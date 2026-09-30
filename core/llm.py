"""One OpenAI-SDK client for every provider (all are OpenAI-compatible)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import requests
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_secret

PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-20b",
        "parallelism": 3,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "parallelism": 2,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "openrouter/auto",
        "parallelism": 4,
        "extra_headers": {
            "HTTP-Referer": "https://research-copilot.streamlit.app",
            "X-Title": "Research Copilot",
        },
    },
    "ollama_cloud": {
        "base_url": "https://ollama.com/v1",
        "key_env": "OLLAMA_API_KEY",
        "default_model": "gpt-oss:20b",
        "parallelism": 3,
    },
    "ollama_local": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "default_api_key": "ollama",
        "default_model": "llama3.1:8b",
        "parallelism": 1,
    },
}

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


class LLMError(Exception):
    """Readable, user-facing LLM failure."""


def _friendly_error(exc: Exception, provider: str, model: str) -> LLMError:
    if isinstance(exc, AuthenticationError):
        return LLMError(f"{provider}: invalid API key.")
    if isinstance(exc, PermissionDeniedError):
        return LLMError(f"{provider}: access forbidden for this key.")
    if isinstance(exc, NotFoundError):
        return LLMError(f"{provider}: model '{model}' not found.")
    if isinstance(exc, APIStatusError) and exc.status_code == 402:
        return LLMError(f"{provider}: out of credit.")
    return LLMError(f"{provider}: {exc}")


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def _create_completion(client: OpenAI, **kwargs):
    return client.chat.completions.create(**kwargs)


# --- streaming think-tag filter -------------------------------------------------
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


class ThinkFilter:
    """Strips <think>...</think> (and a leading orphan '...</think>') from a text
    stream, holding back only the longest trailing suffix that could still become
    a tag boundary-split across chunks.

    # ponytail: detecting a leading orphan </think> is only decidable once we've
    # seen either tag, which true unbounded streaming can't guarantee without
    # buffering forever. We cap that lookahead at MAX_LEADING_LOOKAHEAD chars (a
    # small, fixed startup delay); upgrade path if a provider's orphan reasoning
    # ever exceeds it: make the cap configurable per provider.
    """

    MAX_LEADING_LOOKAHEAD = 64

    def __init__(self):
        self._buf = ""
        self._inside = False
        self._at_start = True

    def feed(self, chunk: str) -> str:
        s = self._buf + chunk
        self._buf = ""
        out = []

        while True:
            if self._inside:
                close_idx = s.find(CLOSE_TAG)
                if close_idx == -1:
                    break
                s = s[close_idx + len(CLOSE_TAG):]
                self._inside = False
                continue

            if self._at_start:
                open_idx = s.find(OPEN_TAG)
                close_idx = s.find(CLOSE_TAG)
                if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                    s = s[close_idx + len(CLOSE_TAG):]
                    self._at_start = False
                    continue
                if open_idx != -1:
                    out.append(s[:open_idx])
                    s = s[open_idx + len(OPEN_TAG):]
                    self._inside = True
                    self._at_start = False
                    continue
                if len(s) < self.MAX_LEADING_LOOKAHEAD:
                    break  # not enough evidence yet; keep buffering (bounded)
                self._at_start = False  # lookahead exhausted: treat as normal text
                continue

            open_idx = s.find(OPEN_TAG)
            if open_idx == -1:
                break
            out.append(s[:open_idx])
            s = s[open_idx + len(OPEN_TAG):]
            self._inside = True

        if self._inside:
            self._buf = self._longest_partial_suffix(s, (CLOSE_TAG,))
        elif self._at_start:
            self._buf = s  # still undecided; all of it stays buffered (bounded above)
        else:
            suffix = self._longest_partial_suffix(s, (OPEN_TAG,))
            out.append(s[: len(s) - len(suffix)] if suffix else s)
            self._buf = suffix

        return "".join(out)

    def finish(self) -> str:
        remainder = self._buf if not self._inside else ""
        self._buf = ""
        return remainder

    @staticmethod
    def _longest_partial_suffix(s: str, tags: tuple[str, ...]) -> str:
        max_len = max(len(t) for t in tags) - 1
        for length in range(min(len(s), max_len), 0, -1):
            suffix = s[-length:]
            if any(tag.startswith(suffix) for tag in tags):
                return suffix
        return ""


def strip_think(text: str) -> str:
    filt = ThinkFilter()
    return filt.feed(text) + filt.finish()


# --- model listing (cached ~1h) --------------------------------------------------
_MODEL_CACHE: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL = 3600


def _is_chat_model(provider: str, model_id: str) -> bool:
    lowered = model_id.lower()
    if provider == "groq":
        return not any(bad in lowered for bad in ("whisper", "guard", "tts"))
    if provider == "gemini":
        return lowered.startswith("gemini") or lowered.startswith("models/gemini")
    return True


def list_models(provider: str, api_key: str | None = None, base_url: str | None = None, force: bool = False) -> list[str]:
    cfg = PROVIDERS[provider]
    base_url = base_url or cfg["base_url"]
    api_key = api_key or _resolve_api_key(provider)
    cache_key = f"{provider}:{base_url}"

    if not force and cache_key in _MODEL_CACHE:
        ts, models = _MODEL_CACHE[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return models

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    ids = []
    for item in data:
        model_id = item.get("id", "")
        if not model_id or not _is_chat_model(provider, model_id):
            continue
        if provider == "gemini" and model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        ids.append(model_id)

    _MODEL_CACHE[cache_key] = (time.time(), ids)
    return ids


def filter_free_openrouter(provider: str, base_url: str | None, api_key: str | None) -> list[str]:
    """Separate raw fetch since free-tier pricing isn't in the plain id list."""
    base_url = base_url or PROVIDERS["openrouter"]["base_url"]
    api_key = api_key or _resolve_api_key("openrouter")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = requests.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    free = []
    for item in data:
        model_id = item.get("id", "")
        pricing = item.get("pricing", {})
        if model_id.endswith(":free") or pricing.get("prompt") in ("0", 0):
            free.append(model_id)
    return free


def _resolve_api_key(provider: str) -> str | None:
    cfg = PROVIDERS[provider]
    if cfg.get("key_env"):
        key = get_secret(cfg["key_env"])
        if key:
            return key
    return cfg.get("default_api_key")


@dataclass
class LLMClient:
    provider: str
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3

    def __post_init__(self):
        cfg = PROVIDERS[self.provider]
        self.base_url = self.base_url or cfg["base_url"]
        self.api_key = self.api_key or _resolve_api_key(self.provider)
        self.model = self.model or cfg["default_model"]
        self._extra_headers = cfg.get("extra_headers", {})
        self._client = OpenAI(api_key=self.api_key or "unset", base_url=self.base_url)

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        try:
            resp = _create_completion(
                self._client,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                extra_headers=self._extra_headers or None,
            )
        except Exception as exc:
            raise _friendly_error(exc, self.provider, self.model) from exc
        return strip_think(resp.choices[0].message.content or "")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        filt = ThinkFilter()
        try:
            stream = _create_completion(
                self._client,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
                extra_headers=self._extra_headers or None,
            )
            for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    piece = filt.feed(delta)
                    if piece:
                        yield piece
            tail = filt.finish()
            if tail:
                yield tail
        except Exception as exc:
            raise _friendly_error(exc, self.provider, self.model) from exc

    def test_connection(self) -> tuple[bool, str]:
        try:
            self.chat([{"role": "user", "content": "Reply with OK."}])
            return True, "Connected."
        except LLMError as e:
            return False, str(e)


DEFAULT_PROVIDER_ORDER = ["ollama_cloud", "openrouter", "groq", "gemini", "ollama_local"]

PROVIDER_DISPLAY_NAMES = {
    "groq": "Groq",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "ollama_cloud": "Ollama Cloud",
    "ollama_local": "Ollama (local)",
}


def display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.replace("_", " ").title())


class FallbackLLMClient:
    """No provider picker: tries providers in order and moves to the next one
    on any LLMError (bad/missing key, rate limit, model gone, connection
    refused for a local Ollama that isn't running, etc.)."""

    def __init__(self, order: list[str] | None = None, temperature: float = 0.3):
        self.order = order or DEFAULT_PROVIDER_ORDER
        self.temperature = temperature
        self._clients = [LLMClient(provider=p, temperature=temperature) for p in self.order]
        self._last_working: str | None = None

    @property
    def provider(self) -> str:
        # for callers that size thread pools off PROVIDERS[client.provider] (e.g. synthesis.py)
        return self._last_working or self.order[0]

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages)

    def chat(self, messages: list[dict]) -> str:
        last_err: LLMError | None = None
        for client in self._clients:
            try:
                result = client.chat(messages)
                self._last_working = client.provider
                return result
            except LLMError as exc:
                last_err = exc
        raise last_err or LLMError("No LLM provider is configured.")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        last_err: LLMError | None = None
        for client in self._clients:
            gen = client.stream(messages)
            try:
                first_chunk = next(gen)
            except StopIteration:
                self._last_working = client.provider
                return
            except LLMError as exc:
                last_err = exc
                continue
            self._last_working = client.provider
            yield first_chunk
            yield from gen
            return
        raise last_err or LLMError("No LLM provider is configured.")

    def test_connection(self) -> tuple[bool, str]:
        for client in self._clients:
            ok, msg = client.test_connection()
            if ok:
                self._last_working = client.provider
                return True, f"Connected via {display_name(client.provider)}."
        return False, "All configured providers failed — check API keys in Data source keys / .env."
