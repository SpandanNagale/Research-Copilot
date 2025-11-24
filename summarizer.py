# robust_summary_llm.py
import os
import re
import socket
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"
MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-4.1-fast:free")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
# Optional fallback HF Space (e.g. https://USERNAME-SPACE.hf.space/api/predict/)
HF_SPACE_URL = os.getenv("HF_SPACE_URL")

def _can_resolve(hostname: str) -> bool:
    try:
        socket.gethostbyname(hostname)
        return True
    except Exception:
        return False

def _requests_session_with_retries(total_retries: int = 3, backoff_factor: float = 0.5):
    session = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST", "GET"),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def _extractive_summary(text: str, num_sentences: int = 3) -> str:
    # cheap fallback: first N sentences from the "Text:" or full text
    parts = re.split(r"Text:\s*", text, flags=re.IGNORECASE)
    src = parts[-1].strip() if parts else text
    sents = re.split(r"(?<=[.!?])\s+", src)
    return "Fallback extractive summary:\n\n" + " ".join(sents[:max(num_sentences, 1)])

def summary(text: str) -> str:
    # If no API key, immediately fallback
    if not OPENROUTER_KEY:
        return _extractive_summary(text, num_sentences=3)

    # DNS check
    if not _can_resolve("api.openrouter.ai"):
        # try HF Space fallback if configured
        if HF_SPACE_URL:
            try:
                return _call_hf_space_summary(text)
            except Exception:
                return "Network/DNS error: cannot resolve api.openrouter.ai and HF fallback failed.\n\n" + _extractive_summary(text, 3)
        return "Network/DNS error: cannot resolve api.openrouter.ai. " + _extractive_summary(text, 3)

    # Prepare payload
    prompt = (
        "Summarize this academic abstract in 3 bullet points. "
        "Preserve key methods, datasets, and results.\n\n"
        f"Text:\n{text}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise academic summarizer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}

    session = _requests_session_with_retries(total_retries=3, backoff_factor=1.0)
    try:
        resp = session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # Try HF space fallback if configured
        if HF_SPACE_URL:
            try:
                return _call_hf_space_summary(text)
            except Exception:
                # fall through to extractive
                pass
        # final fallback
        return f"OpenRouter request failed: {str(e)}\n\n" + _extractive_summary(text, 3)

def _call_hf_space_summary(text: str) -> str:
    """
    Optional: call a Hugging Face Space / custom endpoint that you host as fallback.
    The HF Space should accept JSON payload { "data": [text] } and return predictions.
    Configure HF_SPACE_URL in env if you have one.
    """
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL not configured")
    session = _requests_session_with_retries()
    payload = {"data": [text]}
    resp = session.post(HF_SPACE_URL, json=payload, timeout=20)
    resp.raise_for_status()
    j = resp.json()
    # Adapt depending on your Space's output format — common is j["data"][0] or j["predictions"][0]
    if "data" in j and isinstance(j["data"], list):
        return j["data"][0]
    if "predictions" in j and isinstance(j["predictions"], list):
        return j["predictions"][0]
    # fallback raw
    return str(j)

def batch_summary(abstracts: list[str]) -> list[str]:
    return [summary(t) for t in abstracts]

