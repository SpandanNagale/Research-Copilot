# llm_client.py  (replace your old Groq LLM file with this)

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/r1:free")

def get_llm(prompt: str, system: str = "You are a precise research assistant.") -> str:
    """
    Call OpenRouter chat completions and return the assistant text.
    Works as a drop-in replacement for the previous Groq version.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing. call_llm() will fallback.")

    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text}") from e

    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected OpenRouter response format: {data}")


def call_llm(prompt: str) -> str:
    """
    If OPENROUTER_API_KEY is present → OpenRouter.
    Otherwise → your fallback extractive answer.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return get_llm(prompt)

    # Fallback (unchanged from previous Groq version)
    ctx = re.split(r"Context:\s*", prompt, flags=re.IGNORECASE)
    text = ctx[-1] if len(ctx) > 1 else prompt
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return "No LLM key set. Fallback extractive answer:\n\n" + " ".join(sentences[:8])

