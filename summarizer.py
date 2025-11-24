# summary_llm.py

import os
import requests
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_URL = "https://api.openrouter.ai/v1/chat/completions"

# Default free model — change via .env if you want
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/r1:free")

def summary(text: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "No OPENROUTER_API_KEY set. Cannot generate summary."

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

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
    try:
        resp.raise_for_status()
    except Exception as e:
        return f"OpenRouter error: {resp.text}"

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return f"Unexpected OpenRouter response format: {data}"


def batch_summary(abstracts: list[str]) -> list[str]:
    return [summary(t) for t in abstracts]

