from groq import Groq
import os
from dotenv import load_dotenv

def get_llm(prompt: str, system: str = "You are a precise research assistant.")->str:
    client=Groq(api_key=os.getenv("GROQ_API_KEY"),)
    model="llama-3.1-8b-instant"
    resp=client.chat.completions.create(
         model=model,
         messages=[{"role":"user","content":system},{"role":"user","content":prompt}],
         temperature=0.2
        )
    return resp.choices[0].message.content.strip()

def call_llm(prompt: str) -> str:
    if os.getenv("GROQ_API_KEY"):
        return get_llm(prompt)
    # Fallback (no keys): extractive answer from context – last resort so app still runs
    # This simply returns the first 6-8 sentences of the provided "Context:" block.
    import re
    ctx = re.split(r"Context:\s*", prompt, flags=re.IGNORECASE)
    text = ctx[-1] if len(ctx) > 1 else prompt
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return "No LLM key set. Fallback extractive answer:\n\n" + " ".join(sentences[:8])
