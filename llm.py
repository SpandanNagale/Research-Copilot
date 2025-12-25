import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Default to Flash for speed/cost efficiency in RAG loops
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def get_llm(prompt: str, system_instruction: str = "You are a precise research assistant.") -> str:
    """
    Call Google Gemini API and return the assistant text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing. call_llm() will fallback.")

    try:
        genai.configure(api_key=api_key)
        
        # System instructions are supported in newer Gemini models
        model = genai.GenerativeModel(
            model_name=DEFAULT_MODEL,
            system_instruction=system_instruction
        )

        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=800,
            temperature=0.2,
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Check for safety blocks or empty responses
        if not response.text:
             raise RuntimeError(f"Gemini returned empty response. Finish reason: {response.candidates[0].finish_reason}")
             
        return response.text.strip()

    except Exception as e:
        # Catch SDK specific errors or network issues
        raise RuntimeError(f"Gemini API Error: {str(e)}") from e


def call_llm(prompt: str) -> str:
    """
    If GEMINI_API_KEY is present -> Google Gemini.
    Otherwise -> Fallback extractive answer.
    """
    if os.getenv("GEMINI_API_KEY"):
        return get_llm(prompt)

    # Fallback (Extracts text from prompt if no key provided)
    ctx = re.split(r"Context:\s*", prompt, flags=re.IGNORECASE)
    text = ctx[-1] if len(ctx) > 1 else prompt
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return "No GEMINI_API_KEY set. Fallback extractive answer:\n\n" + " ".join(sentences[:8])
