import os
import asyncio
import logging
import google.generativeai as genai
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_Engine")

# --- 1. GEMINI BACKEND ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_gemini(prompt: str, api_key: str, model_name: str = "gemini-1.5-flash") -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a precise research assistant. Be factual and cite sources."
    )
    
    response = await model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=8192,
            temperature=0.3
        )
    )
    return response.text.strip()

# --- 2. OLLAMA BACKEND (LOCAL) ---
async def _call_ollama(prompt: str, model_name: str = "gemma3") -> str:
    """
    Calls local Ollama instance. Requires 'ollama serve' running.
    """
    try:
        # Ollama python client is sync, so we wrap it in a thread to keep UI responsive
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: ollama.chat(model=model_name, messages=[
                {'role': 'system', 'content': 'You are a precise research assistant.'},
                {'role': 'user', 'content': prompt},
            ])
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama Error: {e}")
        return f"Error connecting to Local LLM: {str(e)}. Is Ollama running?"

# --- 3. UNIFIED GATEWAY ---
async def get_llm_async(prompt: str, config: dict) -> str:
    """
    Router that sends the prompt to the correct backend.
    
    config = {
        "provider": "gemini" | "ollama",
        "api_key": "...", (if gemini)
        "model": "gemini-1.5-flash" | "llama3" | "deepseek-r1"
    }
    """
    provider = config.get("provider", "gemini")
    
    if provider == "gemini":
        if not config.get("api_key"):
            return "Error: Gemini selected but no API Key provided."
        return await _call_gemini(prompt, config["api_key"], config.get("model", "gemini-1.5-flash"))
        
    elif provider == "ollama":
        return await _call_ollama(prompt, config.get("model", "mistral"))
        
    return "Error: Unknown LLM provider."