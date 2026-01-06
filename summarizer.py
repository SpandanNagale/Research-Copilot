import asyncio
import logging
from typing import List, Dict, Any

# Import the unified LLM router
from llm import get_llm_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Summarizer_Engine")

# --- PROMPT TEMPLATE ---
# Forces the model to be concise and consistent.
SUMMARY_PROMPT = """You are an expert academic synthesizer. 
Summarize the following abstract into exactly three structured sections.
Do not use conversational filler or introductory text.

Abstract:
{text}

Output format:
**Problem:** [1 sentence on the research gap]
**Method:** [1-2 sentences on the specific model/dataset/algorithm used]
**Result:** [1 sentence on the key metric or finding]
"""

async def summarize_async(text: str, config: Dict[str, Any]) -> str:
    """
    Summarizes a single abstract using the configured LLM (Gemini or Ollama).
    """
    # 1. Validation
    if not text or len(text) < 50:
        return "Abstract too short or missing."

    # 2. Construction
    prompt = SUMMARY_PROMPT.format(text=text)

    # 3. Execution
    try:
        # We pass the 'config' (provider, api_key, model) down to the LLM router
        summary = await get_llm_async(prompt, config)
        return summary
        
    except Exception as e:
        logger.error(f"Summary failed for text snippet: {text[:30]}... Error: {e}")
        return "Summary unavailable due to LLM error."

async def batch_summary_async(abstracts: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Summarizes multiple abstracts IN PARALLEL.
    
    Args:
        abstracts: List of abstract strings.
        config: Dict containing {"provider": "...", "api_key": "...", "model": "..."}.
        
    Returns:
        List of summary strings in the same order as input.
    """
    # Create a list of async tasks
    tasks = [summarize_async(text, config) for text in abstracts]
    
    # asyncio.gather runs them all simultaneously
    # If using Gemini, 20 papers take ~3-4 seconds total.
    results = await asyncio.gather(*tasks)
    
    return results

# --- SYNC WRAPPER ---
# Use this only if you are calling from a legacy synchronous script.
def batch_summary(abstracts: List[str], config: Dict[str, Any]) -> List[str]:
    return asyncio.run(batch_summary_async(abstracts, config))