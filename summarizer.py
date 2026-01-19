import asyncio
import logging
from typing import List, Dict, Any

# Import the unified LLM router
from LLM import get_llm_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Summarizer_Engine")

# --- PROMPT TEMPLATE ---
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

async def summarize_with_limit(
    sem: asyncio.Semaphore, 
    text: str, 
    config: Dict[str, Any]
) -> str:
    """
    Wraps the summarization logic with a Semaphore to prevent 
    hitting the API Rate Limit (ResourceExhausted).
    """
    async with sem:  # <--- Waits here if 3 requests are already active
        try:
            # 1. Validation
            if not text or len(text) < 50:
                return "Abstract too short or missing."

            # 2. Construction
            prompt = SUMMARY_PROMPT.format(text=text)

            # 3. Execution
            # We add a tiny artificial delay to ensure we don't burst strictly at the limit
            # if the API is feeling sensitive.
            await asyncio.sleep(0.5) 
            
            summary = await get_llm_async(prompt, config)
            return summary
            
        except Exception as e:
            logger.error(f"Summary failed for text: {text[:20]}... Error: {e}")
            return "Summary unavailable due to rate limit/error."

async def batch_summary_async(abstracts: List[str], config: Dict[str, Any]) -> List[str]:
    """
    Summarizes abstracts with CONCURRENCY CONTROL.
    """
    # 1. SET THE LIMIT
    # Gemini Free Tier allows ~15 RPM (Requests Per Minute).
    # Setting concurrency to 3 ensures we don't flood the pipe.
    # If you have a paid tier, you can bump this to 10 or 20.
    limit = 3 
    sem = asyncio.Semaphore(limit)

    # 2. CREATE TASKS
    tasks = [summarize_with_limit(sem, text, config) for text in abstracts]
    
    # 3. RUN WITH PROGRESS LOGGING
    # We use gather, but the Semaphore inside the function controls the speed.
    results = await asyncio.gather(*tasks)
    
    return results

# --- SYNC WRAPPER ---
def batch_summary(abstracts: List[str], config: Dict[str, Any]) -> List[str]:
    return asyncio.run(batch_summary_async(abstracts, config))
