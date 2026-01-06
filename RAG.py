import logging
import asyncio
from typing import List, Tuple, Dict, Any

# Import the unified LLM router we built
from llm import get_llm_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG_Engine")

# --- PROMPT TEMPLATE ---
# We use XML tags because models (both Gemini and Llama 3) follow these boundaries 
# better than plain text, preventing "citation hallucination".
PROMPT_TMPL = """You are a rigorous Research Copilot.
Answer the user's question using ONLY the provided academic abstracts below.

<instructions>
1. Cite sources strictly using the format [1], [2] attached to the statements they support.
2. If the abstracts contain conflicting results, mention both.
3. If the answer is not in the abstracts, state "Insufficient information in the provided sources."
4. Do not hallucinate external knowledge.
5. Keep the answer concise and professional.
</instructions>

<sources>
{context}
</sources>

Question: {question}
Answer:"""

def format_context_structured(hits: List[Tuple[Dict, float, int]]) -> str:
    """
    Formats retrieval hits into clear XML blocks for the LLM.
    """
    blocks = []
    for i, (p, score, idx) in enumerate(hits, start=1):
        # Graceful handling of missing metadata
        title = p.get("title", "Unknown Title")
        url = p.get("pdf_url") or p.get("entry_id") or "N/A"
        
        # Priority: Summary -> Abstract -> Raw Text
        content = p.get("summary") or p.get("abstract") or p.get("text", "")
        # Clean up newlines to save tokens and avoid breaking XML
        content = content.replace("\n", " ").strip()

        block = (
            f'<source id="{i}">'
            f'<title>{title}</title>'
            f'<url>{url}</url>'
            f'<relevance_score>{score:.3f}</relevance_score>'
            f'<text>{content}</text>'
            f'</source>'
        )
        blocks.append(block)
    
    return "\n".join(blocks)

async def RAG_ans_async(vector_store, question: str, config: Dict[str, Any], k: int = 5) -> Tuple[str, List[Any]]:
    """
    Async RAG Pipeline:
    1. Retrieve relevant chunks from VectorStore.
    2. Format them into XML.
    3. Send to LLM (Gemini or Ollama based on 'config').
    
    Args:
        vector_store: Your FAISS/VectorStore instance.
        question: The user's query string.
        config: Dict containing {"provider": "...", "api_key": "...", "model": "..."}.
        k: Number of papers to retrieve.
        
    Returns:
        Tuple(Answer String, List of Source Hits)
    """
    # 1. RETRIEVAL
    # Most local vector stores (FAISS) are CPU-bound and synchronous.
    # We wrap it in to_thread to prevent blocking the main event loop.
    try:
        loop = asyncio.get_running_loop()
        hits = await loop.run_in_executor(None, vector_store.search, question, k)
        
        if not hits:
            return "No relevant research papers found in the database to answer this question.", []
            
    except Exception as e:
        logger.error(f"Vector Store Retrieval Failed: {e}")
        return "Error accessing the knowledge base (Vector Store failure).", []

    # 2. FORMATTING
    context_text = format_context_structured(hits)
    prompt = PROMPT_TMPL.format(question=question, context=context_text)

    # 3. GENERATION
    try:
        # Pass the full config so LLM.py knows whether to use Gemini or Ollama
        answer = await get_llm_async(prompt, config)
        return answer, hits
        
    except Exception as e:
        logger.error(f"RAG Generation Failed: {e}")
        return (
            f"I found relevant papers, but I'm unable to synthesize an answer due to an LLM error.\n"
            f"Error Details: {str(e)}", 
            hits
        )

# --- SYNC WRAPPER ---
# Only use this if you are calling RAG from a legacy sync script.
# In Streamlit, prefer using `asyncio.run(RAG_ans_async(...))` directly in the app.
def RAG_ans(vector_store, question: str, config: Dict[str, Any], k: int = 5):
    return asyncio.run(RAG_ans_async(vector_store, question, config, k))