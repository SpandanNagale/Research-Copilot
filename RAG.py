import os
import re
import time
import traceback
import requests
from typing import List, Tuple, Dict

# Importing from your modified llm_client
from llm_client import call_llm 
# Assuming vectorstore exists in your project structure
from vectorstore import vector_store 

PROMPT_TMPL = """Answer the user's question using ONLY the provided paper abstracts.
Cite sources inline as [1], [2], ... corresponding to the bracketed sources in Context.
Be concise, factual, and avoid speculation. If the answer is uncertain, say so and point to the most relevant sources.

Question:
{question}

Context:
{context}
"""

HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # optional HF fallback

def format_context(snippet: List[Tuple[Dict, float, int]]) -> str:
    blocks = []
    for i, (p, score, idx) in enumerate(snippet, start=1):
        title = p.get("title", "Untitled")
        url = p.get("pdf_url") or p.get("entry_id") or "N/A"
        abstract = p.get("summary") or p.get("abstract") or ""
        blocks.append(f"[{i}] Title: {title}\nURL: {url}\nScore: {score:.3f}\nAbstract: {abstract}")
    return "\n\n".join(blocks)

def _extractive_rag_answer(hits: List[Tuple[Dict, float, int]], question: str, max_sentences: int = 5) -> str:
    question_words = set(re.findall(r"\w+", question.lower()))
    candidates = []
    for i, (p, score, idx) in enumerate(hits, start=1):
        text = (p.get("summary") or p.get("abstract") or "").strip()
        if not text:
            continue
        sents = re.split(r"(?<=[.!?])\s+", text)
        for sent in sents:
            words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(words & question_words)
            candidates.append((overlap, i, sent.strip()))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_sents = [c[2] for c in candidates[:max_sentences]]
    if not top_sents:
        all_text = " ".join([(p.get("summary") or p.get("abstract") or "") for (p,_,_) in hits])
        sents = re.split(r"(?<=[.!?])\s+", all_text)
        top_sents = sents[:max_sentences]
    answer = "Fallback (no LLM) — extractive answer based on retrieved abstracts:\n\n"
    answer += " ".join([s.strip() for s in top_sents if s])
    cited_indices = sorted({i for (_, i, _) in candidates[:max_sentences]})
    if cited_indices:
        answer += "\n\nSources: " + ", ".join(f"[{idx}]" for idx in cited_indices)
    return answer

def _call_hf_space(question: str, context_text: str) -> str:
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL not configured")
    payload = {"data": [f"Question: {question}\n\nContext:\n{context_text}"]}
    session = requests.Session()
    resp = session.post(HF_SPACE_URL, json=payload, timeout=20)
    resp.raise_for_status()
    j = resp.json()
    if isinstance(j, dict):
        if "data" in j and isinstance(j["data"], list):
            return j["data"][0]
        if "predictions" in j and isinstance(j["predictions"], list):
            return j["predictions"][0]
    return str(j)

def RAG_ans(vector_store, question: str, k: int = 4, max_retries: int = 3, backoff_base: float = 1.0):
    """
    LLM-first RAG (Gemini Edition):
    - Retrieve hits
    - Try call_llm(prompt) with retries on network/transient errors
    - If call succeeds -> return LLM answer + hits
    - If call definitively fails -> try HF fallback -> extractive fallback
    """
    hits = vector_store.search(question, k)
    context = format_context(hits)
    prompt = PROMPT_TMPL.format(question=question, context=context)

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        # No key: behave as before but explicit
        fallback = _extractive_rag_answer(hits, question, max_sentences=5)
        return fallback + "\n\n[NOTE] No GEMINI_API_KEY configured; used extractive fallback.", hits

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # Attempt the LLM call
            answer = call_llm(prompt)
            
            # Defensive check for error messages in the text response itself
            if not answer or "gemini error" in answer.lower() or "fallback" in answer.lower():
                raise RuntimeError(f"LLM returned empty/error-like response: {answer}")
            
            return answer, hits
            
        except Exception as e:
            last_exc = e
            traceback.print_exc()
            
            # Retry logic
            if attempt < max_retries:
                sleep_time = backoff_base * (2 ** (attempt - 1))
                print(f"[RAG] Gemini call failed (attempt {attempt}/{max_retries}), retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            else:
                break

    # At this point LLM failed after retries.
    # Try HF fallback if configured
    if HF_SPACE_URL:
        try:
            hf_ans = _call_hf_space(question, context)
            return f"[HF_SPACE fallback used]\n\n{hf_ans}", hits
        except Exception:
            traceback.print_exc()

    # Final: deterministic extractive fallback
    fallback = _extractive_rag_answer(hits, question, max_sentences=5)
    note = ("\n\n[NOTE] Gemini LLM call failed after retries; used extractive fallback. "
            f"Error summary: {str(last_exc).splitlines()[0] if last_exc else 'unknown'}")
    return fallback + note, hits
