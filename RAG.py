# resilient_rag.py
import os
import re
import traceback
import requests
import socket
from typing import List, Tuple, Dict

from llm import call_llm  # your existing client (calls OpenRouter)
from vectorstore import vector_store  # or import the class you use

PROMPT_TMPL = """You are a research assistant. Answer the user's question using ONLY the provided paper abstracts.
Cite sources inline as [1], [2], ... corresponding to the bracketed sources in Context.
Be concise, factual, and avoid speculation. If the answer is uncertain, say so and point to the most relevant sources.

Question:
{question}

Context:
{context}
"""

HF_SPACE_URL = os.getenv("HF_SPACE_URL")  # optional fallback endpoint you can host on HF Spaces

def format_context(snippet: List[Tuple[Dict, float, int]]) -> str:
    blocks = []
    for i, (p, score, idx) in enumerate(snippet, start=1):
        title = p.get("title", "Untitled")
        url = p.get("pdf_url") or p.get("entry_id") or "N/A"
        abstract = p.get("summary", "") or p.get("abstract", "") or ""
        blocks.append(
            f"[{i}] Title: {title}\nURL: {url}\nScore: {score:.3f}\nAbstract: {abstract}"
        )
    return "\n\n".join(blocks)

def _extractive_rag_answer(hits: List[Tuple[Dict, float, int]], question: str, max_sentences: int = 5) -> str:
    """
    Build a deterministic extractive answer from the retrieved abstracts.
    - Concatenate top-k abstracts, pull the top sentences likely to answer the question (simple heuristics).
    - Return an answer and cite sources.
    """
    # naive approach: gather abstracts and pick sentences containing keywords from question
    question_words = set(re.findall(r"\w+", question.lower()))
    candidates = []
    for i, (p, score, idx) in enumerate(hits, start=1):
        text = (p.get("summary") or p.get("abstract") or "").strip()
        if not text:
            continue
        sents = re.split(r"(?<=[.!?])\s+", text)
        # score sentences by overlap with question words
        for sent in sents:
            words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(words & question_words)
            candidates.append((overlap, i, sent.strip()))
    # sort by overlap then by source order
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_sents = [c[2] for c in candidates[:max_sentences]]
    if not top_sents:
        # fallback: first N sentences from concatenated abstracts
        all_text = " ".join([(p.get("summary") or p.get("abstract") or "") for (p,_,_) in hits])
        sents = re.split(r"(?<=[.!?])\s+", all_text)
        top_sents = sents[:max_sentences]
    # Build the answer and add citations for the most referenced sources
    answer = "Fallback (no LLM) — extractive answer based on retrieved abstracts:\n\n"
    answer += " ".join([s.strip() for s in top_sents if s])
    # append quick citation pointers
    cited_indices = sorted({i for (_, i, _) in candidates[:max_sentences]})
    if cited_indices:
        answer += "\n\nSources: " + ", ".join(f"[{idx}]" for idx in cited_indices)
    return answer

def _call_hf_space(question: str, context_text: str) -> str:
    """
    Optional: call a Hugging Face Space or other HTTP fallback which accepts a JSON payload.
    Configure HF_SPACE_URL in your environment if you have one.
    The Space is expected to return a simple string in predictions/data.
    """
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL not configured")
    payload = {"data": [f"Question: {question}\n\nContext:\n{context_text}"]}
    session = requests.Session()
    resp = session.post(HF_SPACE_URL, json=payload, timeout=20)
    resp.raise_for_status()
    j = resp.json()
    # adapt to common Space output shapes:
    if isinstance(j, dict):
        if "data" in j and isinstance(j["data"], list):
            return j["data"][0]
        if "predictions" in j and isinstance(j["predictions"], list):
            return j["predictions"][0]
    return str(j)

def RAG_ans(vector_store, question: str, k: int = 4) -> Tuple[str, List[Tuple[Dict, float, int]]]:
    """
    Resilient RAG answer:
      - Retrieve top-k documents from vector_store
      - Try to call the LLM (call_llm)
      - On network/LLM failure, return deterministic extractive fallback built from hits
      - Also attempt HF Space fallback if configured
    Returns (answer_text, hits)
    """
    hits = vector_store.search(question, k)
    context = format_context(hits)
    prompt = PROMPT_TMPL.format(question=question, context=context)

    # First: quick DNS check for api.openrouter.ai if OPENROUTER_API_KEY set
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        try:
            # attempt LLM call
            answer = call_llm(prompt)
            return answer, hits
        except Exception as e:
            # Detect common network resolution/connection errors
            err_str = repr(e)
            # Log the full traceback to stdout/stderr or your logger; keep user-facing message short
            traceback.print_exc()
            # If HF fallback configured, try it
            if HF_SPACE_URL:
                try:
                    hf_ans = _call_hf_space(question, context)
                    return f"[HF_SPACE fallback used]\n\n{hf_ans}", hits
                except Exception as he:
                    traceback.print_exc()
                    # continue to extractive fallback
            # Use deterministic extractive fallback
            fallback = _extractive_rag_answer(hits, question, max_sentences=5)
            note = ("\n\n[NOTE] Upstream LLM call failed and fallback was used. "
                    "Error summary: " + (str(e).splitlines()[0] if err_str else "unknown"))
            return fallback + note, hits
    else:
        # No LLM key set — use deterministic extractive fallback (same behavior you had before)
        fallback = _extractive_rag_answer(hits, question, max_sentences=5)
        return fallback + "\n\n[NOTE] No OPENROUTER_API_KEY configured; used extractive fallback.", hits
