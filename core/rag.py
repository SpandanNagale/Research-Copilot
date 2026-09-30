"""Grounded Q&A over the knowledge base, streaming, sources numbered per paper."""
from __future__ import annotations

from typing import Iterator

from core.knowledge import KnowledgeBase
from core.llm import LLMClient
from core.models import Paper

SYSTEM_PROMPT = """You are a rigorous Research Copilot. Answer the user's question using ONLY \
the provided sources.
1. Cite sources strictly using [n] matching the source ids, attached to the statements they support.
2. If sources conflict, mention both.
3. If the answer is not in the sources, say "Insufficient information in the provided sources."
4. Do not hallucinate outside knowledge."""

HISTORY_TURNS = 6
SHORT_QUESTION_WORDS = 8


def rewrite_query(question: str, history: list[dict]) -> str:
    if len(question.split()) < SHORT_QUESTION_WORDS and history:
        prev_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
        if prev_user:
            return f"{prev_user} {question}"
    return question


def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """Groups retrieval hits by paper and numbers them per paper, not per chunk."""
    order: list[int] = []
    grouped: dict[int, dict] = {}
    for h in hits:
        pid = id(h["paper"])
        if pid not in grouped:
            grouped[pid] = {"paper": h["paper"], "texts": [], "score": h["score"]}
            order.append(pid)
        grouped[pid]["texts"].append(h["text"])
        grouped[pid]["score"] = max(grouped[pid]["score"], h["score"])

    blocks = []
    sources = []
    for i, pid in enumerate(order, start=1):
        g = grouped[pid]
        paper: Paper = g["paper"]
        text = " ".join(g["texts"])
        blocks.append(f'<source id="{i}"><title>{paper.title}</title><text>{text}</text></source>')
        sources.append({"num": i, "paper": paper, "score": g["score"]})
    return "\n".join(blocks), sources


def _build_messages(question: str, context: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-HISTORY_TURNS:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"<sources>\n{context}\n</sources>\n\nQuestion: {question}\n"
                "Answer using [n] citations matching the source ids above."
            ),
        }
    )
    return messages


def answer_stream(
    kb: KnowledgeBase,
    client: LLMClient,
    question: str,
    history: list[dict] | None = None,
    theme: int | None = None,
    k: int = 5,
) -> tuple[list[dict], Iterator[str]]:
    """Returns (sources, stream) where sources is ready immediately and stream
    yields filtered answer text chunks."""
    history = history or []
    retrieval_query = rewrite_query(question, history)
    hits = kb.search(retrieval_query, k=k, theme=theme)

    if not hits:
        def _empty() -> Iterator[str]:
            yield "No relevant papers found in the knowledge base to answer this question."

        return [], _empty()

    context, sources = build_context(hits)
    messages = _build_messages(question, context, history)
    return sources, client.stream(messages)
