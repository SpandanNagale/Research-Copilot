import pytest

from core.knowledge import Embedder, KnowledgeBase
from core.rag import answer_stream, build_context, rewrite_query
from tests.conftest import FakeLLM


@pytest.fixture
def tfidf_kb(sample_papers, monkeypatch):
    monkeypatch.setattr(
        "core.knowledge._load_sentence_transformer",
        lambda: (_ for _ in ()).throw(RuntimeError("no weights")),
    )
    kb = KnowledgeBase(embedder=Embedder())
    kb.build(sample_papers)
    kb.cluster(k=2)
    return kb


def test_rewrite_query_prepends_previous_question_for_short_followups():
    history = [{"role": "user", "content": "What methods does the paper use?"}, {"role": "assistant", "content": "..."}]
    rewritten = rewrite_query("and results?", history)
    assert rewritten.startswith("What methods does the paper use?")
    assert "and results?" in rewritten


def test_rewrite_query_leaves_long_questions_untouched():
    history = [{"role": "user", "content": "prior question"}]
    q = "This is a sufficiently long question that should not be rewritten at all"
    assert rewrite_query(q, history) == q


def test_build_context_numbers_sources_per_paper_not_per_chunk(sample_papers):
    p = sample_papers[0]
    hits = [
        {"paper": p, "text": "chunk one", "score": 0.9},
        {"paper": p, "text": "chunk two", "score": 0.5},
        {"paper": sample_papers[1], "text": "other paper chunk", "score": 0.8},
    ]
    context, sources = build_context(hits)
    assert len(sources) == 2  # deduped per paper, not per chunk
    assert sources[0]["num"] == 1
    assert sources[0]["score"] == 0.9  # max score across that paper's chunks
    assert "chunk one" in context and "chunk two" in context
    assert '<source id="1">' in context
    assert '<source id="2">' in context


def test_answer_stream_returns_sources_and_streams_filtered_answer(tfidf_kb):
    llm = FakeLLM(static_response="The answer is [1].")
    sources, stream = answer_stream(tfidf_kb, llm, "What is this about?")
    assert sources
    text = "".join(stream)
    assert "[1]" in text


def test_answer_stream_no_hits_returns_empty_sources(tfidf_kb):
    llm = FakeLLM(static_response="unused")
    sources, stream = answer_stream(tfidf_kb, llm, "zzzzz nonexistent gibberish query", theme=999)
    assert sources == []
    assert "No relevant papers" in "".join(stream)
