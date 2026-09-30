import re

import pytest

from core.knowledge import Embedder, KnowledgeBase
from core.synthesis import name_themes_ai, summarize_papers, generate_review, _parse_theme_json
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


def test_summarize_papers_sets_ai_summary(sample_papers):
    llm = FakeLLM(static_response="**Problem:** x\n**Method:** y\n**Result:** z")
    progress_calls = []
    summarize_papers(sample_papers, llm, on_progress=lambda done, total: progress_calls.append((done, total)))
    assert all(p.ai_summary for p in sample_papers)
    assert progress_calls[-1] == (len(sample_papers), len(sample_papers))


def test_summarize_papers_stops_early_after_three_failures(sample_papers):
    def failing(key):
        raise RuntimeError("boom")

    llm = FakeLLM(response_fn=failing)
    summarize_papers(sample_papers, llm)
    processed = [p for p in sample_papers if "Summary unavailable" in p.ai_summary]
    assert len(processed) == 3  # stopped processing results after the first 3 failures


def test_name_themes_ai_parses_json(sample_papers):
    llm = FakeLLM(static_response='{"0": "Vision Models", "1": "Molecular ML"}')
    names = name_themes_ai(sample_papers, llm)
    assert names[0] == "Vision Models"
    assert names[1] == "Molecular ML"


def test_name_themes_ai_falls_back_on_garbage(sample_papers):
    llm = FakeLLM(static_response="not json at all")
    names = name_themes_ai(sample_papers, llm)
    assert names[0] == "Theme 1"
    assert names[1] == "Theme 2"


def test_parse_theme_json_handles_surrounding_prose():
    raw = 'Sure, here it is:\n{"0": "A", "1": "B"}\nHope that helps!'
    result = _parse_theme_json(raw, [0, 1])
    assert result == {0: "A", 1: "B"}


def test_generate_review_citation_numbers_within_reference_range(tfidf_kb, sample_papers):
    def resp_fn(key):
        if "Research gaps and open questions" in key and "Theme sections" in key:
            return (
                "## Introduction\n\nIntro text [1].\n\n"
                "## Research gaps and open questions\n\nGaps text [2].\n\n"
                "## Conclusion\n\nConclusion text [1][2]."
            )
        return "Section body citing [1] and [2]."

    llm = FakeLLM(response_fn=resp_fn)
    review = generate_review(sample_papers, tfidf_kb, llm, papers_per_theme=5)

    assert "## Introduction" in review
    assert "## Research gaps and open questions" in review
    assert "## Conclusion" in review
    assert "## References" in review

    citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", review)]
    assert citation_numbers  # at least one citation present
    assert max(citation_numbers) <= len(sample_papers)
    assert min(citation_numbers) >= 1

    # every reference number in the numbered References section is unique and 1..N
    ref_numbers = sorted(int(n) for n in re.findall(r"^\[(\d+)\]", review, re.MULTILINE))
    assert ref_numbers == list(range(1, len(sample_papers) + 1))
