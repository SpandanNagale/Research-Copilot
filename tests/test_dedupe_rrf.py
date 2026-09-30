import core.sources as sources
from core.models import Paper


def _oa_paper():
    return Paper(
        title="Attention Is All You Need",
        abstract="Short abstract from OpenAlex.",
        authors=["Ashish Vaswani"],
        year=2017,
        citations=90000,
        sources=["openalex"],
        ids={"arxiv": "1706.03762", "openalex": "https://openalex.org/W1"},
    )


def _s2_paper():
    return Paper(
        title="Attention Is All You Need",
        abstract="A considerably longer abstract text describing the Transformer architecture in detail.",
        authors=["Ashish Vaswani"],
        year=2017,
        citations=85000,
        sources=["semantic_scholar"],
        ids={"arxiv": "1706.03762v1", "s2": "abc123"},  # versioned id, still matches
    )


def _unique_openalex_paper():
    return Paper(title="A totally unrelated paper on gene splicing", sources=["openalex"], ids={"openalex": "W2"})


def test_dedupe_merges_via_arxiv_id_and_fills_fields(monkeypatch):
    monkeypatch.setattr(
        sources,
        "SOURCE_FUNCS",
        {
            "openalex": lambda **kw: [_oa_paper(), _unique_openalex_paper()],
            "semantic_scholar": lambda **kw: [_s2_paper()],
        },
    )

    result = sources.search("attention", sources=["openalex", "semantic_scholar"])

    assert result["duplicates_merged"] == 1
    assert result["counts"] == {"openalex": 2, "semantic_scholar": 1}
    assert len(result["papers"]) == 2  # merged pair + the unrelated paper

    merged = next(p for p in result["papers"] if "arxiv" in p.ids)
    assert set(merged.sources) == {"openalex", "semantic_scholar"}
    assert merged.citations == 90000  # max of the two
    assert "considerably longer" in merged.abstract  # longest abstract kept
    assert merged.ids["s2"] == "abc123"  # union of ids


def test_rrf_ranks_multi_source_paper_above_single_source(monkeypatch):
    monkeypatch.setattr(
        sources,
        "SOURCE_FUNCS",
        {
            "openalex": lambda **kw: [_oa_paper(), _unique_openalex_paper()],
            "semantic_scholar": lambda **kw: [_s2_paper()],
        },
    )
    result = sources.search("attention", sources=["openalex", "semantic_scholar"], sort="relevance")
    papers = result["papers"]
    merged = next(p for p in papers if "arxiv" in p.ids)
    unrelated = next(p for p in papers if "arxiv" not in p.ids)
    assert merged.rank_score > unrelated.rank_score
    assert papers[0] is merged  # ranked first


def test_one_failing_source_reports_error_without_killing_run(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("503 Service Unavailable")

    monkeypatch.setattr(
        sources,
        "SOURCE_FUNCS",
        {"openalex": lambda **kw: [_oa_paper()], "semantic_scholar": _boom},
    )
    result = sources.search("x", sources=["openalex", "semantic_scholar"])
    assert result["counts"]["semantic_scholar"] == 0
    assert "semantic_scholar" in result["errors"]
    assert len(result["papers"]) == 1


def test_sort_by_recent_and_citations(monkeypatch):
    old = Paper(title="Old paper text here", year=2010, citations=5)
    new = Paper(title="New paper text here", year=2023, citations=1)
    monkeypatch.setattr(sources, "SOURCE_FUNCS", {"openalex": lambda **kw: [old, new]})

    by_recent = sources.search("x", sources=["openalex"], sort="recent")["papers"]
    assert by_recent[0] is new

    by_citations = sources.search("x", sources=["openalex"], sort="citations")["papers"]
    assert by_citations[0] is old
