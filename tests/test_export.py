from core.export import to_bibtex, to_csv, to_json, to_ris
from core.models import Paper


def test_bibtex_escapes_special_chars_and_double_braces_title():
    p = Paper(title="50% Faster & Better", authors=["Jane Doe"], year=2021)
    bib = to_bibtex([p])
    assert r"50\% Faster \& Better" in bib
    assert "{{50\\% Faster \\& Better}}" in bib


def test_bibtex_dedupes_keys_with_suffixes():
    p1 = Paper(title="Attention Is All You Need", authors=["Vaswani"], year=2017)
    p2 = Paper(title="Attention Is All You Need Revisited", authors=["Vaswani"], year=2017)
    # force identical bibtex_key by using the same title-first-word/author/year
    p2.title = "Attention Is All You Need"
    keys = to_bibtex([p1, p2])
    assert "vaswani2017attentiona," in keys
    assert "vaswani2017attentionb," in keys


def test_bibtex_includes_eprint_for_arxiv():
    p = Paper(title="Deep Learning", authors=["X"], year=2020, ids={"arxiv": "2001.00001"})
    bib = to_bibtex([p])
    assert "eprint = {2001.00001}" in bib
    assert "archivePrefix = {arXiv}" in bib


def test_csv_includes_theme_and_summary_columns():
    p = Paper(title="X", cluster=2, ai_summary="a good summary")
    csv_bytes = to_csv([p])
    text = csv_bytes.decode("utf-8")
    assert "theme" in text and "ai_summary" in text
    assert "a good summary" in text


def test_ris_has_required_fields():
    p = Paper(title="X", authors=["A B"], year=2020, doi="10.1/x")
    ris = to_ris([p])
    assert "TY  - JOUR" in ris
    assert "TI  - X" in ris
    assert "AU  - A B" in ris
    assert "PY  - 2020" in ris
    assert "ER  - " in ris


def test_json_round_trips_title():
    p = Paper(title="X")
    import json

    data = json.loads(to_json([p]))
    assert data[0]["title"] == "X"
