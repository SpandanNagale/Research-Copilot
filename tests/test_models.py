from core.models import Paper


def test_match_keys_prefers_doi_then_arxiv_then_pmid_then_title():
    p = Paper(
        title="A Long Enough Title For Matching",
        doi="10.1000/xyz",
        ids={"arxiv": "1234.5678v2", "pmid": "999"},
    )
    keys = p.match_keys()
    assert keys[0] == "doi:10.1000/xyz"
    assert "arxiv:1234.5678" in keys  # version stripped
    assert "pmid:999" in keys
    assert any(k.startswith("title:") for k in keys)


def test_match_keys_skips_short_titles():
    p = Paper(title="Short")
    assert not any(k.startswith("title:") for k in p.match_keys())


def test_citation_apa_single_author():
    p = Paper(title="Attention Is All You Need", authors=["Vaswani"], year=2017, venue="NeurIPS")
    c = p.citation()
    assert "Vaswani" in c
    assert "(2017)" in c
    assert "Attention Is All You Need." in c
    assert "NeurIPS." in c


def test_citation_apa_multiple_authors():
    p = Paper(title="X", authors=["A", "B", "C", "D"], year=2020)
    c = p.citation()
    assert "A et al." in c


def test_citation_apa_two_to_three_authors():
    p = Paper(title="X", authors=["A", "B"], year=2020)
    assert "A, & B" in p.citation()


def test_bibtex_key():
    p = Paper(title="Attention Is All You Need", authors=["Ashish Vaswani"], year=2017)
    assert p.bibtex_key() == "vaswani2017attention"


def test_bibtex_key_no_author_or_year():
    p = Paper(title="Untitled Work")
    assert p.bibtex_key() == "anonnduntitled"
