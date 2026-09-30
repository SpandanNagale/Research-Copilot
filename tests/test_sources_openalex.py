from core.sources import openalex


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.last_params = params
        return FakeResponse(self.payload)


OPENALEX_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Attention Is All You Need",
    "abstract_inverted_index": {"Attention": [0], "is": [1], "all": [2], "you": [3], "need": [4]},
    "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
    "publication_year": 2017,
    "publication_date": "2017-06-12",
    "primary_location": {
        "source": {"display_name": "NeurIPS"},
        "landing_page_url": "https://arxiv.org/abs/1706.03762",
    },
    "doi": "https://doi.org/10.48550/arxiv.1706.03762",
    "cited_by_count": 90000,
    "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/1706.03762"},
}


def test_rebuild_abstract_from_inverted_index():
    text = openalex.rebuild_abstract(OPENALEX_WORK["abstract_inverted_index"])
    assert text == "Attention is all you need"


def test_rebuild_abstract_handles_empty():
    assert openalex.rebuild_abstract(None) == ""
    assert openalex.rebuild_abstract({}) == ""


def test_search_parses_work_and_sends_api_key_as_query_param(monkeypatch):
    monkeypatch.setattr(openalex, "get_secret", lambda name: "test-key" if name == "OPENALEX_API_KEY" else None)
    session = FakeSession({"results": [OPENALEX_WORK]})

    papers = openalex.search("attention", limit=10, session=session)

    assert session.last_params["api_key"] == "test-key"
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Attention Is All You Need"
    assert p.abstract == "Attention is all you need"
    assert p.venue == "NeurIPS"
    assert p.year == 2017
    assert p.citations == 90000
    assert p.open_access is True
    assert p.doi == "10.48550/arxiv.1706.03762"
    assert p.ids["arxiv"] == "1706.03762"  # extracted from landing page URL
    assert p.ids["openalex"] == "https://openalex.org/W123"


def test_search_filters_include_year_and_oa(monkeypatch):
    monkeypatch.setattr(openalex, "get_secret", lambda name: None)
    session = FakeSession({"results": []})

    openalex.search("x", year_from=2020, year_to=2022, open_access_only=True, session=session)

    filt = session.last_params["filter"]
    assert "from_publication_date:2020-01-01" in filt
    assert "to_publication_date:2022-12-31" in filt
    assert "is_oa:true" in filt
