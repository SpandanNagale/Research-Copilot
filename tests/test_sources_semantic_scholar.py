from core.sources import semantic_scholar


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
        self.last_headers = None
        self.last_params = None

    def get(self, url, params=None, headers=None, timeout=None):
        self.last_params = params
        self.last_headers = headers
        return FakeResponse(self.payload)


S2_PAPER = {
    "paperId": "abc123",
    "title": "Attention Is All You Need",
    "abstract": "We propose the Transformer.",
    "authors": [{"name": "Ashish Vaswani"}],
    "year": 2017,
    "venue": "NeurIPS",
    "citationCount": 90000,
    "externalIds": {"DOI": "10.48550/arxiv.1706.03762", "ArXiv": "1706.03762"},
    "url": "https://www.semanticscholar.org/paper/abc123",
    "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762"},
    "publicationDate": "2017-06-12",
    "isOpenAccess": True,
}


def test_search_parses_paper_and_sends_key_as_header(monkeypatch):
    monkeypatch.setattr(semantic_scholar, "get_secret", lambda name: "s2-key")
    session = FakeSession({"data": [S2_PAPER]})

    papers = semantic_scholar.search("attention", session=session)

    assert session.last_headers == {"x-api-key": "s2-key"}
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Attention Is All You Need"
    assert p.doi == "10.48550/arxiv.1706.03762"
    assert p.ids["arxiv"] == "1706.03762"
    assert p.ids["s2"] == "abc123"
    assert p.citations == 90000
    assert p.open_access is True


def test_year_param_formats():
    assert semantic_scholar._year_param(2020, 2024) == "2020-2024"
    assert semantic_scholar._year_param(2020, None) == "2020-"
    assert semantic_scholar._year_param(None, 2024) == "-2024"
    assert semantic_scholar._year_param(None, None) is None
