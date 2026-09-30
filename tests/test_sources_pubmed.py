from core.sources import pubmed

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue>
          <Title>Journal of Testing</Title>
        </Journal>
        <ArticleTitle>Effects of <i>Drug X</i> on outcomes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText Label="METHODS">Methods text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
          <Author><CollectiveName>Research Group</CollectiveName></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345</ArticleId>
        <ArticleId IdType="doi">10.1000/abc</ArticleId>
        <ArticleId IdType="pmc">PMC7654321</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>999</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue>
          <Title>Other Journal</Title>
        </Journal>
        <ArticleTitle>A second article</ArticleTitle>
        <Abstract>
          <AbstractText>Plain unlabeled abstract.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
        </AuthorList>
        <ELocationID EIdType="doi">10.2000/xyz</ELocationID>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">999</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class FakeResp:
    def __init__(self, json_data=None, content=None):
        self._json = json_data
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeSession:
    def __init__(self, esearch_json, efetch_xml):
        self.esearch_json = esearch_json
        self.efetch_xml = efetch_xml
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if "esearch" in url:
            return FakeResp(json_data=self.esearch_json)
        return FakeResp(content=self.efetch_xml.encode())


def test_search_parses_structured_abstract_collective_name_inline_tags_and_pmc(monkeypatch):
    monkeypatch.setattr(pubmed, "get_secret", lambda name: None)
    session = FakeSession(
        esearch_json={"esearchresult": {"idlist": ["12345", "999"]}},
        efetch_xml=PUBMED_XML,
    )

    papers = pubmed.search("drug x", session=session)

    assert len(papers) == 2
    p1 = papers[0]
    assert p1.title == "Effects of Drug X on outcomes"  # inline <i> flattened via itertext
    assert "BACKGROUND: Background text." in p1.abstract
    assert "METHODS: Methods text." in p1.abstract
    assert p1.authors == ["John Smith", "Research Group"]
    assert p1.doi == "10.1000/abc"
    assert p1.ids["pmcid"] == "PMC7654321"
    assert p1.open_access is True
    assert p1.year == 2020

    p2 = papers[1]
    assert p2.doi == "10.2000/xyz"  # fallback to ELocationID when no ArticleIdList doi
    assert p2.open_access is False


def test_search_preserves_esearch_ranking_order(monkeypatch):
    monkeypatch.setattr(pubmed, "get_secret", lambda name: None)
    session = FakeSession(
        esearch_json={"esearchresult": {"idlist": ["999", "12345"]}},
        efetch_xml=PUBMED_XML,
    )
    papers = pubmed.search("x", session=session)
    assert [p.ids["pmid"] for p in papers] == ["999", "12345"]


def test_open_access_only_appends_filter_string(monkeypatch):
    monkeypatch.setattr(pubmed, "get_secret", lambda name: None)
    session = FakeSession(esearch_json={"esearchresult": {"idlist": []}}, efetch_xml=PUBMED_XML)
    pubmed.search("x", open_access_only=True, session=session)
    esearch_call = session.calls[0]
    assert "free full text[filter]" in esearch_call[1]["term"]


def test_year_range_sets_mindate_maxdate(monkeypatch):
    monkeypatch.setattr(pubmed, "get_secret", lambda name: None)
    session = FakeSession(esearch_json={"esearchresult": {"idlist": []}}, efetch_xml=PUBMED_XML)
    pubmed.search("x", year_from=2018, year_to=2021, session=session)
    params = session.calls[0][1]
    assert params["mindate"] == "2018"
    assert params["maxdate"] == "2021"
    assert params["datetype"] == "pdat"
