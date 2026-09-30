"""PubMed source via NCBI E-utilities. Keeps PubMed's own ranking order."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import requests

from core.config import get_secret
from core.models import Paper

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "research-copilot"


def _common_params() -> dict:
    params = {"tool": TOOL_NAME}
    api_key = get_secret("NCBI_API_KEY")
    email = get_secret("NCBI_EMAIL")
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    return params


def _article_title(article_el: ET.Element) -> str:
    title_el = article_el.find(".//ArticleTitle")
    if title_el is None:
        return ""
    return "".join(title_el.itertext()).strip()


def _abstract(article_el: ET.Element) -> str:
    parts = []
    for ab in article_el.findall(".//Abstract/AbstractText"):
        text = "".join(ab.itertext()).strip()
        label = ab.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _authors(article_el: ET.Element) -> list[str]:
    names = []
    for author_el in article_el.findall(".//AuthorList/Author"):
        collective = author_el.find("CollectiveName")
        if collective is not None and collective.text:
            names.append(collective.text.strip())
            continue
        last = author_el.find("LastName")
        fore = author_el.find("ForeName")
        if last is not None:
            name = last.text or ""
            if fore is not None and fore.text:
                name = f"{fore.text} {name}"
            names.append(name.strip())
    return names


def _doi_and_pmc(pubmed_article_el: ET.Element) -> tuple[str, str]:
    doi = ""
    pmcid = ""
    for id_el in pubmed_article_el.findall(".//ArticleIdList/ArticleId"):
        id_type = id_el.get("IdType", "")
        if id_type == "doi" and id_el.text:
            doi = id_el.text.strip()
        elif id_type == "pmc" and id_el.text:
            pmcid = id_el.text.strip()
    if not doi:
        for eloc in pubmed_article_el.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi" and eloc.text:
                doi = eloc.text.strip()
    return doi, pmcid


def _venue(article_el: ET.Element) -> str:
    journal_el = article_el.find(".//Journal/Title")
    return journal_el.text.strip() if journal_el is not None and journal_el.text else ""


def _year(article_el: ET.Element) -> int | None:
    year_el = article_el.find(".//JournalIssue/PubDate/Year")
    if year_el is not None and year_el.text and year_el.text.isdigit():
        return int(year_el.text)
    medline_date = article_el.find(".//JournalIssue/PubDate/MedlineDate")
    if medline_date is not None and medline_date.text:
        digits = "".join(c for c in medline_date.text[:4] if c.isdigit())
        if digits:
            return int(digits)
    return None


def search(
    query: str,
    limit: int = 25,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    session: requests.Session | None = None,
) -> list[Paper]:
    session = session or requests.Session()

    full_query = query
    if open_access_only:
        full_query = f"{query} AND free full text[filter]"

    esearch_params = {
        **_common_params(),
        "db": "pubmed",
        "retmode": "json",
        "term": full_query,
        "retmax": limit,
        "sort": "relevance",
    }
    if year_from or year_to:
        esearch_params["datetype"] = "pdat"
        esearch_params["mindate"] = str(year_from or 1900)
        esearch_params["maxdate"] = str(year_to or 2100)

    esearch_resp = session.get(ESEARCH_URL, params=esearch_params, timeout=30)
    esearch_resp.raise_for_status()
    id_list = esearch_resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    efetch_params = {
        **_common_params(),
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml",
    }
    efetch_resp = session.get(EFETCH_URL, params=efetch_params, timeout=30)
    efetch_resp.raise_for_status()
    root = ET.fromstring(efetch_resp.content)

    by_pmid: dict[str, Paper] = {}
    for pubmed_article in root.findall(".//PubmedArticle"):
        article_el = pubmed_article.find(".//Article")
        if article_el is None:
            continue
        pmid_el = pubmed_article.find(".//PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
        doi, pmcid = _doi_and_pmc(pubmed_article)
        ids = {"pmid": pmid}
        if pmcid:
            ids["pmcid"] = pmcid

        by_pmid[pmid] = Paper(
            title=_article_title(article_el),
            abstract=_abstract(article_el),
            authors=_authors(article_el),
            year=_year(article_el),
            venue=_venue(article_el),
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
            pdf_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "",
            open_access=bool(pmcid),
            sources=["pubmed"],
            ids=ids,
        )

    # esearch's idlist order is PubMed's own relevance ranking; preserve it.
    return [by_pmid[pmid] for pmid in id_list if pmid in by_pmid]
