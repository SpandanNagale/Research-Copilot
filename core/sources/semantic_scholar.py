"""Semantic Scholar Graph API source."""
from __future__ import annotations

import requests

from core.config import get_secret
from core.models import Paper

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = (
    "title,abstract,authors,year,venue,citationCount,externalIds,url,"
    "openAccessPdf,publicationDate,isOpenAccess"
)


def _year_param(year_from: int | None, year_to: int | None) -> str | None:
    if year_from and year_to:
        return f"{year_from}-{year_to}"
    if year_from:
        return f"{year_from}-"
    if year_to:
        return f"-{year_to}"
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
    api_key = get_secret("SEMANTIC_SCHOLAR_API_KEY")

    params = {"query": query, "limit": min(limit, 100), "fields": FIELDS}
    year_param = _year_param(year_from, year_to)
    if year_param:
        params["year"] = year_param
    if open_access_only:
        params["openAccessPdf"] = ""

    headers = {"x-api-key": api_key} if api_key else {}

    resp = session.get(BASE_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for item in data.get("data", []):
        external_ids = item.get("externalIds") or {}
        oa_pdf = (item.get("openAccessPdf") or {}).get("url", "") or ""
        ids = {"s2": item.get("paperId", "")}
        if external_ids.get("ArXiv"):
            ids["arxiv"] = external_ids["ArXiv"]
        if external_ids.get("PubMed"):
            ids["pmid"] = external_ids["PubMed"]

        papers.append(
            Paper(
                title=item.get("title") or "",
                abstract=item.get("abstract") or "",
                authors=[a.get("name", "") for a in item.get("authors", [])],
                year=item.get("year"),
                published=item.get("publicationDate", "") or "",
                venue=item.get("venue", "") or "",
                doi=external_ids.get("DOI", "") or "",
                url=item.get("url", "") or "",
                pdf_url=oa_pdf,
                citations=item.get("citationCount", 0) or 0,
                open_access=bool(item.get("isOpenAccess")),
                sources=["semantic_scholar"],
                ids=ids,
            )
        )
    return papers
