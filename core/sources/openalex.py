"""OpenAlex source. Requires an API key as of 13 Feb 2026, sent as the api_key query param."""
from __future__ import annotations

import re

import requests

from core.config import get_secret
from core.models import Paper

BASE_URL = "https://api.openalex.org/works"
SELECT_FIELDS = (
    "id,title,abstract_inverted_index,authorships,publication_year,publication_date,"
    "primary_location,doi,cited_by_count,open_access,ids"
)
_ARXIV_ABS_RE = re.compile(r"arxiv\.org/abs/([a-zA-Z0-9.\-/]+)")


def rebuild_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _extract_arxiv_id(work: dict) -> str | None:
    for loc_key in ("primary_location", "best_oa_location"):
        loc = work.get(loc_key) or {}
        landing = loc.get("landing_page_url") or ""
        m = _ARXIV_ABS_RE.search(landing)
        if m:
            return m.group(1)
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
    api_key = get_secret("OPENALEX_API_KEY")

    filters = ["has_abstract:true"]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    if open_access_only:
        filters.append("is_oa:true")

    params = {
        "search": query,
        "filter": ",".join(filters),
        "sort": "relevance_score:desc",
        "per_page": min(limit, 200),
        "select": SELECT_FIELDS,
    }
    if api_key:
        params["api_key"] = api_key

    resp = session.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for work in data.get("results", []):
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
            if a.get("author")
        ]
        primary_location = work.get("primary_location") or {}
        venue = (primary_location.get("source") or {}).get("display_name", "") or ""
        doi = (work.get("doi") or "").replace("https://doi.org/", "")
        oa = work.get("open_access") or {}
        ids = {"openalex": work.get("id", "")}
        arxiv_id = _extract_arxiv_id(work)
        if arxiv_id:
            ids["arxiv"] = arxiv_id

        papers.append(
            Paper(
                title=work.get("title") or "",
                abstract=rebuild_abstract(work.get("abstract_inverted_index")),
                authors=authors,
                year=work.get("publication_year"),
                published=work.get("publication_date", "") or "",
                venue=venue,
                doi=doi,
                url=primary_location.get("landing_page_url", "") or "",
                pdf_url=oa.get("oa_url", "") or "",
                citations=work.get("cited_by_count", 0) or 0,
                open_access=bool(oa.get("is_oa")),
                sources=["openalex"],
                ids=ids,
            )
        )
    return papers
