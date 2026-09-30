"""arXiv source (arxiv package v4.x)."""
from __future__ import annotations

import arxiv

from core.models import Paper


def build_query(query: str, year_from: int | None, year_to: int | None) -> str:
    if not year_from and not year_to:
        return query
    lo = year_from or 1990
    hi = year_to or 2100
    return f"({query}) AND submittedDate:[{lo}01010000 TO {hi}12312359]"


def search(
    query: str,
    limit: int = 25,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    session=None,
) -> list[Paper]:
    full_query = build_query(query, year_from, year_to)
    client = arxiv.Client()
    search_obj = arxiv.Search(
        query=full_query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for r in client.results(search_obj):
        papers.append(
            Paper(
                title=r.title.strip(),
                abstract=r.summary.strip().replace("\n", " "),
                authors=[a.name for a in r.authors],
                year=r.published.year if r.published else None,
                published=str(r.published.date()) if r.published else "",
                venue="arXiv",
                url=r.entry_id,
                pdf_url=r.pdf_url,
                open_access=True,
                sources=["arxiv"],
                ids={"arxiv": r.get_short_id()},
            )
        )
    return papers
