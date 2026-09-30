"""Multi-source aggregator: parallel fetch, dedupe, RRF ranking."""
from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.models import Paper
from core.sources import arxiv_source, openalex, pubmed, semantic_scholar

RRF_K = 60

SOURCE_FUNCS: dict[str, Callable[..., list[Paper]]] = {
    "arxiv": arxiv_source.search,
    "openalex": openalex.search,
    "pubmed": pubmed.search,
    "semantic_scholar": semantic_scholar.search,
}


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _friendly_error(exc: Exception) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return "Unauthorized: check the API key for this source."
    if status == 403:
        return "Forbidden: this source rejected the request (check API key / access)."
    if status == 429:
        return "Rate limited: too many requests to this source right now."
    return str(exc)


def search(
    query: str,
    sources: list[str],
    per_source: int = 25,
    year_from: int | None = None,
    year_to: int | None = None,
    open_access_only: bool = False,
    sort: str = "relevance",
) -> dict:
    """Fan out to each requested source in parallel, dedupe, and RRF-rank.

    Returns {"papers": [...], "counts": {source: n}, "errors": {source: msg}, "duplicates_merged": n}
    """
    session = make_session()
    counts: dict[str, int] = {}
    errors: dict[str, str] = {}
    per_source_results: dict[str, list[Paper]] = {}

    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as pool:
        future_to_source = {
            pool.submit(
                SOURCE_FUNCS[src],
                query=query,
                limit=per_source,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
                session=session,
            ): src
            for src in sources
            if src in SOURCE_FUNCS
        }
        for future in as_completed(future_to_source):
            src = future_to_source[future]
            try:
                papers = future.result()
                per_source_results[src] = papers
                counts[src] = len(papers)
            except Exception as exc:  # one failing source never kills the run
                errors[src] = _friendly_error(exc)
                counts[src] = 0

    merged, duplicates_merged = _dedupe_and_merge(per_source_results)
    _apply_rrf(merged, per_source_results)
    merged = _sort_papers(merged, sort)

    return {
        "papers": merged,
        "counts": counts,
        "errors": errors,
        "duplicates_merged": duplicates_merged,
    }


def _dedupe_and_merge(per_source_results: dict[str, list[Paper]]) -> tuple[list[Paper], int]:
    key_to_paper: dict[str, Paper] = {}
    all_keys_for_paper: dict[int, list[str]] = {}
    order: list[Paper] = []
    duplicates_merged = 0

    for src, papers in per_source_results.items():
        for p in papers:
            keys = p.match_keys()
            existing = None
            for k in keys:
                if k in key_to_paper:
                    existing = key_to_paper[k]
                    break

            if existing is None:
                order.append(p)
                all_keys_for_paper[id(p)] = keys
                for k in keys:
                    key_to_paper[k] = p
            else:
                duplicates_merged += 1
                _merge_into(existing, p)
                for k in keys:
                    key_to_paper[k] = existing

    return order, duplicates_merged


def _merge_into(target: Paper, other: Paper) -> None:
    if len(other.abstract) > len(target.abstract):
        target.abstract = other.abstract
    target.citations = max(target.citations, other.citations)
    target.sources = list(dict.fromkeys(target.sources + other.sources))
    target.ids = {**other.ids, **target.ids}  # keep target's ids, fill gaps from other
    target.open_access = target.open_access or other.open_access
    for attr in ("year", "venue", "doi", "url", "pdf_url", "published"):
        if not getattr(target, attr) and getattr(other, attr):
            setattr(target, attr, getattr(other, attr))
    if not target.authors and other.authors:
        target.authors = other.authors


def _apply_rrf(merged: list[Paper], per_source_results: dict[str, list[Paper]]) -> None:
    id_to_rank: dict[int, dict[str, int]] = defaultdict(dict)
    for src, papers in per_source_results.items():
        for rank, p in enumerate(papers):
            id_to_rank[id(p)][src] = rank

    # Map original per-source paper objects to the (possibly merged-into) survivors by identity
    # of their match keys, since merged papers keep the identity of the first-seen object.
    key_to_merged: dict[str, Paper] = {}
    for p in merged:
        for k in p.match_keys():
            key_to_merged[k] = p

    scores: dict[int, float] = defaultdict(float)
    for src, papers in per_source_results.items():
        for rank, p in enumerate(papers):
            target = None
            for k in p.match_keys():
                if k in key_to_merged:
                    target = key_to_merged[k]
                    break
            if target is None:
                continue
            scores[id(target)] += 1.0 / (RRF_K + rank + 1)

    for p in merged:
        p.rank_score = scores[id(p)]


def _sort_papers(papers: list[Paper], sort: str) -> list[Paper]:
    if sort == "recent":
        return sorted(papers, key=lambda p: p.year or 0, reverse=True)
    if sort == "citations":
        return sorted(papers, key=lambda p: p.citations, reverse=True)
    return sorted(papers, key=lambda p: p.rank_score, reverse=True)
