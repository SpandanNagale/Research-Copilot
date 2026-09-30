"""CSV, BibTeX, RIS, JSON export."""
from __future__ import annotations

import dataclasses
import json
from collections import Counter

from core.models import Paper

_BIBTEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def to_csv(papers: list[Paper]) -> bytes:
    import pandas as pd

    rows = [
        {
            "title": p.title,
            "authors": "; ".join(p.authors),
            "year": p.year,
            "venue": p.venue,
            "doi": p.doi,
            "url": p.url,
            "citations": p.citations,
            "open_access": p.open_access,
            "sources": ";".join(p.sources),
            "theme": p.cluster,
            "ai_summary": p.ai_summary,
            "abstract": p.abstract,
        }
        for p in papers
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _escape_bibtex(s: str) -> str:
    for k, v in _BIBTEX_ESCAPES.items():
        s = s.replace(k, v)
    return s


def _bibtex_keys(papers: list[Paper]) -> list[str]:
    base_keys = [p.bibtex_key() for p in papers]
    counts = Counter(base_keys)
    seen: Counter = Counter()
    keys = []
    for base_key in base_keys:
        if counts[base_key] > 1:
            keys.append(f"{base_key}{chr(ord('a') + seen[base_key])}")
            seen[base_key] += 1
        else:
            keys.append(base_key)
    return keys


def to_bibtex(papers: list[Paper]) -> str:
    keys = _bibtex_keys(papers)
    entries = []
    for p, key in zip(papers, keys):
        fields = [f"  title = {{{{{_escape_bibtex(p.title)}}}}}"]
        if p.authors:
            fields.append(f"  author = {{{' and '.join(_escape_bibtex(a) for a in p.authors)}}}")
        if p.year:
            fields.append(f"  year = {{{p.year}}}")
        if p.venue:
            fields.append(f"  journal = {{{_escape_bibtex(p.venue)}}}")
        if p.doi:
            fields.append(f"  doi = {{{p.doi}}}")
        if p.url:
            fields.append(f"  url = {{{p.url}}}")
        arxiv_id = p.ids.get("arxiv")
        if arxiv_id:
            fields.append(f"  eprint = {{{arxiv_id}}}")
            fields.append("  archivePrefix = {arXiv}")
        entries.append(f"@misc{{{key},\n" + ",\n".join(fields) + "\n}")
    return "\n\n".join(entries)


def to_ris(papers: list[Paper]) -> str:
    lines = []
    for p in papers:
        lines.append("TY  - JOUR")
        for a in p.authors:
            lines.append(f"AU  - {a}")
        lines.append(f"TI  - {p.title}")
        if p.year:
            lines.append(f"PY  - {p.year}")
        if p.venue:
            lines.append(f"JO  - {p.venue}")
        if p.doi:
            lines.append(f"DO  - {p.doi}")
        if p.url:
            lines.append(f"UR  - {p.url}")
        if p.abstract:
            lines.append(f"AB  - {p.abstract}")
        lines.append("ER  - ")
        lines.append("")
    return "\n".join(lines)


def to_json(papers: list[Paper]) -> str:
    return json.dumps([dataclasses.asdict(p) for p in papers], indent=2, default=str)
