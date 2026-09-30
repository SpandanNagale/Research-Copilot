"""Paper data model shared by every source, the knowledge base, and export."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def _strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    published: str = ""
    venue: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    citations: int = 0
    open_access: bool = False
    sources: list[str] = field(default_factory=list)
    ids: dict = field(default_factory=dict)  # arxiv, pmid, pmcid, openalex, s2
    full_text: str = ""
    cluster: int | None = None
    ai_summary: str = ""
    rank_score: float = 0.0

    def match_keys(self) -> list[str]:
        """Keys other papers can be deduped against, most authoritative first."""
        keys = []
        if self.doi:
            keys.append(f"doi:{self.doi.strip().lower()}")
        arxiv_id = self.ids.get("arxiv")
        if arxiv_id:
            keys.append(f"arxiv:{_strip_arxiv_version(arxiv_id).lower()}")
        pmid = self.ids.get("pmid")
        if pmid:
            keys.append(f"pmid:{str(pmid).strip()}")
        norm_title = _normalize_title(self.title)
        if len(norm_title) > 20:
            keys.append(f"title:{norm_title}")
        return keys

    def citation(self) -> str:
        """APA-style single line."""
        if not self.authors:
            author_str = "Anon."
        elif len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) <= 3:
            author_str = ", ".join(self.authors[:-1]) + f", & {self.authors[-1]}"
        else:
            author_str = f"{self.authors[0]} et al."

        year_str = f"({self.year})" if self.year else "(n.d.)"
        parts = [author_str, year_str, f"{self.title}."]
        if self.venue:
            parts.append(f"{self.venue}.")
        link = self.doi and f"https://doi.org/{self.doi}" or self.url
        if link:
            parts.append(link)
        return " ".join(parts)

    def bibtex_key(self) -> str:
        last_name = "anon"
        if self.authors:
            last_name = re.sub(r"[^a-z]", "", self.authors[0].split()[-1].lower()) or "anon"
        year = str(self.year) if self.year else "nd"
        words = re.findall(r"[a-zA-Z]+", self.title)
        first_word = words[0].lower() if words else "untitled"
        return f"{last_name}{year}{first_word}"
