"""PDF upload parsing: full text kept (chunked for retrieval elsewhere), not truncated."""
from __future__ import annotations

import re

from pypdf import PdfReader

from core.models import Paper

_ABSTRACT_RE = re.compile(
    r"abstract\s*[:.\-]?\s*(.+?)(?:\n\s*\n|\nkeywords|\nintroduction|\n1[.\s]+introduction)",
    re.IGNORECASE | re.DOTALL,
)


def _guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if 15 <= len(line) <= 200 and not line.lower().startswith("abstract"):
            return line
    return fallback


def _guess_abstract(text: str) -> str:
    m = _ABSTRACT_RE.search(text)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(1)).strip()[:3000]


def parse(uploaded_file) -> Paper | None:
    """uploaded_file: a file-like object (e.g. Streamlit's UploadedFile) or path."""
    try:
        reader = PdfReader(uploaded_file)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text).strip()
    except Exception:
        return None

    if not full_text:
        return None  # scanned PDF with no text layer

    meta = reader.metadata or {}
    title = (meta.get("/Title") or "").strip()
    filename = getattr(uploaded_file, "name", "uploaded.pdf")
    if not title:
        title = _guess_title(full_text, fallback=filename)

    return Paper(
        title=title,
        abstract=_guess_abstract(full_text),
        authors=[a.strip() for a in (meta.get("/Author") or "").split(";") if a.strip()],
        full_text=full_text,
        sources=["pdf"],
        url="",
    )
