"""Per-paper summaries, AI theme naming, and map-reduce literature review generation."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from core.knowledge import KnowledgeBase
from core.llm import PROVIDERS, LLMClient
from core.models import Paper

SUMMARY_PROMPT = """Summarize the following abstract into exactly three structured sections. \
No conversational filler.

Abstract:
{text}

Output format:
**Problem:** [1 sentence on the research gap]
**Method:** [1-2 sentences on the specific model/dataset/algorithm used]
**Result:** [1 sentence on the key metric or finding]
"""


def summarize_papers(
    papers: list[Paper],
    client: LLMClient,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Mutates paper.ai_summary in place. Stops submitting new work once the
    first 3 completed calls have all failed."""
    workers = PROVIDERS[client.provider]["parallelism"]

    def _do(p: Paper) -> str:
        return client.complete(SUMMARY_PROMPT.format(text=f"{p.title}\n\n{p.abstract}"))

    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_do, p): p for p in papers}
        for future in as_completed(futures):
            p = futures[future]
            completed += 1
            try:
                p.ai_summary = future.result()
            except Exception as exc:
                p.ai_summary = f"Summary unavailable: {exc}"
                failures += 1
            if on_progress:
                on_progress(completed, len(papers))
            if completed == 3 and failures == 3:
                for f in futures:
                    f.cancel()
                break


def name_themes_ai(papers: list[Paper], client: LLMClient) -> dict[int, str]:
    clusters = sorted({p.cluster for p in papers if p.cluster is not None})
    rep_titles = {
        str(c): [p.title for p in papers if p.cluster == c][:5] for c in clusters
    }
    prompt = (
        "Given these clusters of paper titles, respond with ONLY a JSON object mapping "
        "each cluster number (as a string) to a short 2-4 word theme name.\n\n"
        f"Clusters:\n{json.dumps(rep_titles, indent=2)}\n\nJSON:"
    )
    try:
        raw = client.complete(prompt, system="You output only valid JSON, no prose.")
    except Exception:
        return {c: f"Theme {c + 1}" for c in clusters}
    return _parse_theme_json(raw, clusters)


def _parse_theme_json(raw: str, clusters: list[int]) -> dict[int, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    result = {}
    for c in clusters:
        val = data.get(str(c)) or data.get(c)
        result[c] = val.strip() if isinstance(val, str) and val.strip() else f"Theme {c + 1}"
    return result


REVIEW_SECTION_PROMPT = """Write a literature review section (~{words} words) for the theme \
"{theme}" using ONLY the numbered sources below. Cite with [n] matching the reference numbers \
given, attached to the statements they support.

{refs}

Output only the section body (no heading).
"""

REDUCE_PROMPT = """You are given the theme sections of a literature review (with [n] citations \
already assigned). Write exactly these three sections, in this order, using the SAME [n] numbers:

## Introduction
## Research gaps and open questions
## Conclusion

Theme sections:
{sections}

Output ONLY those three headed sections.
"""

_SECTION_SPLIT_RE = re.compile(
    r"##\s*(Introduction|Research gaps and open questions|Conclusion)\s*\n", re.IGNORECASE
)


def _split_reduce_sections(text: str) -> tuple[str, str, str]:
    matches = list(_SECTION_SPLIT_RE.finditer(text))
    if len(matches) < 3:
        return (
            f"## Introduction\n\n{text.strip()}",
            "## Research gaps and open questions\n\n(Not generated.)",
            "## Conclusion\n\n(Not generated.)",
        )
    parts: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = m.group(1).lower().split()[0]  # introduction / research / conclusion
        parts[key] = text[m.start():end].strip()
    return (
        parts.get("introduction", "## Introduction\n\n(missing)"),
        parts.get("research", "## Research gaps and open questions\n\n(missing)"),
        parts.get("conclusion", "## Conclusion\n\n(missing)"),
    )


LENGTH_WORDS = {"short": 150, "medium": 300, "long": 500}


def generate_review(
    papers: list[Paper],
    kb: KnowledgeBase,
    client: LLMClient,
    papers_per_theme: int = 10,
    length: str = "medium",
    theme_names: dict[int, str] | None = None,
) -> str:
    clusters = sorted({p.cluster for p in papers if p.cluster is not None})
    theme_names = theme_names or {c: f"Theme {c + 1}" for c in clusters}
    words = LENGTH_WORDS.get(length, 300)

    # Global reference numbers assigned before any prompting so [n] stays consistent.
    ref_number = {id(p): i + 1 for i, p in enumerate(papers)}

    def _map_section(c: int) -> tuple[int, str]:
        selected = kb.centroid_nearest(c, papers_per_theme)
        refs = "\n".join(
            f"[{ref_number[id(p)]}] {p.title}: {p.abstract[:500]}" for p in selected
        )
        prompt = REVIEW_SECTION_PROMPT.format(words=words, theme=theme_names[c], refs=refs)
        body = client.complete(prompt, system="You are a rigorous academic writer.")
        return c, f"## {theme_names[c]}\n\n{body}"

    workers = PROVIDERS[client.provider]["parallelism"]
    sections: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for c, text in pool.map(_map_section, clusters):
            sections[c] = text

    section_bodies = "\n\n".join(sections[c] for c in clusters)

    reduce_text = client.complete(
        REDUCE_PROMPT.format(sections=section_bodies),
        system="You are a rigorous academic writer.",
    )
    intro, gaps, conclusion = _split_reduce_sections(reduce_text)

    references = "\n".join(f"[{ref_number[id(p)]}] {p.citation()}" for p in papers)

    return "\n\n".join([intro, section_bodies, gaps, conclusion, f"## References\n\n{references}"])
