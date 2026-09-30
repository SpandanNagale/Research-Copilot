"""Reusable UI pieces: library cards, cluster map, empty/error states."""
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.models import Paper

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


_CITATION_RE = re.compile(r"\[(\d+)\]")


def highlight_citations(text: str) -> str:
    """Wraps [n] citation markers in the highlighter-yellow accent span."""
    return _CITATION_RE.sub(r'<span class="citation-marker">[\1]</span>', text)


def _summary_to_html(text: str) -> str:
    # ai_summary is our own fixed **label:** markdown; a raw <div> wrapper (needed
    # for the highlighter-yellow rule) won't re-parse markdown inside it, so
    # convert the small subset we actually emit (bold + line breaks) ourselves.
    html = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    return html.replace("\n", "<br>")


def paper_card(p: Paper, theme_names: dict[int, str] | None = None) -> None:
    with st.container(border=True):
        link = p.url or p.pdf_url
        title_md = f"**[{p.title}]({link})**" if link else f"**{p.title}**"
        st.markdown(title_md)
        st.caption(p.citation())

        badges = " ".join(f'<span class="source-badge">{s}</span>' for s in p.sources)
        oa = '<span class="oa-marker">Open access</span>' if p.open_access else ""
        theme_label = ""
        if theme_names is not None and p.cluster is not None:
            theme_label = f" &middot; {theme_names.get(p.cluster, f'Theme {p.cluster + 1}')}"
        st.markdown(
            f"{badges} {oa} &nbsp;&middot;&nbsp; {p.citations} citations{theme_label}",
            unsafe_allow_html=True,
        )

        if p.abstract:
            with st.expander("Abstract"):
                st.write(p.abstract)

        if p.ai_summary:
            st.markdown(f'<div class="ai-summary">{_summary_to_html(p.ai_summary)}</div>', unsafe_allow_html=True)


def cluster_map(papers: list[Paper], coords: np.ndarray, theme_names: dict[int, str]) -> None:
    if len(papers) == 0:
        empty_state("No papers to plot yet.", "Run a search first.")
        return

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "title": [p.title for p in papers],
            "year": [p.year or "" for p in papers],
            "theme": [
                theme_names.get(p.cluster, f"Theme {p.cluster + 1}") if p.cluster is not None else "Unclustered"
                for p in papers
            ],
            "type": ["Upload" if "pdf" in p.sources else "Search result" for p in papers],
            "size": [np.log1p(max(p.citations, 0)) + 4 for p in papers],
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="theme",
        symbol="type",
        size="size",
        size_max=18,
        hover_data=["title", "year"],
        template="plotly_white",
    )
    fig.update_layout(
        font_family="Public Sans, sans-serif",
        legend_title_text="",
        xaxis_title=None,
        yaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")


def empty_state(message: str, action: str = "") -> None:
    st.info(f"{message}" + (f" {action}" if action else ""))


def source_errors(errors: dict[str, str]) -> None:
    for src, msg in errors.items():
        st.warning(f"{src}: {msg}")
