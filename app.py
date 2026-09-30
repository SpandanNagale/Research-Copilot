import os

import streamlit as st

from core.config import get_secret
from core.export import to_bibtex, to_csv, to_json, to_ris
from core.knowledge import KnowledgeBase, label_themes_tfidf
from core.llm import DEFAULT_PROVIDER_ORDER, FallbackLLMClient, LLMError, display_name
from core.rag import answer_stream
from core.sources import search as source_search
from core.sources import pdf as pdf_source
from core.synthesis import generate_review, name_themes_ai, summarize_papers
from ui import theme
from ui.components import cluster_map, empty_state, highlight_citations, paper_card, source_errors

st.set_page_config(page_title="Research Copilot", layout="wide", page_icon="\U0001F4DA")
theme.inject()

DATA_SOURCE_KEYS = {
    "OpenAlex": "OPENALEX_API_KEY",
    "NCBI (PubMed)": "NCBI_API_KEY",
    "NCBI email": "NCBI_EMAIL",
    "Semantic Scholar": "SEMANTIC_SCHOLAR_API_KEY",
}

# --- session state ---------------------------------------------------------
for key, default in {
    "papers": [],
    "kb": None,
    "coords": None,
    "theme_names": {},
    "run_summary": "",
    "run_errors": {},
    "messages": [],
    "review_md": "",
    "authenticated": False,
}.items():
    st.session_state.setdefault(key, default)


# --- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Engine")
    st.caption(
        "Tries " + " → ".join(display_name(p) for p in DEFAULT_PROVIDER_ORDER)
        + " automatically, moving on if one fails."
    )

    app_password = get_secret("APP_PASSWORD")
    server_keys_usable = True
    if app_password and not st.session_state.authenticated:
        pw = st.text_input("App password", type="password")
        if pw and pw == app_password:
            st.session_state.authenticated = True
            st.rerun()
        server_keys_usable = False
        st.warning("Enter the app password to use the configured LLM providers.")

    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)

    def _resolve_client() -> FallbackLLMClient | None:
        if not server_keys_usable:
            return None
        return FallbackLLMClient(temperature=temperature)

    if st.button("Test connection"):
        client = _resolve_client()
        if client is None:
            st.error("Enter the app password first.")
        else:
            ok, msg = client.test_connection()
            (st.success if ok else st.error)(msg)

    with st.expander("Data source keys"):
        for label, env_name in DATA_SOURCE_KEYS.items():
            val = st.text_input(label, value="", type="password" if "email" not in env_name.lower() else "default", key=f"dsk_{env_name}")
            if val:
                os.environ[env_name] = val


# --- hero --------------------------------------------------------------------
st.markdown("### What are you researching?")
query = st.text_input("Research query", placeholder="e.g. graph neural networks for drug discovery", label_visibility="collapsed")

ALL_SOURCES = ["arxiv", "openalex", "pubmed", "semantic_scholar"]
selected_sources = st.pills("Sources", ALL_SOURCES, default=ALL_SOURCES, selection_mode="multi", format_func=lambda s: s.replace("_", " ").title())

uploaded_files = st.file_uploader("Upload PDFs (optional)", type=["pdf"], accept_multiple_files=True)

with st.expander("Filters"):
    c1, c2, c3 = st.columns(3)
    per_source = c1.slider("Results per source", 5, 50, 20)
    n_themes = c2.slider("Number of themes", 2, 10, 5)
    papers_per_theme = c3.slider("Papers per theme in review", 3, 20, 10)
    year_from, year_to = st.slider("Year range", 1990, 2026, (2015, 2026))
    sort = st.selectbox("Sort by", ["relevance", "recent", "citations"])
    oa_only = st.checkbox("Open access only")
    auto_summarize = st.checkbox("Auto-summarize papers after search")

run_btn = st.button("Search", type="primary")


def run_pipeline() -> None:
    st.session_state.messages = []
    st.session_state.review_md = ""

    with st.status("Working...", expanded=True) as status:
        papers = []
        errors = {}
        counts = {}
        duplicates_merged = 0

        if uploaded_files:
            status.write(f"Reading {len(uploaded_files)} uploaded PDF(s)...")
            for f in uploaded_files:
                parsed = pdf_source.parse(f)
                if parsed:
                    papers.append(parsed)
                else:
                    errors["pdf"] = f"Couldn't read {getattr(f, 'name', 'a PDF')} (no text layer?)."

        if query.strip() and selected_sources:
            status.write(f"Searching {', '.join(selected_sources)}...")
            result = source_search(
                query,
                sources=selected_sources,
                per_source=per_source,
                year_from=year_from,
                year_to=year_to,
                open_access_only=oa_only,
                sort=sort,
            )
            papers.extend(result["papers"])
            errors.update(result["errors"])
            counts = result["counts"]
            duplicates_merged = result["duplicates_merged"]

        if not papers:
            status.update(label="No results. Try a broader query or upload a PDF.", state="error")
            st.session_state.papers = []
            return

        status.write(f"Analyzing {len(papers)} documents...")
        kb = KnowledgeBase()
        kb.build(papers)
        kb.cluster(k=n_themes)
        coords = kb.project_2d()
        theme_names = label_themes_tfidf(papers)

        if auto_summarize:
            client = _resolve_client()
            if client is not None:
                status.write("Summarizing papers...")
                summarize_papers(papers, client)

        st.session_state.papers = papers
        st.session_state.kb = kb
        st.session_state.coords = coords
        st.session_state.theme_names = theme_names

        n_multi = sum(1 for p in papers if len(p.sources) > 1)
        n_clusters = len(set(p.cluster for p in papers))
        st.session_state.run_summary = (
            f"{len(papers)} papers from {len(counts) or 1} source(s), "
            f"{n_multi} found in more than one, grouped into {n_clusters} themes."
        )
        st.session_state.run_errors = errors
        status.update(label="Done.", state="complete", expanded=False)


if run_btn:
    if not query.strip() and not uploaded_files:
        st.error("Enter a research query or upload a PDF.")
    else:
        run_pipeline()

if st.session_state.run_summary:
    st.success(st.session_state.run_summary)
    source_errors(st.session_state.run_errors)


# --- tabs ----------------------------------------------------------------
if not st.session_state.papers:
    empty_state("No papers yet.", "Enter a topic above or upload a PDF, then click Search.")
else:
    papers = st.session_state.papers
    kb: KnowledgeBase = st.session_state.kb
    theme_names = st.session_state.theme_names

    tab_library, tab_map, tab_review, tab_ask, tab_export = st.tabs(
        ["Library", "Map", "Review", "Ask", "Export"]
    )

    with tab_library:
        col_a, col_b, col_c = st.columns(3)
        theme_options = ["All"] + [theme_names.get(c, f"Theme {c + 1}") for c in sorted(set(p.cluster for p in papers))]
        theme_filter = col_a.selectbox("Theme", theme_options)
        lib_sort = col_b.selectbox("Sort by", ["relevance", "recent", "citations"], key="lib_sort")
        search_within = col_c.text_input("Search within results", key="lib_search")

        if st.button("Summarize papers", help="On-demand: generates AI summaries for the papers below."):
            client = _resolve_client()
            if client is None:
                st.error("Enter the app password in the sidebar to use the LLM providers.")
            else:
                progress = st.progress(0.0)

                def _on_progress(done, total):
                    progress.progress(done / total)

                summarize_papers(papers, client, on_progress=_on_progress)
                progress.empty()

        shown = papers
        if theme_filter != "All":
            shown = [p for p in shown if theme_names.get(p.cluster, f"Theme {p.cluster + 1}") == theme_filter]
        if search_within:
            needle = search_within.lower()
            shown = [p for p in shown if needle in p.title.lower() or needle in p.abstract.lower()]
        if lib_sort == "recent":
            shown = sorted(shown, key=lambda p: p.year or 0, reverse=True)
        elif lib_sort == "citations":
            shown = sorted(shown, key=lambda p: p.citations, reverse=True)
        else:
            shown = sorted(shown, key=lambda p: p.rank_score, reverse=True)

        if not shown:
            empty_state("No papers match these filters.")
        for p in shown:
            paper_card(p, theme_names)

    with tab_map:
        cluster_map(papers, st.session_state.coords, theme_names)
        st.caption("Dot size scales with log(citations). Shape distinguishes uploads from search results.")

    with tab_review:
        length = st.select_slider("Length", options=["short", "medium", "long"], value="medium")
        if st.button("Name themes with AI"):
            client = _resolve_client()
            if client is None:
                st.error("No API key available.")
            else:
                st.session_state.theme_names = name_themes_ai(papers, client)
                st.rerun()

        if st.button("Write review", type="primary"):
            client = _resolve_client()
            if client is None:
                st.error("Enter the app password in the sidebar to use the LLM providers.")
            else:
                try:
                    with st.spinner("Writing literature review..."):
                        st.session_state.review_md = generate_review(
                            papers, kb, client, papers_per_theme=papers_per_theme, length=length, theme_names=theme_names
                        )
                except LLMError as exc:
                    st.error(str(exc))

        if st.session_state.review_md:
            st.markdown(highlight_citations(st.session_state.review_md), unsafe_allow_html=True)
            col_md, col_bib = st.columns(2)
            col_md.download_button("Download review (.md)", st.session_state.review_md.encode("utf-8"), "literature_review.md", "text/markdown")
            col_bib.download_button("Download references (.bib)", to_bibtex(papers).encode("utf-8"), "references.bib", "text/plain")
        else:
            empty_state("No review yet.", "Click 'Write review' to generate one.")

    with tab_ask:
        theme_filter_ask = st.selectbox(
            "Limit to theme (optional)", ["All"] + [theme_names.get(c, f"Theme {c + 1}") for c in sorted(set(p.cluster for p in papers))], key="ask_theme"
        )
        theme_id = None
        if theme_filter_ask != "All":
            theme_id = next(c for c in set(p.cluster for p in papers) if theme_names.get(c, f"Theme {c + 1}") == theme_filter_ask)

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                content = highlight_citations(m["content"]) if m["role"] == "assistant" else m["content"]
                st.markdown(content, unsafe_allow_html=True)

        if question := st.chat_input("Ask about these papers..."):
            client = _resolve_client()
            if client is None:
                st.error("Enter the app password in the sidebar to use the LLM providers.")
            else:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    try:
                        sources, stream = answer_stream(
                            kb, client, question, st.session_state.messages[:-1], theme=theme_id
                        )
                        placeholder = st.empty()
                        answer = placeholder.write_stream(stream)
                        placeholder.markdown(highlight_citations(answer), unsafe_allow_html=True)
                        if sources:
                            with st.expander("Sources"):
                                for s in sources:
                                    link = s["paper"].url or s["paper"].pdf_url
                                    st.markdown(f"**[{s['num']}] {s['paper'].title}** (score: {s['score']:.2f})")
                                    if link:
                                        st.caption(link)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except LLMError as exc:
                        st.error(str(exc))
                        st.session_state.messages.pop()  # drop the unanswered user turn

    with tab_export:
        st.write("Download the current results.")
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("CSV", to_csv(papers), "papers.csv", "text/csv")
        c2.download_button("BibTeX", to_bibtex(papers).encode("utf-8"), "papers.bib", "text/plain")
        c3.download_button("RIS", to_ris(papers).encode("utf-8"), "papers.ris", "text/plain")
        c4.download_button("JSON", to_json(papers).encode("utf-8"), "papers.json", "application/json")
