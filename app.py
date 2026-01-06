import streamlit as st
import pandas as pd
from Arxiv import fetch_papers
from embeddings import EmbedCluster
from summarizer import batch_summary
from vectorstore import vector_store
from RAG import RAG_ans

st.set_page_config(page_title="Research Copilot (Phase 1–3)", layout="wide")
st.title("📚 Research Copilot – Phase 1–3")
st.caption("Phase 1: Embed/Cluster/Summarize • Phase 2: Semantic Search (FAISS) • Phase 3: RAG Q&A with citations")

with st.sidebar:
    st.header("Settings")
    query = st.text_input("Topic/query", value="graph neural network drug discovery")
    max_paper = st.slider("Max Result", 10, 100, 40, step=5)
    cluster = st.slider("No. of clusters", 2, 10, 5)
    top_k = st.slider("Top-k search/RAG", 1, 10, 5)
    go = st.button("Run Pipeline")

if go:
    with st.spinner("Fetching papers from arXiv..."):
        papers = fetch_papers(query=query, max_paper=max_paper)
    if not papers:
        st.error("No results. Try a broader query.")
        st.stop()

    df = pd.DataFrame(papers)
    st.subheader("Raw Result")
    st.dataframe(df[["title","published","pdf_url","summary"]],
                 use_container_width=True, hide_index=True)

    with st.spinner("Embedding & clustering..."):
        texts = [p["summary"] for p in papers]
        ec = EmbedCluster()
        ec.fit(texts, papers)
        labels, _ = ec.kmeans(k=cluster)

    with st.spinner("Summarizing abstracts..."):
        summaries = batch_summary(texts)

    for i, p in enumerate(papers):
        p["cluster"] = int(labels[i])
        p["short_summary"] = summaries[i]

    st.subheader("Clustered View")
    tabs = st.tabs([f"Cluster {i}" for i in sorted(set(labels))])
    for idx, tab in enumerate(tabs):
        with tab:
            subset = [p for p in papers if p["cluster"] == idx]
            for p in subset:
                with st.container(border=True):
                    st.markdown(f"### {p['title']}")
                    st.markdown(f"- **Authors:** {', '.join(p['authors'])}")
                    st.markdown(f"- **Published:** {p['published'].date() if p['published'] else 'N/A'}")
                    st.markdown(f"- **PDF:** [{p['pdf_url']}]({p['pdf_url']})")
                    st.markdown("**Summary:**")
                    st.write(p["short_summary"])

    # Build FAISS over abstracts with metadata preserved
    vs = vector_store()
    vs.build_index(papers)
    st.session_state["vectorstore"] = vs
    st.session_state["papers"] = papers
    st.success("✅ Index ready for semantic search & RAG.")

# Semantic Search UI
if "vectorstore" in st.session_state:
    st.subheader("🔎 Semantic Search on Papers")
    user_query = st.text_input("Ask a semantic query over abstracts")
    if user_query:
        hits = st.session_state["vectorstore"].search(user_query, k=top_k)
        for i, (p, score, idx) in enumerate(hits, start=1):
            with st.container(border=True):
                st.markdown(f"### Result {i}: {p['title']}")
                st.write(f"**Relevance (cosine):** {score:.3f}")
                st.markdown(f"- **PDF:** [{p['pdf_url']}]({p['pdf_url']})")
                st.markdown("**Abstract:**")
                st.write(p["summary"])
        st.write("---")

# RAG Q&A UI
if "vectorstore" in st.session_state:
    st.subheader("🧠 Ask a Question (RAG – grounded answers with citations)")
    qa_q = st.text_input("Your research question")
    if st.button("Get Answer") and qa_q:
        with st.spinner("Thinking with your papers..."):
            answer, hits = RAG_ans(st.session_state["vectorstore"], qa_q, k=top_k)
        st.markdown("### 🧾 Answer")
        st.write(answer)

        st.markdown("### 📌 References")
        for i, (p, score, idx) in enumerate(hits, start=1):
            st.markdown(f"[{i}] **{p['title']}** — [PDF]({p['pdf_url']})")
