import streamlit as st
import pandas as pd
import asyncio
import os

# Import your modules
from Arxiv import fetch_papers 
from embeddings import EmbedCluster 
from vectorstore import vector_store 

# Async wrappers
from summarizer import batch_summary_async 
from RAG import RAG_ans_async 

st.set_page_config(page_title="Research Copilot Dual-Mode", layout="wide", page_icon="⚔️")

# --- HELPER ---
def run_async(coro):
    """Helper to run async code in Streamlit."""
    return asyncio.run(coro)

# --- SIDEBAR: SETTINGS & ENGINE ---
with st.sidebar:
    st.title("⚙️ Engine Room")
    
    # 1. Choose Provider
    provider = st.radio("LLM Backend", ["Google Gemini (Cloud)", "Ollama (Local)"])
    
    llm_config = {}
    
    if provider == "Google Gemini (Cloud)":
        llm_config["provider"] = "gemini"
        # Input for API Key
        api_key_input = st.text_input("Gemini API Key", type="password")
        
        # Auto-load from environment if available
        final_key = api_key_input if api_key_input else os.getenv("GEMINI_API_KEY")
        
        if not final_key:
            st.warning("⚠️ API Key required!")
        else:
            st.success("✅ Key Detected")
            llm_config["api_key"] = final_key
            
        llm_config["model"] = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
        
    else:
        llm_config["provider"] = "ollama"
        st.info("Ensure 'ollama serve' is running.")
        llm_config["model"] = st.text_input("Local Model Name", value="mistral")
        st.caption("Recommended: mistral, llama3, deepseek-r1")

    # Save config to session
    st.session_state["llm_config"] = llm_config

    st.markdown("---")
    st.header("Research Settings")
    query = st.text_input("Research Topic", value="Agentic AI workflows")
    max_paper = st.number_input("Max Papers", 10, 100, 15)
    clusters = st.slider("Cluster Groups", 2, 8, 3)
    
    start_btn = st.button("🚀 Run Analysis", type="primary")

# --- MAIN LOGIC ---

# Initialize Session State
if "papers" not in st.session_state:
    st.session_state["papers"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "trigger" not in st.session_state:
    st.session_state["trigger"] = False

# Button Logic
if start_btn:
    # Validation
    if llm_config["provider"] == "gemini" and "api_key" not in llm_config:
        st.error("Cannot start: Missing Gemini API Key.")
        st.stop()
    st.session_state["trigger"] = True

# Execution Loop
if st.session_state["trigger"]:
    st.session_state["trigger"] = False
    
    with st.status("Agents Working...", expanded=True) as status:
        # 1. FETCH
        st.write("Fetching Arxiv Data...")
        papers = fetch_papers(query, max_paper)
        
        if not papers:
            st.error("No papers found on Arxiv for this topic.")
            st.stop()
        
        # 2. SUMMARIZE (Async & Parallel)
        st.write(f"Summarizing {len(papers)} papers using {llm_config['provider']}...")
        texts = [p["summary"] for p in papers]
        
        # Pass config to the async summarizer
        summaries = run_async(batch_summary_async(texts, llm_config))
        
        for i, p in enumerate(papers):
            p["short_summary"] = summaries[i]
            
        # 3. EMBED & CLUSTER
        st.write("Clustering embeddings...")
        ec = EmbedCluster()
        ec.fit(texts, papers)
        labels, _ = ec.kmeans(k=clusters)
        
        for i, p in enumerate(papers):
            p["cluster"] = int(labels[i])
            
        # 4. INDEX
        st.write("Building Semantic Index...")
        vs = vector_store()
        vs.build_index(papers)
        
        # Save results to session
        st.session_state["vectorstore"] = vs
        st.session_state["papers"] = papers
        
        status.update(label="Research Complete!", state="complete", expanded=False)

# --- DISPLAY RESULTS ---
if st.session_state["papers"]:
    papers = st.session_state["papers"]
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Knowledge Graph", "💬 Chat with Papers", "📥 Export Data"])
    
    # TAB 1: CLUSTERED VIEW
    with tab1:
        st.subheader(f"Found {len(papers)} papers on '{query}'")
        
        # Create tabs for each cluster
        c_tabs = st.tabs([f"Group {i+1}" for i in range(clusters)])
        
        for idx, tab in enumerate(c_tabs):
            with tab:
                # Filter papers for this cluster
                subset = [p for p in papers if p["cluster"] == idx]
                
                for p in subset:
                    # --- THE CARD UI ---
                    with st.container(border=True):
                        # Title & Link
                        st.markdown(f"### [{p['title']}]({p['pdf_url']})")
                        
                        # Metadata Row
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            # Handle author list gracefully
                            authors = p.get('authors', [])
                            if isinstance(authors, list):
                                author_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                            else:
                                author_str = str(authors)
                            st.markdown(f"**Authors:** *{author_str}*")
                        with c2:
                            published_date = p.get('published', 'N/A')
                            # Try to format date if it's a datetime object
                            if hasattr(published_date, 'date'):
                                published_date = published_date.date()
                            st.caption(f"📅 {published_date}")

                        st.divider()
                        
                        # AI Summary (Highlighted)
                        st.markdown("**✨ AI Summary:**")
                        st.info(p["short_summary"])
                        
                        # Original Abstract (Expandable)
                        with st.expander("📖 Read Original Abstract"):
                            st.write(p["summary"])

    # TAB 2: CHAT (RAG)
    with tab2:
        st.subheader("Deep Dive Q&A")
        
        # Chat History Container
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Render History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about methodologies, results, or comparisons..."):
            # User Msg
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant Msg
            with st.chat_message("assistant"):
                with st.spinner("Synthesizing answer..."):
                    # Call Async RAG with Config
                    answer, hits = run_async(RAG_ans_async(
                        st.session_state["vectorstore"], 
                        prompt, 
                        config=st.session_state["llm_config"]
                    ))
                    
                    st.markdown(answer)
                    
                    # Sources Dropdown
                    with st.expander("📚 Sources Used"):
                        for i, (p, score, idx) in enumerate(hits, 1):
                            st.markdown(f"**[{i}] {p['title']}** (Score: {score:.2f})")
                            st.caption(f"[PDF]({p['pdf_url']})")
            
            # Save Assistant Msg
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # TAB 3: EXPORT
    with tab3:
        st.write("Download the research data for your own analysis.")
        df = pd.DataFrame(papers)
        
        # Prepare CSV
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download CSV",
            csv,
            "research_data.csv",
            "text/csv",
            key='download-csv'

        )
