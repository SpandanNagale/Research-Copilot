import streamlit as st
import pandas as pd
import asyncio
import os

# --- IMPORT MODULES ---
# Ensure you have updated Arxiv.py, summarizer.py, and RAG.py 
# with the latest versions provided in our chat.
from Arxiv import fetch_papers 
from embeddings import EmbedCluster 
from vectorstore import vector_store 

# Async wrappers from our upgraded files
from summarizer import batch_summary_async 
from RAG import RAG_ans_async 

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Research Copilot 2.0", 
    layout="wide", 
    page_icon="🧠"
)

# --- ASYNC HELPER ---
def run_async(coro):
    """Helper to run async code within Streamlit's sync context."""
    return asyncio.run(coro)

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Engine Room")
    
    # 1. LLM BACKEND SELECTION
    provider = st.radio("LLM Backend", ["Google Gemini (Cloud)", "Ollama (Local)"])
    
    llm_config = {}
    
    if provider == "Google Gemini (Cloud)":
        llm_config["provider"] = "gemini"
        # Secure API Key Entry
        api_key_input = st.text_input("Gemini API Key", type="password")
        
        # Fallback to env variable
        final_key = api_key_input if api_key_input else os.getenv("GEMINI_API_KEY")
        
        if not final_key:
            st.warning("⚠️ API Key required!")
        else:
            st.success("✅ Key Detected")
            llm_config["api_key"] = final_key
            
        llm_config["model"] = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
        
    else:
        llm_config["provider"] = "ollama"
        st.info("Ensure 'ollama serve' is running locally.")
        llm_config["model"] = st.text_input("Local Model Name", value="mistral")
        st.caption("Recommended: mistral, llama3, deepseek-r1")

    # Save LLM config to session state
    st.session_state["llm_config"] = llm_config

    st.markdown("---")
    
    # 2. RESEARCH SETTINGS
    st.header("Research Parameters")
    query = st.text_input("Research Topic", value="Agentic AI workflows")
    
    # New Sorting Feature
    sort_option = st.selectbox(
        "Sort Papers By",
        options=["Relevance", "SubmittedDate", "LastUpdatedDate"],
        index=0
    )
    
    max_paper = st.number_input("Max Papers", 10, 100, 15)
    clusters = st.slider("Cluster Groups", 2, 8, 3)
    
    start_btn = st.button("🚀 Run Analysis", type="primary")

# --- MAIN LOGIC ---

# Initialize Session State Variables
if "papers" not in st.session_state:
    st.session_state["papers"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "trigger" not in st.session_state:
    st.session_state["trigger"] = False

# Trigger Logic
if start_btn:
    # Validation
    if llm_config["provider"] == "gemini" and "api_key" not in llm_config:
        st.error("Cannot start: Missing Gemini API Key.")
        st.stop()
    st.session_state["trigger"] = True

# Execution Pipeline
if st.session_state["trigger"]:
    st.session_state["trigger"] = False # Reset trigger
    
    with st.status("Agents Working...", expanded=True) as status:
        
        # STEP 1: FETCH FROM ARXIV
        st.write(f"Fetching Arxiv Data (Sorted by {sort_option})...")
        papers = fetch_papers(query, max_paper, sort_by=sort_option.lower())
        
        if not papers:
            st.error("No papers found. Try a different query.")
            st.stop()
        
        # STEP 2: SUMMARIZE (Async & Parallel)
        st.write(f"Summarizing {len(papers)} papers using {llm_config['provider']}...")
        texts = [p["summary"] for p in papers]
        
        # Execute the async batch summarizer
        summaries = run_async(batch_summary_async(texts, llm_config))
        
        # Attach summaries to paper objects
        for i, p in enumerate(papers):
            p["short_summary"] = summaries[i]
            
        # STEP 3: EMBED & CLUSTER
        st.write("Clustering embeddings...")
        ec = EmbedCluster()
        ec.fit(texts, papers)
        labels, _ = ec.kmeans(k=clusters)
        
        for i, p in enumerate(papers):
            p["cluster"] = int(labels[i])
            
        # STEP 4: INDEX FOR RAG
        st.write("Building Semantic Index...")
        vs = vector_store()
        vs.build_index(papers)
        
        # Store results in Session State
        st.session_state["vectorstore"] = vs
        st.session_state["papers"] = papers
        
        status.update(label="Research Complete!", state="complete", expanded=False)

# --- DISPLAY RESULTS ---

if st.session_state["papers"]:
    papers = st.session_state["papers"]
    
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Knowledge Graph", "💬 Chat with Papers", "📥 Export Data"])
    
    # --- TAB 1: CLUSTERED VIEW ---
    with tab1:
        st.subheader(f"Found {len(papers)} papers on '{query}'")
        
        # Tabs for each cluster group
        c_tabs = st.tabs([f"Group {i+1}" for i in range(clusters)])
        
        for idx, tab in enumerate(c_tabs):
            with tab:
                # Filter papers belonging to this cluster
                subset = [p for p in papers if p["cluster"] == idx]
                
                for p in subset:
                    with st.container(border=True):
                        # Title & PDF Link
                        st.markdown(f"### [{p['title']}]({p['pdf_url']})")
                        
                        # Metadata
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            authors = p.get('authors', [])
                            author_str = ", ".join(authors[:3]) + ("..." if len(authors) > 3 else "")
                            st.markdown(f"**Authors:** *{author_str}*")
                        with c2:
                            published = p.get('published', 'N/A')
                            st.caption(f"📅 {published}")

                        st.divider()
                        
                        # AI Generated Summary
                        st.markdown("**✨ AI Insight:**")
                        st.info(p["short_summary"])
                        
                        # Original Abstract (Collapsible)
                        with st.expander("📖 Read Original Abstract"):
                            st.write(p["summary"])

    # --- TAB 2: RAG CHAT ---
    with tab2:
        st.subheader("Deep Dive Q&A")
        
        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask about specific methodologies, results, or comparisons..."):
            
            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Assistant Message
            with st.chat_message("assistant"):
                with st.spinner("Analyzing papers..."):
                    # Call Async RAG with the stored config
                    answer, hits = run_async(RAG_ans_async(
                        st.session_state["vectorstore"], 
                        prompt, 
                        config=st.session_state["llm_config"]
                    ))
                    
                    st.markdown(answer)
                    
                    # Show Sources
                    with st.expander("📚 Sources Used"):
                        for i, (p, score, idx) in enumerate(hits, 1):
                            st.markdown(f"**[{i}] {p['title']}** (Score: {score:.2f})")
                            st.caption(f"[PDF]({p['pdf_url']})")
            
            # Save Assistant Message
            st.session_state.messages.append({"role": "assistant", "content": answer})

    # --- TAB 3: EXPORT ---
    with tab3:
        st.write("Download the structured research data for your own analysis.")
        df = pd.DataFrame(papers)
        
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "Download Research CSV",
            csv,
            "research_data.csv",
            "text/csv",
            key='download-csv'
        )
