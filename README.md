# 📚 Research Copilot

A lightweight research assistant that helps you **search, cluster, summarize, and query scientific papers** from arXiv.  
Built with **Streamlit, FAISS, and Local Embeddings**, it’s designed as a modular pipeline for academic exploration.  

---

## 🚀 Features
- 🔎 **Paper Retrieval** – Fetch papers from arXiv using a keyword/topic query.  
- 🧩 **Clustering** – Group papers into clusters of related work using embeddings + KMeans.  
- ✍️ **Summarization** – Generate concise summaries of abstracts.  
- 📖 **RAG (Retrieval-Augmented Generation)** – Ask natural language questions over papers, powered by FAISS vector search.  
- 🎛️ **Interactive UI** – Browse raw results, clusters, and query answers via Streamlit.  

---

## 🛠️ Installation
Clone the repo and set up a Python virtual environment:

```bash
git clone https://github.com/SpandanNagale/Research-Copilot
cd research-copilot

# Create venv
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

