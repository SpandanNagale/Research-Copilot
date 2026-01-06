# Research Copilot 2.0 🧠⚔️

> **Dual-Mode (Cloud + Local) • Asynchronous • RAG-Optimized**

Research Copilot is a high-performance AI tool designed to accelerate academic research. It fetches papers from Arxiv, summarizes them in parallel, clusters them by topic, and allows you to chat with the entire collection using grounded RAG (Retrieval-Augmented Generation).

**Key Upgrade:** This version is fully **Asynchronous**. It summarizes 20+ papers in seconds (vs. minutes in v1) and supports **Local LLMs (Ollama)** for privacy and zero-cost operation.

---

## 🚀 Key Features

* **⚡ Async Pipeline:** Uses `asyncio` and `aiohttp` concepts to summarize dozens of papers simultaneously.
* **⚔️ Dual-Mode Engine:**
    * **Cloud:** Google Gemini 2.5 Flash/Pro (High speed, 1M+ context).
    * **Local:** Ollama (Llama 3, Mistral, DeepSeek) for privacy and offline use.
* **📊 Smart Clustering:** Automatically groups papers into thematic clusters using K-Means embeddings.
* **🔍 Grounded RAG:** Chat with papers using strict citation rules (`[1]`, `[2]`) to prevent hallucinations.
* **📑 Exportable Data:** Download your entire research session as a CSV.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **AI Orchestration:** Python `asyncio`, `tenacity` (Retries)
* **LLMs:** Google Generative AI (Gemini), Ollama (Local)
* **Vector Database:** FAISS (CPU) + SentenceTransformers
* **Data Source:** Arxiv API

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/SpandanNagale/Research-Copilot.git](https://github.com/SpandanNagale/Research-Copilot.git)
cd Research-Copilot
python -m venv myenv
.\myenv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
