from llm import call_llm
from vectorstore import vector_store

def format_context(snippet:list[tuple[dict,float,int]])->str:
    blocks=[]
    for i , (p,score,idx) in enumerate(snippet , start=1):
        title=p.get("title","Untitled")
        url = p.get("pdf_url") or p.get("entry_id") or "N/A"
        abstract = p.get("summary","")
        blocks.append(
            f"[{i}] Title: {title}\nURL: {url}\nScore: {score:.3f}\nAbstract: {abstract}"
        )
    return "\n\n".join(blocks)

PROMPT_TMPL = """You are a research assistant. Answer the user's question using ONLY the provided paper abstracts.
Cite sources inline as [1], [2], ... corresponding to the bracketed sources in Context.
Be concise, factual, and avoid speculation. If the answer is uncertain, say so and point to the most relevant sources.

Question:
{question}

Context:
{context}
"""
def RAG_ans(vector_store , question : str , k : int = 4):
    hits=vector_store.search(question,k)
    context=format_context(hits)
    prompt=PROMPT_TMPL.format(question=question,context=context)
    answer=call_llm(prompt)
    return answer , hits

