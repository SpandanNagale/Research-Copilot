from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

def summary(text:str)->str:
    client=Groq(api_key=os.getenv("GROQ_API_KEY"),)
    model="llama3-8b-8192"
    prompt = f"Summarize this academic abstract in 3 bullet points, preserve key methods/datasets/results. Text:\n{text}"
    resp=client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def batch_summary(abstract:list[str])->list[str]:
    return [summary(t) for t in abstract]
