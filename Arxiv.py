import arxiv

def fetch_papers(query:str,max_paper:int=30, sort_by:str="relevance")->list[dict]:
    sort_map={
        "relevance": arxiv.SortCriterion.Relevance,
        "Lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
        "Submitteddate": arxiv.SortCriterion.SubmittedDate 
    }

    search= arxiv.Search(query=query ,max_results=max_paper , sort_by=sort_map["relevance"] , sort_order=arxiv.SortOrder.Descending )
    results=[]
    for r in search.results():
        results.append({
            "title": r.title.strip(),
            "authors": [a.name for a in r.authors],
            "summary": r.summary.strip().replace("\n", " "),
            "published": r.published,
            "updated": r.updated,
            "pdf_url": r.pdf_url,
            "entry_id": r.entry_id,
            "primary_category": getattr(r, "primary_category", None),
        })
    
    return results


