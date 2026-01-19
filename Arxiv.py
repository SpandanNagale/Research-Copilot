import arxiv
import logging

logger = logging.getLogger("Arxiv_Scraper")

def fetch_papers(query: str, max_paper: int = 30, sort_by: str = "relevance") -> list[dict]:
    """
    Fetches papers from ArXiv API and standardizes the metadata.
    """
    # 1. Map string to API Enum safely
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastupdateddate": arxiv.SortCriterion.LastUpdatedDate,
        "submitteddate": arxiv.SortCriterion.SubmittedDate 
    }
    
    # Fallback to relevance if string doesn't match
    target_sort = sort_map.get(sort_by.lower(), arxiv.SortCriterion.Relevance)

    # 2. Add defensive Try/Except block
    try:
        search = arxiv.Search(
            query=query, 
            max_results=max_paper, 
            sort_by=target_sort, # FIXED: Uses the dynamic variable now
            sort_order=arxiv.SortOrder.Descending 
        )
        
        results = []
        # Client configuration sets a timeout so the UI doesn't hang indefinitely
        client = arxiv.Client(page_size=max_paper, delay_seconds=1.0, num_retries=2)
        
        for r in client.results(search):
            results.append({
                "title": r.title.strip(),
                "authors": [a.name for a in r.authors],
                "summary": r.summary.strip().replace("\n", " "),
                # Converting datetime to string makes it safer for JSON exports and UI tables
                "published": r.published.strftime("%Y-%m-%d"), 
                "pdf_url": r.pdf_url,
                "entry_id": r.entry_id,
                "primary_category": getattr(r, "primary_category", "Unknown"),
            })
            
        return results

    except Exception as e:
        logger.error(f"ArXiv API Failed: {str(e)}")
        # Returning an empty list prevents the app from crashing. 
        # Streamlit will just show "No results found."
        return []



