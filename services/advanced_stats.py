from typing import List, Optional
from collections import Counter
import httpx

HAL_API_URL = "https://api.archives-ouvertes.fr/search/"

async def get_aggregated_stats(researchers: List[str], start_year: Optional[int], end_year: Optional[int]):
    """
    Fetches and deduplicates publications for a list of researchers.
    """
    # Construct HAL query for multiple authors
    # Example: authFullName_s:("Flavien" OR "Sébastien")
    
    author_queries = [f'"{name}"' for name in researchers]
    query = f'authFullName_t:({" OR ".join(author_queries)})'
    
    fl = "halId_s,title_s,producedDateY_i,docType_s,authFullName_s,journalTitle_s,keyword_s"
    
    params = {
        "q": query,
        "wt": "json",
        "fl": fl,
        "rows": 500, # Assume max 500 for now, could be paginated
        "sort": "producedDateY_i desc"
    }
    
    if start_year or end_year:
        s = start_year if start_year else "*"
        e = end_year if end_year else "*"
        params["fq"] = f"producedDateY_i:[{s} TO {e}]"
        
    async with httpx.AsyncClient() as client:
        try:
            print(f"DEBUG: HAL Query Params: {params}", flush=True)
            response = await client.get(HAL_API_URL, params=params, timeout=10.0)
            print(f"DEBUG: HAL Request URL: {response.url}", flush=True)
            response.raise_for_status()
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            
            # Deduplicate by halId_s
            unique_docs = {}
            for d in docs:
                hal_id = d.get("halId_s")
                if hal_id and hal_id not in unique_docs:
                    unique_docs[hal_id] = d
            
            final_docs = list(unique_docs.values())
            
            # Aggregate stats
            years = [d.get("producedDateY_i") for d in final_docs if d.get("producedDateY_i")]
            types = [d.get("docType_s") for d in final_docs if d.get("docType_s")]
            
            return {
                "total_publications": len(final_docs),
                "years_distribution": dict(Counter(years)),
                "types_distribution": dict(Counter(types)),
                "publications": final_docs
            }
            
        except Exception as e:
            print(f"Error fetching aggregated stats: {e}")
            return {"error": str(e)}

async def compute_collaborations(publications: List[dict], researchers: List[str]):
    """
    Computes collaboration pairs and triples from a list of publications.
    """
    pairs = Counter()
    triples = Counter()
    
    # Normalize input researchers to lowercase for matching
    target_researchers = {r.lower() for r in researchers}
    
    for pub in publications:
        authors = pub.get("authFullName_s", [])
        if isinstance(authors, str):
            authors = [authors]
            
        # Filter authors to only those in our target list
        matched_authors = []
        for author in authors:
            # Simple matching, could be improved with fuzzing
            for target in target_researchers:
                if target in author.lower():
                    matched_authors.append(target)
                    break
        
        # Deduplicate matched authors for this publication
        matched_authors = sorted(list(set(matched_authors)))
        n = len(matched_authors)
        
        # Pairs
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    pair = f"{matched_authors[i]} | {matched_authors[j]}"
                    pairs[pair] += 1
                    
        # Triples
        if n >= 3:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        triple = f"{matched_authors[i]} | {matched_authors[j]} | {matched_authors[k]}"
                        triples[triple] += 1
                        
    # Find unconnected researchers
    connected = set()
    for pair in pairs.keys():
        a, b = pair.split(" | ")
        connected.add(a)
        connected.add(b)
        
    unconnected = list(target_researchers - connected)
    
    return {
        "pairs": [{"pair": k, "count": v} for k, v in pairs.most_common()],
        "triples": [{"triple": k, "count": v} for k, v in triples.most_common()],
        "unconnected": unconnected
    }
