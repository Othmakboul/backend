from typing import List, Optional
from collections import Counter
import httpx

HAL_API_URL = "https://api.archives-ouvertes.fr/search/"

async def get_aggregated_stats(db, researchers: List[str], start_year: Optional[int], end_year: Optional[int]):
    """
    Fetches deduplicated publications from MongoDB for a list of researcher names.
    Since 'publications' already stores deduplicated entries, we just query it.
    """
    try:
        # 1. Map researcher names to _unique_id (since we store listic_author_ids)
        cursor = db.researchers.find({"name": {"$in": researchers}}, {"_unique_id": 1, "name": 1})
        matched_researchers = await cursor.to_list(length=1000)
        
        researcher_ids = [r["_unique_id"] for r in matched_researchers]
        
        if not researcher_ids:
            return {
                "total_publications": 0,
                "years_distribution": {},
                "types_distribution": {},
                "publications": []
            }
            
        # 2. Build the query for publications
        query = {
            "listic_author_ids": {"$in": researcher_ids}
        }
        
        if start_year or end_year:
            s = start_year if start_year else 0
            e = end_year if end_year else 9999
            query["producedDateY_i"] = {"$gte": s, "$lte": e}
            
        # 3. Fetch from DB
        pubs_cursor = db.publications.find(query).sort("producedDateY_i", -1)
        final_docs = await pubs_cursor.to_list(length=5000)
        
        # 4. Aggregate stats
        years = [d.get("producedDateY_i") for d in final_docs if d.get("producedDateY_i")]
        types = [d.get("docType_s") for d in final_docs if d.get("docType_s")]
        
        # 5. Remove _id for JSON serialization
        for d in final_docs:
            d.pop("_id", None)
        
        return {
            "total_publications": len(final_docs),
            "years_distribution": dict(Counter(years)),
            "types_distribution": dict(Counter(types)),
            "publications": final_docs
        }
        
    except Exception as e:
        print(f"Error fetching aggregated stats from DB: {e}")
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
