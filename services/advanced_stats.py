import unicodedata
from typing import List, Optional
from collections import Counter


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def names_match(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    return na in nb or nb in na


async def get_aggregated_stats(db, researchers: List[str], start_year: Optional[int], end_year: Optional[int]):
    """
    Fetches deduplicated publications + PhD students from MongoDB
    for a list of researcher names.
    """
    try:
        cursor = db.researchers.find({"name": {"$in": researchers}}, {"_unique_id": 1, "name": 1})
        matched = await cursor.to_list(length=1000)
        researcher_ids = [r["_unique_id"] for r in matched]

        if not researcher_ids:
            return {"total_publications": 0, "years_distribution": {}, "types_distribution": {}, "publications": [], "phd_students": []}

        query = {"listic_author_ids": {"$in": researcher_ids}}
        if start_year or end_year:
            query["producedDateY_i"] = {"$gte": start_year or 0, "$lte": end_year or 9999}

        pubs_cursor = db.publications.find(query).sort("producedDateY_i", -1)
        final_docs = await pubs_cursor.to_list(length=5000)

        years = [d.get("producedDateY_i") for d in final_docs if d.get("producedDateY_i")]
        types = [d.get("docType_s") for d in final_docs if d.get("docType_s")]

        for d in final_docs:
            d.pop("_id", None)

        # Fetch PhD students for this set of researchers
        all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
        phd_students = []
        for doc in all_docs:
            directors = [
                doc.get("director", ""),
                doc.get("co_director", ""),
                doc.get("co_advisor", ""),
            ] + [cd.get("name", "") for cd in doc.get("co_directors", [])]
            for res_name in researchers:
                if any(names_match(res_name, d) for d in directors if d):
                    entry = {
                        "name": doc.get("name"),
                        "sujet": doc.get("Sujet", ""),
                        "director": doc.get("director"),
                        "co_director": doc.get("co_director"),
                        "start_date": doc.get("start_date"),
                        "status": doc.get("status", "active"),
                        "theme": doc.get("Theme"),
                    }
                    if entry not in phd_students:
                        phd_students.append(entry)
                    break

        return {
            "total_publications": len(final_docs),
            "years_distribution": dict(Counter(years)),
            "types_distribution": dict(Counter(types)),
            "publications": final_docs,
            "phd_students": phd_students,
        }

    except Exception as e:
        print(f"Error in get_aggregated_stats: {e}")
        return {"error": str(e)}


async def compute_collaborations(db, publications: List[dict], researchers: List[str]):
    """
    Computes publication-based collaboration pairs/triples AND
    co-supervision pairs from the doctorants collection.
    """
    pairs = Counter()
    triples = Counter()
    target_norms = {normalize_name(r): r for r in researchers}

    # ── Publication-based collaborations ──────────────────────────────────
    for pub in publications:
        authors = pub.get("authFullName_s", [])
        if isinstance(authors, str):
            authors = [authors]

        matched = []
        for author in authors:
            norm_author = normalize_name(author)
            for norm_target, orig in target_norms.items():
                if norm_target in norm_author or norm_author in norm_target:
                    matched.append(orig)
                    break

        matched = sorted(list(set(matched)))
        n = len(matched)

        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    pairs[f"{matched[i]} | {matched[j]}"] += 1

        if n >= 3:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        triples[f"{matched[i]} | {matched[j]} | {matched[k]}"] += 1

    # ── Co-supervision-based collaborations ───────────────────────────────
    cosupervision_pairs = []
    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)

    for doc in all_docs:
        # Collect all directors for this thesis
        all_directors_raw = [
            doc.get("director", ""),
            doc.get("co_director", ""),
            doc.get("co_advisor", ""),
        ] + [cd.get("name", "") for cd in doc.get("co_directors", [])]
        all_directors_raw = [d for d in all_directors_raw if d]

        # Find which target researchers are directors of this thesis
        matched_researchers = []
        for norm_target, orig in target_norms.items():
            if any(names_match(norm_target, d) for d in all_directors_raw):
                matched_researchers.append(orig)

        # If 2+ target researchers co-supervise this thesis → collaboration pair
        matched_researchers = sorted(list(set(matched_researchers)))
        if len(matched_researchers) >= 2:
            for i in range(len(matched_researchers)):
                for j in range(i + 1, len(matched_researchers)):
                    cosupervision_pairs.append({
                        "pair": f"{matched_researchers[i]} | {matched_researchers[j]}",
                        "doctorant": doc.get("name", ""),
                        "sujet": doc.get("Sujet", ""),
                        "start_date": doc.get("start_date"),
                        "status": doc.get("status", "active"),
                    })

    # ── Unconnected researchers ────────────────────────────────────────────
    connected = set()
    for pair_key in pairs:
        a, b = pair_key.split(" | ")
        connected.add(a)
        connected.add(b)
    for cp in cosupervision_pairs:
        a, b = cp["pair"].split(" | ")
        connected.add(a)
        connected.add(b)

    unconnected = [r for r in researchers if r not in connected]

    return {
        "pairs": [{"pair": k, "count": v} for k, v in pairs.most_common()],
        "triples": [{"triple": k, "count": v} for k, v in triples.most_common()],
        "cosupervision_pairs": cosupervision_pairs,
        "unconnected": unconnected,
    }
