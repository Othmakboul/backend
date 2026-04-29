"""
ADUM / theses.fr integration.
Uses the official French doctoral registry (theses.fr) API to fetch
LISTIC doctoral theses and cross-reference them with our local database.
"""
import httpx
import unicodedata
from typing import List, Optional

THESES_FR_API = "https://theses.fr/api/v1/theses/recherche"


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _names_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na and nb and (na in nb or nb in na))


def _person_name(d: dict) -> str:
    return " ".join(filter(None, [d.get("prenom", ""), d.get("nom", "")])).strip()


def _normalize_thesis(t: dict) -> dict:
    auteurs = t.get("auteurs") or []
    directeurs = t.get("directeurs") or []

    student_name = _person_name(auteurs[0]) if auteurs else ""
    director_name = _person_name(directeurs[0]) if directeurs else ""
    co_directors = [_person_name(d) for d in directeurs[1:] if d]

    defense = (t.get("dateSoutenance") or "")[:10]
    start = (t.get("dateDebut") or "")[:10]
    statut = t.get("statut", "")
    status = "defended" if (defense or "en cours" not in statut.lower()) and defense else "active"

    etablissements = [e.get("label", "") for e in (t.get("etablissements") or [])]

    return {
        "theses_fr_id": t.get("id", ""),
        "name": student_name,
        "title": t.get("titre", ""),
        "director": director_name,
        "co_directors": co_directors,
        "doctoral_school": ((t.get("ecoleDoctorale") or {}).get("label") or ""),
        "etablissements": etablissements,
        "start_date": start,
        "defense_date": defense,
        "status": status,
        "keywords": t.get("mots_cles") or [],
        "abstract": (t.get("resume") or "")[:600],
        "source": "theses.fr",
        "theses_fr_url": f"https://theses.fr/{t.get('id', '')}" if t.get("id") else "",
    }


async def fetch_listic_theses() -> List[dict]:
    """
    Fetch all LISTIC doctoral theses from theses.fr.
    Tries multiple query terms and deduplicates by ID.
    """
    results: List[dict] = []
    queries = ["LISTIC USMB", "LISTIC Savoie Mont Blanc", "Laboratoire d'Informatique Systèmes Traitement de l'Information et de la Connaissance"]

    async with httpx.AsyncClient(timeout=25.0) as client:
        for query in queries:
            params = {"q": query, "nombre": 500, "tri": "dateDesc"}
            try:
                r = await client.get(THESES_FR_API, params=params)
                r.raise_for_status()
                theses = r.json().get("theses", [])
                for t in theses:
                    results.append(_normalize_thesis(t))
                print(f"theses.fr [{query}]: {len(theses)} results")
            except Exception as e:
                print(f"Error fetching theses.fr [{query}]: {e}")

    # Deduplicate by theses_fr_id
    seen: set = set()
    unique: List[dict] = []
    for t in results:
        tid = t["theses_fr_id"]
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(t)
        elif not tid:
            unique.append(t)

    return unique


async def compare_with_db(db, listic_theses: Optional[List[dict]] = None) -> dict:
    """
    Cross-reference theses.fr data with our local doctorants collection.
    Detects: missing entries, director mismatches, status discrepancies.
    """
    if listic_theses is None:
        listic_theses = await fetch_listic_theses()

    db_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
    db_norms = {_norm(d.get("name", "")): d for d in db_docs if d.get("name")}

    matched = []
    missing_in_db = []

    for thesis in listic_theses:
        norm_name = _norm(thesis["name"])
        found_doc = None
        for db_norm, doc in db_norms.items():
            if _names_match(norm_name, db_norm):
                found_doc = doc
                break

        if found_doc:
            discrepancies = []
            db_dir = _norm(found_doc.get("director", ""))
            th_dir = _norm(thesis["director"])
            if th_dir and db_dir and not _names_match(th_dir, db_dir):
                discrepancies.append({
                    "field": "director",
                    "db_value": found_doc.get("director", ""),
                    "adum_value": thesis["director"],
                })
            db_status = found_doc.get("status", "active")
            if thesis["status"] != db_status:
                discrepancies.append({
                    "field": "status",
                    "db_value": db_status,
                    "adum_value": thesis["status"],
                })
            matched.append({
                **thesis,
                "in_db": True,
                "db_name": found_doc.get("name", ""),
                "discrepancies": discrepancies,
            })
        else:
            missing_in_db.append({**thesis, "in_db": False, "discrepancies": []})

    # Entries in our DB not found in theses.fr
    theses_norms = [_norm(t["name"]) for t in listic_theses if t.get("name")]
    missing_in_adum = []
    for db_norm, doc in db_norms.items():
        found = any(_names_match(db_norm, tn) for tn in theses_norms)
        if not found:
            missing_in_adum.append(doc)

    with_discrepancies = [m for m in matched if m["discrepancies"]]

    return {
        "total_theses_fr": len(listic_theses),
        "total_db": len(db_docs),
        "matched": len(matched),
        "missing_in_db": missing_in_db,
        "missing_in_adum": missing_in_adum,
        "with_discrepancies": with_discrepancies,
        "all_theses": listic_theses,
    }
