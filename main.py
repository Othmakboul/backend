import os
import json
import unicodedata
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from services.hal import get_hal_stats, get_project_stats, get_listic_stats
from services.dblp import get_dblp_stats
from services.advanced_stats import get_aggregated_stats, compute_collaborations
from services.ml_clustering import perform_clustering
from pydantic import BaseModel

app = FastAPI(title="LISTIC Dashboard API")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "listic_db"

DATA_PATH = os.getenv(
    "DATA_PATH_RESEARCHERS",
    "/home/skudo/Desktop/LISTIC/listic-database/listic personnes/listic_personnes.complete_structure.json"
)
DATA_PATH_PROJECTS = os.getenv(
    "DATA_PATH_PROJECTS",
    "/home/skudo/Desktop/LISTIC/listic-database/listic_projet/listic_projets.complete_structure.json"
)
DATA_PATH_DOCTORANTS = os.getenv(
    "DATA_PATH_DOCTORANTS",
    "/home/skudo/Desktop/LISTIC/listic-database/listic personnes/listic_personnes.doctorants.json"
)

client = None
db = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AggregatedRequest(BaseModel):
    researchers: List[str]
    start_year: Optional[int] = None
    end_year: Optional[int] = None


def normalize_name(s: str) -> str:
    """Lowercase + strip accents for fuzzy name matching."""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def names_match(a: str, b: str) -> bool:
    """True if normalized last-name token of a appears in normalized b."""
    na, nb = normalize_name(a), normalize_name(b)
    # Check bidirectional substring (handles 'Trouvé' vs 'trouve')
    return na in nb or nb in na


async def seed_data():
    # ── Researchers ────────────────────────────────────────────────────────
    if await db.researchers.count_documents({}) == 0:
        if os.path.exists(DATA_PATH):
            try:
                print(f"Seeding researchers from {DATA_PATH}...")
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                root = raw[0] if isinstance(raw, list) else raw
                all_persons = []
                for category, persons in root.get("data", {}).items():
                    if isinstance(persons, list):
                        for p in persons:
                            if isinstance(p, dict):
                                p["category"] = category
                                if "_unique_id" not in p:
                                    p["_unique_id"] = p.get("name")
                                all_persons.append(p)
                if all_persons:
                    await db.researchers.insert_many(all_persons)
                    print(f"Inserted {len(all_persons)} researchers.")
            except Exception as e:
                print(f"Error seeding researchers: {e}")

    # ── Projects ───────────────────────────────────────────────────────────
    if await db.projects.count_documents({}) == 0:
        if os.path.exists(DATA_PATH_PROJECTS):
            try:
                print(f"Seeding projects from {DATA_PATH_PROJECTS}...")
                with open(DATA_PATH_PROJECTS, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                root = raw[0] if isinstance(raw, list) else raw
                all_projects = []
                for cat, projs in root.get("data", {}).items():
                    if isinstance(projs, list):
                        for p in projs:
                            if isinstance(p, dict):
                                p["type"] = cat
                                if "_unique_id" not in p:
                                    p["_unique_id"] = p.get("NOM")
                                all_projects.append(p)
                if all_projects:
                    await db.projects.insert_many(all_projects)
                    print(f"Inserted {len(all_projects)} projects.")
            except Exception as e:
                print(f"Error seeding projects: {e}")

    # ── Doctorants ─────────────────────────────────────────────────────────
    if await db.doctorants.count_documents({}) == 0:
        if os.path.exists(DATA_PATH_DOCTORANTS):
            try:
                print(f"Seeding doctorants from {DATA_PATH_DOCTORANTS}...")
                with open(DATA_PATH_DOCTORANTS, "r", encoding="utf-8") as f:
                    docs = json.load(f)
                if docs:
                    await db.doctorants.insert_many(docs)
                    print(f"Inserted {len(docs)} doctorants.")
            except Exception as e:
                print(f"Error seeding doctorants: {e}")


@app.on_event("startup")
async def startup_event():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    await seed_data()


@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()


@app.get("/")
def read_root():
    return {"message": "LISTIC Dashboard API is running with MongoDB"}


# ── Global stats ────────────────────────────────────────────────────────────
@app.get("/global-stats")
async def get_global_stats(start_year: Optional[int] = None, end_year: Optional[int] = None):
    hal_data = await get_listic_stats(start_year, end_year)
    return {
        "hal": hal_data,
        "dblp": {"note": "Global DBLP statistics not available natively via API"}
    }


# ── Researchers list ─────────────────────────────────────────────────────────
@app.get("/researchers")
async def get_researchers(category: Optional[str] = None):
    query = {}
    if category:
        query["category"] = category
    cursor = db.researchers.find(query, {"_id": 0})
    return await cursor.to_list(length=1000)


# ── Projects list ─────────────────────────────────────────────────────────────
@app.get("/projects")
async def get_projects():
    cursor = db.projects.find({}, {"_id": 0})
    return await cursor.to_list(length=1000)


# ── Project detail ───────────────────────────────────────────────────────────
@app.get("/project/{uid}")
async def get_project_details(uid: str):
    proj = await db.projects.find_one({"_unique_id": uid}, {"_id": 0})
    if not proj:
        proj = await db.projects.find_one({"NOM": uid}, {"_id": 0})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    hal_stats = await get_project_stats(proj.get("NOM"))
    return {"profile": proj, "stats": {"hal": hal_stats}}


# ── Researcher detail ─────────────────────────────────────────────────────────
@app.get("/researcher/{uid}")
async def get_researcher_details(
    uid: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    keyword: Optional[str] = None
):
    person = await db.researchers.find_one({"_unique_id": uid}, {"_id": 0})
    if not person:
        person = await db.researchers.find_one({"name": uid}, {"_id": 0})
    if not person:
        raise HTTPException(status_code=404, detail="Researcher not found")

    name = person.get("name", "")
    norm_name = normalize_name(name)

    # Fetch PhD students supervised by this researcher
    all_docs_cursor = db.doctorants.find({}, {"_id": 0})
    all_docs = await all_docs_cursor.to_list(length=500)
    phd_students = []
    for doc in all_docs:
        director = doc.get("director", "")
        co_director = doc.get("co_director", "")
        co_directors = [cd.get("name", "") for cd in doc.get("co_directors", [])]
        all_directors = [director, co_director] + co_directors + [
            doc.get("co_advisor", "")
        ]
        if any(names_match(norm_name, normalize_name(d)) for d in all_directors if d):
            phd_students.append({
                "name": doc.get("name"),
                "sujet": doc.get("Sujet", ""),
                "start_date": doc.get("start_date"),
                "status": doc.get("status", "active"),
                "defense_date": doc.get("defense_date"),
                "director": doc.get("director"),
                "co_director": doc.get("co_director"),
                "doctoral_school": doc.get("doctoral_school"),
                "theme": doc.get("Theme"),
            })

    hal_data = await get_hal_stats(name, start_year, end_year, keyword)  # type: ignore[misc]
    dblp_data = await get_dblp_stats(name)  # type: ignore[misc]

    return {
        "profile": person,
        "phd_students": phd_students,
        "stats": {"hal": hal_data, "dblp": dblp_data}
    }


# ── Researcher cards (fast) ───────────────────────────────────────────────────
@app.get("/api/researchers/cards")
async def get_researcher_cards():
    cursor = db.researchers.find({}, {"_id": 0})
    researchers = await cursor.to_list(length=1000)

    # Pre-fetch all projects and doctorants once
    all_projects = await db.projects.find({}, {"_id": 0, "NOM": 1, "members": 1}).to_list(1000)
    all_docs = await db.doctorants.find(
        {}, {"_id": 0, "name": 1, "director": 1, "co_director": 1, "co_directors": 1, "co_advisor": 1, "status": 1}
    ).to_list(500)

    result = []
    for r in researchers:
        uid = r["_unique_id"]
        name = r.get("name", "")
        norm = normalize_name(name)

        # Publications from local warehouse
        pub_count = await db.publications.count_documents({"listic_author_ids": uid})

        # Projects: check if name appears in members array (accent-tolerant)
        proj_count = sum(
            1 for p in all_projects
            if any(names_match(norm, normalize_name(m)) for m in (p.get("members") or []))
        )

        # PhD students supervised
        phd_count = 0
        for doc in all_docs:
            directors = [
                doc.get("director", ""),
                doc.get("co_director", ""),
                doc.get("co_advisor", ""),
            ] + [cd.get("name", "") for cd in doc.get("co_directors", [])]
            if any(names_match(norm, normalize_name(d)) for d in directors if d):
                phd_count += 1

        result.append({**r, "pub_count": pub_count, "project_count": proj_count, "phd_count": phd_count})

    return result


# ── Researcher publications (local warehouse) ─────────────────────────────────
@app.get("/api/researchers/{uid}/publications")
async def get_researcher_publications(
    uid: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
):
    query = {"listic_author_ids": uid}
    if start_year or end_year:
        query["producedDateY_i"] = {"$gte": start_year or 0, "$lte": end_year or 9999}
    cursor = db.publications.find(query, {"_id": 0}).sort("producedDateY_i", -1)
    pubs = await cursor.to_list(length=5000)
    return {"total": len(pubs), "publications": pubs}


# ── PhD students for a researcher ────────────────────────────────────────────
@app.get("/api/researchers/{uid}/phd-students")
async def get_researcher_phd_students(uid: str):
    person = await db.researchers.find_one({"_unique_id": uid}, {"_id": 0, "name": 1})
    if not person:
        raise HTTPException(status_code=404, detail="Researcher not found")

    norm = normalize_name(person.get("name", ""))
    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)

    phd_students = []
    for doc in all_docs:
        directors = [
            doc.get("director", ""),
            doc.get("co_director", ""),
            doc.get("co_advisor", ""),
        ] + [cd.get("name", "") for cd in doc.get("co_directors", [])]
        if any(names_match(norm, normalize_name(d)) for d in directors if d):
            phd_students.append({k: v for k, v in doc.items() if k != "_id"})

    active = [d for d in phd_students if d.get("status") != "defended"]
    defended = [d for d in phd_students if d.get("status") == "defended"]
    return {
        "researcher": person.get("name"),
        "total": len(phd_students),
        "active": active,
        "defended": defended,
    }


# ── Doctorants list ───────────────────────────────────────────────────────────
@app.get("/api/doctorants")
async def get_doctorants(status: Optional[str] = None):
    query = {}
    if status:
        query["status"] = status
    cursor = db.doctorants.find(query, {"_id": 0})
    docs = await cursor.to_list(length=500)
    return {"total": len(docs), "doctorants": docs}


# ── Aggregated multi-researcher stats ────────────────────────────────────────
@app.post("/api/advanced/aggregated-stats")
async def fetch_aggregated_stats(request: AggregatedRequest):
    if not request.researchers:
        return {"error": "No researchers provided."}

    stats = await get_aggregated_stats(db, request.researchers, request.start_year, request.end_year)

    raw_pubs = stats.get("publications")
    publications = raw_pubs if isinstance(raw_pubs, list) else []
    collaborations = await compute_collaborations(db, publications, request.researchers)

    return {"stats": stats, "collaborations": collaborations}


# ── Clustering ────────────────────────────────────────────────────────────────
@app.get("/api/analytics/cluster")
async def get_clustering(granularity: float = 0.5):
    return await perform_clustering(db, granularity)


# ── Inconsistencies ───────────────────────────────────────────────────────────
@app.get("/api/analytics/inconsistencies")
async def get_inconsistencies():
    from datetime import date

    cursor = db.researchers.find({}, {"_id": 0, "name": 1, "_unique_id": 1, "category": 1})
    researchers = await cursor.to_list(length=1000)
    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
    all_projects = await db.projects.find({}, {"_id": 0, "NOM": 1, "members": 1, "PÉRIODE": 1}).to_list(1000)

    issues = []
    today = date.today()

    # Build set of researcher names for director validation
    researcher_norms = {normalize_name(r["name"]): r for r in researchers}

    for r in researchers:
        uid = r["_unique_id"]
        name = r.get("name", "")
        norm = normalize_name(name)
        category = r.get("category", "")
        pub_count = await db.publications.count_documents({"listic_author_ids": uid})

        if category == "enseignants_chercheurs" and pub_count == 0:
            issues.append({
                "severity": "high",
                "type": "missing_hal_data",
                "researcher": name,
                "researcher_id": uid,
                "message": f"'{name}' has 0 publications indexed in HAL warehouse.",
                "sources": ["Lab Website: Active", "HAL warehouse: 0 publications"]
            })
        elif category == "enseignants_chercheurs" and pub_count < 3:
            issues.append({
                "severity": "medium",
                "type": "low_hal_coverage",
                "researcher": name,
                "researcher_id": uid,
                "message": f"'{name}' has only {pub_count} publication(s) indexed — possible incomplete HAL profile.",
                "sources": [f"HAL warehouse: {pub_count} publication(s)"]
            })

    # Rule: Doctorant without director
    for doc in all_docs:
        if not doc.get("director"):
            issues.append({
                "severity": "medium",
                "type": "missing_director",
                "researcher": doc.get("name", "Unknown"),
                "researcher_id": doc.get("_unique_id", ""),
                "message": f"PhD student '{doc.get('name')}' has no director recorded.",
                "sources": ["LISTIC DB: no director field"]
            })

    # Rule: Doctorant marked 'active' but defense_date is in the past
    for doc in all_docs:
        defense_str = doc.get("defense_date")
        if defense_str and doc.get("status") == "active":
            try:
                parts = defense_str.split("-")
                defense_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                if defense_date < today:
                    issues.append({
                        "severity": "high",
                        "type": "status_mismatch",
                        "researcher": doc.get("name", ""),
                        "researcher_id": doc.get("_unique_id", ""),
                        "message": f"'{doc.get('name')}' is marked 'active' but defense date {defense_str} has passed.",
                        "sources": [f"DB status: active", f"Defense date: {defense_str}"]
                    })
            except Exception:
                pass

    # Rule: Director name not matching any LISTIC researcher (external director OK, flag if unexpected)
    for doc in all_docs:
        director = doc.get("director", "")
        if director:
            norm_dir = normalize_name(director)
            director_in_listic = any(norm_dir in rn or rn in norm_dir for rn in researcher_norms)
            director_external = doc.get("director_affiliation", "LISTIC") not in ("LISTIC", "")
            if not director_in_listic and not director_external:
                issues.append({
                    "severity": "medium",
                    "type": "unknown_director",
                    "researcher": doc.get("name", ""),
                    "researcher_id": doc.get("_unique_id", ""),
                    "message": f"Director '{director}' of '{doc.get('name')}' not found in LISTIC researchers — may be external or misspelled.",
                    "sources": ["doctorants.json: director field", "researchers collection"]
                })

    # Rule: Project with no members recorded
    for proj in all_projects:
        if not proj.get("members"):
            issues.append({
                "severity": "low",
                "type": "project_no_members",
                "researcher": proj.get("NOM", ""),
                "researcher_id": "",
                "message": f"Project '{proj.get('NOM')}' ({proj.get('PÉRIODE', '?')}) has no team members recorded.",
                "sources": ["projects DB: members = []"]
            })

    return {
        "total_issues": len(issues),
        "high_severity": sum(1 for i in issues if i["severity"] == "high"),
        "medium_severity": sum(1 for i in issues if i["severity"] == "medium"),
        "low_severity": sum(1 for i in issues if i.get("severity") == "low"),
        "issues": sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3))
    }


# ── HAL sync trigger ──────────────────────────────────────────────────────────
@app.get("/api/sync/hal")
async def trigger_hal_sync():
    import asyncio
    from services.sync_worker import sync_hal_publications
    asyncio.create_task(sync_hal_publications())
    return {"message": "HAL sync started in background."}
