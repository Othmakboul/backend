import os
import json
import asyncio
import unicodedata
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from services.hal import get_hal_stats, get_project_stats, get_listic_stats
from services.dblp import get_dblp_stats
from services.advanced_stats import get_aggregated_stats, compute_collaborations
from services.ml_clustering import perform_clustering
from services.adum import fetch_listic_theses, compare_with_db as adum_compare
from services.listic_scraper import compare_website_with_db
from pydantic import BaseModel

app = FastAPI(title="LISTIC Dashboard API", version="2.0.0")

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

# ── In-memory sync status tracker ─────────────────────────────────────────────
sync_status = {
    "hal": {"status": "never", "last_sync": None, "count": 0},
    "adum": {"status": "never", "last_sync": None, "count": 0},
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────
class AggregatedRequest(BaseModel):
    researchers: List[str]
    start_year: Optional[int] = None
    end_year: Optional[int] = None


class ClusterRequest(BaseModel):
    granularity: float = 0.5
    researcher_ids: Optional[List[str]] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def names_match(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    return na in nb or nb in na


# ── Database seeding ───────────────────────────────────────────────────────────
async def seed_data():
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


# ── Auto-sync background tasks ─────────────────────────────────────────────────
async def _run_hal_sync_loop():
    """Run HAL sync every 6 hours automatically."""
    await asyncio.sleep(10)  # small delay after startup
    while True:
        try:
            sync_status["hal"]["status"] = "syncing"
            from services.sync_worker import sync_hal_publications
            await sync_hal_publications()
            sync_status["hal"]["status"] = "completed"
            sync_status["hal"]["last_sync"] = datetime.utcnow().isoformat()
            count = await db.publications.count_documents({})
            sync_status["hal"]["count"] = count
            print(f"Auto HAL sync completed. {count} publications in warehouse.")
        except Exception as e:
            sync_status["hal"]["status"] = "error"
            print(f"Auto HAL sync error: {e}")
        await asyncio.sleep(6 * 3600)  # repeat every 6 hours


async def _run_adum_sync_loop():
    """Run ADUM (theses.fr) sync every 24 hours."""
    await asyncio.sleep(30)
    while True:
        try:
            sync_status["adum"]["status"] = "syncing"
            theses = await fetch_listic_theses()
            sync_status["adum"]["status"] = "completed"
            sync_status["adum"]["last_sync"] = datetime.utcnow().isoformat()
            sync_status["adum"]["count"] = len(theses)
            sync_status["adum"]["last_data"] = theses
            print(f"Auto ADUM sync completed. {len(theses)} theses fetched.")
        except Exception as e:
            sync_status["adum"]["status"] = "error"
            print(f"Auto ADUM sync error: {e}")
        await asyncio.sleep(24 * 3600)


@app.on_event("startup")
async def startup_event():
    global client, db
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    await seed_data()
    # Start auto-sync loops in background
    asyncio.create_task(_run_hal_sync_loop())
    asyncio.create_task(_run_adum_sync_loop())


@app.on_event("shutdown")
async def shutdown_event():
    if client:
        client.close()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "LISTIC Dashboard API v2.0 — Multi-source research intelligence platform"}


# ── Sync status ────────────────────────────────────────────────────────────────
@app.get("/api/sync/status")
async def get_sync_status():
    pub_count = await db.publications.count_documents({})
    doc_count = await db.doctorants.count_documents({})
    res_count = await db.researchers.count_documents({})
    return {
        "hal": {**sync_status["hal"], "total_publications": pub_count},
        "adum": {**sync_status["adum"]},
        "database": {
            "researchers": res_count,
            "doctorants": doc_count,
            "publications": pub_count,
        }
    }


# ── Manual sync triggers ───────────────────────────────────────────────────────
@app.get("/api/sync/hal")
async def trigger_hal_sync():
    from services.sync_worker import sync_hal_publications
    asyncio.create_task(sync_hal_publications())
    sync_status["hal"]["status"] = "syncing"
    return {"message": "HAL sync started in background. Check /api/sync/status for progress."}


@app.get("/api/sync/adum")
async def trigger_adum_sync():
    async def _do_sync():
        sync_status["adum"]["status"] = "syncing"
        try:
            theses = await fetch_listic_theses()
            sync_status["adum"]["status"] = "completed"
            sync_status["adum"]["last_sync"] = datetime.utcnow().isoformat()
            sync_status["adum"]["count"] = len(theses)
            sync_status["adum"]["last_data"] = theses
        except Exception as e:
            sync_status["adum"]["status"] = "error"
            print(f"ADUM sync error: {e}")
    asyncio.create_task(_do_sync())
    return {"message": "ADUM (theses.fr) sync started in background."}


# ── Global stats ───────────────────────────────────────────────────────────────
@app.get("/global-stats")
async def get_global_stats(start_year: Optional[int] = None, end_year: Optional[int] = None):
    hal_data = await get_listic_stats(start_year, end_year)
    return {"hal": hal_data, "dblp": {"note": "Global DBLP statistics not available natively via API"}}


# ── Researchers list ───────────────────────────────────────────────────────────
@app.get("/researchers")
async def get_researchers(category: Optional[str] = None):
    query = {}
    if category:
        query["category"] = category
    cursor = db.researchers.find(query, {"_id": 0})
    return await cursor.to_list(length=1000)


# ── Projects list ──────────────────────────────────────────────────────────────
@app.get("/projects")
async def get_projects():
    cursor = db.projects.find({}, {"_id": 0})
    return await cursor.to_list(length=1000)


# ── Project detail ─────────────────────────────────────────────────────────────
@app.get("/project/{uid}")
async def get_project_details(uid: str):
    proj = await db.projects.find_one({"_unique_id": uid}, {"_id": 0})
    if not proj:
        proj = await db.projects.find_one({"NOM": uid}, {"_id": 0})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    hal_stats = await get_project_stats(proj.get("NOM"))
    return {"profile": proj, "stats": {"hal": hal_stats}}


# ── Researcher detail ──────────────────────────────────────────────────────────
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

    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
    phd_students = []
    for doc in all_docs:
        director = doc.get("director", "")
        co_director = doc.get("co_director", "")
        co_directors = [cd.get("name", "") for cd in doc.get("co_directors", [])]
        all_directors = [director, co_director] + co_directors + [doc.get("co_advisor", "")]
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

    hal_data = await get_hal_stats(name, start_year, end_year, keyword)
    dblp_data = await get_dblp_stats(name)

    return {"profile": person, "phd_students": phd_students, "stats": {"hal": hal_data, "dblp": dblp_data}}


# ── Researcher cards (fast) ────────────────────────────────────────────────────
@app.get("/api/researchers/cards")
async def get_researcher_cards():
    cursor = db.researchers.find({}, {"_id": 0})
    researchers = await cursor.to_list(length=1000)

    all_projects = await db.projects.find({}, {"_id": 0, "NOM": 1, "members": 1}).to_list(1000)
    all_docs = await db.doctorants.find(
        {}, {"_id": 0, "name": 1, "director": 1, "co_director": 1, "co_directors": 1, "co_advisor": 1, "status": 1}
    ).to_list(500)

    result = []
    for r in researchers:
        uid = r["_unique_id"]
        name = r.get("name", "")
        norm = normalize_name(name)

        pub_count = await db.publications.count_documents({"listic_author_ids": uid})

        proj_count = sum(
            1 for p in all_projects
            if any(names_match(norm, normalize_name(m)) for m in (p.get("members") or []))
        )

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


# ── Researcher publications (local warehouse) ──────────────────────────────────
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


# ── PhD students for a researcher ─────────────────────────────────────────────
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
    return {"researcher": person.get("name"), "total": len(phd_students), "active": active, "defended": defended}


# ── Doctorants list ───────────────────────────────────────────────────────────
@app.get("/api/doctorants")
async def get_doctorants(status: Optional[str] = None):
    query = {}
    if status:
        query["status"] = status
    cursor = db.doctorants.find(query, {"_id": 0})
    docs = await cursor.to_list(length=500)
    return {"total": len(docs), "doctorants": docs}


# ── ADUM / theses.fr data ─────────────────────────────────────────────────────
@app.get("/api/adum/theses")
async def get_adum_theses():
    """Fetch all LISTIC theses from theses.fr and compare with DB."""
    theses = await fetch_listic_theses()
    comparison = await adum_compare(db, theses)
    return comparison


# ── Aggregated multi-researcher stats ─────────────────────────────────────────
@app.post("/api/advanced/aggregated-stats")
async def fetch_aggregated_stats(request: AggregatedRequest):
    if not request.researchers:
        return {"error": "No researchers provided."}
    stats = await get_aggregated_stats(db, request.researchers, request.start_year, request.end_year)
    raw_pubs = stats.get("publications")
    publications = raw_pubs if isinstance(raw_pubs, list) else []
    collaborations = await compute_collaborations(db, publications, request.researchers)
    return {"stats": stats, "collaborations": collaborations}


# ── Clustering (GET — all researchers) ────────────────────────────────────────
@app.get("/api/analytics/cluster")
async def get_clustering(granularity: float = 0.5):
    return await perform_clustering(db, granularity)


# ── Clustering (POST — filtered researchers + granularity) ────────────────────
@app.post("/api/analytics/cluster")
async def post_clustering(request: ClusterRequest):
    return await perform_clustering(db, request.granularity, request.researcher_ids)


# ── Data Sources comparison ───────────────────────────────────────────────────
@app.get("/api/analytics/data-sources")
async def get_data_sources():
    """
    Cross-source comparison: HAL warehouse vs ADUM (theses.fr) vs LISTIC website.
    Returns researcher-level status in each source.
    """
    # HAL status per researcher
    researchers = await db.researchers.find({}, {"_id": 0}).to_list(1000)
    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
    all_projects = await db.projects.find({}, {"_id": 0, "NOM": 1, "members": 1}).to_list(1000)

    # Run ADUM, website comparison, and pub counts in parallel
    async def _get_adum():
        try:
            theses = await fetch_listic_theses()
            return await adum_compare(db, theses)
        except Exception as e:
            print(f"ADUM comparison error: {e}")
            return {"total_theses_fr": 0, "matched": 0, "missing_in_db": [], "missing_in_adum": [], "with_discrepancies": [], "all_theses": []}

    async def _get_website():
        try:
            return await compare_website_with_db(db)
        except Exception as e:
            print(f"Website comparison error: {e}")
            return {"total_website": 0, "matched": 0, "coverage_rate": 0, "in_db_not_website": [], "in_website_not_db": []}

    async def _get_pub_counts():
        pipeline = [
            {"$unwind": "$listic_author_ids"},
            {"$group": {"_id": "$listic_author_ids", "count": {"$sum": 1}}}
        ]
        cursor = db.publications.aggregate(pipeline)
        return {doc["_id"]: doc["count"] async for doc in cursor}

    adum_data, website_data, pub_counts_map = await asyncio.gather(
        _get_adum(), _get_website(), _get_pub_counts()
    )

    # Build per-researcher status
    adum_matched_names = {normalize_name(m.get("name", "")) for m in (adum_data.get("all_theses") or [])}
    website_missing_norms = {normalize_name(r.get("name", "")) for r in website_data.get("in_db_not_website", [])}

    researcher_status = []
    for r in researchers:
        uid = r["_unique_id"]
        name = r.get("name", "")
        norm = normalize_name(name)
        category = r.get("category", "")

        pub_count = pub_counts_map.get(uid, 0)

        phd_count = 0
        for doc in all_docs:
            dirs = [doc.get("director", ""), doc.get("co_director", ""), doc.get("co_advisor", "")] + \
                   [cd.get("name", "") for cd in doc.get("co_directors", [])]
            if any(names_match(norm, normalize_name(d)) for d in dirs if d):
                phd_count += 1

        # HAL status
        if category == "enseignants_chercheurs":
            if pub_count == 0:
                hal_status = "missing"
            elif pub_count < 3:
                hal_status = "low"
            else:
                hal_status = "ok"
        else:
            hal_status = "na"

        # ADUM status (only relevant for doctorants)
        adum_status = "na"
        if category == "doctorants":
            matched_adum = any(names_match(norm, an) for an in adum_matched_names)
            adum_status = "ok" if matched_adum else "missing"

        # Website status
        website_status = "missing" if norm in website_missing_norms else "ok"

        researcher_status.append({
            "uid": uid,
            "name": name,
            "category": category,
            "pub_count": pub_count,
            "phd_count": phd_count,
            "hal": {"status": hal_status, "pub_count": pub_count},
            "adum": {"status": adum_status},
            "listic_website": {"status": website_status},
        })

    return {
        "summary": {
            "hal": {
                "total_publications": await db.publications.count_documents({}),
                "researchers_with_pubs": sum(1 for r in researcher_status if r["hal"]["pub_count"] > 0),
                "researchers_missing": sum(1 for r in researcher_status if r["hal"]["status"] == "missing"),
                "last_sync": sync_status["hal"].get("last_sync"),
                "sync_status": sync_status["hal"]["status"],
            },
            "adum": {
                "total_theses_fr": adum_data.get("total_theses_fr", 0),
                "matched": adum_data.get("matched", 0),
                "missing_in_db": len(adum_data.get("missing_in_db", [])),
                "missing_in_adum": len(adum_data.get("missing_in_adum", [])),
                "with_discrepancies": len(adum_data.get("with_discrepancies", [])),
                "last_sync": sync_status["adum"].get("last_sync"),
                "sync_status": sync_status["adum"]["status"],
            },
            "listic_website": {
                "total_website": website_data.get("total_website", 0),
                "coverage_rate": website_data.get("coverage_rate", 0),
                "in_db_not_website": len(website_data.get("in_db_not_website", [])),
                "in_website_not_db": len(website_data.get("in_website_not_db", [])),
            }
        },
        "adum_details": {
            "missing_in_db": adum_data.get("missing_in_db", []),
            "missing_in_adum": adum_data.get("missing_in_adum", []),
            "with_discrepancies": adum_data.get("with_discrepancies", []),
        },
        "website_details": {
            "in_db_not_website": website_data.get("in_db_not_website", []),
            "in_website_not_db": website_data.get("in_website_not_db", []),
        },
        "researchers": researcher_status,
    }


# ── Inconsistencies (enhanced with cross-source checks) ───────────────────────
@app.get("/api/analytics/inconsistencies")
async def get_inconsistencies():
    from datetime import date

    cursor = db.researchers.find({}, {"_id": 0, "name": 1, "_unique_id": 1, "category": 1})
    researchers = await cursor.to_list(length=1000)
    all_docs = await db.doctorants.find({}, {"_id": 0}).to_list(500)
    all_projects = await db.projects.find({}, {"_id": 0, "NOM": 1, "members": 1, "PÉRIODE": 1}).to_list(1000)

    issues = []
    today = date.today()
    researcher_norms = {normalize_name(r["name"]): r for r in researchers}

    # ── HAL publication checks ─────────────────────────────────────────────
    for r in researchers:
        uid = r["_unique_id"]
        name = r.get("name", "")
        category = r.get("category", "")
        pub_count = await db.publications.count_documents({"listic_author_ids": uid})

        if category == "enseignants_chercheurs" and pub_count == 0:
            issues.append({
                "severity": "high",
                "type": "missing_hal_data",
                "researcher": name,
                "researcher_id": uid,
                "message": f"'{name}' has 0 publications indexed in the HAL warehouse.",
                "sources": ["LISTIC Website: Active researcher", "HAL warehouse: 0 publications"],
                "action": "Trigger HAL sync or verify name spelling in HAL"
            })
        elif category == "enseignants_chercheurs" and pub_count < 3:
            issues.append({
                "severity": "medium",
                "type": "low_hal_coverage",
                "researcher": name,
                "researcher_id": uid,
                "message": f"'{name}' has only {pub_count} publication(s) indexed — possible incomplete HAL profile.",
                "sources": [f"HAL warehouse: {pub_count} publication(s)"],
                "action": "Check HAL author profile and verify name variants"
            })

    # ── PhD student checks ─────────────────────────────────────────────────
    for doc in all_docs:
        if not doc.get("director"):
            issues.append({
                "severity": "medium",
                "type": "missing_director",
                "researcher": doc.get("name", "Unknown"),
                "researcher_id": doc.get("_unique_id", ""),
                "message": f"PhD student '{doc.get('name')}' has no director recorded.",
                "sources": ["LISTIC DB: director field is empty"],
                "action": "Update the doctorants database with the correct director name"
            })

    # ── Status mismatch: marked active but defense date passed ─────────────
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
                        "sources": [f"DB status: active", f"Defense date: {defense_str}"],
                        "action": "Update status to 'defended' in the database"
                    })
            except Exception:
                pass

    # ── Unknown director (not in LISTIC researchers) ───────────────────────
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
                    "sources": ["doctorants DB: director field", "researchers collection"],
                    "action": "Verify director affiliation or update researcher database"
                })

    # ── Cross-source: ADUM vs DB comparison ───────────────────────────────
    try:
        theses = await fetch_listic_theses()
        adum_result = await adum_compare(db, theses)

        for t in adum_result.get("missing_in_db", []):
            issues.append({
                "severity": "medium",
                "type": "missing_in_local_db",
                "researcher": t.get("name", ""),
                "researcher_id": "",
                "message": f"PhD student '{t.get('name')}' is registered in theses.fr (ADUM) but not found in our local database.",
                "sources": ["theses.fr: registered", "Local DB: not found"],
                "action": f"Add this entry — director: {t.get('director', 'unknown')}, started: {t.get('start_date', '?')}",
                "adum_url": t.get("theses_fr_url", "")
            })

        for entry in adum_result.get("with_discrepancies", []):
            for disc in entry.get("discrepancies", []):
                issues.append({
                    "severity": "medium",
                    "type": f"adum_mismatch_{disc['field']}",
                    "researcher": entry.get("name", ""),
                    "researcher_id": "",
                    "message": f"Discrepancy for '{entry.get('name')}': {disc['field']} differs between ADUM and local DB.",
                    "sources": [f"ADUM: {disc['adum_value']}", f"Local DB: {disc['db_value']}"],
                    "action": f"Verify and correct the {disc['field']} field",
                    "adum_url": entry.get("theses_fr_url", "")
                })
    except Exception as e:
        print(f"ADUM cross-check error: {e}")

    # ── Project with no members ────────────────────────────────────────────
    for proj in all_projects:
        if not proj.get("members"):
            issues.append({
                "severity": "low",
                "type": "project_no_members",
                "researcher": proj.get("NOM", ""),
                "researcher_id": "",
                "message": f"Project '{proj.get('NOM')}' ({proj.get('PÉRIODE', '?')}) has no team members recorded.",
                "sources": ["projects DB: members = []"],
                "action": "Update project entry with participating researchers"
            })

    return {
        "total_issues": len(issues),
        "high_severity": sum(1 for i in issues if i["severity"] == "high"),
        "medium_severity": sum(1 for i in issues if i["severity"] == "medium"),
        "low_severity": sum(1 for i in issues if i.get("severity") == "low"),
        "issues": sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["severity"], 3))
    }


# ── Matchmaking: hidden synergies ─────────────────────────────────────────────
@app.get("/api/researchers/{uid}/matchmaking")
async def get_matchmaking(uid: str, limit: int = 5):
    researcher = await db.researchers.find_one({"_unique_id": uid}, {"_id": 0, "name": 1})
    if not researcher:
        raise HTTPException(status_code=404, detail="Researcher not found")

    target_name = researcher["name"]
    target_norm = normalize_name(target_name)

    my_pubs = await db.publications.find(
        {"listic_author_ids": uid},
        {"authFullName_s": 1, "keyword_s": 1}
    ).to_list(length=10000)

    my_keywords: set = set()
    my_collaborator_norms: set = set()

    for pub in my_pubs:
        kw = pub.get("keyword_s", [])
        if isinstance(kw, list):
            my_keywords.update(k.lower().strip() for k in kw if k and k.strip())
        elif kw:
            my_keywords.add(kw.lower().strip())

        authors = pub.get("authFullName_s", [])
        if isinstance(authors, str):
            authors = [authors]
        for a in (authors or []):
            n = normalize_name(a)
            if not names_match(n, target_norm):
                my_collaborator_norms.add(n)

    candidate_categories = ["enseignants_chercheurs", "chercheurs_associes", "emerites"]
    all_researchers = await db.researchers.find(
        {"_unique_id": {"$ne": uid}, "category": {"$in": candidate_categories}},
        {"_id": 0, "_unique_id": 1, "name": 1}
    ).to_list(length=500)

    suggestions = []

    for r in all_researchers:
        r_norm = normalize_name(r["name"])
        if any(names_match(r_norm, collab) for collab in my_collaborator_norms):
            continue

        their_pubs = await db.publications.find(
            {"listic_author_ids": r["_unique_id"]},
            {"keyword_s": 1}
        ).to_list(length=2000)

        their_keywords: set = set()
        for pub in their_pubs:
            kw = pub.get("keyword_s", [])
            if isinstance(kw, list):
                their_keywords.update(k.lower().strip() for k in kw if k and k.strip())
            elif kw:
                their_keywords.add(kw.lower().strip())

        if not their_keywords or not my_keywords:
            continue

        common = my_keywords & their_keywords
        if not common:
            continue

        union = my_keywords | their_keywords
        score = round(len(common) / len(union) * 100, 1) if union else 0

        suggestions.append({
            "researcher_id": r["_unique_id"],
            "name": r["name"],
            "common_keywords": sorted(list(common))[:6],
            "score": score,
            "pub_count": len(their_pubs),
        })

    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return {
        "researcher": target_name,
        "total_my_keywords": len(my_keywords),
        "suggestions": suggestions[:limit],
    }


# ── Intelligence insights ─────────────────────────────────────────────────────
@app.get("/api/analytics/insights")
async def get_insights():
    researchers = await db.researchers.find(
        {"category": "enseignants_chercheurs"},
        {"_id": 0, "_unique_id": 1, "name": 1}
    ).to_list(200)

    researcher_stats = []
    for r in researchers:
        count = await db.publications.count_documents({"listic_author_ids": r["_unique_id"]})
        researcher_stats.append({"id": r["_unique_id"], "name": r["name"], "count": count})

    researcher_stats.sort(key=lambda x: x["count"], reverse=True)

    pipeline = [
        {"$match": {"producedDateY_i": {"$gte": 2010, "$lte": 2026}}},
        {"$group": {"_id": "$producedDateY_i", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    year_agg = await db.publications.aggregate(pipeline).to_list(50)
    year_data = {d["_id"]: d["count"] for d in year_agg if d["_id"]}

    recent_3y = sum(year_data.get(y, 0) for y in [2022, 2023, 2024])
    prev_3y = sum(year_data.get(y, 0) for y in [2019, 2020, 2021])
    growth_pct = round(((recent_3y - prev_3y) / prev_3y * 100) if prev_3y else 0, 1)

    isolated = [r for r in researcher_stats if r["count"] == 0]
    top_collab_pipeline = [
        {"$match": {"listic_author_ids": {"$exists": True, "$not": {"$size": 0}}}},
        {"$project": {"pairs": {"$reduce": {
            "input": {"$range": [0, {"$subtract": [{"$size": "$listic_author_ids"}, 1]}]},
            "initialValue": [],
            "in": {"$concatArrays": ["$$value", [{"$arrayElemAt": ["$listic_author_ids", "$$this"]}]]}
        }}}},
    ]

    total_pubs = await db.publications.count_documents({})
    total_researchers = await db.researchers.count_documents({"category": "enseignants_chercheurs"})
    active_count = len([r for r in researcher_stats if r["count"] > 0])
    peak_year = max(year_data.items(), key=lambda x: x[1])[0] if year_data else None
    peak_count = year_data.get(peak_year, 0) if peak_year else 0

    return {
        "total_publications": total_pubs,
        "total_researchers": total_researchers,
        "active_researchers": active_count,
        "isolated_count": len(isolated),
        "isolated_researchers": [r["name"] for r in isolated[:4]],
        "top_researcher": researcher_stats[0] if researcher_stats else None,
        "second_researcher": researcher_stats[1] if len(researcher_stats) > 1 else None,
        "growth_3y_pct": growth_pct,
        "year_distribution": year_data,
        "peak_year": peak_year,
        "peak_count": peak_count,
        "recent_3y": recent_3y,
        "prev_3y": prev_3y,
    }


# ── Yearly statistics (for Time Machine) ─────────────────────────────────────
@app.get("/api/analytics/yearly-stats")
async def get_yearly_stats():
    pipeline = [
        {"$match": {"producedDateY_i": {"$gte": 2010, "$lte": 2026}}},
        {"$group": {"_id": "$producedDateY_i", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    year_agg = await db.publications.aggregate(pipeline).to_list(50)

    researchers = await db.researchers.find(
        {"category": {"$in": ["enseignants_chercheurs", "chercheurs_associes"]}},
        {"_id": 0, "_unique_id": 1, "name": 1}
    ).to_list(200)

    researcher_by_year = []
    for r in researchers:
        r_pipeline = [
            {"$match": {"listic_author_ids": r["_unique_id"], "producedDateY_i": {"$gte": 2010, "$lte": 2026}}},
            {"$group": {"_id": "$producedDateY_i", "count": {"$sum": 1}}},
        ]
        r_agg = await db.publications.aggregate(r_pipeline).to_list(30)
        if r_agg:
            researcher_by_year.append({
                "id": r["_unique_id"],
                "name": r["name"],
                "by_year": {d["_id"]: d["count"] for d in r_agg}
            })

    return {
        "global_by_year": {d["_id"]: d["count"] for d in year_agg},
        "researchers": researcher_by_year,
    }


# ── Cluster bridges (Venn intersections) ─────────────────────────────────────
@app.post("/api/analytics/cluster-bridges")
async def get_cluster_bridges(req: ClusterRequest):
    result = await perform_clustering(db, req.granularity, req.researcher_ids)
    clusters = result.get("clusters", [])

    # Pairwise keyword Jaccard similarity → fast, no extra DB queries
    bridges = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            ca, cb = clusters[i], clusters[j]
            kw_a = set(kw.lower() for kw in ca.get("keywords", []))
            kw_b = set(kw.lower() for kw in cb.get("keywords", []))
            union = kw_a | kw_b
            common = kw_a & kw_b
            if not union:
                continue
            similarity = len(common) / len(union)
            bridges.append({
                "cluster_a_id": ca["cluster_id"],
                "cluster_b_id": cb["cluster_id"],
                "cluster_a_name": ca["name"],
                "cluster_b_name": cb["name"],
                "common_keywords": list(common)[:6],
                "similarity": round(similarity * 100, 1),
            })

    return {**result, "bridges": bridges}


# ── Bridge researchers (lazy-loaded on hover) ─────────────────────────────────
class BridgeRequest(BaseModel):
    ids_a: List[str]
    ids_b: List[str]


@app.post("/api/analytics/bridge-researchers")
async def get_bridge_researchers(req: BridgeRequest):
    if not req.ids_a or not req.ids_b:
        return {"collab_pub_count": 0, "bridge_researchers_a": [], "bridge_researchers_b": [], "bridge_pairs_count": 0}

    query = {
        "$and": [
            {"listic_author_ids": {"$in": req.ids_a}},
            {"listic_author_ids": {"$in": req.ids_b}},
        ]
    }
    pubs = await db.publications.find(
        query, {"listic_author_ids": 1, "title_s": 1, "producedDateY_i": 1}
    ).to_list(300)

    ids_a_set = set(req.ids_a)
    ids_b_set = set(req.ids_b)

    bridge_pairs: set = set()
    for pub in pubs:
        auth_ids = pub.get("listic_author_ids", [])
        a_in_pub = [aid for aid in auth_ids if aid in ids_a_set]
        b_in_pub = [bid for bid in auth_ids if bid in ids_b_set]
        for a in a_in_pub:
            for b in b_in_pub:
                bridge_pairs.add((a, b))

    all_ids = list(ids_a_set | ids_b_set)
    researchers = await db.researchers.find(
        {"_unique_id": {"$in": all_ids}}, {"_id": 0, "_unique_id": 1, "name": 1}
    ).to_list(200)
    name_map = {r["_unique_id"]: r["name"] for r in researchers}

    bridge_a_ids = {a for a, _ in bridge_pairs}
    bridge_b_ids = {b for _, b in bridge_pairs}

    return {
        "collab_pub_count": len(pubs),
        "bridge_researchers_a": [{"id": aid, "name": name_map.get(aid, aid)} for aid in bridge_a_ids],
        "bridge_researchers_b": [{"id": bid, "name": name_map.get(bid, bid)} for bid in bridge_b_ids],
        "bridge_pairs_count": len(bridge_pairs),
    }


# ── Lab health score ───────────────────────────────────────────────────────────
@app.get("/api/analytics/health-score")
async def get_health_score():
    # Publication rate
    ec_researchers = await db.researchers.find(
        {"category": "enseignants_chercheurs"}, {"_id": 0, "_unique_id": 1, "name": 1}
    ).to_list(200)

    pub_counts = []
    for r in ec_researchers:
        count = await db.publications.count_documents({"listic_author_ids": r["_unique_id"]})
        pub_counts.append(count)

    total_ec = len(ec_researchers)
    active_count = sum(1 for c in pub_counts if c > 0)
    pub_rate = round((active_count / total_ec * 100) if total_ec else 0, 1)

    # Data quality
    total_issues = 0
    high_issues = 0
    try:
        from services.adum import fetch_listic_theses, compare_with_db as adum_compare
        from services.listic_scraper import compare_website_with_db
    except Exception:
        pass
    all_docs = await db.doctorants.find({}, {"_id": 0, "director": 1, "status": 1, "defense_date": 1}).to_list(500)
    for doc in all_docs:
        if not doc.get("director"):
            total_issues += 1
    high_issues = sum(1 for r_count in pub_counts if r_count == 0)
    quality_score = round(max(0, 100 - high_issues * 4 - total_issues * 1.5), 1)

    # Year trend
    pipeline = [
        {"$match": {"producedDateY_i": {"$gte": 2019, "$lte": 2024}}},
        {"$group": {"_id": "$producedDateY_i", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    year_agg = await db.publications.aggregate(pipeline).to_list(10)
    year_data = {d["_id"]: d["count"] for d in year_agg}
    recent = sum(year_data.get(y, 0) for y in [2022, 2023, 2024])
    prev = sum(year_data.get(y, 0) for y in [2019, 2020, 2021])
    growth_pct = round(((recent - prev) / prev * 100) if prev else 0, 1)
    growth_score = min(100, max(0, 50 + growth_pct))

    # PhD activity
    active_phd = await db.doctorants.count_documents({"status": {"$ne": "defended"}})
    phd_score = min(100, round(active_phd / max(total_ec, 1) * 33, 1))

    # Collaboration density (from pub co-authorships)
    collab_pipeline = [
        {"$match": {"listic_author_ids": {"$exists": True}}},
        {"$project": {"count": {"$size": {"$ifNull": ["$listic_author_ids", []]}}}},
        {"$group": {"_id": None, "avg": {"$avg": "$count"}}},
    ]
    collab_agg = await db.publications.aggregate(collab_pipeline).to_list(1)
    avg_authors = collab_agg[0]["avg"] if collab_agg else 1
    collab_score = min(100, round((avg_authors - 1) * 30, 1))

    overall = round((pub_rate * 0.30 + quality_score * 0.25 + growth_score * 0.20 + phd_score * 0.15 + collab_score * 0.10), 1)

    return {
        "overall": overall,
        "dimensions": [
            {"key": "publication_rate", "label": "Taux de Publication", "score": pub_rate,
             "detail": f"{active_count}/{total_ec} chercheurs avec publications HAL", "color": "#3b82f6"},
            {"key": "data_quality", "label": "Qualité des Données", "score": quality_score,
             "detail": f"{high_issues} chercheurs sans HAL · {total_issues} incohérences directeur", "color": "#10b981"},
            {"key": "growth", "label": "Croissance Triennale", "score": growth_score,
             "detail": f"{growth_pct:+.1f}% vs période précédente", "color": "#6366f1"},
            {"key": "phd_activity", "label": "Vitalité Doctorale", "score": phd_score,
             "detail": f"{active_phd} thèses actives pour {total_ec} permanents", "color": "#f59e0b"},
            {"key": "collaboration", "label": "Densité Collaborative", "score": collab_score,
             "detail": f"{avg_authors:.1f} auteurs LISTIC en moyenne par publication", "color": "#ec4899"},
        ],
        "growth_pct": growth_pct,
        "total_publications": await db.publications.count_documents({}),
        "active_phd": active_phd,
    }
