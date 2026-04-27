import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from services.hal import get_hal_stats, get_project_stats, get_listic_stats
from services.dblp import get_dblp_stats
from services.advanced_stats import get_aggregated_stats, compute_collaborations
from pydantic import BaseModel

app = FastAPI(title="LISTIC Dashboard API")

# Environment variables
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "listic_db"

# Data paths (can be overridden by env vars, e.g. for Docker volume)
DATA_PATH = os.getenv("DATA_PATH_RESEARCHERS", "/home/skudo/Desktop/LISTIC/listic-database/listic personnes/listic_personnes.complete_structure.json")
DATA_PATH_PROJECTS = os.getenv("DATA_PATH_PROJECTS", "/home/skudo/Desktop/LISTIC/listic-database/listic_projet/listic_projets.complete_structure.json")

# Database client
client = None
db = None

# CORS setup
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

async def seed_data():
    """Reads JSON files and populates MongoDB if collections are empty."""
    
    # 1. Seed Researchers
    if await db.researchers.count_documents({}) == 0:
        if os.path.exists(DATA_PATH):
            try:
                print(f"Seeding researchers from {DATA_PATH}...")
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    all_persons = []
                    root = raw_data[0] if isinstance(raw_data, list) and len(raw_data) > 0 else raw_data
                    data_content = root.get("data", {})
                    
                    for category, persons in data_content.items():
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
        else:
            print(f"Data file not found: {DATA_PATH}")

    # 2. Seed Projects
    if await db.projects.count_documents({}) == 0:
        if os.path.exists(DATA_PATH_PROJECTS):
            try:
                print(f"Seeding projects from {DATA_PATH_PROJECTS}...")
                with open(DATA_PATH_PROJECTS, "r", encoding="utf-8") as f:
                    raw_projects = json.load(f)
                    root_proj = raw_projects[0] if isinstance(raw_projects, list) and len(raw_projects) > 0 else raw_projects
                    data_content_proj = root_proj.get("data", {})
                    
                    all_projects_list = []
                    for cat, projs in data_content_proj.items():
                        if isinstance(projs, list):
                            for p in projs:
                                if isinstance(p, dict):
                                    p["type"] = cat
                                    # Ensure unique ID
                                    if "_unique_id" not in p:
                                        p["_unique_id"] = p.get("NOM")
                                    all_projects_list.append(p)
                    
                    if all_projects_list:
                        await db.projects.insert_many(all_projects_list)
                        print(f"Inserted {len(all_projects_list)} projects.")
            except Exception as e:
                print(f"Error seeding projects: {e}")
        else:
            print(f"Projects file not found: {DATA_PATH_PROJECTS}")

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

@app.get("/global-stats")
async def get_global_stats(start_year: Optional[int] = None, end_year: Optional[int] = None):
    """
    Get global statistics for LISTIC lab (HAL, DBLP).
    """
    hal_data = await get_listic_stats(start_year, end_year)
    return {
        "hal": hal_data,
        "dblp": {"note": "Global DBLP statistics not available natively via API"}
    }

@app.get("/researchers")
async def get_researchers(category: Optional[str] = None):
    query = {}
    if category:
        query["category"] = category
    
    # Exclude _id from result or map it
    cursor = db.researchers.find(query, {"_id": 0})
    return await cursor.to_list(length=1000)

@app.get("/projects")
async def get_projects():
    cursor = db.projects.find({}, {"_id": 0})
    return await cursor.to_list(length=1000)

@app.get("/project/{uid}")
async def get_project_details(uid: str):
    # Try finding by _unique_id first
    proj = await db.projects.find_one({"_unique_id": uid}, {"_id": 0})
    
    # Fallback by NOM if not found (for backward compatibility if uid is name)
    if not proj:
        proj = await db.projects.find_one({"NOM": uid}, {"_id": 0})
        
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    
    name = proj.get("NOM")
    hal_stats = await get_project_stats(name)
    
    return {
        "profile": proj,
        "stats": {
            "hal": hal_stats
        }
    }

@app.get("/researcher/{uid}")
async def get_researcher_details(uid: str, start_year: Optional[int] = None, end_year: Optional[int] = None, keyword: Optional[str] = None):
    person = await db.researchers.find_one({"_unique_id": uid}, {"_id": 0})
    
    # Fallback search by name if needed
    if not person:
         person = await db.researchers.find_one({"name": uid}, {"_id": 0})
         
    if not person:
        raise HTTPException(status_code=404, detail="Researcher not found")
    
    name = person.get("name")
    hal_data = await get_hal_stats(name, start_year, end_year, keyword)
    dblp_data = await get_dblp_stats(name)
    
    return {
        "profile": person,
        "stats": {
            "hal": hal_data,
            "dblp": dblp_data
        }
    }

@app.post("/api/advanced/aggregated-stats")
async def fetch_aggregated_stats(request: AggregatedRequest):
    if not request.researchers:
        return {"error": "No researchers provided."}
        
    stats = await get_aggregated_stats(request.researchers, request.start_year, request.end_year)
    
    collaborations = {"pairs": [], "triples": [], "unconnected": []}
    if "publications" in stats and stats["publications"]:
        collaborations = await compute_collaborations(stats["publications"], request.researchers)
        
    return {
        "stats": stats,
        "collaborations": collaborations
    }
