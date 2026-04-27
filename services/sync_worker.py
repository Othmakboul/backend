import asyncio
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
import os
import time

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "listic_db"
HAL_API_URL = "https://api.archives-ouvertes.fr/search/"

async def sync_hal_publications():
    """
    Background worker to sync and deduplicate HAL publications into the Data Warehouse.
    """
    print("Starting HAL Synchronization Worker...")
    start_time = time.time()
    
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    
    # 1. Get all researchers
    researchers_cursor = db.researchers.find({}, {"name": 1, "_unique_id": 1})
    researchers = await researchers_cursor.to_list(length=1000)
    
    if not researchers:
        print("No researchers found in the database. Sync aborted.")
        return
        
    print(f"Found {len(researchers)} researchers. Querying HAL...")
    
    # Process in batches to avoid enormous query strings
    batch_size = 20
    total_upserted = 0
    
    async with httpx.AsyncClient() as http_client:
        for i in range(0, len(researchers), batch_size):
            batch = researchers[i:i+batch_size]
            names = [r.get("name") for r in batch if r.get("name")]
            
            author_queries = [f'"{name}"' for name in names]
            query = f'authFullName_t:({" OR ".join(author_queries)})'
            fl = "halId_s,title_s,producedDateY_i,docType_s,authFullName_s,journalTitle_s,keyword_s"
            
            params = {
                "q": query,
                "wt": "json",
                "fl": fl,
                "rows": 2000, # Maximize rows to catch bulk history
                "sort": "producedDateY_i desc"
            }
            
            try:
                response = await http_client.get(HAL_API_URL, params=params, timeout=30.0)
                response.raise_for_status()
                docs = response.json().get("response", {}).get("docs", [])
                
                if not docs:
                    continue
                    
                # 2. Upsert (Deduplicate) into 'publications' collection
                operations = []
                for doc in docs:
                    hal_id = doc.get("halId_s")
                    if not hal_id:
                        continue
                        
                    # Map authors back to our internal researcher IDs (if they exist)
                    doc_authors = doc.get("authFullName_s", [])
                    if isinstance(doc_authors, str):
                        doc_authors = [doc_authors]
                        
                    matched_ids = []
                    for author in doc_authors:
                        for r in researchers:
                            if r["name"].lower() in author.lower():
                                matched_ids.append(r["_unique_id"])
                                break
                                
                    doc["listic_author_ids"] = list(set(matched_ids))
                    doc["last_synced"] = time.time()
                    
                    # Update if halId_s exists, otherwise insert
                    operations.append(
                        UpdateOne(
                            {"halId_s": hal_id},
                            {"$set": doc},
                            upsert=True
                        )
                    )
                
                if operations:
                    result = await db.publications.bulk_write(operations)
                    total_upserted += (result.upserted_count + result.modified_count)
                    
            except Exception as e:
                print(f"Error processing batch {i}-{i+batch_size}: {e}")
                
            # Rate limiting delay
            await asyncio.sleep(1)
            
    print(f"HAL Synchronization complete in {time.time() - start_time:.2f}s. Upserted/Modified {total_upserted} publications.")
    client.close()

if __name__ == "__main__":
    asyncio.run(sync_hal_publications())
