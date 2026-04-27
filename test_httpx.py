import asyncio
from services.advanced_stats import get_aggregated_stats
import json

async def main():
    stats = await get_aggregated_stats(["Ilham ALLOUI", "Abdourrahmane ATTO"], 2020, 2026)
    print(json.dumps(stats, indent=2))

asyncio.run(main())
