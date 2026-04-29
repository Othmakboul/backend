"""
LISTIC website scraper.
Fetches researcher data from the official LISTIC lab website
(https://www.listic.univ-smb.fr) and cross-references with our database.
"""
import httpx
import re
import unicodedata
from typing import List

LISTIC_BASE = "https://www.listic.univ-smb.fr"
MEMBERS_URL = f"{LISTIC_BASE}/membres/"


def _norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _names_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    return bool(na and nb and (na in nb or nb in na))


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


async def scrape_listic_members() -> List[dict]:
    """
    Scrape the LISTIC members/persons listing page.
    Returns a list of dicts with name, url, and source fields.
    """
    members: List[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LISTIC-Dashboard/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        try:
            resp = await client.get(MEMBERS_URL, headers=headers)
            resp.raise_for_status()
            html = resp.text

            # Match typical WordPress/Drupal person listing links
            patterns = [
                r'href="(' + re.escape(LISTIC_BASE) + r'/[^"]*(?:membre|person|chercheur|enseignant)[^"]*)"[^>]*>\s*([A-ZÀ-Öa-zà-ö][^<]{2,60})',
                r'href="(' + re.escape(LISTIC_BASE) + r'/membres/[^"]+)"[^>]*>([A-ZÀ-Ö][a-zà-ö][^<]{1,50})',
                r'href="(/membres/[^"]+)"[^>]*>([A-ZÀ-Ö][a-zà-ö][^<]{1,50})',
            ]

            seen: set = set()
            for pat in patterns:
                for m in re.finditer(pat, html, re.IGNORECASE):
                    url = m.group(1)
                    name = _strip_tags(m.group(2)).strip()
                    if not url.startswith("http"):
                        url = LISTIC_BASE + url
                    key = _norm(name)
                    if key and key not in seen and 3 < len(name) < 70:
                        seen.add(key)
                        members.append({
                            "name": name,
                            "url": url,
                            "source": "listic_website",
                        })

            # Fallback: look for any person card with a name pattern
            if len(members) < 5:
                # Try to find h3/h4 tags near member links
                card_pattern = r'<(?:h[234]|strong)[^>]*>([A-ZÀ-Ö][a-zà-ö][\w\s\-\.À-ö]{3,50})</(?:h[234]|strong)>'
                for m in re.finditer(card_pattern, html):
                    name = _strip_tags(m.group(1)).strip()
                    key = _norm(name)
                    if key and key not in seen:
                        seen.add(key)
                        members.append({"name": name, "url": "", "source": "listic_website"})

            print(f"LISTIC website: found {len(members)} member entries")

        except httpx.HTTPStatusError as e:
            print(f"HTTP error scraping LISTIC members page: {e.response.status_code}")
        except Exception as e:
            print(f"Error scraping LISTIC members page: {e}")

    return members


async def compare_website_with_db(db) -> dict:
    """
    Compare scraped LISTIC website members with our local researchers collection.
    """
    website_members = await scrape_listic_members()
    db_researchers = await db.researchers.find(
        {}, {"_id": 0, "name": 1, "email": 1, "category": 1, "_unique_id": 1}
    ).to_list(1000)

    db_norms = {_norm(r["name"]): r for r in db_researchers if r.get("name")}
    website_norms = [_norm(m["name"]) for m in website_members if m.get("name")]

    in_db_not_website = []
    for dn, researcher in db_norms.items():
        found = any(_names_match(dn, wn) for wn in website_norms)
        if not found:
            in_db_not_website.append(researcher)

    in_website_not_db = []
    for wn, member in zip(website_norms, website_members):
        found = any(_names_match(wn, dn) for dn in db_norms)
        if not found:
            in_website_not_db.append(member)

    matched = len(db_researchers) - len(in_db_not_website)

    return {
        "total_website": len(website_members),
        "total_db": len(db_researchers),
        "matched": matched,
        "in_db_not_website": in_db_not_website,
        "in_website_not_db": in_website_not_db,
        "coverage_rate": round(matched / max(len(db_researchers), 1) * 100, 1),
    }
