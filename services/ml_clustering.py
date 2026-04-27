import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def names_match(a: str, b: str) -> bool:
    na, nb = normalize_name(a), normalize_name(b)
    return na in nb or nb in na


async def perform_clustering(db, granularity: float):
    """
    Clusters researchers by publication keywords + thesis subjects of their PhD students.
    Granularity 0.0 = 1 big cluster, 1.0 = one cluster per researcher.
    """
    researchers_cursor = db.researchers.find({}, {"_unique_id": 1, "name": 1})
    researchers = await researchers_cursor.to_list(length=1000)

    if not researchers:
        return {"clusters": []}

    # Pre-fetch all doctorants once
    all_docs = await db.doctorants.find({}, {"_id": 0, "name": 1, "Sujet": 1, "Theme": 1,
                                             "director": 1, "co_director": 1, "co_directors": 1,
                                             "co_advisor": 1, "hal_data": 1}).to_list(500)

    corpus = []
    researcher_names = []
    researcher_ids = []

    for r in researchers:
        uid = r["_unique_id"]
        r_norm = normalize_name(r["name"])

        # ── Publication keywords from HAL warehouse ────────────────────────
        pubs_cursor = db.publications.find({"listic_author_ids": uid}, {"keyword_s": 1})
        pubs = await pubs_cursor.to_list(length=10000)

        all_keywords = []
        for p in pubs:
            kw = p.get("keyword_s")
            if kw:
                if isinstance(kw, list):
                    all_keywords.extend(kw)
                else:
                    all_keywords.append(kw)

        # ── PhD students' thesis subjects ──────────────────────────────────
        for doc in all_docs:
            directors = [
                doc.get("director", ""),
                doc.get("co_director", ""),
                doc.get("co_advisor", ""),
            ] + [cd.get("name", "") for cd in doc.get("co_directors", [])]
            if any(names_match(r_norm, d) for d in directors if d):
                sujet = doc.get("Sujet", "")
                theme = doc.get("Theme", "")
                if sujet and sujet not in ("Non trouve", "Mots-clefs"):
                    # Treat words in thesis subject as additional keywords
                    all_keywords.extend(sujet.split())
                if theme and theme not in ("Non trouve", "AFuTe ou ReGaRD"):
                    all_keywords.append(theme.replace(" ", "_"))
                # Also include doctorant HAL keywords if present
                hal = doc.get("hal_data", {})
                for kw in hal.get("keywords", []):
                    all_keywords.append(kw)

        if all_keywords:
            doc_text = " ".join(k.lower().replace(" ", "_") for k in all_keywords)
            corpus.append(doc_text)
            researcher_names.append(r["name"])
            researcher_ids.append(uid)

    if not corpus:
        return {"clusters": []}

    # ── TF-IDF Vectorization ───────────────────────────────────────────────
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)

    X_dense = X.toarray()
    nonzero_mask = X_dense.sum(axis=1) > 0
    X_filtered = X_dense[nonzero_mask]
    names_filtered = [n for n, keep in zip(researcher_names, nonzero_mask) if keep]
    ids_filtered = [i for i, keep in zip(researcher_ids, nonzero_mask) if keep]

    if len(X_filtered) < 2:
        return {
            "granularity": granularity,
            "total_clusters": 1,
            "clusters": [{
                "cluster_id": 0,
                "name": "LISTIC Lab",
                "keywords": [],
                "members": [{"id": researcher_ids[i], "name": researcher_names[i]}
                            for i in range(len(researcher_names))],
                "size": len(researcher_names),
            }]
        }

    # ── Agglomerative Clustering ───────────────────────────────────────────
    n_members = len(X_filtered)
    n_clusters = max(1, int(round(granularity * (n_members - 1))) + 1)
    n_clusters = min(n_clusters, n_members)

    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="euclidean", linkage="ward"
    ).fit_predict(X_filtered)

    # ── Build result ───────────────────────────────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
    clusters_map = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_map[int(label)].append({"id": ids_filtered[idx], "name": names_filtered[idx]})

    result_clusters = []
    for cluster_id, members in clusters_map.items():
        member_indices = [ids_filtered.index(m["id"]) for m in members]
        cluster_tfidf = np.array(X_filtered[member_indices]).sum(axis=0)
        top_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_keywords = [
            feature_names[i].replace("_", " ")
            for i in top_indices if cluster_tfidf[i] > 0
        ]
        cluster_name = " | ".join(top_keywords[:3]).title() if top_keywords else "Mixed Theme"
        result_clusters.append({
            "cluster_id": cluster_id,
            "name": cluster_name,
            "keywords": top_keywords,
            "members": members,
            "size": len(members),
        })

    result_clusters.sort(key=lambda x: x["size"], reverse=True)

    return {
        "granularity": granularity,
        "total_clusters": len(result_clusters),
        "clusters": result_clusters,
    }
