import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import json

async def perform_clustering(db, granularity: float):
    """
    Performs clustering on all researchers based on their publication keywords.
    Granularity is a float between 0.0 (one big cluster) and 1.0 (many individual clusters).
    """
    # 1. Fetch all researchers
    researchers_cursor = db.researchers.find({}, {"_unique_id": 1, "name": 1})
    researchers = await researchers_cursor.to_list(length=1000)
    
    if not researchers:
        return {"clusters": []}
        
    # 2. For each researcher, gather all keywords from their publications
    corpus = []
    researcher_names = []
    researcher_ids = []
    
    for r in researchers:
        uid = r["_unique_id"]
        # Find all publications where listic_author_ids contains uid
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
                    
        if all_keywords:
            # Join keywords into a single string document for TF-IDF
            doc = " ".join([k.lower().replace(" ", "_") for k in all_keywords])
            corpus.append(doc)
            researcher_names.append(r["name"])
            researcher_ids.append(uid)
            
    if not corpus:
        return {"clusters": []}
        
    # 3. TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1)
    try:
        X = vectorizer.fit_transform(corpus)
    except ValueError:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        
    X_dense = X.toarray()
    
    # Remove zero vectors (researchers with no useful keywords after vectorization)
    nonzero_mask = X_dense.sum(axis=1) > 0
    X_filtered = X_dense[nonzero_mask]
    researcher_names_filtered = [n for n, keep in zip(researcher_names, nonzero_mask) if keep]
    researcher_ids_filtered = [i for i, keep in zip(researcher_ids, nonzero_mask) if keep]
    
    if len(X_filtered) < 2:
        # Not enough data to cluster - return everyone in one cluster
        return {
            "granularity": granularity,
            "total_clusters": 1,
            "clusters": [{
                "cluster_id": 0,
                "name": "LISTIC Lab",
                "members": [{"id": researcher_ids[i], "name": researcher_names[i]} for i in range(len(researcher_names))],
                "size": len(researcher_names)
            }]
        }
    
    # 4. Clustering: map granularity to number of clusters
    # granularity 0.0 = 1 cluster, 1.0 = N clusters (one per researcher)
    n_members = len(X_filtered)
    n_clusters = max(1, int(round(granularity * (n_members - 1))) + 1)
    n_clusters = min(n_clusters, n_members)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='euclidean',
        linkage='ward'
    )
    
    labels = clustering.fit_predict(X_filtered)
    researcher_names = researcher_names_filtered
    researcher_ids = researcher_ids_filtered
    X_dense = X_filtered
    
    # 5. Format Output
    clusters_map = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_map[int(label)].append({
            "id": researcher_ids[idx],
            "name": researcher_names[idx]
        })
        
    # Extract top keywords for each cluster to label it
    feature_names = vectorizer.get_feature_names_out()
    result_clusters = []
    
    for cluster_id, members in clusters_map.items():
        # Find centroid or simply sum TF-IDF scores for members
        member_indices = [researcher_ids.index(m["id"]) for m in members]
        cluster_tfidf = np.array(X_dense[member_indices]).sum(axis=0)
        
        # Get top 3 keywords
        top_indices = cluster_tfidf.argsort()[-3:][::-1]
        top_keywords = [feature_names[i].replace("_", " ") for i in top_indices if cluster_tfidf[i] > 0]
        
        cluster_name = " | ".join(top_keywords) if top_keywords else "Unknown Theme"
        
        result_clusters.append({
            "cluster_id": cluster_id,
            "name": cluster_name.title(),
            "members": members,
            "size": len(members)
        })
        
    # Sort clusters by size
    result_clusters.sort(key=lambda x: x["size"], reverse=True)
    
    return {
        "granularity": granularity,
        "total_clusters": len(result_clusters),
        "clusters": result_clusters
    }
