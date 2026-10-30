"""Elasticsearch index + kNN search for the multimodal demo."""
import numpy as np
from elasticsearch import Elasticsearch, helpers

import eis

es = Elasticsearch(eis.ES_URL, api_key=eis.API_KEY, verify_certs=eis.VERIFY_TLS, ssl_show_warn=False, request_timeout=60)

MAPPING = {
    "properties": {
        "modality": {"type": "keyword"},
        "title": {"type": "text"},
        "caption": {"type": "text"},
        "path": {"type": "keyword"},
        "describes": {"type": "keyword"},
        "embedding": {"type": "dense_vector", "dims": eis.DIMS, "index": True, "similarity": "cosine"},
    }
}


def recreate_index():
    if es.indices.exists(index=eis.INDEX):
        es.indices.delete(index=eis.INDEX)
    es.indices.create(index=eis.INDEX, mappings=MAPPING)


def index_exists() -> bool:
    return es.indices.exists(index=eis.INDEX)


def count_by_modality() -> dict:
    if not index_exists():
        return {}
    r = es.search(index=eis.INDEX, size=0, aggs={"m": {"terms": {"field": "modality"}}})
    return {b["key"]: b["doc_count"] for b in r["aggregations"]["m"]["buckets"]}


def index_items(items: list[dict], progress=None):
    """Embed every item with its own modality and bulk-index. Text is batched; media is one call each."""
    recreate_index()
    docs, done = [], 0
    texts = [i for i in items if i["modality"] == "text"]
    vecs = eis.embed_texts([t["caption"] for t in texts])
    for it, v in zip(texts, vecs):
        docs.append({**it, "embedding": v.tolist()})
    done += len(texts)
    for it in (i for i in items if i["modality"] != "text"):
        docs.append({**it, "embedding": eis.embed_path(it["modality"], it["path"]).tolist()})
        done += 1
        if progress:
            progress(done / len(items), f"Embedded {it['id']}")
    helpers.bulk(es, ({"_index": eis.INDEX, "_id": d["id"], "_source": d} for d in docs), refresh=True)
    return len(docs)


def search(vec: np.ndarray, modalities: list[str], k: int = 5, dims: int = eis.DIMS) -> dict[str, list[dict]]:
    """Top-k per target modality. Cross-modal scores are lower than same-modality scores,
    so ranking within each modality is more informative than one mixed list."""
    out = {}
    for m in modalities:
        field = "embedding"
        r = es.search(index=eis.INDEX, size=k, source_excludes=["embedding"],
                      knn={"field": field, "query_vector": vec.tolist(), "k": k, "num_candidates": 50,
                           "filter": {"term": {"modality": m}}})
        out[m] = [{**h["_source"], "score": h["_score"]} for h in r["hits"]["hits"]]
    return out
