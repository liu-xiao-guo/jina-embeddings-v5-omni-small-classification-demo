"""Thin client for Elastic Inference Service (EIS) embedding endpoints."""
import base64
import mimetypes
import os
from pathlib import Path

import numpy as np
import requests
import urllib3
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ES_URL = os.environ["ELASTICSEARCH_URL"].rstrip("/")
API_KEY = os.environ["ELASTICSEARCH_API_KEY"]
VERIFY_TLS = os.getenv("ELASTICSEARCH_INSECURE_TLS", "false").lower() != "true"
INDEX = os.getenv("ELASTICSEARCH_INDEX", "omni_classification_demo")
OMNI_ID = os.getenv("OMNI_INFERENCE_ID", ".jina-embeddings-v5-omni-small")
TEXT_ID = os.getenv("TEXT_INFERENCE_ID", ".jina-embeddings-v5-text-small")
DIMS = 1024
MAX_BATCH = 16

if not VERIFY_TLS:
    urllib3.disable_warnings()

HEADERS = {"Authorization": f"ApiKey {API_KEY}", "Content-Type": "application/json"}

# The model card requires this prefix on text; without it text is off-distribution.
DOC_PREFIX = "Document: "

MIME_FALLBACK = {".flac": "audio/flac", ".m4a": "audio/mp4", ".opus": "audio/ogg", ".webm": "video/webm"}


def _post(path: str, body: dict, timeout: int = 120) -> dict:
    r = requests.post(f"{ES_URL}{path}", headers=HEADERS, json=body, verify=VERIFY_TLS, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"EIS {r.status_code}: {_root_reason(r)}")
    return r.json()


def _root_reason(r: requests.Response) -> str:
    try:
        d = r.json()["error"]
        while d.get("caused_by"):
            d = d["caused_by"]
        return d.get("reason", r.text[:300])
    except Exception:
        return r.text[:300]


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def media_block(kind: str, data: bytes, filename: str) -> dict:
    """Build an EIS content block (image | audio | video | pdf) from raw bytes."""
    mime = mimetypes.guess_type(filename)[0] or MIME_FALLBACK.get(Path(filename).suffix.lower(), f"{kind}/octet-stream")
    return {"type": kind, "format": "base64", "value": _data_uri(data, mime)}


def _normalize(vecs: list[list[float]], dims: int) -> np.ndarray:
    """Matryoshka-truncate to `dims` and re-normalize to unit length."""
    a = np.asarray(vecs, dtype=np.float32)[:, :dims]
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def embed_texts(texts: list[str], dims: int = DIMS, prefix: bool = True, endpoint: str = OMNI_ID) -> np.ndarray:
    """Embed texts. Omni endpoint uses the multimodal `embedding` shape; the
    text-only endpoint (task text_embedding) takes plain strings."""
    texts = [(DOC_PREFIX + t) if prefix else t for t in texts]
    vecs = []
    for i in range(0, len(texts), MAX_BATCH):  # EIS rejects more than 16 inputs per request
        chunk = texts[i:i + MAX_BATCH]
        if endpoint == OMNI_ID:
            body = {"input": [{"content": [{"type": "text", "value": t}]} for t in chunk]}
            vecs += [e["embedding"] for e in _post(f"/_inference/embedding/{endpoint}", body)["embeddings"]]
        else:
            out = _post(f"/_inference/text_embedding/{endpoint}", {"input": chunk})
            vecs += [e["embedding"] for e in out["text_embedding"]]
    return _normalize(vecs, dims)


def embed_media(kind: str, data: bytes, filename: str, dims: int = DIMS) -> np.ndarray:
    """Embed one image / audio / video file. Returns shape (dims,)."""
    body = {"input": [{"content": [media_block(kind, data, filename)]}]}
    out = _post(f"/_inference/embedding/{OMNI_ID}", body, timeout=300)["embeddings"]
    return _normalize([out[0]["embedding"]], dims)[0]


def embed_path(kind: str, path: str | Path, dims: int = DIMS) -> np.ndarray:
    p = Path(path)
    return embed_media(kind, p.read_bytes(), p.name, dims)
