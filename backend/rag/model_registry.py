"""
model_registry.py
-----------------
Thread-safe, lazy initialisation of the three heavy singletons:
  - SentenceTransformer  (all-mpnet-base-v2, 768-d)
  - CrossEncoder         (ms-marco-MiniLM-L-6-v2)
  - QdrantClient         (local persistent store)

Call `get_models()` from any module — blocks only on first call.
Call `init_models()` once at app startup to surface errors early.
"""

import os, sys, threading
from .config import (
    EMBED_MODEL_NAME, RERANK_MODEL_NAME, VECTOR_DIM,
    QDRANT_COLLECTION, QDRANT_DB_PATH, emit,
)

_lock         = threading.Lock()
_embed_model  = None
_reranker     = None
_qdrant       = None
_init_error   = None   # stores the first error string so we stop retrying


def _log(message: str, status: str = "processing") -> None:
    """Emit over SocketIO if available, always print to terminal."""
    print(f"[model_registry:{status}] {message}", flush=True)
    try:
        emit(message, status)
    except Exception:
        pass   # SocketIO not yet ready at startup — terminal log is enough


def init_models() -> bool:
    """
    Eagerly initialise all three models.
    Returns True on success, False on failure.
    Call this once from app.py after socketio is set up.
    """
    result = get_models()
    return result is not None


def get_models() -> tuple | None:
    """
    Returns (embed_model, reranker, qdrant_client) or None if init failed.
    Safe to call from multiple threads concurrently.
    After a failure the error is stored and retried on next call to avoid
    permanent silent failures if a transient issue (e.g. file lock) clears.
    """
    global _embed_model, _reranker, _qdrant, _init_error

    # Fast path — fully ready
    if _embed_model is not None and _reranker is not None and _qdrant is not None:
        return _embed_model, _reranker, _qdrant

    with _lock:
        # Re-check inside lock (double-checked locking)
        if _embed_model is not None and _reranker is not None and _qdrant is not None:
            return _embed_model, _reranker, _qdrant

        # ── Embedding model ──────────────────────────────────────────────────
        if _embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _log(f"Loading embedding model: {EMBED_MODEL_NAME} ({VECTOR_DIM}-d)…")
                _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
                _log(f"Embedding model ready ({EMBED_MODEL_NAME}).", "success")
            except Exception as e:
                _init_error = f"Embedding model load failed: {e}"
                _log(_init_error, "error")
                return None

        # ── Cross-encoder reranker ───────────────────────────────────────────
        if _reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                _log(f"Loading reranker: {RERANK_MODEL_NAME}…")
                _reranker = CrossEncoder(RERANK_MODEL_NAME)
                _log("Reranker ready.", "success")
            except Exception as e:
                _init_error = f"Reranker load failed: {e}"
                _log(_init_error, "error")
                return None

        # ── Qdrant client ────────────────────────────────────────────────────
        if _qdrant is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                os.makedirs(QDRANT_DB_PATH, exist_ok=True)
                _log(f"Connecting to Qdrant at: {QDRANT_DB_PATH}")
                _qdrant = QdrantClient(path=QDRANT_DB_PATH)
                if not _qdrant.collection_exists(QDRANT_COLLECTION):
                    _log(f"Creating collection '{QDRANT_COLLECTION}' ({VECTOR_DIM}-d)…")
                    _qdrant.create_collection(
                        collection_name=QDRANT_COLLECTION,
                        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                    )
                    _log(f"Collection '{QDRANT_COLLECTION}' created.", "success")
                else:
                    _log(f"Collection '{QDRANT_COLLECTION}' already exists.", "success")
                _log("Qdrant ready.", "success")
            except Exception as e:
                _init_error = f"Qdrant init failed: {e}"
                _log(_init_error, "error")
                return None

        _init_error = None   # clear any previous transient error
        _log("All models ready.", "success")
        return _embed_model, _reranker, _qdrant


def get_init_error() -> str | None:
    """Returns the last init error string, or None if everything is healthy."""
    return _init_error
