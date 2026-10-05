"""
offline/indexer.py
------------------
Embeds a list of text chunks using the shared model registry and upserts
the resulting vectors + payloads into Qdrant.
"""

from ..config import QDRANT_COLLECTION, emit
from ..model_registry import get_models
from .chunker import build_points


def session_already_indexed(session_id: str) -> bool:
    """
    Return True if any point with this session_id already exists.
    Uses a simple scroll with a payload filter dict — avoids the
    qdrant_client.models.Filter pydantic issues in v1.12.x.
    """
    models = get_models()
    if models is None:
        return False
    _, _, qdrant = models
    try:
        # Use the dict-based filter form which is stable across client versions
        results, _ = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={
                "must": [
                    {
                        "key":   "session_id",
                        "match": {"value": session_id},
                    }
                ]
            },
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0
    except Exception as e:
        # If the check itself fails, allow indexing to proceed
        emit(f"session_already_indexed check failed ({e}) — proceeding with index.", "error")
        return False


def embed_and_index(
    chunks:      list[str],
    metadata:    dict,
    session_id:  str,
    source_name: str = "source",
) -> int:
    """
    Embed chunks with all-mpnet-base-v2, build payloads, upsert to Qdrant.
    Returns number of points successfully upserted (0 on failure).
    """
    if not chunks:
        emit(f"[{source_name}] No chunks to index.", "error")
        return 0

    models = get_models()
    if models is None:
        emit(f"[{source_name}] Model registry unavailable.", "error")
        return 0

    embed_model, _, qdrant = models

    # ── Batch embed ───────────────────────────────────────────────────────────
    emit(f"[{source_name}] Embedding {len(chunks)} chunks (all-mpnet-base-v2)…")
    try:
        vectors = embed_model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emit(f"[{source_name}] Embedding complete.", "success")
    except Exception as e:
        emit(f"[{source_name}] Embedding failed: {e}", "error")
        return 0

    # ── Build point structs ───────────────────────────────────────────────────
    points = build_points(
        chunks=chunks,
        vectors=[v.tolist() for v in vectors],
        metadata=metadata,
        session_id=session_id,
        source_name=source_name,
    )

    # ── Upsert ───────────────────────────────────────────────────────────────
    emit(f"[{source_name}] Upserting {len(points)} vectors into Qdrant…")
    try:
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
        emit(f"[{source_name}] Indexed {len(points)} chunks successfully.", "success")
        return len(points)
    except Exception as e:
        emit(f"[{source_name}] Qdrant upsert failed: {e}", "error")
        return 0
