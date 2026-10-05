"""
offline/chunker.py
------------------
Splits cleaned lecture text into overlapping chunks and assigns each
a UUID derived deterministically from the chunk text so upserts are
idempotent (same content always gets the same UUID).

Qdrant requires IDs to be either unsigned integers or valid UUID strings.
We use uuid5(NAMESPACE_OID, chunk_text) which is deterministic and
always produces a valid UUID — no silent upsert failures.
"""

import uuid
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_id(text: str) -> str:
    """
    Deterministic UUID v5 from chunk text.
    Same text always produces the same UUID — safe to upsert repeatedly.
    Returns a lowercase UUID string e.g. '550e8400-e29b-41d4-a716-446655440000'
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))


def chunk_text(text: str) -> list[str]:
    """
    Split text using recursive character splitting.
    Boundaries tried in order: paragraph → newline → sentence → word → char.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_text(text)


def build_points(
    chunks:      list[str],
    vectors:     list[list[float]],
    metadata:    dict,
    session_id:  str,
    source_name: str,
) -> list[dict]:
    """
    Pair each chunk with its vector and attach full metadata as payload.
    Returns a list of PointStruct objects ready for QdrantClient.upsert().
    """
    from qdrant_client.models import PointStruct

    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        cid = chunk_id(chunk)
        payload = {
            "text":            chunk,
            "session_id":      session_id,
            "chunk_index":     i,
            "chunk_id":        cid,
            "source_name":     source_name,
            "lecture_title":   metadata.get("lecture_title",  "Unknown"),
            "topics":          metadata.get("topics",          []),
            "key_terms":       metadata.get("key_terms",       []),
            "difficulty":      metadata.get("difficulty",      "unknown"),
            "lecture_summary": metadata.get("summary",         ""),
        }
        points.append(PointStruct(id=cid, vector=vec, payload=payload))

    return points
