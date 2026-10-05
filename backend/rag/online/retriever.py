"""
online/retriever.py
-------------------
  1. query_rewrite(raw_query)  → clean query string   (fixes STT noise via Azure)
  2. retrieve(query, top_k)    → candidate list        (ANN cosine search)
  3. rerank(query, candidates) → top-k list            (cross-encoder scoring)
  4. build_context(chunks)     → trimmed context str   (token-budget assembly)
"""

import time
from ..config import (
    QDRANT_COLLECTION, LLM_CONTEXT_CHARS,
    TOP_K_RETRIEVE, TOP_K_RERANK,
    get_azure_client, emit,
)
from ..model_registry import get_models

_REWRITE_PROMPT = """\
You are a query preprocessor for a university lecture Q&A system.
The student's question was captured via speech-to-text and may contain:
  - Mis-heard words ("eig en values" → "eigenvalues")
  - Run-together words ("whatis" → "what is")
  - Wrong capitalisation or punctuation
Rewrite the question to be grammatically clean and well-formed.
If the question is already correct, return it UNCHANGED.
Return ONLY the rewritten question. No explanation. No quotes."""


# ── Step 1: Query rewrite ─────────────────────────────────────────────────────

def query_rewrite(raw_query: str) -> str:
    """Fix STT noise before embedding. Falls back to original if Azure unavailable."""
    if not raw_query.strip():
        return raw_query

    result = get_azure_client()
    if result is None:
        return raw_query
    client, deployment = result

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": _REWRITE_PROMPT},
                {"role": "user",   "content": raw_query},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and rewritten != raw_query:
            emit(f"Query rewritten: '{raw_query}' → '{rewritten}'")
        return rewritten or raw_query
    except Exception as e:
        emit(f"Query rewrite failed ({e}), using original.", "error")
        return raw_query


# ── Step 2: ANN retrieval ─────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Embed query and run cosine ANN search in Qdrant."""
    models = get_models()
    if models is None:
        return []
    embed_model, _, qdrant = models

    t0  = time.perf_counter()
    vec = embed_model.encode([query], convert_to_numpy=True)[0]
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vec.tolist(),
        limit=top_k,
    )
    ms = (time.perf_counter() - t0) * 1000
    emit(f"Retrieved {len(hits)} candidates in {ms:.0f} ms.")
    return [
        {"text": h.payload.get("text", ""), "score": h.score, "payload": h.payload}
        for h in hits
    ]


# ── Step 3: Cross-encoder rerank ──────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_RERANK) -> list[dict]:
    """Score all candidates with cross-encoder, return top_k highest scoring."""
    if not candidates:
        return []

    models = get_models()
    if models is None:
        return candidates[:top_k]

    _, reranker, _ = models

    t0     = time.perf_counter()
    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [c for _, c in ranked[:top_k]]
    ms     = (time.perf_counter() - t0) * 1000
    emit(f"Reranked {len(candidates)} → {len(top)} in {ms:.0f} ms.", "success")
    return top


# ── Step 4: Context assembly ──────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Concatenate reranked chunks up to the token budget."""
    parts: list[str] = []
    total = 0
    for i, c in enumerate(chunks):
        snippet = f"[Excerpt {i + 1}]\n{c['text']}"
        if total + len(snippet) > LLM_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)
