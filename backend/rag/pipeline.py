"""
pipeline.py
-----------
The single public surface for app.py and stt_service.py.
Imports nothing from offline/* or online/* at module level —
all imports are local to keep startup instant.

Offline pipeline
----------------
background_index_transcript(text, session_id, source_name)
    Fire-and-forget. Runs: cleaner → chunker → indexer in a daemon thread.

process_lecture_transcript(text, session_id, source_name)
    Blocking version. Call directly if you need to await completion.

Online pipeline
---------------
ask_stream(query, top_k_retrieve, top_k_rerank)
    Full online path. Yields TTS-ready sentences.
    Wire into tts_service.text_to_speech_stream() per sentence.

ask(query, top_k_retrieve, top_k_rerank) -> str
    Blocking wrapper. Returns complete answer string.

search(query, top_k) -> list[dict]
    Retrieval-only (no LLM). Returns ranked chunks for debugging.
"""

import threading
from datetime import datetime
from .config import TOP_K_RETRIEVE, TOP_K_RERANK, emit


# ═══════════════════════════════════════════════════════════════════════════════
#  OFFLINE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_lecture_transcript(
    transcript_text: str,
    session_id:   str | None = None,
    source_name:  str = "Microphone Recording",
) -> None:
    """
    Blocking offline pipeline.
    Steps: clean + extract metadata → chunk → embed → upsert to Qdrant.
    """
    from .offline.cleaner  import clean_and_extract
    from .offline.chunker  import chunk_text
    from .offline.indexer  import embed_and_index, session_already_indexed
    from .model_registry   import get_models, get_init_error

    if not transcript_text.strip():
        emit(f"[{source_name}] Empty transcript — nothing to index.", "error")
        return

    models = get_models()
    if models is None:
        error = get_init_error() or "Unknown init error"
        emit(f"[{source_name}] Model init failed: {error}", "error")
        return

    sid = session_id or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emit(f"[{source_name}] Session ID: {sid}")

    if session_already_indexed(sid):
        emit(f"[{source_name}] Session already indexed — skipping.", "success")
        return

    emit(f"[{source_name}] ── OFFLINE PIPELINE START ──")
    emit(f"[{source_name}] Input: {len(transcript_text)} chars")

    # Step 1 — knowledge-aware clean + metadata extraction
    # Falls back to raw text gracefully if OPENAI_API_KEY is not set
    emit(f"[{source_name}] Step 1/3: cleaning + extracting metadata…")
    cleaned_text, metadata = clean_and_extract(transcript_text, source_name)
    emit(f"[{source_name}] Cleaned text: {len(cleaned_text)} chars, metadata keys: {list(metadata.keys())}")

    # Step 2 — semantic chunking
    emit(f"[{source_name}] Step 2/3: chunking…")
    chunks = chunk_text(cleaned_text)
    if not chunks:
        emit(f"[{source_name}] Chunking produced 0 chunks — aborting.", "error")
        return
    emit(f"[{source_name}] {len(chunks)} chunks created.", "success")

    # Step 3 — embed + upsert
    emit(f"[{source_name}] Step 3/3: embedding + indexing…")
    n = embed_and_index(chunks, metadata, sid, source_name)
    if n > 0:
        emit(f"[{source_name}] ── OFFLINE PIPELINE COMPLETE: {n} vectors stored ──", "success")
    else:
        emit(f"[{source_name}] ── OFFLINE PIPELINE FAILED: 0 vectors stored ──", "error")


def background_index_transcript(
    text:        str,
    session_id:  str | None = None,
    source_name: str = "Microphone Recording",
) -> None:
    """Non-blocking wrapper. Returns immediately; work runs in a daemon thread."""
    t = threading.Thread(
        target=process_lecture_transcript,
        args=(text, session_id, source_name),
        daemon=True,
        name=f"offline-{session_id or 'new'}",
    )
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
#  ONLINE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def ask_stream(
    query:          str,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_rerank:   int = TOP_K_RERANK,
):
    """
    Full online pipeline. Yields TTS-ready sentence strings.

    Steps
    -----
    1. Query rewrite  (fix STT noise)
    2. ANN retrieval  (top_k_retrieve candidates from Qdrant)
    3. Cross-encoder rerank  (top_k_rerank kept)
    4. LLM agent      (streams answer, buffered into sentences)

    Usage
    -----
    for sentence in pipeline.ask_stream(query):
        for audio in tts_service.text_to_speech_stream(sentence):
            socketio.emit("audio_chunk", audio)
    """
    from .online.retriever  import query_rewrite, retrieve, rerank
    from .online.llm_agent  import answer_sentences
    from .model_registry    import get_models, get_init_error

    models = get_models()
    if models is None:
        error = get_init_error() or "Unknown init error — check server terminal."
        emit(f"Model init failed: {error}", "error")
        yield f"System error: {error}"
        return

    emit(f"Online pipeline: '{query}'")

    # 1. Rewrite
    clean_query = query_rewrite(query)

    # 2. Retrieve
    candidates = retrieve(clean_query, top_k=top_k_retrieve)
    if not candidates:
        yield "I couldn't find any relevant information in the lecture knowledge base."
        return

    # 3. Rerank
    top_chunks = rerank(clean_query, candidates, top_k=top_k_rerank)

    # 4. LLM → sentence stream (TTS-ready)
    yield from answer_sentences(clean_query, top_chunks)


def ask(
    query:          str,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_rerank:   int = TOP_K_RERANK,
) -> str:
    """Blocking wrapper. Returns the full answer as one string."""
    return " ".join(ask_stream(query, top_k_retrieve, top_k_rerank))


def search(query: str, top_k: int = TOP_K_RERANK) -> list[dict]:
    """
    Retrieval-only helper (no LLM call) — useful for debugging and inspection.
    Returns reranked chunks: [{text, score, session_id, lecture_title}, …]
    """
    from .online.retriever import query_rewrite, retrieve, rerank
    from .model_registry   import get_models, get_init_error

    if get_models() is None:
        error = get_init_error() or "Unknown init error"
        emit(f"Model init failed: {error}", "error")
        return []

    clean_query = query_rewrite(query)
    candidates  = retrieve(clean_query, top_k=top_k * 3)
    top         = rerank(clean_query, candidates, top_k=top_k)
    return [
        {
            "text":          c["text"],
            "score":         round(c["score"], 4),
            "session_id":    c["payload"].get("session_id"),
            "lecture_title": c["payload"].get("lecture_title", "Unknown"),
            "topics":        c["payload"].get("topics", []),
        }
        for c in top
    ]
