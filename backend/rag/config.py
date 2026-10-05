"""
config.py
---------
Single source of truth for all environment variables and constants.
Every other module imports from here — nothing reads os.environ directly.
"""

import os, sys
from dotenv import load_dotenv

# Explicitly find the .env file at backend/.env regardless of cwd.
# config.py lives at  backend/rag/config.py
# .env lives at       backend/.env
# So we go one level up from this file's directory.
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Print confirmation so you can see it in the terminal on startup
_key_preview = os.environ.get("AZURE_OPENAI_API_KEY", "")
print(
    f"[config] .env loaded from: {os.path.abspath(_ENV_PATH)}\n"
    f"[config] AZURE_OPENAI_API_KEY set: {bool(_key_preview)} "
    f"(starts with: {_key_preview[:6]!r})",
    flush=True,
)

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME : str = os.environ.get("EMBED_MODEL_NAME", "all-mpnet-base-v2")
VECTOR_DIM       : int = int(os.environ.get("VECTOR_DIM", "768"))

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANK_MODEL_NAME: str = os.environ.get(
    "RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "lecture_transcripts")
QDRANT_DB_PATH   : str = os.environ.get(
    "QDRANT_DB_PATH",
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qdrant_db")
    ),
)

# ── Azure OpenAI ─────────────────────────────────────────────────────────────
AZURE_OPENAI_API_VERSION: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_DEPLOYMENT : str = os.environ.get("AZURE_OPENAI_DEPLOYMENT",  "gpt-4.1-mini")

# ── Offline pipeline ─────────────────────────────────────────────────────────
CHUNK_SIZE   : int = int(os.environ.get("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "50"))

# ── Online pipeline ──────────────────────────────────────────────────────────
TOP_K_RETRIEVE   : int = int(os.environ.get("TOP_K_RETRIEVE", "10"))
TOP_K_RERANK     : int = int(os.environ.get("TOP_K_RERANK",   "3"))
LLM_CONTEXT_CHARS: int = int(os.environ.get("LLM_CONTEXT_TOKENS", "1200")) * 4


# ── Azure client factory ──────────────────────────────────────────────────────
def get_azure_client():
    """
    Reads env vars fresh on every call — never uses cached module-level strings.
    This guarantees the key is always current even if dotenv loaded late.
    Returns a configured AzureOpenAI client, or None if keys are missing.
    """
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY",  "")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    version  = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    deploy   = os.environ.get("AZURE_OPENAI_DEPLOYMENT",  "gpt-4o-mini")

    if not api_key or not endpoint:
        print(
            f"[config] get_azure_client: missing vars — "
            f"api_key={'set' if api_key else 'MISSING'}, "
            f"endpoint={'set' if endpoint else 'MISSING'}",
            flush=True,
        )
        return None

    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key        = api_key,
        azure_endpoint = endpoint,
        api_version    = version,
    ), deploy   # returns (client, deployment_name) tuple


# ── Shared SocketIO status emitter ───────────────────────────────────────────
def emit(message: str, status: str = "processing") -> None:
    """Broadcast a status update over SocketIO if available, else log to stderr."""
    try:
        from app import socketio
        socketio.emit("rag_status", {"message": message, "status": status})
    except Exception:
        print(f"[rag:{status}] {message}", file=sys.stderr)
