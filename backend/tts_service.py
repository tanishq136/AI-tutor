"""
tts_service.py
==============
Two entry points:

text_to_speech(text)
    Blocking — sends entire text, saves MP3, returns filename.
    Use for short one-shot utterances.

text_to_speech_stream(text) -> Iterator[bytes]
    Streaming — yields MP3 audio chunks as they arrive.
    Use this in the online pipeline to minimise TTFB:

        from rag.pipeline import ask_stream
        for sentence in ask_stream(query):
            for audio_chunk in tts_service.text_to_speech_stream(sentence):
                socketio.emit("audio_chunk", audio_chunk)

Model  : ElevenLabs eleven_turbo_v2_5  (low-latency, recommended for real-time)
Voice  : configurable via ELEVENLABS_VOICE_ID (default: Rachel)
"""

import os, re, requests
from typing import Iterator

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE_DIR, "tts_output")

# eleven_turbo_v2_5 has the lowest TTFB (~200 ms) while still being high quality.
# Do NOT revert to eleven_monolingual_v1 — that model is deprecated.
TTS_MODEL   = os.environ.get("ELEVENLABS_TTS_MODEL",  "eleven_turbo_v2_5")
TTS_VOICE   = os.environ.get("ELEVENLABS_VOICE_ID",    "21m00Tcm4TlvDq8ikWAM")  # Rachel
TTS_TIMEOUT = int(os.environ.get("ELEVENLABS_TTS_TIMEOUT", "30"))


def _is_sentence_end(text: str) -> bool:
    """True if text ends with a sentence-terminal punctuation mark (ignoring trailing spaces)."""
    return bool(re.search(r'[.!?]\s*$', text))


def _api_key() -> str | None:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key or key == "your_elevenlabs_api_key_here":
        print("TTS: ELEVENLABS_API_KEY not set.")
        return None
    return key


# ── Streaming entry point (use this in the online pipeline) ──────────────────

def text_to_speech_stream(text: str) -> Iterator[bytes]:
    """
    Yields raw MP3 bytes as they stream from ElevenLabs.
    First chunk typically arrives within ~200 ms for eleven_turbo_v2_5.

    Wire directly into a SocketIO emit loop for sub-second audio TTFB.
    """
    key = _api_key()
    if not key or not text.strip():
        return

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE}/stream"
    headers = {
        "Accept":       "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key":   key,
    }
    payload = {
        "text":     text,
        "model_id": TTS_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    try:
        with requests.post(url, json=payload, headers=headers,
                           stream=True, timeout=TTS_TIMEOUT) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=2048):
                if chunk:
                    yield chunk
    except requests.Timeout:
        print(f"TTS stream: timeout after {TTS_TIMEOUT}s (voice={TTS_VOICE}, model={TTS_MODEL})")
    except requests.HTTPError as e:
        print(f"TTS stream: HTTP {e.response.status_code} — {e.response.text[:200]}")
    except Exception as e:
        print(f"TTS stream: {e}")


# ── Blocking entry point (simple use-case / fallback) ───────────────────────

def text_to_speech(text: str, output_filename: str = "answer.mp3") -> str | None:
    """
    Blocking TTS — waits for the complete audio before returning.
    Returns the output filename on success, None on failure.
    """
    key = _api_key()
    if not key or not text.strip():
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, output_filename)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE}"
    headers = {
        "Accept":       "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key":   key,
    }
    payload = {
        "text":     text,
        "model_id": TTS_MODEL,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    print(f"[TTS] POST to {url}", flush=True)
    print(f"[TTS] Writing to: {filepath!r}", flush=True)
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=TTS_TIMEOUT)
        resp.raise_for_status()
        print(f"[TTS] HTTP {resp.status_code} — writing audio bytes…", flush=True)
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"[TTS] Saved: {filepath}", flush=True)
        return output_filename
    except requests.Timeout:
        print(f"[TTS] Timeout after {TTS_TIMEOUT}s", flush=True)
        return None
    except requests.HTTPError as e:
        print(f"[TTS] HTTP error {e.response.status_code}: {e.response.text[:200]}", flush=True)
        return None
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[TTS] Error: {e}", flush=True)
        return None


# ── Online pipeline glue: sentence-streaming LLM → TTS ──────────────────────

def stream_answer_to_audio(token_iterator, emit_audio_chunk, emit_text_token=None):
    """
    Bridges the LLM sentence stream and TTS stream for minimum TTFB.

    Parameters
    ----------
    token_iterator   : Iterator[str]   — from rag.pipeline.ask_stream()
    emit_audio_chunk : Callable[bytes] — called for each MP3 chunk (e.g. socketio.emit)
    emit_text_token  : Callable[str]   — optional, called for each text token (live caption)

    How it works
    ------------
    Accumulates LLM tokens until a sentence boundary is detected ([.!?]),
    then immediately fires a TTS streaming request for that sentence.
    Audio starts playing before the LLM has finished generating the next sentence.

    Example wiring in app.py
    ------------------------
    @socketio.on("ask")
    def handle_ask(data):
        query = data["query"]
        from rag.pipeline import ask_stream
        tts_service.stream_answer_to_audio(
            ask_stream(query),
            emit_audio_chunk=lambda b: socketio.emit("audio_chunk", {"data": b.hex()}),
            emit_text_token =lambda t: socketio.emit("text_token",  {"token": t}),
        )
    """
    buffer = ""
    for token in token_iterator:
        buffer += token
        if emit_text_token:
            emit_text_token(token)

        # Flush on sentence boundary — enables parallel TTS while LLM continues
        if _is_sentence_end(buffer) and len(buffer.strip()) > 10:
            for audio_chunk in text_to_speech_stream(buffer.strip()):
                emit_audio_chunk(audio_chunk)
            buffer = ""

    # Flush any remaining text (last sentence may not end with punctuation)
    if buffer.strip():
        for audio_chunk in text_to_speech_stream(buffer.strip()):
            emit_audio_chunk(audio_chunk)
