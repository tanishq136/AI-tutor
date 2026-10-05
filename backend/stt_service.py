"""
stt_service.py
==============
Streams PCM audio from the browser to ElevenLabs scribe_v2 in 1.25-second WAV chunks.
Transcribed text is accumulated in a txt file (thread-safe) and handed off to the
offline RAG pipeline when recording stops.

Model  : ElevenLabs scribe_v2  (locked — do not change)
Format : 16-bit mono PCM, 16 kHz
"""

import os, io, wave, threading, base64, sqlite3, requests
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(_BASE_DIR, "transcripts.db")

STT_MODEL   = "scribe_v2"          # ElevenLabs model identifier
STT_LANG    = "en"
CHUNK_SECS  = 1.25                 # seconds of audio per STT call
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2                   # 16-bit = 2 bytes
CHANNELS     = 1
BYTES_PER_CHUNK = int(SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_SECS)
MIN_TAIL_BYTES  = int(SAMPLE_RATE * SAMPLE_WIDTH * 0.5)   # ignore tail < 0.5 s


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS transcriptions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  text TEXT)""")
    conn.commit()
    conn.close()

_init_db()


class ElevenLabsSTTStreamer:
    """
    Lifecycle
    ---------
    1.  Instantiate once per recording session.
    2.  Call start().
    3.  Feed base64-encoded PCM chunks via send_chunk().
    4.  Call stop() when recording ends — returns immediately; cleanup is async.
    """

    def __init__(self, sid: str, on_transcription_callback, mode: str = "lecture"):
        self.sid      = sid
        self.callback = on_transcription_callback
        self.api_key  = os.environ.get("ELEVENLABS_API_KEY", "")
        self.is_running = False
        # mode="lecture" → transcript is indexed into RAG after recording stops
        # mode="ask"     → transcript is NOT indexed (it's a question, not a document)
        self.mode     = mode

        self._pcm_buffer      = bytearray()   # rolling chunk buffer
        self._full_pcm        = bytearray()   # entire session, for WAV save
        self._thread_lock     = threading.Lock()
        self._txt_lock        = threading.Lock()
        self._active_threads: list[threading.Thread] = []

        uploads_dir = os.path.join(_BASE_DIR, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        self._txt_path = os.path.join(uploads_dir, f"recording_{sid}.txt")
        self._wav_path = os.path.join(uploads_dir, f"recording_{sid}.wav")

        # Clear txt for this session
        with open(self._txt_path, "w", encoding="utf-8") as f:
            f.write("")

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        self.is_running = True

    def send_chunk(self, base64_audio: str):
        if not self.is_running:
            return
        try:
            raw = base64.b64decode(base64_audio)
            self._pcm_buffer.extend(raw)
            self._full_pcm.extend(raw)

            while len(self._pcm_buffer) >= BYTES_PER_CHUNK:
                chunk = bytes(self._pcm_buffer[:BYTES_PER_CHUNK])
                del self._pcm_buffer[:BYTES_PER_CHUNK]
                self._dispatch_chunk(chunk)
        except Exception as e:
            print(f"[STT {self.sid}] send_chunk error: {e}")

    def stop(self):
        """Signal stop, flush tail audio synchronously, then hand off cleanup."""
        self.is_running = False

        # Flush tail if long enough (runs on caller thread — small, fast)
        if len(self._pcm_buffer) >= MIN_TAIL_BYTES:
            self._stt_request(bytes(self._pcm_buffer))
        self._pcm_buffer.clear()

        # All remaining work (join threads, save WAV, DB write, RAG) in background
        t = threading.Thread(target=self._cleanup, daemon=True, name=f"stt-cleanup-{self.sid}")
        t.start()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _dispatch_chunk(self, pcm: bytes):
        t = threading.Thread(target=self._stt_request, args=(pcm,), daemon=True)
        with self._thread_lock:
            self._active_threads.append(t)
            self._active_threads = [th for th in self._active_threads if th.is_alive()]
        t.start()

    def _stt_request(self, pcm: bytes):
        if not self.api_key or self.api_key == "your_elevenlabs_api_key_here":
            self.callback(self.sid, "[Missing ELEVENLABS_API_KEY]", True)
            return

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as w:
            w.setnchannels(CHANNELS)
            w.setsampwidth(SAMPLE_WIDTH)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        wav_io.seek(0)

        try:
            resp = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.api_key},
                data={"model_id": STT_MODEL, "language_code": STT_LANG},
                files={"file": ("chunk.wav", wav_io, "audio/wav")},
                timeout=30,
            )
            if resp.ok:
                text = resp.json().get("text", "").strip()
                if text:
                    self.callback(self.sid, text, True)
                    with self._txt_lock:
                        with open(self._txt_path, "a", encoding="utf-8") as f:
                            f.write(text + " ")
            else:
                msg = f"[STT {resp.status_code}]"
                print(f"[STT {self.sid}] {msg}: {resp.text[:200]}")
                self.callback(self.sid, msg, True)
        except requests.Timeout:
            self.callback(self.sid, "[STT timeout]", True)
        except Exception as e:
            self.callback(self.sid, f"[STT error: {e}]", True)

    def _cleanup(self):
        # Wait for all in-flight chunk threads
        with self._thread_lock:
            threads = list(self._active_threads)
        for t in threads:
            t.join(timeout=15)

        # Save full-session WAV
        if len(self._full_pcm) > 0:
            try:
                with wave.open(self._wav_path, "wb") as w:
                    w.setnchannels(CHANNELS)
                    w.setsampwidth(SAMPLE_WIDTH)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(self._full_pcm)
            except Exception as e:
                print(f"[STT {self.sid}] WAV save error: {e}")
        self._full_pcm.clear()

        # Read final transcript
        try:
            with open(self._txt_path, "r", encoding="utf-8") as f:
                final_text = f.read().strip()
        except Exception:
            final_text = ""

        if not final_text:
            return

        # Persist to SQLite
        try:
            session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                "INSERT INTO transcriptions (timestamp, text) VALUES (?, ?)",
                (session_time, final_text),
            )
            conn.commit()
            conn.close()
            print(f"[STT {self.sid}] Saved to DB.")
        except Exception as e:
            print(f"[STT {self.sid}] DB save error: {e}")
            session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Only index into RAG for lecture recordings, not for ask-questions
        if self.mode == "lecture":
            from rag.pipeline import background_index_transcript
            background_index_transcript(final_text, session_time, source_name=f"session:{self.sid}")
            print(f"[STT {self.sid}] Handed off to RAG pipeline (lecture mode).")
        else:
            print(f"[STT {self.sid}] Ask mode — skipping RAG indexing, transcript saved to txt only.")
