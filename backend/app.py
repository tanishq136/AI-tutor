import os
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from stt_service import ElevenLabsSTTStreamer
import pypdf
import docx

load_dotenv()

# Resolve paths relative to this file so the server works from any cwd
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
_FRONTEND  = os.path.normpath(os.path.join(_BASE_DIR, '..', 'frontend'))
_TTS_DIR   = os.path.join(_BASE_DIR, 'tts_output')   # matches tts_service.OUTPUT_DIR

app = Flask(__name__, static_folder=_FRONTEND)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = os.path.join(_BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(_TTS_DIR, exist_ok=True)

# Active STT streamers keyed by socket session ID
active_streamers: dict[str, ElevenLabsSTTStreamer] = {}


def on_transcription(sid, text, is_final):
    socketio.emit('transcription_chunk', {'text': text, 'isFinal': is_final}, to=sid)


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Health-check endpoint — visit /diagnostics in the browser to debug."""
    import sys
    report = {}

    # ── 1. Paths ──────────────────────────────────────────────────────────────
    from rag.config import QDRANT_DB_PATH, QDRANT_COLLECTION, EMBED_MODEL_NAME, VECTOR_DIM
    report["qdrant_db_path"]   = QDRANT_DB_PATH
    report["qdrant_db_exists"] = os.path.isdir(QDRANT_DB_PATH)
    report["collection"]       = QDRANT_COLLECTION
    report["embed_model"]      = EMBED_MODEL_NAME
    report["vector_dim"]       = VECTOR_DIM

    # ── 2. Model registry status ──────────────────────────────────────────────
    from rag.model_registry import get_models, get_init_error
    models      = get_models()
    init_error  = get_init_error()
    report["models_ready"]  = models is not None
    report["init_error"]    = init_error

    # ── 3. Qdrant collection info ─────────────────────────────────────────────
    if models is not None:
        try:
            _, _, qdrant = models
            info = qdrant.get_collection(QDRANT_COLLECTION)
            report["vectors_count"]    = info.vectors_count
            report["points_count"]     = info.points_count
            report["collection_status"]= str(info.status)
        except Exception as e:
            report["qdrant_error"] = str(e)

    # ── 4. Azure OpenAI keys present (read fresh from env, not cached module vars) ──
    report["azure_key_set"]      = bool(os.environ.get("AZURE_OPENAI_API_KEY", ""))
    report["azure_endpoint_set"] = bool(os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
    report["azure_deployment"]   = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "not set")
    report["azure_key_preview"]  = os.environ.get("AZURE_OPENAI_API_KEY", "")[:6] + "…" 

    # ── 5. ElevenLabs key present ─────────────────────────────────────────────
    report["elevenlabs_key_set"] = bool(os.environ.get("ELEVENLABS_API_KEY", ""))

    return jsonify(report)


@app.route('/clear_index', methods=['POST'])
def clear_index():
    """
    Wipe the entire Qdrant collection and recreate it empty.
    Call this from the browser console when you want to start fresh:
        fetch('/clear_index', {method:'POST'}).then(r=>r.json()).then(console.log)
    """
    try:
        from rag.model_registry import get_models
        from rag.config import QDRANT_COLLECTION, VECTOR_DIM
        models = get_models()
        if models is None:
            return jsonify({"error": "Models not ready"}), 500
        _, _, qdrant = models
        qdrant.delete_collection(QDRANT_COLLECTION)
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"size": VECTOR_DIM, "distance": "Cosine"},
        )
        return jsonify({"success": True, "message": f"Collection '{QDRANT_COLLECTION}' cleared and recreated."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['GET'])
def search_transcripts():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        from rag.pipeline import search
        results = search(query, top_k=3)
        return jsonify({
            "query":                    query,
            "retrieval_quality_metric": "Cosine Similarity (0 - 1)",
            "results":                  results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data  = request.json
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Full online pipeline: rewrite → retrieve → rerank → LLM
        print(f"[ask] Running pipeline for: {query!r}", flush=True)
        from rag.pipeline import ask as pipeline_ask
        answer_text = pipeline_ask(query)
        print(f"[ask] Pipeline done. Answer length: {len(answer_text)} chars", flush=True)

        # Convert answer to speech and save to tts_output/
        print(f"[ask] Calling TTS. OUTPUT_DIR={_TTS_DIR!r}", flush=True)
        from tts_service import text_to_speech
        audio_file = text_to_speech(answer_text, "answer.mp3")
        print(f"[ask] TTS done. audio_file={audio_file!r}", flush=True)

        return jsonify({
            "answer":    answer_text,
            "audio_url": f"/audio/{audio_file}" if audio_file else None,
        })
    except Exception as e:
        # Print full traceback to terminal so we can see the exact line
        traceback.print_exc()
        return jsonify({"error": str(e), "detail": traceback.format_exc()}), 500


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve TTS-generated MP3 files from backend/tts_output/."""
    return send_from_directory(_TTS_DIR, filename)


@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No selected file"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.txt', '.md', '.pdf', '.docx', '.srt']:
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        text_content = ""
        if ext in ['.txt', '.md', '.srt']:
            text_content = file.read().decode('utf-8', errors='ignore')
        elif ext == '.pdf':
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text_content += (page.extract_text() or "") + "\n"
        elif ext == '.docx':
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text_content += para.text + "\n"

        if not text_content.strip():
            return jsonify({"error": "File was empty or unreadable"}), 400

        # background_index_transcript already spawns a daemon thread internally
        from rag.pipeline import background_index_transcript
        background_index_transcript(text_content, source_name=file.filename)
        return jsonify({"success": True, "message": f"'{file.filename}' received — indexing started."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Static file serving ───────────────────────────────────────────────────────

@app.route('/')
def index():
    resp = send_from_directory(app.static_folder, 'index.html')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.route('/<path:path>')
def static_proxy(path):
    resp = send_from_directory(app.static_folder, path)
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    sid = request.sid
    print(f"[{sid}] Client connected")
    emit('status', {'message': 'Connected to server'})


@socketio.on('audio_source_selected')
def handle_audio_source(data):
    emit('source_ack', {'message': f"Audio source acknowledged: {data.get('label', 'Unknown')}"})


@socketio.on('start_stream')
def handle_start_stream():
    """Lecture recording mode — transcript IS indexed into RAG after stop."""
    sid = request.sid
    print(f"[{sid}] Lecture stream started")
    if sid in active_streamers:
        active_streamers[sid].stop()
    streamer = ElevenLabsSTTStreamer(sid, on_transcription, mode="lecture")
    streamer.start()
    active_streamers[sid] = streamer


@socketio.on('start_ask_stream')
def handle_start_ask_stream():
    """Ask-question mode — transcript is NOT indexed into RAG after stop."""
    sid = request.sid
    print(f"[{sid}] Ask stream started")
    if sid in active_streamers:
        active_streamers[sid].stop()
    streamer = ElevenLabsSTTStreamer(sid, on_transcription, mode="ask")
    streamer.start()
    active_streamers[sid] = streamer


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid        = request.sid
    b64_audio  = data.get('audio')
    if sid in active_streamers and b64_audio:
        active_streamers[sid].send_chunk(b64_audio)


@socketio.on('stop_stream')
def handle_stop_stream():
    sid = request.sid
    print(f"[{sid}] Stream stopped")
    if sid in active_streamers:
        active_streamers[sid].stop()
        del active_streamers[sid]


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in active_streamers:
        active_streamers[sid].stop()
        del active_streamers[sid]
    print(f"[{sid}] Client disconnected")


if __name__ == '__main__':
    # Eagerly load models before accepting requests so the first /ask
    # doesn't pay the 30-60s download + load penalty. Any init errors
    # print to the terminal here rather than silently failing on first use.
    import threading
    def _preload():
        from rag.model_registry import init_models
        ok = init_models()
        if not ok:
            print("[startup] ERROR: model init failed — check terminal above for details.", flush=True)
        else:
            print("[startup] All models loaded and ready.", flush=True)
    threading.Thread(target=_preload, daemon=True, name="model-preload").start()

    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
