document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // ── UI elements ───────────────────────────────────────────────────────────
    const wsDot          = document.getElementById('ws-dot');
    const wsStatus       = document.getElementById('ws-status');
    const audioSelect    = document.getElementById('audio-source');
    const systemAck      = document.getElementById('system-ack');
    const ragTerminal    = document.getElementById('rag-terminal');

    // Upload
    const uploadBtn      = document.getElementById('upload-btn');
    const fileInput      = document.getElementById('file-upload');
    const uploadStatus   = document.getElementById('upload-status');

    // Ask-by-voice
    const askMicBtn      = document.getElementById('ask-mic-btn');
    const askMicLabel    = document.getElementById('ask-mic-label');
    const questionPreview= document.getElementById('question-preview');
    const questionText   = document.getElementById('question-text');
    const chatAnswer     = document.getElementById('chat-answer');
    const answerText     = document.getElementById('answer-text');
    const answerAudio    = document.getElementById('answer-audio');

    // ── Audio recording state ─────────────────────────────────────────────────
    let isAskRecording   = false;
    let selectedDeviceId = null;
    let audioContext, scriptProcessor, mediaStreamSource;
    let accumulatedQuestion = '';   // built up from transcription_chunk events

    // ── Helpers ───────────────────────────────────────────────────────────────

    function logToTerminal(message, status = 'processing') {
        if (!ragTerminal) return;
        const placeholder = ragTerminal.querySelector('.placeholder');
        if (placeholder) placeholder.remove();
        const p = document.createElement('p');
        p.className = `terminal-line ${status}`;
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        ragTerminal.appendChild(p);
        ragTerminal.scrollTop = ragTerminal.scrollHeight;
    }

    // ── WebSocket status ──────────────────────────────────────────────────────

    socket.on('connect', () => {
        wsDot.className     = 'dot green';
        wsStatus.textContent = 'Server connected via WebSockets';
    });

    socket.on('disconnect', () => {
        wsDot.className     = 'dot red';
        wsStatus.textContent = 'Server disconnected';
        if (isAskRecording) stopAskRecording(false);  // stop without auto-submitting
    });

    socket.on('source_ack', (data) => {
        systemAck.textContent = data.message;
        systemAck.classList.remove('hidden');
        setTimeout(() => systemAck.classList.add('hidden'), 4000);
    });

    // ── RAG terminal ──────────────────────────────────────────────────────────

    socket.on('rag_status', (data) => {
        logToTerminal(data.message, data.status);
    });

    // ── Transcription chunks — accumulate into the question ───────────────────
    // These come from the same ElevenLabsSTTStreamer used for lecture recording.
    // When the user stops the ask-mic, we collect everything into accumulatedQuestion.

    socket.on('transcription_chunk', (data) => {
        if (!isAskRecording) return;
        const chunk = data.text.trim();
        if (!chunk) return;

        accumulatedQuestion = (accumulatedQuestion + ' ' + chunk).trim();

        // Show live preview above the answer panel
        questionText.textContent  = accumulatedQuestion;
        questionPreview.classList.remove('hidden');

        // Log each chunk to the terminal so it's visible
        logToTerminal(`STT: "${chunk}"`, 'processing');
    });

    // ── Audio device enumeration ──────────────────────────────────────────────

    async function getAudioDevices() {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
            const devices     = await navigator.mediaDevices.enumerateDevices();
            const audioInputs = devices.filter(d => d.kind === 'audioinput');

            audioSelect.innerHTML = '';
            audioInputs.forEach(device => {
                const opt = document.createElement('option');
                opt.value = device.deviceId;
                opt.text  = device.label || `Microphone ${audioSelect.length + 1}`;
                audioSelect.appendChild(opt);
            });

            if (audioInputs.length > 0) {
                selectedDeviceId = audioInputs[0].deviceId;
                notifyBackendOfSource(audioInputs[0]);
            }
        } catch (err) {
            logToTerminal(`Mic permission denied: ${err.message}`, 'error');
            wsStatus.textContent = 'Mic permission denied';
            wsDot.className = 'dot red';
        }
    }

    function notifyBackendOfSource(device) {
        socket.emit('audio_source_selected', {
            deviceId: device.deviceId,
            label:    device.label || 'Default Microphone',
        });
    }

    audioSelect.addEventListener('change', e => {
        selectedDeviceId = e.target.value;
        const opt = e.target.options[e.target.selectedIndex];
        notifyBackendOfSource({ deviceId: opt.value, label: opt.text });
    });

    // ── Ask-by-voice recording ────────────────────────────────────────────────

    async function startAskRecording() {
        try {
            const constraints = {
                audio: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true,
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);

            audioContext      = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            mediaStreamSource = audioContext.createMediaStreamSource(stream);
            scriptProcessor   = audioContext.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (event) => {
                if (!isAskRecording) return;

                // Float32 → 16-bit PCM
                const inputData = event.inputBuffer.getChannelData(0);
                const pcm16     = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    const s  = Math.max(-1, Math.min(1, inputData[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                // PCM → Base64
                let binary = '';
                const bytes = new Uint8Array(pcm16.buffer);
                for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
                socket.emit('audio_chunk', { audio: window.btoa(binary) });
            };

            mediaStreamSource.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);

            // Reset state for this new question
            accumulatedQuestion  = '';
            questionText.textContent = '';
            questionPreview.classList.add('hidden');
            chatAnswer.classList.add('hidden');
            answerAudio.classList.add('hidden');

            isAskRecording = true;

            // Update button to "Stop & Ask" state
            askMicBtn.className  = 'btn recording ask-mic-recording';
            askMicLabel.textContent = 'Stop & Ask';

            logToTerminal('Listening for question…', 'processing');
            // start_ask_stream tells the backend NOT to index this into RAG —
            // it is a question, not a lecture document
            socket.emit('start_ask_stream');

        } catch (err) {
            logToTerminal(`Mic error: ${err.message}`, 'error');
        }
    }

    function stopAskRecording(autoSubmit = true) {
        if (!isAskRecording) return;
        isAskRecording = false;

        // Tear down audio pipeline
        if (scriptProcessor)   { scriptProcessor.disconnect();   scriptProcessor   = null; }
        if (mediaStreamSource) { mediaStreamSource.disconnect(); mediaStreamSource = null; }
        if (audioContext)      { audioContext.close();           audioContext      = null; }

        // Reset button
        askMicBtn.className     = 'btn primary ask-mic-idle';
        askMicLabel.textContent = 'Ask a Question';

        // Tell the backend — this triggers _cleanup → saves txt → indexes to RAG
        socket.emit('stop_stream');

        const query = accumulatedQuestion.trim();
        logToTerminal(`Question captured: "${query}"`, 'success');

        // Auto-submit to /ask after a short pause to let the final STT chunks arrive
        if (autoSubmit && query) {
            setTimeout(() => submitQuestion(query), 800);
        } else if (autoSubmit && !query) {
            logToTerminal('No speech detected — please try again.', 'error');
        }
    }

    askMicBtn.addEventListener('click', () => {
        if (!isAskRecording) {
            startAskRecording();
        } else {
            stopAskRecording(true);
        }
    });

    // ── Submit question to /ask ───────────────────────────────────────────────

    async function submitQuestion(query) {
        if (!query) return;

        askMicBtn.disabled      = true;
        chatAnswer.classList.remove('hidden');
        answerText.textContent  = 'Searching knowledge base…';
        answerAudio.classList.add('hidden');
        logToTerminal(`Querying agent: "${query}"`, 'processing');

        try {
            const resp = await fetch('/ask', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ query }),
            });
            const data = await resp.json();

            if (resp.ok) {
                answerText.textContent = data.answer;
                logToTerminal('Answer generated.', 'success');

                if (data.audio_url) {
                    answerAudio.src = data.audio_url + '?t=' + Date.now();
                    answerAudio.classList.remove('hidden');
                    answerAudio.play().catch(e => console.log('Auto-play blocked:', e));
                }
            } else {
                answerText.textContent = `Error: ${data.error}`;
                logToTerminal(`Ask error: ${data.error}`, 'error');
            }
        } catch (err) {
            answerText.textContent = 'Network error.';
            logToTerminal(`Network error: ${err.message}`, 'error');
        } finally {
            askMicBtn.disabled = false;
        }
    }

    // ── Document upload ───────────────────────────────────────────────────────

    uploadBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a file first.';
            uploadStatus.style.color = 'var(--danger-color)';
            return;
        }

        uploadStatus.textContent = 'Uploading…';
        uploadStatus.style.color = 'var(--primary-color)';
        uploadBtn.disabled       = true;
        logToTerminal(`Uploading: ${file.name}`, 'processing');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp   = await fetch('/upload_doc', { method: 'POST', body: formData });
            const result = await resp.json();

            if (resp.ok) {
                uploadStatus.textContent = `Done: ${result.message}`;
                uploadStatus.style.color = 'var(--success-color)';
                fileInput.value          = '';
                logToTerminal(`Accepted: ${file.name}`, 'success');
            } else {
                uploadStatus.textContent = `Error: ${result.error || 'Upload failed'}`;
                uploadStatus.style.color = 'var(--danger-color)';
                logToTerminal(`Upload error: ${result.error}`, 'error');
            }
        } catch (e) {
            uploadStatus.textContent = `Network error: ${e.message}`;
            uploadStatus.style.color = 'var(--danger-color)';
            logToTerminal(`Upload network error: ${e.message}`, 'error');
        } finally {
            uploadBtn.disabled = false;
        }
    });

    // ── Boot ──────────────────────────────────────────────────────────────────
    getAudioDevices();
});
