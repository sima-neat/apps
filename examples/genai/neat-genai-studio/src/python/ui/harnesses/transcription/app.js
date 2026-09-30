/* Transcription API harness: exercises the Studio's OpenAI-compatible
 * speech-to-text endpoint from the same origin. No external requests.
 *
 *   POST ../../v1/audio/transcriptions  multipart: file, language?, model?, response_format
 *
 * Audio comes from the microphone (MediaRecorder, same MIME preference order as
 * the Studio's own recorder) or a chosen file. The clip is uploaded with its
 * real filename and MIME type; the model server decodes WebM/Opus, MP4/AAC,
 * WAV and friends. WAV is the portable choice for scripted use.
 */
(function () {
  'use strict';

  const API_BASE = new URL('../../', window.location.href).href.replace(/\/$/, '');
  const LANGUAGES = [
    ['auto', 'auto (detect)'], ['en', 'en · English'], ['fr', 'fr · French'], ['es', 'es · Spanish'],
    ['de', 'de · German'], ['it', 'it · Italian'], ['pt', 'pt · Portuguese'], ['ja', 'ja · Japanese'],
    ['ko', 'ko · Korean'], ['zh', 'zh · Chinese'], ['vi', 'vi · Vietnamese'], ['no', 'no · Norwegian'],
  ];
  const MIME_PREFERENCE = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/wav'];

  const $ = (id) => document.getElementById(id);
  const dom = {
    record: $('record'), stopRec: $('stopRec'), recDot: $('recDot'), recTime: $('recTime'),
    file: $('file'), clipInfo: $('clipInfo'), preview: $('preview'), language: $('language'),
    format: $('format'), model: $('model'), transcribe: $('transcribe'), cancel: $('cancel'),
    status: $('status'), result: $('result'), meta: $('meta'), curl: $('curl'), raw: $('raw'),
    conn: $('conn'), home: $('home-button'),
    mModel: $('mModel'), mLanguage: $('mLanguage'), mNoSpeech: $('mNoSpeech'),
    mLogprob: $('mLogprob'), mIgnored: $('mIgnored'), mElapsed: $('mElapsed'),
  };

  let clip = null;            // { blob, name, type }
  let recorder = null;
  let stream = null;
  let chunks = [];
  let timer = null;
  let previewUrl = null;
  let controller = null;

  function setStatus(text, kind) {
    dom.status.textContent = text || '';
    dom.status.className = 'status' + (kind ? ' ' + kind : '');
  }

  function pickMime() {
    if (!window.MediaRecorder || !MediaRecorder.isTypeSupported) return '';
    return MIME_PREFERENCE.find((m) => MediaRecorder.isTypeSupported(m)) || '';
  }

  function extensionFor(type) {
    const base = (type || '').split(';')[0];
    return { 'audio/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/wav': 'wav',
             'audio/x-wav': 'wav', 'audio/mpeg': 'mp3', 'audio/flac': 'flac' }[base] || 'bin';
  }

  function setClip(blob, name, type) {
    clip = { blob, name, type: type || blob.type || 'application/octet-stream' };
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(blob);
    dom.preview.src = previewUrl;
    dom.preview.hidden = false;
    dom.clipInfo.textContent = `${name} · ${clip.type} · ${(blob.size / 1024).toFixed(0)} KiB`;
    dom.transcribe.disabled = false;
    updateCurl();
  }

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      setStatus(`Microphone unavailable: ${err.message}`, 'error');
      return;
    }
    const mime = pickMime();
    try {
      recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
    } catch (err) {
      setStatus(`Recording not supported here: ${err.message}`, 'error');
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    chunks = [];
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const type = recorder.mimeType || mime || 'audio/webm';
      const blob = new Blob(chunks, { type });
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
      clearInterval(timer);
      dom.recDot.classList.remove('live');
      dom.record.disabled = false;
      dom.stopRec.disabled = true;
      if (!blob.size) { setStatus('Nothing was recorded.', 'error'); return; }
      setClip(blob, `recording.${extensionFor(type)}`, type);
      setStatus('Clip ready.');
    };
    recorder.start();
    const started = Date.now();
    dom.recTime.textContent = '0.0 s';
    timer = setInterval(() => { dom.recTime.textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`; }, 200);
    dom.recDot.classList.add('live');
    dom.record.disabled = true;
    dom.stopRec.disabled = false;
    setStatus('Recording… press Stop when done.');
  }

  function stopRecording() {
    if (recorder && recorder.state !== 'inactive') recorder.stop();
  }

  function formData() {
    const fd = new FormData();
    fd.append('file', clip.blob, clip.name);
    if (dom.language.value && dom.language.value !== 'auto') fd.append('language', dom.language.value);
    if (dom.model.value.trim()) fd.append('model', dom.model.value.trim());
    fd.append('response_format', dom.format.value);
    return fd;
  }

  function updateCurl() {
    const parts = [`curl -k -X POST ${API_BASE}/v1/audio/transcriptions \\`,
                   `  -F 'file=@${clip ? clip.name : 'recording.wav'}' \\`];
    if (dom.language.value && dom.language.value !== 'auto') parts.push(`  -F 'language=${dom.language.value}' \\`);
    if (dom.model.value.trim()) parts.push(`  -F 'model=${dom.model.value.trim()}' \\`);
    parts.push(`  -F 'response_format=${dom.format.value}' -D -`);
    dom.curl.textContent = parts.join('\n');
  }

  function fmt(value, digits) {
    return (typeof value === 'number' && Number.isFinite(value)) ? value.toFixed(digits) : (value == null ? '–' : String(value));
  }

  async function transcribe() {
    if (!clip) { setStatus('Record or choose a clip first.', 'error'); return; }
    controller = new AbortController();
    dom.transcribe.disabled = true;
    dom.cancel.disabled = false;
    dom.result.textContent = '';
    dom.meta.hidden = true;
    setStatus('Transcribing…');
    const started = performance.now();
    try {
      const res = await fetch(`${API_BASE}/v1/audio/transcriptions`, {
        method: 'POST', body: formData(), signal: controller.signal,
      });
      const contentType = res.headers.get('Content-Type') || '';
      const rawText = await res.text();
      dom.raw.textContent = rawText;
      if (!res.ok) {
        let message = `HTTP ${res.status}`;
        try { const err = JSON.parse(rawText); message = err.error || message; if (err.param) message += ` (param: ${err.param})`; } catch (e) { /* not JSON */ }
        throw new Error(message);
      }
      const ms = performance.now() - started;
      dom.mModel.textContent = res.headers.get('X-ASR-Model') || '–';
      dom.mElapsed.textContent = fmt(Number(res.headers.get('X-Elapsed-Time')), 2);
      if (contentType.startsWith('text/plain')) {
        dom.result.textContent = rawText.trim() || '(empty transcript)';
        dom.mLanguage.textContent = dom.mNoSpeech.textContent = dom.mLogprob.textContent = dom.mIgnored.textContent = '–';
      } else {
        const data = JSON.parse(rawText);
        dom.result.textContent = data.text || '(empty transcript)';
        dom.mLanguage.textContent = data.language ? `${data.language}${data.language_detected ? ' (detected)' : ''}` : '–';
        dom.mNoSpeech.textContent = fmt(data.no_speech_prob, 3);
        dom.mLogprob.textContent = fmt(data.avg_logprob, 3);
        dom.mIgnored.textContent = data.ignored ? `yes · ${data.reason || ''}` : (data.ignored === false ? 'no' : '–');
      }
      dom.meta.hidden = false;
      setStatus(`OK in ${(ms / 1000).toFixed(2)} s round trip`, 'ok');
    } catch (err) {
      if (err.name === 'AbortError') setStatus('Cancelled.');
      else setStatus(err.message, 'error');
    } finally {
      controller = null;
      dom.transcribe.disabled = !clip;
      dom.cancel.disabled = true;
    }
  }

  LANGUAGES.forEach(([code, label]) => {
    const o = document.createElement('option');
    o.value = code; o.textContent = label;
    dom.language.appendChild(o);
  });
  dom.record.addEventListener('click', startRecording);
  dom.stopRec.addEventListener('click', stopRecording);
  dom.file.addEventListener('change', () => {
    const f = dom.file.files && dom.file.files[0];
    if (f) { setClip(f, f.name, f.type); setStatus('File selected.'); }
  });
  ['change', 'input'].forEach((ev) => {
    dom.language.addEventListener(ev, updateCurl);
    dom.format.addEventListener(ev, updateCurl);
    dom.model.addEventListener(ev, updateCurl);
  });
  dom.transcribe.addEventListener('click', transcribe);
  dom.cancel.addEventListener('click', () => { if (controller) controller.abort(); });
  dom.home.addEventListener('click', () => {
    stopRecording();
    if (window.parent && window.parent !== window) window.parent.postMessage({ type: 'sima-sentry:home' }, '*');
    else window.location.assign('../index.html');
  });
  window.addEventListener('pagehide', () => { stopRecording(); if (stream) stream.getTracks().forEach((t) => t.stop()); });

  dom.conn.textContent = window.isSecureContext ? (navigator.mediaDevices ? 'microphone available' : 'no microphone API') : 'HTTPS needed for the microphone';
  updateCurl();
})();
