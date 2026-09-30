/* Audio API playground for Neat GenAI Studio.
 *
 * Three modes on one same-origin page:
 *   Speech         POST /v1/audio/speech         (GET /v1/audio/voices for the pickers)
 *   Transcription  POST /v1/audio/transcriptions  clip (record / drop a file) or live (hands-free)
 *   Echo           speak → transcriptions → speech → played back, hands-free
 *
 * Playback goes through the Web Audio API from the user's click (the Studio does
 * the same for chat replies) so it works inside the Studio's embedded frame; the
 * visible <audio> controls are a scrubbable fallback. The live modes detect
 * speech in the browser (energy VAD over the microphone), cut utterances at a
 * pause and upload each one as 16 kHz WAV. Nothing leaves this origin.
 */
(() => {
  'use strict';

  const API = window.location.origin;
  const PREFS_KEY = 'audio-playground';
  const LANGUAGE_NAMES = {
    ar: 'Arabic', bg: 'Bulgarian', cs: 'Czech', da: 'Danish', de: 'German', el: 'Greek', en: 'English',
    es: 'Spanish', et: 'Estonian', fi: 'Finnish', fr: 'French', hi: 'Hindi', hr: 'Croatian', hu: 'Hungarian',
    id: 'Indonesian', it: 'Italian', ja: 'Japanese', ko: 'Korean', lt: 'Lithuanian', lv: 'Latvian', nl: 'Dutch',
    no: 'Norwegian', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian', ru: 'Russian', sk: 'Slovak', sl: 'Slovenian',
    sv: 'Swedish', tr: 'Turkish', uk: 'Ukrainian', vi: 'Vietnamese', zh: 'Chinese',
  };
  const ASR_LANGUAGES = ['auto', 'en', 'fr', 'es', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'vi', 'no'];
  const MIME_PREFERENCE = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/wav'];
  const TABS = ['speech', 'transcription', 'echo'];

  const $ = (id) => document.getElementById(id);
  const embedded = window.parent && window.parent !== window;
  const langLabel = (c) => (c === 'auto' ? 'auto · detect' : (LANGUAGE_NAMES[c] ? `${c} · ${LANGUAGE_NAMES[c]}` : c));

  // ---- theme: follow the Studio ------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    $('themeSun').hidden = theme === 'dark';
    $('themeMoon').hidden = theme !== 'dark';
  }
  function storedTheme() {
    try { return localStorage.getItem('theme') || 'dark'; } catch (e) { return 'dark'; }
  }
  applyTheme(storedTheme());
  $('themeButton').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
    applyTheme(next);
  });
  window.addEventListener('storage', (e) => { if (e.key === 'theme' && e.newValue) applyTheme(e.newValue); });

  // ---- header: tabs, back, prefs -----------------------------------------
  function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}'); } catch (e) { return {}; } }
  function savePrefs(patch) {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(Object.assign(loadPrefs(), patch))); } catch (e) { /* ignore */ }
  }
  let currentTab = null;
  let leaveTab = () => {};        // set once the mode objects exist (below)
  function showTab(name) {
    if (!TABS.includes(name)) name = 'speech';
    if (currentTab && currentTab !== name) leaveTab(currentTab);
    currentTab = name;
    document.querySelectorAll('.pg-tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    document.querySelectorAll('.pg-panel').forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
    savePrefs({ tab: name });
    if (history.replaceState) history.replaceState(null, '', `#${name}`);
  }
  document.querySelectorAll('.pg-tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));

  // "Back to Studio" is a plain link to / (target=_top) so it works even if this
  // script never runs. When embedded in the Studio it closes the modal in place
  // instead of reloading; standalone it simply navigates to the Studio.
  const closePlayground = () => window.parent.postMessage({ type: 'sima-studio:close-playground' }, window.location.origin);
  if (embedded) {
    $('backButton').addEventListener('click', (e) => { e.preventDefault(); closePlayground(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePlayground(); });
  } else {
    $('newTab').hidden = true;   // already a tab of its own
  }
  document.querySelectorAll('.copy-btn[data-copy]').forEach((b) => b.addEventListener('click', async () => {
    const text = $(b.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; } catch (e) { b.textContent = 'Select & copy'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  }));

  // ---- small helpers -----------------------------------------------------
  function setStatus(el, text, kind, busy) {
    el.innerHTML = '';
    if (busy) { const s = document.createElement('i'); s.className = 'spinner'; el.appendChild(s); }
    el.appendChild(document.createTextNode(text || ''));
    el.className = 'status' + (kind ? ' ' + kind : '');
  }
  function option(value, label) { const o = document.createElement('option'); o.value = value; o.textContent = label; return o; }
  function fmt(v, d) { return (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(d) : (v == null || v === '' ? '–' : String(v)); }
  function bindSlider(input, output, format) {
    const show = () => { output.value = format(Number(input.value)); };
    input.addEventListener('input', show); show();
    return show;
  }
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function avg(list) { return list.length ? list.reduce((a, b) => a + b, 0) / list.length : NaN; }
  function clock() { const d = new Date(); return d.toTimeString().slice(0, 8); }

  // ---- audio helpers -----------------------------------------------------
  let audioCtx = null;
  function ensureAudioContext() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume().catch(() => {});
    return audioCtx;
  }

  function drawWave(canvas, buffer) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = Math.max(1, Math.floor(w * dpr));
    canvas.height = Math.max(1, Math.floor(h * dpr));
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!buffer) return;
    const data = buffer.getChannelData(0);
    const bars = Math.max(40, Math.floor(w / 3));
    const step = Math.max(1, Math.floor(data.length / bars));
    const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#0f9d8f';
    ctx.fillStyle = accent;
    let peak = 0;
    const peaks = [];
    for (let i = 0; i < bars; i++) {
      let max = 0;
      const start = i * step;
      for (let j = start; j < start + step && j < data.length; j += 4) { const v = Math.abs(data[j]); if (v > max) max = v; }
      peaks.push(max); if (max > peak) peak = max;
    }
    const scale = peak > 0 ? 0.92 / peak : 1;
    const bw = w / bars;
    for (let i = 0; i < bars; i++) {
      const amp = Math.max(1.5, peaks[i] * scale * h);
      ctx.globalAlpha = 0.9;
      ctx.fillRect(i * bw + bw * 0.2, (h - amp) / 2, bw * 0.6, amp);
    }
    ctx.globalAlpha = 1;
  }

  /** One decoded clip with Web Audio playback and a scrolling cursor. play()
   *  resolves when playback ends (or is stopped). */
  function makePlayer({ waveEl, canvas, button, audioEl }) {
    let buffer = null, source = null, startedAt = 0, raf = 0, resolveEnd = null, objectUrl = null;
    let loadGen = 0;                 // stop() bumps it: a load still decoding then goes nowhere
    const cursor = waveEl.querySelector('.cursor');
    const placeholder = waveEl.querySelector('.placeholder');
    const label = button.querySelector('span');

    function stopSource() {          // Web Audio playback only
      if (source) { const s = source; source = null; try { s.stop(); } catch (e) { /* already stopped */ } }
      cancelAnimationFrame(raf);
      waveEl.classList.remove('playing');
      label.textContent = 'Play';
      if (resolveEnd) { const r = resolveEnd; resolveEnd = null; r(); }
    }
    function stop() {                // everything, including a load still in progress
      loadGen += 1;
      stopSource();
      if (!audioEl.paused) { try { audioEl.pause(); } catch (e) { /* not playable */ } }
    }
    function tick() {
      if (!source || !buffer) return;
      const t = (audioCtx.currentTime - startedAt) / buffer.duration;
      cursor.style.left = `${Math.min(100, t * 100)}%`;
      raf = requestAnimationFrame(tick);
    }
    function play() {
      if (!buffer) return Promise.resolve();
      if (source) { stop(); return Promise.resolve(); }
      const ctx = ensureAudioContext();
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = stopSource;
      startedAt = ctx.currentTime;
      source.start();
      waveEl.classList.add('playing');
      label.textContent = 'Stop';
      raf = requestAnimationFrame(tick);
      return new Promise((resolve) => { resolveEnd = resolve; });
    }
    /** Decode and show a clip; resolves when autoplay finishes. A stop() while
     *  decoding (tab left, Cancel) discards the result: nothing is shown or played. */
    async function load(blob, { autoplay } = {}) {
      stop();
      const gen = loadGen;
      buffer = null;
      const ctx = ensureAudioContext();
      const bytes = await blob.arrayBuffer();
      if (gen !== loadGen) return Promise.resolve();
      let decoded = null;
      try { decoded = await ctx.decodeAudioData(bytes.slice(0)); } catch (e) { decoded = null; }
      if (gen !== loadGen) return Promise.resolve();
      buffer = decoded;
      if (buffer) {
        drawWave(canvas, buffer);
        placeholder.hidden = true;
      } else {
        drawWave(canvas, null);
        placeholder.hidden = false;
        placeholder.textContent = 'This clip cannot be decoded by the browser';
      }
      // The scrubbable <audio> plays a WAV re-encoded from the decoded samples:
      // always a document this page produced, never the uploaded file itself.
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
      audioEl.removeAttribute('src');
      audioEl.hidden = !buffer;
      if (buffer) {
        objectUrl = URL.createObjectURL(encodeWav(buffer.getChannelData(0), buffer.sampleRate, buffer.sampleRate));
        audioEl.src = objectUrl;
      }
      button.disabled = !buffer;
      if (autoplay && buffer) return play();
      return Promise.resolve();
    }
    button.addEventListener('click', () => { play(); });
    audioEl.addEventListener('play', () => { if (source) stopSource(); });   // don't double-play
    window.addEventListener('resize', () => { if (buffer) drawWave(canvas, buffer); });
    return { load, stop, play, get buffer() { return buffer; }, get playing() { return !!source; } };
  }

  // ---- WAV encoding (16 kHz mono PCM16) for the live modes --------------
  function resampleTo(samples, fromRate, toRate) {
    if (fromRate === toRate) return samples;
    const ratio = fromRate / toRate;
    const out = new Float32Array(Math.floor(samples.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const pos = i * ratio, i0 = Math.floor(pos), i1 = Math.min(samples.length - 1, i0 + 1), f = pos - i0;
      out[i] = samples[i0] * (1 - f) + samples[i1] * f;
    }
    return out;
  }
  function encodeWav(samples, sampleRate, targetRate = 16000) {
    const pcm = resampleTo(samples, sampleRate, targetRate);
    const buf = new ArrayBuffer(44 + pcm.length * 2);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE');
    str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, targetRate, true); v.setUint32(28, targetRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, pcm.length * 2, true);
    for (let i = 0, o = 44; i < pcm.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, pcm[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  // ---- hands-free microphone with energy VAD -----------------------------
  // Frames (~21 ms) come from an AudioWorklet (ScriptProcessor fallback). A
  // slow-rising / fast-falling noise floor sets the threshold; an utterance
  // starts after two voiced frames (with pre-roll), ends after `silenceMs` of
  // quiet, and is emitted as a 16 kHz WAV blob. pause() mutes the detector
  // (used while the Echo reply plays), resume() re-arms it.
  const WORKLET_SRC = `
    class PgCapture extends AudioWorkletProcessor {
      constructor() { super(); this.buf = new Float32Array(1024); this.n = 0; }
      process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;
        for (let i = 0; i < ch.length; i++) {
          this.buf[this.n++] = ch[i];
          if (this.n === this.buf.length) { this.port.postMessage(this.buf); this.buf = new Float32Array(1024); this.n = 0; }
        }
        return true;
      }
    }
    registerProcessor('pg-capture', PgCapture);`;
  let workletReady = null;

  function createLiveMic(opts) {
    const o = Object.assign({ silenceMs: 700, minSpeechMs: 250, maxSpeechMs: 20000, prerollMs: 320, sensitivity: 0.5 }, opts);
    const h = {};
    const mic = {
      state: 'idle', paused: false,
      on(ev, fn) { h[ev] = fn; return mic; },
      set(patch) { Object.assign(o, patch); },
    };
    const emit = (ev, ...a) => { if (h[ev]) h[ev](...a); };
    let stream = null, node = null, srcNode = null, rate = 48000;
    let floor = -60, voicedRun = 0, quietMs = 0, speechMs = 0, segment = [], segmentMs = 0;
    const preroll = [];
    let prerollMs = 0;

    function reset() { voicedRun = 0; quietMs = 0; speechMs = 0; segment = []; segmentMs = 0; preroll.length = 0; prerollMs = 0; }
    function setState(s) { if (mic.state !== s) { mic.state = s; emit('state', s); } }
    function finish(force) {
      const blob = segment.length && speechMs >= o.minSpeechMs ? encodeWav(concat(segment), rate) : null;
      const dur = segmentMs / 1000;
      reset();
      setState('listening');
      if (blob) emit('segment', blob, dur, !!force);
    }
    function concat(parts) {
      let n = 0; parts.forEach((p) => { n += p.length; });
      const out = new Float32Array(n); let off = 0;
      parts.forEach((p) => { out.set(p, off); off += p.length; });
      return out;
    }
    function frame(samples) {
      const ms = (samples.length / rate) * 1000;
      let sum = 0;
      for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
      const rms = Math.sqrt(sum / samples.length);
      const db = 20 * Math.log10(rms + 1e-7);
      if (mic.paused) { emit('level', 0, false, 0); return; }
      // noise floor: drops quickly, rises slowly (so speech doesn't lift it)
      floor += (db - floor) * (db < floor ? 0.25 : 0.008);
      floor = Math.max(-80, Math.min(-25, floor));
      const margin = 6 + (1 - o.sensitivity) * 14;            // 20 dB (deaf) … 6 dB (keen)
      const threshold = Math.max(floor + margin, -58 + (1 - o.sensitivity) * 10);
      const voiced = db > threshold;
      const norm = (x) => Math.max(0, Math.min(1, (x + 70) / 60));
      emit('level', norm(db), voiced, norm(threshold));
      if (mic.state === 'speech') {
        segment.push(samples); segmentMs += ms;
        if (voiced) { quietMs = 0; speechMs += ms; } else quietMs += ms;
        if (quietMs >= o.silenceMs) finish(false);
        else if (segmentMs >= o.maxSpeechMs) finish(true);
        return;
      }
      // listening: keep a pre-roll ring so the first syllable isn't lost
      preroll.push(samples); prerollMs += ms;
      while (prerollMs > o.prerollMs && preroll.length > 1) prerollMs -= (preroll.shift().length / rate) * 1000;
      voicedRun = voiced ? voicedRun + 1 : 0;
      if (voicedRun >= 2) {
        segment = preroll.slice(); segmentMs = prerollMs; speechMs = ms * voicedRun; quietMs = 0;
        preroll.length = 0; prerollMs = 0;
        setState('speech');
      }
    }

    let startToken = 0;               // invalidated by stop(): a permission prompt may outlive it
    /** Resolves true when listening, false when cancelled by stop() meanwhile
     *  (or already running / starting); rejects when the microphone is denied. */
    mic.start = async function start() {
      if (stream) return true;
      if (mic.starting) return false;
      const ctx = ensureAudioContext();
      rate = ctx.sampleRate;
      const token = ++startToken;
      mic.starting = true;
      let granted;
      try {
        granted = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      } finally { mic.starting = false; }
      if (token !== startToken) {     // stopped (tab left, Stop pressed) while the prompt was open
        granted.getTracks().forEach((t) => t.stop());
        return false;
      }
      stream = granted;
      srcNode = ctx.createMediaStreamSource(stream);
      let usedWorklet = false;
      if (ctx.audioWorklet) {
        try {
          if (!workletReady) workletReady = ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'text/javascript' })));
          await workletReady;
          node = new AudioWorkletNode(ctx, 'pg-capture', { numberOfInputs: 1, numberOfOutputs: 0 });
          node.port.onmessage = (e) => frame(e.data);
          srcNode.connect(node);
          usedWorklet = true;
        } catch (e) { workletReady = null; node = null; }
      }
      if (!usedWorklet) {
        node = ctx.createScriptProcessor(1024, 1, 1);
        node.onaudioprocess = (e) => frame(new Float32Array(e.inputBuffer.getChannelData(0)));
        srcNode.connect(node); node.connect(ctx.destination);   // Chrome needs the sink for the callback to run
      }
      reset(); floor = -60; mic.paused = false;
      setState('listening');
      return true;
    };
    mic.stop = function stop() {
      startToken += 1;                // cancels a start() still waiting on the prompt
      if (!stream) return;
      try { srcNode.disconnect(); } catch (e) { /* ignore */ }
      try { node.disconnect(); } catch (e) { /* ignore */ }
      if (node && node.port) node.port.onmessage = null;
      stream.getTracks().forEach((t) => t.stop());
      stream = null; node = null; srcNode = null;
      reset();
      setState('idle');
      emit('level', 0, false, 0);
    };
    mic.pause = function pause() { mic.paused = true; reset(); if (mic.state !== 'idle') setState('paused'); };
    mic.resume = function resume() { if (!stream) return; mic.paused = false; reset(); floor = Math.max(floor, -60); setState('listening'); };
    return mic;
  }

  /** Level meter + threshold marker for a VAD mic. */
  function bindMeter(mic, fill, marker) {
    const bar = fill.parentElement;
    mic.on('level', (level, voiced, thr) => {
      fill.style.width = `${(level * 100).toFixed(1)}%`;
      marker.style.left = `${(thr * 100).toFixed(1)}%`;
      bar.classList.toggle('armed', thr > 0);
      bar.classList.toggle('voiced', voiced);
    });
  }

  /** POST one clip to /v1/audio/transcriptions (verbose_json) → {text, data, headers, secs}. */
  async function transcribeBlob(blob, name, { language, model, signal } = {}) {
    const fd = new FormData();
    fd.append('file', blob, name);
    if (language && language !== 'auto') fd.append('language', language);
    if (model) fd.append('model', model);
    fd.append('response_format', 'verbose_json');
    const t0 = performance.now();
    const res = await fetch(`${API}/v1/audio/transcriptions`, { method: 'POST', body: fd, signal });
    const raw = await res.text();
    let data = {};
    try { data = JSON.parse(raw); } catch (e) { /* not JSON */ }
    if (!res.ok) throw new Error(data.error ? `${data.error}${data.param ? ` (param: ${data.param})` : ''}` : `HTTP ${res.status}`);
    return { text: (data.text || '').trim(), data, headers: res.headers, secs: (performance.now() - t0) / 1000 };
  }

  /** POST /v1/audio/speech → {blob, headers, secs}. */
  async function speakText(body, signal) {
    const t0 = performance.now();
    const res = await fetch(`${API}/v1/audio/speech`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const parts = [err.error || `HTTP ${res.status}`];
      if (err.param) parts.push(`(param: ${err.param})`);
      if (err.reason) parts.push(`(${err.reason}${err.engine ? ', ' + err.engine : ''})`);
      throw new Error(parts.join(' '));
    }
    return { blob: await res.blob(), headers: res.headers, secs: (performance.now() - t0) / 1000 };
  }

  // =====================================================================
  // Voices listing (shared by Speech and Echo)
  // =====================================================================
  let listing = null;
  const listingWaiters = [];
  function engineEntry(key) { return (listing && listing.engines || []).find((e) => e.key === key) || null; }

  function fillEngineSelect(select, wanted) {
    select.innerHTML = '';
    select.appendChild(option('default', 'default · router picks'));
    (listing.engines || []).forEach((e) => select.appendChild(option(e.key, `${e.label || e.key}${e.loaded ? '' : ' · loads on first use'}`)));
    const w = wanted || listing.default_engine || 'default';
    select.value = [...select.options].some((o) => o.value === w) ? w : 'default';
  }
  // Supertonic voices are per request. The Piper engines speak with the voice
  // loaded for the language (chosen in the Studio's voice settings) and the
  // speech route refuses any other, so those are listed but not selectable.
  function fillVoiceSelect(select, engineKey, wanted) {
    const entry = engineEntry(engineKey);
    const perRequest = engineKey === 'supertonic';
    select.innerHTML = '';
    select.appendChild(option('default', entry ? (perRequest ? 'engine default' : 'voice loaded for the language') : 'router choice'));
    (entry && entry.voices || []).forEach((v) => {
      const bits = [v.label || v.id];
      if (v.language) bits.push(v.language);
      if (v.default) bits.push('current');
      else if (!perRequest) bits.push('select in Studio settings');
      else if (v.installed === false) bits.push('downloads on select');
      const o = option(v.id, bits.join(' · '));
      if (!perRequest && !v.default) o.disabled = true;
      select.appendChild(o);
    });
    if (wanted && [...select.options].some((o) => o.value === wanted && !o.disabled)) select.value = wanted;
  }
  function engineLanguages(engineKey) {
    const entry = engineEntry(engineKey);
    return (entry && entry.languages && entry.languages.length) ? entry.languages : (listing && listing.languages && listing.languages.length ? listing.languages : ['en']);
  }

  async function loadVoices() {
    const pill = $('conn');
    try {
      const res = await fetch(`${API}/v1/audio/voices`, { headers: { Accept: 'application/json' } });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      listing = data;
      const n = listing.engines.length;
      pill.className = 'pg-pill ok';
      pill.lastElementChild.textContent = `${n} engine${n === 1 ? '' : 's'} · default ${listing.default_engine || 'router'}`;
    } catch (err) {
      listing = { engines: [], languages: ['en'], default_engine: null, error: err.message };
      pill.className = 'pg-pill err';
      pill.lastElementChild.textContent = 'voices unavailable';
    }
    listingWaiters.splice(0).forEach((fn) => fn());
  }

  // =====================================================================
  // Speech
  // =====================================================================
  const sp = {
    input: $('sp-input'), engine: $('sp-engine'), voice: $('sp-voice'), language: $('sp-language'),
    speed: $('sp-speed'), speedOut: $('sp-speedOut'), run: $('sp-run'), stop: $('sp-stop'), status: $('sp-status'),
    size: $('sp-size'), req: $('sp-req'), curl: $('sp-curl'), headers: $('sp-headers'), voicesJson: $('sp-voicesJson'),
    download: $('sp-download'),
  };
  const spPlayer = makePlayer({ waveEl: $('sp-wave'), canvas: $('sp-canvas'), button: $('sp-play'), audioEl: $('sp-audio') });
  let spController = null;

  function fillSpeechLanguages(wanted) {
    const languages = engineLanguages(sp.engine.value);
    const w = wanted || sp.language.value;
    sp.language.innerHTML = '';
    languages.forEach((c) => sp.language.appendChild(option(c, langLabel(c))));
    sp.language.value = languages.includes(w) ? w : (languages.includes('en') ? 'en' : languages[0]);
  }
  function speechBody() {
    const body = { input: sp.input.value, model: sp.engine.value, language: sp.language.value,
                   speed: Number(sp.speed.value), response_format: 'wav' };
    if (sp.voice.value && sp.voice.value !== 'default') body.voice = sp.voice.value;
    return body;
  }
  function updateSpeechPreview() {
    const body = speechBody();
    sp.req.textContent = JSON.stringify(body, null, 2);
    const json = JSON.stringify(body).replace(/'/g, "'\\''");
    sp.curl.textContent = `curl -k -X POST ${API}/v1/audio/speech \\\n  -H 'Content-Type: application/json' \\\n  -d '${json}' -o speech.wav -D -`;
    savePrefs({ engine: sp.engine.value, voice: sp.voice.value, language: sp.language.value, speed: sp.speed.value, input: sp.input.value });
  }
  function initSpeechPickers() {
    const prefs = loadPrefs();
    sp.voicesJson.textContent = JSON.stringify(listing, null, 2);
    if (listing.error) setStatus(sp.status, `Could not list voices: ${listing.error}`, 'err');
    fillEngineSelect(sp.engine, prefs.engine);
    fillVoiceSelect(sp.voice, sp.engine.value, prefs.voice);
    fillSpeechLanguages(prefs.language);
    if (prefs.speed) sp.speed.value = prefs.speed;
    if (prefs.input) sp.input.value = prefs.input;
    sp.speedOut.value = `${Number(sp.speed.value).toFixed(2)}×`;
    updateSpeechPreview();
  }

  async function synthesize() {
    if (spController) return;                 // one request at a time (Run is disabled; shortcut checks too)
    const body = speechBody();
    if (!body.input.trim()) { setStatus(sp.status, 'Enter some text first.', 'err'); return; }
    ensureAudioContext();                     // created on the click: playback is allowed from here on
    updateSpeechPreview();
    spController = new AbortController();
    sp.run.disabled = true; sp.stop.disabled = false; sp.download.hidden = true;
    setStatus(sp.status, 'Synthesizing…', '', true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/v1/audio/speech`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: spController.signal,
      });
      const lines = [];
      res.headers.forEach((v, k) => { if (/^x-|^content-/i.test(k)) lines.push(`${k}: ${v}`); });
      sp.headers.textContent = lines.join('\n') || '(none)';
      const h = (n) => res.headers.get(n) || '–';
      $('st-engine').textContent = h('X-Engine'); $('st-voice').textContent = h('X-Voice'); $('st-speed').textContent = h('X-Speed');
      $('st-dur').textContent = fmt(Number(res.headers.get('X-Audio-Duration')), 2);
      $('st-gen').textContent = fmt(Number(res.headers.get('X-Elapsed-Time')), 2);
      $('st-rtf').textContent = fmt(Number(res.headers.get('X-RTF')), 3);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const parts = [err.error || `HTTP ${res.status}`];
        if (err.param) parts.push(`(param: ${err.param})`);
        if (err.reason) parts.push(`(${err.reason}${err.engine ? ', ' + err.engine : ''})`);
        throw new Error(parts.join(' '));
      }
      const blob = await res.blob();
      sp.size.textContent = `${(blob.size / 1024).toFixed(0)} KiB · ${((performance.now() - t0) / 1000).toFixed(2)} s round trip`;
      sp.download.href = URL.createObjectURL(blob);
      sp.download.hidden = false;
      setStatus(sp.status, 'Playing', 'ok');
      await spPlayer.load(blob, { autoplay: true });
      setStatus(sp.status, 'Done', 'ok');
    } catch (err) {
      if (err.name === 'AbortError') setStatus(sp.status, 'Cancelled.');
      else setStatus(sp.status, err.message, 'err');
    } finally {
      spController = null;
      sp.run.disabled = false; sp.stop.disabled = true;
    }
  }

  sp.engine.addEventListener('change', () => { fillVoiceSelect(sp.voice, sp.engine.value); fillSpeechLanguages(); updateSpeechPreview(); });
  [sp.voice, sp.language].forEach((e) => e.addEventListener('change', updateSpeechPreview));
  sp.input.addEventListener('input', updateSpeechPreview);
  sp.speed.addEventListener('input', () => { sp.speedOut.value = `${Number(sp.speed.value).toFixed(2)}×`; updateSpeechPreview(); });
  sp.run.addEventListener('click', synthesize);
  sp.stop.addEventListener('click', () => { if (spController) spController.abort(); spPlayer.stop(); });
  sp.input.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !spController) synthesize(); });

  // =====================================================================
  // Transcription: clip mode
  // =====================================================================
  const tr = {
    rec: $('tr-rec'), recStop: $('tr-recStop'), recTime: $('tr-recTime'), level: $('tr-level'), drop: $('tr-drop'),
    file: $('tr-file'), clipInfo: $('tr-clipInfo'), language: $('tr-language'),
    format: $('tr-format'), model: $('tr-model'), run: $('tr-run'), cancel: $('tr-cancel'), status: $('tr-status'),
    result: $('tr-result'), meta: $('tr-meta'), elapsed: $('tr-elapsed'), curl: $('tr-curl'), raw: $('tr-raw'),
  };
  const trPlayer = makePlayer({ waveEl: $('tr-wave'), canvas: $('tr-canvas'), button: $('tr-play'), audioEl: $('tr-audio') });
  let clip = null, recorder = null, stream = null, chunks = [], timer = 0, meterRaf = 0, trController = null;

  ASR_LANGUAGES.forEach((c) => { tr.language.appendChild(option(c, langLabel(c))); $('tl-language').appendChild(option(c, langLabel(c))); $('ec-asrLanguage').appendChild(option(c, langLabel(c))); });

  function extensionFor(type) {
    const base = (type || '').split(';')[0];
    return { 'audio/webm': 'webm', 'video/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/x-m4a': 'm4a',
             'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/wave': 'wav', 'audio/mpeg': 'mp3', 'audio/flac': 'flac' }[base] || 'bin';
  }
  async function setClip(blob, name, type) {
    clip = { blob, name, type: type || blob.type || 'application/octet-stream' };
    tr.clipInfo.textContent = `${name} · ${clip.type} · ${(blob.size / 1024).toFixed(0)} KiB`;
    tr.run.disabled = false;
    updateTrPreview();
    await trPlayer.load(blob, { autoplay: false });
  }

  function meterLoop(analyser, data) {
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sum += v * v; }
    const rms = Math.sqrt(sum / data.length);
    tr.level.style.width = `${Math.min(100, rms * 260)}%`;
    meterRaf = requestAnimationFrame(() => meterLoop(analyser, data));
  }

  let recToken = 0, recStarting = false;   // a pending permission prompt is cancelled by stopRecording()
  async function startRecording() {
    if (recStarting || (recorder && recorder.state !== 'inactive')) return;
    const token = ++recToken;
    recStarting = true;
    let granted;
    try { granted = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (err) { setStatus(tr.status, `Microphone unavailable: ${err.message}`, 'err'); return; }
    finally { recStarting = false; }
    if (token !== recToken) { granted.getTracks().forEach((t) => t.stop()); return; }   // mode/tab left meanwhile
    stream = granted;
    const mime = (window.MediaRecorder && MediaRecorder.isTypeSupported) ? MIME_PREFERENCE.find((m) => MediaRecorder.isTypeSupported(m)) : '';
    try { recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream); }
    catch (err) { setStatus(tr.status, `Recording not supported here: ${err.message}`, 'err'); stream.getTracks().forEach((t) => t.stop()); return; }
    const ctx = ensureAudioContext();
    const analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    meterLoop(analyser, new Uint8Array(analyser.fftSize));
    chunks = [];
    recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    recorder.onstop = async () => {
      const type = recorder.mimeType || mime || 'audio/webm';
      const blob = new Blob(chunks, { type });
      stream.getTracks().forEach((t) => t.stop()); stream = null;
      clearInterval(timer); cancelAnimationFrame(meterRaf); tr.level.style.width = '0';
      tr.rec.classList.remove('recording'); tr.rec.querySelector('span').textContent = 'Record';
      tr.rec.disabled = false; tr.recStop.disabled = true;
      if (!blob.size) { setStatus(tr.status, 'Nothing was recorded.', 'err'); return; }
      await setClip(blob, `recording.${extensionFor(type)}`, type);
      setStatus(tr.status, 'Clip ready. Press Transcribe.', 'ok');
    };
    recorder.start();
    const started = Date.now();
    tr.recTime.textContent = '0.0 s';
    timer = setInterval(() => { tr.recTime.textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`; }, 200);
    tr.rec.classList.add('recording'); tr.rec.querySelector('span').textContent = 'Recording';
    tr.rec.disabled = true; tr.recStop.disabled = false;
    setStatus(tr.status, 'Recording… press Stop when done.');
  }
  function stopRecording() { recToken += 1; if (recorder && recorder.state !== 'inactive') recorder.stop(); }

  function trFormData() {
    const fd = new FormData();
    fd.append('file', clip.blob, clip.name);
    if (tr.language.value !== 'auto') fd.append('language', tr.language.value);
    if (tr.model.value.trim()) fd.append('model', tr.model.value.trim());
    fd.append('response_format', tr.format.value);
    return fd;
  }
  function updateTrPreview() {
    const parts = [`curl -k -X POST ${API}/v1/audio/transcriptions \\`, `  -F 'file=@${clip ? clip.name : 'recording.wav'}' \\`];
    if (tr.language.value !== 'auto') parts.push(`  -F 'language=${tr.language.value}' \\`);
    if (tr.model.value.trim()) parts.push(`  -F 'model=${tr.model.value.trim()}' \\`);
    parts.push(`  -F 'response_format=${tr.format.value}' -D -`);
    tr.curl.textContent = parts.join('\n');
    savePrefs({ trLanguage: tr.language.value, trFormat: tr.format.value });
  }

  async function transcribe() {
    if (!clip) { setStatus(tr.status, 'Record or choose a clip first.', 'err'); return; }
    trController = new AbortController();
    tr.run.disabled = true; tr.cancel.disabled = false; tr.meta.hidden = true;
    tr.result.classList.add('empty'); tr.result.textContent = 'Transcribing…';
    setStatus(tr.status, 'Transcribing…', '', true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/v1/audio/transcriptions`, { method: 'POST', body: trFormData(), signal: trController.signal });
      const type = res.headers.get('Content-Type') || '';
      const rawText = await res.text();
      tr.raw.textContent = rawText;
      if (!res.ok) {
        let message = `HTTP ${res.status}`;
        try { const err = JSON.parse(rawText); message = err.error || message; if (err.param) message += ` (param: ${err.param})`; } catch (e) { /* not JSON */ }
        throw new Error(message);
      }
      tr.elapsed.textContent = `${((performance.now() - t0) / 1000).toFixed(2)} s round trip · server ${fmt(Number(res.headers.get('X-Elapsed-Time')), 2)} s`;
      $('m-model').textContent = res.headers.get('X-ASR-Model') || '–';
      let text = '';
      if (type.startsWith('text/plain')) {
        text = rawText.trim();
        ['m-lang', 'm-nospeech', 'm-logprob', 'm-ignored'].forEach((id) => { $(id).textContent = '–'; });
      } else {
        const data = JSON.parse(rawText);
        text = data.text || '';
        $('m-lang').textContent = data.language ? `${data.language}${data.language_detected ? ' (detected)' : ''}` : '–';
        $('m-nospeech').textContent = fmt(data.no_speech_prob, 3);
        $('m-logprob').textContent = fmt(data.avg_logprob, 3);
        const verdict = $('m-ignored'); verdict.textContent = '';
        if (data.ignored == null) verdict.textContent = '–';
        else verdict.appendChild(el('span', data.ignored ? 'badge warn' : 'badge ok', data.ignored ? `would be ignored · ${data.reason || ''}` : 'accepted'));
      }
      tr.result.classList.toggle('empty', !text);
      tr.result.textContent = text || '(empty transcript)';
      tr.meta.hidden = false;
      setStatus(tr.status, 'Done', 'ok');
    } catch (err) {
      tr.result.classList.add('empty'); tr.result.textContent = 'The transcript appears here.';
      if (err.name === 'AbortError') setStatus(tr.status, 'Cancelled.');
      else setStatus(tr.status, err.message, 'err');
    } finally {
      trController = null;
      tr.run.disabled = !clip; tr.cancel.disabled = true;
    }
  }

  tr.rec.addEventListener('click', startRecording);
  tr.recStop.addEventListener('click', stopRecording);
  tr.file.addEventListener('change', () => { const f = tr.file.files && tr.file.files[0]; if (f) { setClip(f, f.name, f.type); setStatus(tr.status, 'File selected.'); } });
  ['dragenter', 'dragover'].forEach((ev) => tr.drop.addEventListener(ev, (e) => { e.preventDefault(); tr.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => tr.drop.addEventListener(ev, (e) => { e.preventDefault(); tr.drop.classList.remove('over'); }));
  tr.drop.addEventListener('drop', (e) => { const f = e.dataTransfer.files && e.dataTransfer.files[0]; if (f) { setClip(f, f.name, f.type); setStatus(tr.status, 'File dropped.'); } });
  [tr.language, tr.format].forEach((e) => e.addEventListener('change', updateTrPreview));
  tr.model.addEventListener('input', updateTrPreview);
  tr.run.addEventListener('click', transcribe);
  tr.cancel.addEventListener('click', () => { if (trController) trController.abort(); });

  // =====================================================================
  // Transcription: live mode
  // =====================================================================
  const tl = {
    start: $('tl-start'), stop: $('tl-stop'), state: $('tl-state'), level: $('tl-level'), threshold: $('tl-threshold'),
    language: $('tl-language'), model: $('tl-model'), sens: $('tl-sens'), silence: $('tl-silence'),
    hideIgnored: $('tl-hideIgnored'), status: $('tl-status'), log: $('tl-log'), copy: $('tl-copy'), clear: $('tl-clear'),
  };
  const liveMic = createLiveMic({ silenceMs: 700, sensitivity: 0.5 });
  bindMeter(liveMic, tl.level, tl.threshold);
  const liveStats = { count: 0, words: 0, speech: 0, latencies: [], rtfs: [] };
  let liveQueue = Promise.resolve();
  let liveController = null;

  function setChip(chipEl, cls, text) { chipEl.className = `state-chip ${cls}`; chipEl.lastElementChild.textContent = text; }
  liveMic.on('state', (s) => {
    const map = { idle: ['idle', 'idle'], listening: ['listening', 'listening'], speech: ['voice', 'hearing you'], paused: ['busy', 'paused'] };
    const [cls, text] = map[s] || ['idle', s];
    setChip(tl.state, cls, text);
  });
  function trMode(mode) {
    document.querySelectorAll('.seg-btn[data-trmode]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.trmode === mode)));
    $('tr-clipMode').hidden = mode !== 'clip'; $('tr-clipOut').hidden = mode !== 'clip';
    $('tr-liveMode').hidden = mode !== 'live'; $('tr-liveOut').hidden = mode !== 'live';
    if (mode !== 'live') stopLive();
    if (mode !== 'clip') { stopRecording(); trPlayer.stop(); }
    savePrefs({ trMode: mode });
  }
  document.querySelectorAll('.seg-btn[data-trmode]').forEach((b) => b.addEventListener('click', () => trMode(b.dataset.trmode)));

  function updateLiveStats() {
    $('tl-count').textContent = String(liveStats.count);
    $('tl-words').textContent = String(liveStats.words);
    $('tl-speech').textContent = liveStats.speech.toFixed(1);
    $('tl-latency').textContent = fmt(avg(liveStats.latencies), 2);
    $('tl-rtf').textContent = fmt(avg(liveStats.rtfs), 2);
  }
  function addSegmentRow(dur) {
    const ph = tl.log.querySelector('.placeholder'); if (ph) ph.remove();
    const row = el('div', 'seg-item pending');
    row.appendChild(el('span', 't', clock()));
    row.appendChild(el('span', 'x', `transcribing ${dur.toFixed(1)} s of speech…`));
    const meta = el('span', 'm'); row.appendChild(meta);
    tl.log.appendChild(row);
    tl.log.scrollTop = tl.log.scrollHeight;
    return row;
  }
  liveMic.on('segment', (blob, dur, forced) => {
    const session = liveController;          // the capture session this segment belongs to
    if (!session) return;                    // stopped between cut and delivery
    const row = addSegmentRow(dur);
    liveQueue = liveQueue.then(async () => {
      if (session.signal.aborted) { row.remove(); return; }   // stopped while queued
      try {
        const r = await transcribeBlob(blob, 'utterance.wav', { language: tl.language.value, model: tl.model.value.trim(), signal: session.signal });
        const d = r.data;
        row.classList.remove('pending');
        row.querySelector('.x').textContent = r.text || '(no words)';
        const bits = [`${dur.toFixed(1)} s`, `${r.secs.toFixed(2)} s`, `RTF ${(r.secs / Math.max(dur, 0.1)).toFixed(2)}`];
        if (d.language) bits.push(d.language + (d.language_detected ? '*' : ''));
        if (forced) bits.push('cut at max length');
        if (d.ignored) { bits.push(`ignored: ${d.reason || 'filtered'}`); row.classList.add('ignored'); if (tl.hideIgnored.checked) row.hidden = true; }
        row.querySelector('.m').textContent = bits.join(' · ');
        if (!d.ignored && r.text) {
          liveStats.count += 1; liveStats.words += r.text.split(/\s+/).filter(Boolean).length;
          liveStats.speech += dur; liveStats.latencies.push(r.secs); liveStats.rtfs.push(r.secs / Math.max(dur, 0.1));
          updateLiveStats();
        }
      } catch (err) {
        row.classList.remove('pending');
        if (err.name === 'AbortError') { row.remove(); return; }
        row.classList.add('err'); row.querySelector('.x').textContent = err.message;
        setStatus(tl.status, err.message, 'err');
      }
      tl.log.scrollTop = tl.log.scrollHeight;
    });
  });
  async function startLive() {
    if (liveController) return;              // already listening or waiting on the prompt
    ensureAudioContext();
    liveMic.set({ sensitivity: Number(tl.sens.value), silenceMs: Number(tl.silence.value) });
    setStatus(tl.status, '');
    const session = new AbortController();   // before the mic runs: segments bind to it
    liveController = session;
    tl.start.disabled = true; tl.start.querySelector('span').textContent = 'Starting…';   // no second click meanwhile
    tl.stop.disabled = false;
    let started = false;
    try { started = await liveMic.start(); }
    catch (err) {
      if (liveController === session) { liveController = null; resetLiveControls(); }
      setStatus(tl.status, `Microphone unavailable: ${err.message}`, 'err'); setChip(tl.state, 'err', 'no microphone');
      return;
    }
    if (!started || liveController !== session) {   // stopped while the permission prompt was open
      if (liveController === session) { liveController = null; resetLiveControls(); }
      return;
    }
    tl.start.classList.add('recording'); tl.start.querySelector('span').textContent = 'Listening';
  }
  function resetLiveControls() {
    tl.start.disabled = false; tl.start.classList.remove('recording'); tl.start.querySelector('span').textContent = 'Start listening';
    tl.stop.disabled = true;
  }
  function stopLive() {
    liveMic.stop();
    if (liveController) { liveController.abort(); liveController = null; }
    resetLiveControls();
  }
  tl.start.addEventListener('click', startLive);
  tl.stop.addEventListener('click', stopLive);
  bindSlider(tl.sens, $('tl-sensOut'), (v) => v.toFixed(2));
  bindSlider(tl.silence, $('tl-silenceOut'), (v) => `${v} ms`);
  tl.sens.addEventListener('input', () => { liveMic.set({ sensitivity: Number(tl.sens.value) }); savePrefs({ tlSens: tl.sens.value }); });
  tl.silence.addEventListener('input', () => { liveMic.set({ silenceMs: Number(tl.silence.value) }); savePrefs({ tlSilence: tl.silence.value }); });
  tl.language.addEventListener('change', () => savePrefs({ tlLanguage: tl.language.value }));
  tl.hideIgnored.addEventListener('change', () => { tl.log.querySelectorAll('.seg-item.ignored').forEach((r) => { r.hidden = tl.hideIgnored.checked; }); savePrefs({ tlHide: tl.hideIgnored.checked }); });
  tl.clear.addEventListener('click', () => {
    tl.log.innerHTML = '<div class="placeholder">Start listening and speak. Each utterance appears here as soon as it is transcribed.</div>';
    Object.assign(liveStats, { count: 0, words: 0, speech: 0, latencies: [], rtfs: [] }); updateLiveStats();
  });
  tl.copy.addEventListener('click', async () => {
    const text = [...tl.log.querySelectorAll('.seg-item:not(.ignored):not(.pending):not(.err) .x')].map((n) => n.textContent).join('\n');
    try { await navigator.clipboard.writeText(text); tl.copy.textContent = 'Copied'; } catch (e) { tl.copy.textContent = 'Select & copy'; }
    setTimeout(() => { tl.copy.textContent = 'Copy all'; }, 1500);
  });

  // =====================================================================
  // Echo: speak → transcribe → speak back
  // =====================================================================
  const ec = {
    start: $('ec-start'), state: $('ec-state'), level: $('ec-level'), threshold: $('ec-threshold'),
    engine: $('ec-engine'), voice: $('ec-voice'), asrLanguage: $('ec-asrLanguage'), ttsLanguage: $('ec-ttsLanguage'),
    speed: $('ec-speed'), silence: $('ec-silence'), sens: $('ec-sens'), status: $('ec-status'),
    turns: $('ec-turns'), clear: $('ec-clear'),
  };
  const ecPlayer = makePlayer({ waveEl: $('ec-wave'), canvas: $('ec-canvas'), button: $('ec-play'), audioEl: $('ec-audio') });
  const echoMic = createLiveMic({ silenceMs: 600, sensitivity: 0.5, maxSpeechMs: 15000 });
  bindMeter(echoMic, ec.level, ec.threshold);
  const echoStats = { count: 0, asr: [], tts: [], turn: [] };
  let echoOn = false, echoBusy = false, echoController = null;

  function setEchoState(cls, text) {
    ec.state.className = `echo-state ${cls}`; ec.state.textContent = text;
    ec.start.className = `echo-btn ${echoOn ? 'on ' : ''}${cls}`;
  }
  echoMic.on('state', (s) => {
    if (!echoOn || echoBusy) return;
    if (s === 'listening') setEchoState('listening', 'Listening… say something.');
    else if (s === 'speech') setEchoState('voice', 'Hearing you…');
  });
  function fillEchoTtsLanguages() {
    const wanted = ec.ttsLanguage.value || 'auto';
    ec.ttsLanguage.innerHTML = '';
    ec.ttsLanguage.appendChild(option('auto', 'match what was heard'));
    engineLanguages(ec.engine.value).forEach((c) => ec.ttsLanguage.appendChild(option(c, langLabel(c))));
    ec.ttsLanguage.value = [...ec.ttsLanguage.options].some((o) => o.value === wanted) ? wanted : 'auto';
  }
  function initEchoPickers() {
    const prefs = loadPrefs();
    fillEngineSelect(ec.engine, prefs.ecEngine);
    fillVoiceSelect(ec.voice, ec.engine.value, prefs.ecVoice);
    if (prefs.ecTtsLanguage) ec.ttsLanguage.value = prefs.ecTtsLanguage;
    fillEchoTtsLanguages();
    if (prefs.ecTtsLanguage && [...ec.ttsLanguage.options].some((o) => o.value === prefs.ecTtsLanguage)) ec.ttsLanguage.value = prefs.ecTtsLanguage;
  }
  function saveEchoPrefs() {
    savePrefs({ ecEngine: ec.engine.value, ecVoice: ec.voice.value, ecTtsLanguage: ec.ttsLanguage.value, ecAsrLanguage: ec.asrLanguage.value,
                ecSpeed: ec.speed.value, ecSilence: ec.silence.value, ecSens: ec.sens.value });
  }
  function updateEchoStats() {
    $('ec-count').textContent = String(echoStats.count);
    $('ec-asr').textContent = fmt(avg(echoStats.asr), 2);
    $('ec-tts').textContent = fmt(avg(echoStats.tts), 2);
    $('ec-turn').textContent = fmt(avg(echoStats.turn), 2);
  }
  function bubble(kind, who, text) {
    const ph = ec.turns.querySelector('.placeholder'); if (ph) ph.remove();
    const m = el('div', `message ${kind}`);
    m.appendChild(el('span', 'who', who));
    m.appendChild(el('span', 'txt', text));
    m.appendChild(el('span', 'meta'));
    ec.turns.appendChild(m);
    ec.turns.scrollTop = ec.turns.scrollHeight;
    return m;
  }
  const setBubble = (m, text, meta, cls) => {
    m.querySelector('.txt').textContent = text;
    m.querySelector('.meta').textContent = meta || '';
    m.classList.remove('pending'); if (cls) m.classList.add(cls);
    ec.turns.scrollTop = ec.turns.scrollHeight;
  };

  echoMic.on('segment', async (blob, dur) => {
    if (!echoOn || echoBusy) return;
    echoBusy = true;
    echoMic.pause();
    echoController = new AbortController();
    const tTurn = performance.now();
    const you = bubble('user pending', 'You said', `transcribing ${dur.toFixed(1)} s…`);
    try {
      setEchoState('busy', 'Transcribing…');
      const asr = await transcribeBlob(blob, 'utterance.wav', { language: ec.asrLanguage.value, signal: echoController.signal });
      const d = asr.data;
      if (d.ignored || !asr.text) {
        setBubble(you, asr.text || '(nothing recognisable)', `${dur.toFixed(1)} s · ${asr.secs.toFixed(2)} s · ${d.ignored ? `ignored: ${d.reason || 'filtered'}` : 'empty'}`, 'ignored');
        you.style.opacity = '.6';
        return;
      }
      setBubble(you, asr.text, `${dur.toFixed(1)} s · ASR ${asr.secs.toFixed(2)} s${d.language ? ` · ${d.language}${d.language_detected ? '*' : ''}` : ''}`);
      const back = bubble('assistant pending', 'Spoken back', 'synthesizing…');
      setEchoState('busy', 'Synthesizing the reply…');
      const langs = engineLanguages(ec.engine.value);
      let language = ec.ttsLanguage.value;
      if (language === 'auto') language = (d.tts_language && langs.includes(d.tts_language)) ? d.tts_language : (langs.includes(d.language) ? d.language : (langs.includes('en') ? 'en' : langs[0]));
      const body = { input: asr.text, model: ec.engine.value, language, speed: Number(ec.speed.value), response_format: 'wav' };
      if (ec.voice.value !== 'default') body.voice = ec.voice.value;
      const tts = await speakText(body, echoController.signal);
      const h = (n) => tts.headers.get(n) || '–';
      const audioSecs = Number(tts.headers.get('X-Audio-Duration'));
      setBubble(back, asr.text, `${h('X-Engine')} · ${h('X-Voice')} · ${h('X-Language')} · TTS ${tts.secs.toFixed(2)} s · ${fmt(audioSecs, 1)} s audio · RTF ${h('X-RTF')}`);
      echoStats.count += 1; echoStats.asr.push(asr.secs); echoStats.tts.push(tts.secs); echoStats.turn.push((performance.now() - tTurn) / 1000);
      updateEchoStats();
      setEchoState('speaking', 'Speaking back…');
      back.classList.add('speaking');
      await ecPlayer.load(tts.blob, { autoplay: true });
      back.classList.remove('speaking');
    } catch (err) {
      if (err.name !== 'AbortError') {
        const last = ec.turns.lastElementChild;
        if (last && last.classList.contains('pending')) setBubble(last, err.message, '', 'err');
        else bubble('assistant err', 'Error', err.message);
        setStatus(ec.status, err.message, 'err');
      }
    } finally {
      echoBusy = false; echoController = null;
      if (echoOn) { echoMic.resume(); setEchoState('listening', 'Listening… say something.'); }
    }
  });
  async function startEcho() {
    ensureAudioContext();
    echoMic.set({ sensitivity: Number(ec.sens.value), silenceMs: Number(ec.silence.value) });
    setStatus(ec.status, '');
    let started = false;
    try { started = await echoMic.start(); }
    catch (err) { setEchoState('err', `Microphone unavailable: ${err.message}`); return; }
    if (!started) return;                     // stopped while the permission prompt was open
    echoOn = true;
    setEchoState('listening', 'Listening… say something.');
  }
  function stopEcho() {
    echoOn = false;
    echoMic.stop();
    if (echoController) echoController.abort();
    ecPlayer.stop();
    setEchoState('idle', 'Press to start, then just talk. What you say is transcribed and spoken straight back.');
  }
  ec.start.addEventListener('click', () => { if (echoOn) stopEcho(); else startEcho(); });
  ec.engine.addEventListener('change', () => { fillVoiceSelect(ec.voice, ec.engine.value); fillEchoTtsLanguages(); saveEchoPrefs(); });
  [ec.voice, ec.ttsLanguage, ec.asrLanguage].forEach((e) => e.addEventListener('change', saveEchoPrefs));
  bindSlider(ec.speed, $('ec-speedOut'), (v) => `${v.toFixed(2)}×`);
  bindSlider(ec.silence, $('ec-silenceOut'), (v) => `${v} ms`);
  bindSlider(ec.sens, $('ec-sensOut'), (v) => v.toFixed(2));
  ec.speed.addEventListener('input', saveEchoPrefs);
  ec.silence.addEventListener('input', () => { echoMic.set({ silenceMs: Number(ec.silence.value) }); saveEchoPrefs(); });
  ec.sens.addEventListener('input', () => { echoMic.set({ sensitivity: Number(ec.sens.value) }); saveEchoPrefs(); });
  ec.clear.addEventListener('click', () => {
    ec.turns.innerHTML = '<div class="placeholder">Each turn shows what was heard and the reply that was spoken back, with timings.</div>';
    Object.assign(echoStats, { count: 0, asr: [], tts: [], turn: [] }); updateEchoStats();
  });

  // Leaving a tab stops whatever it was doing (playback, recording, listening).
  leaveTab = (name) => {
    if (name === 'speech') { if (spController) spController.abort(); spPlayer.stop(); }
    else if (name === 'transcription') { stopLive(); stopRecording(); trPlayer.stop(); }
    else if (name === 'echo') stopEcho();
  };

  // ---- boot --------------------------------------------------------------
  window.addEventListener('pagehide', () => {
    stopRecording(); if (stream) stream.getTracks().forEach((t) => t.stop());
    stopLive(); stopEcho(); spPlayer.stop(); trPlayer.stop();
  });
  const prefs = loadPrefs();
  if (prefs.trLanguage) tr.language.value = prefs.trLanguage;
  if (prefs.trFormat) tr.format.value = prefs.trFormat;
  if (prefs.tlLanguage) tl.language.value = prefs.tlLanguage;
  if (prefs.tlSens) tl.sens.value = prefs.tlSens;
  if (prefs.tlSilence) tl.silence.value = prefs.tlSilence;
  if (prefs.tlHide) tl.hideIgnored.checked = true;
  if (prefs.ecAsrLanguage) ec.asrLanguage.value = prefs.ecAsrLanguage;
  if (prefs.ecSpeed) ec.speed.value = prefs.ecSpeed;
  if (prefs.ecSilence) ec.silence.value = prefs.ecSilence;
  if (prefs.ecSens) ec.sens.value = prefs.ecSens;
  [tl.sens, tl.silence, ec.speed, ec.silence, ec.sens].forEach((i) => i.dispatchEvent(new Event('input')));
  updateTrPreview();
  trMode(prefs.trMode === 'live' ? 'live' : 'clip');
  const initial = (location.hash || '').replace('#', '') || prefs.tab || 'speech';
  showTab(initial);
  listingWaiters.push(initSpeechPickers, initEchoPickers);
  loadVoices();
})();
