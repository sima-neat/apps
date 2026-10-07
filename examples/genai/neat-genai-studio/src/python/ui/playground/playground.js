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
  const TABS = ['speech', 'transcription', 'echo', 'translate'];

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
  let enterTab = () => {};
  function showTab(name) {
    if (!TABS.includes(name)) name = 'speech';
    if (currentTab && currentTab !== name) leaveTab(currentTab);
    currentTab = name;
    document.querySelectorAll('.pg-tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    document.querySelectorAll('.pg-panel').forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
    savePrefs({ tab: name });
    if (history.replaceState) history.replaceState(null, '', `#${name}`);
    enterTab(name);
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
  const sliderShows = [];            // refresh every slider's readout (boot, after restoring prefs)
  function bindSlider(input, output, format) {
    const show = () => { output.value = format(Number(input.value)); };
    input.addEventListener('input', show); show();
    sliderShows.push(show);
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
  /** onPlayback(true|false) is called whenever audible playback (Web Audio or
   *  the native controls) starts or stops, so listening modes can pause. */
  function makePlayer({ waveEl, canvas, button, audioEl, onPlayback }) {
    let buffer = null, source = null, startedAt = 0, raf = 0, resolveEnd = null, objectUrl = null;
    let nativeWaiters = [];          // play() promises handed over to the <audio> controls
    let loadGen = 0;                 // stop() bumps it: a load still decoding then goes nowhere
    const cursor = waveEl.querySelector('.cursor');
    const placeholder = waveEl.querySelector('.placeholder');
    const emptyText = placeholder.textContent;   // shown again after reset()
    const label = button.querySelector('span');

    let audible = false;
    const isAudible = () => !!source || (!audioEl.paused && !audioEl.ended);
    function notify() {              // report playback start/stop transitions once
      const now = isAudible();
      if (now !== audible) { audible = now; if (onPlayback) onPlayback(now); }
    }
    function stopSource() {          // Web Audio playback only
      if (source) { const s = source; source = null; try { s.stop(); } catch (e) { /* already stopped */ } }
      cancelAnimationFrame(raf);
      waveEl.classList.remove('playing');
      label.textContent = 'Play';
      if (resolveEnd) { const r = resolveEnd; resolveEnd = null; r(); }
      notify();
    }
    function settleNative() {        // the native playback a turn was waiting on ended
      const waiters = nativeWaiters; nativeWaiters = [];
      waiters.forEach((r) => r());
    }
    function stop() {                // everything, including a load still in progress
      loadGen += 1;
      stopSource();
      if (!audioEl.paused) { try { audioEl.pause(); } catch (e) { /* not playable */ } }
      settleNative();
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
      if (!audioEl.paused) { try { audioEl.pause(); } catch (e) { /* ignore */ } }   // one player at a time
      const ctx = ensureAudioContext();
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = stopSource;
      startedAt = ctx.currentTime;
      source.start();
      notify();
      waveEl.classList.add('playing');
      label.textContent = 'Stop';
      raf = requestAnimationFrame(tick);
      return new Promise((resolve) => { resolveEnd = resolve; });
    }
    /** Decode and show a clip; resolves when autoplay finishes. A stop() while
     *  decoding (tab left, Cancel) discards the result: nothing is shown or played. */
    function clear() {               // show nothing until the new clip is decoded
      buffer = null;
      drawWave(canvas, null);
      placeholder.hidden = false;
      placeholder.textContent = 'Decoding…';
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
      audioEl.removeAttribute('src');
      try { audioEl.load(); } catch (e) { /* ignore */ }
      audioEl.hidden = true;
      button.disabled = true;
    }
    async function load(blob, { autoplay } = {}) {
      stop();
      const gen = loadGen;
      clear();
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
    /** Back to empty: stop, drop the clip and its object URL, restore the placeholder. */
    function reset() {
      stop();
      clear();
      placeholder.textContent = emptyText;
      waveEl.classList.remove('playing');
    }
    button.addEventListener('click', () => { play(); });
    // Switching to the native controls mid-playback: stop the Web Audio copy, but
    // a turn awaiting play() (Echo, Translate keep the microphone paused until
    // then) keeps waiting until the native playback pauses or ends.
    audioEl.addEventListener('play', () => {
      if (!source) return;
      if (resolveEnd) { nativeWaiters.push(resolveEnd); resolveEnd = null; }
      stopSource();
    });
    ['pause', 'ended', 'emptied'].forEach((ev) => audioEl.addEventListener(ev, () => { settleNative(); notify(); }));
    audioEl.addEventListener('playing', notify);
    window.addEventListener('resize', () => { if (buffer) drawWave(canvas, buffer); });
    return { load, stop, play, reset, get buffer() { return buffer; }, get playing() { return isAudible(); } };
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
  // The capture processor is registered once per AudioContext; only a failed
  // addModule is retried (registering the same name twice would throw).
  function loadWorklet(ctx) {
    if (!workletReady) {
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'text/javascript' }));
      const ready = ctx.audioWorklet.addModule(url);
      ready.then(() => URL.revokeObjectURL(url), () => { URL.revokeObjectURL(url); if (workletReady === ready) workletReady = null; });
      workletReady = ready;
    }
    return workletReady;
  }

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

    let startToken = 0;               // stop() bumps it: every await in a start re-checks it
    let pending = null;               // the start in flight, if any
    const cancelled = (token) => token !== startToken;

    // Build the capture graph from a granted stream. Everything is held in locals
    // and only published (stream/srcNode/node) once no stop() happened during any
    // await; otherwise it is torn down here and the start reports false.
    async function startOnce(token) {
      const ctx = ensureAudioContext();
      const granted = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      let src = null, n = null;
      const teardown = () => {
        try { if (src) src.disconnect(); } catch (e) { /* ignore */ }
        try { if (n) n.disconnect(); } catch (e) { /* ignore */ }
        if (n && n.port) n.port.onmessage = null;
        if (n && 'onaudioprocess' in n) n.onaudioprocess = null;
        granted.getTracks().forEach((t) => t.stop());
      };
      if (cancelled(token)) { teardown(); return false; }        // stopped during the permission prompt
      try {
        rate = ctx.sampleRate;
        src = ctx.createMediaStreamSource(granted);
        let workletOk = false;
        if (ctx.audioWorklet) {
          try { await loadWorklet(ctx); workletOk = true; } catch (e) { workletOk = false; }
          if (cancelled(token)) { teardown(); return false; }    // stopped while the worklet loaded
        }
        if (workletOk) {
          try {
            n = new AudioWorkletNode(ctx, 'pg-capture', { numberOfInputs: 1, numberOfOutputs: 0 });
            n.port.onmessage = (e) => frame(e.data);
            src.connect(n);
          } catch (e) { n = null; }
        }
        if (!n) {
          n = ctx.createScriptProcessor(1024, 1, 1);
          n.onaudioprocess = (e) => frame(new Float32Array(e.inputBuffer.getChannelData(0)));
          src.connect(n); n.connect(ctx.destination);   // Chrome needs the sink for the callback to run
        }
      } catch (err) { teardown(); throw err; }
      stream = granted; srcNode = src; node = n;
      reset(); floor = -60; mic.paused = false;
      setState('listening');
      return true;
    }

    /** Resolves true once listening, false when stop() cancelled this start;
     *  rejects when the microphone is denied or capture cannot be set up. A
     *  start issued while a cancelled one is still settling waits for it and
     *  then proceeds, so a quick stop/start is never silently dropped. */
    mic.start = async function start() {
      if (stream) return true;
      if (pending) {
        await pending.catch(() => false);
        if (stream) return true;
      }
      const token = ++startToken;
      const p = startOnce(token);
      pending = p; mic.starting = true;
      try { return await p; } finally {
        if (pending === p) { pending = null; mic.starting = false; }
      }
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
  /** POST one clip to /v1/audio/transcriptions (or `endpoint: 'translations'`,
   *  Whisper's speech-to-English) as verbose_json → {text, data, headers, secs}. */
  async function transcribeBlob(blob, name, { language, model, signal, endpoint = 'transcriptions' } = {}) {
    const fd = new FormData();
    fd.append('file', blob, name);
    if (language && language !== 'auto') fd.append('language', language);
    if (model) fd.append('model', model);
    fd.append('response_format', 'verbose_json');
    const t0 = performance.now();
    const route = endpoint === 'translations' ? 'translations' : 'transcriptions';
    const res = await fetch(`${API}/v1/audio/${route}`, { method: 'POST', body: fd, signal });
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
  let spClearGen = 0;                // Clear bumps it: a synthesis it interrupted reports nothing

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
    if (!listing) return;          // pickers not filled yet: saving now would erase the stored choices
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

  let downloadUrl = null;
  function setDownload(blob) {
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
    downloadUrl = blob ? URL.createObjectURL(blob) : null;
    if (downloadUrl) sp.download.href = downloadUrl; else sp.download.removeAttribute('href');
    sp.download.hidden = !downloadUrl;
  }
  async function synthesize() {
    if (spController) return;                 // one request at a time (Run is disabled; shortcut checks too)
    const body = speechBody();
    if (!body.input.trim()) { setStatus(sp.status, 'Enter some text first.', 'err'); return; }
    ensureAudioContext();                     // created on the click: playback is allowed from here on
    updateSpeechPreview();
    const controller = new AbortController();
    spController = controller;
    const clearGen = spClearGen;
    const cleared = () => clearGen !== spClearGen;
    sp.run.disabled = true; sp.stop.disabled = false; sp.download.hidden = true;
    setStatus(sp.status, 'Synthesizing…', '', true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/v1/audio/speech`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal,
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
      setDownload(blob);
      setStatus(sp.status, 'Playing', 'ok');
      await spPlayer.load(blob, { autoplay: true });
      if (cleared()) return;                   // Clear emptied the output on purpose
      if (controller.signal.aborted) { setStatus(sp.status, 'Cancelled.'); return; }   // Cancel during playback
      if (spPlayer.buffer) setStatus(sp.status, 'Done', 'ok');
      else setStatus(sp.status, 'The browser could not decode the returned audio (the download link still has it).', 'err');
    } catch (err) {
      if (cleared()) return;                   // Clear's own abort: keep the cleared output empty
      if (err.name === 'AbortError') setStatus(sp.status, 'Cancelled.');
      else setStatus(sp.status, err.message, 'err');
    } finally {
      if (spController === controller) spController = null;
      sp.run.disabled = !!spController; sp.stop.disabled = !spController;
    }
  }

  sp.engine.addEventListener('change', () => { fillVoiceSelect(sp.voice, sp.engine.value); fillSpeechLanguages(); updateSpeechPreview(); });
  [sp.voice, sp.language].forEach((e) => e.addEventListener('change', updateSpeechPreview));
  sp.input.addEventListener('input', updateSpeechPreview);
  sp.speed.addEventListener('input', () => { sp.speedOut.value = `${Number(sp.speed.value).toFixed(2)}×`; updateSpeechPreview(); });
  sp.run.addEventListener('click', synthesize);
  $('sp-clear').addEventListener('click', () => {
    spClearGen += 1;                               // the interrupted synthesis reports nothing
    if (spController) spController.abort();        // a synthesis still running is dropped too
    spPlayer.reset();
    setDownload(null);
    ['st-engine', 'st-voice', 'st-speed', 'st-dur', 'st-gen', 'st-rtf'].forEach((id) => { $(id).textContent = '–'; });
    sp.size.textContent = '–';
    sp.headers.textContent = '';
    setStatus(sp.status, '');
  });
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
  let clip = null, trController = null;

  ASR_LANGUAGES.forEach((c) => { tr.language.appendChild(option(c, langLabel(c))); $('tl-language').appendChild(option(c, langLabel(c))); $('ec-asrLanguage').appendChild(option(c, langLabel(c))); });

  function extensionFor(type) {
    const base = (type || '').split(';')[0];
    return { 'audio/webm': 'webm', 'video/webm': 'webm', 'audio/ogg': 'ogg', 'audio/mp4': 'm4a', 'audio/x-m4a': 'm4a',
             'audio/wav': 'wav', 'audio/x-wav': 'wav', 'audio/wave': 'wav', 'audio/mpeg': 'mp3', 'audio/flac': 'flac' }[base] || 'bin';
  }
  async function setClip(blob, name, type) {
    // A result for the previous clip must never appear under the new one:
    // abort its request (its finally re-enables Transcribe).
    if (trController) trController.abort();
    clip = { blob, name, type: type || blob.type || 'application/octet-stream' };
    tr.clipInfo.textContent = `${name} · ${clip.type} · ${(blob.size / 1024).toFixed(0)} KiB`;
    tr.run.disabled = !!trController;          // one request at a time: Run returns when it ends
    tr.meta.hidden = true;
    tr.result.classList.add('empty'); tr.result.textContent = 'The transcript appears here.';
    updateTrPreview();
    await trPlayer.load(blob, { autoplay: false });
  }



  let recToken = 0, recStarting = false;   // a pending permission prompt is cancelled by stopRecording()
  // Each recording owns its stream, recorder, meter and timer (captured in the
  // closure), so a late onstop from a previous recording can never tear down a
  // newer one.
  let active = null;                 // { rec, cleanup } of the recording in progress
  async function startRecording() {
    if (recStarting || active) return;
    const token = ++recToken;
    recStarting = true;
    let granted;
    try { granted = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (err) {
      if (token !== recToken) return;        // superseded: leave the UI to the current attempt
      setStatus(tr.status, `Microphone unavailable: ${err.message}`, 'err');
      return;
    }
    finally { if (token === recToken) recStarting = false; }
    if (token !== recToken) { granted.getTracks().forEach((t) => t.stop()); return; }   // mode/tab left meanwhile
    const mime = (window.MediaRecorder && MediaRecorder.isTypeSupported) ? MIME_PREFERENCE.find((m) => MediaRecorder.isTypeSupported(m)) : '';
    let rec;
    try { rec = mime ? new MediaRecorder(granted, { mimeType: mime }) : new MediaRecorder(granted); }
    catch (err) { setStatus(tr.status, `Recording not supported here: ${err.message}`, 'err'); granted.getTracks().forEach((t) => t.stop()); return; }
    // Level meter. Everything after the permission grant is undone if it fails,
    // so the microphone never stays on without a recording that can stop it.
    let analyser, src;
    try {
      const ctx = ensureAudioContext();
      analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
      src = ctx.createMediaStreamSource(granted);
      src.connect(analyser);
    } catch (err) {
      try { if (src) src.disconnect(); } catch (e) { /* ignore */ }
      granted.getTracks().forEach((t) => t.stop());
      setStatus(tr.status, `Recording could not start: ${err.message}`, 'err');
      return;
    }
    const data = new Uint8Array(analyser.fftSize);
    let raf = 0;
    const meter = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) { const v = (data[i] - 128) / 128; sum += v * v; }
      tr.level.style.width = `${Math.min(100, Math.sqrt(sum / data.length) * 260)}%`;
      raf = requestAnimationFrame(meter);
    };
    meter();
    const started = Date.now();
    const tm = setInterval(() => { tr.recTime.textContent = `${((Date.now() - started) / 1000).toFixed(1)} s`; }, 200);
    const parts = [];
    let cleaned = false;
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      clearInterval(tm); cancelAnimationFrame(raf);
      try { src.disconnect(); } catch (e) { /* ignore */ }
      granted.getTracks().forEach((t) => t.stop());
      if (active && active.rec === rec) active = null;
      tr.level.style.width = '0';
      tr.rec.classList.remove('recording'); tr.rec.querySelector('span').textContent = 'Record';
      tr.rec.disabled = false; tr.recStop.disabled = true;
    };
    rec.ondataavailable = (e) => { if (e.data && e.data.size) parts.push(e.data); };
    rec.onerror = (e) => { recording.discard = true; cleanup(); setStatus(tr.status, `Recording failed: ${(e.error && e.error.message) || 'unknown error'}`, 'err'); };
    const recording = { rec, cleanup, discard: false };
    rec.onstop = async () => {
      cleanup();
      if (recording.discard) return;               // Clear: throw the captured audio away
      const type = rec.mimeType || mime || 'audio/webm';
      const blob = new Blob(parts, { type });
      if (!blob.size) { setStatus(tr.status, 'Nothing was recorded.', 'err'); return; }
      await setClip(blob, `recording.${extensionFor(type)}`, type);
      setStatus(tr.status, 'Clip ready. Press Transcribe.', 'ok');
    };
    active = recording;
    try { rec.start(); }
    catch (err) { cleanup(); setStatus(tr.status, `Recording could not start: ${err.message}`, 'err'); return; }
    tr.recTime.textContent = '0.0 s';
    tr.rec.classList.add('recording'); tr.rec.querySelector('span').textContent = 'Recording';
    tr.rec.disabled = true; tr.recStop.disabled = false;
    setStatus(tr.status, 'Recording… press Stop when done.');
  }
  function stopRecording({ discard = false } = {}) {
    recToken += 1;                   // cancels a start still waiting on the permission prompt
    const cur = active;
    if (!cur) return;
    if (discard) cur.discard = true;
    if (cur.rec.state !== 'inactive') cur.rec.stop();   // onstop finishes the clip and cleans up
    else cur.cleanup();
  }

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
    if (trController) return;                    // single-flight: Cancel first
    const controller = new AbortController();
    trController = controller;
    tr.run.disabled = true; tr.cancel.disabled = false; tr.meta.hidden = true;
    tr.result.classList.add('empty'); tr.result.textContent = 'Transcribing…';
    setStatus(tr.status, 'Transcribing…', '', true);
    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/v1/audio/transcriptions`, { method: 'POST', body: trFormData(), signal: controller.signal });
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
      if (trController === controller) trController = null;
      tr.run.disabled = !clip || !!trController; tr.cancel.disabled = !trController;
    }
  }

  tr.rec.addEventListener('click', startRecording);
  tr.recStop.addEventListener('click', stopRecording);
  $('tr-clear').addEventListener('click', () => {
    stopRecording({ discard: true });
    if (trController) trController.abort();        // its finally re-enables the controls
    trPlayer.reset();
    clip = null;
    tr.file.value = '';
    tr.clipInfo.textContent = ''; tr.recTime.textContent = '';
    tr.run.disabled = true;
    tr.meta.hidden = true;
    tr.result.classList.add('empty'); tr.result.textContent = 'The transcript appears here.';
    tr.elapsed.textContent = '–'; tr.raw.textContent = '';
    updateTrPreview();
    setStatus(tr.status, '');
  });
  // A chosen or dropped file replaces the clip: discard any recording in
  // progress (or still waiting on the permission prompt) so its onstop cannot
  // overwrite the file afterwards.
  function useFile(f, message) {
    if (!f) return;
    stopRecording({ discard: true });
    setClip(f, f.name, f.type);
    setStatus(tr.status, message);
  }
  tr.file.addEventListener('change', () => useFile(tr.file.files && tr.file.files[0], 'File selected.'));
  ['dragenter', 'dragover'].forEach((ev) => tr.drop.addEventListener(ev, (e) => { e.preventDefault(); tr.drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => tr.drop.addEventListener(ev, (e) => { e.preventDefault(); tr.drop.classList.remove('over'); }));
  tr.drop.addEventListener('drop', (e) => useFile(e.dataTransfer.files && e.dataTransfer.files[0], 'File dropped.'));
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
  let liveClearGen = 0;
  let liveQueue = Promise.resolve();
  let liveController = null;           // the capture session (Start → Stop)
  // The batch of transcription requests the log shows. Clear aborts it and
  // starts a new batch with an empty queue while listening carries on; Stop
  // aborts it with the session.
  let liveBatch = null;
  function newLiveBatch() {
    if (liveBatch) liveBatch.abort();
    liveBatch = new AbortController();
    liveQueue = Promise.resolve();
    return liveBatch;
  }

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
    if (mode !== 'clip') { stopRecording(); trPlayer.stop(); if (trController) trController.abort(); }
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
    // Settings as they were when this utterance was spoken, not when its turn
    // in the queue comes.
    const language = tl.language.value, model = tl.model.value.trim();
    const clearGen = liveClearGen;
    const batch = liveBatch || newLiveBatch();
    const dropped = () => session.signal.aborted || batch.signal.aborted || clearGen !== liveClearGen;
    liveQueue = liveQueue.then(async () => {
      if (dropped()) { row.remove(); return; }   // stopped or cleared while queued
      try {
        const r = await transcribeBlob(blob, 'utterance.wav', { language, model, signal: batch.signal });
        if (dropped()) return;                   // the log was cleared meanwhile
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
        if (err.name === 'AbortError' || dropped()) { row.remove(); return; }   // not this log's business any more
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
    newLiveBatch();
    tl.start.disabled = true; tl.start.querySelector('span').textContent = 'Starting…';   // no second click meanwhile
    tl.stop.disabled = false;
    let started = false;
    try { started = await liveMic.start(); }
    catch (err) {
      // A superseded attempt must not report anything: the replacement may have
      // already succeeded, and this error would sit over a working microphone.
      if (liveController !== session) return;
      liveController = null; resetLiveControls();
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
    if (liveBatch) { liveBatch.abort(); liveBatch = null; }
    liveQueue = Promise.resolve();
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
    liveClearGen += 1;                         // results still in flight belong to the cleared log
    if (liveBatch) newLiveBatch();             // cancel in-flight and queued requests; keep listening
    setStatus(tl.status, '');
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
  // A manual replay while listening pauses the microphone until it ends (turns
  // manage the microphone themselves while echoBusy).
  const ecPlayer = makePlayer({ waveEl: $('ec-wave'), canvas: $('ec-canvas'), button: $('ec-play'), audioEl: $('ec-audio'),
    onPlayback: (playing) => {
      if (!echoOn || echoBusy) return;
      if (playing) { echoMic.pause(); setEchoState('speaking', 'Replaying… listening resumes when it ends.'); }
      else { echoMic.resume(); setEchoState('listening', 'Listening… say something.'); }
    } });
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
    if (!listing) return;          // pickers not filled yet: keep the stored choices
    savePrefs({ ecEngine: ec.engine.value, ecVoice: ec.voice.value, ecTtsLanguage: ec.ttsLanguage.value, ecAsrLanguage: ec.asrLanguage.value,
                ecSpeed: ec.speed.value, ecSilence: ec.silence.value, ecSens: ec.sens.value });
  }
  function updateEchoStats() {
    $('ec-count').textContent = String(echoStats.count);
    $('ec-asr').textContent = fmt(avg(echoStats.asr), 2);
    $('ec-tts').textContent = fmt(avg(echoStats.tts), 2);
    $('ec-turn').textContent = fmt(avg(echoStats.turn), 2);
  }
  function bubble(kind, who, text, container = ec.turns) {
    const ph = container.querySelector('.placeholder'); if (ph) ph.remove();
    const m = el('div', `message ${kind}`);
    m.appendChild(el('span', 'who', who));
    m.appendChild(el('span', 'txt', text));
    m.appendChild(el('span', 'meta'));
    container.appendChild(m);
    container.scrollTop = container.scrollHeight;
    return m;
  }
  const setBubble = (m, text, meta, cls) => {
    m.querySelector('.txt').textContent = text;
    m.querySelector('.meta').textContent = meta || '';
    m.classList.remove('pending'); if (cls) m.classList.add(cls);
    if (m.parentElement) m.parentElement.scrollTop = m.parentElement.scrollHeight;
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
      if (echoOn && ecPlayer.playing) { setEchoState('speaking', 'Replaying… listening resumes when it ends.'); }
      else if (echoOn) { echoMic.resume(); setEchoState('listening', 'Listening… say something.'); }
    }
  });
  let echoStarting = false;
  // Bumped by every start and stop. getUserMedia can still be pending when the
  // user cancels and presses again: without a token the OLD attempt's rejection
  // clears the shared flag under the new one, and an old SUCCESS leaves the
  // microphone live while the UI reports it unavailable.
  let echoAttempt = 0;
  async function startEcho() {
    if (echoOn || echoStarting) return;
    const attempt = ++echoAttempt;
    ensureAudioContext();
    echoMic.set({ sensitivity: Number(ec.sens.value), silenceMs: Number(ec.silence.value) });
    setStatus(ec.status, '');
    echoStarting = true;
    setEchoState('busy', 'Starting the microphone… press again to cancel.');
    let started = false;
    try { started = await echoMic.start(); }
    catch (err) {
      if (attempt !== echoAttempt) return;     // superseded; leave state alone
      echoStarting = false;
      setEchoState('err', `Microphone unavailable: ${err.message}`);
      return;
    }
    if (attempt !== echoAttempt) {
      // A newer attempt (or a stop) owns the UI now. Release this stream rather
      // than leaving the microphone open with nothing reading it.
      if (started) echoMic.stop();
      return;
    }
    if (!echoStarting || !started) return;    // cancelled while the prompt / setup was pending
    echoStarting = false;
    echoOn = true;
    // A cancelled turn may still be unwinding: stay paused until its finally
    // resumes the microphone, so no utterance is dropped by the busy guard.
    if (echoBusy) { echoMic.pause(); setEchoState('busy', 'Finishing the previous turn…'); }
    else if (ecPlayer.playing) { echoMic.pause(); setEchoState('speaking', 'Replaying… listening resumes when it ends.'); }
    else setEchoState('listening', 'Listening… say something.');
  }
  function stopEcho() {
    echoAttempt++;                 // invalidate any startup still in flight
    echoStarting = false;
    echoOn = false;
    echoMic.stop();
    if (echoController) echoController.abort();
    ecPlayer.stop();
    setEchoState('idle', 'Press to start, then just talk. What you say is transcribed and spoken straight back.');
  }
  ec.start.addEventListener('click', () => { if (echoOn || echoStarting) stopEcho(); else startEcho(); });
  ec.engine.addEventListener('change', () => { fillVoiceSelect(ec.voice, ec.engine.value); fillEchoTtsLanguages(); saveEchoPrefs(); });
  [ec.voice, ec.ttsLanguage, ec.asrLanguage].forEach((e) => e.addEventListener('change', saveEchoPrefs));
  bindSlider(ec.speed, $('ec-speedOut'), (v) => `${v.toFixed(2)}×`);
  bindSlider(ec.silence, $('ec-silenceOut'), (v) => `${v} ms`);
  bindSlider(ec.sens, $('ec-sensOut'), (v) => v.toFixed(2));
  ec.speed.addEventListener('input', saveEchoPrefs);
  ec.silence.addEventListener('input', () => { echoMic.set({ silenceMs: Number(ec.silence.value) }); saveEchoPrefs(); });
  ec.sens.addEventListener('input', () => { echoMic.set({ sensitivity: Number(ec.sens.value) }); saveEchoPrefs(); });
  ec.clear.addEventListener('click', () => {
    if (echoController) echoController.abort();   // the turn in progress belongs to the cleared history
    ecPlayer.reset();                              // and the last reply's audio with it
    ec.turns.innerHTML = '<div class="placeholder">Each turn shows what was heard and the reply that was spoken back, with timings.</div>';
    Object.assign(echoStats, { count: 0, asr: [], tts: [], turn: [] }); updateEchoStats();
  });

  // =====================================================================
  // Translate: speak or type in one language, get it in another
  // =====================================================================
  // Into English, speech goes through Whisper's translate task
  // (/v1/audio/translations). Into any other language, Whisper transcribes in
  // the spoken language and the loaded chat model translates the text through
  // the Studio's /v1/chat/completions. Typed text always goes to the chat model.
  const xl = {
    source: $('xl-source'), target: $('xl-target'), tone: $('xl-tone'), swap: $('xl-swap'), route: $('xl-route'), model: $('xl-model'),
    start: $('xl-start'), state: $('xl-state'), level: $('xl-level'), threshold: $('xl-threshold'),
    text: $('xl-text'), send: $('xl-send'), speak: $('xl-speak'), engine: $('xl-engine'), voice: $('xl-voice'),
    speed: $('xl-speed'), silence: $('xl-silence'), sens: $('xl-sens'), status: $('xl-status'),
    turns: $('xl-turns'), copy: $('xl-copy'), clear: $('xl-clear'),
  };
  const XL_PLACEHOLDER = '<div class="placeholder">Each turn shows the original with its detected language and the translation, with ASR, LLM and TTS timings.</div>';
  const xlPlayer = makePlayer({ waveEl: $('xl-wave'), canvas: $('xl-canvas'), button: $('xl-play'), audioEl: $('xl-audio'),
    onPlayback: (playing) => {
      if (!xlOn || xlBusy) return;               // a turn manages the microphone itself
      if (playing) { xlMic.pause(); setXlState('speaking', 'Replaying… listening resumes when it ends.'); }
      else { xlMic.resume(); setXlState('listening', 'Listening… speak, then pause.'); }
    } });
  const xlMic = createLiveMic({ silenceMs: 700, sensitivity: 0.5, maxSpeechMs: 15000 });
  bindMeter(xlMic, xl.level, xl.threshold);
  const xlStats = { count: 0, asr: [], llm: [], tts: [] };
  const xlPairs = [];                // [original, translation] for Copy all
  let xlOn = false, xlStarting = false, xlBusy = false, xlController = null;
  let xlChatModel = null, xlModelGen = 0;
  const XL_IDLE = 'Press to listen and speak, or type below. Each utterance is translated as soon as you pause.';
  const langName = (code) => LANGUAGE_NAMES[code] || code;
  const langTag = (code) => (code ? `${code} · ${langName(code)}` : 'language not detected');

  function setXlState(cls, text) {
    xl.state.className = `echo-state ${cls}`; xl.state.textContent = text;
    xl.start.className = `echo-btn ${(xlOn || xlStarting) ? 'on ' : ''}${cls}`;
  }
  xlMic.on('state', (st) => {
    if (!xlOn || xlBusy) return;
    if (st === 'listening') setXlState('listening', 'Listening… speak, then pause.');
    else if (st === 'speech') setXlState('voice', 'Hearing you…');
  });

  function speakable(code, engine) { return engineLanguages(engine).includes(code); }
  function fillTranslateSources(wanted) {
    const w = wanted || xl.source.value || 'auto';
    xl.source.innerHTML = '';
    ASR_LANGUAGES.forEach((c) => xl.source.appendChild(option(c, c === 'auto' ? 'auto · detect' : langTag(c))));
    xl.source.value = ASR_LANGUAGES.includes(w) ? w : 'auto';
  }
  function fillTranslateTargets(wanted) {
    const w = wanted || xl.target.value || 'en';
    const engineLabel = (engineEntry(xl.engine.value) || {}).label || 'the router';
    const codes = Object.keys(LANGUAGE_NAMES).sort((a, b) => (a === 'en' ? -1 : b === 'en' ? 1 : langName(a).localeCompare(langName(b))));
    xl.target.innerHTML = '';
    codes.forEach((c) => xl.target.appendChild(option(c, `${langTag(c)}${speakable(c, xl.engine.value) ? '' : ` · not speakable by ${engineLabel}`}`)));
    xl.target.value = codes.includes(w) ? w : 'en';
  }
  // Tone (register) of the translation. Whisper's translate task has none, so
  // any tone but neutral goes through the chat model, into English too.
  const TONES = {
    neutral: { label: 'neutral', rule: '' },
    formal: { label: 'formal', rule: 'Use a formal, polite register: formal forms of address (for example vous, Sie, usted, Lei), honorifics where the language has them, no slang or contractions.' },
    casual: { label: 'casual', rule: 'Use a casual, everyday register as between friends: informal forms of address (for example tu, du, tú), contractions and natural colloquial phrasing are fine.' },
    friendly: { label: 'friendly', rule: 'Use a warm, friendly and upbeat register, informal but polite.' },
    business: { label: 'business', rule: 'Use a professional business register: clear, courteous and concise, as in a work email.' },
    simple: { label: 'simple', rule: 'Use plain, simple words and short sentences that a language learner can understand.' },
  };
  const toneOf = () => (TONES[xl.tone.value] ? xl.tone.value : 'neutral');
  // Whisper's translate task only when nothing but a literal English rendering is wanted.
  const whisperOnly = (target, tone) => target === 'en' && tone === 'neutral';
  function refreshRoute() {
    const target = xl.target.value, tone = toneOf();
    xl.route.textContent = whisperOnly(target, tone) ? 'translations → speech' : 'transcriptions → chat → speech';
    const note = xl.model.querySelector('span');
    if (xlChatModel) {
      xl.model.className = 'model-note ok';
      note.textContent = `Chat model: ${xlChatModel}${whisperOnly(target, tone) ? ' (used for typed text)' : ''}`;
    } else if (xlModelGen === 0) {
      xl.model.className = 'model-note idle'; note.textContent = 'Checking the chat model…';
    } else {
      xl.model.className = 'model-note warn';
      note.textContent = whisperOnly(target, tone)
        ? 'No chat model loaded: spoken English translation works; typed text and tones need a chat model (Studio → Settings → Models).'
        : `No chat model loaded: load one in the Studio (Settings → Models) to translate into ${langName(target)}${tone === 'neutral' ? '' : ` with a ${TONES[tone].label} tone`}. Spoken → English (neutral) works without one.`;
    }
  }
  /** The loaded chat/VLM model from /models/status (first loaded non-ASR entry). */
  async function refreshChatModel() {
    const gen = ++xlModelGen;
    let name = null;
    try {
      const res = await fetch(`${API}/models/status`, { headers: { Accept: 'application/json' } });
      const data = await res.json().catch(() => ({}));
      const loaded = (data.catalog || []).find((m) => m.loaded && (m.type || 'chat') !== 'asr');
      name = loaded ? loaded.name : null;
    } catch (e) { name = null; }
    if (gen !== xlModelGen) return xlChatModel;   // a newer check superseded this one
    xlChatModel = name;
    refreshRoute();
    return xlChatModel;
  }
  function initTranslatePickers() {
    const prefs = loadPrefs();
    fillEngineSelect(xl.engine, prefs.xlEngine);
    fillVoiceSelect(xl.voice, xl.engine.value, prefs.xlVoice);
    fillTranslateTargets(prefs.xlTarget || 'en');
    refreshRoute();
  }
  function saveTranslatePrefs() {
    if (!listing) return;          // pickers not filled yet: keep the stored choices
    savePrefs({ xlSource: xl.source.value, xlTarget: xl.target.value, xlTone: toneOf(), xlEngine: xl.engine.value, xlVoice: xl.voice.value,
                xlSpeak: xl.speak.checked, xlSpeed: xl.speed.value, xlSilence: xl.silence.value, xlSens: xl.sens.value });
  }
  function updateXlStats() {
    $('xl-count').textContent = String(xlStats.count);
    $('xl-asr').textContent = fmt(avg(xlStats.asr), 2);
    $('xl-llm').textContent = fmt(avg(xlStats.llm), 2);
    $('xl-tts').textContent = fmt(avg(xlStats.tts), 2);
  }

  // Small models sometimes wrap the answer: drop reasoning blocks, leaked stop
  // tokens, a "Translation:" label and surrounding quotes.
  function cleanTranslation(text) {
    let t = String(text || '');
    t = t.replace(/<think>[\s\S]*?<\/think>/gi, '');
    const open = t.search(/<think>/i);
    if (open >= 0) t = t.slice(0, open);          // still thinking: show nothing of it
    t = t.replace(/<\|?(end_of_turn|eot_id|im_end|endoftext|end)\|?>/gi, '');
    t = t.trim().replace(/^(translation|translated text)\s*:\s*/i, '');
    const m = t.match(/^(["“«„'])([\s\S]*)(["”»“'])$/);
    if (m) t = m[2].trim();
    return t;
  }
  /** Translate `text` with the loaded chat model, streaming partial text to onText. */
  function translationPrompt({ source, target, tone }) {
    const rule = (TONES[tone] || TONES.neutral).rule;
    const only = 'Reply with the result only: no explanations, notes, transliteration or quotes.';
    if (source && source !== 'auto' && source === target) {   // same language: restyle only
      return `You are an editor. Rewrite the user's text in ${langName(target)}, keeping its meaning. ${rule} ${only}`.replace(/\s+/g, ' ').trim();
    }
    const from = source && source !== 'auto' ? ` from ${langName(source)}` : '';
    return `You are a translator. Translate the user's text${from} into ${langName(target)}. ${rule} ${only}`.replace(/\s+/g, ' ').trim();
  }
  async function translateWithLlm(text, { model, source, target, tone = 'neutral', signal, onText }) {
    const body = {
      model, stream: true, temperature: 0, max_tokens: 512,
      messages: [
        { role: 'system', content: translationPrompt({ source, target, tone }) },
        { role: 'user', content: text },
      ],
    };
    const t0 = performance.now();
    const res = await fetch(`${API}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' }, body: JSON.stringify(body), signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const e = err.error;
      throw new Error((e && (e.message || (typeof e === 'string' ? e : ''))) || `Chat model error (HTTP ${res.status})`);
    }
    let acc = '';
    if (!(res.headers.get('Content-Type') || '').includes('text/event-stream') || !res.body) {
      const data = await res.json().catch(() => ({}));
      acc = ((((data.choices || [])[0] || {}).message || {}).content) || '';
    } else {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '', done = false;
      while (!done) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buf += decoder.decode(chunk.value, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (payload === '[DONE]') { done = true; break; }
          try {
            const delta = ((((JSON.parse(payload).choices || [])[0] || {}).delta || {}).content) || '';
            if (delta) { acc += delta; if (onText) onText(cleanTranslation(acc)); }
          } catch (e) { /* keep-alive or partial line */ }
        }
      }
      if (done) { try { await reader.cancel(); } catch (e) { /* ignore */ } }
    }
    return { text: cleanTranslation(acc), secs: (performance.now() - t0) / 1000 };
  }

  /** One translation turn from a spoken utterance ({blob, dur}) or typed text. */
  async function runTranslation({ blob, dur, typed }) {
    if (xlBusy) return;
    xlBusy = true;
    xl.send.disabled = true;
    const session = new AbortController();
    xlController = session;
    // Settings as they are now, not when a later await resumes.
    const source = xl.source.value, target = xl.target.value, engine = xl.engine.value, tone = toneOf();
    const voice = xl.voice.value, speed = Number(xl.speed.value), speak = xl.speak.checked;
    if (xlOn) xlMic.pause();                     // no listening while we translate and speak
    const signal = session.signal;
    const turn = [];                              // bubbles of this turn, for cancellation
    const add = (kind, who, text) => { const b = bubble(kind, who, text, xl.turns); turn.push(b); return b; };
    const you = add('user pending', blob ? 'You said' : 'You typed', blob ? `listening to ${dur.toFixed(1)} s of speech…` : typed);
    const timing = {};
    let out = null;
    try {
      let original = typed || '', detected = source !== 'auto' ? source : null, translation = '', how = '';
      if (blob) {
        const toEnglish = whisperOnly(target, tone);
        setXlState('busy', toEnglish ? 'Translating to English…' : 'Transcribing…');
        // Into English: Whisper's translate task gives the translation, and a
        // transcription of the same clip (requested alongside) gives the original.
        const [r, src] = await Promise.all([
          transcribeBlob(blob, 'utterance.wav', { language: source, signal, endpoint: toEnglish ? 'translations' : 'transcriptions' }),
          toEnglish ? transcribeBlob(blob, 'utterance.wav', { language: source, signal }).catch((e) => { if (e.name === 'AbortError') throw e; return null; })
                    : Promise.resolve(null),
        ]);
        timing.asr = Math.max(r.secs, src ? src.secs : 0);
        detected = r.data.language || (src && src.data.language) || detected;
        if (r.data.ignored || !r.text) {
          setBubble(you, r.text || '(nothing recognisable)', `${dur.toFixed(1)} s · ${r.data.ignored ? `ignored: ${r.data.reason || 'filtered'}` : 'no words'}`, 'ignored');
          you.style.opacity = '.6';
          return;
        }
        if (toEnglish) {
          original = (src && src.text) || '';
          setBubble(you, original || `(${dur.toFixed(1)} s of ${detected ? langName(detected) : 'speech'}; source transcript unavailable)`,
                    `${langTag(detected)} · ASR ${timing.asr.toFixed(2)} s`);
          translation = r.text; how = `Whisper translate ${r.secs.toFixed(2)} s`;
        } else {
          original = r.text;
          setBubble(you, original, `${langTag(detected)} · ASR ${r.secs.toFixed(2)} s`);
        }
      } else {
        setBubble(you, typed, detected ? langTag(detected) : 'typed');
      }
      const targetName = langName(target);
      const toneLabel = tone === 'neutral' ? '' : TONES[tone].label;
      const who = toneLabel ? `${targetName} · ${toneLabel}` : targetName;
      if (!translation) {
        if (detected && detected === target && tone === 'neutral') {
          translation = original; how = `already in ${targetName}`;
        } else {
          if (!xlChatModel) await refreshChatModel();
          if (signal.aborted) return;
          const model = xlChatModel;               // this turn's model: request and label agree
          if (!model) throw new Error(`Load a chat model in the Studio (Settings → Models) to translate into ${targetName}${toneLabel ? ` with a ${toneLabel} tone` : ''}.`);
          setXlState('busy', detected === target ? `Rewriting in a ${toneLabel} tone…` : `Translating into ${targetName}${toneLabel ? ` (${toneLabel})` : ''}…`);
          out = add('assistant pending', who, 'translating…');
          const llm = await translateWithLlm(original, { model, source: detected || source, target, tone, signal,
            onText: (t) => { if (!signal.aborted && t) out.querySelector('.txt').textContent = t; } });
          timing.llm = llm.secs;
          translation = llm.text; how = `${model} · LLM ${llm.secs.toFixed(2)} s`;
          if (!translation) throw new Error('The chat model returned no translation.');
        }
      }
      if (signal.aborted) return;
      if (!out) out = add('assistant', who, translation);
      setBubble(out, translation, how);
      xlPairs.push([original || `(${langTag(detected)} speech)`, translation]);
      xlStats.count += 1;
      if (timing.asr != null) xlStats.asr.push(timing.asr);
      if (timing.llm != null) xlStats.llm.push(timing.llm);
      if (speak && speakable(target, engine)) {
        setXlState('busy', 'Synthesizing…');
        const body = { input: translation, model: engine, language: target, speed, response_format: 'wav' };
        if (voice && voice !== 'default') body.voice = voice;
        const tts = await speakText(body, signal);
        timing.tts = tts.secs; xlStats.tts.push(tts.secs);
        const h = (n) => tts.headers.get(n) || '–';
        setBubble(out, translation, `${how} · ${h('X-Engine')} ${h('X-Voice')} TTS ${tts.secs.toFixed(2)} s`);
        updateXlStats();
        if (signal.aborted) return;
        setXlState('speaking', 'Speaking the translation…');
        out.classList.add('speaking');
        await xlPlayer.load(tts.blob, { autoplay: true });
        out.classList.remove('speaking');
      } else {
        if (speak) setBubble(out, translation, `${how} · not spoken: ${(engineEntry(engine) || {}).label || 'no engine'} has no ${targetName} voice`);
        updateXlStats();
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        const pendingBubble = turn.find((b) => b.classList.contains('pending'));
        if (pendingBubble) setBubble(pendingBubble, err.message, '', 'err');
        else add('assistant err', 'Error', err.message);
        setStatus(xl.status, err.message, 'err');
      }
    } finally {
      turn.forEach((b) => { if (b.classList.contains('pending')) { setBubble(b, '(cancelled)', '', 'ignored'); b.style.opacity = '.6'; } });
      if (out) out.classList.remove('speaking');
      if (xlController === session) xlController = null;
      xlBusy = false;
      xl.send.disabled = false;
      if (xlOn && xlPlayer.playing) { setXlState('speaking', 'Replaying… listening resumes when it ends.'); }
      else if (xlOn) { xlMic.resume(); setXlState('listening', 'Listening… speak, then pause.'); }
      else if (!xlStarting) setXlState('idle', XL_IDLE);
    }
  }

  xlMic.on('segment', (blob, dur) => { if (xlOn && !xlBusy) runTranslation({ blob, dur }); });
  let xlAttempt = 0;                 // see echoAttempt: per-attempt ownership
  async function startTranslate() {
    if (xlOn || xlStarting) return;
    const attempt = ++xlAttempt;
    ensureAudioContext();
    xlMic.set({ sensitivity: Number(xl.sens.value), silenceMs: Number(xl.silence.value) });
    setStatus(xl.status, '');
    xlStarting = true;
    setXlState('busy', 'Starting the microphone… press again to cancel.');
    let started = false;
    try { started = await xlMic.start(); }
    catch (err) {
      if (attempt !== xlAttempt) return;
      xlStarting = false;
      setXlState('err', `Microphone unavailable: ${err.message}`);
      return;
    }
    if (attempt !== xlAttempt) {
      if (started) xlMic.stop();
      return;
    }
    if (!xlStarting || !started) return;          // cancelled while the prompt / setup was pending
    xlStarting = false;
    xlOn = true;
    if (xlBusy) xlMic.pause();                     // a typed translation is still running
    else if (xlPlayer.playing) { xlMic.pause(); setXlState('speaking', 'Replaying… listening resumes when it ends.'); }
    else setXlState('listening', 'Listening… speak, then pause.');
  }
  function stopTranslate() {
    xlAttempt++;
    xlStarting = false;
    xlOn = false;
    xlMic.stop();
    if (xlController) xlController.abort();
    xlPlayer.stop();
    setXlState('idle', XL_IDLE);
  }
  function sendTyped() {
    const text = xl.text.value.trim();
    if (!text) { setStatus(xl.status, 'Type something to translate.', 'err'); return; }
    if (xlBusy) return;
    ensureAudioContext();                          // created on the click: playback is allowed
    setStatus(xl.status, '');
    runTranslation({ typed: text });
  }
  xl.start.addEventListener('click', () => { if (xlOn || xlStarting) stopTranslate(); else startTranslate(); });
  xl.send.addEventListener('click', sendTyped);
  xl.text.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') sendTyped(); });
  xl.engine.addEventListener('change', () => { fillVoiceSelect(xl.voice, xl.engine.value); fillTranslateTargets(); saveTranslatePrefs(); });
  xl.target.addEventListener('change', () => { refreshRoute(); saveTranslatePrefs(); });
  xl.tone.addEventListener('change', () => { refreshRoute(); saveTranslatePrefs(); });
  [xl.source, xl.voice, xl.speak].forEach((e) => e.addEventListener('change', saveTranslatePrefs));
  xl.swap.addEventListener('click', () => {
    const s = xl.source.value, t = xl.target.value;
    if (s !== 'auto' && [...xl.target.options].some((o) => o.value === s)) xl.target.value = s;
    xl.source.value = ASR_LANGUAGES.includes(t) ? t : 'auto';
    refreshRoute(); saveTranslatePrefs();
  });
  bindSlider(xl.speed, $('xl-speedOut'), (v) => `${v.toFixed(2)}×`);
  bindSlider(xl.silence, $('xl-silenceOut'), (v) => `${v} ms`);
  bindSlider(xl.sens, $('xl-sensOut'), (v) => v.toFixed(2));
  xl.speed.addEventListener('input', saveTranslatePrefs);
  xl.silence.addEventListener('input', () => { xlMic.set({ silenceMs: Number(xl.silence.value) }); saveTranslatePrefs(); });
  xl.sens.addEventListener('input', () => { xlMic.set({ sensitivity: Number(xl.sens.value) }); saveTranslatePrefs(); });
  xl.clear.addEventListener('click', () => {
    if (xlController) xlController.abort();       // the turn in progress belongs to the cleared history
    xlPlayer.reset();                              // and the last translation's audio with it
    xl.turns.innerHTML = XL_PLACEHOLDER;
    xlPairs.length = 0;
    Object.assign(xlStats, { count: 0, asr: [], llm: [], tts: [] }); updateXlStats();
  });
  xl.copy.addEventListener('click', async () => {
    const text = xlPairs.map(([a, b]) => `${a}\n→ ${b}`).join('\n\n');
    try { await navigator.clipboard.writeText(text); xl.copy.textContent = 'Copied'; } catch (e) { xl.copy.textContent = 'Select & copy'; }
    setTimeout(() => { xl.copy.textContent = 'Copy all'; }, 1500);
  });
  window.addEventListener('focus', () => { if (currentTab === 'translate') refreshChatModel(); });
  fillTranslateSources();

  enterTab = (name) => { if (name === 'translate') refreshChatModel(); };
  // Leaving a tab stops whatever it was doing (playback, recording, listening).
  leaveTab = (name) => {
    if (name === 'speech') { if (spController) spController.abort(); spPlayer.stop(); }
    else if (name === 'transcription') { stopLive(); stopRecording(); trPlayer.stop(); if (trController) trController.abort(); }
    else if (name === 'echo') stopEcho();
    else if (name === 'translate') stopTranslate();
  };

  // ---- boot --------------------------------------------------------------
  window.addEventListener('pagehide', () => {
    stopRecording();
    stopLive(); stopEcho(); stopTranslate(); spPlayer.stop(); trPlayer.stop();
    if (downloadUrl) URL.revokeObjectURL(downloadUrl);
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
  if (prefs.xlSource) fillTranslateSources(prefs.xlSource);
  if (prefs.xlTone && [...xl.tone.options].some((o) => o.value === prefs.xlTone)) xl.tone.value = prefs.xlTone;
  if (prefs.xlSpeak === false) xl.speak.checked = false;
  if (prefs.xlSpeed) xl.speed.value = prefs.xlSpeed;
  if (prefs.xlSilence) xl.silence.value = prefs.xlSilence;
  if (prefs.xlSens) xl.sens.value = prefs.xlSens;
  // Readouts only: firing 'input' here would run the save handlers while the
  // voice pickers are still empty and overwrite the stored choices.
  sliderShows.forEach((show) => show());
  updateTrPreview();
  trMode(prefs.trMode === 'live' ? 'live' : 'clip');
  const initial = (location.hash || '').replace('#', '') || prefs.tab || 'speech';
  showTab(initial);
  listingWaiters.push(initSpeechPickers, initEchoPickers, initTranslatePickers);
  loadVoices();
})();
