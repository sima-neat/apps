/* Audio API playground for Neat GenAI Studio.
 *
 * Same-origin only:
 *   GET  /v1/audio/voices           engines, voices, languages
 *   POST /v1/audio/speech           WAV + X-* headers
 *   POST /v1/audio/transcriptions   json | verbose_json | text
 *
 * Playback goes through the Web Audio API (decodeAudioData + a buffer source,
 * started from the user's click, exactly as the Studio's chat playback does)
 * so it works inside the Studio's embedded frame; the visible <audio> controls
 * are there for scrubbing and replay. The theme follows the Studio's own
 * localStorage 'theme' key. No external requests.
 */
(function () {
  'use strict';

  const API = window.location.origin;
  const PREFS_KEY = 'sima-studio:audio-playground';
  const LANGUAGE_NAMES = {
    ar: 'Arabic', bg: 'Bulgarian', cs: 'Czech', da: 'Danish', de: 'German', el: 'Greek', en: 'English',
    es: 'Spanish', et: 'Estonian', fi: 'Finnish', fr: 'French', hi: 'Hindi', hr: 'Croatian', hu: 'Hungarian',
    id: 'Indonesian', it: 'Italian', ja: 'Japanese', ko: 'Korean', lt: 'Lithuanian', lv: 'Latvian', nl: 'Dutch',
    no: 'Norwegian', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian', ru: 'Russian', sk: 'Slovak', sl: 'Slovenian',
    sv: 'Swedish', tr: 'Turkish', uk: 'Ukrainian', vi: 'Vietnamese', zh: 'Chinese',
  };
  const ASR_LANGUAGES = ['auto', 'en', 'fr', 'es', 'de', 'it', 'pt', 'ja', 'ko', 'zh', 'vi', 'no'];
  const MIME_PREFERENCE = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/wav'];

  const $ = (id) => document.getElementById(id);
  const embedded = window.parent && window.parent !== window;

  // ---- theme: follow the Studio ------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    $('themeSun').hidden = theme === 'dark';
    $('themeMoon').hidden = theme !== 'dark';
    $('logo').src = theme === 'dark' ? '/static/icons/logo_dark.png' : '/static/icons/logo_bright.png';
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

  // ---- header: tabs, close, prefs ----------------------------------------
  function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}'); } catch (e) { return {}; } }
  function savePrefs(patch) {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(Object.assign(loadPrefs(), patch))); } catch (e) { /* ignore */ }
  }
  function showTab(name) {
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
  document.querySelectorAll('.copy-btn').forEach((b) => b.addEventListener('click', async () => {
    const text = $(b.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied'; } catch (e) { b.textContent = 'Select & copy'; }
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  }));

  // ---- audio helpers ---------------------------------------------------
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

  /** One decoded clip with Web Audio playback and a scrolling cursor. */
  function makePlayer({ waveEl, canvas, button, audioEl }) {
    let buffer = null, source = null, startedAt = 0, raf = 0;
    const cursor = waveEl.querySelector('.cursor');
    const label = button.querySelector('span');

    function stop() {
      if (source) { try { source.stop(); } catch (e) { /* already stopped */ } source = null; }
      cancelAnimationFrame(raf);
      waveEl.classList.remove('playing');
      label.textContent = 'Play';
      button.classList.remove('playing');
    }
    function tick() {
      if (!source || !buffer) return;
      const t = (audioCtx.currentTime - startedAt) / buffer.duration;
      cursor.style.left = `${Math.min(100, t * 100)}%`;
      raf = requestAnimationFrame(tick);
    }
    async function play() {
      if (!buffer) return;
      if (source) { stop(); return; }
      const ctx = ensureAudioContext();
      source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      source.onended = stop;
      startedAt = ctx.currentTime;
      source.start();
      waveEl.classList.add('playing');
      label.textContent = 'Stop';
      raf = requestAnimationFrame(tick);
    }
    async function load(blob, { autoplay } = {}) {
      stop();
      buffer = null;
      const ctx = ensureAudioContext();
      const bytes = await blob.arrayBuffer();
      try {
        buffer = await ctx.decodeAudioData(bytes.slice(0));
        drawWave(canvas, buffer);
        waveEl.querySelector('.placeholder').hidden = true;
      } catch (e) {
        drawWave(canvas, null);
        waveEl.querySelector('.placeholder').hidden = false;
        waveEl.querySelector('.placeholder').textContent = 'Cannot decode this clip for the waveform (the <audio> player may still play it)';
      }
      if (audioEl.src) URL.revokeObjectURL(audioEl.src);
      audioEl.src = URL.createObjectURL(blob);
      audioEl.hidden = false;
      button.disabled = !buffer;
      if (autoplay && buffer) await play();
      return buffer;
    }
    button.addEventListener('click', play);
    audioEl.addEventListener('play', () => { if (source) stop(); });   // don't double-play
    window.addEventListener('resize', () => { if (buffer) drawWave(canvas, buffer); });
    return { load, stop, get buffer() { return buffer; } };
  }

  function setStatus(el, text, kind, busy) {
    el.innerHTML = '';
    if (busy) { const s = document.createElement('i'); s.className = 'spinner'; el.appendChild(s); }
    el.appendChild(document.createTextNode(text || ''));
    el.className = 'status' + (kind ? ' ' + kind : '');
  }
  function option(value, label) { const o = document.createElement('option'); o.value = value; o.textContent = label; return o; }
  function fmt(v, d) { return (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(d) : (v == null || v === '' ? '–' : String(v)); }

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
  let listing = null;
  let spController = null;

  function engineEntry(key) { return (listing && listing.engines || []).find((e) => e.key === key) || null; }

  function fillVoices(prefs) {
    const entry = engineEntry(sp.engine.value);
    sp.voice.innerHTML = '';
    sp.voice.appendChild(option('default', entry ? 'engine default' : 'router choice'));
    (entry && entry.voices || []).forEach((v) => {
      const bits = [v.label || v.id];
      if (v.language) bits.push(v.language);
      if (v.installed === false) bits.push('downloads on select');
      if (v.default) bits.push('current');
      sp.voice.appendChild(option(v.id, bits.join(' · ')));
    });
    if (prefs && prefs.voice && [...sp.voice.options].some((o) => o.value === prefs.voice)) sp.voice.value = prefs.voice;
    const languages = (entry && entry.languages && entry.languages.length) ? entry.languages : (listing && listing.languages || ['en']);
    const wanted = (prefs && prefs.language) || sp.language.value;
    sp.language.innerHTML = '';
    languages.forEach((c) => sp.language.appendChild(option(c, LANGUAGE_NAMES[c] ? `${c} · ${LANGUAGE_NAMES[c]}` : c)));
    sp.language.value = languages.includes(wanted) ? wanted : (languages.includes('en') ? 'en' : languages[0]);
  }

  async function loadVoices() {
    const prefs = loadPrefs();
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
      listing = { engines: [], languages: ['en'], default_engine: null };
      pill.className = 'pg-pill err';
      pill.lastElementChild.textContent = 'voices unavailable';
      setStatus(sp.status, `Could not list voices: ${err.message}`, 'err');
    }
    sp.voicesJson.textContent = JSON.stringify(listing, null, 2);
    sp.engine.innerHTML = '';
    sp.engine.appendChild(option('default', 'default · router picks'));
    listing.engines.forEach((e) => sp.engine.appendChild(option(e.key, `${e.label || e.key}${e.loaded ? '' : ' · loads on first use'}`)));
    const wanted = prefs.engine || listing.default_engine || 'default';
    sp.engine.value = [...sp.engine.options].some((o) => o.value === wanted) ? wanted : 'default';
    fillVoices(prefs);
    if (prefs.speed) sp.speed.value = prefs.speed;
    if (prefs.input) sp.input.value = prefs.input;
    sp.speedOut.value = `${Number(sp.speed.value).toFixed(2)}×`;
    updateSpeechPreview();
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

  async function synthesize() {
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
      sp.size.textContent = `${(blob.size / 1024).toFixed(0)} KiB · ${(performance.now() - t0) / 1000 | 0}.${String(Math.round(((performance.now() - t0) % 1000) / 10)).padStart(2, '0')} s round trip`;
      sp.download.href = URL.createObjectURL(blob);
      sp.download.hidden = false;
      await spPlayer.load(blob, { autoplay: true });
      setStatus(sp.status, 'Playing', 'ok');
    } catch (err) {
      if (err.name === 'AbortError') setStatus(sp.status, 'Cancelled.');
      else setStatus(sp.status, err.message, 'err');
    } finally {
      spController = null;
      sp.run.disabled = false; sp.stop.disabled = true;
    }
  }

  sp.engine.addEventListener('change', () => { fillVoices(null); updateSpeechPreview(); });
  [sp.voice, sp.language].forEach((el) => el.addEventListener('change', updateSpeechPreview));
  sp.input.addEventListener('input', updateSpeechPreview);
  sp.speed.addEventListener('input', () => { sp.speedOut.value = `${Number(sp.speed.value).toFixed(2)}×`; updateSpeechPreview(); });
  sp.run.addEventListener('click', synthesize);
  sp.stop.addEventListener('click', () => { if (spController) spController.abort(); spPlayer.stop(); });
  sp.input.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') synthesize(); });

  // =====================================================================
  // Transcription
  // =====================================================================
  const tr = {
    rec: $('tr-rec'), recStop: $('tr-recStop'), recTime: $('tr-recTime'), level: $('tr-level'), drop: $('tr-drop'),
    file: $('tr-file'), dropText: $('tr-dropText'), clipInfo: $('tr-clipInfo'), language: $('tr-language'),
    format: $('tr-format'), model: $('tr-model'), run: $('tr-run'), cancel: $('tr-cancel'), status: $('tr-status'),
    result: $('tr-result'), meta: $('tr-meta'), elapsed: $('tr-elapsed'), curl: $('tr-curl'), raw: $('tr-raw'),
  };
  const trPlayer = makePlayer({ waveEl: $('tr-wave'), canvas: $('tr-canvas'), button: $('tr-play'), audioEl: $('tr-audio') });
  let clip = null, recorder = null, stream = null, chunks = [], timer = 0, meterRaf = 0, trController = null;

  ASR_LANGUAGES.forEach((c) => tr.language.appendChild(option(c, c === 'auto' ? 'auto · detect' : `${c} · ${LANGUAGE_NAMES[c] || c}`)));

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

  async function startRecording() {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (err) { setStatus(tr.status, `Microphone unavailable: ${err.message}`, 'err'); return; }
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
  function stopRecording() { if (recorder && recorder.state !== 'inactive') recorder.stop(); }

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
    tr.result.textContent = ''; tr.result.classList.add('empty'); tr.result.textContent = 'Transcribing…';
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
        ['m-lang', 'm-nospeech', 'm-logprob'].forEach((id) => { $(id).textContent = '–'; });
        $('m-ignored').textContent = '–';
      } else {
        const data = JSON.parse(rawText);
        text = data.text || '';
        $('m-lang').textContent = data.language ? `${data.language}${data.language_detected ? ' (detected)' : ''}` : '–';
        $('m-nospeech').textContent = fmt(data.no_speech_prob, 3);
        $('m-logprob').textContent = fmt(data.avg_logprob, 3);
        $('m-ignored').innerHTML = data.ignored == null ? '–'
          : (data.ignored ? `<span class="badge warn">would be ignored · ${data.reason || ''}</span>` : '<span class="badge ok">accepted</span>');
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
  [tr.language, tr.format].forEach((el) => el.addEventListener('change', updateTrPreview));
  tr.model.addEventListener('input', updateTrPreview);
  tr.run.addEventListener('click', transcribe);
  tr.cancel.addEventListener('click', () => { if (trController) trController.abort(); });
  window.addEventListener('pagehide', () => { stopRecording(); if (stream) stream.getTracks().forEach((t) => t.stop()); spPlayer.stop(); trPlayer.stop(); });

  // ---- boot ------------------------------------------------------------
  const prefs = loadPrefs();
  if (prefs.trLanguage) tr.language.value = prefs.trLanguage;
  if (prefs.trFormat) tr.format.value = prefs.trFormat;
  updateTrPreview();
  const initial = (location.hash || '').replace('#', '') || prefs.tab || 'speech';
  showTab(initial === 'transcription' ? 'transcription' : 'speech');
  loadVoices();
})();
