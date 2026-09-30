/* Speech API harness: exercises the Studio's OpenAI-compatible speech endpoint
 * from the same origin. No external requests, no build step.
 *
 *   GET  ../../v1/audio/voices   -> engines, voices, languages (fills the pickers)
 *   POST ../../v1/audio/speech   -> audio/wav + X-* headers (played, downloadable)
 */
(function () {
  'use strict';

  const API_BASE = new URL('../../', window.location.href).href.replace(/\/$/, '');
  const STORE_KEY = 'sima-studio:speech-harness';
  const LANGUAGE_NAMES = {
    ar: 'Arabic', bg: 'Bulgarian', cs: 'Czech', da: 'Danish', de: 'German', el: 'Greek', en: 'English',
    es: 'Spanish', et: 'Estonian', fi: 'Finnish', fr: 'French', hi: 'Hindi', hr: 'Croatian',
    hu: 'Hungarian', id: 'Indonesian', it: 'Italian', ja: 'Japanese', ko: 'Korean', lt: 'Lithuanian',
    lv: 'Latvian', nl: 'Dutch', no: 'Norwegian', pl: 'Polish', pt: 'Portuguese', ro: 'Romanian',
    ru: 'Russian', sk: 'Slovak', sl: 'Slovenian', sv: 'Swedish', tr: 'Turkish', uk: 'Ukrainian',
    vi: 'Vietnamese', zh: 'Chinese',
  };

  const $ = (id) => document.getElementById(id);
  const dom = {
    input: $('input'), engine: $('engine'), voice: $('voice'), language: $('language'),
    speed: $('speed'), speedLabel: $('speedLabel'), synthesize: $('synthesize'), stop: $('stop'),
    download: $('download'), status: $('status'), player: $('player'), conn: $('conn'),
    sEngine: $('sEngine'), sVoice: $('sVoice'), sSpeed: $('sSpeed'), sDuration: $('sDuration'),
    sElapsed: $('sElapsed'), sRtf: $('sRtf'), reqJson: $('reqJson'), curl: $('curl'),
    respHeaders: $('respHeaders'), voicesJson: $('voicesJson'), home: $('home-button'),
  };

  let listing = null;         // GET /v1/audio/voices payload
  let controller = null;      // AbortController of the in-flight request
  let objectUrl = null;       // Blob URL of the last WAV

  function setStatus(text, kind) {
    dom.status.textContent = text || '';
    dom.status.className = 'status' + (kind ? ' ' + kind : '');
  }

  function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch (e) { return {}; }
  }
  function savePrefs() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        engine: dom.engine.value, voice: dom.voice.value, language: dom.language.value,
        speed: dom.speed.value, input: dom.input.value,
      }));
    } catch (e) { /* storage may be unavailable */ }
  }

  function option(value, label) {
    const o = document.createElement('option');
    o.value = value;
    o.textContent = label;
    return o;
  }

  function engineEntry(key) {
    return (listing && listing.engines || []).find((e) => e.key === key) || null;
  }

  function fillVoices(prefs) {
    const entry = engineEntry(dom.engine.value);
    dom.voice.innerHTML = '';
    dom.voice.appendChild(option('default', 'default (engine choice)'));
    (entry && entry.voices || []).forEach((v) => {
      const bits = [v.label || v.id];
      if (v.language) bits.push(v.language);
      if (v.installed === false) bits.push('download on select');
      if (v.default) bits.push('current');
      dom.voice.appendChild(option(v.id, bits.join(' · ')));
    });
    if (prefs && prefs.voice && [...dom.voice.options].some((o) => o.value === prefs.voice)) {
      dom.voice.value = prefs.voice;
    }
    const languages = entry && entry.languages && entry.languages.length
      ? entry.languages : (listing && listing.languages || ['en']);
    const current = dom.language.value;
    dom.language.innerHTML = '';
    languages.forEach((code) => dom.language.appendChild(
      option(code, LANGUAGE_NAMES[code] ? `${code} · ${LANGUAGE_NAMES[code]}` : code)));
    const wanted = (prefs && prefs.language) || current;
    dom.language.value = languages.includes(wanted) ? wanted : (languages.includes('en') ? 'en' : languages[0]);
  }

  async function loadVoices() {
    const prefs = loadPrefs();
    dom.conn.textContent = 'loading voices…';
    try {
      const res = await fetch(`${API_BASE}/v1/audio/voices`, { headers: { Accept: 'application/json' } });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      listing = data;
    } catch (err) {
      listing = { engines: [], languages: ['en'], default_engine: null };
      dom.conn.textContent = '';
      setStatus(`Could not list voices: ${err.message}`, 'error');
    }
    dom.voicesJson.textContent = JSON.stringify(listing, null, 2);
    dom.engine.innerHTML = '';
    dom.engine.appendChild(option('default', 'default (router picks)'));
    listing.engines.forEach((e) => dom.engine.appendChild(
      option(e.key, `${e.label || e.key}${e.loaded ? '' : ' · loads on first use'}`)));
    const wanted = prefs.engine || listing.default_engine || 'default';
    dom.engine.value = [...dom.engine.options].some((o) => o.value === wanted) ? wanted : 'default';
    fillVoices(prefs);
    if (prefs.speed) { dom.speed.value = prefs.speed; }
    if (prefs.input) { dom.input.value = prefs.input; }
    dom.speedLabel.textContent = `${Number(dom.speed.value).toFixed(2)}×`;
    dom.conn.textContent = listing.engines.length
      ? `${listing.engines.length} engine${listing.engines.length === 1 ? '' : 's'} · default ${listing.default_engine || 'router'}`
      : 'no server-side engines';
  }

  function buildBody() {
    const body = { input: dom.input.value, model: dom.engine.value, language: dom.language.value,
                   speed: Number(dom.speed.value), response_format: 'wav' };
    if (dom.voice.value && dom.voice.value !== 'default') body.voice = dom.voice.value;
    return body;
  }

  function curlFor(body) {
    const json = JSON.stringify(body).replace(/'/g, "'\\''");
    return `curl -k -X POST ${API_BASE}/v1/audio/speech \\\n  -H 'Content-Type: application/json' \\\n  -d '${json}' -o speech.wav -D -`;
  }

  function showHeaders(res) {
    const lines = [];
    res.headers.forEach((value, key) => { if (/^x-|^content-/i.test(key)) lines.push(`${key}: ${value}`); });
    dom.respHeaders.textContent = lines.join('\n') || '(none)';
    const h = (name) => res.headers.get(name) || '–';
    dom.sEngine.textContent = h('X-Engine');
    dom.sVoice.textContent = h('X-Voice');
    dom.sSpeed.textContent = h('X-Speed');
    const dur = Number(res.headers.get('X-Audio-Duration'));
    const gen = Number(res.headers.get('X-Elapsed-Time'));
    const rtf = Number(res.headers.get('X-RTF'));
    dom.sDuration.textContent = Number.isFinite(dur) && dur ? dur.toFixed(2) : '–';
    dom.sElapsed.textContent = Number.isFinite(gen) && gen ? gen.toFixed(2) : '–';
    dom.sRtf.textContent = Number.isFinite(rtf) && rtf ? rtf.toFixed(3) : '–';
  }

  async function synthesize() {
    const body = buildBody();
    if (!body.input.trim()) { setStatus('Enter some text first.', 'error'); return; }
    savePrefs();
    dom.reqJson.textContent = JSON.stringify(body, null, 2);
    dom.curl.textContent = curlFor(body);
    controller = new AbortController();
    dom.synthesize.disabled = true;
    dom.stop.disabled = false;
    dom.download.hidden = true;
    setStatus('Synthesizing…');
    const started = performance.now();
    try {
      const res = await fetch(`${API_BASE}/v1/audio/speech`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body), signal: controller.signal,
      });
      showHeaders(res);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const detail = [err.error || `HTTP ${res.status}`];
        if (err.param) detail.push(`(param: ${err.param})`);
        if (err.reason) detail.push(`(reason: ${err.reason}${err.engine ? ', engine: ' + err.engine : ''})`);
        throw new Error(detail.join(' '));
      }
      const blob = await res.blob();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(blob);
      dom.player.src = objectUrl;
      dom.player.hidden = false;
      dom.download.href = objectUrl;
      dom.download.hidden = false;
      const ms = performance.now() - started;
      setStatus(`OK · ${(blob.size / 1024).toFixed(0)} KiB WAV in ${(ms / 1000).toFixed(2)} s round trip`, 'ok');
      try { await dom.player.play(); } catch (e) { /* autoplay may need a gesture */ }
    } catch (err) {
      if (err.name === 'AbortError') setStatus('Cancelled.');
      else setStatus(err.message, 'error');
    } finally {
      controller = null;
      dom.synthesize.disabled = false;
      dom.stop.disabled = true;
    }
  }

  dom.engine.addEventListener('change', () => { fillVoices(null); savePrefs(); });
  dom.speed.addEventListener('input', () => { dom.speedLabel.textContent = `${Number(dom.speed.value).toFixed(2)}×`; });
  dom.synthesize.addEventListener('click', synthesize);
  dom.stop.addEventListener('click', () => { if (controller) controller.abort(); dom.player.pause(); });
  dom.input.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') synthesize(); });
  dom.home.addEventListener('click', () => {
    if (window.parent && window.parent !== window) window.parent.postMessage({ type: 'sima-sentry:home' }, '*');
    else window.location.assign('../index.html');
  });

  loadVoices();
})();
