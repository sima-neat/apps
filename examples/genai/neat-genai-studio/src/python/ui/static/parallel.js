// ---- Parallel panels: a chat panel per loaded model, each sent on its own ----
// Every loaded chat/VLM model gets a panel with its own prompt box, Send
// button and conversation. Each panel streams from the same-origin
// /v1/chat/completions proxy (the OpenAI-compatible model server), so the
// panels run at the same time and independently of the main chat, whose
// server-side history they never touch. Panels keep their history in the page.
(function () {
  'use strict';

  const SYSTEM_PROMPT = 'Answer clearly and concisely. Use Markdown formatting when it helps. '
    + 'Answer the question in the language it was asked in.';
  const WITH_IMAGE = ' The current message includes an image. Answer using what you see in it.';
  const WITHOUT_IMAGE = ' No image is attached to the current message; if asked about one, say so.';
  const SPECTRUM_SVG = '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" '
    + 'stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="5" height="16" rx="1.2"/>'
    + '<rect x="9.5" y="4" width="5" height="16" rx="1.2"/><rect x="16" y="4" width="5" height="16" rx="1.2"/></svg>';

  // model name -> { messages: [{role, text, image}|{role:'assistant', ...}], image, controller }
  const panels = new Map();
  let view = null;
  let open = false;
  let shownModels = '';
  // The loaded models ticked in the Models dropdown get a panel; null means
  // all of them (until the user changes the ticks).
  let chosen = null;
  const drafts = new Map();   // model name -> unsent text in its prompt box
  // 'different': each panel has its own prompt box and Send. 'same': one
  // shared box sends the same prompt to every panel at once.
  let promptMode = 'different';
  const SHARED = '__shared__';  // the shared box's key for pictures and the camera
  let sharedImage = null;

  // The loaded chat/VLM models, from the catalog newui.js keeps up to date.
  const loaded = () => (typeof _catalog !== 'undefined' && Array.isArray(_catalog)
    ? _catalog.filter((m) => (m.type || 'chat') !== 'asr' && m.loaded)
    : []);
  const isVision = (m) => !!m.supportsVision;
  const esc = (s) => (typeof escHtml === 'function' ? escHtml(s) : String(s));
  const answerOf = (text) => (typeof splitThinking === 'function' ? splitThinking(text || '').answer : (text || ''));

  function panelState(name) {
    if (!panels.has(name)) panels.set(name, { messages: [], image: null, controller: null });
    return panels.get(name);
  }

  // The 3.0 accelerator driver now and then refuses a job while several models
  // run at once ("no free bank", rc=-11); one retry gets through.
  function acceleratorBusy(message) {
    return /rc=-11|resource temporarily unavailable|no free bank|queued wait failed/i.test(String(message || ''));
  }

  function fmt(n, digits) { return Number.isFinite(n) ? n.toFixed(digits) : '—'; }

  function statsText(s) {
    const parts = [];
    if (s.tokens) parts.push(`${s.tokens} token${s.tokens === 1 ? '' : 's'}`);
    if (s.tps != null) parts.push(`${fmt(s.tps, 1)} tok/s`);
    if (s.ttftS != null) parts.push(`first token ${fmt(s.ttftS, 2)}s`);
    if (s.totalS != null) parts.push(`${fmt(s.totalS, 1)}s`);
    return parts.join(' · ');
  }

  // ---- header button ----

  function ensureButton() {
    if (document.getElementById('parallelButton')) return;
    const anchor = document.getElementById('benchmarkButton');
    if (!anchor) return;
    const btn = document.createElement('button');
    btn.id = 'parallelButton';
    btn.className = 'header-icon-btn';
    btn.type = 'button';
    btn.style.display = 'none';
    btn.setAttribute('aria-pressed', 'false');
    btn.innerHTML = SPECTRUM_SVG;
    btn.addEventListener('click', () => setOpen(!open));
    anchor.insertAdjacentElement('beforebegin', btn);
  }

  function syncButton() {
    ensureButton();
    const btn = document.getElementById('parallelButton');
    if (!btn) return;
    const enough = loaded().length >= 2;
    btn.style.display = enough || open ? '' : 'none';
    btn.classList.toggle('is-on', open);
    btn.setAttribute('aria-pressed', open ? 'true' : 'false');
    const label = open ? 'Back to the chat' : 'Parallel: a chat panel for each loaded model, each with its own Send';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }

  // ---- the view ----

  function setOpen(on) {
    open = !!on && (loaded().length >= 2 || !on);
    if (!view) buildView();
    view.hidden = !open;
    document.body.classList.toggle('parallel-open', open);
    if (open) { placeView(); renderPanels(true); }
    syncButton();
  }

  function placeView() {
    const header = document.querySelector('.app-header');
    const top = header ? Math.round(header.getBoundingClientRect().bottom) : 64;
    view.style.top = `${top}px`;
  }

  function buildView() {
    view = document.createElement('section');
    view.id = 'parallelView';
    view.className = 'parallel-view';
    view.hidden = true;
    view.setAttribute('aria-label', 'Parallel panels');
    view.innerHTML = '<div class="parallel-head">'
      + '<div><div class="parallel-title">Parallel · <span class="parallel-count"></span></div>'
      + '<div class="parallel-sub">Each model has its own prompt and Send. They run at the same time on the accelerator.</div></div>'
      + '<div class="parallel-head-actions">'
      + '<label class="parallel-mode-pick">Prompt <select class="parallel-mode-select" aria-label="Same or different prompts">'
      + '<option value="different">Different prompts</option><option value="same">Same prompt</option></select></label>'
      + '<div class="parallel-models-pick">'
      + '<button type="button" class="parallel-btn parallel-models-btn" aria-haspopup="true" aria-expanded="false"></button>'
      + '<div class="parallel-models-menu" role="group" aria-label="Models with a panel" hidden></div></div>'
      + '<button type="button" class="parallel-btn parallel-stop-all">Stop all</button>'
      + '<button type="button" class="parallel-btn parallel-close">Back to chat</button></div></div>'
      + '<div class="parallel-grid"></div>'
      + '<div class="parallel-shared" hidden>'
      + '<div class="parallel-attach parallel-shared-attach"></div>'
      + '<div class="parallel-composer">'
      + '<textarea rows="2" placeholder="One prompt for every panel" aria-label="Prompt for every panel"></textarea>'
      + '<div class="parallel-composer-row">'
      + '<button type="button" class="parallel-btn parallel-picture">Picture</button>'
      + '<button type="button" class="parallel-btn parallel-cam">Camera</button>'
      + '<span class="parallel-shared-note"></span><span class="parallel-spacer"></span>'
      + '<button type="button" class="parallel-btn parallel-send parallel-send-all">Send to all</button></div></div></div>'
      + '<input type="file" accept="image/*" class="parallel-file" hidden>';
    view.querySelector('.parallel-close').addEventListener('click', () => setOpen(false));
    const menuBtn = view.querySelector('.parallel-models-btn');
    const menu = view.querySelector('.parallel-models-menu');
    const setMenu = (on) => { menu.hidden = !on; menuBtn.setAttribute('aria-expanded', on ? 'true' : 'false'); };
    menuBtn.addEventListener('click', () => setMenu(menu.hidden));
    document.addEventListener('mousedown', (e) => {
      if (!menu.hidden && !e.target.closest('.parallel-models-pick')) setMenu(false);
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !menu.hidden) setMenu(false); });
    menu.addEventListener('change', (e) => {
      const box = e.target.closest('input[type=checkbox]');
      if (!box) return;
      const names = loaded().map((m) => m.name);
      const set = new Set(chosen || names);
      if (box.checked) set.add(box.value);
      else set.delete(box.value);
      if (!set.size) { box.checked = true; return; }   // keep at least one panel
      chosen = names.filter((n) => set.has(n));
      renderPanels(true);
    });
    view.querySelector('.parallel-stop-all').addEventListener('click', () => {
      panels.forEach((p) => p.controller && p.controller.abort());
    });
    view.querySelector('.parallel-file').addEventListener('change', onFileChosen);
    view.querySelector('.parallel-mode-select').addEventListener('change', (e) => {
      promptMode = e.target.value === 'same' ? 'same' : 'different';
      syncMode();
    });
    const shared = view.querySelector('.parallel-shared');
    const sharedTa = shared.querySelector('textarea');
    sharedTa.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendShared(); }
    });
    sharedTa.addEventListener('input', syncShared);
    shared.querySelector('.parallel-send-all').addEventListener('click', () => {
      if (anyBusy()) panels.forEach((p) => p.controller && p.controller.abort());
      else sendShared();
    });
    shared.querySelector('.parallel-picture').addEventListener('click', () => {
      fileTarget = SHARED;
      view.querySelector('.parallel-file').click();
    });
    shared.querySelector('.parallel-cam').addEventListener('click', () => openCamera(SHARED));
    syncMode();
    document.body.appendChild(view);
    window.addEventListener('resize', () => { if (open) placeView(); });
    document.addEventListener('keydown', (e) => { if (open && e.key === 'Escape' && !view.querySelector('.parallel-camera')) setOpen(false); });
  }

  let fileTarget = null;
  async function onFileChosen(e) {
    const file = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!file || !fileTarget) return;
    try {
      const image = await scaleImage(file);
      if (fileTarget === SHARED) { sharedImage = image; renderSharedAttachment(); return; }
      panelState(fileTarget).image = image;
      renderPanels(true);
    } catch (err) {
      if (fileTarget !== SHARED) alertInPanel(fileTarget, 'That file could not be read as an image.');
    }
  }

  async function scaleImage(blob, maxSide = 896) {
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    if (bitmap.close) bitmap.close();
    return canvas.toDataURL('image/jpeg', 0.9);
  }

  // Rebuild the panels when the loaded models or the ticked models change;
  // otherwise keep each panel (and anything typed in it) as it is. Models not
  // shown keep their conversations for when they are ticked again.
  function renderPanels(force) {
    if (!view) return;
    const models = loaded();
    const names = models.map((m) => m.name);
    let shown = chosen ? chosen.filter((n) => names.includes(n)) : names;
    if (!shown.length) { chosen = null; shown = names; }
    view.querySelector('.parallel-count').textContent =
      `${shown.length} of ${names.length} loaded model${names.length === 1 ? '' : 's'}`;
    view.querySelector('.parallel-models-btn').textContent = `Models · ${shown.length} ▾`;
    const key = `${names.join('|')}#${shown.join('|')}`;
    if (!force && key === shownModels) return;
    shownModels = key;
    const menu = view.querySelector('.parallel-models-menu');
    menu.innerHTML = models.map((m) => {
      const on = shown.includes(m.name);
      const last = on && shown.length === 1;
      return `<label class="parallel-models-option${last ? ' is-locked' : ''}">`
        + `<input type="checkbox" value="${esc(m.name)}"${on ? ' checked' : ''}${last ? ' disabled' : ''}>`
        + `<span>${esc(m.name)}</span>`
        + `<span class="parallel-badge${isVision(m) ? ' is-vlm' : ''}">${isVision(m) ? 'Sees images' : 'Text only'}</span></label>`;
    }).join('') + '<div class="parallel-models-hint">Ticked models get a panel.</div>';
    const grid = view.querySelector('.parallel-grid');
    grid.style.setProperty('--panels', String(Math.max(1, shown.length)));
    grid.innerHTML = '';
    shown.forEach((n) => grid.appendChild(buildPanel(models.find((m) => m.name === n))));
    requestAnimationFrame(syncShared);
    if (open && names.length < 2) setOpen(false);
  }

  function buildPanel(model) {
    const state = panelState(model.name);
    const vision = isVision(model);
    const el = document.createElement('div');
    el.className = 'parallel-panel';
    el.dataset.model = model.name;
    el.innerHTML = '<div class="parallel-panel-head">'
      + `<span class="parallel-panel-name" title="${esc(model.name)}">${esc(model.name)}</span>`
      + `<span class="parallel-badge${vision ? ' is-vlm' : ''}">${vision ? 'Sees images' : 'Text only'}</span>`
      + '<button type="button" class="parallel-link parallel-new">New chat</button></div>'
      + '<div class="parallel-thread"></div>'
      + '<div class="parallel-attach"></div>'
      + '<div class="parallel-composer">'
      + `<textarea rows="2" placeholder="Message ${esc(model.name)}" aria-label="Message ${esc(model.name)}"></textarea>`
      + '<div class="parallel-composer-row">'
      + (vision ? '<button type="button" class="parallel-btn parallel-picture">Picture</button>'
        + '<button type="button" class="parallel-btn parallel-cam">Camera</button>' : '')
      + '<span class="parallel-spacer"></span>'
      + '<button type="button" class="parallel-btn parallel-send">Send</button></div></div>';
    const ta = el.querySelector('textarea');
    ta.value = drafts.get(model.name) || '';
    ta.addEventListener('input', () => drafts.set(model.name, ta.value));
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(model); }
    });
    el.querySelector('.parallel-send').addEventListener('click', () => {
      if (state.controller) state.controller.abort();
      else send(model);
    });
    el.querySelector('.parallel-new').addEventListener('click', () => {
      if (state.controller) state.controller.abort();
      state.messages = [];
      renderThread(model.name);
    });
    if (vision) {
      el.querySelector('.parallel-picture').addEventListener('click', () => {
        fileTarget = model.name;
        view.querySelector('.parallel-file').click();
      });
      el.querySelector('.parallel-cam').addEventListener('click', () => openCamera(model.name));
    }
    // Fill the thread after the panel is in the DOM.
    requestAnimationFrame(() => { renderThread(model.name); renderAttachment(model.name); syncSend(model.name); });
    return el;
  }

  const panelEl = (name) => (view ? Array.from(view.querySelectorAll('.parallel-panel')).find((p) => p.dataset.model === name) : null);

  function syncSend(name) {
    const el = panelEl(name);
    if (!el) return;
    const busy = !!panelState(name).controller;
    const btn = el.querySelector('.parallel-send');
    btn.textContent = busy ? 'Stop' : 'Send';
    btn.classList.toggle('is-stop', busy);
  }

  function renderAttachment(name) {
    const el = panelEl(name);
    if (!el) return;
    const box = el.querySelector('.parallel-attach');
    const image = panelState(name).image;
    box.innerHTML = image ? `<img src="${image}" alt="Picture for the next message"><button type="button" class="parallel-link">Remove</button>` : '';
    if (image) box.querySelector('button').addEventListener('click', () => { panelState(name).image = null; renderAttachment(name); });
  }

  function alertInPanel(name, text) {
    const el = panelEl(name);
    if (!el) return;
    const note = document.createElement('div');
    note.className = 'parallel-note is-error';
    note.textContent = text;
    el.querySelector('.parallel-thread').appendChild(note);
  }

  // Draw a panel's conversation. Streaming replies update their own bubble.
  function renderThread(name) {
    const el = panelEl(name);
    if (!el) return;
    const thread = el.querySelector('.parallel-thread');
    const state = panelState(name);
    thread.innerHTML = '';
    if (!state.messages.length) {
      thread.innerHTML = '<div class="parallel-empty">Type a message and press Send. This panel keeps its own conversation.</div>';
      return;
    }
    state.messages.forEach((m) => thread.appendChild(messageEl(m)));
    thread.scrollTop = thread.scrollHeight;
  }

  function messageEl(m) {
    const div = document.createElement('div');
    if (m.role === 'note') {
      div.className = 'parallel-note';
      div.textContent = m.text;
      return div;
    }
    if (m.role === 'user') {
      div.className = 'parallel-msg is-user';
      div.innerHTML = (m.image ? `<img src="${m.image}" alt="Sent with this message">` : '') + `<div>${esc(m.text)}</div>`;
      return div;
    }
    div.className = 'parallel-msg is-assistant';
    div.innerHTML = '<div class="parallel-body message-text"></div><div class="parallel-foot"></div>';
    m.bodyEl = div.querySelector('.parallel-body');
    m.footEl = div.querySelector('.parallel-foot');
    paintReply(m, true);
    return div;
  }

  function paintReply(m, final) {
    if (!m.bodyEl) return;
    const answer = answerOf(m.text);
    if (final || typeof setMarkdownThrottled !== 'function') {
      if (typeof cancelPendingRender === 'function') cancelPendingRender(m.bodyEl);
      if (typeof renderMarkdownInto === 'function') renderMarkdownInto(m.bodyEl, answer || (m.pending ? '…' : ''));
      else m.bodyEl.textContent = answer;
    } else {
      setMarkdownThrottled(m.bodyEl, answer);
    }
    const bits = [];
    const stats = statsText(m);
    if (stats) bits.push(`<span class="parallel-stats">${stats}</span>`);
    if (m.state) bits.push(`<span class="parallel-state is-${m.stateKind || 'busy'}">${esc(m.state)}</span>`);
    m.footEl.innerHTML = bits.join('');
  }

  // ---- sending ----

  async function send(model, given) {
    const name = model.name;
    const el = panelEl(name);
    const state = panelState(name);
    if (!el || state.controller) return;
    const ta = el.querySelector('textarea');
    const vision = isVision(model);
    const image = vision ? (given ? given.image : state.image) : null;
    const text = (given ? given.text : ta.value.trim()) || (image ? 'Describe this image.' : '');
    if (!text) return;
    if (!loaded().some((m) => m.name === name)) {
      alertInPanel(name, `${name} is no longer loaded. Load it again in Settings.`);
      return;
    }
    const history = state.messages
      .filter((m) => m.role === 'user' || (m.role === 'assistant' && !m.failed && answerOf(m.text).trim()))
      .map((m) => ({ role: m.role, content: m.role === 'assistant' ? answerOf(m.text) : m.text }));
    const content = image
      ? [{ type: 'text', text }, { type: 'image_url', image_url: { url: image } }]
      : text;
    const messages = [{ role: 'system', content: SYSTEM_PROMPT + (image ? WITH_IMAGE : WITHOUT_IMAGE) }, ...history, { role: 'user', content }];

    if (!given) {
      ta.value = '';
      drafts.delete(name);
      state.image = null;
      renderAttachment(name);
    }
    const reply = { role: 'assistant', text: '', tokens: 0, tps: null, ttftS: null, totalS: null, pending: true, state: 'Waiting for the first token…', stateKind: 'busy' };
    state.messages.push({ role: 'user', text, image }, reply);
    renderThread(name);
    const controller = new AbortController();
    state.controller = controller;
    syncSend(name);

    const t0 = performance.now();
    for (let attempt = 0; ; attempt += 1) {
      try {
        await stream(name, messages, reply, controller, t0);
        reply.pending = false;
        reply.totalS = (performance.now() - t0) / 1000;
        if (!answerOf(reply.text).trim()) {
          reply.failed = true;
          reply.state = `No answer: this conversation is probably longer than ${name} can read at once. Press New chat.`;
          reply.stateKind = 'error';
        } else {
          reply.state = '';
        }
        break;
      } catch (err) {
        const stopped = err && err.name === 'AbortError';
        if (!stopped && attempt === 0 && acceleratorBusy(err && err.message)) {
          reply.text = ''; reply.tokens = 0; reply.ttftS = null;
          reply.state = 'The accelerator was busy; trying again…';
          paintReply(reply, true);
          await new Promise((r) => setTimeout(r, 400));
          if (!controller.signal.aborted) continue;
        }
        reply.pending = false;
        reply.totalS = (performance.now() - t0) / 1000;
        reply.failed = !stopped;
        reply.state = stopped ? 'Stopped' : `Failed: ${err && err.message ? err.message : err}`;
        reply.stateKind = stopped ? 'muted' : 'error';
        break;
      }
    }
    state.controller = null;
    paintReply(reply, true);
    syncSend(name);
    const thread = panelEl(name) && panelEl(name).querySelector('.parallel-thread');
    if (thread) thread.scrollTop = thread.scrollHeight;
  }

  async function stream(name, messages, reply, controller, t0) {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: name, messages, stream: true }),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) {
      const detail = await resp.text().catch(() => '');
      let message = `HTTP ${resp.status}`;
      try { const j = JSON.parse(detail); message = (j.error && (j.error.message || j.error)) || message; } catch (e) { /* not JSON */ }
      throw new Error(String(message));
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let tFirst = null;
    let tLast = null;
    let serverTps = null;
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') {
          if (serverTps != null) reply.tps = serverTps;
          else if (tFirst != null && tLast > tFirst && reply.tokens > 1) reply.tps = (reply.tokens - 1) / ((tLast - tFirst) / 1000);
          return;
        }
        let obj;
        try { obj = JSON.parse(data); } catch (e) { continue; }
        if (obj.error) throw new Error(String(obj.error.message || obj.error));
        if (obj.tps != null && Number.isFinite(Number(obj.tps))) serverTps = Number(obj.tps);
        if (obj.generated_tokens != null && Number.isFinite(Number(obj.generated_tokens))) reply.tokens = Number(obj.generated_tokens);
        const delta = obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content;
        if (!delta) continue;
        const now = performance.now();
        if (tFirst == null) { tFirst = now; reply.ttftS = (now - t0) / 1000; }
        tLast = now;
        reply.tokens += 1;
        reply.text += delta;
        const parts = typeof splitThinking === 'function' ? splitThinking(reply.text) : { present: false };
        reply.state = parts.present && !parts.closed ? 'Thinking…' : 'Generating…';
        if (serverTps != null) reply.tps = serverTps;
        paintReply(reply, false);
        const thread = panelEl(name) && panelEl(name).querySelector('.parallel-thread');
        if (thread) thread.scrollTop = thread.scrollHeight;
      }
    }
    throw new Error('The reply stopped early: the connection closed before it finished.');
  }

  // ---- same prompt for every panel ----

  const shownNames = () => Array.from(view.querySelectorAll('.parallel-panel')).map((p) => p.dataset.model);
  const anyBusy = () => shownNames().some((n) => panelState(n).controller);

  function syncMode() {
    if (!view) return;
    const same = promptMode === 'same';
    view.classList.toggle('is-same', same);
    view.querySelector('.parallel-shared').hidden = !same;
    view.querySelector('.parallel-mode-select').value = promptMode;
    view.querySelector('.parallel-sub').textContent = same
      ? 'One prompt goes to every panel at once; each panel keeps its own conversation.'
      : 'Each model has its own prompt and Send. They run at the same time on the accelerator.';
    syncShared();
  }

  function renderSharedAttachment() {
    const box = view.querySelector('.parallel-shared-attach');
    box.innerHTML = sharedImage ? `<img src="${sharedImage}" alt="Picture for the next message"><button type="button" class="parallel-link">Remove</button>` : '';
    if (sharedImage) box.querySelector('button').addEventListener('click', () => { sharedImage = null; renderSharedAttachment(); });
    syncShared();
  }

  // The shared box's buttons and its note about pictures and text-only models.
  function syncShared() {
    if (!view) return;
    const shared = view.querySelector('.parallel-shared');
    const models = loaded().filter((m) => shownNames().includes(m.name));
    const seeing = models.filter(isVision);
    shared.querySelector('.parallel-picture').hidden = !seeing.length;
    shared.querySelector('.parallel-cam').hidden = !seeing.length;
    const busy = anyBusy();
    const btn = shared.querySelector('.parallel-send-all');
    btn.textContent = busy ? 'Stop all' : `Send to ${models.length}`;
    btn.classList.toggle('is-stop', busy);
    const blind = models.filter((m) => !isVision(m));
    shared.querySelector('.parallel-shared-note').textContent = sharedImage && blind.length
      ? `${blind.map((m) => m.name).join(' and ')} can't take images, so only ${seeing.map((m) => m.name).join(' and ')} will answer.`
      : '';
  }

  // Send the shared prompt to every shown panel at once. A model that can't
  // see the attached picture sits that message out, with a note in its panel.
  function sendShared() {
    const shared = view.querySelector('.parallel-shared');
    const ta = shared.querySelector('textarea');
    const image = sharedImage;
    const text = ta.value.trim() || (image ? 'Describe this image.' : '');
    if (!text || anyBusy()) return;
    const models = loaded().filter((m) => shownNames().includes(m.name));
    ta.value = '';
    sharedImage = null;
    renderSharedAttachment();
    models.forEach((m) => {
      if (image && !isVision(m)) {
        panelState(m.name).messages.push({ role: 'note', text: `“${text}” had a picture: ${m.name} can't take images, so it didn't answer.` });
        renderThread(m.name);
        return;
      }
      send(m, { text, image: isVision(m) ? image : null }).finally(syncShared);
    });
    setTimeout(syncShared, 50);
  }

  // ---- camera (one at a time) ----

  let camStream = null;
  async function openCamera(name) {
    const el = name === SHARED ? view.querySelector('.parallel-shared') : panelEl(name);
    if (!el || view.querySelector('.parallel-camera')) return;
    try {
      camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    } catch (err) {
      if (name !== SHARED) alertInPanel(name, `The camera isn't available (${err.message}). Allow camera access for this page, then try again.`);
      else view.querySelector('.parallel-shared-note').textContent = "The camera isn't available: allow camera access for this page.";
      return;
    }
    const box = document.createElement('div');
    box.className = 'parallel-camera';
    box.innerHTML = '<video autoplay playsinline muted></video><div class="parallel-composer-row">'
      + '<button type="button" class="parallel-btn parallel-snap">Use this picture</button>'
      + '<button type="button" class="parallel-link parallel-cancel">Cancel</button></div>';
    const video = box.querySelector('video');
    video.srcObject = camStream;
    const close = () => { if (camStream) camStream.getTracks().forEach((t) => t.stop()); camStream = null; box.remove(); };
    box.querySelector('.parallel-cancel').addEventListener('click', close);
    box.querySelector('.parallel-snap').addEventListener('click', () => {
      if (!video.videoWidth) return;
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, 896 / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      const image = canvas.toDataURL('image/jpeg', 0.9);
      close();
      if (name === SHARED) { sharedImage = image; renderSharedAttachment(); return; }
      panelState(name).image = image;
      renderAttachment(name);
    });
    el.querySelector('.parallel-attach').before(box);
  }

  // Follow the loaded models: the button appears with two or more, and the
  // panels change when a model is loaded or unloaded.
  function tick() {
    syncButton();
    if (open) { renderPanels(false); syncShared(); }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', tick);
  else tick();
  setInterval(tick, 1500);
})();
