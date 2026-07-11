// SiMaSentry Mission Control — air-gapped, zero-dependency portal.
// All state in DOM + localStorage. The single outbound network use is the
// iframe loading a harness page from the same origin; the portal itself
// makes no fetch() calls.
'use strict';

(function () {
  const CONFIG_KEY = 'sima-sentry:config';

  // Public mode names → folder paths. Decouples brand naming from disk layout.
  const MODE_ROUTES = {
    medical:  { folder: 'health',   label: 'SiMaSentry-Med',  accent: 'medical' },
    safety:   { folder: 'safety',   label: 'SiMaSentry-Safe', accent: 'safety' },
    security: { folder: 'security', label: 'SiMaSentry-Sec',  accent: 'security' },
  };

  // Legacy hash routes from the previous launcher version.
  const LEGACY_HASH = { health: 'medical', safety: 'safety', security: 'security' };

  const DEFAULTS = {
    provider: 'ollama',
    baseUrl:  'http://localhost:9998/v1/chat/completions',
    apiKey:   '',
    model:    'gemma3',
  };

  const $ = (id) => document.getElementById(id);

  const dom = {
    body: document.body,
    cards: document.querySelectorAll('.mc-card'),
    endpoint: $('mc-endpoint'),
    model: $('mc-model'),
    provider: $('mc-provider'),
    openConfigTriggers: document.querySelectorAll('[data-action="open-config"]'),
    netPill: $('mc-net-pill'),
    netLabel: $('mc-net-label'),

    // Drawer
    configBtn: $('config-btn'),
    drawer: $('config-drawer'),
    scrim: $('config-scrim'),
    drawerClose: $('config-close'),
    form: $('config-form'),
    cfgBaseUrl: $('cfg-base-url'),
    cfgModel: $('cfg-model'),
    cfgApiKey: $('cfg-api-key'),
    cfgApiKeyRow: $('cfg-api-key-row'),
    providerInputs: document.querySelectorAll('input[name="provider"]'),
    saveBtn: $('config-save'),
    clearBtn: $('config-clear'),
    message: $('config-message'),

    // Frame
    frameView: $('frame-view'),
    frame: $('harness-frame'),
    frameBack: $('frame-back-btn'),
    frameSwitches: document.querySelectorAll('.frame-switch'),
  };

  // ── Storage helpers (private-mode safe) ─────────────────
  function safeGet(key) {
    try { return window.localStorage.getItem(key); }
    catch { return null; }
  }
  function safeSet(key, value) {
    try { window.localStorage.setItem(key, value); return true; }
    catch { return false; }
  }
  function safeRemove(key) {
    try { window.localStorage.removeItem(key); }
    catch { /* ignore */ }
  }
  function loadConfig() {
    const raw = safeGet(CONFIG_KEY);
    if (!raw) return { ...DEFAULTS };
    try {
      const obj = JSON.parse(raw);
      return {
        provider: obj.provider === 'openai' ? 'openai' : 'ollama',
        baseUrl:  typeof obj.baseUrl === 'string' ? obj.baseUrl : DEFAULTS.baseUrl,
        model:    typeof obj.model === 'string'   ? obj.model   : DEFAULTS.model,
        apiKey:   typeof obj.apiKey === 'string'  ? obj.apiKey  : '',
      };
    } catch {
      return { ...DEFAULTS };
    }
  }
  function persistConfig(cfg) {
    safeSet(CONFIG_KEY, JSON.stringify(cfg));
  }

  // ── Drawer ──────────────────────────────────────────────
  let lastFocused = null;

  function openDrawer() {
    const cfg = loadConfig();
    fillDrawerForm(cfg);
    dom.drawer.hidden = false;
    dom.scrim.hidden = false;
    // Use rAF so the transform animation runs even when going from hidden.
    requestAnimationFrame(() => dom.body.classList.add('drawer-open'));
    lastFocused = document.activeElement;
    setTimeout(() => dom.cfgBaseUrl.focus(), 60);
  }

  function closeDrawer() {
    dom.body.classList.remove('drawer-open');
    setTimeout(() => {
      dom.drawer.hidden = true;
      dom.scrim.hidden = true;
    }, 220);
    if (lastFocused && typeof lastFocused.focus === 'function') {
      lastFocused.focus();
    }
  }

  function fillDrawerForm(cfg) {
    for (const radio of dom.providerInputs) {
      radio.checked = radio.value === cfg.provider;
    }
    dom.cfgBaseUrl.value = cfg.baseUrl;
    dom.cfgModel.value = cfg.model;
    dom.cfgApiKey.value = cfg.apiKey || '';
    syncProviderUI();
    hideMessage();
  }

  function selectedProvider() {
    for (const r of dom.providerInputs) if (r.checked) return r.value;
    return DEFAULTS.provider;
  }

  function syncProviderUI() {
    dom.cfgApiKeyRow.hidden = selectedProvider() !== 'openai';
  }

  function readDrawer() {
    return {
      provider: selectedProvider(),
      baseUrl:  dom.cfgBaseUrl.value.trim() || DEFAULTS.baseUrl,
      model:    dom.cfgModel.value.trim()   || DEFAULTS.model,
      apiKey:   dom.cfgApiKey.value.trim(),
    };
  }

  function showMessage(tone, text) {
    dom.message.hidden = false;
    dom.message.dataset.tone = tone;
    dom.message.textContent = text;
  }
  function hideMessage() {
    dom.message.hidden = true;
    delete dom.message.dataset.tone;
  }

  function onSave(event) {
    event.preventDefault();
    const cfg = readDrawer();
    persistConfig(cfg);
    paintStatus(cfg);
    showMessage('ok',
      `Saved. Each card will launch its harness with this base URL ` +
      `pre-applied via URL parameters.`);
  }

  function onClear() {
    safeRemove(CONFIG_KEY);
    fillDrawerForm({ ...DEFAULTS });
    paintStatus({ ...DEFAULTS });
    showMessage('warn', 'Cleared. Cards will launch with shipped defaults.');
  }

  // ── Network status pill ─────────────────────────────────
  // Browser-level reachability — uses navigator.onLine and the corresponding
  // events. This reflects whether the device has *any* network interface up
  // (which covers an air-gapped LAN to the Modalix box). It does not probe
  // the llima endpoint specifically.
  function paintNetStatus() {
    if (!dom.netPill || !dom.netLabel) return;
    const online = navigator.onLine !== false;
    dom.netPill.dataset.state = online ? 'online' : 'offline';
    dom.netLabel.textContent = online ? 'Online' : 'Offline';
  }

  // ── System Status (static, derived from saved config) ───
  function paintStatus(cfg) {
    if (dom.endpoint) {
      dom.endpoint.textContent = cfg.baseUrl;
      dom.endpoint.title = cfg.baseUrl;
    }
    if (dom.model) {
      dom.model.textContent = cfg.model;
      dom.model.title = cfg.model;
    }
    if (dom.provider) dom.provider.textContent = cfg.provider === 'openai' ? 'OpenAI' : 'Ollama / llima';
  }

  // ── Mode routing ────────────────────────────────────────
  function modeFromLocation() {
    const params = new URLSearchParams(window.location.search);
    const m = params.get('mode');
    if (m && MODE_ROUTES[m]) return m;
    // Soft fallback for old #hash links from the previous launcher.
    const hash = (window.location.hash || '').replace(/^#/, '');
    if (LEGACY_HASH[hash]) return LEGACY_HASH[hash];
    return null;
  }

  function buildHarnessUrl(mode) {
    const route = MODE_ROUTES[mode];
    if (!route) return null;
    const cfg = loadConfig();
    const params = new URLSearchParams();
    if (cfg.provider) params.set('provider', cfg.provider);
    if (cfg.baseUrl)  params.set('base_url', cfg.baseUrl);
    if (cfg.model)    params.set('model',    cfg.model);
    if (cfg.provider === 'openai' && cfg.apiKey) params.set('api_key', cfg.apiKey);
    const qs = params.toString();
    return `${route.folder}/index.html${qs ? `?${qs}` : ''}`;
  }

  function openMode(mode, { pushHistory = true } = {}) {
    const route = MODE_ROUTES[mode];
    if (!route) return;
    const url = buildHarnessUrl(mode);
    dom.body.classList.add('in-frame');
    dom.body.classList.remove('is-returning');
    dom.frameView.hidden = false;
    dom.frame.title = route.label;

    if (dom.frame.dataset.mode !== mode) {
      const switching = !!dom.frame.dataset.mode;
      if (switching) {
        // Cross-fade: dim the current iframe, swap src, fade back in on load.
        dom.frame.style.opacity = '0';
        setTimeout(() => {
          dom.frame.src = url;
          dom.frame.dataset.mode = mode;
        }, 200);
      } else {
        // First entry: the .mc-frame container animates in; the iframe itself
        // starts visible.
        dom.frame.style.opacity = '1';
        dom.frame.src = url;
        dom.frame.dataset.mode = mode;
      }
    }

    for (const btn of dom.frameSwitches) {
      const active = btn.dataset.target === mode;
      if (active) btn.setAttribute('aria-current', 'page');
      else btn.removeAttribute('aria-current');
    }
    if (pushHistory) {
      const target = `?mode=${encodeURIComponent(mode)}`;
      if (window.location.search !== `?mode=${mode}`) {
        history.pushState({ mode }, '', target);
      }
    }
    if (dom.frameBack) dom.frameBack.focus({ preventScroll: true });
  }

  function closeFrame({ pushHistory = true } = {}) {
    dom.body.classList.remove('in-frame');
    dom.frameView.hidden = true;
    // One-shot class to re-trigger the landing-page entrance animation;
    // cleared after the keyframe completes.
    dom.body.classList.add('is-returning');
    setTimeout(() => dom.body.classList.remove('is-returning'), 320);
    // Free the iframe so background fetches / camera streams stop.
    dom.frame.removeAttribute('src');
    dom.frame.style.opacity = '';
    delete dom.frame.dataset.mode;
    for (const btn of dom.frameSwitches) btn.removeAttribute('aria-current');
    if (pushHistory && window.location.search) {
      history.pushState({ mode: null }, '', window.location.pathname);
    }
  }

  function onCardClick(event) {
    if (event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const card = event.currentTarget;
    const mode = card.dataset.mode;
    if (!mode || !MODE_ROUTES[mode]) return;
    event.preventDefault();
    openMode(mode);
  }

  function onPopState() {
    const mode = modeFromLocation();
    if (mode) openMode(mode, { pushHistory: false });
    else closeFrame({ pushHistory: false });
  }

  // ── Home button postMessage from harnesses ─────────────
  function onMessage(event) {
    // Only trust messages coming from our own iframe's window.
    if (!dom.frame || event.source !== dom.frame.contentWindow) return;
    const data = event.data;
    if (!data || typeof data !== 'object') return;
    if (data.type === 'sima-sentry:home') {
      closeFrame();
    }
  }

  // ── Init ────────────────────────────────────────────────
  function init() {
    const cfg = loadConfig();
    paintStatus(cfg);
    paintNetStatus();
    window.addEventListener('online', paintNetStatus);
    window.addEventListener('offline', paintNetStatus);

    // Drawer wiring
    dom.configBtn.addEventListener('click', openDrawer);
    for (const trigger of dom.openConfigTriggers) {
      trigger.addEventListener('click', openDrawer);
    }
    dom.drawerClose.addEventListener('click', closeDrawer);
    dom.scrim.addEventListener('click', closeDrawer);
    dom.form.addEventListener('submit', onSave);
    dom.clearBtn.addEventListener('click', onClear);
    for (const r of dom.providerInputs) r.addEventListener('change', syncProviderUI);
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && dom.body.classList.contains('drawer-open')) {
        closeDrawer();
      }
    });

    // Card + frame wiring
    for (const card of dom.cards) card.addEventListener('click', onCardClick);
    for (const btn of dom.frameSwitches) {
      btn.addEventListener('click', () => openMode(btn.dataset.target));
    }
    dom.frameBack.addEventListener('click', () => closeFrame());
    // Fade the iframe back in once the new harness has loaded — this
    // completes the cross-fade started in openMode().
    dom.frame.addEventListener('load', () => {
      if (dom.frame.dataset.mode) dom.frame.style.opacity = '1';
    });
    window.addEventListener('popstate', onPopState);
    window.addEventListener('message', onMessage);

    // Initial routing.
    const initial = modeFromLocation();
    if (initial) {
      // If the user landed via legacy hash, normalise the URL to ?mode=.
      if (window.location.hash) {
        history.replaceState({ mode: initial }, '', `?mode=${initial}`);
      }
      openMode(initial, { pushHistory: false });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
