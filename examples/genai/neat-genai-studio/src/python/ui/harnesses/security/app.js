// SiMaSentry-Sec — air-gapped, zero-dependency client.
// All state in instance fields, single class, no globals beyond window.__security.
'use strict';

const DEFAULT_SYSTEM_PROMPT = `You are a Senior Security Analyst. Monitor surveillance for unauthorized access, weapons, and suspicious behavior. Provide objective assessments.

Begin every assessment with a single line:
THREAT: <INFO|LOW|MEDIUM|HIGH|CRITICAL>
followed by your analysis and, where relevant, a short bulleted list of recommended actions. Keep the THREAT: line itself in English so the console can color-code the alert.`;

const THREAT_ANALYSIS_PROMPT = 'Analyze this specific object. Is it a weapon, an unauthorized person in a restricted area, or a suspicious abandoned package? Assess the threat level (Low/Medium/High).';

const CHANGE_DETECTION_SYSTEM_PROMPT = `You are an expert in Secure Facility Change Detection. You compare two photographs of the same area — a BASELINE reference and a CURRENT feed — and report visible differences between them. You are NOT a threat-assessment tool: do not rate severity, do not output "THREAT:" lines, do not call anything "critical" or "low". Just describe what changed.

You will receive two images, in this exact order, each preceded by an explicit label:
  • IMAGE A — BASELINE REFERENCE (the prior, known-good photograph)
  • IMAGE B — CURRENT FEED (the new photograph to compare against the baseline)

All bounding-box coordinates you report describe locations on IMAGE B (the Current Feed) only.

Output a discrepancy list — one per line — in this EXACT single-line format:
[CATEGORY] (x, y, w, h) — short description

Where:
  • CATEGORY is one of NEW (something present in B but not in A), MISSING (something present in A but not in B), or STRUCTURAL (something altered between A and B, e.g. an open door, broken window, moved furniture).
  • x = horizontal position of the LEFT edge of the box, as a fraction of IMAGE B's width (0.0 = left edge, 1.0 = right edge).
  • y = vertical position of the TOP edge of the box, as a fraction of IMAGE B's height (0.0 = top edge, 1.0 = bottom edge).
  • w = box width, as a fraction of IMAGE B's width (1.0 = full image width).
  • h = box height, as a fraction of IMAGE B's height (1.0 = full image height).

Concrete example (do not copy the values, only the format):
[NEW] (0.42, 0.30, 0.18, 0.22) — Cardboard box on the loading-dock floor

If you cannot determine a precise bounding box for an item, omit the coordinates entirely and write:
[CATEGORY] — short description

After the list, write a single paragraph summarising the differences. If the two frames look identical, say so explicitly and emit zero discrepancy lines.`;

const CHANGE_DETECTION_USER_PROMPT = 'Compare IMAGE A (BASELINE) against IMAGE B (CURRENT FEED). List every visible difference. Use the bounding-box format defined in your instructions; coordinates describe locations on IMAGE B (Current Feed) only. Do not include severity labels or "THREAT" lines — this is a descriptive audit, not a threat assessment.';

// Discrepancy bbox stroke colours, keyed by category.
const DISCREPANCY_COLOURS = {
  new: '#f85149',         // red — same hue as --threat-critical
  missing: '#d29922',     // yellow — --threat-medium
  structural: '#f0883e',  // orange — --threat-high
};

// Line-by-line regex extractors for `[CATEGORY] (x, y, w, h) — text` and the
// no-bbox variant. The model occasionally drops the brackets, so they're optional.
const RX_DISCREPANCY_BBOX = /^\s*[-*•]?\s*\[?(NEW|MISSING|STRUCTURAL)\]?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*[—–\-:]\s*(.+?)\s*$/i;
const RX_DISCREPANCY_NOBBOX = /^\s*[-*•]?\s*\[?(NEW|MISSING|STRUCTURAL)\]?\s*[—–\-:]\s*(.+?)\s*$/i;

// Maps the BCP-47 codes from the language dropdown to natural-language names
// the model will actually reason about. Without an explicit name, the model
// often ignores the locale tag and picks whatever it inferred from the prompt.
const LANGUAGE_NAMES = {
  'en-US': 'English (United States)',
  'en-GB': 'English (United Kingdom)',
  'es-ES': 'Spanish (Spain)',
  'es-MX': 'Spanish (Mexico)',
  'pt-BR': 'Brazilian Portuguese',
  'fr-FR': 'French',
  'de-DE': 'German',
  'it-IT': 'Italian',
  'nl-NL': 'Dutch',
  'ru-RU': 'Russian',
  'ar-SA': 'Arabic',
  'hi-IN': 'Hindi',
  'zh-CN': 'Simplified Chinese',
  'zh-TW': 'Traditional Chinese',
  'ja-JP': 'Japanese',
  'ko-KR': 'Korean',
};

const PROVIDER_DEFAULTS = {
  ollama: {
    baseUrl: 'http://localhost:11434/v1/chat/completions',
    model: 'gemma3',
  },
  openai: {
    baseUrl: 'https://api.openai.com/v1/chat/completions',
    model: 'gpt-4o',
  },
};

const DEFAULT_SETTINGS = Object.freeze({
  provider: 'ollama',
  baseUrl: PROVIDER_DEFAULTS.ollama.baseUrl,
  apiKey: '',
  model: PROVIDER_DEFAULTS.ollama.model,
  language: 'en-US',
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
  autoRead: true,
  sttDisabled: false,
  highContrast: false,
});

const STORAGE_PREFIX = 'security-';
const STORAGE_KEYS = Object.fromEntries(
  Object.keys(DEFAULT_SETTINGS).map((k) => [k, STORAGE_PREFIX + k]),
);

const URL_PARAM_ALIASES = {
  provider: ['provider'],
  baseUrl: ['baseUrl', 'base_url'],
  apiKey: ['apiKey', 'api_key'],
  model: ['model'],
  language: ['language', 'lang'],
  systemPrompt: ['systemPrompt', 'system_prompt'],
  autoRead: ['autoRead', 'auto_read'],
  sttDisabled: ['sttDisabled', 'stt_disabled', 'stt'],
  highContrast: ['highContrast', 'high_contrast'],
};

const MAX_IMAGE_DIM = 1280;
const VIDEO_FRAME_SEEK_SECONDS = 0.1;
const THREAT_LEVELS = ['info', 'low', 'medium', 'high', 'critical'];

// Models occasionally leak their chat-template stop tokens into streamed
// `delta.content` (Gemma's <end_of_turn>, Llama-3's <|eot_id|>, ChatML's
// <|im_end|>, etc.). When we see one, truncate at that point and treat the
// stream as finished — anything after is template artefact.
const STOP_TOKENS = [
  '<end_of_turn>',
  '<start_of_turn>',
  '<|im_end|>',
  '<|im_start|>',
  '<|eot_id|>',
  '<|eot|>',
  '<|endoftext|>',
  '<|end_of_text|>',
  '<|end|>',
  '<|start_header_id|>',
  '<|end_header_id|>',
  '</s>',
  '<s>',
];
const STOP_TOKEN_REGEX = new RegExp(
  STOP_TOKENS.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
);
const STOP_TOKEN_MAX_LEN = Math.max(...STOP_TOKENS.map((t) => t.length));

class SecurityConsole {
  constructor(root = document) {
    this.root = root;
    this.dom = this.queryDom();
    this.settings = { ...DEFAULT_SETTINGS };
    this.history = [];
    this.attachment = null; // { dataUri, filename, width, height, source: 'image'|'frame' }
    this.recognition = null;
    this.recognitionActive = false;
    this.sttSupported = false;
    this.sttPermanentlyDisabled = false;
    this.voicesCache = [];
    this.missingVoiceWarned = false;
    this.pendingRequest = null;
    this.activeTab = 'workbench';
    this.alerts = [];
    this.alertSeq = 0;
    this.taImage = null;
    this.taBox = null;          // { xN, yN, wN, hN } finalized
    this.taDragState = null;    // { startX, startY, currentX, currentY, pointerId, started } during drag
    this.taPendingRequest = null;
    // Change Detection state
    this.cdBaselineImage = null;
    this.cdBaselineDataUri = null;
    this.cdCurrentImage = null;
    this.cdCurrentDataUri = null;
    this.cdDiscrepancies = [];
    this.cdPendingRequest = null;
  }

  queryDom() {
    const $ = (id) => this.root.getElementById(id);
    return {
      body: this.root.body,
      messages: $('messages'),
      emptyState: $('empty-state'),
      composer: $('composer'),
      composerInput: $('composer-input'),
      sendButton: $('send-button'),
      micButton: $('mic-button'),
      speakerButton: $('speaker-button'),
      stopSpeakingButton: $('stop-speaking-button'),
      clearButton: $('clear-button'),
      imageButton: $('image-button'),
      imageInput: $('image-input'),
      videoButton: $('video-button'),
      videoInput: $('video-input'),
      imagePreview: $('image-preview'),
      imagePreviewImg: $('image-preview-img'),
      imagePreviewMeta: $('image-preview-meta'),
      imageRemove: $('image-remove'),
      status: $('status'),
      drawer: $('settings-drawer'),
      drawerBackdrop: this.root.querySelector('.drawer-backdrop'),
      settingsButton: $('settings-button'),
      settingsForm: $('settings-form'),
      settingProvider: () => this.root.querySelector('input[name="provider"]:checked'),
      providerChip: $('provider-chip'),
      providerChipLabel: this.root.querySelector('.provider-chip-label'),
      settingBaseUrl: $('setting-base-url'),
      settingApiKey: $('setting-api-key'),
      apiKeyField: $('api-key-field'),
      settingModel: $('setting-model'),
      settingLanguage: $('setting-language'),
      settingSystemPrompt: $('setting-system-prompt'),
      settingAutoRead: $('setting-auto-read'),
      settingSttDisabled: $('setting-stt-disabled'),
      settingHighContrast: $('setting-high-contrast'),
      messageTemplate: $('message-template'),
      videoExtractor: $('frame-extractor-video'),

      tabButtons: Array.from(this.root.querySelectorAll('.tab-btn')),
      tabPanels: Array.from(this.root.querySelectorAll('.tab-panel')),
      mobileNavButtons: Array.from(this.root.querySelectorAll('.mobile-nav-btn')),

      alertsList: $('alerts-list'),
      alertsEmpty: $('alerts-empty'),
      alertsClear: $('alerts-clear'),
      alertTemplate: $('alert-template'),

      taImageInput: $('ta-image-input'),
      taLoadImage: $('ta-load-image'),
      taClearMark: $('ta-clear-mark'),
      taReadout: $('ta-readout'),
      taCanvas: $('ta-canvas'),
      taCanvasWrap: this.root.querySelector('#ta-mode-mark .ta-canvas-wrap'),
      taContext: $('ta-context'),
      taAnalyze: $('ta-analyze'),
      taResult: $('ta-result'),
      taResultPill: $('ta-result-pill'),
      taResultTime: $('ta-result-time'),
      taResultContent: $('ta-result-content'),
      taResultStats: $('ta-result-stats'),

      cdBaselineInput: $('cd-baseline-input'),
      cdBaselineLoad: $('cd-baseline-load'),
      cdBaselineCanvas: $('cd-baseline-canvas'),
      cdBaselineReadout: $('cd-baseline-readout'),
      cdBaselineWrap: this.root.querySelector('.cd-slot[data-slot="baseline"] .cd-canvas-wrap'),
      cdCurrentInput: $('cd-current-input'),
      cdCurrentLoad: $('cd-current-load'),
      cdCurrentCanvas: $('cd-current-canvas'),
      cdCurrentReadout: $('cd-current-readout'),
      cdCurrentWrap: this.root.querySelector('.cd-slot[data-slot="current"] .cd-canvas-wrap'),
      cdContext: $('cd-context'),
      cdAnalyze: $('cd-analyze'),
      cdResult: $('cd-result'),
      cdResultTime: $('cd-result-time'),
      cdResultContent: $('cd-result-content'),
      cdResultStats: $('cd-result-stats'),
    };
  }

  init() {
    this.loadSettings();
    this.applySettingsToForm();
    this.applyProviderUI();
    this.applyHighContrast(this.settings.highContrast);
    this.setupTabs();
    this.setupThreatAnalysis();
    this.setupChangeDetection();
    this.wireEvents();
    this.setupTTS();
    this.setupSTT();

    // ?feature=changedetection deep-links into the Change Detection tab.
    // Run after wireEvents so all listeners exist.
    const params = new URLSearchParams(window.location.search);
    if (params.get('feature') === 'changedetection') {
      this.setActiveTab('change-detection');
    }

    this.dom.composerInput.focus();
  }

  // ── Settings ──────────────────────────────────────────────────────────────

  loadSettings() {
    const fromStorage = {};
    for (const key of Object.keys(DEFAULT_SETTINGS)) {
      const stored = this.safeStorageGet(STORAGE_KEYS[key]);
      if (stored !== null) fromStorage[key] = this.coerce(key, stored);
    }
    const fromUrl = this.parseUrlParams();
    this.settings = { ...DEFAULT_SETTINGS, ...fromStorage, ...fromUrl };

    // ?mode=security applies a bundle of SOC-appropriate defaults: high-contrast
    // theme, force local Ollama, preemptively disable mic. Explicit URL params
    // for those keys still win — `?mode=security&provider=openai` is honoured.
    const params = new URLSearchParams(window.location.search);
    const modeBundleApplied = params.get('mode') === 'security';
    if (modeBundleApplied) {
      this.settings.highContrast = true;
      if (!params.has('provider')) this.settings.provider = 'ollama';
      if (!params.has('baseUrl') && !params.has('base_url')) {
        this.settings.baseUrl = PROVIDER_DEFAULTS.ollama.baseUrl;
      }
      if (!params.has('stt') && !params.has('sttDisabled') && !params.has('stt_disabled')) {
        this.settings.sttDisabled = true;
      }
    }

    // Persist resolved settings so reload-without-params keeps the kiosk preset.
    if (Object.keys(fromUrl).length > 0 || modeBundleApplied) this.saveSettings();
  }

  parseUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const out = {};
    for (const [key, aliases] of Object.entries(URL_PARAM_ALIASES)) {
      for (const alias of aliases) {
        if (params.has(alias)) {
          const raw = params.get(alias);
          // ?stt=on/off is read as the *enabled* state, so invert it for sttDisabled.
          if (key === 'sttDisabled' && alias === 'stt') {
            const v = String(raw).toLowerCase().trim();
            out[key] = !(v === '1' || v === 'true' || v === 'yes' || v === 'on');
          } else {
            out[key] = this.coerce(key, raw);
          }
          break;
        }
      }
    }
    return out;
  }

  coerce(key, raw) {
    if (key === 'autoRead' || key === 'sttDisabled' || key === 'highContrast') {
      const v = String(raw).toLowerCase().trim();
      return v === '1' || v === 'true' || v === 'yes' || v === 'on';
    }
    if (key === 'provider') {
      const v = String(raw).toLowerCase();
      return v === 'openai' ? 'openai' : 'ollama';
    }
    return String(raw);
  }

  safeStorageGet(key) {
    try { return window.localStorage.getItem(key); }
    catch { return null; }
  }
  safeStorageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch { /* private mode etc */ }
  }
  safeStorageRemove(key) {
    try { window.localStorage.removeItem(key); } catch { /* */ }
  }

  saveSettings() {
    for (const [key, value] of Object.entries(this.settings)) {
      this.safeStorageSet(STORAGE_KEYS[key], String(value));
    }
  }

  applySettingsToForm() {
    const s = this.settings;
    const providerInput = this.root.querySelector(`input[name="provider"][value="${s.provider}"]`);
    if (providerInput) providerInput.checked = true;
    this.dom.settingBaseUrl.value = s.baseUrl;
    this.dom.settingApiKey.value = s.apiKey;
    this.dom.settingModel.value = s.model;
    this.dom.settingLanguage.value = s.language;
    this.dom.settingSystemPrompt.value = s.systemPrompt;
    this.dom.settingAutoRead.checked = !!s.autoRead;
    this.dom.settingSttDisabled.checked = !!s.sttDisabled;
    this.dom.settingHighContrast.checked = !!s.highContrast;
    this.dom.speakerButton.setAttribute('aria-pressed', s.autoRead ? 'true' : 'false');
  }

  readSettingsFromForm() {
    const providerEl = this.dom.settingProvider();
    return {
      provider: providerEl ? providerEl.value : 'ollama',
      baseUrl: this.dom.settingBaseUrl.value.trim(),
      apiKey: this.dom.settingApiKey.value,
      model: this.dom.settingModel.value.trim(),
      language: this.dom.settingLanguage.value,
      systemPrompt: this.dom.settingSystemPrompt.value,
      autoRead: this.dom.settingAutoRead.checked,
      sttDisabled: this.dom.settingSttDisabled.checked,
      highContrast: this.dom.settingHighContrast.checked,
    };
  }

  applyProviderUI() {
    const provider = this.settings.provider;
    this.dom.providerChip.dataset.provider = provider;
    this.dom.providerChipLabel.textContent = provider === 'openai' ? 'OpenAI' : 'Ollama (local)';
    // Hide API key field for Ollama (not used; remains in DOM so values aren't lost).
    this.dom.apiKeyField.classList.toggle('hidden', provider === 'ollama');
    // Update placeholders so an empty field hints at the right default.
    const def = PROVIDER_DEFAULTS[provider];
    this.dom.settingBaseUrl.placeholder = def.baseUrl;
    this.dom.settingModel.placeholder = def.model;
  }

  // ── Event wiring ──────────────────────────────────────────────────────────

  wireEvents() {
    // Composer
    this.dom.composer.addEventListener('submit', (e) => {
      e.preventDefault();
      this.sendMessage();
    });
    this.dom.composerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        this.sendMessage();
      }
    });
    this.dom.composerInput.addEventListener('input', () => this.autoSizeTextarea());

    // Image attach
    this.dom.imageButton.addEventListener('click', () => this.dom.imageInput.click());
    this.dom.imageInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this.attachImageFile(file);
      e.target.value = '';
    });
    this.dom.imageRemove.addEventListener('click', () => this.clearAttachment());

    // Video frame capture
    this.dom.videoButton.addEventListener('click', () => this.dom.videoInput.click());
    this.dom.videoInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this.captureFrameFromVideoFile(file);
      e.target.value = '';
    });

    // Voice
    this.dom.micButton.addEventListener('click', () => this.toggleSTT());
    this.dom.speakerButton.addEventListener('click', () => this.toggleAutoRead());
    this.dom.stopSpeakingButton.addEventListener('click', () => this.stopSpeaking());

    // Conversation
    this.dom.clearButton.addEventListener('click', () => this.clearConversation());

    // Settings drawer
    this.dom.settingsButton.addEventListener('click', () => this.openSettings());
    this.root.addEventListener('click', (e) => {
      const action = e.target.closest('[data-action]')?.dataset.action;
      if (!action) return;
      if (action === 'open-settings') this.openSettings();
      else if (action === 'close-settings') this.closeSettings();
      else if (action === 'reset-system-prompt') {
        this.dom.settingSystemPrompt.value = DEFAULT_SYSTEM_PROMPT;
      } else if (action === 'clear-storage') {
        this.clearStoredSettings();
      } else if (action === 'goto-threat-analysis') {
        this.setActiveTab('threat-analysis');
      } else if (action === 'goto-workbench') {
        this.setActiveTab('workbench');
      }
    });

    // Alerts sidebar
    this.dom.alertsClear.addEventListener('click', () => this.clearAlerts());
    this.dom.settingsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      this.saveAndCloseSettings();
    });
    this.root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (!this.dom.drawer.hidden) this.closeSettings();
        else if (this.recognitionActive) this.stopSTT();
        else if (window.speechSynthesis?.speaking) this.stopSpeaking();
      }
    });

    // Provider radio: live-preview placeholder swap (does not save until submit).
    this.dom.settingsForm.querySelectorAll('input[name="provider"]').forEach((el) => {
      el.addEventListener('change', () => {
        const provider = el.value;
        const def = PROVIDER_DEFAULTS[provider];
        // Only swap if the field still holds the *other* provider's default — preserves operator overrides.
        if (this.dom.settingBaseUrl.value === PROVIDER_DEFAULTS.ollama.baseUrl ||
            this.dom.settingBaseUrl.value === PROVIDER_DEFAULTS.openai.baseUrl ||
            this.dom.settingBaseUrl.value === '') {
          this.dom.settingBaseUrl.value = def.baseUrl;
        }
        if (this.dom.settingModel.value === PROVIDER_DEFAULTS.ollama.model ||
            this.dom.settingModel.value === PROVIDER_DEFAULTS.openai.model ||
            this.dom.settingModel.value === '') {
          this.dom.settingModel.value = def.model;
        }
        this.dom.apiKeyField.classList.toggle('hidden', provider === 'ollama');
        this.dom.settingBaseUrl.placeholder = def.baseUrl;
        this.dom.settingModel.placeholder = def.model;
      });
    });
  }

  autoSizeTextarea() {
    const ta = this.dom.composerInput;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 240) + 'px';
  }

  // ── Settings drawer actions ───────────────────────────────────────────────

  openSettings() {
    this.applySettingsToForm();
    this.dom.drawer.hidden = false;
    // Defer focus so the drawer is in the layout before we move focus.
    requestAnimationFrame(() => {
      this.dom.settingsForm.querySelector('input,select,textarea,button')?.focus();
    });
  }

  closeSettings() {
    this.dom.drawer.hidden = true;
    this.dom.settingsButton.focus();
  }

  saveAndCloseSettings() {
    this.settings = { ...this.settings, ...this.readSettingsFromForm() };
    this.saveSettings();
    this.applyProviderUI();
    this.applyHighContrast(this.settings.highContrast);
    // Re-evaluate STT availability with new settings.
    if (this.recognition && this.recognition.lang !== this.settings.language) {
      this.recognition.lang = this.settings.language;
    }
    this.updateMicAvailability();
    this.closeSettings();
    this.setStatus('Settings saved.', 'info');
    setTimeout(() => this.setStatus('', null), 1500);
  }

  applyHighContrast(enabled) {
    document.body.classList.toggle('high-contrast', !!enabled);
  }

  clearStoredSettings() {
    for (const key of Object.values(STORAGE_KEYS)) this.safeStorageRemove(key);
    this.settings = { ...DEFAULT_SETTINGS };
    this.applySettingsToForm();
    this.applyProviderUI();
    this.applyHighContrast(this.settings.highContrast);
    this.setStatus('Stored settings cleared.', 'info');
  }

  // ── Conversation ──────────────────────────────────────────────────────────

  sendMessage() {
    if (this.pendingRequest) return; // one in-flight request at a time
    const text = this.dom.composerInput.value.trim();
    if (!text && !this.attachment) return;

    const userEntry = {
      role: 'user',
      content: text,
      attachment: this.attachment,
    };
    this.history.push(userEntry);
    this.appendMessage({ role: 'user', content: text, attachment: this.attachment });

    this.dom.composerInput.value = '';
    this.autoSizeTextarea();
    const sentAttachment = this.attachment;
    this.clearAttachment();

    this.dispatchToProvider({ userText: text, attachment: sentAttachment });
  }

  async dispatchToProvider({ userText, attachment }) {
    const controller = new AbortController();
    this.pendingRequest = controller;
    this.setStatus('Analysing surveillance frame…', 'info');
    this.dom.sendButton.disabled = true;

    let assistantLi = null;
    const requestStartedAt = performance.now();

    try {
      const messages = this.buildApiMessages({ userText, attachment });
      const result = await this.callChatCompletion({
        messages,
        signal: controller.signal,
        onDelta: (_chunk, full) => {
          if (!assistantLi) {
            assistantLi = this.appendMessage({ role: 'assistant', content: full, streaming: true });
          } else {
            this.updateStreamingMessage(assistantLi, full);
          }
        },
      });

      const completedAt = performance.now();
      const reply = result.full;

      // Final render — flushes the last delta, applies definitive threat level, drops the streaming cursor.
      if (!assistantLi) {
        assistantLi = this.appendMessage({ role: 'assistant', content: reply });
      }
      this.updateStreamingMessage(assistantLi, reply);
      this.finalizeStreamingMessage(assistantLi);

      const stats = this.computeStats({ requestStartedAt, completedAt, result });
      this.setMessageStats(assistantLi, stats);

      const threatLevel = this.extractThreatLevel(reply);
      this.history.push({ role: 'assistant', content: reply, threatLevel });

      if (threatLevel) {
        this.addAlert({
          threatLevel,
          summary: this.summaryFromReply(reply),
          snapshotUri: attachment?.dataUri || null,
          source: 'workbench',
          messageEl: assistantLi,
        });
      }

      if (this.settings.autoRead) this.speakText(reply);
      this.setStatus('', null);
    } catch (err) {
      if (assistantLi) this.finalizeStreamingMessage(assistantLi);
      if (err.name === 'AbortError') return;
      const msg = this.formatError(err);
      this.appendMessage({ role: 'error', content: msg });
      this.setStatus(msg, 'error');
    } finally {
      this.pendingRequest = null;
      this.dom.sendButton.disabled = false;
    }
  }

  computeStats({ requestStartedAt, completedAt, result }) {
    const totalMs = completedAt - requestStartedAt;
    const ttftMs = result.firstDeltaAt != null ? result.firstDeltaAt - requestStartedAt : null;
    // Generation time = wall time spent producing tokens, i.e. excluding model loading / prompt processing.
    const generationMs = result.firstDeltaAt != null ? completedAt - result.firstDeltaAt : totalMs;

    let tokens = null;
    let tokenSource = null;
    if (result.usage && typeof result.usage.completion_tokens === 'number') {
      tokens = result.usage.completion_tokens;
      tokenSource = 'usage';
    } else if (typeof result.deltaCount === 'number' && result.deltaCount > 0) {
      tokens = result.deltaCount;
      tokenSource = 'deltas';
    } else if (typeof result.full === 'string' && result.full.length) {
      // Rough fallback: average ~4 chars per token across English-ish content.
      tokens = Math.max(1, Math.round(result.full.length / 4));
      tokenSource = 'estimated';
    }

    const tps = tokens != null && generationMs > 0 ? tokens / (generationMs / 1000) : null;
    return {
      ttftMs,
      totalMs,
      generationMs,
      tokens,
      tokenSource,
      tps,
      promptTokens: result.usage?.prompt_tokens ?? null,
    };
  }

  setMessageStats(li, stats) {
    if (!li || !stats) return;
    const el = li.querySelector('.message-stats');
    if (!el) return;
    this.renderStats(el, stats);
  }

  renderStats(el, stats) {
    if (!el || !stats) return;
    el.replaceChildren();

    const addStat = (label, value, valueClass = null, title = null) => {
      const wrap = document.createElement('span');
      wrap.className = 'stat';
      if (title) wrap.title = title;
      const lbl = document.createElement('span');
      lbl.className = 'stat-label';
      lbl.textContent = label;
      const val = document.createElement('span');
      val.className = 'stat-value' + (valueClass ? ' ' + valueClass : '');
      val.textContent = value;
      wrap.append(lbl, ' ', val);
      el.appendChild(wrap);
    };

    if (stats.ttftMs != null) {
      const cls = stats.ttftMs < 500 ? 'is-good' : stats.ttftMs < 2000 ? 'is-warn' : 'is-slow';
      addStat('TTFT', this.formatMs(stats.ttftMs), cls);
    }
    if (stats.tps != null) {
      const cls = stats.tps >= 30 ? 'is-good' : stats.tps >= 10 ? 'is-warn' : 'is-slow';
      addStat('TPS', `${stats.tps.toFixed(1)} tok/s`, cls);
    }
    if (stats.tokens != null) {
      const suffix =
        stats.tokenSource === 'usage' ? '' :
        stats.tokenSource === 'deltas' ? ' (Δ)' :
        stats.tokenSource === 'estimated' ? ' (~)' : '';
      const title =
        stats.tokenSource === 'usage' ? 'Reported by provider (usage.completion_tokens)' :
        stats.tokenSource === 'deltas' ? 'Counted from SSE chunks (≈1 token per chunk)' :
        stats.tokenSource === 'estimated' ? 'Estimated from response length (~4 chars/token)' : '';
      addStat('tokens', `${stats.tokens}${suffix}`, null, title);
    }
    if (stats.promptTokens != null) {
      addStat('prompt', `${stats.promptTokens}`, null, 'Prompt tokens reported by provider');
    }
    addStat('total', this.formatMs(stats.totalMs));

    el.hidden = false;
  }

  formatMs(ms) {
    if (!isFinite(ms)) return '—';
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  }

  buildLanguageDirective({ withThreatNote = true } = {}) {
    const lang = this.settings.language || 'en-US';
    const langName = LANGUAGE_NAMES[lang] || 'English';
    // Lock the response language to the operator's configured choice. Phrased
    // as a strict directive because some models otherwise mirror the operator's
    // input language and bounce between languages turn-to-turn.
    const parts = [
      'OUTPUT LANGUAGE — STRICT:',
      `Always respond in ${langName} (locale ${lang}).`,
      'Do not detect or mirror the operator\'s typing language.',
      'Do not switch languages between turns; every assessment body, recommendation, and prose line must be in ' + langName + '.',
    ];
    if (withThreatNote) {
      parts.push('The single leading "THREAT: <LEVEL>" line stays in English so the console can color-code the alert; everything after it is in ' + langName + '.');
    } else {
      // For descriptive flows (Change Detection): the structural tokens we DO
      // need to keep in English are the [CATEGORY] tags (NEW/MISSING/STRUCTURAL)
      // so the parser can read them; the prose around them is in the chosen language.
      parts.push('The bracketed [NEW] / [MISSING] / [STRUCTURAL] category tokens stay in English so the console can parse them; everything else is in ' + langName + '.');
    }
    return parts.join(' ');
  }

  buildSystemContent() {
    const base = (this.settings.systemPrompt || DEFAULT_SYSTEM_PROMPT).trim();
    return `${base}\n\n${this.buildLanguageDirective()}`;
  }

  buildChangeDetectionSystemContent() {
    return `${CHANGE_DETECTION_SYSTEM_PROMPT}\n\n${this.buildLanguageDirective({ withThreatNote: false })}`;
  }

  buildApiMessages({ userText, attachment }) {
    const systemMsg = {
      role: 'system',
      content: this.buildSystemContent(),
    };

    // Collect prior turns (excluding the just-pushed current user message).
    // Prior attachments collapse to a textual marker so we don't re-upload images.
    const prior = [];
    for (let i = 0; i < this.history.length - 1; i++) {
      const m = this.history[i];
      if (m.role === 'user') {
        let content = m.content || '';
        if (m.attachment) {
          const note = `[surveillance frame attached: ${m.attachment.filename}]`;
          content = content ? `${content}\n${note}` : note;
        }
        prior.push({ role: 'user', content });
      } else if (m.role === 'assistant') {
        prior.push({ role: 'assistant', content: m.content });
      }
    }

    // Current user — multimodal if there's an attachment, otherwise plain string.
    const currentUser = attachment
      ? {
          role: 'user',
          content: [
            { type: 'text', text: userText || 'Analyse this surveillance frame for threats.' },
            { type: 'image_url', image_url: { url: attachment.dataUri } },
          ],
        }
      : { role: 'user', content: userText };

    // Strict templates (Llama-family chat templates, etc.) require loop_messages
    // to alternate user/assistant/user/assistant starting from user. If a previous
    // turn errored, the failed user turn stays in history and a naive concatenation
    // would produce two consecutive user messages. Merge same-role neighbours and
    // ensure the first non-system message is a user turn.
    return [systemMsg, ...this.enforceAlternation([...prior, currentUser])];
  }

  enforceAlternation(messages) {
    const out = [];
    for (const m of messages) {
      const last = out[out.length - 1];
      if (last && last.role === m.role) {
        out[out.length - 1] = this.mergeSameRoleMessages(last, m);
      } else {
        out.push(m);
      }
    }
    // Drop any leading assistant turns so loop_messages starts with user.
    while (out.length && out[0].role !== 'user') out.shift();
    return out;
  }

  mergeSameRoleMessages(a, b) {
    const aIsArr = Array.isArray(a.content);
    const bIsArr = Array.isArray(b.content);
    if (!aIsArr && !bIsArr) {
      return { role: a.role, content: `${a.content || ''}\n\n${b.content || ''}`.trim() };
    }
    // At least one side is multimodal — keep the array shape and weave the string text in.
    const arrSide = bIsArr ? b : a;
    const strSide = bIsArr ? a : b;
    const strText = typeof strSide.content === 'string' ? strSide.content : '';
    const newContent = arrSide.content.map((p) => ({ ...p }));
    const textIdx = newContent.findIndex((p) => p.type === 'text');
    if (textIdx === -1) {
      newContent.unshift({ type: 'text', text: strText });
    } else {
      // strSide is the older one when strSide === a; prepend its text to preserve order.
      const prefix = strSide === a ? `${strText}\n\n` : '';
      const suffix = strSide === b ? `\n\n${strText}` : '';
      newContent[textIdx] = {
        ...newContent[textIdx],
        text: `${prefix}${newContent[textIdx].text || ''}${suffix}`.trim(),
      };
    }
    return { role: a.role, content: newContent };
  }

  async callChatCompletion({ messages, signal, onDelta }) {
    const { provider, baseUrl, apiKey, model } = this.settings;
    if (!baseUrl) throw new Error('Base URL is not configured. Open settings to set it.');
    if (!model) throw new Error('Vision model is not configured. Open settings to set it.');

    const headers = { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' };
    if (provider === 'openai' && apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const payload = { model, messages, stream: true };
    // Ask OpenAI for token usage in the trailing chunk. Ollama emits usage natively in
    // its final OpenAI-compatible chunk and ignores stream_options when sent.
    if (provider === 'openai') payload.stream_options = { include_usage: true };
    const body = JSON.stringify(payload);

    let response;
    try {
      response = await fetch(baseUrl, { method: 'POST', headers, body, signal });
    } catch (err) {
      // TypeError from fetch can mean: CORS, mixed content (HTTPS page → HTTP endpoint),
      // CSP block, refused connection, or DNS failure. We can't tell from the error alone.
      const isLocalhost = /localhost|127\.0\.0\.1/i.test(baseUrl);
      const hint = isLocalhost
        ? `Cannot reach ${baseUrl}. Is \`ollama serve\` running, and was it started with \`OLLAMA_ORIGINS=*\`?`
        : `Cannot reach ${baseUrl}. Possible causes: (1) the server isn't running or this device can't route to it; (2) CORS — the server must return \`Access-Control-Allow-Origin\` for this page's origin; (3) mixed content — if this page is served over HTTPS, http:// endpoints are blocked. Check DevTools → Console for the exact browser error.`;
      const wrapped = new Error(hint);
      wrapped.cause = err;
      throw wrapped;
    }

    if (!response.ok) {
      let detail = '';
      try { detail = (await response.text()).slice(0, 500); } catch { /* */ }
      throw new Error(`HTTP ${response.status} ${response.statusText}${detail ? `\n${detail}` : ''}`);
    }

    const ct = response.headers.get('content-type') || '';
    if (!ct.includes('text/event-stream') || !response.body) {
      // Provider ignored stream:true (or stream isn't readable here) — fall back to a one-shot JSON read.
      const data = await response.json();
      const rawReply = data?.choices?.[0]?.message?.content;
      if (typeof rawReply !== 'string' || !rawReply.trim()) {
        throw new Error('Provider returned an empty response.');
      }
      const reply = this.stripStopTokens(rawReply);
      if (onDelta) onDelta(reply, reply);
      return {
        full: reply,
        firstDeltaAt: null,
        deltaCount: null,
        usage: data?.usage || null,
      };
    }

    return await this.readSSEStream(response.body, onDelta);
  }

  // Truncate at the first occurrence of any known chat-template stop token.
  stripStopTokens(text) {
    if (!text) return text;
    const m = STOP_TOKEN_REGEX.exec(text);
    return m ? text.slice(0, m.index) : text;
  }

  async readSSEStream(stream, onDelta) {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let full = '';            // committed (rendered) text
    let bufferedSuffix = '';  // tail held back because it could be a stop-token prefix
    let firstDeltaAt = null;
    let deltaCount = 0;
    let usage = null;

    const commitDelta = (text) => {
      if (!text || !text.length) return;
      if (firstDeltaAt === null) firstDeltaAt = performance.now();
      deltaCount++;
      full += text;
      if (onDelta) onDelta(text, full);
    };

    const processDelta = (rawDelta) => {
      // Combine with any held-back tail so a stop token split across chunks is detected.
      const combined = bufferedSuffix + rawDelta;
      const m = STOP_TOKEN_REGEX.exec(combined);
      if (m) {
        commitDelta(combined.slice(0, m.index));
        bufferedSuffix = '';
        return 'done';
      }
      // Hold back any trailing chars that could start a stop token, in case the
      // remainder of the token arrives in the next delta.
      const pendLen = this.pendingStopTokenSuffixLength(combined);
      const safeEnd = combined.length - pendLen;
      commitDelta(combined.slice(0, safeEnd));
      bufferedSuffix = combined.slice(safeEnd);
      return 'continue';
    };

    const flushEvent = (eventText) => {
      for (const rawLine of eventText.split('\n')) {
        const line = rawLine.replace(/\r$/, '');
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (!payload) continue;
        if (payload === '[DONE]') return 'done';
        let obj;
        try { obj = JSON.parse(payload); } catch { continue; }
        // Capture usage if the provider includes it (OpenAI: trailing chunk; Ollama: final chunk).
        if (obj?.usage) usage = obj.usage;
        // Some servers signal end via finish_reason on a delta chunk.
        const finishReason = obj?.choices?.[0]?.finish_reason;
        // OpenAI streams `delta.content`; some providers stream `message.content` cumulatively.
        const delta =
          obj?.choices?.[0]?.delta?.content ??
          obj?.choices?.[0]?.message?.content ??
          '';
        if (typeof delta === 'string' && delta.length) {
          if (processDelta(delta) === 'done') return 'done';
        }
        // finish_reason is null on intermediate chunks, "stop"/"length"/"tool_calls"/etc on the terminal one.
        if (finishReason) return 'done';
      }
      return 'continue';
    };

    // If the stream ended with a held-back suffix that turned out NOT to be a
    // stop token, the chars are genuine content — commit them.
    const finalCommit = () => {
      if (!bufferedSuffix) return;
      const stripped = this.stripStopTokens(bufferedSuffix);
      bufferedSuffix = '';
      if (stripped.length) commitDelta(stripped);
    };

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const event = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (flushEvent(event) === 'done') {
            finalCommit();
            try { reader.cancel(); } catch { /* */ }
            if (!full.trim()) throw new Error('Provider returned an empty response.');
            return { full, firstDeltaAt, deltaCount, usage };
          }
        }
      }
      // Stream ended without [DONE] — process any remainder as a final event.
      if (buffer.trim()) flushEvent(buffer);
      finalCommit();
    } finally {
      try { reader.releaseLock(); } catch { /* */ }
    }

    // Belt-and-braces: cumulative-content responses might still embed a stop token.
    full = this.stripStopTokens(full);
    if (!full.trim()) throw new Error('Provider returned an empty response.');
    return { full, firstDeltaAt, deltaCount, usage };
  }

  // Length of the longest suffix of `s` that is a strict prefix of any STOP_TOKEN.
  // Used to hold back chars that might be the start of a stop token split across SSE chunks.
  pendingStopTokenSuffixLength(s) {
    if (!s) return 0;
    const start = Math.min(s.length, STOP_TOKEN_MAX_LEN - 1);
    for (let n = start; n > 0; n--) {
      const tail = s.slice(s.length - n);
      for (const tok of STOP_TOKENS) {
        if (tok.length > tail.length && tok.startsWith(tail)) return n;
      }
    }
    return 0;
  }

  formatError(err) {
    const base = err?.message || String(err);
    return `Threat assessment failed: ${base}`;
  }

  appendMessage({ role, content, attachment, threatLevel, streaming }) {
    const tpl = this.dom.messageTemplate.content.cloneNode(true);
    const li = tpl.querySelector('.message');
    li.dataset.role = role;
    if (threatLevel) li.dataset.threatLevel = threatLevel;
    if (streaming) li.classList.add('streaming');

    const roleEl = li.querySelector('.message-role');
    roleEl.textContent = role === 'assistant' ? 'AI Analyst' : role === 'system' ? 'System' : role === 'error' ? 'Error' : 'Operator';

    const pill = li.querySelector('.threat-pill');
    if (role === 'assistant' && threatLevel) {
      pill.hidden = false;
      pill.dataset.threatLevel = threatLevel;
      pill.textContent = `Threat · ${threatLevel.toUpperCase()}`;
    }

    const contentEl = li.querySelector('.message-content');
    if (role === 'assistant') {
      contentEl.innerHTML = this.renderMarkdown(content || '');
    } else {
      contentEl.textContent = content || '';
    }

    const attachmentsEl = li.querySelector('.message-attachments');
    if (attachment?.dataUri) {
      const img = document.createElement('img');
      img.src = attachment.dataUri;
      img.alt = `Surveillance frame: ${attachment.filename}`;
      attachmentsEl.appendChild(img);
    } else {
      attachmentsEl.remove();
    }

    this.dom.messages.appendChild(li);
    this.dom.body.classList.add('has-messages');
    li.scrollIntoView({ block: 'end', behavior: 'smooth' });
    return li;
  }

  updateStreamingMessage(li, fullText) {
    if (!li) return;
    const contentEl = li.querySelector('.message-content');
    if (contentEl) contentEl.innerHTML = this.renderMarkdown(fullText);

    // Apply the threat level as soon as the first line is parseable.
    const level = this.extractThreatLevel(fullText);
    if (level && li.dataset.threatLevel !== level) {
      li.dataset.threatLevel = level;
      const pill = li.querySelector('.threat-pill');
      if (pill) {
        pill.hidden = false;
        pill.dataset.threatLevel = level;
        pill.textContent = `Threat · ${level.toUpperCase()}`;
      }
    }

    if (this.isScrolledNearBottom()) li.scrollIntoView({ block: 'end', behavior: 'auto' });
  }

  finalizeStreamingMessage(li) {
    if (!li) return;
    li.classList.remove('streaming');
  }

  isScrolledNearBottom(threshold = 80) {
    const main = this.root.querySelector('.app-main');
    if (!main) return true;
    return main.scrollTop + main.clientHeight >= main.scrollHeight - threshold;
  }

  // ── Markdown renderer (zero-dependency, XSS-safe) ─────────────────────────
  // Escape first, then apply transformations on the escaped text. The output
  // contains only an allow-listed set of tags: p, h1-h3, pre, code, strong, em,
  // ul, ol, li, br, a (with href validated to http(s)/mailto/relative).

  escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  renderMarkdown(src) {
    if (!src) return '';
    const lines = String(src).split(/\r?\n/);
    const out = [];
    let i = 0;
    let listType = null;
    let listItems = [];
    const flushList = () => {
      if (!listType) return;
      const items = listItems.map((item) => `<li>${this.renderInline(item)}</li>`).join('');
      out.push(`<${listType}>${items}</${listType}>`);
      listType = null;
      listItems = [];
    };

    while (i < lines.length) {
      const line = lines[i];

      // Code fence: ```lang … ```  (handles unclosed fences during streaming)
      const fence = line.match(/^```\s*([\w+-]*)\s*$/);
      if (fence) {
        flushList();
        const lang = fence[1] || '';
        const codeLines = [];
        i++;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) {
          codeLines.push(lines[i]);
          i++;
        }
        // skip the closing fence (or fall through end-of-input gracefully)
        if (i < lines.length) i++;
        const escaped = this.escapeHtml(codeLines.join('\n'));
        out.push(`<pre><code${lang ? ` class="lang-${this.escapeHtml(lang)}"` : ''}>${escaped}</code></pre>`);
        continue;
      }

      // Headings: # H1, ## H2, ### H3
      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      if (heading) {
        flushList();
        out.push(`<h${heading[1].length}>${this.renderInline(heading[2])}</h${heading[1].length}>`);
        i++;
        continue;
      }

      // Unordered list: -, *, +
      const ul = line.match(/^\s{0,3}[-*+]\s+(.*)$/);
      if (ul) {
        if (listType !== 'ul') { flushList(); listType = 'ul'; }
        listItems.push(ul[1]);
        i++;
        continue;
      }

      // Ordered list: 1. 2. …
      const ol = line.match(/^\s{0,3}\d+\.\s+(.*)$/);
      if (ol) {
        if (listType !== 'ol') { flushList(); listType = 'ol'; }
        listItems.push(ol[1]);
        i++;
        continue;
      }

      // Blank line — paragraph break
      if (!line.trim()) {
        flushList();
        i++;
        continue;
      }

      // Paragraph: greedily collect consecutive non-empty, non-special lines.
      flushList();
      const para = [line];
      i++;
      while (
        i < lines.length &&
        lines[i].trim() &&
        !/^(#{1,3}\s|```|\s{0,3}[-*+]\s|\s{0,3}\d+\.\s)/.test(lines[i])
      ) {
        para.push(lines[i]);
        i++;
      }
      out.push(`<p>${this.renderInline(para.join('\n'))}</p>`);
    }
    flushList();
    return out.join('');
  }

  renderInline(s) {
    let t = this.escapeHtml(s);
    // Inline code: `code`  (handle first so its content isn't touched by other inline rules)
    t = t.replace(/`([^`\n]+)`/g, (_, c) => `<code>${c}</code>`);
    // Bold: **text**
    t = t.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
    // Italic: *text* (not part of a remaining ** pair) and _text_
    t = t.replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
    t = t.replace(/(^|[^_\w])_([^_\n]+?)_(?!_)/g, '$1<em>$2</em>');
    // Links: [text](url) — validate scheme to block javascript:, data:, vbscript:, etc.
    t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, txt, url) => {
      if (/^(https?:\/\/|mailto:|\.{0,2}\/|#)/i.test(url)) {
        return `<a href="${url}" target="_blank" rel="noopener noreferrer">${txt}</a>`;
      }
      return m;
    });
    // Soft line breaks inside a paragraph
    t = t.replace(/\n/g, '<br>');
    return t;
  }

  clearConversation() {
    if (this.pendingRequest) this.pendingRequest.abort();
    this.history = [];
    this.dom.messages.replaceChildren();
    this.dom.body.classList.remove('has-messages');
    this.clearAttachment();
    if (window.speechSynthesis?.speaking) this.stopSpeaking();
    this.setStatus('Threat log cleared.', 'info');
    setTimeout(() => this.setStatus('', null), 1500);
  }

  // ── Threat-level extraction ───────────────────────────────────────────────

  extractThreatLevel(text) {
    if (typeof text !== 'string') return null;
    const head = text.slice(0, 240);
    const m = head.match(/THREAT\s*[:\-]\s*\[?\s*(INFO|LOW|MEDIUM|HIGH|CRITICAL)\s*\]?/i)
      || head.match(/\[(INFO|LOW|MEDIUM|HIGH|CRITICAL)\]/i);
    if (!m) return null;
    const level = m[1].toLowerCase();
    return THREAT_LEVELS.includes(level) ? level : null;
  }

  // ── Vision: image attach ──────────────────────────────────────────────────

  async attachImageFile(file) {
    try {
      this.setStatus(`Encoding ${file.name}…`, 'info');
      const dataUri = await this.fileToDownscaledDataUri(file);
      const dims = await this.dataUriDimensions(dataUri);
      this.attachment = {
        dataUri,
        filename: file.name,
        width: dims.width,
        height: dims.height,
        source: 'image',
      };
      this.renderAttachmentPreview();
      this.setStatus('', null);
    } catch (err) {
      this.setStatus(`Image attach failed: ${err.message}`, 'error');
    }
  }

  fileToDataUri(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error || new Error('FileReader error'));
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(file);
    });
  }

  async fileToDownscaledDataUri(file) {
    const original = await this.fileToDataUri(file);
    return this.downscaleDataUri(original);
  }

  async downscaleDataUri(dataUri) {
    const img = await this.loadImage(dataUri);
    const { width, height } = img;
    const longEdge = Math.max(width, height);
    if (longEdge <= MAX_IMAGE_DIM) return dataUri;
    const scale = MAX_IMAGE_DIM / longEdge;
    const w = Math.round(width * scale);
    const h = Math.round(height * scale);
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
    return canvas.toDataURL('image/jpeg', 0.85);
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Could not decode image data.'));
      img.src = src;
    });
  }

  async dataUriDimensions(dataUri) {
    const img = await this.loadImage(dataUri);
    return { width: img.naturalWidth, height: img.naturalHeight };
  }

  // ── Vision: video frame capture ───────────────────────────────────────────

  async captureFrameFromVideoFile(file) {
    const objectUrl = URL.createObjectURL(file);
    try {
      this.setStatus(`Capturing frame from ${file.name}…`, 'info');
      const video = this.dom.videoExtractor;
      video.src = objectUrl;
      await this.waitForVideoReady(video);
      const seekTime = Math.min(VIDEO_FRAME_SEEK_SECONDS, Math.max(0, (video.duration || 0) - 0.05));
      await this.seekVideo(video, seekTime);
      const canvas = document.createElement('canvas');
      const w = video.videoWidth || 1280;
      const h = video.videoHeight || 720;
      const longEdge = Math.max(w, h);
      const scale = longEdge > MAX_IMAGE_DIM ? MAX_IMAGE_DIM / longEdge : 1;
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUri = canvas.toDataURL('image/jpeg', 0.85);
      this.attachment = {
        dataUri,
        filename: `${file.name} @ ${seekTime.toFixed(2)}s`,
        width: canvas.width,
        height: canvas.height,
        source: 'frame',
      };
      this.renderAttachmentPreview();
      this.setStatus('', null);
    } catch (err) {
      this.setStatus(`Frame capture failed: ${err.message}`, 'error');
    } finally {
      URL.revokeObjectURL(objectUrl);
      this.dom.videoExtractor.removeAttribute('src');
      this.dom.videoExtractor.load();
    }
  }

  waitForVideoReady(video) {
    return new Promise((resolve, reject) => {
      const onLoaded = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('Video could not be decoded by this browser.')); };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoaded);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('loadedmetadata', onLoaded, { once: true });
      video.addEventListener('error', onError, { once: true });
    });
  }

  seekVideo(video, time) {
    return new Promise((resolve, reject) => {
      const onSeeked = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('Seek failed.')); };
      const cleanup = () => {
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('seeked', onSeeked, { once: true });
      video.addEventListener('error', onError, { once: true });
      try { video.currentTime = time; } catch (err) { cleanup(); reject(err); }
    });
  }

  renderAttachmentPreview() {
    if (!this.attachment) {
      this.dom.imagePreview.hidden = true;
      this.dom.imagePreviewImg.removeAttribute('src');
      return;
    }
    this.dom.imagePreviewImg.src = this.attachment.dataUri;
    const { filename, width, height, source } = this.attachment;
    const tag = source === 'frame' ? 'frame' : 'image';
    this.dom.imagePreviewMeta.textContent = `${filename} · ${width}×${height} ${tag}`;
    this.dom.imagePreview.hidden = false;
  }

  clearAttachment() {
    this.attachment = null;
    this.renderAttachmentPreview();
  }

  // ── Voice: STT ────────────────────────────────────────────────────────────

  setupSTT() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      this.sttSupported = false;
      this.dom.micButton.disabled = true;
      this.dom.micButton.title = 'Speech recognition is not supported in this browser.';
      return;
    }
    this.sttSupported = true;
    this.recognition = new Recognition();
    this.recognition.lang = this.settings.language;
    this.recognition.continuous = false;
    this.recognition.interimResults = true;
    this.recognition.onresult = (e) => this.handleSTTResult(e);
    this.recognition.onend = () => this.handleSTTEnd();
    this.recognition.onerror = (e) => this.handleSTTError(e);
    this.updateMicAvailability();
  }

  updateMicAvailability() {
    if (!this.sttSupported || this.sttPermanentlyDisabled) {
      this.dom.micButton.disabled = true;
      return;
    }
    if (this.settings.sttDisabled) {
      this.dom.micButton.disabled = true;
      this.dom.micButton.title = 'Microphone disabled in settings (recommended on air-gapped terminals).';
      return;
    }
    this.dom.micButton.disabled = false;
    this.dom.micButton.title = '';
  }

  toggleSTT() {
    if (this.recognitionActive) this.stopSTT();
    else this.startSTT();
  }

  startSTT() {
    if (!this.recognition) return;
    try {
      this.recognition.lang = this.settings.language;
      this.recognition.start();
      this.recognitionActive = true;
      this.dom.micButton.classList.add('recording');
      this.dom.micButton.setAttribute('aria-pressed', 'true');
      this.setStatus('Listening…', 'info');
    } catch (err) {
      // start() throws if already active; ignore silently.
    }
  }

  stopSTT() {
    if (!this.recognition) return;
    try { this.recognition.stop(); } catch { /* */ }
  }

  handleSTTResult(event) {
    let interim = '';
    let final = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const r = event.results[i];
      if (r.isFinal) final += r[0].transcript;
      else interim += r[0].transcript;
    }
    const ta = this.dom.composerInput;
    if (final) {
      const sep = ta.value && !ta.value.endsWith(' ') ? ' ' : '';
      ta.value = (ta.value + sep + final).trimStart();
      this.autoSizeTextarea();
    } else if (interim) {
      this.setStatus(`Listening… "${interim}"`, 'info');
    }
  }

  handleSTTEnd() {
    this.recognitionActive = false;
    this.dom.micButton.classList.remove('recording');
    this.dom.micButton.setAttribute('aria-pressed', 'false');
    if (this.dom.status.dataset.persistent !== 'true') this.setStatus('', null);
  }

  handleSTTError(event) {
    const code = event?.error || 'unknown';
    switch (code) {
      case 'network':
        this.sttPermanentlyDisabled = true;
        this.dom.micButton.disabled = true;
        this.dom.micButton.title = 'Speech-to-text requires network access in this browser. Disabled for this air-gapped session.';
        this.setStatus(
          'Speech-to-text requires network access in this browser and is unavailable on this air-gapped device. Use the keyboard or a browser with on-device dictation (e.g. Edge with Windows Speech Services).',
          'warn',
          true,
        );
        break;
      case 'not-allowed':
      case 'service-not-allowed':
        this.setStatus('Microphone permission denied — enable it in your browser settings.', 'error');
        break;
      case 'no-speech':
      case 'aborted':
        // Silent reset; mic stays usable.
        break;
      default:
        this.setStatus(`Speech recognition error: ${code}`, 'error');
        break;
    }
  }

  // ── Voice: TTS ────────────────────────────────────────────────────────────

  setupTTS() {
    if (!('speechSynthesis' in window)) return;
    const refresh = () => { this.voicesCache = window.speechSynthesis.getVoices() || []; };
    refresh();
    if (typeof window.speechSynthesis.onvoiceschanged !== 'undefined') {
      window.speechSynthesis.addEventListener('voiceschanged', refresh);
    }
  }

  toggleAutoRead() {
    this.settings.autoRead = !this.settings.autoRead;
    this.dom.settingAutoRead.checked = this.settings.autoRead;
    this.dom.speakerButton.setAttribute('aria-pressed', this.settings.autoRead ? 'true' : 'false');
    this.safeStorageSet(STORAGE_KEYS.autoRead, String(this.settings.autoRead));
    if (!this.settings.autoRead && window.speechSynthesis?.speaking) this.stopSpeaking();
  }

  pickVoiceFor(lang) {
    const local = (this.voicesCache.length ? this.voicesCache : (window.speechSynthesis?.getVoices() || []))
      .filter((v) => v.localService === true);
    if (!local.length) return null;
    const exact = local.find((v) => v.lang === lang);
    if (exact) return exact;
    const prefix = lang.split('-')[0];
    const partial = local.find((v) => v.lang && v.lang.toLowerCase().startsWith(prefix.toLowerCase()));
    if (partial) return partial;
    return null;
  }

  speakText(text) {
    if (!('speechSynthesis' in window) || !text) return;
    const voice = this.pickVoiceFor(this.settings.language);
    if (!voice) {
      if (!this.missingVoiceWarned) {
        this.missingVoiceWarned = true;
        this.setStatus(
          `No offline voice installed for ${this.settings.language}. Install OS-level voices or change language.`,
          'warn',
        );
        setTimeout(() => { if (this.dom.status.dataset.persistent !== 'true') this.setStatus('', null); }, 6000);
      }
      return;
    }
    try { window.speechSynthesis.cancel(); } catch { /* */ }
    const utter = new SpeechSynthesisUtterance(text);
    utter.voice = voice;
    utter.lang = voice.lang || this.settings.language;
    utter.rate = 1;
    utter.pitch = 1;
    utter.onstart = () => this.setSpeakingState(true);
    utter.onend = () => this.setSpeakingState(false);
    utter.onerror = () => this.setSpeakingState(false);
    // Pre-emptively flip the UI — onstart fires only once the utterance is
    // actually playing, which can lag the speak() call by a frame or two.
    this.setSpeakingState(true);
    window.speechSynthesis.speak(utter);
  }

  stopSpeaking() {
    try { window.speechSynthesis?.cancel(); } catch { /* */ }
    this.setSpeakingState(false);
  }

  setSpeakingState(active) {
    const btn = this.dom.stopSpeakingButton;
    if (!btn) return;
    btn.disabled = !active;
    btn.classList.toggle('is-speaking', !!active);
  }

  // ── Status bar ────────────────────────────────────────────────────────────

  setStatus(text, tone, persistent = false) {
    this.dom.status.textContent = text || '';
    if (tone) this.dom.status.dataset.tone = tone;
    else delete this.dom.status.dataset.tone;
    if (persistent) this.dom.status.dataset.persistent = 'true';
    else delete this.dom.status.dataset.persistent;
  }

  // ── Tabs ──────────────────────────────────────────────────────────────────

  setupTabs() {
    for (const btn of this.dom.tabButtons) {
      btn.addEventListener('click', () => this.setActiveTab(btn.dataset.tab));
      btn.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          const tabs = this.dom.tabButtons;
          const idx = tabs.indexOf(btn);
          const next = e.key === 'ArrowRight' ? (idx + 1) % tabs.length : (idx - 1 + tabs.length) % tabs.length;
          this.setActiveTab(tabs[next].dataset.tab);
          tabs[next].focus();
        }
      });
    }
    // Mobile bottom nav — same setActiveTab dispatch, no roving tabindex
    // (single-row tablist; touch users don't need arrow-key navigation).
    for (const btn of this.dom.mobileNavButtons) {
      btn.addEventListener('click', () => this.setActiveTab(btn.dataset.tab));
    }
  }

  setActiveTab(name) {
    if (this.activeTab === name) return;
    this.activeTab = name;
    for (const btn of this.dom.tabButtons) {
      const sel = btn.dataset.tab === name;
      btn.setAttribute('aria-selected', sel ? 'true' : 'false');
      btn.tabIndex = sel ? 0 : -1;
    }
    for (const btn of this.dom.mobileNavButtons) {
      btn.setAttribute('aria-selected', btn.dataset.tab === name ? 'true' : 'false');
    }
    for (const panel of this.dom.tabPanels) {
      panel.hidden = panel.id !== `tab-${name}`;
    }
  }

  // ── Alerts sidebar ────────────────────────────────────────────────────────

  addAlert({ threatLevel, summary, snapshotUri, source, messageEl }) {
    const id = ++this.alertSeq;
    const alert = {
      id,
      threatLevel,
      summary: summary || '',
      snapshotUri: snapshotUri || null,
      source: source || 'workbench',
      timestamp: new Date(),
      messageRef: messageEl ? new WeakRef(messageEl) : null,
    };
    this.alerts.push(alert);
    this.appendAlertElement(alert);
    this.dom.alertsClear.hidden = false;
  }

  appendAlertElement(alert) {
    const tpl = this.dom.alertTemplate.content.cloneNode(true);
    const li = tpl.querySelector('.alert-item');
    li.dataset.alertId = String(alert.id);
    if (alert.threatLevel) li.dataset.threatLevel = alert.threatLevel;

    const wrap = li.querySelector('.alert-snapshot-wrap');
    const img = li.querySelector('.alert-snapshot');
    if (alert.snapshotUri) {
      img.src = alert.snapshotUri;
      img.alt = `Snapshot ${alert.id}`;
    } else {
      img.remove();
      wrap.classList.add('is-empty');
    }

    const pill = li.querySelector('.alert-pill');
    if (alert.threatLevel) {
      pill.dataset.threatLevel = alert.threatLevel;
      pill.textContent = alert.threatLevel.toUpperCase();
    } else {
      pill.remove();
    }

    li.querySelector('.alert-time').textContent = this.formatAlertTime(alert.timestamp);
    li.querySelector('.alert-source').textContent = alert.source === 'threat-analysis' ? 'Threat Analysis' : 'Workbench';
    li.querySelector('.alert-summary').textContent = alert.summary || '(no summary)';

    const btn = li.querySelector('.alert-button');
    btn.addEventListener('click', () => this.navigateToAlert(alert));

    // Newest alert at top.
    this.dom.alertsList.prepend(li);
  }

  clearAlerts() {
    this.alerts = [];
    this.dom.alertsList.replaceChildren();
    this.dom.alertsClear.hidden = true;
  }

  navigateToAlert(alert) {
    if (alert.source === 'threat-analysis') {
      this.setActiveTab('threat-analysis');
      return;
    }
    this.setActiveTab('workbench');
    const el = alert.messageRef?.deref();
    if (el && el.isConnected) {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.classList.add('alert-target');
      setTimeout(() => el.classList.remove('alert-target'), 1800);
    }
  }

  formatAlertTime(d) {
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  }

  summaryFromReply(reply) {
    if (!reply) return '';
    const stripped = reply.replace(/^\s*THREAT\s*[:\-]\s*\[?\s*\w+\s*\]?\s*\n+/i, '');
    const firstLine = stripped.split('\n').find((l) => l.trim()) || stripped.trim();
    return firstLine.length > 140 ? firstLine.slice(0, 137) + '…' : firstLine;
  }

  // ── Threat Analysis tab ───────────────────────────────────────────────────

  setupThreatAnalysis() {
    this.dom.taLoadImage.addEventListener('click', () => this.dom.taImageInput.click());
    this.dom.taImageInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this.taLoadImageFile(file);
      e.target.value = '';
    });
    this.dom.taClearMark.addEventListener('click', () => this.taClearBox());

    // Bounding-box drag via Pointer events — covers mouse, touch, and pen.
    const canvas = this.dom.taCanvas;
    canvas.addEventListener('pointerdown', (e) => this.taPointerDown(e));
    canvas.addEventListener('pointermove', (e) => this.taPointerMove(e));
    canvas.addEventListener('pointerup', (e) => this.taPointerUp(e));
    canvas.addEventListener('pointercancel', (e) => this.taPointerCancel(e));
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        // Keyboard fallback: drop a centred 25%-of-frame box.
        e.preventDefault();
        this.taSetBox(0.375, 0.375, 0.25, 0.25);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        if (this.taDragState) this.taPointerCancel(e);
        else if (this.taBox) this.taClearBox();
      }
    });

    this.dom.taAnalyze.addEventListener('click', () => this.taAnalyze());
  }

  async taLoadImageFile(file) {
    try {
      this.setStatus(`Loading ${file.name}…`, 'info');
      const dataUri = await this.fileToDownscaledDataUri(file);
      const img = await this.loadImage(dataUri);
      this.taImage = img;
      this.taBox = null;
      this.taDragState = null;
      const canvas = this.dom.taCanvas;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      // Paint pixels FIRST while the canvas is still display:none (the .has-image
      // class is what reveals it). Otherwise the browser would paint one frame
      // of `background: #000` showing through the still-transparent canvas
      // before our drawImage lands.
      this.taDrawCanvas();
      canvas.classList.add('has-image');
      this.dom.taCanvasWrap.classList.add('has-image');
      this.dom.taReadout.textContent = `${img.naturalWidth}×${img.naturalHeight} · click and drag to outline an object`;
      this.dom.taClearMark.disabled = true;
      this.dom.taAnalyze.disabled = true;
      this.setStatus('', null);
    } catch (err) {
      this.setStatus(`Image load failed: ${err.message}`, 'error');
    }
  }

  // Convert a pointer event's clientX/Y to canvas-internal pixel coordinates.
  taPointerToCanvas(event) {
    const canvas = this.dom.taCanvas;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);
    return { x: Math.max(0, Math.min(canvas.width, x)), y: Math.max(0, Math.min(canvas.height, y)) };
  }

  taPointerDown(event) {
    if (!this.taImage || event.button !== 0) return;
    const p = this.taPointerToCanvas(event);
    if (!p) return;
    try { this.dom.taCanvas.setPointerCapture(event.pointerId); } catch { /* */ }
    this.taDragState = {
      pointerId: event.pointerId,
      startX: p.x, startY: p.y,
      currentX: p.x, currentY: p.y,
      startClientX: event.clientX, startClientY: event.clientY,
      moved: false,
    };
    event.preventDefault();
  }

  taPointerMove(event) {
    if (!this.taDragState || event.pointerId !== this.taDragState.pointerId) return;
    const p = this.taPointerToCanvas(event);
    if (!p) return;
    this.taDragState.currentX = p.x;
    this.taDragState.currentY = p.y;
    // Movement threshold in CSS pixels (what the operator perceives), so we
    // don't reject genuine drags on devices with scaled canvases.
    const dxCss = event.clientX - this.taDragState.startClientX;
    const dyCss = event.clientY - this.taDragState.startClientY;
    if (Math.abs(dxCss) >= 6 || Math.abs(dyCss) >= 6) this.taDragState.moved = true;
    this.taDrawCanvas();
  }

  taPointerUp(event) {
    if (!this.taDragState || event.pointerId !== this.taDragState.pointerId) return;
    const drag = this.taDragState;
    this.taDragState = null;
    try { this.dom.taCanvas.releasePointerCapture(event.pointerId); } catch { /* */ }

    if (!drag.moved) {
      // Treat as an accidental click — don't create a tiny degenerate box.
      this.taDrawCanvas();
      this.setStatus('Click and drag to outline an object — a single click is too small.', 'info');
      setTimeout(() => { if (this.dom.status.dataset.persistent !== 'true') this.setStatus('', null); }, 2400);
      return;
    }

    const canvas = this.dom.taCanvas;
    const x0 = Math.min(drag.startX, drag.currentX);
    const y0 = Math.min(drag.startY, drag.currentY);
    const w = Math.abs(drag.currentX - drag.startX);
    const h = Math.abs(drag.currentY - drag.startY);
    this.taSetBox(x0 / canvas.width, y0 / canvas.height, w / canvas.width, h / canvas.height);
  }

  taPointerCancel(event) {
    if (this.taDragState && event && event.pointerId !== this.taDragState.pointerId) return;
    this.taDragState = null;
    if (event) {
      try { this.dom.taCanvas.releasePointerCapture(event.pointerId); } catch { /* */ }
    }
    this.taDrawCanvas();
  }

  // Set a finalized bounding box. Inputs are normalized 0–1 on the canvas.
  taSetBox(xN, yN, wN, hN) {
    const clamp = (v) => Math.max(0, Math.min(1, v));
    const x = clamp(xN);
    const y = clamp(yN);
    const w = Math.max(0, Math.min(1 - x, wN));
    const h = Math.max(0, Math.min(1 - y, hN));
    if (w < 0.005 || h < 0.005) return; // sub-half-percent boxes are noise
    this.taBox = { xN: x, yN: y, wN: w, hN: h };
    this.taDrawCanvas();
    const canvas = this.dom.taCanvas;
    const xPx = Math.round(x * canvas.width);
    const yPx = Math.round(y * canvas.height);
    const wPx = Math.round(w * canvas.width);
    const hPx = Math.round(h * canvas.height);
    this.dom.taReadout.textContent = `Box at (${xPx}, ${yPx}, ${wPx}×${hPx}) px · normalized (${x.toFixed(3)}, ${y.toFixed(3)}, ${w.toFixed(3)}, ${h.toFixed(3)})`;
    this.dom.taClearMark.disabled = false;
    this.dom.taAnalyze.disabled = !!this.taPendingRequest;
  }

  taClearBox() {
    this.taBox = null;
    this.taDragState = null;
    this.taDrawCanvas();
    if (this.taImage) {
      this.dom.taReadout.textContent = `${this.dom.taCanvas.width}×${this.dom.taCanvas.height} · click and drag to outline an object`;
    } else {
      this.dom.taReadout.textContent = 'No image loaded.';
    }
    this.dom.taClearMark.disabled = true;
    this.dom.taAnalyze.disabled = true;
  }

  taDrawCanvas() {
    const canvas = this.dom.taCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (this.taImage) ctx.drawImage(this.taImage, 0, 0, canvas.width, canvas.height);

    // Live preview during drag.
    if (this.taDragState) {
      const d = this.taDragState;
      const x0 = Math.min(d.startX, d.currentX);
      const y0 = Math.min(d.startY, d.currentY);
      const w = Math.abs(d.currentX - d.startX);
      const h = Math.abs(d.currentY - d.startY);
      this.taStrokeBox(ctx, x0, y0, w, h, true);
      return;
    }

    if (this.taBox) {
      const x = this.taBox.xN * canvas.width;
      const y = this.taBox.yN * canvas.height;
      const w = this.taBox.wN * canvas.width;
      const h = this.taBox.hN * canvas.height;
      this.taStrokeBox(ctx, x, y, w, h, false);
    }
  }

  taStrokeBox(ctx, x, y, w, h, dashed) {
    const longEdge = Math.max(ctx.canvas.width, ctx.canvas.height);
    const stroke = Math.max(3, Math.round(longEdge * 0.0035));
    const haloStroke = stroke + 4;
    // Black halo first for contrast on bright backgrounds…
    ctx.save();
    ctx.lineWidth = haloStroke;
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.setLineDash(dashed ? [stroke * 3, stroke * 2] : []);
    ctx.strokeRect(x, y, w, h);
    // …then red foreground.
    ctx.lineWidth = stroke;
    ctx.strokeStyle = '#f85149';
    ctx.strokeRect(x, y, w, h);
    // Tiny corner ticks for visibility against busy frames.
    if (!dashed) {
      const tick = Math.min(w, h, longEdge * 0.02);
      ctx.lineWidth = stroke;
      ctx.strokeStyle = '#f85149';
      ctx.beginPath();
      // top-left
      ctx.moveTo(x, y + tick); ctx.lineTo(x, y); ctx.lineTo(x + tick, y);
      // top-right
      ctx.moveTo(x + w - tick, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + tick);
      // bottom-right
      ctx.moveTo(x + w, y + h - tick); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - tick, y + h);
      // bottom-left
      ctx.moveTo(x + tick, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - tick);
      ctx.stroke();
    }
    ctx.restore();
  }

  async taAnalyze() {
    if (!this.taImage || !this.taBox || this.taPendingRequest) return;

    const canvas = this.dom.taCanvas;
    const annotatedDataUri = canvas.toDataURL('image/jpeg', 0.85);
    const xPx = Math.round(this.taBox.xN * canvas.width);
    const yPx = Math.round(this.taBox.yN * canvas.height);
    const wPx = Math.round(this.taBox.wN * canvas.width);
    const hPx = Math.round(this.taBox.hN * canvas.height);
    const w = canvas.width;
    const h = canvas.height;
    const extraContext = this.dom.taContext.value.trim();

    const coordHint = `[Object outlined in a red box at pixel (x=${xPx}, y=${yPx}, w=${wPx}, h=${hPx}) on a ${w}×${h} frame; normalized (${this.taBox.xN.toFixed(3)}, ${this.taBox.yN.toFixed(3)}, ${this.taBox.wN.toFixed(3)}, ${this.taBox.hN.toFixed(3)}).]`;
    const userText = extraContext
      ? `${THREAT_ANALYSIS_PROMPT}\n\nAdditional context: ${extraContext}\n\n${coordHint}`
      : `${THREAT_ANALYSIS_PROMPT}\n\n${coordHint}`;

    const messages = [
      { role: 'system', content: this.buildSystemContent() },
      {
        role: 'user',
        content: [
          { type: 'text', text: userText },
          { type: 'image_url', image_url: { url: annotatedDataUri } },
        ],
      },
    ];

    const resultEl = this.dom.taResult;
    resultEl.hidden = false;
    resultEl.classList.add('streaming');
    resultEl.removeAttribute('data-threat-level');
    this.dom.taResultPill.hidden = true;
    this.dom.taResultPill.removeAttribute('data-threat-level');
    this.dom.taResultPill.textContent = '';
    this.dom.taResultContent.innerHTML = '';
    this.dom.taResultStats.hidden = true;
    this.dom.taResultStats.replaceChildren();
    this.dom.taResultTime.textContent = this.formatAlertTime(new Date());

    const controller = new AbortController();
    this.taPendingRequest = controller;
    this.dom.taAnalyze.disabled = true;
    this.setStatus('Analysing marked object…', 'info');

    const requestStartedAt = performance.now();

    try {
      const result = await this.callChatCompletion({
        messages,
        signal: controller.signal,
        onDelta: (_chunk, full) => {
          this.dom.taResultContent.innerHTML = this.renderMarkdown(full);
          const level = this.extractThreatLevel(full);
          if (level && resultEl.dataset.threatLevel !== level) {
            resultEl.dataset.threatLevel = level;
            this.dom.taResultPill.hidden = false;
            this.dom.taResultPill.dataset.threatLevel = level;
            this.dom.taResultPill.textContent = `Threat · ${level.toUpperCase()}`;
          }
        },
      });
      const completedAt = performance.now();
      const reply = result.full;

      this.dom.taResultContent.innerHTML = this.renderMarkdown(reply);
      resultEl.classList.remove('streaming');

      const threatLevel = this.extractThreatLevel(reply);
      if (threatLevel) {
        resultEl.dataset.threatLevel = threatLevel;
        this.dom.taResultPill.hidden = false;
        this.dom.taResultPill.dataset.threatLevel = threatLevel;
        this.dom.taResultPill.textContent = `Threat · ${threatLevel.toUpperCase()}`;
      }

      const stats = this.computeStats({ requestStartedAt, completedAt, result });
      this.renderStats(this.dom.taResultStats, stats);

      this.addAlert({
        threatLevel,
        summary: this.summaryFromReply(reply),
        snapshotUri: annotatedDataUri,
        source: 'threat-analysis',
      });

      if (this.settings.autoRead) this.speakText(reply);
      this.setStatus('', null);
    } catch (err) {
      resultEl.classList.remove('streaming');
      if (err.name === 'AbortError') {
        this.setStatus('Analysis cancelled.', 'info');
      } else {
        const msg = this.formatError(err);
        this.dom.taResultContent.textContent = msg;
        this.setStatus(msg, 'error');
      }
    } finally {
      this.taPendingRequest = null;
      this.dom.taAnalyze.disabled = !this.taBox;
    }
  }

  // ── Change Detection sub-mode ─────────────────────────────────────────────

  setupChangeDetection() {
    this.dom.cdBaselineLoad.addEventListener('click', () => this.dom.cdBaselineInput.click());
    this.dom.cdCurrentLoad.addEventListener('click', () => this.dom.cdCurrentInput.click());
    this.dom.cdBaselineInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this.cdLoadFile('baseline', file);
      e.target.value = '';
    });
    this.dom.cdCurrentInput.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) this.cdLoadFile('current', file);
      e.target.value = '';
    });
    this.dom.cdAnalyze.addEventListener('click', () => this.cdAnalyze());
  }

  async cdLoadFile(slot, file) {
    try {
      this.setStatus(`Loading ${slot} (${file.name})…`, 'info');
      const dataUri = await this.fileToDownscaledDataUri(file);
      const img = await this.loadImage(dataUri);
      const canvas = slot === 'baseline' ? this.dom.cdBaselineCanvas : this.dom.cdCurrentCanvas;
      const wrap = slot === 'baseline' ? this.dom.cdBaselineWrap : this.dom.cdCurrentWrap;
      const readout = slot === 'baseline' ? this.dom.cdBaselineReadout : this.dom.cdCurrentReadout;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      // Paint pixels BEFORE revealing the canvas (.has-image flips display
      // from none to block). Otherwise the browser shows one frame of
      // `background: #000` through the still-transparent canvas.
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      wrap.classList.add('has-image');
      readout.textContent = `${img.naturalWidth}×${img.naturalHeight} · ${file.name}`;
      if (slot === 'baseline') {
        this.cdBaselineImage = img;
        this.cdBaselineDataUri = dataUri;
      } else {
        this.cdCurrentImage = img;
        this.cdCurrentDataUri = dataUri;
        // New current frame invalidates any prior overlay.
        this.cdDiscrepancies = [];
      }
      this.dom.cdAnalyze.disabled = !(this.cdBaselineImage && this.cdCurrentImage) || !!this.cdPendingRequest;
      this.setStatus('', null);
    } catch (err) {
      this.setStatus(`${slot} load failed: ${err.message}`, 'error');
    }
  }

  async cdAnalyze() {
    if (!this.cdBaselineImage || !this.cdCurrentImage || this.cdPendingRequest) return;

    const extraContext = this.dom.cdContext.value.trim();
    const trailingText = extraContext
      ? `${CHANGE_DETECTION_USER_PROMPT}\n\nAdditional context for the operator's facility: ${extraContext}`
      : CHANGE_DETECTION_USER_PROMPT;

    // Interleave a text label IMMEDIATELY before each image so the model can't
    // mis-bind which image is the baseline vs the current. Trailing instructions
    // come last so the operator's task summary is the most recent thing the
    // model sees before generating.
    const messages = [
      { role: 'system', content: this.buildChangeDetectionSystemContent() },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'The next image is IMAGE A — BASELINE REFERENCE (the prior, known-good photograph of the area).' },
          { type: 'image_url', image_url: { url: this.cdBaselineDataUri } },
          { type: 'text', text: 'The next image is IMAGE B — CURRENT FEED (the new photograph). All bounding-box coordinates you report describe locations on this image, IMAGE B, only.' },
          { type: 'image_url', image_url: { url: this.cdCurrentDataUri } },
          { type: 'text', text: trailingText },
        ],
      },
    ];

    const resultEl = this.dom.cdResult;
    resultEl.hidden = false;
    resultEl.classList.add('streaming');
    this.dom.cdResultContent.innerHTML = '';
    this.dom.cdResultStats.hidden = true;
    this.dom.cdResultStats.replaceChildren();
    this.dom.cdResultTime.textContent = this.formatAlertTime(new Date());

    // Reset Current Feed canvas back to the un-annotated image.
    this.cdDiscrepancies = [];
    this.cdRedrawCurrent();

    const controller = new AbortController();
    this.cdPendingRequest = controller;
    this.dom.cdAnalyze.disabled = true;
    this.setStatus('Comparing baseline vs current…', 'info');

    const requestStartedAt = performance.now();

    try {
      const result = await this.callChatCompletion({
        messages,
        signal: controller.signal,
        onDelta: (_chunk, full) => {
          this.dom.cdResultContent.innerHTML = this.renderMarkdown(full);
          // Re-parse on each delta so the canvas overlay grows alongside the
          // text. Cheap: a few regex per line, only redrawn if discrepancy
          // count changes. (No threat-level extraction here — Change Detection
          // is purely descriptive; threat assessment lives in Threat Analysis.)
          const parsed = this.cdParseDiscrepancies(full);
          if (parsed.length !== this.cdDiscrepancies.length) {
            this.cdDiscrepancies = parsed;
            this.cdRedrawCurrent();
          }
        },
      });
      const completedAt = performance.now();
      const reply = result.full;

      this.dom.cdResultContent.innerHTML = this.renderMarkdown(reply);
      resultEl.classList.remove('streaming');

      this.cdDiscrepancies = this.cdParseDiscrepancies(reply);
      this.cdRedrawCurrent();

      const stats = this.computeStats({ requestStartedAt, completedAt, result });
      this.renderStats(this.dom.cdResultStats, stats);

      if (this.settings.autoRead) this.speakText(reply);
      this.setStatus('', null);
    } catch (err) {
      resultEl.classList.remove('streaming');
      if (err.name === 'AbortError') {
        this.setStatus('Comparison cancelled.', 'info');
      } else {
        const msg = this.formatError(err);
        this.dom.cdResultContent.textContent = msg;
        this.setStatus(msg, 'error');
      }
    } finally {
      this.cdPendingRequest = null;
      this.dom.cdAnalyze.disabled = !(this.cdBaselineImage && this.cdCurrentImage);
    }
  }

  cdParseDiscrepancies(text) {
    if (!text) return [];
    const out = [];
    const seen = new Set(); // dedupe identical lines (model occasionally repeats)
    for (const rawLine of text.split('\n')) {
      const line = rawLine.replace(/\r$/, '').trim();
      if (!line) continue;
      let match = line.match(RX_DISCREPANCY_BBOX);
      if (match) {
        const [, cat, x, y, w, h, label] = match;
        const bbox = this.cdNormalizeBBox(parseFloat(x), parseFloat(y), parseFloat(w), parseFloat(h));
        if (!bbox) continue;
        const key = `${cat.toLowerCase()}|${bbox.xN.toFixed(3)}|${bbox.yN.toFixed(3)}|${label.trim().toLowerCase()}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ category: cat.toLowerCase(), label: label.trim(), bbox });
        continue;
      }
      match = line.match(RX_DISCREPANCY_NOBBOX);
      if (match) {
        const [, cat, label] = match;
        const key = `${cat.toLowerCase()}|nobbox|${label.trim().toLowerCase()}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ category: cat.toLowerCase(), label: label.trim(), bbox: null });
      }
    }
    return out;
  }

  // VLMs are unreliable about bbox format. We asked for normalized [x, y, w, h]
  // but commonly receive: 0-100 percent, 0-1000 PaliGemma scale, raw pixels, or
  // [x_min, y_min, x_max, y_max] (xyxy). Detect the format from the values and
  // canonicalise to normalized 0-1 xywh on Image B.
  cdNormalizeBBox(a, b, c, d) {
    if (![a, b, c, d].every((v) => Number.isFinite(v) && v >= 0)) return null;
    const max = Math.max(a, b, c, d);

    // Step 1: scale-detect. Pick a divisor so all four values land in [0, ~1.05].
    let scale = 1;
    if (max > 1.05) {
      if (max <= 100.5) scale = 100;            // percentage
      else if (max <= 1024.5) scale = 1000;     // PaliGemma 0-1000 (1024 close enough)
      else if (this.cdCurrentCanvas && (max <= this.cdCurrentCanvas.width || max <= this.cdCurrentCanvas.height)) {
        // Plausibly raw pixels on Image B's natural resolution.
        scale = Math.max(this.cdCurrentCanvas.width, this.cdCurrentCanvas.height);
      } else {
        // Fall back: divide by the max to bring it into range. Less accurate but
        // better than discarding the bbox entirely.
        scale = max;
      }
    }
    a /= scale; b /= scale; c /= scale; d /= scale;

    // Step 2: format-detect xywh vs xyxy.
    // If treating (c, d) as width/height would overflow the image edge by more
    // than 5%, the model probably emitted xyxy (corner-to-corner). Convert.
    const xywhOverflow = Math.max(0, (a + c) - 1) + Math.max(0, (b + d) - 1);
    const looksXyxy = (a + c > 1.05 || b + d > 1.05) && c > a && d > b;
    let xN, yN, wN, hN;
    if (looksXyxy) {
      // (a, b) = xMin, yMin; (c, d) = xMax, yMax
      xN = Math.min(a, c);
      yN = Math.min(b, d);
      wN = Math.abs(c - a);
      hN = Math.abs(d - b);
    } else if (xywhOverflow > 0.001) {
      // Slight overflow that isn't xyxy — clamp width/height to image edge.
      xN = a; yN = b;
      wN = Math.min(c, 1 - a);
      hN = Math.min(d, 1 - b);
    } else {
      xN = a; yN = b; wN = c; hN = d;
    }

    // Final clamp + sanity floor to drop sub-half-percent boxes (parser noise).
    xN = Math.max(0, Math.min(1, xN));
    yN = Math.max(0, Math.min(1, yN));
    wN = Math.max(0, Math.min(1 - xN, wN));
    hN = Math.max(0, Math.min(1 - yN, hN));
    if (wN < 0.005 || hN < 0.005) return null;
    return { xN, yN, wN, hN };
  }

  cdRedrawCurrent() {
    const canvas = this.dom.cdCurrentCanvas;
    const img = this.cdCurrentImage;
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    const longEdge = Math.max(canvas.width, canvas.height);
    const stroke = Math.max(3, Math.round(longEdge * 0.0035));
    const halo = stroke + 4;

    let drawIndex = 0;
    for (const d of this.cdDiscrepancies) {
      if (!d.bbox) continue;
      drawIndex++;
      const colour = DISCREPANCY_COLOURS[d.category] || '#cccccc';
      const x = d.bbox.xN * canvas.width;
      const y = d.bbox.yN * canvas.height;
      const w = d.bbox.wN * canvas.width;
      const h = d.bbox.hN * canvas.height;

      // Halo
      ctx.save();
      ctx.lineWidth = halo;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.strokeRect(x, y, w, h);
      // Coloured stroke
      ctx.lineWidth = stroke;
      ctx.strokeStyle = colour;
      ctx.strokeRect(x, y, w, h);

      // Numbered category badge anchored to the top-left of the box.
      const label = `${drawIndex}. ${d.category.toUpperCase()}`;
      const fontSize = Math.max(12, Math.round(longEdge * 0.018));
      ctx.font = `bold ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textBaseline = 'top';
      const padX = Math.round(fontSize * 0.4);
      const padY = Math.round(fontSize * 0.2);
      const textW = ctx.measureText(label).width;
      const badgeW = textW + padX * 2;
      const badgeH = fontSize + padY * 2;
      let bx = x;
      let by = y - badgeH - 2;
      if (by < 0) by = y + 2;
      ctx.fillStyle = colour;
      ctx.fillRect(bx, by, badgeW, badgeH);
      // Text with a black outline for legibility on any background.
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
      ctx.strokeText(label, bx + padX, by + padY);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, bx + padX, by + padY);
      ctx.restore();
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const app = new SecurityConsole(document);
  app.init();
  window.__security = app;
});

// ─── Mission Control "Home" button ────────────────────────────────────────
// Independent of the security console's main bootstrap. Posts to the parent
// when embedded in Mission Control; otherwise navigates one level up.
(function () {
  function wireHome() {
    var btn = document.getElementById('home-button');
    if (!btn) return;
    btn.addEventListener('click', function () {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: 'sima-sentry:home' }, '*');
      } else {
        window.location.assign('../index.html');
      }
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireHome, { once: true });
  } else {
    wireHome();
  }
})();

// ─── Mobile bottom navigation ─────────────────────────────────────────────
// Forwards data-tab clicks to the matching top-tab button so the existing
// tab logic owns the actual switching. The Home button posts to the
// parent (Mission Control) when embedded, otherwise navigates one up.
(function () {
  function wireBottomNav() {
    var nav = document.querySelector('.mc-bottom-nav');
    if (!nav) return;

    var tabBtns = nav.querySelectorAll('[data-tab]');
    var homeBtn = nav.querySelector('[data-action="home"]');

    tabBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = document.querySelector('[role="tab"][data-tab="' + btn.dataset.tab + '"]');
        if (target) target.click();
      });
    });

    if (homeBtn) {
      homeBtn.addEventListener('click', function () {
        if (window.parent && window.parent !== window) {
          window.parent.postMessage({ type: 'sima-sentry:home' }, '*');
        } else {
          window.location.assign('../index.html');
        }
      });
    }

    function syncActive() {
      var active = document.querySelector('[role="tab"][aria-selected="true"]');
      var tab = active && active.dataset && active.dataset.tab;
      tabBtns.forEach(function (btn) {
        if (btn.dataset.tab === tab) btn.setAttribute('aria-current', 'page');
        else btn.removeAttribute('aria-current');
      });
    }
    syncActive();
    document.querySelectorAll('[role="tab"]').forEach(function (tab) {
      tab.addEventListener('click', function () { setTimeout(syncActive, 0); });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireBottomNav, { once: true });
  } else {
    wireBottomNav();
  }
})();
