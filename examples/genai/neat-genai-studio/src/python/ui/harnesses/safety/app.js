/* SiMaSentry-Safe — zero-dependency, air-gapped occupational-safety client.
 * All state lives in browser memory + localStorage. No build step, no CDNs. */
(() => {
  "use strict";

  // ─── Constants ──────────────────────────────────────────────────────────
  const DEFAULT_SYSTEM_PROMPT =
    `You are an experienced Occupational Safety Inspector (OSHA level) with ` +
    `20 years of field experience. Detect ACTUAL hazards, missing PPE, and ` +
    `unsafe machinery operation that you can clearly see. Be concise and ` +
    `factual — not alarmist.\n\n` +
    `DEFAULT to NO VIOLATION. Only flag what a trained inspector would ` +
    `actually write up on a real inspection report. Do NOT cite:\n` +
    `  • hypothetical or speculative concerns\n` +
    `  • stylistic, cosmetic, or housekeeping preferences\n` +
    `  • issues that aren't clearly visible in the image\n` +
    `  • best-practice suggestions that aren't actual violations\n` +
    `If unsure, do not flag.\n\n` +
    `When you do find a clear violation, classify its severity:\n` +
    `  CRITICAL — imminent danger of serious injury or death (active fall ` +
    `hazard with no protection, exposed energized conductors, unguarded ` +
    `blade in operation, person in the path of moving machinery). Reserved ` +
    `for situations where someone could be seriously hurt in the next few ` +
    `minutes. Use sparingly.\n` +
    `  HIGH — clear OSHA-recordable issue needing same-shift correction.\n` +
    `  MEDIUM — genuine but non-urgent issue; correct within 24 hours.\n` +
    `  LOW — minor housekeeping issue; correct when convenient.\n\n` +
    `Calibration check before tagging CRITICAL: "Could this realistically ` +
    `kill someone in the next few minutes?" If not, downgrade. Most ` +
    `findings are LOW or MEDIUM. CRITICAL should be rare.`;

  // Human-readable name for each BCP-47 code in the Settings dropdown.
  // Used to build the per-request language lock directive.
  const LANGUAGE_NAMES = {
    "en-US": "English",
    "en-GB": "English",
    "es-ES": "Spanish",
    "es-MX": "Spanish",
    "pt-BR": "Portuguese",
    "fr-FR": "French",
    "de-DE": "German",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "ru-RU": "Russian",
    "ar-SA": "Arabic",
    "hi-IN": "Hindi",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
  };

  const ZONE_USER_PROMPT = (zoneName) =>
    `Zone ${zoneName} — Inspect this zone for clearly visible safety ` +
    `violations. DEFAULT to no violation. Only flag what a trained ` +
    `inspector would actually write up on a real inspection report.\n\n` +
    `Your reply MUST begin with EXACTLY one of these literal tokens on ` +
    `its own first line (in English uppercase, regardless of response ` +
    `language):\n` +
    `  • "VIOLATION: <SEVERITY>" — where <SEVERITY> is CRITICAL, HIGH, ` +
    `MEDIUM, or LOW. Use only if you see a clear, unambiguous violation.\n` +
    `  • "NO VIOLATION:" — use when the zone looks acceptable, when issues ` +
    `are speculative, or when nothing is clearly wrong.\n\n` +
    `On the following lines, briefly describe what you actually see. Do ` +
    `not speculate, do not cite stylistic preferences, do not invent ` +
    `concerns to fill space.`;

  // Strict header form: ZONE_USER_PROMPT now requires the model to begin
  // with EITHER "VIOLATION: <SEVERITY>" OR "NO VIOLATION:" — so we only flag
  // when the explicit violation marker is at the very start of the response.
  // We deliberately do NOT fall back on bare-word matching anymore: the model
  // often uses "violation" descriptively in prose ("to avoid a violation,
  // ensure goggles are worn") and that triggered too many false positives.
  const VIOLATION_HEADER_RE =
    /^\s*violation\s*[:\-–—]\s*(critical|high|medium|low)\b/i;

  /** Returns the severity level ("CRITICAL"/"HIGH"/"MEDIUM"/"LOW") or null. */
  const classifyViolation = (text) => {
    if (!text) return null;
    const header = VIOLATION_HEADER_RE.exec(text);
    return header ? header[1].toUpperCase() : null;
  };

  const PROVIDER_DEFAULTS = {
    openai: {
      baseUrl: "https://api.openai.com/v1/chat/completions",
      model: "gpt-4o",
    },
    ollama: {
      baseUrl: "http://localhost:11434/v1/chat/completions",
      model: "gemma3",
    },
  };

  const STORAGE_KEY = "safety-ai-harness:settings";

  const URL_PARAM_MAP = {
    provider: "provider",
    base_url: "baseUrl", baseUrl: "baseUrl",
    api_key: "apiKey", apiKey: "apiKey",
    model: "model",
    lang: "language", language: "language",
    system_prompt: "systemPrompt", systemPrompt: "systemPrompt",
    auto_read: "autoRead", autoRead: "autoRead",
  };

  // STT error codes that mean "the speech engine couldn't reach a server."
  // On air-gapped devices these are expected; surface a friendly message.
  const STT_OFFLINE_ERRORS = new Set(["network", "service-not-allowed"]);

  // Chat-template stop tokens that some servers leak into streamed delta.content
  // instead of just terminating the stream. Strip them client-side and end the
  // stream when we see one.
  const STOP_TOKEN_RE =
    /<\|eot_id\|>|<\|end_of_text\|>|<\|im_end\|>|<\|im_start\|>|<end_of_turn>|<start_of_turn>|<\/s>|<\|endoftext\|>/;
  const STOP_TOKEN_MAX_LEN = 16; // longest of the above, for split-delta lookahead

  // ─── SettingsStore ──────────────────────────────────────────────────────
  class SettingsStore extends EventTarget {
    constructor() {
      super();
      this.values = this._loadInitial();
    }

    get all() { return { ...this.values }; }
    get(key) { return this.values[key]; }

    update(patch, { persist = true, fromUrl = false } = {}) {
      const before = this.values;
      this.values = { ...before, ...patch };
      if (persist) this._persist();
      this.dispatchEvent(new CustomEvent("change", {
        detail: { before, after: this.values, patch, fromUrl },
      }));
    }

    resetSystemPrompt() {
      this.update({ systemPrompt: DEFAULT_SYSTEM_PROMPT });
    }

    _defaults() {
      return {
        provider: "ollama",
        baseUrl: PROVIDER_DEFAULTS.ollama.baseUrl,
        apiKey: "",
        model: PROVIDER_DEFAULTS.ollama.model,
        language: (navigator.language || "en-US"),
        systemPrompt: DEFAULT_SYSTEM_PROMPT,
        autoRead: false,
      };
    }

    _loadInitial() {
      const defaults = this._defaults();
      let stored = {};
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) stored = JSON.parse(raw);
      } catch (err) {
        console.warn("Settings: localStorage unreadable", err);
      }
      const merged = { ...defaults, ...stored };
      const fromUrl = this._parseUrl();
      const final = { ...merged, ...fromUrl };

      if (fromUrl.provider && !fromUrl.baseUrl) {
        final.baseUrl = PROVIDER_DEFAULTS[fromUrl.provider]?.baseUrl || final.baseUrl;
      }
      if (fromUrl.provider && !fromUrl.model && !stored.model) {
        final.model = PROVIDER_DEFAULTS[fromUrl.provider]?.model || final.model;
      }

      // ?mode=safety: lock to a local vision model if the user hasn't already
      // chosen one (via storage or URL). The safety harness is intrinsically
      // safety-themed, so the theme bullet is satisfied just by being here.
      const mode = new URLSearchParams(window.location.search).get("mode");
      if (mode === "safety") {
        if (!stored.provider && !fromUrl.provider) {
          final.provider = "ollama";
          if (!stored.baseUrl && !fromUrl.baseUrl) {
            final.baseUrl = PROVIDER_DEFAULTS.ollama.baseUrl;
          }
        }
        if (!stored.model && !fromUrl.model) {
          final.model = "llava";
        }
      }

      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(final)); }
      catch (err) { console.warn("Settings: localStorage write failed", err); }

      if (Object.keys(fromUrl).length) this._cleanUrl();
      return final;
    }

    _parseUrl() {
      const params = new URLSearchParams(window.location.search);
      const out = {};
      for (const [raw, value] of params.entries()) {
        const key = URL_PARAM_MAP[raw];
        if (!key) continue;
        if (key === "autoRead") out[key] = value === "1" || value === "true";
        else if (key === "provider") {
          if (value === "openai" || value === "ollama") out[key] = value;
        } else if (typeof value === "string") {
          out[key] = value;
        }
      }
      return out;
    }

    _cleanUrl() {
      try {
        const url = new URL(window.location.href);
        if (url.searchParams.has("api_key") || url.searchParams.has("apiKey")) {
          url.searchParams.delete("api_key");
          url.searchParams.delete("apiKey");
          window.history.replaceState({}, document.title, url.pathname + url.search + url.hash);
        }
      } catch { /* best effort */ }
    }

    _persist() {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.values)); }
      catch (err) { console.warn("Settings: localStorage write failed", err); }
    }
  }

  // ─── ChatClient ─────────────────────────────────────────────────────────
  class ApiError extends Error {
    constructor(status, body) {
      super(`HTTP ${status}: ${body?.slice?.(0, 200) ?? ""}`);
      this.status = status;
      this.body = body;
    }
  }

  class ChatClient {
    constructor(settings) { this.settings = settings; }

    buildPayload(messages) {
      const { model, systemPrompt, language } = this.settings.all;
      const langName = LANGUAGE_NAMES[language] || language || "English";
      // Authoritative language lock — overrides any "respond in user's language"
      // wording in either the default prompt or a user-edited system prompt.
      // Built fresh on every request so changing Settings → Language takes
      // effect immediately without restart.
      const langDirective =
        `Always respond exclusively in ${langName} (${language || "en-US"}). ` +
        `Do not switch languages even if the user writes in another language. ` +
        `Translate the user's input to ${langName} internally before answering.`;
      const sys = systemPrompt
        ? `${systemPrompt}\n\n${langDirective}`
        : langDirective;
      const head = [{ role: "system", content: sys }, ...messages];
      return {
        model: model || PROVIDER_DEFAULTS.ollama.model,
        messages: head,
        stream: true,
        temperature: 0.2,
      };
    }

    headers() {
      const { provider, apiKey } = this.settings.all;
      const h = { "Content-Type": "application/json" };
      if (provider === "openai" && apiKey) h.Authorization = `Bearer ${apiKey}`;
      return h;
    }

    async *stream(messages, signal) {
      const { baseUrl } = this.settings.all;
      if (!baseUrl) throw new Error("No base URL configured.");
      let response;
      try {
        response = await fetch(baseUrl, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify(this.buildPayload(messages)),
          signal,
        });
      } catch (err) {
        if (err.name === "AbortError") return;
        throw new Error(`Network error reaching ${baseUrl}: ${err.message}`);
      }

      if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new ApiError(response.status, text);
      }

      const ct = response.headers.get("content-type") || "";
      if (!ct.includes("event-stream") && !ct.includes("text/plain")) {
        const data = await response.json();
        const content = data?.choices?.[0]?.message?.content;
        if (content) yield content;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      // `pending` holds streamed content not yet emitted. We hold back up to
      // STOP_TOKEN_MAX_LEN-1 trailing chars so a stop token split across two
      // deltas (e.g. "<end_of_" + "turn>") still gets caught. When a complete
      // stop token appears, emit everything before it and terminate.
      let pending = "";
      const consume = (delta) => {
        pending += delta;
        const m = STOP_TOKEN_RE.exec(pending);
        if (m) {
          const head = pending.slice(0, m.index);
          pending = "";
          return { chunks: head ? [head] : [], terminated: true };
        }
        const chunks = [];
        if (pending.length > STOP_TOKEN_MAX_LEN - 1) {
          const safe = pending.slice(0, pending.length - (STOP_TOKEN_MAX_LEN - 1));
          pending = pending.slice(pending.length - (STOP_TOKEN_MAX_LEN - 1));
          if (safe) chunks.push(safe);
        }
        return { chunks, terminated: false };
      };

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";
          for (const raw of lines) {
            const line = raw.trim();
            if (!line || !line.startsWith("data:")) continue;
            const payload = line.slice(5).trim();
            if (payload === "[DONE]") {
              if (pending) { yield pending; pending = ""; }
              return;
            }
            let json;
            try { json = JSON.parse(payload); } catch { continue; }
            const delta = json?.choices?.[0]?.delta?.content;
            if (!delta) continue;
            const { chunks, terminated } = consume(delta);
            for (const chunk of chunks) yield chunk;
            if (terminated) return;
          }
        }
        // Stream ended without [DONE] or stop token — flush the held-back tail.
        if (pending) yield pending;
      } finally {
        try { reader.releaseLock(); } catch { /* ok */ }
      }
    }
  }

  // ─── ImageHandler ───────────────────────────────────────────────────────
  class ImageHandler {
    static MAX_RAW_BYTES = 10 * 1024 * 1024;
    static DOWNSCALE_TRIGGER = 4 * 1024 * 1024;
    static MAX_EDGE = 1536;

    async toDataUri(file) {
      if (!file.type.startsWith("image/")) {
        throw new Error("Selected file is not an image.");
      }
      if (file.size > ImageHandler.MAX_RAW_BYTES) {
        throw new Error(`Image too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is ${ImageHandler.MAX_RAW_BYTES / 1024 / 1024} MB.`);
      }
      const raw = await this._read(file);
      if (file.size <= ImageHandler.DOWNSCALE_TRIGGER) return raw;
      try {
        return await this._downscale(raw);
      } catch (err) {
        console.warn("Downscale failed, sending original:", err);
        return raw;
      }
    }

    /** Process a data URI (e.g., from a camera capture) through the same
     *  downscale pipeline as a file upload. Always JPEG out if downscaled. */
    async processDataUri(dataUri) {
      try {
        return await this._downscale(dataUri);
      } catch (err) {
        console.warn("Downscale failed, sending original:", err);
        return dataUri;
      }
    }

    _read(file) {
      return new Promise((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result));
        r.onerror = () => reject(r.error || new Error("File read failed."));
        r.readAsDataURL(file);
      });
    }

    async _downscale(dataUri) {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = dataUri;
      await img.decode();
      const longEdge = Math.max(img.width, img.height);
      if (longEdge <= ImageHandler.MAX_EDGE) return dataUri;
      const scale = ImageHandler.MAX_EDGE / longEdge;
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      return canvas.toDataURL("image/jpeg", 0.9);
    }
  }

  // ─── CameraController ──────────────────────────────────────────────────
  class CameraController {
    constructor(videoEl, panelEl) {
      this.video = videoEl;
      this.panel = panelEl;
      this.stream = null;
    }

    get supported() {
      return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    async open() {
      if (!this.supported) throw new Error("Camera not supported in this browser.");
      // Prefer the rear (environment) camera on tablets; fall back to any cam.
      const tryConstraints = [
        { video: { facingMode: { ideal: "environment" } }, audio: false },
        { video: true, audio: false },
      ];
      let lastErr;
      for (const c of tryConstraints) {
        try {
          this.stream = await navigator.mediaDevices.getUserMedia(c);
          break;
        } catch (err) {
          lastErr = err;
        }
      }
      if (!this.stream) throw lastErr || new Error("Could not open camera.");
      this.video.srcObject = this.stream;
      this.panel.hidden = false;
      try { await this.video.play(); } catch { /* autoplay blocks ignored */ }
    }

    capture() {
      if (!this.stream) throw new Error("Camera is not open.");
      const w = this.video.videoWidth || 1280;
      const h = this.video.videoHeight || 720;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(this.video, 0, 0, w, h);
      return canvas.toDataURL("image/jpeg", 0.9);
    }

    close() {
      if (this.stream) {
        for (const track of this.stream.getTracks()) track.stop();
        this.stream = null;
      }
      this.video.srcObject = null;
      this.panel.hidden = true;
    }
  }

  // ─── VoiceController ────────────────────────────────────────────────────
  class VoiceController extends EventTarget {
    constructor(settings) {
      super();
      this.settings = settings;
      this.recognition = null;
      this.listening = false;
      this.voices = [];
      this.sttSupported = false;
      this.ttsSupported = "speechSynthesis" in window;
      this._speaking = false;
      // Track the live utterance so cancelSpeech() can detach its handlers
      // before tearing it down — otherwise a delayed onend from a cancelled
      // utterance can fire AFTER the user starts a new one and prematurely
      // hide the stop button.
      this._currentUtter = null;
      this._initSTT();
      if (this.ttsSupported) this._loadVoices();
    }

    get speaking() { return this._speaking; }

    _initSTT() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) return;
      const r = new SR();
      r.continuous = false;
      r.interimResults = true;
      this.recognition = r;
      this.sttSupported = true;
    }

    _loadVoices() {
      const populate = () => { this.voices = speechSynthesis.getVoices(); };
      populate();
      try { speechSynthesis.addEventListener("voiceschanged", populate); }
      catch { speechSynthesis.onvoiceschanged = populate; }
    }

    /** Strict offline mode: prefer voices marked `localService === true`.
     *  Returns { voice, isLocal } so the caller can warn if no local voice
     *  exists for the requested language (cloud voices won't speak air-gapped). */
    pickVoice(lang) {
      if (!this.voices.length) this.voices = speechSynthesis.getVoices?.() || [];
      const localOnly = this.voices.filter(v => v.localService === true);
      const pickFrom = (list) => {
        const primary = (lang || "").split("-")[0];
        return list.find(v => v.lang === lang)
            || list.find(v => v.lang.replace("_", "-") === lang)
            || list.find(v => v.lang.startsWith(primary + "-"))
            || list.find(v => v.lang === primary)
            || list.find(v => v.default)
            || list[0]
            || null;
      };
      const local = pickFrom(localOnly);
      if (local) return { voice: local, isLocal: true };
      const any = pickFrom(this.voices);
      return { voice: any, isLocal: false };
    }

    startListening({ onPartial, onFinal, onError, onEnd }) {
      if (!this.recognition || this.listening) return;
      const r = this.recognition;
      r.lang = this.settings.get("language");
      let finalText = "";
      r.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          if (result.isFinal) finalText += result[0].transcript;
          else interim += result[0].transcript;
        }
        onPartial?.((finalText + interim).trim());
      };
      r.onerror = (e) => {
        const code = e.error || "unknown";
        const offline = STT_OFFLINE_ERRORS.has(code);
        onError?.(code, { offline });
      };
      r.onend = () => {
        this.listening = false;
        onFinal?.(finalText.trim());
        onEnd?.();
      };
      this.listening = true;
      try { r.start(); }
      catch (err) {
        this.listening = false;
        onError?.(err.message || "start failed", { offline: false });
      }
    }

    stopListening() {
      if (this.listening && this.recognition) this.recognition.stop();
    }

    /** Speak `text` in the configured language. Returns a status string the
     *  caller can surface, or null on success. */
    speak(text) {
      if (!this.ttsSupported || !text) return null;
      // Detach the previous utterance BEFORE cancelling. Otherwise its
      // onend/onerror fires later (after we've already started a new
      // utterance) and erroneously flips _speaking back to false.
      this._detachCurrent();
      speechSynthesis.cancel();

      const lang = this.settings.get("language");
      const { voice, isLocal } = this.pickVoice(lang);
      if (!voice) return "No speech voice available on this device.";
      const utter = new SpeechSynthesisUtterance(text);
      utter.voice = voice;
      utter.lang = voice.lang || lang;
      utter.rate = 1;
      utter.pitch = 1;
      utter.onstart = () => {
        if (this._currentUtter === utter) this._setSpeaking(true);
      };
      utter.onend = () => {
        if (this._currentUtter === utter) {
          this._currentUtter = null;
          this._setSpeaking(false);
        }
      };
      utter.onerror = () => {
        if (this._currentUtter === utter) {
          this._currentUtter = null;
          this._setSpeaking(false);
        }
      };
      this._currentUtter = utter;
      speechSynthesis.speak(utter);
      // Some engines fire `start` synchronously before this returns; others
      // delay. Mark speaking immediately so the stop button shows up without
      // waiting for a round-trip.
      this._setSpeaking(true);
      if (!isLocal) {
        return `No local voice for ${lang}; using "${voice.name}" (may require network).`;
      }
      return null;
    }

    cancelSpeech() {
      if (!this.ttsSupported) return;
      this._detachCurrent();
      // Some Chromium builds don't fully stop the active utterance after a
      // single cancel(); a second cancel on the next tick reliably stops it.
      // (Known bug: crbug.com/679743 and friends.)
      try { speechSynthesis.cancel(); } catch { /* ok */ }
      setTimeout(() => {
        try { speechSynthesis.cancel(); } catch { /* ok */ }
      }, 60);
      this._setSpeaking(false);
    }

    _detachCurrent() {
      const u = this._currentUtter;
      if (!u) return;
      u.onstart = null;
      u.onend = null;
      u.onerror = null;
      this._currentUtter = null;
    }

    _setSpeaking(on) {
      const next = !!on;
      if (next === this._speaking) return;
      this._speaking = next;
      this.dispatchEvent(new Event(next ? "speakstart" : "speakend"));
    }
  }

  // ─── ChatRenderer ───────────────────────────────────────────────────────
  class ChatRenderer {
    constructor(listEl, templateEl) {
      this.list = listEl;
      this.template = templateEl;
    }

    clear() {
      this.list.replaceChildren();
      document.body.classList.remove("has-messages");
    }

    addMessage(role, { text = "", imageDataUri = null, streaming = false } = {}) {
      const node = this.template.content.firstElementChild.cloneNode(true);
      node.dataset.role = role;
      node.querySelector(".message-role").textContent =
        role === "assistant" ? "Assistant" : role === "system" ? "System" : role === "error" ? "Error" : "You";
      const content = node.querySelector(".message-content");
      const attachments = node.querySelector(".message-attachments");
      const metrics = node.querySelector(".message-metrics");
      if (role !== "assistant") metrics?.remove();

      if (role === "assistant" && streaming) {
        node.classList.add("streaming");
        this._renderInto(content, text);
      } else if (role === "user") {
        content.textContent = text;
      } else {
        this._renderInto(content, text);
      }

      if (imageDataUri) {
        const img = document.createElement("img");
        img.src = imageDataUri;
        img.alt = "Attached safety image";
        attachments.appendChild(img);
      } else {
        attachments.remove();
      }

      this.list.appendChild(node);
      document.body.classList.add("has-messages");
      this._scrollToBottom();

      // Streaming buffer + rAF-coalesced re-render: parse markdown progressively
      // as tokens arrive, but at most once per animation frame.
      let buffer = role === "assistant" && streaming ? text : "";
      let rafId = null;
      const cancelPending = () => {
        if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
      };

      return {
        node,
        appendToken: (token) => {
          buffer += token;
          if (rafId != null) return;
          rafId = requestAnimationFrame(() => {
            rafId = null;
            this._renderInto(content, buffer);
            this._scrollToBottom();
          });
        },
        finalize: (fullText) => {
          cancelPending();
          node.classList.remove("streaming");
          this._renderInto(content, fullText);
          this._scrollToBottom();
        },
        replaceText: (newText) => {
          cancelPending();
          buffer = newText;
          content.textContent = newText;
        },
        setMetrics: (txt) => {
          if (metrics && metrics.isConnected) metrics.textContent = txt;
        },
      };
    }

    _scrollToBottom() {
      const main = this.list.parentElement;
      if (!main) return;
      const distance = main.scrollHeight - main.scrollTop - main.clientHeight;
      if (distance < 160) main.scrollTop = main.scrollHeight;
    }

    _renderInto(target, text) {
      target.replaceChildren();
      if (!text) return;
      const blocks = this._splitBlocks(text);
      for (const block of blocks) {
        target.appendChild(this._renderBlock(block));
      }
    }

    _splitBlocks(text) {
      const lines = text.replace(/\r\n/g, "\n").split("\n");
      const blocks = [];
      let buf = [];
      let inFence = false;
      let fenceLang = "";
      let fenceLines = [];
      const flush = () => {
        if (buf.length) {
          blocks.push({ kind: "text", lines: buf });
          buf = [];
        }
      };
      for (const line of lines) {
        const fence = /^```(\w*)\s*$/.exec(line);
        if (fence) {
          if (!inFence) {
            flush();
            inFence = true;
            fenceLang = fence[1] || "";
            fenceLines = [];
          } else {
            blocks.push({ kind: "code", lang: fenceLang, lines: fenceLines });
            inFence = false;
            fenceLang = "";
            fenceLines = [];
          }
          continue;
        }
        if (inFence) { fenceLines.push(line); continue; }
        if (line.trim() === "") {
          flush();
        } else {
          buf.push(line);
        }
      }
      if (inFence) blocks.push({ kind: "code", lang: fenceLang, lines: fenceLines });
      flush();
      return blocks;
    }

    _renderBlock(block) {
      if (block.kind === "code") {
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        if (block.lang) code.dataset.lang = block.lang;
        code.textContent = block.lines.join("\n");
        pre.appendChild(code);
        return pre;
      }
      const allBullets = block.lines.every(l => /^\s*[-*]\s+/.test(l));
      const allOrdered = block.lines.every(l => /^\s*\d+\.\s+/.test(l));
      if (allBullets || allOrdered) {
        const list = document.createElement(allOrdered ? "ol" : "ul");
        for (const line of block.lines) {
          const li = document.createElement("li");
          const inner = line.replace(/^\s*(?:[-*]|\d+\.)\s+/, "");
          this._renderInline(li, inner);
          list.appendChild(li);
        }
        return list;
      }
      const p = document.createElement("p");
      this._renderInline(p, block.lines.join(" "));
      return p;
    }

    _renderInline(parent, text) {
      const re = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\*([^*]+)\*)/g;
      let last = 0;
      let m;
      while ((m = re.exec(text))) {
        if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
        let el;
        if (m[1]) { el = document.createElement("strong"); el.textContent = m[2]; }
        else if (m[3]) { el = document.createElement("code"); el.textContent = m[4]; }
        else { el = document.createElement("em"); el.textContent = m[6]; }
        parent.appendChild(el);
        last = re.lastIndex;
      }
      if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  // ─── ZoneManager ────────────────────────────────────────────────────────
  // Click-and-drag rectangular zones on a snapshot. Two stacked canvases:
  // base (image) + overlay (zones + drag preview). Coordinates are stored in
  // image pixel space, so cropping a zone always yields the original-resolution
  // sub-image regardless of CSS display size.
  class ZoneManager {
    static PALETTE = ["#d97706", "#dc2626", "#2563eb", "#16a34a", "#7c3aed", "#db2777", "#0891b2"];
    static MIN_ZONE_PX = 12;

    constructor({ baseCanvas, overlayCanvas, listEl, emptyEl, onZoneCreated, onZoneRemoved }) {
      this.base = baseCanvas;
      this.overlay = overlayCanvas;
      this.listEl = listEl;
      this.emptyEl = emptyEl;
      this.onZoneCreated = onZoneCreated;
      this.onZoneRemoved = onZoneRemoved;
      this.image = null;
      this.zones = [];
      this._dragStart = null;
      this._dragRect = null;
      this._nextId = 1;
      this._bindPointer();
    }

    _bindPointer() {
      this.overlay.addEventListener("pointerdown", (e) => this._onPointerDown(e));
      this.overlay.addEventListener("pointermove", (e) => this._onPointerMove(e));
      this.overlay.addEventListener("pointerup", (e) => this._onPointerUp(e));
      this.overlay.addEventListener("pointercancel", () => this._cancelDrag());
    }

    setImage(htmlImage, sourceDataUri = null) {
      this.image = htmlImage;
      // Keep the original snapshot data URI around so addFullImageZone can
      // hand it to the VLM without a re-encode round-trip.
      this.imageDataUri = sourceDataUri;
      // Wipe existing zones when a new base image is loaded.
      const removed = this.zones;
      this.zones = [];
      this._nextId = 1;
      this._fitCanvases();
      this._drawBase();
      this._drawOverlay();
      this._renderList();
      if (this.emptyEl) this.emptyEl.hidden = true;
      this.base.hidden = false;
      this.overlay.hidden = false;
      for (const r of removed) this.onZoneRemoved?.(r);
    }

    /** Programmatically create a zone covering the entire image. */
    addFullImageZone() {
      if (!this.image) return null;
      const id = this._nextId++;
      const name = this._zoneName(id);
      const color = ZoneManager.PALETTE[(id - 1) % ZoneManager.PALETTE.length];
      const r = { x: 0, y: 0, w: this.image.width, h: this.image.height };
      // Reuse the original data URI if we have it — no need to re-encode the
      // full image through a canvas just to copy every pixel.
      const dataUri = this.imageDataUri || this._cropToDataUri(r);
      const zone = { id, name, color, ...r, dataUri };
      this.zones.push(zone);
      this._drawOverlay();
      this._renderList();
      this.onZoneCreated?.(zone);
      return zone;
    }

    clearAll() {
      const removed = this.zones;
      this.zones = [];
      this._nextId = 1;
      this._drawOverlay();
      this._renderList();
      for (const r of removed) this.onZoneRemoved?.(r);
    }

    removeZone(id) {
      const idx = this.zones.findIndex((z) => z.id === id);
      if (idx === -1) return;
      const [removed] = this.zones.splice(idx, 1);
      this._drawOverlay();
      this._renderList();
      this.onZoneRemoved?.(removed);
    }

    _fitCanvases() {
      const wrap = this.base.parentElement;
      // wrap has 6px padding each side under the new grid layout.
      const maxW = Math.max(120, (wrap?.clientWidth || 720) - 12);
      const maxH = Math.max(280, Math.min(560, window.innerHeight * 0.55));
      const ar = this.image.width / this.image.height;
      let dispW = Math.min(maxW, this.image.width);
      let dispH = dispW / ar;
      if (dispH > maxH) {
        dispH = maxH;
        dispW = dispH * ar;
      }
      for (const c of [this.base, this.overlay]) {
        c.width = this.image.width;
        c.height = this.image.height;
        c.style.width = dispW + "px";
        c.style.height = dispH + "px";
      }
    }

    _drawBase() {
      const ctx = this.base.getContext("2d");
      ctx.clearRect(0, 0, this.base.width, this.base.height);
      if (this.image) ctx.drawImage(this.image, 0, 0);
    }

    _drawOverlay() {
      const ctx = this.overlay.getContext("2d");
      ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
      for (const z of this.zones) this._drawZone(ctx, z);
      if (this._dragRect) this._drawPreview(ctx, this._dragRect);
    }

    _drawZone(ctx, z) {
      const lw = Math.max(2, this.overlay.width / 400);
      ctx.save();
      ctx.lineWidth = lw;
      ctx.strokeStyle = z.color;
      ctx.fillStyle = z.color + "33";
      ctx.fillRect(z.x, z.y, z.w, z.h);
      ctx.strokeRect(z.x, z.y, z.w, z.h);
      // Label badge
      const fontSize = Math.max(16, Math.round(this.overlay.width / 38));
      ctx.font = `bold ${fontSize}px -apple-system, "Segoe UI", Roboto, sans-serif`;
      const label = `Zone ${z.name}`;
      const padX = 8, padY = 4;
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = z.color;
      ctx.fillRect(z.x, Math.max(0, z.y - fontSize - padY * 2), tw + padX * 2, fontSize + padY * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, z.x + padX, Math.max(fontSize, z.y - padY));
      ctx.restore();
    }

    _drawPreview(ctx, r) {
      const lw = Math.max(2, this.overlay.width / 400);
      ctx.save();
      ctx.setLineDash([Math.max(6, lw * 3), Math.max(3, lw * 1.5)]);
      ctx.lineWidth = lw;
      ctx.strokeStyle = "#1a1408";
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.restore();
    }

    _eventToImageCoords(e) {
      const rect = this.overlay.getBoundingClientRect();
      const sx = this.overlay.width / rect.width;
      const sy = this.overlay.height / rect.height;
      return {
        x: Math.max(0, Math.min(this.overlay.width, (e.clientX - rect.left) * sx)),
        y: Math.max(0, Math.min(this.overlay.height, (e.clientY - rect.top) * sy)),
      };
    }

    _onPointerDown(e) {
      if (!this.image) return;
      e.preventDefault();
      try { this.overlay.setPointerCapture(e.pointerId); } catch { /* ok */ }
      const p = this._eventToImageCoords(e);
      this._dragStart = p;
      this._dragRect = { x: p.x, y: p.y, w: 0, h: 0 };
      this._drawOverlay();
    }

    _onPointerMove(e) {
      if (!this._dragStart) return;
      e.preventDefault();
      const p = this._eventToImageCoords(e);
      this._dragRect = {
        x: Math.min(p.x, this._dragStart.x),
        y: Math.min(p.y, this._dragStart.y),
        w: Math.abs(p.x - this._dragStart.x),
        h: Math.abs(p.y - this._dragStart.y),
      };
      this._drawOverlay();
    }

    _onPointerUp(e) {
      if (!this._dragStart || !this._dragRect) return;
      try { this.overlay.releasePointerCapture(e.pointerId); } catch { /* ok */ }
      const r = this._dragRect;
      this._dragStart = null;
      this._dragRect = null;
      if (r.w < ZoneManager.MIN_ZONE_PX || r.h < ZoneManager.MIN_ZONE_PX) {
        this._drawOverlay();
        return;
      }
      const id = this._nextId++;
      const name = this._zoneName(id);
      const color = ZoneManager.PALETTE[(id - 1) % ZoneManager.PALETTE.length];
      const dataUri = this._cropToDataUri(r);
      const zone = { id, name, color, x: r.x, y: r.y, w: r.w, h: r.h, dataUri };
      this.zones.push(zone);
      this._drawOverlay();
      this._renderList();
      this.onZoneCreated?.(zone);
    }

    _cancelDrag() {
      this._dragStart = null;
      this._dragRect = null;
      this._drawOverlay();
    }

    _zoneName(id) {
      // 1→A, 2→B, …, 26→Z, 27→AA, …
      let n = id;
      let s = "";
      while (n > 0) {
        const r = (n - 1) % 26;
        s = String.fromCharCode(65 + r) + s;
        n = Math.floor((n - 1) / 26);
      }
      return s;
    }

    _cropToDataUri(r) {
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(r.w));
      c.height = Math.max(1, Math.round(r.h));
      c.getContext("2d").drawImage(this.base, r.x, r.y, r.w, r.h, 0, 0, c.width, c.height);
      return c.toDataURL("image/jpeg", 0.9);
    }

    _renderList() {
      this.listEl.replaceChildren();
      for (const z of this.zones) {
        const li = document.createElement("li");
        li.className = "zone-item";
        li.dataset.zoneId = String(z.id);

        const dot = document.createElement("span");
        dot.className = "zone-dot";
        dot.style.background = z.color;

        const label = document.createElement("strong");
        label.textContent = `Zone ${z.name}`;

        const dim = document.createElement("span");
        dim.className = "zone-dim";
        dim.textContent = ` ${Math.round(z.w)}×${Math.round(z.h)} px`;

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "icon-button danger";
        removeBtn.setAttribute("aria-label", `Remove Zone ${z.name}`);
        removeBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M18.3 5.71 12 12l6.3 6.29-1.41 1.42L10.59 13.4 4.29 19.71 2.88 18.3 9.17 12 2.88 5.71 4.29 4.29l6.3 6.3 6.29-6.3z"/></svg>';
        removeBtn.addEventListener("click", () => this.removeZone(z.id));

        li.append(dot, label, dim, removeBtn);
        this.listEl.appendChild(li);
      }
    }
  }

  // ─── App ────────────────────────────────────────────────────────────────
  class App {
    constructor() {
      this.settings = new SettingsStore();
      this.client = new ChatClient(this.settings);
      this.images = new ImageHandler();
      this.voice = new VoiceController(this.settings);
      this.renderer = new ChatRenderer(
        document.getElementById("messages"),
        document.getElementById("message-template"),
      );

      this.messages = [];
      this.pendingImage = null;
      this.streamCtrl = null;
      this.zoneStreams = new Map(); // zoneId -> AbortController

      this._cacheElements();
      this.camera = new CameraController(this.el.cameraVideo, this.el.cameraPanel);
      this.zoneCamera = new CameraController(this.el.zoneCameraVideo, this.el.zoneCameraPanel);
      this.zones = new ZoneManager({
        baseCanvas: this.el.zoneCanvas,
        overlayCanvas: this.el.zoneOverlay,
        listEl: this.el.zoneList,
        emptyEl: this.el.canvasEmpty,
        onZoneCreated: (zone) => this._onZoneCreated(zone),
        onZoneRemoved: (zone) => this._onZoneRemoved(zone),
      });

      this._bind();
      this._applySettings();
      this._populateSettingsForm();
      this._refreshMicSupport();
      this._refreshCameraSupport();
      this._applyModeFromUrl();
    }

    _cacheElements() {
      this.el = {
        composer: document.getElementById("composer"),
        input: document.getElementById("composer-input"),
        sendBtn: document.getElementById("send-button"),
        micBtn: document.getElementById("mic-button"),
        speakerBtn: document.getElementById("speaker-button"),
        clearBtn: document.getElementById("clear-button"),
        imageBtn: document.getElementById("image-button"),
        imageInput: document.getElementById("image-input"),
        imagePreview: document.getElementById("image-preview"),
        imagePreviewImg: document.getElementById("image-preview-img"),
        imagePreviewMeta: document.getElementById("image-preview-meta"),
        imageRemove: document.getElementById("image-remove"),
        cameraBtn: document.getElementById("camera-button"),
        cameraPanel: document.getElementById("camera-panel"),
        cameraVideo: document.getElementById("camera-video"),
        cameraCapture: document.getElementById("camera-capture"),
        cameraCancel: document.getElementById("camera-cancel"),
        // Tabs (top strip + bottom nav both carry data-tab attributes)
        tabButtons: Array.from(document.querySelectorAll(".tab-button[data-tab]")),
        bottomNavItems: Array.from(document.querySelectorAll(".bottom-nav-item[data-tab]")),
        tabPanels: Array.from(document.querySelectorAll(".tab-panel[data-tab]")),
        headerHomeBtn: document.getElementById("home-button"),
        bottomHomeBtn: document.getElementById("bottom-home-btn"),
        // Compliance Monitor
        zoneUploadBtn: document.getElementById("zone-upload-btn"),
        zoneCameraBtn: document.getElementById("zone-camera-btn"),
        zoneWholeBtn: document.getElementById("zone-whole-btn"),
        zoneClearBtn: document.getElementById("zone-clear-btn"),
        zoneImageInput: document.getElementById("zone-image-input"),
        zoneCameraPanel: document.getElementById("zone-camera-panel"),
        zoneCameraVideo: document.getElementById("zone-camera-video"),
        zoneCameraCapture: document.getElementById("zone-camera-capture"),
        zoneCameraCancel: document.getElementById("zone-camera-cancel"),
        zoneCanvas: document.getElementById("zone-canvas"),
        zoneOverlay: document.getElementById("zone-overlay"),
        canvasEmpty: document.getElementById("canvas-empty"),
        zoneList: document.getElementById("zone-list"),
        complianceLog: document.getElementById("compliance-log"),
        complianceEntryTemplate: document.getElementById("compliance-entry-template"),
        complianceStatus: document.getElementById("compliance-status"),
        // Status (Inspector tab)
        status: document.getElementById("status"),
        providerChip: document.getElementById("provider-chip"),
        providerChipLabel: document.querySelector("#provider-chip .provider-chip-label"),
        settingsBtn: document.getElementById("settings-button"),
        ttsStopBtn: document.getElementById("tts-stop-button"),
        drawer: document.getElementById("settings-drawer"),
        settingsForm: document.getElementById("settings-form"),
        apiKeyField: document.getElementById("api-key-field"),
        emptyState: document.getElementById("empty-state"),
      };
    }

    _bind() {
      this.el.composer.addEventListener("submit", (e) => { e.preventDefault(); this._send(); });
      this.el.input.addEventListener("input", this._autosize);
      this.el.input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this._send();
        }
      });

      this.el.imageBtn.addEventListener("click", () => this.el.imageInput.click());
      this.el.imageInput.addEventListener("change", (e) => this._onImageSelected(e));
      this.el.imageRemove.addEventListener("click", () => this._clearPendingImage());

      this.el.cameraBtn.addEventListener("click", () => this._openCamera());
      this.el.cameraCapture.addEventListener("click", () => this._captureFromCamera());
      this.el.cameraCancel.addEventListener("click", () => this._closeCamera());

      this.el.micBtn.addEventListener("click", () => this._toggleMic());
      this.el.speakerBtn.addEventListener("click", () => this._toggleAutoRead());

      this.el.clearBtn.addEventListener("click", () => this._clearConversation());

      this.el.settingsBtn.addEventListener("click", () => this._openSettings());
      document.querySelectorAll('[data-action="open-settings"]').forEach((b) =>
        b.addEventListener("click", () => this._openSettings())
      );
      document.querySelectorAll('[data-action="close-settings"]').forEach((b) =>
        b.addEventListener("click", () => this._closeSettings())
      );
      document.querySelectorAll('[data-action="reset-system-prompt"]').forEach((b) =>
        b.addEventListener("click", () => {
          this.el.settingsForm.elements.systemPrompt.value = DEFAULT_SYSTEM_PROMPT;
        })
      );

      this.el.settingsForm.querySelectorAll('input[name="provider"]').forEach((input) =>
        input.addEventListener("change", (e) => this._onProviderChanged(e.target.value))
      );

      this.el.settingsForm.addEventListener("submit", (e) => { e.preventDefault(); this._saveSettings(); });

      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
          if (!this.el.drawer.hidden) this._closeSettings();
          else if (!this.el.cameraPanel.hidden) this._closeCamera();
        }
      });

      this.settings.addEventListener("change", () => this._applySettings());

      // Stop the camera if the page is hidden (tab switch / lock screen on a tablet).
      document.addEventListener("visibilitychange", () => {
        if (document.hidden && !this.el.cameraPanel.hidden) this._closeCamera();
        if (document.hidden && !this.el.zoneCameraPanel.hidden) this._closeZoneCamera();
      });

      // ── Tabs (top strip + bottom nav share switching logic) ──────────
      const allTabBtns = [...this.el.tabButtons, ...this.el.bottomNavItems];
      allTabBtns.forEach((btn) => {
        btn.addEventListener("click", () => this._switchTab(btn.dataset.tab));
      });
      // Arrow-key navigation only on the top strip (the bottom nav uses
      // direct taps in mobile contexts).
      this.el.tabButtons.forEach((btn) => {
        btn.addEventListener("keydown", (e) => {
          if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
          e.preventDefault();
          const idx = this.el.tabButtons.indexOf(btn);
          const next = e.key === "ArrowRight"
            ? (idx + 1) % this.el.tabButtons.length
            : (idx - 1 + this.el.tabButtons.length) % this.el.tabButtons.length;
          this.el.tabButtons[next].focus();
          this._switchTab(this.el.tabButtons[next].dataset.tab);
        });
      });
      // Mobile Home button forwards to the header Home button so any
      // future Mission Control wiring needs to be done in only one place.
      if (this.el.bottomHomeBtn) {
        this.el.bottomHomeBtn.addEventListener("click", () => {
          this.el.headerHomeBtn?.click();
        });
      }

      // ── Compliance Monitor ────────────────────────────────────────────
      this.el.zoneUploadBtn.addEventListener("click", () => this.el.zoneImageInput.click());
      this.el.zoneImageInput.addEventListener("change", (e) => this._onZoneImageSelected(e));
      this.el.zoneCameraBtn.addEventListener("click", () => this._openZoneCamera());
      this.el.zoneCameraCapture.addEventListener("click", () => this._captureZoneFromCamera());
      this.el.zoneCameraCancel.addEventListener("click", () => this._closeZoneCamera());
      this.el.zoneClearBtn.addEventListener("click", () => this._clearAllZones());
      this.el.zoneWholeBtn.addEventListener("click", () => this._useWholeImageAsZone());

      // ── TTS stop button (visible only while speech is playing) ─────────
      this.el.ttsStopBtn.addEventListener("click", () => {
        this.voice.cancelSpeech();
      });
      this.voice.addEventListener("speakstart", () => {
        if (this.voice.ttsSupported) this.el.ttsStopBtn.hidden = false;
      });
      this.voice.addEventListener("speakend", () => {
        this.el.ttsStopBtn.hidden = true;
      });
    }

    _autosize = (e) => {
      const ta = e.currentTarget;
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    };

    _applySettings() {
      const s = this.settings.all;
      this.el.providerChip.dataset.provider = s.provider;
      this.el.providerChipLabel.textContent = s.provider === "openai" ? "OpenAI" : "Ollama";
      this.el.speakerBtn.setAttribute("aria-pressed", String(!!s.autoRead));
      if (this.voice.recognition) this.voice.recognition.lang = s.language;
    }

    _populateSettingsForm() {
      const s = this.settings.all;
      const f = this.el.settingsForm.elements;
      [...f.provider].forEach((r) => { r.checked = (r.value === s.provider); });
      f.baseUrl.value = s.baseUrl || "";
      f.apiKey.value = s.apiKey || "";
      f.model.value = s.model || "";
      f.language.value = s.language || "en-US";
      f.systemPrompt.value = s.systemPrompt || DEFAULT_SYSTEM_PROMPT;
      f.autoRead.checked = !!s.autoRead;
      this._onProviderChanged(s.provider);
    }

    _onProviderChanged(provider) {
      const baseUrlInput = this.el.settingsForm.elements.baseUrl;
      const modelInput = this.el.settingsForm.elements.model;
      const defaults = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.ollama;
      const otherDefault = PROVIDER_DEFAULTS[provider === "openai" ? "ollama" : "openai"];
      if (!baseUrlInput.value || baseUrlInput.value === otherDefault.baseUrl) {
        baseUrlInput.value = defaults.baseUrl;
      }
      if (!modelInput.value || modelInput.value === otherDefault.model) {
        modelInput.value = defaults.model;
      }
      this.el.apiKeyField.classList.toggle("hidden", provider !== "openai");
    }

    _openSettings() {
      this._populateSettingsForm();
      this.el.drawer.hidden = false;
      const first = this.el.drawer.querySelector("input, select, textarea, button");
      first?.focus();
    }

    _closeSettings() { this.el.drawer.hidden = true; this.el.settingsBtn.focus(); }

    _saveSettings() {
      const f = this.el.settingsForm.elements;
      const provider = [...f.provider].find((r) => r.checked)?.value || "ollama";
      const patch = {
        provider,
        baseUrl: f.baseUrl.value.trim(),
        apiKey: f.apiKey.value,
        model: f.model.value.trim(),
        language: f.language.value,
        systemPrompt: f.systemPrompt.value,
        autoRead: !!f.autoRead.checked,
      };
      this.settings.update(patch);
      this._closeSettings();
      this._setStatus("Settings saved.", "info");
      setTimeout(() => this._setStatus(""), 1800);
    }

    _setStatus(text, tone = "") {
      this.el.status.textContent = text;
      if (tone) this.el.status.dataset.tone = tone;
      else delete this.el.status.dataset.tone;
    }

    _clearConversation() {
      if (this.streamCtrl) { this.streamCtrl.abort(); this.streamCtrl = null; }
      this.voice.cancelSpeech();
      this.messages = [];
      this.renderer.clear();
      this._clearPendingImage();
      this.el.input.value = "";
      this._autosize({ currentTarget: this.el.input });
      this._setStatus("");
    }

    // ── Image handling ───────────────────────────────────────────────────
    async _onImageSelected(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      try {
        this._setStatus(`Processing image (${(file.size / 1024).toFixed(0)} KB)…`, "info");
        const dataUri = await this.images.toDataUri(file);
        this._setPendingImage({ dataUri, name: file.name, size: file.size });
        this._setStatus("");
      } catch (err) {
        this._setStatus(err.message || "Image processing failed.", "error");
      }
    }

    _setPendingImage({ dataUri, name, size }) {
      this.pendingImage = { dataUri, name, size };
      this.el.imagePreviewImg.src = dataUri;
      const sizeLabel = size != null ? ` · ${(size / 1024).toFixed(0)} KB` : "";
      this.el.imagePreviewMeta.textContent = `${name}${sizeLabel}`;
      this.el.imagePreview.hidden = false;
    }

    _clearPendingImage() {
      this.pendingImage = null;
      this.el.imagePreviewImg.removeAttribute("src");
      this.el.imagePreviewMeta.textContent = "";
      this.el.imagePreview.hidden = true;
    }

    // ── Camera ───────────────────────────────────────────────────────────
    _refreshCameraSupport() {
      if (!this.camera.supported) {
        this.el.cameraBtn.disabled = true;
        this.el.cameraBtn.title = "Camera not supported in this browser. Serve over HTTPS or localhost.";
      }
    }

    async _openCamera() {
      if (!this.el.cameraPanel.hidden) { this._closeCamera(); return; }
      this._setStatus("Opening camera…", "info");
      try {
        await this.camera.open();
        this.el.cameraBtn.setAttribute("aria-pressed", "true");
        this._setStatus("");
      } catch (err) {
        const name = err?.name || "";
        let msg;
        if (name === "NotAllowedError") msg = "Camera permission denied.";
        else if (name === "NotFoundError") msg = "No camera found on this device.";
        else if (name === "NotReadableError") msg = "Camera is in use by another application.";
        else msg = `Camera error: ${err?.message || err}`;
        this._setStatus(msg, "error");
        this.camera.close();
      }
    }

    async _captureFromCamera() {
      try {
        const raw = this.camera.capture();
        this.camera.close();
        this.el.cameraBtn.setAttribute("aria-pressed", "false");
        this._setStatus("Processing capture…", "info");
        const dataUri = await this.images.processDataUri(raw);
        const approxSize = Math.round((dataUri.length - dataUri.indexOf(",") - 1) * 3 / 4);
        const stamp = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
        this._setPendingImage({ dataUri, name: `capture-${stamp}.jpg`, size: approxSize });
        this._setStatus("");
      } catch (err) {
        this._setStatus(`Capture failed: ${err.message || err}`, "error");
      }
    }

    _closeCamera() {
      this.camera.close();
      this.el.cameraBtn.setAttribute("aria-pressed", "false");
    }

    // ── Voice ────────────────────────────────────────────────────────────
    _refreshMicSupport() {
      if (this.voice.sttSupported) {
        this.el.micBtn.disabled = false;
        this.el.micBtn.title = "Start voice input";
      } else {
        this.el.micBtn.disabled = true;
        this.el.micBtn.title = "Speech recognition not supported in this browser.";
      }
    }

    _toggleMic() {
      if (!this.voice.sttSupported) return;
      if (this.voice.listening) {
        this.voice.stopListening();
        return;
      }
      const restoreMic = () => {
        this.el.micBtn.classList.remove("recording");
        this.el.micBtn.setAttribute("aria-pressed", "false");
        this.el.micBtn.setAttribute("aria-label", "Start voice input");
      };
      this.el.micBtn.classList.add("recording");
      this.el.micBtn.setAttribute("aria-pressed", "true");
      this.el.micBtn.setAttribute("aria-label", "Stop voice input");
      this._setStatus("Listening…", "info");
      this.voice.startListening({
        onPartial: (text) => { this.el.input.value = text; this._autosize({ currentTarget: this.el.input }); },
        onFinal: (text) => { if (text) this.el.input.value = text; },
        onError: (code, { offline } = {}) => {
          restoreMic();
          let msg;
          if (offline) {
            msg = "Speech recognition unavailable offline — type your report instead.";
          } else if (code === "not-allowed") {
            msg = "Microphone permission denied. Allow it in browser settings.";
          } else if (code === "no-speech") {
            msg = "No speech detected.";
          } else if (code === "audio-capture") {
            msg = "No microphone found on this device.";
          } else {
            msg = `Mic error: ${code}`;
          }
          this._setStatus(msg, "error");
        },
        onEnd: () => {
          restoreMic();
          if (this.el.status.textContent === "Listening…") this._setStatus("");
        },
      });
    }

    _toggleAutoRead() {
      const next = !this.settings.get("autoRead");
      this.settings.update({ autoRead: next });
      if (!next) this.voice.cancelSpeech();
    }

    // ── Send / stream ────────────────────────────────────────────────────
    async _send() {
      if (this.streamCtrl) {
        this.streamCtrl.abort();
        this.streamCtrl = null;
        return;
      }

      const text = this.el.input.value.trim();
      const image = this.pendingImage;
      if (!text && !image) return;

      const s = this.settings.all;
      if (s.provider === "openai" && !s.apiKey) {
        this._setStatus("OpenAI API key required. Open Settings to add one.", "error");
        return;
      }
      if (!s.baseUrl) {
        this._setStatus("Base URL is required. Open Settings to configure.", "error");
        return;
      }

      const userContent = image
        ? [
            { type: "text", text: text || "Analyze this image for safety hazards and PPE compliance." },
            { type: "image_url", image_url: { url: image.dataUri } },
          ]
        : text;

      this.messages.push({ role: "user", content: userContent });
      this.renderer.addMessage("user", { text: text || "(image)", imageDataUri: image?.dataUri || null });
      this.el.input.value = "";
      this._autosize({ currentTarget: this.el.input });
      this._clearPendingImage();

      const bubble = this.renderer.addMessage("assistant", { text: "", streaming: true });

      this.streamCtrl = new AbortController();
      this.el.sendBtn.querySelector("span").textContent = "Stop";
      this._setStatus("Analyzing…", "info");

      const requestStart = performance.now();
      let firstTokenAt = null;
      let tokenCount = 0;
      let acc = "";
      let aborted = false;
      let failed = false;
      try {
        for await (const token of this.client.stream(this.messages, this.streamCtrl.signal)) {
          if (firstTokenAt == null) {
            firstTokenAt = performance.now();
            bubble.setMetrics(`TTFT ${Math.round(firstTokenAt - requestStart)} ms · streaming…`);
          }
          tokenCount++;
          acc += token;
          bubble.appendToken(token);
        }
      } catch (err) {
        if (err.name === "AbortError") {
          aborted = true;
        } else if (err instanceof ApiError) {
          failed = true;
          const friendly = this._friendlyApiError(err);
          this.renderer.addMessage("error", { text: friendly });
          this._setStatus(friendly, "error");
        } else {
          failed = true;
          const msg = err.message || String(err);
          this.renderer.addMessage("error", { text: msg });
          this._setStatus(msg, "error");
        }
      } finally {
        this.streamCtrl = null;
        this.el.sendBtn.querySelector("span").textContent = "Send";
        if (!failed) this._setStatus("");
      }

      // If this turn produced no usable assistant content, roll back the user
      // turn we optimistically pushed — otherwise the next send creates two
      // consecutive user messages, which strict Llama-style chat templates
      // (e.g. LLiMa) reject with "roles must alternate".
      if (failed || (!acc && !aborted) || (aborted && !acc)) {
        bubble.node.remove();
        this.messages.pop();
        if (!failed && !acc && !aborted) {
          this.renderer.addMessage("error", { text: "Empty response from provider." });
        }
        return;
      }

      bubble.finalize(aborted ? `${acc}\n\n*(stopped)*` : acc);
      this.messages.push({ role: "assistant", content: acc });

      // Final performance metrics. Token count is SSE-delta count — a close
      // proxy for generated tokens since OpenAI/Ollama stream ~1 token per delta.
      if (firstTokenAt != null) {
        const endAt = performance.now();
        const ttftMs = Math.round(firstTokenAt - requestStart);
        const genSec = (endAt - firstTokenAt) / 1000;
        const tps = genSec > 0 ? tokenCount / genSec : 0;
        bubble.setMetrics(`TTFT ${ttftMs} ms · ${tps.toFixed(1)} tok/s · ${tokenCount} tok${aborted ? " · stopped" : ""}`);
      }

      if (!aborted && this.settings.get("autoRead") && acc) {
        const warn = this.voice.speak(acc);
        if (warn) this._setStatus(warn, "warn");
      }
    }

    // ── Tabs ─────────────────────────────────────────────────────────────
    _switchTab(name) {
      // Sync both the top tab strip and the mobile bottom nav.
      [...this.el.tabButtons, ...this.el.bottomNavItems].forEach((btn) => {
        const active = btn.dataset.tab === name;
        btn.setAttribute("aria-selected", String(active));
        btn.tabIndex = active ? 0 : -1;
      });
      this.el.tabPanels.forEach((panel) => {
        panel.hidden = panel.dataset.tab !== name;
      });
      // Stop any open camera in the leaving panel.
      if (name !== "inspector" && !this.el.cameraPanel.hidden) this._closeCamera();
      if (name !== "compliance" && !this.el.zoneCameraPanel.hidden) this._closeZoneCamera();
    }

    // ── URL ?mode=safety ────────────────────────────────────────────────
    _applyModeFromUrl() {
      const mode = new URLSearchParams(window.location.search).get("mode");
      if (mode !== "safety") return;
      document.body.classList.add("mode-safety");
      this._switchTab("compliance");
    }

    // ── Compliance: snapshot loading ─────────────────────────────────────
    async _onZoneImageSelected(e) {
      const file = e.target.files?.[0];
      e.target.value = "";
      if (!file) return;
      try {
        this._setComplianceStatus(`Loading snapshot (${(file.size / 1024).toFixed(0)} KB)…`, "info");
        const dataUri = await this.images.toDataUri(file);
        await this._loadComplianceSnapshot(dataUri);
        this._setComplianceStatus("");
      } catch (err) {
        this._setComplianceStatus(err.message || "Snapshot load failed.", "error");
      }
    }

    async _openZoneCamera() {
      if (!this.el.zoneCameraPanel.hidden) { this._closeZoneCamera(); return; }
      this._setComplianceStatus("Opening camera…", "info");
      try {
        await this.zoneCamera.open();
        this._setComplianceStatus("");
      } catch (err) {
        const name = err?.name || "";
        let msg;
        if (name === "NotAllowedError") msg = "Camera permission denied.";
        else if (name === "NotFoundError") msg = "No camera found on this device.";
        else if (name === "NotReadableError") msg = "Camera is in use by another application.";
        else msg = `Camera error: ${err?.message || err}`;
        this._setComplianceStatus(msg, "error");
        this.zoneCamera.close();
      }
    }

    async _captureZoneFromCamera() {
      try {
        const raw = this.zoneCamera.capture();
        this.zoneCamera.close();
        this._setComplianceStatus("Processing capture…", "info");
        const dataUri = await this.images.processDataUri(raw);
        await this._loadComplianceSnapshot(dataUri);
        this._setComplianceStatus("");
      } catch (err) {
        this._setComplianceStatus(`Capture failed: ${err.message || err}`, "error");
      }
    }

    _closeZoneCamera() {
      this.zoneCamera.close();
    }

    _loadComplianceSnapshot(dataUri) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          // Aborting any in-flight zone analyses; we're starting fresh.
          this._abortAllZoneAnalyses();
          this.el.complianceLog.replaceChildren();
          this.zones.setImage(img, dataUri);
          this.el.zoneWholeBtn.disabled = false;
          resolve();
        };
        img.onerror = () => reject(new Error("Failed to decode snapshot image."));
        img.src = dataUri;
      });
    }

    _useWholeImageAsZone() {
      this.zones.addFullImageZone();
    }

    _clearAllZones() {
      this._abortAllZoneAnalyses();
      this.zones.clearAll();
      this.el.complianceLog.replaceChildren();
      this._setComplianceStatus("");
    }

    _abortAllZoneAnalyses() {
      for (const ctrl of this.zoneStreams.values()) { try { ctrl.abort(); } catch { /* ok */ } }
      this.zoneStreams.clear();
    }

    // ── Compliance: zone created → analyze ───────────────────────────────
    _onZoneCreated(zone) {
      const entry = this._renderComplianceEntry(zone);
      this._analyzeZone(zone, entry);
    }

    _onZoneRemoved(zone) {
      const ctrl = this.zoneStreams.get(zone.id);
      if (ctrl) { try { ctrl.abort(); } catch { /* ok */ } this.zoneStreams.delete(zone.id); }
      const node = this.el.complianceLog.querySelector(`.compliance-entry[data-zone-id="${zone.id}"]`);
      if (node) node.remove();
    }

    _renderComplianceEntry(zone) {
      const node = this.el.complianceEntryTemplate.content.firstElementChild.cloneNode(true);
      node.dataset.zoneId = String(zone.id);
      node.classList.add("streaming");
      node.querySelector(".compliance-entry-dot").style.background = zone.color;
      node.querySelector(".compliance-entry-title").textContent = `Zone ${zone.name}`;
      const thumb = node.querySelector(".compliance-entry-thumb img");
      thumb.src = zone.dataUri;
      thumb.alt = `Zone ${zone.name} preview`;
      const content = node.querySelector(".compliance-entry-content");
      const metrics = node.querySelector(".compliance-entry-metrics");
      const violationBadge = node.querySelector(".compliance-entry-violation");

      // Buffered re-render: same pattern as ChatRenderer.
      let buffer = "";
      let rafId = null;
      const cancelPending = () => {
        if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
      };

      this.el.complianceLog.appendChild(node);
      return {
        node,
        appendToken: (token) => {
          buffer += token;
          if (rafId != null) return;
          rafId = requestAnimationFrame(() => {
            rafId = null;
            this.renderer._renderInto(content, buffer);
          });
        },
        finalize: (full) => {
          cancelPending();
          node.classList.remove("streaming");
          this.renderer._renderInto(content, full);
        },
        markViolation: (severity) => {
          const sev = (severity || "HIGH").toUpperCase();
          node.classList.add("violation", `severity-${sev.toLowerCase()}`);
          violationBadge.hidden = false;
          violationBadge.textContent = sev;
          violationBadge.dataset.severity = sev;
          node.querySelector(".compliance-entry-dot").style.background = "var(--danger)";
        },
        setMetrics: (txt) => { metrics.textContent = txt; },
        setError: (msg) => {
          cancelPending();
          node.classList.remove("streaming");
          node.classList.add("error");
          content.textContent = msg;
        },
      };
    }

    async _analyzeZone(zone, entry) {
      const s = this.settings.all;
      if (s.provider === "openai" && !s.apiKey) {
        entry.setError("OpenAI API key required. Open Settings to add one.");
        return;
      }
      if (!s.baseUrl) {
        entry.setError("Base URL is required. Open Settings to configure.");
        return;
      }

      const userContent = [
        { type: "text", text: ZONE_USER_PROMPT(zone.name) },
        { type: "image_url", image_url: { url: zone.dataUri } },
      ];
      const messages = [{ role: "user", content: userContent }];

      const ctrl = new AbortController();
      this.zoneStreams.set(zone.id, ctrl);

      const requestStart = performance.now();
      let firstTokenAt = null;
      let tokenCount = 0;
      let acc = "";
      let aborted = false;
      let failed = false;
      try {
        for await (const token of this.client.stream(messages, ctrl.signal)) {
          if (firstTokenAt == null) {
            firstTokenAt = performance.now();
            entry.setMetrics(`TTFT ${Math.round(firstTokenAt - requestStart)} ms · streaming…`);
          }
          tokenCount++;
          acc += token;
          entry.appendToken(token);
        }
      } catch (err) {
        if (err.name === "AbortError") {
          aborted = true;
        } else if (err instanceof ApiError) {
          failed = true;
          entry.setError(this._friendlyApiError(err));
        } else {
          failed = true;
          entry.setError(err.message || String(err));
        }
      } finally {
        this.zoneStreams.delete(zone.id);
      }

      if (failed) return;
      if (!acc) {
        entry.setError(aborted ? "Analysis stopped." : "Empty response from provider.");
        return;
      }

      entry.finalize(aborted ? `${acc}\n\n*(stopped)*` : acc);

      if (firstTokenAt != null) {
        const endAt = performance.now();
        const ttftMs = Math.round(firstTokenAt - requestStart);
        const genSec = (endAt - firstTokenAt) / 1000;
        const tps = genSec > 0 ? tokenCount / genSec : 0;
        entry.setMetrics(`TTFT ${ttftMs} ms · ${tps.toFixed(1)} tok/s · ${tokenCount} tok${aborted ? " · stopped" : ""}`);
      }

      if (!aborted) {
        const severity = classifyViolation(acc);
        if (severity) {
          entry.markViolation(severity);
          // Audio alert — fires regardless of the autoRead setting because this
          // is a safety alert, not a preference. Phrase varies by severity:
          // CRITICAL gets the urgent halt-operation wording; the rest are
          // labeled by severity.
          const phrase = severity === "CRITICAL"
            ? `Critical Safety Violation Detected in Zone ${zone.name}. Halt operation immediately.`
            : `${severity.charAt(0)}${severity.slice(1).toLowerCase()} severity safety violation in Zone ${zone.name}.`;
          const warn = this.voice.speak(phrase);
          if (warn) this._setComplianceStatus(warn, "warn");
        }
      }
    }

    _setComplianceStatus(text, tone = "") {
      this.el.complianceStatus.textContent = text;
      if (tone) this.el.complianceStatus.dataset.tone = tone;
      else delete this.el.complianceStatus.dataset.tone;
    }

    _friendlyApiError(err) {
      const status = err.status;
      if (status === 401) return "Authentication failed (401). Check your API key in Settings.";
      if (status === 403) return "Forbidden (403). Your key lacks access to this model.";
      if (status === 404) return "Endpoint or model not found (404). Check Base URL and Model in Settings.";
      if (status === 429) return "Rate limited (429). Try again in a moment.";
      if (status >= 500) return `Provider error (${status}). The endpoint returned a server error.`;
      return `Request failed (${status}). ${err.body?.slice(0, 240) || ""}`.trim();
    }
  }

  // ─── Boot ───────────────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => new App());
  } else {
    new App();
  }
})();

// ─── Mission Control "Home" button ────────────────────────────────────────
// Independent of the harness's main IIFE: the home button is purely a
// navigation control with no shared state. Posts to the parent when the
// harness is embedded in Mission Control; otherwise navigates one level up.
(function () {
  function wireHome() {
    var btn = document.getElementById("home-button");
    if (!btn) return;
    btn.addEventListener("click", function () {
      if (window.parent && window.parent !== window) {
        window.parent.postMessage({ type: "sima-sentry:home" }, "*");
      } else {
        window.location.assign("../index.html");
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wireHome, { once: true });
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
