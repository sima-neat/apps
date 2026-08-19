/* SiMaSentry-Med — zero-dependency client.
 * Credentials stay in memory; non-sensitive preferences use localStorage. */
(() => {
  "use strict";

  // ─── Constants ──────────────────────────────────────────────────────────
  const DEFAULT_SYSTEM_PROMPT =
    "You are an expert Medical AI and Radiologist. Provide descriptive, " +
    "objective findings based on the visual data. Always include a standard " +
    "medical disclaimer.";

  /** Human-readable names for the BCP-47 codes the harness exposes in the
   *  Settings language picker. Used to build the runtime language directive
   *  appended to the system prompt. */
  const LANGUAGE_NAMES = {
    "en-US": "English (US)",
    "en-GB": "English (British)",
    "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexican)",
    "pt-BR": "Brazilian Portuguese",
    "fr-FR": "French",
    "de-DE": "German",
    "it-IT": "Italian",
    "nl-NL": "Dutch",
    "ru-RU": "Russian",
    "ar-SA": "Arabic",
    "hi-IN": "Hindi",
    "zh-CN": "Simplified Chinese",
    "zh-TW": "Traditional Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
  };

  function languageDirective(code) {
    if (!code) return "";
    const name = LANGUAGE_NAMES[code] || code;
    return `Respond exclusively in ${name} (${code}), regardless of the language used in the user's input. Do not switch languages.`;
  }

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

  const STORAGE_KEY = "medical-ai-harness:settings";

  // Recognized URL params (and their canonical setting key).
  const URL_PARAM_MAP = {
    provider: "provider",
    base_url: "baseUrl", baseUrl: "baseUrl",
    model: "model",
    lang: "language", language: "language",
    system_prompt: "systemPrompt", systemPrompt: "systemPrompt",
    auto_read: "autoRead", autoRead: "autoRead",
    mode: "mode",
  };

  const KNOWN_MODES = new Set(["medical"]);

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
        mode: "medical", // this harness is medical-focused by default
      };
    }

    _loadInitial() {
      const defaults = this._defaults();
      let stored = {};
      try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) stored = JSON.parse(raw);
        if (!stored || typeof stored !== "object") stored = {};
        else delete stored.apiKey;
      } catch (err) {
        console.warn("Settings: localStorage unreadable", err);
      }
      const merged = { ...defaults, ...stored };
      const fromUrl = this._parseUrl();
      const final = { ...merged, ...fromUrl };

      // If the URL specified a provider but not a base URL, snap to that
      // provider's default base URL so the user gets a sane endpoint.
      if (fromUrl.provider && !fromUrl.baseUrl) {
        final.baseUrl = PROVIDER_DEFAULTS[fromUrl.provider]?.baseUrl || final.baseUrl;
      }
      // Same logic for model.
      if (fromUrl.provider && !fromUrl.model && !stored.model) {
        final.model = PROVIDER_DEFAULTS[fromUrl.provider]?.model || final.model;
      }
      // ?mode=medical resets the system prompt back to the default radiologist
      // prompt, even if the user had previously customized it. Spec'd behavior.
      if (fromUrl.mode === "medical" && !fromUrl.systemPrompt) {
        final.systemPrompt = DEFAULT_SYSTEM_PROMPT;
      }

      // Persist non-sensitive settings only. The API key stays in memory.
      this._persistValues(final);

      // Remove legacy credential parameters without consuming them.
      this._cleanUrl();

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
        } else if (key === "mode") {
          if (KNOWN_MODES.has(value)) out[key] = value;
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
      this._persistValues(this.values);
    }

    _persistValues(values) {
      const persistable = {
        provider: values.provider,
        baseUrl: values.baseUrl,
        model: values.model,
        language: values.language,
        systemPrompt: values.systemPrompt,
        autoRead: values.autoRead,
        mode: values.mode,
      };
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(persistable)); }
      catch (err) { console.warn("Settings: localStorage write failed", err); }
    }
  }

  // ─── StopTokenFilter ────────────────────────────────────────────────────
  /** Strip leaked chat-template stop tokens from streamed content. Some
   *  inference servers don't have the right `eos_token_id` configured for
   *  a given model's chat template, so tokens like `<end_of_turn>` (Gemma),
   *  `<|im_end|>` (ChatML), `<|eot_id|>` (Llama 3), `</s>` (Mistral) leak
   *  into the visible output. We hold back a small tail buffer so we can
   *  catch tokens that are split across two SSE chunks. */
  class StopTokenFilter {
    static STOP_TOKENS = [
      "<end_of_turn>", "<start_of_turn>",
      "<eos>", "<bos>",
      "<|eot_id|>", "<|begin_of_text|>",
      "<|start_header_id|>", "<|end_header_id|>",
      "<|im_start|>", "<|im_end|>",
      "<|endoftext|>",
      "<|end|>", "<|user|>", "<|assistant|>", "<|system|>",
      "</s>", "<s>",
      "[INST]", "[/INST]",
    ];
    static MAX_TOKEN_LEN = (() => {
      let m = 0;
      for (const t of StopTokenFilter.STOP_TOKENS) if (t.length > m) m = t.length;
      return m;
    })();

    constructor() {
      this.pending = "";
      this.stopped = false;
    }

    /** Feed a streamed delta. Returns text that is safe to emit. Once a
     *  full stop token is seen, the filter latches and returns "" forever. */
    feed(text) {
      if (this.stopped || !text) return "";
      this.pending += text;
      let earliest = -1;
      for (const tok of StopTokenFilter.STOP_TOKENS) {
        const idx = this.pending.indexOf(tok);
        if (idx >= 0 && (earliest < 0 || idx < earliest)) earliest = idx;
      }
      if (earliest >= 0) {
        const out = this.pending.slice(0, earliest);
        this.pending = "";
        this.stopped = true;
        return out;
      }
      // Hold back the last MAX_TOKEN_LEN chars in case they're the start of
      // a stop token split across chunks.
      if (this.pending.length <= StopTokenFilter.MAX_TOKEN_LEN) return "";
      const cut = this.pending.length - StopTokenFilter.MAX_TOKEN_LEN;
      const out = this.pending.slice(0, cut);
      this.pending = this.pending.slice(cut);
      return out;
    }

    /** Flush remaining buffer at stream end, stripping any trailing partial
     *  stop-token prefix (e.g. ending with `<end_of_` mid-token). */
    finish() {
      if (this.stopped) return "";
      let tail = this.pending;
      this.pending = "";
      let bestStrip = 0;
      for (const tok of StopTokenFilter.STOP_TOKENS) {
        const max = Math.min(tail.length, tok.length - 1);
        for (let n = max; n > bestStrip; n--) {
          if (tok.startsWith(tail.slice(tail.length - n))) {
            bestStrip = n;
            break;
          }
        }
      }
      if (bestStrip > 0) tail = tail.slice(0, tail.length - bestStrip);
      return tail;
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
      const conv = ChatClient.sanitizeAlternation(messages);
      // Compose the system message dynamically with a hard language directive
      // so the model doesn't drift to whatever language the user typed in.
      const directive = languageDirective(language);
      const sysParts = [];
      if (systemPrompt) sysParts.push(systemPrompt);
      if (directive) sysParts.push(directive);
      const sysContent = sysParts.join("\n\n");
      const head = sysContent
        ? [{ role: "system", content: sysContent }, ...conv]
        : conv;
      return {
        model: model || "gpt-4o",
        messages: head,
        stream: true,
        stream_options: { include_usage: true },
        temperature: 0.2,
      };
    }

    /** Enforce strict user/assistant/user/assistant alternation, starting with
     *  user. Templates like Gemma 3's raise on any deviation. We drop leading
     *  non-user turns and replace (not duplicate) on consecutive same-role
     *  messages, keeping the most recent. Vision/tool messages keep their
     *  array content untouched. */
    static sanitizeAlternation(messages) {
      const out = [];
      for (const m of messages) {
        if (m.role !== "user" && m.role !== "assistant") continue;
        if (out.length === 0) {
          if (m.role === "user") out.push(m);
          continue;
        }
        const expected = out[out.length - 1].role === "user" ? "assistant" : "user";
        if (m.role === expected) out.push(m);
        else out[out.length - 1] = m; // collapse repeated same-role
      }
      return out;
    }

    headers() {
      const { provider, apiKey } = this.settings.all;
      const h = { "Content-Type": "application/json" };
      if (provider === "openai" && apiKey) h.Authorization = `Bearer ${apiKey}`;
      return h;
    }

    async *stream(messages, { signal, onUsage } = {}) {
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

      const filter = new StopTokenFilter();
      const ct = response.headers.get("content-type") || "";
      if (!ct.includes("event-stream") && !ct.includes("text/plain")) {
        // Non-streaming JSON fallback.
        const data = await response.json();
        if (data?.usage && typeof onUsage === "function") onUsage(data.usage);
        const content = data?.choices?.[0]?.message?.content;
        if (content) {
          const safe = filter.feed(content);
          if (safe) yield safe;
        }
        const tail = filter.finish();
        if (tail) yield tail;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let stopped = false;
      while (!stopped) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const raw of lines) {
          const line = raw.trim();
          if (!line || !line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") { stopped = true; break; }
          let json;
          try { json = JSON.parse(payload); } catch { continue; }
          if (json?.usage && typeof onUsage === "function") onUsage(json.usage);
          const delta = json?.choices?.[0]?.delta?.content;
          if (delta) {
            const safe = filter.feed(delta);
            if (safe) yield safe;
            if (filter.stopped) { stopped = true; break; }
          }
        }
      }
      const tail = filter.finish();
      if (tail) yield tail;
    }
  }

  // ─── ImageHandler ───────────────────────────────────────────────────────
  class ImageHandler {
    static MAX_RAW_BYTES = 10 * 1024 * 1024; // 10 MB
    static DOWNSCALE_TRIGGER = 4 * 1024 * 1024; // 4 MB
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
      this.speaking = false;
      this._initSTT();
      if (this.ttsSupported) this._loadVoices();
    }

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

    /** Strict-local voice picker for air-gapped operation.
     *  Returns { voice, exact, matchedPrimary, fallback } so callers can warn
     *  when the chosen voice doesn't match the requested language. */
    pickVoice(lang) {
      if (!this.voices.length) this.voices = speechSynthesis.getVoices?.() || [];
      // Strict: only voices that run on-device. Excludes Chrome's Google cloud voices.
      const local = this.voices.filter(v => v.localService === true);
      // Resilience: if a browser doesn't populate `localService` truthfully for
      // any voice (rare), fall back to the full list so TTS isn't dead in the
      // water — but flag it so we don't pretend it's verified-local.
      const pool = local.length ? local : this.voices;
      const verifiedLocal = local.length > 0;
      const primary = (lang || "").split("-")[0];

      const exact = pool.find(v => v.lang === lang)
          || pool.find(v => v.lang.replace("_", "-") === lang);
      const partial = !exact && (
            pool.find(v => v.lang.startsWith(primary + "-"))
         || pool.find(v => v.lang === primary)
      );
      const fallback = !exact && !partial && (
            pool.find(v => v.default)
         || pool[0]
         || null
      );
      const chosen = exact || partial || fallback || null;

      return {
        voice: chosen,
        exact: !!exact,
        matchedPrimary: !!partial,
        fallback: !!fallback,
        verifiedLocal,
      };
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
      r.onerror = (e) => onError?.(e.error || "unknown");
      r.onend = () => {
        this.listening = false;
        onFinal?.(finalText.trim());
        onEnd?.();
      };
      this.listening = true;
      try { r.start(); }
      catch (err) {
        this.listening = false;
        onError?.(err.message || "start failed");
      }
    }

    stopListening() {
      if (this.listening && this.recognition) this.recognition.stop();
    }

    speak(text) {
      if (!this.ttsSupported || !text) return null;
      speechSynthesis.cancel();
      const lang = this.settings.get("language");
      const utter = new SpeechSynthesisUtterance(text);
      const result = this.pickVoice(lang);
      if (result.voice) utter.voice = result.voice;
      utter.lang = (result.voice && result.voice.lang) || lang;
      utter.rate = 1;
      utter.pitch = 1;
      utter.onstart = () => {
        this.speaking = true;
        this.dispatchEvent(new CustomEvent("speakstart"));
      };
      const finish = () => {
        if (!this.speaking) return;
        this.speaking = false;
        this.dispatchEvent(new CustomEvent("speakend"));
      };
      utter.onend = finish;
      utter.onerror = finish;
      speechSynthesis.speak(utter);
      return result;
    }

    cancelSpeech() {
      if (!this.ttsSupported) return;
      speechSynthesis.cancel();
      // Some browsers don't fire onend when cancel() is called — emit ourselves.
      if (this.speaking) {
        this.speaking = false;
        this.dispatchEvent(new CustomEvent("speakend"));
      }
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

      let acc = text;
      let rafId = 0;

      const renderNow = () => {
        rafId = 0;
        this._renderInto(content, acc);
        if (node.classList.contains("streaming")) this._appendStreamCursor(content);
        this._scrollToBottom();
      };
      const scheduleRender = () => {
        if (rafId) return;
        rafId = requestAnimationFrame(renderNow);
      };

      if (role === "assistant" && streaming) {
        node.classList.add("streaming");
        if (acc) this._renderInto(content, acc);
        this._appendStreamCursor(content);
      } else if (role === "user") {
        content.textContent = text;
      } else {
        this._renderInto(content, text);
      }

      if (imageDataUri) {
        const img = document.createElement("img");
        img.src = imageDataUri;
        img.alt = "Attached medical image";
        attachments.appendChild(img);
      } else {
        attachments.remove();
      }

      this.list.appendChild(node);
      document.body.classList.add("has-messages");
      this._scrollToBottom();
      return {
        node,
        appendToken: (token) => {
          acc += token;
          scheduleRender();
        },
        finalize: (fullText) => {
          if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
          acc = fullText;
          node.classList.remove("streaming");
          this._renderInto(content, fullText);
          this._scrollToBottom();
        },
        replaceText: (newText) => {
          if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
          acc = newText;
          content.textContent = newText;
        },
        setMetrics: (metrics) => {
          const slot = node.querySelector(".message-metrics");
          if (!slot || !metrics || !metrics.length) return;
          slot.replaceChildren();
          for (const { label, value, title } of metrics) {
            const m = document.createElement("span");
            m.className = "metric";
            if (title) m.title = title;
            const l = document.createElement("span");
            l.className = "metric-label";
            l.textContent = label;
            const v = document.createElement("span");
            v.className = "metric-value";
            v.textContent = value;
            m.append(l, v);
            slot.appendChild(m);
          }
          slot.hidden = false;
          this._scrollToBottom();
        },
      };
    }

    /** Append a blinking cursor inside the last block-level child, so it
     *  flows on the same line as the last visible text. */
    _appendStreamCursor(content) {
      const cursor = document.createElement("span");
      cursor.className = "stream-cursor";
      cursor.setAttribute("aria-hidden", "true");
      cursor.textContent = "▍";
      const last = content.lastElementChild;
      if (!last) { content.appendChild(cursor); return; }
      if (last.tagName === "P" || last.tagName === "LI") {
        last.appendChild(cursor);
      } else if (last.tagName === "UL" || last.tagName === "OL") {
        const lastLi = last.lastElementChild;
        (lastLi || content).appendChild(cursor);
      } else {
        content.appendChild(cursor);
      }
    }

    _scrollToBottom() {
      const main = this.list.parentElement;
      if (!main) return;
      // Only auto-scroll if the user is already near the bottom.
      const distance = main.scrollHeight - main.scrollTop - main.clientHeight;
      if (distance < 160) main.scrollTop = main.scrollHeight;
    }

    /** Render lightweight markdown safely into `target`. */
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
      // text block — could be a list or a paragraph
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

    /** Inline tokens: **bold**, *italic*, `code`. Always via DOM nodes. */
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

  // ─── CameraCapture ──────────────────────────────────────────────────────
  /** Live-camera capture via getUserMedia. Renders a modal with the camera
   *  preview, captures a still frame to a JPEG data URI on demand, and feeds
   *  it through the same image pipeline as a file upload. */
  class CameraCapture {
    constructor({ onCapture, onError }) {
      this.onCapture = onCapture;
      this.onError = onError;
      this.stream = null;
      this.facingMode = "environment"; // rear camera by default
      this._cacheElements();
      this._bind();
    }

    _cacheElements() {
      this.el = {
        modal: document.getElementById("camera-modal"),
        video: document.getElementById("camera-video"),
        error: document.getElementById("camera-error"),
        capture: document.getElementById("camera-capture"),
        switchBtn: document.getElementById("camera-switch"),
      };
    }

    _bind() {
      this.el.capture.addEventListener("click", () => this._capture());
      this.el.switchBtn.addEventListener("click", () => this._switchCamera());
      document.querySelectorAll('[data-action="close-camera"]').forEach((b) =>
        b.addEventListener("click", () => this.close())
      );
      document.addEventListener("keydown", (e) => {
        if (!this.el.modal.hidden && e.key === "Escape") this.close();
      });
    }

    isSupported() {
      return !!(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function");
    }

    async open() {
      if (!this.isSupported()) {
        this.onError?.("Camera not available in this browser. Use 'Choose image' instead.");
        return;
      }
      this.el.modal.hidden = false;
      this.el.error.hidden = true;
      this.el.error.textContent = "";
      await this._startStream();
    }

    async _startStream() {
      this._stopStream();
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: this.facingMode },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
          },
          audio: false,
        });
        this.el.video.srcObject = this.stream;
      } catch (err) {
        // If "environment" was rejected (most laptops have only a user-facing
        // camera), retry without the constraint before giving up.
        if (this.facingMode === "environment" && err.name === "OverconstrainedError") {
          this.facingMode = "user";
          try {
            this.stream = await navigator.mediaDevices.getUserMedia({
              video: { facingMode: { ideal: "user" } },
              audio: false,
            });
            this.el.video.srcObject = this.stream;
            return;
          } catch (err2) { err = err2; }
        }
        this._showError(this._friendlyError(err));
      }
    }

    _stopStream() {
      if (this.stream) {
        for (const t of this.stream.getTracks()) t.stop();
        this.stream = null;
      }
      if (this.el.video) this.el.video.srcObject = null;
    }

    async _switchCamera() {
      this.facingMode = this.facingMode === "environment" ? "user" : "environment";
      this.el.error.hidden = true;
      await this._startStream();
    }

    _capture() {
      if (!this.stream) return;
      const video = this.el.video;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) {
        this._showError("Camera not ready yet — try again in a moment.");
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, w, h);
      const dataUri = canvas.toDataURL("image/jpeg", 0.92);
      const name = `capture-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.jpg`;
      const size = Math.round(dataUri.length * 0.75); // base64 → bytes estimate
      this.onCapture?.({ dataUri, name, size });
      this.close();
    }

    _showError(message) {
      this.el.error.textContent = message;
      this.el.error.hidden = false;
      this._stopStream();
    }

    _friendlyError(err) {
      if (!err) return "Failed to start camera.";
      if (err.name === "NotAllowedError" || err.name === "SecurityError")
        return "Camera permission denied. Allow camera access in your browser settings, then try again.";
      if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError")
        return "No camera detected on this device.";
      if (err.name === "NotReadableError" || err.name === "TrackStartError")
        return "Camera is in use by another application.";
      if (err.name === "OverconstrainedError" || err.name === "ConstraintNotSatisfiedError")
        return "Requested camera setup unavailable. Try Switch Camera.";
      return err.message || "Failed to start camera.";
    }

    close() {
      this._stopStream();
      this.el.modal.hidden = true;
    }
  }

  // ─── WorkbenchController ────────────────────────────────────────────────
  class WorkbenchController {
    constructor({ app }) {
      this.app = app;
      this.image = null;        // { dataUri, name, naturalWidth, naturalHeight }
      this.findings = [];       // [{ id, x, y, status, text, metrics }]
      this.selectedId = null;
      this._cacheElements();
      this.findingTemplate = document.getElementById("finding-template");
      this._bind();
    }

    _cacheElements() {
      this.el = {
        stage: document.getElementById("viewer-stage"),
        frame: document.getElementById("viewer-frame"),
        img: document.getElementById("viewer-image"),
        overlay: document.getElementById("viewer-overlay"),
        markersGroup: document.getElementById("viewer-markers"),
        empty: document.getElementById("viewer-empty"),
        coords: document.getElementById("viewer-coords"),
        hint: document.getElementById("viewer-hint"),
        list: document.getElementById("findings-list"),
        listEmpty: document.getElementById("findings-empty"),
        count: document.getElementById("findings-count"),
        replace: document.getElementById("viewer-replace"),
        clear: document.getElementById("viewer-clear"),
        upload: document.getElementById("viewer-upload"),
        status: document.getElementById("workbench-status"),
      };
    }

    _bind() {
      this.el.frame.addEventListener("click", (e) => this._onClick(e));
      // Pointer events cover mouse, touch, and pen with one API.
      this.el.img.addEventListener("pointermove", (e) => this._onMove(e));
      this.el.img.addEventListener("pointerleave", () => { this.el.coords.textContent = ""; });
      // On touchscreens, surface coords briefly on first contact too so the
      // user sees what they're about to analyze before the click fires.
      this.el.img.addEventListener("pointerdown", (e) => this._onMove(e));
      this.el.clear.addEventListener("click", () => this.clearFindings());
      this.el.replace.addEventListener("click", () => this.app.triggerImageUpload());
      this.el.upload.addEventListener("click", () => this.app.triggerImageUpload());

      // When the image finishes loading we have the natural dimensions.
      this.el.img.addEventListener("load", () => this._onImageLoaded());

      // Re-fit the frame whenever the stage resizes — including when the
      // Workbench tab becomes visible after being hidden (clientWidth was 0).
      if (typeof ResizeObserver !== "undefined") {
        this._ro = new ResizeObserver(() => this._fitFrame());
        this._ro.observe(this.el.stage);
      } else {
        window.addEventListener("resize", () => this._fitFrame());
      }
      this._refreshButtons();
    }

    /** Called by App after switching to the Workbench tab — gives us a chance
     *  to compute frame dimensions now that the stage is actually visible. */
    onShown() { this._fitFrame(); }

    setImage(image) {
      if (!image) {
        this.image = null;
        this.el.img.removeAttribute("src");
        this.el.frame.hidden = true;
        this.el.frame.style.removeProperty("width");
        this.el.frame.style.removeProperty("height");
        this.el.empty.hidden = false;
        this.el.stage.dataset.hasImage = "false";
        this.clearFindings();
        this._refreshButtons();
        return;
      }
      this.image = { ...image, naturalWidth: 0, naturalHeight: 0 };
      this.el.img.src = image.dataUri;
      this.el.empty.hidden = true;
      this.el.frame.hidden = false;
      this.el.stage.dataset.hasImage = "true";
      this.clearFindings();
      this._refreshButtons();
    }

    _onImageLoaded() {
      if (!this.image) return;
      this.image.naturalWidth = this.el.img.naturalWidth || 1;
      this.image.naturalHeight = this.el.img.naturalHeight || 1;
      this.el.overlay.setAttribute("viewBox", `0 0 ${this.image.naturalWidth} ${this.image.naturalHeight}`);
      this._fitFrame();
    }

    /** Compute explicit pixel dimensions for the frame so it's never 0×0.
     *  Letterboxes the image inside the stage, preserving aspect ratio. */
    _fitFrame() {
      if (!this.image || !this.image.naturalWidth) return;
      const stage = this.el.stage;
      const sw = stage.clientWidth;
      const sh = stage.clientHeight;
      if (sw <= 0 || sh <= 0) return; // stage hidden or unmeasured
      const w = this.image.naturalWidth;
      const h = this.image.naturalHeight;
      const ratio = w / h;
      const stageRatio = sw / sh;
      const fw = ratio > stageRatio ? sw : sh * ratio;
      const fh = ratio > stageRatio ? sw / ratio : sh;
      this.el.frame.style.width = `${Math.floor(fw)}px`;
      this.el.frame.style.height = `${Math.floor(fh)}px`;
    }

    _refreshButtons() {
      const has = !!this.image;
      this.el.clear.disabled = !has || this.findings.length === 0;
      this.el.replace.disabled = !has;
      this.el.hint.textContent = has
        ? "Click anywhere on the scan to analyze that region."
        : "Upload a scan to begin point analysis.";
    }

    _onMove(e) {
      if (!this.image) return;
      const rect = this.el.img.getBoundingClientRect();
      if (rect.width === 0) return;
      const x = Math.round((e.clientX - rect.left) * (this.image.naturalWidth / rect.width));
      const y = Math.round((e.clientY - rect.top) * (this.image.naturalHeight / rect.height));
      this.el.coords.textContent = `(${x}, ${y})`;
    }

    _onClick(e) {
      if (!this.image) return;
      // Don't trigger when clicking an existing marker.
      if (e.target.closest(".marker")) return;
      const rect = this.el.img.getBoundingClientRect();
      if (rect.width === 0) return;
      const xDisplay = e.clientX - rect.left;
      const yDisplay = e.clientY - rect.top;
      if (xDisplay < 0 || yDisplay < 0 || xDisplay > rect.width || yDisplay > rect.height) return;
      const x = Math.round(xDisplay * (this.image.naturalWidth / rect.width));
      const y = Math.round(yDisplay * (this.image.naturalHeight / rect.height));
      this.startAnalysis(x, y);
    }

    startAnalysis(x, y) {
      if (!this.image) return;
      const finding = {
        id: "f-" + Date.now() + "-" + Math.random().toString(36).slice(2, 7),
        x, y,
        status: "pending",
        text: "Analyzing…",
        metrics: null,
      };
      this.findings.push(finding);
      this._renderFinding(finding);
      this._renderMarker(finding);
      this._select(finding.id);
      this._scrollToFinding(finding.id);
      this._refreshButtons();

      const w = this.image.naturalWidth;
      const h = this.image.naturalHeight;
      const prompt =
        `Analyze the area around these specific coordinates [x=${x}, y=${y}] ` +
        `on the attached medical image (image dimensions ${w}×${h} pixels) for ` +
        `potential medical anomalies. Identify the anatomy at this location if ` +
        `visible, describe any findings objectively, and include the standard ` +
        `medical disclaimer.`;

      this.app._send({
        text: prompt,
        imageDataUri: this.image.dataUri,
        imageName: this.image.name,
        displayText: `Analyze region near (${x}, ${y})`,
        // Don't show the image in the chat bubble — it's already on the workbench.
        displayImageDataUri: null,
        onProgress: (acc) => {
          finding.text = acc;
          this._renderFindingText(finding);
        },
        onComplete: ({ acc, aborted, failed, metrics }) => {
          if (failed) {
            finding.status = "error";
            finding.text = "Analysis failed. See chat tab for details.";
          } else if (aborted && !acc) {
            // Cancelled before any output — drop the orphan finding.
            this._removeFinding(finding.id);
            return;
          } else {
            finding.status = aborted ? "complete" : "complete";
            finding.text = acc;
            finding.metrics = metrics;
          }
          this._renderFinding(finding);
          this._renderMarker(finding);
        },
      });
    }

    _renderMarker(finding) {
      const SVG_NS = "http://www.w3.org/2000/svg";
      let g = this.el.markersGroup.querySelector(`[data-finding-id="${finding.id}"]`);
      if (!g) {
        g = document.createElementNS(SVG_NS, "g");
        g.setAttribute("data-finding-id", finding.id);
        g.classList.add("marker");
        const circle = document.createElementNS(SVG_NS, "circle");
        circle.classList.add("marker-circle");
        const dot = document.createElementNS(SVG_NS, "circle");
        dot.classList.add("marker-dot");
        const label = document.createElementNS(SVG_NS, "text");
        label.classList.add("marker-label");
        g.append(circle, dot, label);
        g.addEventListener("click", (e) => {
          e.stopPropagation();
          this._select(finding.id);
          this._scrollToFinding(finding.id);
        });
        this.el.markersGroup.appendChild(g);
      }
      g.dataset.status = finding.status;
      g.classList.toggle("selected", this.selectedId === finding.id);

      const w = this.image.naturalWidth;
      const h = this.image.naturalHeight;
      const radius = Math.max(28, Math.min(w, h) * 0.025);
      const circle = g.querySelector(".marker-circle");
      circle.setAttribute("cx", finding.x);
      circle.setAttribute("cy", finding.y);
      circle.setAttribute("r", radius);
      const dot = g.querySelector(".marker-dot");
      dot.setAttribute("cx", finding.x);
      dot.setAttribute("cy", finding.y);
      dot.setAttribute("r", Math.max(3, radius * 0.12));
      const label = g.querySelector(".marker-label");
      label.setAttribute("x", finding.x);
      label.setAttribute("y", finding.y - radius - radius * 0.35);
      label.textContent = String(this.findings.findIndex(f => f.id === finding.id) + 1);
    }

    _renderFinding(finding) {
      let li = this.el.list.querySelector(`[data-finding-id="${finding.id}"]`);
      if (!li) {
        const tpl = this.findingTemplate.content.firstElementChild.cloneNode(true);
        tpl.dataset.findingId = finding.id;
        tpl.addEventListener("click", () => this._select(finding.id));
        tpl.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            this._select(finding.id);
          }
        });
        this.el.list.appendChild(tpl);
        li = tpl;
      }
      li.dataset.status = finding.status;
      li.classList.toggle("selected", this.selectedId === finding.id);
      const idx = this.findings.findIndex(f => f.id === finding.id) + 1;
      li.querySelector(".finding-num").textContent = String(idx);
      li.querySelector(".finding-coords").textContent =
        `(${finding.x}, ${finding.y}) · ${finding.status === "pending" ? "analyzing…" : finding.status}`;
      this._renderFindingText(finding);
      this._renderFindingMetrics(finding);
      this._updateCount();
    }

    _renderFindingText(finding) {
      const li = this.el.list.querySelector(`[data-finding-id="${finding.id}"]`);
      if (!li) return;
      const target = li.querySelector(".finding-text");
      this.app.renderer._renderInto(target, finding.text || "");
    }

    _renderFindingMetrics(finding) {
      const li = this.el.list.querySelector(`[data-finding-id="${finding.id}"]`);
      if (!li) return;
      const slot = li.querySelector(".finding-metrics");
      if (!finding.metrics || !finding.metrics.length) {
        slot.hidden = true;
        slot.replaceChildren();
        return;
      }
      slot.replaceChildren();
      // Compact subset for the side panel: TTFT and TPS are most useful.
      const wanted = new Set(["TTFT", "TPS"]);
      for (const { label, value, title } of finding.metrics) {
        if (!wanted.has(label)) continue;
        const m = document.createElement("span");
        m.className = "metric";
        if (title) m.title = title;
        const l = document.createElement("span");
        l.className = "metric-label";
        l.textContent = label;
        const v = document.createElement("span");
        v.className = "metric-value";
        v.textContent = value;
        m.append(l, v);
        slot.appendChild(m);
      }
      slot.hidden = false;
    }

    _select(id) {
      this.selectedId = id;
      this.el.list.querySelectorAll(".finding").forEach((el) => {
        el.classList.toggle("selected", el.dataset.findingId === id);
      });
      this.el.markersGroup.querySelectorAll(".marker").forEach((g) => {
        g.classList.toggle("selected", g.dataset.findingId === id);
      });
    }

    _scrollToFinding(id) {
      const li = this.el.list.querySelector(`[data-finding-id="${id}"]`);
      if (li) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    _removeFinding(id) {
      this.findings = this.findings.filter(f => f.id !== id);
      const li = this.el.list.querySelector(`[data-finding-id="${id}"]`);
      if (li) li.remove();
      const m = this.el.markersGroup.querySelector(`[data-finding-id="${id}"]`);
      if (m) m.remove();
      if (this.selectedId === id) this.selectedId = null;
      this._updateCount();
      // Renumber remaining
      this.findings.forEach((f) => {
        this._renderMarker(f);
        const li2 = this.el.list.querySelector(`[data-finding-id="${f.id}"]`);
        if (li2) {
          const idx = this.findings.findIndex(x => x.id === f.id) + 1;
          li2.querySelector(".finding-num").textContent = String(idx);
        }
      });
      this._refreshButtons();
    }

    _updateCount() { this.el.count.textContent = String(this.findings.length); }

    clearFindings() {
      this.findings = [];
      this.selectedId = null;
      this.el.markersGroup.replaceChildren();
      this.el.list.replaceChildren();
      this._updateCount();
      this._refreshButtons();
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

      this._cacheElements();
      this.workbench = new WorkbenchController({ app: this });
      this.camera = new CameraCapture({
        onCapture: (image) => this._useImage(image),
        onError: (msg) => this._setStatus(msg, "error"),
      });
      this._bind();
      this._applySettings();
      this._populateSettingsForm();
      this._refreshMicSupport();
      this._setTab("chat");
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
        cameraBtn: document.getElementById("camera-button"),
        imageInput: document.getElementById("image-input"),
        imagePreview: document.getElementById("image-preview"),
        imagePreviewImg: document.getElementById("image-preview-img"),
        imagePreviewMeta: document.getElementById("image-preview-meta"),
        imageRemove: document.getElementById("image-remove"),
        viewerCameraBtn: document.getElementById("viewer-camera"),
        ttsControls: document.getElementById("tts-controls"),
        stopTtsBtn: document.getElementById("stop-tts-button"),
        stopVoiceBtn: document.getElementById("stop-voice-button"),
        status: document.getElementById("status"),
        providerChip: document.getElementById("provider-chip"),
        providerChipLabel: document.querySelector("#provider-chip .provider-chip-label"),
        settingsBtn: document.getElementById("settings-button"),
        drawer: document.getElementById("settings-drawer"),
        settingsForm: document.getElementById("settings-form"),
        apiKeyField: document.getElementById("api-key-field"),
        emptyState: document.getElementById("empty-state"),
      };
    }

    _bind() {
      // Composer
      this.el.composer.addEventListener("submit", (e) => { e.preventDefault(); this._send(); });
      this.el.input.addEventListener("input", this._autosize);
      this.el.input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this._send();
        }
      });

      // Image
      this.el.imageBtn.addEventListener("click", () => this.el.imageInput.click());
      this.el.imageInput.addEventListener("change", (e) => this._onImageSelected(e));
      this.el.imageRemove.addEventListener("click", () => this._clearPendingImage());

      // Camera
      this.el.cameraBtn?.addEventListener("click", () => this.triggerCameraCapture());
      this.el.viewerCameraBtn?.addEventListener("click", () => this.triggerCameraCapture());
      if (!this.camera.isSupported()) {
        if (this.el.cameraBtn) {
          this.el.cameraBtn.disabled = true;
          this.el.cameraBtn.title = "Camera not supported in this browser.";
        }
        if (this.el.viewerCameraBtn) {
          this.el.viewerCameraBtn.disabled = true;
          this.el.viewerCameraBtn.title = "Camera not supported in this browser.";
        }
      }

      // Voice
      this.el.micBtn.addEventListener("click", () => this._toggleMic());
      this.el.speakerBtn.addEventListener("click", () => this._toggleAutoRead());

      // Clear
      this.el.clearBtn.addEventListener("click", () => this._clearConversation());

      // Settings drawer
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

      // Provider radio toggles base URL placeholder + api key visibility
      this.el.settingsForm.querySelectorAll('input[name="provider"]').forEach((input) =>
        input.addEventListener("change", (e) => this._onProviderChanged(e.target.value))
      );

      this.el.settingsForm.addEventListener("submit", (e) => { e.preventDefault(); this._saveSettings(); });

      // Esc closes drawer / cancels TTS
      document.addEventListener("keydown", (e) => {
        if (e.key !== "Escape") return;
        if (!this.el.drawer.hidden) { this._closeSettings(); return; }
        if (this.voice.speaking) { this.voice.cancelSpeech(); return; }
      });

      // Stop-TTS pill (floating) + permanent composer Stop-voice button.
      // Either control cancels active TTS via voice.cancelSpeech().
      this.el.stopTtsBtn?.addEventListener("click", () => this.voice.cancelSpeech());
      this.el.stopVoiceBtn?.addEventListener("click", () => this.voice.cancelSpeech());
      if (!this.voice.ttsSupported && this.el.stopVoiceBtn) {
        this.el.stopVoiceBtn.disabled = true;
        this.el.stopVoiceBtn.title = "Speech synthesis not supported in this browser.";
      }
      this.voice.addEventListener("speakstart", () => {
        this.el.ttsControls.hidden = false;
        this.el.stopVoiceBtn?.classList.add("speaking");
      });
      this.voice.addEventListener("speakend", () => {
        this.el.ttsControls.hidden = true;
        this.el.stopVoiceBtn?.classList.remove("speaking");
      });

      // Tab switching. The Home tab is an action (return to Mission Control),
      // not a panel switch — forward to the dedicated #home-button which has
      // its own postMessage / navigation handler.
      document.querySelectorAll(".tab-button").forEach((b) => {
        b.addEventListener("click", () => {
          if (b.dataset.action === "home") {
            document.getElementById("home-button")?.click();
            return;
          }
          if (b.dataset.tab) this._setTab(b.dataset.tab);
        });
      });

      // Settings change — re-apply
      this.settings.addEventListener("change", () => this._applySettings());
    }

    _setTab(tab) {
      if (tab !== "chat" && tab !== "workbench") tab = "chat";
      document.body.dataset.tab = tab;
      const panels = { chat: "tab-chat", workbench: "tab-workbench" };
      for (const [name, id] of Object.entries(panels)) {
        document.getElementById(id).hidden = name !== tab;
      }
      document.querySelectorAll(".tab-button").forEach((b) => {
        const active = b.dataset.tab === tab;
        b.setAttribute("aria-selected", String(active));
        b.tabIndex = active ? 0 : -1;
      });
      // Workbench frame can't measure itself while hidden — refit now that
      // the stage has real dimensions. Defer one frame so layout settles.
      if (tab === "workbench" && this.workbench) {
        requestAnimationFrame(() => this.workbench.onShown());
      }
    }

    triggerImageUpload() { this.el.imageInput.click(); }

    triggerCameraCapture() { this.camera.open(); }

    _autosize = (e) => {
      const ta = e.currentTarget;
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    };

    _applySettings() {
      const s = this.settings.all;
      // Provider chip
      this.el.providerChip.dataset.provider = s.provider;
      this.el.providerChipLabel.textContent = s.provider === "openai" ? "OpenAI" : "Ollama";
      // Speaker pressed state
      this.el.speakerBtn.setAttribute("aria-pressed", String(!!s.autoRead));
      // Update STT lang if currently constructed
      if (this.voice.recognition) this.voice.recognition.lang = s.language;
      // Apply mode theme to body
      if (s.mode) document.body.dataset.mode = s.mode;
      else delete document.body.dataset.mode;
    }

    _populateSettingsForm() {
      const s = this.settings.all;
      const f = this.el.settingsForm.elements;
      // provider radio
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
      // Only autofill if empty or matches the *other* provider's default
      const otherDefault = PROVIDER_DEFAULTS[provider === "openai" ? "ollama" : "openai"];
      if (!baseUrlInput.value || baseUrlInput.value === otherDefault.baseUrl) {
        baseUrlInput.value = defaults.baseUrl;
      }
      if (!modelInput.value || modelInput.value === otherDefault.model) {
        modelInput.value = defaults.model;
      }
      // Hide/show api key
      this.el.apiKeyField.classList.toggle("hidden", provider !== "openai");
    }

    _openSettings() {
      this._populateSettingsForm();
      this.el.drawer.hidden = false;
      // Focus first input for keyboard accessibility
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
      e.target.value = ""; // allow re-selecting same file later
      if (!file) return;
      try {
        this._setStatus(`Processing image (${(file.size / 1024).toFixed(0)} KB)…`, "info");
        const dataUri = await this.images.toDataUri(file);
        await this._useImage({ dataUri, name: file.name, size: file.size });
        this._setStatus("");
      } catch (err) {
        this._setStatus(err.message || "Image processing failed.", "error");
      }
    }

    /** Common entry point for any newly-acquired image (file upload OR camera
     *  capture). Downscales if needed, updates the chat preview chip, and
     *  syncs to the Workbench tab. */
    async _useImage(image) {
      let final = image;
      // For camera captures (which arrive already as data URIs), apply the
      // same downscale rule as file uploads.
      if (image.size && image.size > ImageHandler.DOWNSCALE_TRIGGER) {
        try {
          const dataUri = await this.images._downscale(image.dataUri);
          final = { ...image, dataUri, size: Math.round(dataUri.length * 0.75) };
        } catch (err) {
          console.warn("Camera/image downscale failed; using original:", err);
        }
      }
      this.pendingImage = final;
      this.el.imagePreviewImg.src = final.dataUri;
      this.el.imagePreviewMeta.textContent =
        `${final.name} · ${(final.size / 1024).toFixed(0)} KB`;
      this.el.imagePreview.hidden = false;
      this.workbench.setImage(final);
    }

    _clearPendingImage() {
      this.pendingImage = null;
      this.el.imagePreviewImg.removeAttribute("src");
      this.el.imagePreviewMeta.textContent = "";
      this.el.imagePreview.hidden = true;
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
        onError: (err) => {
          restoreMic();
          const msg = this._friendlyMicError(err);
          if (msg) this._setStatus(msg, "error");
          console.warn("STT error:", err);
        },
        onEnd: () => {
          restoreMic();
          if (this.el.status.textContent === "Listening…") this._setStatus("");
        },
      });
    }

    _friendlyMicError(code) {
      switch (code) {
        case "network":
        case "service-not-allowed":
          return "Speech recognition needs the browser's online engine, which is unavailable on this air-gapped device. Use the keyboard, or a browser with on-device STT (Safari iOS 14+ / macOS 14+).";
        case "not-allowed":
          return "Microphone permission denied. Allow microphone access in your browser settings.";
        case "no-speech":
          return "Didn't catch that — try again.";
        case "audio-capture":
          return "No microphone detected.";
        case "aborted":
          return ""; // user-initiated stop, suppress message
        default:
          return `Speech recognition failed: ${code}.`;
      }
    }

    _toggleAutoRead() {
      const next = !this.settings.get("autoRead");
      this.settings.update({ autoRead: next });
      if (!next) this.voice.cancelSpeech();
    }

    // ── Send / stream ────────────────────────────────────────────────────
    async _send(opts = null) {
      const isProgrammatic = !!opts;

      // Composer Send button click while streaming = stop.
      if (!isProgrammatic && this.streamCtrl) {
        this.streamCtrl.abort();
        this.streamCtrl = null;
        return;
      }
      // Programmatic call (e.g. Workbench point analysis): abort any prior
      // in-flight turn and start fresh.
      if (isProgrammatic && this.streamCtrl) {
        this.streamCtrl.abort();
        this.streamCtrl = null;
      }

      const text = (opts?.text ?? this.el.input.value.trim()).trim();
      const image = opts?.imageDataUri
        ? { dataUri: opts.imageDataUri, name: opts.imageName || "image" }
        : this.pendingImage;
      const displayText = opts?.displayText ?? (text || "(image)");
      const displayImageDataUri = "displayImageDataUri" in (opts || {})
        ? opts.displayImageDataUri
        : (image?.dataUri || null);

      if (!text && !image) {
        opts?.onComplete?.({ acc: "", failed: true, aborted: false, metrics: null });
        return;
      }

      // Validate
      const s = this.settings.all;
      if (s.provider === "openai" && !s.apiKey) {
        this._setStatus("OpenAI API key required. Open Settings to add one.", "error");
        opts?.onComplete?.({ acc: "", failed: true, aborted: false, metrics: null });
        return;
      }
      if (!s.baseUrl) {
        this._setStatus("Base URL is required. Open Settings to configure.", "error");
        opts?.onComplete?.({ acc: "", failed: true, aborted: false, metrics: null });
        return;
      }

      // Build user message turn
      const userContent = image
        ? [
            { type: "text", text: text || "Please analyze this medical image." },
            { type: "image_url", image_url: { url: image.dataUri } },
          ]
        : text;

      const userMessage = { role: "user", content: userContent };
      this.messages.push(userMessage);
      this.renderer.addMessage("user", { text: displayText, imageDataUri: displayImageDataUri });
      if (!isProgrammatic) {
        this.el.input.value = "";
        this._autosize({ currentTarget: this.el.input });
        this._clearPendingImage();
      }

      const popOrphanUser = () => {
        // Preserve strict alternation when the assistant turn fails to land.
        const idx = this.messages.lastIndexOf(userMessage);
        if (idx >= 0) this.messages.splice(idx, 1);
      };

      // Streaming bubble
      const bubble = this.renderer.addMessage("assistant", { text: "", streaming: true });

      this.streamCtrl = new AbortController();
      this.el.sendBtn.querySelector("span").textContent = "Stop";
      this._setStatus("Thinking…", "info");

      let acc = "";
      let aborted = false;
      let failed = false;
      let usage = null;
      const t0 = performance.now();
      let tFirst = 0;
      try {
        for await (const token of this.client.stream(this.messages, {
          signal: this.streamCtrl.signal,
          onUsage: (u) => { usage = u; },
        })) {
          if (!tFirst) tFirst = performance.now();
          acc += token;
          bubble.appendToken(token);
          opts?.onProgress?.(acc, token);
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

      if (failed) {
        popOrphanUser();
        bubble.node.remove();
        opts?.onComplete?.({ acc: "", failed: true, aborted: false, metrics: null });
        return;
      }
      if (!acc && !aborted) {
        popOrphanUser();
        bubble.node.remove();
        this.renderer.addMessage("error", { text: "Empty response from provider." });
        opts?.onComplete?.({ acc: "", failed: true, aborted: false, metrics: null });
        return;
      }
      if (aborted && !acc) {
        popOrphanUser();
        bubble.node.remove();
        opts?.onComplete?.({ acc: "", aborted: true, failed: false, metrics: null });
        return;
      }

      bubble.finalize(aborted ? `${acc}\n\n*(stopped)*` : acc);
      if (acc) this.messages.push({ role: "assistant", content: acc });

      const tEnd = performance.now();
      const metrics = this._computeMetrics({ t0, tFirst, tEnd, acc, usage, aborted });
      if (metrics) bubble.setMetrics(metrics);

      opts?.onComplete?.({ acc, aborted, failed: false, metrics });

      if (!aborted && this.settings.get("autoRead") && acc) {
        const r = this.voice.speak(acc);
        if (r && r.voice && !r.exact) {
          const lang = this.settings.get("language");
          this._setStatus(
            `No exact local voice for ${lang}; using ${r.voice.name} (${r.voice.lang}).`,
            "info"
          );
          setTimeout(() => { if (this.el.status.dataset.tone === "info") this._setStatus(""); }, 3500);
        } else if (r && !r.voice) {
          this._setStatus("No local voice installed; install an OS voice for TTS.", "error");
          setTimeout(() => { if (this.el.status.dataset.tone === "error") this._setStatus(""); }, 4000);
        }
      }
    }

    _computeMetrics({ t0, tFirst, tEnd, acc, usage, aborted }) {
      if (!tFirst || !acc) return null;

      const ttftMs = tFirst - t0;
      const genMs = tEnd - tFirst;
      const totalMs = tEnd - t0;

      // Prefer authoritative count from `usage`. Fall back to a char/4 estimate.
      const reportedTokens = usage?.completion_tokens
        ?? usage?.eval_count   // Ollama native key, in case a future server merges fields
        ?? null;
      const estimated = reportedTokens == null;
      const tokens = reportedTokens ?? Math.max(1, Math.round(acc.length / 4));
      const tps = genMs > 0 ? (tokens / (genMs / 1000)) : 0;

      const fmtMs = (ms) => ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
      const fmtTok = (n) => `${estimated ? "~" : ""}${n.toLocaleString()} tok`;

      const out = [
        { label: "TTFT",  value: fmtMs(ttftMs), title: "Time to first token (request → first stream chunk)" },
        { label: "TPS",   value: `${tps.toFixed(1)} tok/s`, title: estimated ? "Tokens per second (estimated from char count)" : "Tokens per second (from server-reported usage)" },
        { label: "Out",   value: fmtTok(tokens), title: estimated ? "Estimated output tokens (~chars/4)" : "Output tokens reported by server" },
        { label: "Total", value: fmtMs(totalMs), title: "Total wall-clock time for the request" },
      ];
      if (usage?.prompt_tokens != null) {
        out.splice(3, 0, { label: "In", value: `${usage.prompt_tokens.toLocaleString()} tok`, title: "Prompt tokens reported by server" });
      }
      if (aborted) out.push({ label: "", value: "stopped", title: "Generation was stopped by the user" });
      return out;
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

  function boot() {
    const app = new App();
    window.addEventListener("message", (event) => {
      if (event.source !== window.parent || event.origin !== window.location.origin) return;
      if (!event.data || event.data.type !== "sima-sentry:config") return;
      const cfg = event.data.config || {};
      const patch = {};
      if (cfg.provider === "openai" || cfg.provider === "ollama") patch.provider = cfg.provider;
      for (const key of ["baseUrl", "model", "apiKey"]) {
        if (typeof cfg[key] === "string") patch[key] = cfg[key];
      }
      app.settings.update(patch);
      app._populateSettingsForm();
    });
  }

  // ─── Boot ───────────────────────────────────────────────────────────────
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
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
