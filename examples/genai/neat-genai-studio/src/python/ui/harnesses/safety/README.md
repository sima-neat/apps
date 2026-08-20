# SiMaSentry-Safe

A zero-dependency, browser-only AI harness for occupational safety monitoring on **air-gapped factory devices**. Three static files plus a README — no build tools, no npm, no CDNs, no remote fonts, no remote scripts. Ship the folder; serve it locally; analyze hazards.

```
safety/
├── index.html   ← markup, all icons inline SVG, no <link>/<script src> to anywhere remote
├── style.css    ← hand-written, system font stack, no @import
├── app.js       ← single IIFE, plain ES2022, only fetch() target is your configured baseUrl
└── README.md    ← this file
```

## Overview

The harness has two tabs:

- **Chat** — chat with a vision-capable model. Type, attach a snapshot, capture from camera, or use the mic.
- **PPE Inspector** — load a factory-floor snapshot, click-and-drag to mark rectangular **Safety Zones** (or use the whole image as a single zone); each zone is automatically cropped, sent to the VLM with a PPE-focused prompt, and analyzed independently. If the model reports a **violation**, the harness speaks an audio alert regardless of the auto-read setting and color-codes the entry by severity (CRITICAL / HIGH / MEDIUM / LOW).

Both tabs share:

- **Vision**: base64 data URIs sent as the OpenAI vision-content array to any `/v1/chat/completions` endpoint.
- **Voice**: native browser STT (`SpeechRecognition`) for hands-free reporting and TTS (`speechSynthesis`) for spoken warnings. TTS prefers OS-local voices for offline operation.
- **Local-first**: defaults to `http://localhost:11434/v1/chat/completions` with model `gemma3`. OpenAI supported but secondary.
- **Deep linking**: every setting can be passed via URL parameters so a tablet can be deployed with a single bookmark. `api_key` is stripped from the URL after load. `?mode=safety` is a convenience preset (see below).

The whole `safety/` folder can be copied to a USB stick, pushed to a tablet, served from a local web server, and used with **zero internet egress**.

---

## Local Ollama setup (the primary path)

This is how you'll actually run it on the factory floor.

### 1. Pull a vision-capable model

On a connected machine (or on-device with one-time setup access):

```bash
ollama pull gemma3           # default — multimodal, fast on tablet hardware
# or:
ollama pull llava            # ~4.5 GB, classic vision baseline
ollama pull bakllava         # alternative
```

For fully air-gapped deployment, copy the contents of `~/.ollama/models/` from a primed machine onto the tablet's `~/.ollama/models/` directory.

### 2. Configure CORS so the in-browser harness can reach Ollama

The browser will refuse to call `http://localhost:11434` from a page served on a different origin (e.g. `http://localhost:8000`) unless Ollama explicitly allows it. Set `OLLAMA_ORIGINS`:

**macOS (LaunchAgent / desktop app):**
```bash
launchctl setenv OLLAMA_ORIGINS "*"
# Then quit and restart the Ollama menu bar app.
```

**Linux (systemd):** edit `/etc/systemd/system/ollama.service` and add to `[Service]`:
```ini
Environment="OLLAMA_ORIGINS=*"
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Windows:** System Properties → Environment Variables → New system variable `OLLAMA_ORIGINS=*`. Restart Ollama.

For tighter security, replace `*` with the exact origin serving the harness, e.g. `http://localhost:8000`.

### 3. Start Ollama and the harness

```bash
ollama serve                                # if not auto-started
cd safety
python3 -m http.server 8000                 # any static file server works
```

Open `http://localhost:8000`. The harness ships pre-pointed at `http://localhost:11434/v1/chat/completions` with model `gemma3`, so you should be able to send a hazard query immediately.

---

## Run locally

Any static file server works. The page **must** be served over `http://localhost` (or HTTPS) for the camera to function — `getUserMedia` refuses to run from `file://`.

```bash
cd safety
python3 -m http.server 8000
# or:
npx http-server -p 8000      # if available offline
# or any nginx / caddy file-server pointed at this directory
```

---

## Settings

Open the gear icon (top right). Settings persist to `localStorage` under the key `safety-ai-harness:settings`.

| Field | What it does |
|---|---|
| **Provider** | `Ollama (local)` or `OpenAI`. Switching auto-fills the matching default Base URL and Model if the field is empty or still on the other provider's default. |
| **Base URL** | The full `/v1/chat/completions` endpoint. Default: `http://localhost:11434/v1/chat/completions`. |
| **API Key** | Only used when Provider is OpenAI. Sent as `Authorization: Bearer <key>`. Stored locally; never logged. |
| **Model** | Vision-capable model name. `gemma3` (default), `llava`, `bakllava` for Ollama; `gpt-4o` for OpenAI. |
| **Language** | BCP-47 code (e.g. `es-ES`). Drives both `SpeechRecognition.lang` and TTS voice selection. |
| **System Prompt** | The safety inspector instructions. The default explicitly tells the model to detect the user's language and reply in the same language. "Reset to default" restores it. |
| **Auto-read replies aloud** | Speaks each assistant response via `speechSynthesis` with a local OS voice in the configured language. |

---

## Deep-link URLs

Every setting can be passed via URL parameters. Precedence: defaults < `localStorage` < URL parameters. Recognized parameters (snake_case and camelCase both accepted):

| Parameter | Setting |
|---|---|
| `provider` | `ollama` or `openai` |
| `base_url` / `baseUrl` | full endpoint URL |
| `api_key` / `apiKey` | OpenAI key (stripped from the URL after load) |
| `model` | model name |
| `lang` / `language` | BCP-47 code |
| `system_prompt` / `systemPrompt` | URL-encoded prompt override |
| `auto_read` / `autoRead` | `1` / `true` to enable auto-TTS on load |
| `mode` | `safety` opens the PPE Inspector tab and locks defaults to a local vision model (`llava`) when no model is otherwise stored or specified |

### Examples

**Local Ollama, Spanish-speaking floor terminal (primary deployment):**
```
http://factory-tablet-12.local:8000/?provider=ollama&model=gemma3&lang=es-ES&base_url=http://localhost:11434/v1/chat/completions
```

**Custom zone-specific prompt + auto-read warnings:**
```
http://factory-tablet-12.local:8000/?system_prompt=Focus%20on%20Zone%20B%20chemical%20handling%3A%20require%20goggles%2C%20gloves%2C%20and%20splash%20aprons.%20Flag%20any%20unsealed%20container.&auto_read=1
```

**OpenAI fallback (only on internet-connected devices):**
```
http://localhost:8000/?provider=openai&model=gpt-4o&lang=en-US&api_key=sk-...
```
After load, the `api_key` parameter is removed from the address bar via `history.replaceState` to keep it out of browser history and out of any screenshots.

**PPE Inspector preset (boots straight into the zone tool):**
```
http://localhost:8000/?mode=safety
```
This opens the PPE Inspector tab on load and forces the model default to `llava` if the user hasn't already chosen one. The safety-themed UI is intrinsic to this harness, so the param's "theme" effect is just to add a `body.mode-safety` hook for any future per-mode CSS.

---

## Voice notes (for offline deployments)

**Speech-to-Text** is browser-and-OS-dependent:
- Safari (macOS / iOS / iPadOS) and Microsoft Edge tend to ship local recognizers that work offline.
- Desktop Chrome historically routes recognition through a Google endpoint; on an air-gapped device the `SpeechRecognition` will fire `onerror` with `error: "network"` or `"service-not-allowed"`. The harness detects this and surfaces a clear status message — *"Speech recognition unavailable offline — type your report instead."* — and resets the mic button to idle.
- If STT is unsupported entirely, the mic button is disabled with a tooltip.

**Text-to-Speech** uses `speechSynthesis`. The harness filters voices by `voice.localService === true` first to ensure speech works on air-gapped devices. If no local voice is available for the configured language, it falls back to the best non-local match and surfaces a status warning so the operator knows TTS may be silent until a local voice for that language is installed.

To install offline TTS voices on Linux (where coverage is often lean):
```bash
# Debian/Ubuntu — speech-dispatcher with espeak-ng
sudo apt-get install speech-dispatcher espeak-ng espeak-ng-data
```
On macOS and Windows, additional voices can be downloaded once via OS settings (System Settings → Accessibility → Spoken Content on macOS; Settings → Time & Language → Speech on Windows) before the device is air-gapped.

---

## PPE Inspector

The second tab is a zone-based PPE compliance workflow:

1. Click **Upload snapshot** (or **Capture from camera**) to load a factory-floor image.
2. Click and drag on the snapshot to mark a rectangular **Safety Zone**. Each zone is auto-named (Zone A, Zone B, …) and color-coded.
3. Each zone is **automatically analyzed** the moment you finish drawing — the harness crops the zone, sends it as a vision payload with the prompt:

   > *Zone X — Monitor the individuals within this marked zone. Are they wearing required PPE (Hardhats, High-Vis Vests, Safety Glasses)? Report any spills or hazards inside this specific area. If you detect a safety violation, begin your reply with the literal token "VIOLATION:" on the first line (in English, regardless of response language).*

4. The streaming response appears as a card in the **Analyses** list under the snapshot.
5. **Severity classification**: the system prompt instructs the model to grade each violation on a four-tier OSHA-style scale, emitted on the first line as `VIOLATION: <SEVERITY>`:

   | Level | Meaning |
   |---|---|
   | `CRITICAL` | Imminent danger to life or limb; halt operation immediately |
   | `HIGH` | Serious risk; remediate this shift before resuming work |
   | `MEDIUM` | Meaningful risk; remediate within 24 hours |
   | `LOW` | Minor issue; remediate at next available opportunity |

   The card's left rail and badge are color-coded by level (CRITICAL pulses red, HIGH solid red, MEDIUM amber, LOW dark yellow). If the model emits the word "violation" (or a translation: *violación*, *violação*, *violazione*, *infraction*, *Verstoß*, 違反) but doesn't include a severity tag, the harness treats it as `HIGH`.

6. **Audio alert on violation**: fires regardless of the auto-read setting — it's a safety alert, not a preference. Phrasing varies by severity:

   - `CRITICAL` → *"Critical Safety Violation Detected in Zone N. Halt operation immediately."*
   - `HIGH` / `MEDIUM` / `LOW` → *"`<Severity>` severity safety violation in Zone N."*

   The alert phrase is fixed English; the full assistant response remains visible in the configured response language.

Multiple zones can be drawn on the same snapshot — each is analyzed independently. Removing a zone from the sidebar aborts its in-flight analysis (if any) and removes its card. **Clear all zones** wipes all zones and analyses but keeps the snapshot loaded.

## Camera

The camera button opens a live preview directly below the message log:

- Requires a secure context — `localhost` (treated as secure) or HTTPS. Will not work from `file://`.
- Prefers the rear camera (`facingMode: { ideal: "environment" }`) on tablets, falls back to any available camera.
- "Capture" grabs the current frame onto a canvas, downscales it through the same pipeline as a file upload (max long edge 1536px, JPEG q=0.9), and attaches it as the pending image.
- Frames are kept entirely in browser memory; nothing leaves the device unless the user submits to a remote provider.
- The camera stream is automatically stopped when the panel is closed, when the user submits, or when the tab loses visibility.

---

## Air-gapped deployment checklist

1. Copy the `safety/` folder onto the device.
2. Install Ollama and configure `OLLAMA_ORIGINS` per the section above.
3. Pull (or pre-stage) a vision model: `ollama pull gemma3`.
4. Install OS TTS voices for any languages you'll use, before disconnecting the device.
5. Serve the harness over `localhost` with a static file server.
6. Open DevTools → Network tab, check **Offline**, and reload the page. **Zero failed requests** confirms no remote dependency slipped in. The first chat request will fail (Ollama also blocked by the offline switch) — uncheck Offline to actually use it. The DevTools test only validates that the *page itself* needs nothing remote.
7. Bookmark a deep-link URL with the device's deployment-specific settings.

---

## Architecture summary

`app.js` is a single IIFE wrapping these classes:

| Class | Responsibility |
|---|---|
| `SettingsStore` | Defaults → localStorage → URL params merge; `change` events; URL cleanup of `api_key`. |
| `ApiError`, `ChatClient` | Build the OpenAI-format payload (with vision content array when an image is attached); SSE streaming parser with non-streaming JSON fallback. |
| `ImageHandler` | File-or-data-URI → base64; 10 MB hard cap; canvas downscale to 1536px long edge at JPEG q=0.9 above 4 MB. |
| `CameraController` | `getUserMedia` with environment-camera preference; `<video>` → canvas → JPEG data URI. Two instances — one each for the Chat and PPE Inspector tabs. |
| `ZoneManager` | Two stacked canvases (base + overlay) with pointer-event drag handling; emits `onZoneCreated` / `onZoneRemoved` callbacks. Each zone stores its image-pixel bounds, auto-name (A, B, C…), color, and a cropped data URI. |
| `VoiceController` | Native `SpeechRecognition` (with offline-error handling) + `speechSynthesis` (local-voice-preferred). |
| `ChatRenderer` | DOM templating + safe lightweight markdown (bold / italic / code / lists / fences). Used for both chat messages and zone analyses. |
| `App` | Wires everything to the DOM, manages tabs, routes zone analyses to independent `AbortController`s, and fires the violation audio alert. |

**Stop-token sanitization**: `ChatClient.stream` strips chat-template stop tokens that some servers leak into `delta.content` (`<end_of_turn>`, `<|eot_id|>`, `<|im_end|>`, `<|end_of_text|>`, `<|endoftext|>`, `<start_of_turn>`, `<|im_start|>`, `</s>`) and terminates the stream cleanly when one is detected. Includes a small lookahead buffer to catch tokens that are split across two SSE deltas.

State lives in the `App` instance and `SettingsStore`. There is no global state and no framework runtime.
