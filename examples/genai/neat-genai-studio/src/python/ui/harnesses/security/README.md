# SiMaSentry-Sec

A zero-dependency, **air-gapped** browser harness that turns a CCTV snapshot or a frame from a recorded video into a Vision-Language-Model threat assessment, with native browser voice input/output. Designed for factory-floor and Security Operations Center (SOC) terminals with no internet egress.

The console has three views, switchable via tabs in the header:

- **Workbench** — the conversational threat-log view. Attach a frame, ask freeform questions, get streamed assessments. The right-hand sidebar logs every alert (timestamp, snapshot, threat level) and clicking an entry jumps back to the originating message.
- **Threat Analysis** — single-image marker tool. Load a frame, click-and-drag a bounding box around the suspicious person / bag / vehicle, optionally add facility context, hit *Analyze marked object*. The frame is sent to the VLM with the red box burned in and the literal prompt: *"Analyze this specific object. Is it a weapon, an unauthorized person in a restricted area, or a suspicious abandoned package? Assess the threat level (Low/Medium/High)."*
- **Change Detection** — dual-image differential audit. Load a Baseline Reference frame and a Current Feed frame, hit *Compare & Detect*, and the VLM returns a list of NEW / MISSING / STRUCTURAL discrepancies. When the model includes coordinates the harness draws colour-coded boxes on the Current Feed (red = new, yellow = missing, orange = structural). Change Detection is descriptive — it does not produce threat levels or post to the alerts sidebar; run anything suspicious through Threat Analysis afterwards.

```
security/
├── index.html   # markup + inline SVG icons + inline data-URI favicon
├── style.css   # hand-written, dark-only SOC theme
├── app.js     # SecurityConsole class — settings, fetch, vision, voice
└── README.md   # this file
```

## Why this exists

A trained operator watching a wall of monitors can miss things. A local VLM running on the same network — sometimes the same machine — can act as a second pair of eyes. This console gives the operator a fast way to ask the model "what's wrong with this frame", colour-codes the threat level, and reads the answer aloud in the operator's language.

It is an **operator aid**, not an autonomous response system. The disclaimer at the top of the page is not decorative.

## Architecture — strictly offline

The console makes **zero external network requests at load time**. Open it in DevTools' Network tab on a fresh load and you will see only the three local files (`index.html`, `style.css`, `app.js`) plus an inline data-URI favicon. There are:

- No CDN links (no Google Fonts, FontAwesome, Tailwind, Bootstrap, etc.)
- No `@font-face`, no `@import`, no webfont URLs anywhere
- No remote favicon, no analytics, no preconnect/prefetch hints
- All icons are inline `<svg>` elements (mic, speaker, attach, capture-frame, settings, send, clear, remove, shield)

A locked-down `Content-Security-Policy` `<meta>` tag enforces this at the browser level:

```
default-src 'self' data: blob:;
script-src 'self';
style-src 'self';
img-src 'self' data: blob:;
media-src 'self' blob: data:;
connect-src 'self' http://localhost:* http://127.0.0.1:* https://api.openai.com;
font-src 'self';
object-src 'none';
base-uri 'self';
frame-ancestors 'none';
```

The only outbound request the page ever makes is the chat-completion call to whichever endpoint the operator configured. The default is `http://localhost:11434` (Ollama).

## Quick start — offline / Ollama (recommended)

This is the path designed for production air-gapped terminals.

1. **Install Ollama** on the terminal (one-off, can be done from a USB image or staged through your normal software-distribution channel).
2. **Pull a vision model.** On a connected staging machine: `ollama pull gemma3` (or `llava` / `llama3.2-vision`), then transfer `~/.ollama` model blobs to the air-gapped device. Or do a one-time pull on the terminal itself if it has temporary internet access.
3. **Start Ollama with browser CORS enabled** so the static page can talk to it:

   - macOS / Linux: `OLLAMA_ORIGINS="*" ollama serve`
   - Linux (systemd service): drop a file at `/etc/systemd/system/ollama.service.d/override.conf` with
     ```
     [Service]
     Environment="OLLAMA_ORIGINS=*"
     ```
     then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
   - macOS (LaunchAgent installs of the Ollama app): `launchctl setenv OLLAMA_ORIGINS "*"`, then quit and relaunch the app.
   - Windows: `setx OLLAMA_ORIGINS "*"` in PowerShell (or set in *System Properties → Environment Variables*), then **fully restart Ollama**. A new shell does not propagate the variable to an already-running service.

   Restart is mandatory — Ollama only reads the env var at start.
4. **Copy the `security/` folder to the terminal** and either double-click `index.html` (opens via `file://`) or serve it with any static server: `python -m http.server 8000` then visit `http://localhost:8000/security/`.
5. **Verify settings.** Provider should already read `Ollama (local)`, base URL `http://localhost:11434/v1/chat/completions`, model `gemma3`. That's it.

## CORS troubleshooting

If a request fails with `net::ERR_FAILED` and DevTools shows "blocked by CORS policy", `OLLAMA_ORIGINS` is not set in the environment Ollama actually inherited. Check the value Ollama is using:

- **systemd**: `systemctl show ollama -p Environment`
- **macOS LaunchAgent**: `launchctl getenv OLLAMA_ORIGINS`
- **Windows**: `Get-ChildItem Env:OLLAMA_ORIGINS` from a *new* PowerShell window, then verify Ollama was restarted *after* the variable was set.

You can scope `OLLAMA_ORIGINS` more tightly than `*` — for example, `OLLAMA_ORIGINS="file://* http://localhost:8000"` if you only ever serve the harness from those origins.

If you see `Cannot reach http://localhost:11434/...` in the status bar, the harness is telling you the connection itself failed (Ollama not running, wrong port, firewall) — that is a different problem than CORS.

## Configuration via URL parameters (deep links)

The harness reads URL parameters at startup and writes them through to local storage so the configuration sticks across reloads. Precedence: **defaults < `localStorage` < URL parameters**.

| URL key                     | Setting        | Notes                                  |
|-----------------------------|----------------|----------------------------------------|
| `provider`                  | provider       | `ollama` (default) or `openai`         |
| `baseUrl`, `base_url`       | baseUrl        | Full chat-completions endpoint URL     |
| `apiKey`, `api_key`         | apiKey         | Avoid in URL — see security caveats    |
| `model`                     | model          | e.g. `gemma3`, `llava`, `gpt-4o`       |
| `language`, `lang`          | language       | BCP-47, e.g. `en-US`, `es-ES`, `zh-CN` |
| `systemPrompt`, `system_prompt` | systemPrompt | URL-encoded                          |
| `autoRead`, `auto_read`     | autoRead       | `true`/`false`/`1`/`0`/`yes`/`no`      |
| `stt`, `sttDisabled`        | sttDisabled    | `stt=off` disables the mic; `stt=on` enables |
| `highContrast`, `high_contrast` | highContrast | Boolean — applies the high-contrast SOC theme |
| `mode`                      | (bundle)       | `mode=security` applies a SOC bundle (see below) |
| `feature`                   | (one-shot nav) | `feature=changedetection` deep-links straight to the Change Detection tab |

### `?mode=security` — SOC kiosk preset

Adding `?mode=security` to the URL applies a curated bundle of defaults appropriate for a locked-down SOC terminal:

- **High-contrast theme** — brighter cyan accent, sharper borders, larger base font, pure-black background.
- **Provider** — forced to `ollama` (unless `?provider=...` is also supplied).
- **Base URL** — defaults to the local Ollama endpoint (`http://localhost:11434/v1/chat/completions`) unless `?baseUrl=` overrides.
- **Microphone** — preemptively disabled (avoids the failed-first-attempt UX on Chromium, since `SpeechRecognition` requires network).

Explicit URL params still win — `index.html?mode=security&provider=openai` is honoured for staging machines that want the contrast theme without the local-Ollama default.

### Worked examples

Default factory terminal, English, audio alerts on:
```
index.html?provider=ollama&model=gemma3&lang=en-US&autoRead=true
```

Spanish-language operator on a different vision model:
```
index.html?provider=ollama&baseUrl=http://localhost:11434/v1/chat/completions&model=llama3.2-vision&lang=es-ES
```

Air-gapped kiosk with the mic preemptively disabled (avoids the failed-first-attempt UX):
```
index.html?provider=ollama&model=gemma3&lang=en-US&stt=off
```

Kiosk preset with a facility-specific addition to the system prompt (URL-encoded):
```
index.html?provider=ollama&model=gemma3&lang=en-US&systemPrompt=You%20are%20an%20expert%20AI%20security%20analyst%20monitoring%20Plant%2042.%20Loading%20bay%20doors%20must%20remain%20closed%20outside%2006%3A00-18%3A00%20local%20time.%20Begin%20every%20response%20with%20THREAT%3A%20%3CINFO%7CLOW%7CMEDIUM%7CHIGH%7CCRITICAL%3E.
```

OpenAI staging (only valid on a connected machine, see below):
```
index.html?provider=openai&model=gpt-4o&lang=en-US
```

## Voice on air-gapped devices

### Audio alerts (TTS)

The console uses the browser's `speechSynthesis` API. To guarantee no voice request leaves the device, the voice picker filters strictly to OS-installed voices (`voice.localService === true`) before doing language matching. Cloud-only voices are excluded outright.

Install offline voices for the languages you support:

- **macOS**: *System Settings → Accessibility → Spoken Content → System Voice → Manage Voices…*
- **Windows**: *Settings → Time & Language → Speech → Add voices*
- **Linux**: install `espeak-ng` and/or `speech-dispatcher` via your package manager; some distributions also support Mozilla TTS / Piper for higher-quality offline voices.

If no offline voice is installed for the chosen language, the harness shows a notice in the status bar and skips auto-read for that message. The text is still rendered normally.

### Voice input (STT) — important caveat

The browser's `SpeechRecognition` API (and its `webkitSpeechRecognition` alias) is **not on-device** in Chrome / Chromium / Edge — those browsers ship transcription via Google's cloud service. On an air-gapped machine the first attempt will fail with a `network` error.

The harness handles this gracefully:

- It catches `error.error === 'network'` and shows a persistent banner: *"Speech-to-text requires network access in this browser and is unavailable on this air-gapped device."*
- It then disables the mic button for the rest of the session, so the operator does not retry into a wall.

Workarounds for offline STT:

- **Settings → Disable microphone** (or `?stt=off` in the URL) preemptively hides the failure altogether on terminals you know are offline.
- **Edge on Windows** can use Windows Speech Services (on-device) for some languages — try it before deploying widely.
- Otherwise, rely on the keyboard, a barcode scanner, or an external dictation tool that types into the textarea.

The harness never requires STT — every feature is reachable via the keyboard.

## Vision

- Supported attachment types: any image (`<input accept="image/*">`) and any video (`<input accept="video/*">`) that the host browser can decode.
- Images are downscaled to a long-edge maximum of **1280 px** before base64-encoding (re-encoded as JPEG @ 0.85 quality). This keeps payloads under typical token budgets and makes Ollama responses materially faster on CPU-only terminals.
- "Capture frame" works by loading the chosen video file into a hidden `<video>` element, seeking ~0.1 s in (avoids occasional black first frame), drawing the current frame to a `<canvas>`, and reading it back via `toDataURL('image/jpeg', 0.85)`. The video file's object URL is revoked immediately after.
- The harness never requests `MediaStream` / camera access — there is no live-camera capture on purpose. If you want to feed live video, point the model at your VMS recording and use *Capture frame* on the most recent clip.
- Vision payload follows the standard OpenAI multipart format:
  ```json
  {
    "role": "user",
    "content": [
      { "type": "text", "text": "…" },
      { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,…" } }
    ]
  }
  ```
  Both `gpt-4o` and Ollama's OpenAI-compatible `/v1/chat/completions` endpoint accept this shape for vision-capable models.

## Threat Analysis tab — bounding-box marker

1. Click **Load surveillance frame** and pick a JPG/PNG.
2. Click and drag on the canvas to outline the suspicious person, bag, or vehicle. A live preview rectangle appears during the drag; on release it solidifies into a red 4-sided box with corner ticks and a black halo for contrast on bright backgrounds. The readout shows pixel + normalized box coordinates.
   - **Movement threshold**: drags shorter than 6 CSS pixels are treated as accidental clicks and rejected with a hint to drag.
   - **Esc** during a drag cancels it; **Esc** afterwards clears the finalized box.
   - **Enter / Space** with the canvas focused drops a centred default 25%-of-frame box (keyboard fallback).
   - Drawing a second box replaces the first.
3. Optionally type a one-liner of facility context (e.g. *"Loading bay 3, after-hours"*).
4. Click **Analyze marked object**. The harness:
   - Burns the red box into the JPEG (so the model has both the visual cue and explicit numeric coordinates),
   - Builds the user message with the literal prompt above + your context + a `[Object outlined in a red box at pixel (x, y, w, h) on a W×H frame; normalized (xN, yN, wN, hN)]` hint,
   - Streams the VLM's response into the result pane, colour-coded by detected threat level,
   - Posts the result to the *Current Alerts* sidebar tagged with source = "Threat Analysis".

The pointer-event handler (`pointerdown / pointermove / pointerup / pointercancel`) covers mouse, touch, and pen with one set of listeners — no separate touch path.

## Change Detection tab — baseline vs current

The dual-image workflow for spotting differences between a known-good reference and a live frame. Lives in its own top-level tab; deep-link with `?feature=changedetection`.

1. Open the **Change Detection** tab from the header.
2. Load a **Baseline Reference** snapshot (Image A — the known-good state of the area, e.g. taken at the start of the shift).
3. Load a **Current Feed** snapshot (Image B — the live frame under suspicion). The Current Feed canvas is what gets annotated with discrepancy boxes.
4. Optionally add facility context (e.g. *"Server room, baseline taken 06:00, last access logged 17:30"*).
5. Click **Compare & Detect**.

The harness sends *both* images in a single `/v1/chat/completions` request as a multi-image content array (Image A first, Image B second), under a Change-Detection-specific system prompt that instructs the model to use a structured discrepancy format:

```
[CATEGORY] (x, y, w, h) — short description
```

…where CATEGORY ∈ {NEW, MISSING, STRUCTURAL} and `(x, y, w, h)` are normalized 0–1 bounding-box coordinates on the Current Feed (Image B). The harness parses each line as it streams in and progressively draws coloured boxes on the Current Feed canvas:

- **Red** — new / suspicious object (potential threat or abandoned package)
- **Yellow** — missing object (theft, tampering, or removed item)
- **Orange** — structural change (forced entry, open door, broken window)

Each box gets a numbered category badge in the top-left so the visual matches the order in the result panel. Discrepancies the model returns *without* coordinates are still listed in the result text — they just don't get a box drawn.

Change Detection has its own dedicated system prompt (no shared role with Threat Analysis), and is intentionally descriptive — **it does not produce a `THREAT:` line, and it does not post to the alerts sidebar**. If a discrepancy looks suspicious, take a screenshot of the Current Feed, switch to the Threat Analysis tab, and run the marker tool against it.

Note: smaller VLMs (including most Ollama-served models) are unreliable at precise bounding boxes. Treat the boxes as advisory — read the discrepancy list as the source of truth.

## Alerts sidebar (Workbench tab)

Every assistant response that includes a parseable `THREAT: <LEVEL>` line is automatically posted to the right-hand alerts sidebar, regardless of which tab triggered it. Each entry shows:

- Snapshot of the originating frame (Workbench: the user's attached image; Threat Analysis: the marker-annotated frame).
- Threat-level pill, colour-coded.
- Timestamp (HH:MM:SS, 24-hour).
- Source label — *Workbench* or *Threat Analysis*.
- A two-line summary (the first non-empty line of the response, with the THREAT directive stripped).

Click an alert from a Workbench assessment to jump to its message in the threat log (with a brief outline flash to draw the eye). Click a Threat-Analysis alert to switch tabs to the analysis view.

The sidebar is in-memory only — clearing it (or refreshing the page) drops the list. This is intentional: the harness is an aid, not a system of record.

## Streaming output

Assistant responses are streamed via Server-Sent Events (`stream: true` on the chat-completions request) and rendered into the threat-log bubble as each delta arrives. A blinking caret marks the in-flight message; the bubble's threat-level border colour updates the moment the `THREAT: <LEVEL>` line is parseable, well before the rest of the assessment finishes generating.

If the configured provider ignores `stream: true` and replies with a single JSON document, the harness detects this from the response `Content-Type` and falls back to a one-shot render.

### Stop-token handling

Some Ollama-served chat templates (Gemma3's `<end_of_turn>`, Llama-3's `<|eot_id|>`, ChatML's `<|im_end|>`, etc.) occasionally leak their template stop tokens into `delta.content` instead of being filtered server-side. The harness detects a fixed list of common stop tokens (`STOP_TOKENS` in `app.js`) and:

- Truncates the rendered text at the first occurrence of any stop token.
- Stops reading from the stream cleanly (cancels the reader, returns the current buffer).
- Holds back any tail of the rendered text that *could* be the prefix of a stop token, so a token split across two SSE chunks (e.g. `<end_of` then `_turn>`) is detected on the second chunk before the partial prefix gets shown to the operator.
- Honours `finish_reason` on a chunk as a normal end-of-stream signal.

If a model you deploy uses a stop token not in the default list, add it to `STOP_TOKENS` near the top of `app.js`.

The renderer is a small (~80 lines), zero-dependency markdown subset that supports headings (`#`/`##`/`###`), bold (`**`), italic (`*`/`_`), inline code (`` ` ``), fenced code blocks with language hints, ordered + unordered lists, and links. All input is HTML-escaped first, and only an allow-listed set of tags is emitted; link `href` values are validated to `http(s)://`, `mailto:`, or relative URLs (so an `[x](javascript:…)` from an attacker-controlled response stays inert).

## Performance metrics (TTFT / TPS)

Every assistant message gets a small footer line with timing data sampled by the browser:

- **TTFT** — time-to-first-token: wall-clock duration from the moment `fetch()` was issued to the first SSE delta arriving. This includes network round-trip + the model's prompt-processing / load time. On a healthy local Ollama, expect ≲500 ms; on cold-start (model loaded into VRAM on the first request), several seconds is normal.
- **TPS** — tokens per second across the *generation* window only (i.e. `completedAt − firstDeltaAt`), so model-load latency does not deflate the rate.
- **tokens** — completion tokens. Source is shown via a suffix:
  - no suffix → reported by the provider (`usage.completion_tokens` from OpenAI's `stream_options.include_usage` trailing chunk, or Ollama's final OpenAI-compatible chunk). Most accurate.
  - `(Δ)` → counted from SSE chunks (≈1 token per chunk). Off-by-a-bit when the provider batches.
  - `(~)` → estimated from response character length (~4 chars/token). Used only when the provider neither emits `usage` nor streams.
- **prompt** — prompt tokens, when the provider returns it.
- **total** — full wall-clock from `fetch()` to stream close.

Values are colour-coded as a fast at-a-glance health check on the terminal:

| Metric | Green (good) | Amber | Red (slow) |
|--------|:------------:|:-----:|:----------:|
| TTFT   | < 500 ms     | < 2 s | ≥ 2 s      |
| TPS    | ≥ 30 tok/s   | ≥ 10  | < 10 tok/s |

These thresholds reflect what feels acceptable on a workstation-class GPU with `gemma3` or similar. Tune them in `style.css` (`is-good` / `is-warn` / `is-slow` classes) if your factory hardware is materially different.

## Threat-level extraction

The default system prompt instructs the model to begin every response with a single line:
```
THREAT: <INFO|LOW|MEDIUM|HIGH|CRITICAL>
```
The console parses that line and applies a colour-coded left border + a small pill to the message bubble:

| Level    | Colour |
|----------|--------|
| INFO     | Blue   |
| LOW      | Green  |
| MEDIUM   | Amber  |
| HIGH     | Orange |
| CRITICAL | Red    |

If the model omits the THREAT line, no colour is applied — the message renders in the neutral border. Override the system prompt in *Settings → System prompt* to add facility-specific rules; keep the THREAT line in English so the colour mapping stays consistent.

## Running with OpenAI (staging only)

OpenAI is supported as a secondary path for non-air-gapped staging environments where you want a sanity check against a known-good model.

1. Open *Settings*, switch *Provider* to OpenAI.
2. Paste your API key into the *API Key* field. **The key never leaves the browser** — it is stored in `localStorage` and only sent as the `Authorization` header on the chat-completions request you trigger.
3. Set *Vision model* to `gpt-4o` (or any other vision-capable OpenAI model).

This requires internet egress from the device. **Do not enable it on production air-gapped terminals.**

## Deep-link / kiosk deployment

For shipping a preconfigured shortcut to operators:

- Build a single URL with `provider`, `model`, `lang`, `autoRead`, and (optionally) `systemPrompt` baked in. URL-encode the system prompt.
- Drop a `.url` file (Windows) / `.desktop` file (Linux) / `.webloc` file (macOS) on the operator's desktop pointing at that URL.
- **Do not** put `apiKey` in the URL on a kiosk that uses OpenAI — the URL ends up in browser history, in HTTP `Referer` headers, and in any logs the surrounding system keeps. Use the settings drawer instead, or operate Ollama-only on production terminals.

## Security caveats

- Settings (including any API key) are stored in the browser's `localStorage` for the page's origin. On a kiosk that several operators share, anything stored in `localStorage` is visible to everyone with access to that browser profile.
- The console is an **analyst aid**. It produces probabilistic assessments. A trained operator must verify every alert before acting on it. Nothing here triggers any automated response.
- `?apiKey=` in URLs leaks the key into browser history, Referer headers, and any HTTP intermediary that logs URLs. Avoid this for OpenAI deployments.
- The locked-down CSP prevents most injection vectors, but the harness still trusts the model's raw text. The threat extractor uses a strict regex on the first 240 characters of the response — but the message body is rendered as `textContent` (never `innerHTML`), so an attacker-controlled response cannot inject markup.

## Sibling harnesses

This is one of a small suite of lightweight, single-page AI harnesses in `apps-llima-harnesses/`. The companion `health/` harness uses the same conventions (split `index.html` / `style.css` / `app.js`, drawer-based settings, BCP-47 language list, `localStorage` per-harness prefix) for medical-image analysis. Look there if you want a reference implementation in a slightly different domain — the patterns are intentionally consistent.

## Verification checklist

A short sanity-check to walk through after deploying to a new terminal:

1. **Air-gap audit**: open DevTools → Network, reload the page; confirm zero external requests beyond the three local files. Disable the network adapter, reload, confirm the page still renders identically.
2. **Provider chip** reads "Ollama (local)" by default. The dot is green.
3. **Settings round-trip**: change provider/model/language in the drawer, save, reload — values persist. *Clear stored settings* + reload returns to defaults.
4. **URL-param override**: load `index.html?lang=es-ES&autoRead=true` — *Settings → Language* is `es-ES`, *Auto-read* is checked.
5. **Image attach**: click the attach button, pick a JPG/PNG, confirm preview chip with filename + dimensions appears.
6. **Frame capture**: click the film/clapper button, pick a small `.mp4`, confirm the captured frame appears in the preview chip and the filename includes the seek timestamp.
7. **Send + Ollama**: type "What's in this frame?", click Send Alert, confirm a response arrives within a few seconds and the message bubble has a colour-coded left border. Open DevTools → Network and inspect the request body — it should contain the multipart `content` array.
8. **TTS**: enable *Auto-read*, send another message, confirm it is spoken. Check that no outbound voice request appears in DevTools → Network.
9. **TTS — missing voice**: change language to one with no installed local voice, confirm the status bar warns and the message still renders silently.
10. **STT — offline failure** (on an air-gapped device): click the mic; confirm the harness shows the "STT requires network access" banner and the mic disables itself.
11. **STT preemptive disable**: load `index.html?stt=off`, confirm the mic button is disabled with the correct tooltip.
12. **Threat colours**: send five prompts that the model treats with different severity (or temporarily override the system prompt to force a specific level) — confirm the five colours render distinctly.
