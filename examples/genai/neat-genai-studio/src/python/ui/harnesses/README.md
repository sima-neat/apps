# SiMaSentry Intelligence Agent

**SiMaSentry Intelligence Agent** is the air-gapped, browser-native product layer for **SiMa.ai Llima** edge AI on the [SiMa.ai MLSoC Modalix SOM](https://sima.ai/boards-devkits/?section=som). It ships as a single static-file repo: a **Mission Control** portal at the root, three vertical harnesses, and nothing else. No build step, no install on the operator's workstation, no internet at runtime.

| Mode | Product card | What it does | Default model |
|---|---|---|---|
| `?mode=medical`  | **SiMaSentry-Med**  ([`health/`](health/README.md))   | *Clinical-Grade Edge Radiology* — symptom triage, radiology / medical-image review, vision + voice. | `gemma3` |
| `?mode=safety`   | **SiMaSentry-Safe** ([`safety/`](safety/README.md))   | *Industrial PPE & Hazard Compliance* — factory-floor hazard detection from camera, image, or video. | `gemma3` |
| `?mode=security` | **SiMaSentry-Sec**  ([`security/`](security/README.md)) | *High-Security SOC & Change Detection* — threat assessment from CCTV stills and video frames. | `gemma3` |

Open `index.html` and pick a card. The harness loads embedded inside Mission Control via an iframe; click **Home** in the harness header to return.

---

## Hard offline rules (strict requirement)

This repo runs on **air-gapped factory devices that never see the internet**. Every page — the Mission Control portal and each harness — obeys these rules:

1. **Zero external network requests at runtime.** No CDN. No Google Fonts, no FontAwesome, no Tailwind, no Bootstrap, no analytics. Nothing is fetched from the public internet at any time.
2. **All assets are local.** CSS hand-written in `style.css`. UI icons are inline `<svg>` or local PNGs. Each page's favicon is a local PNG (`SiMaSentry.png`, `SiMaSentry-Med.png`, etc.). Scripts are local `<script src="app.js">`, no ES-module imports from a URL.
3. **The portal makes no `fetch()` calls of its own.** It is purely a launcher: the only network request is the iframe loading a harness from the same origin. The harnesses' own chat requests go directly to your configured engine.
4. **All settings live in `localStorage`.** No remote sync, no cookies sent off-device.

You can audit the whole tree with one command:

```bash
# Should return only local asset references:
#   - href="style.css", src="app.js", and harness folder paths
#   - href="SiMaSentry.png" / src="SiMaSentry.png" (Mission Control branding + favicon)
#   - href="SiMaSentry-Med.png" / -Safe / -Sec  (per-harness favicons + card icons)
grep -nE 'src=|href=|@import|@font-face|url\(' \
  index.html style.css app.js \
  health/index.html health/style.css health/app.js \
  safety/index.html safety/style.css safety/app.js \
  security/index.html security/style.css security/app.js
```

---

## Prerequisites

1. **A running llima server.** Follow [docs.sima.ai/pages/genai/runtime.html](https://docs.sima.ai/pages/genai/runtime.html). On the Modalix:
   ```bash
   cd /media/nvme/llima
   ./run.sh                   # or: llima run <model> --mode web
   ```
   llima listens on **port 9998** by default and exposes `POST /v1/chat/completions` (OpenAI-compatible).

2. **A modern browser** (Chrome, Edge, or Safari recent). Camera and microphone features require `https://` or `http://localhost` — `file://` will block them.

3. *(Optional)* The harnesses also work against a local [Ollama](https://ollama.com/) install on port `11434` as a fallback runtime.

---

## Quick start

From the repo root:

```bash
python3 -m http.server 8080
```

Then in the browser:

1. Open `http://localhost:8080`. Mission Control fades in: header pill, hero, three product cards, Edge Diagnostics dropdown, value-prop section, footer — staggered.
2. *(Optional)* Click **Configure** in the header to open the Global Configuration drawer. Set provider, base URL, and model. Click **Save**. These values pre-fill each harness via URL parameters when you launch a card.
3. Click any product card — *SiMaSentry-Med*, *SiMaSentry-Safe*, or *SiMaSentry-Sec*. The harness fades into the iframe view; the URL becomes `?mode=…` (bookmarkable, shareable).
4. Use the **Home** button in the harness header to return to Mission Control. Switching between harnesses uses the in-frame *Med / Safe / Sec* tabs and cross-fades the iframe.
5. Each harness has its own gear icon — open it to override per-app settings (system prompt, language, etc.). The portal does **not** write into harness storage; the harness owns its state.

### Deep links

Skip Mission Control and land directly on a harness:

```
http://localhost:8080/?mode=medical
http://localhost:8080/?mode=safety
http://localhost:8080/?mode=security
```

The legacy hash form (`#health`, `#safety`, `#security`) auto-redirects to `?mode=` on load.

---

## Mission Control at a glance

The landing page has five regions:

- **Header** — SiMaSentry shield logomark, eyebrow + product title, **Online / Offline** network pill (driven by `navigator.onLine` + the `online`/`offline` browser events; no probing of the engine), and a **Configure** button that opens the slide-out drawer.
- **Hero** — short pitch ("Three verticals. One edge runtime.").
- **Product cards** — three side-by-side cards using the per-vertical PNG logos as their icons. Each card has its own SiMa-palette accent color (lavender for Med, rust orange for Safe, lime for Sec — sampled from the shield logo tiles). Click to launch.
- **Edge Diagnostics dropdown** — collapsed by default. Click to expand a 3 × 2 stat grid showing Hardware (linked to the SiMa.ai MLSoC Modalix SOM product page), Environment, Engine, Configured endpoint, Default model, Provider. The endpoint cell never wraps — long URLs ellipsis with the full string available on hover.
- **Why SiMaSentry** — value-proposition section with the SiMaSentry brand graphic on the left and four positive highlight tiles on the right (Operator-grade UI, Domain-tuned prompts, Mission Control orchestration, Air-gapped zero install).

Transitions: a staggered fade-in on first paint, a 320 ms scale-fade when entering a harness, a cross-fade on the iframe when switching between harnesses, and a slide-fade when returning to the landing page. All animations honor `prefers-reduced-motion: reduce`.

---

## How settings hand off (decoupling)

Mission Control persists its own configuration in `localStorage` under the key **`sima-sentry:config`** (`{ provider, baseUrl, model, apiKey }`). It **never** writes into the harnesses' own storage keys.

When you click a card, the portal builds the iframe URL with the saved config as URL parameters:

```
health/index.html?provider=ollama&base_url=http%3A%2F%2Flocalhost%3A9998%2Fv1%2Fchat%2Fcompletions&model=gemma3
```

Each harness already supports these URL parameters and merges them into its own `localStorage` on load (see each harness's README under "Deep-linking with URL parameters"). After the first launch, the harness owns those settings — direct visits without the portal pick them up. If you change something inside a harness's gear, your change wins for that harness; the portal's drawer is just a seed.

The harness header also gets a **Home** button. When the harness is loaded inside Mission Control, the button posts a `{type: "sima-sentry:home"}` message to the parent window, which closes the iframe. When the harness is opened directly (no parent), the button navigates to `../index.html`. Same-origin only — the portal's listener checks `event.source === iframe.contentWindow` for defense in depth.

---

## CORS / cross-origin

If you serve the portal from a workstation (`http://workstation:8080`) and llima runs on a Modalix box (`http://modalix:9998`), the browser fires a CORS preflight before every chat request. llima does not currently advertise CORS headers, so the preflight fails and harness chats are blocked.

**Recommended fix — reverse proxy.** Put both Mission Control and `/v1/*` behind a single origin. Minimal nginx on the Modalix:

```nginx
server {
  listen 8080;
  root /path/to/apps-llima-harnesses;
  index index.html;

  location / { try_files $uri $uri/ =404; }

  location /v1/ {
    proxy_pass         http://127.0.0.1:9998;
    proxy_http_version 1.1;
    proxy_set_header   Host $host;
    proxy_buffering    off;          # for streaming responses
  }
}
```

Then in the **Global Configuration** drawer, set Base URL to `/v1/chat/completions` (relative — same origin, no preflight). Save, click a card, done.

**Ollama fallback.** If you're testing against local Ollama instead of llima, Ollama supports CORS via `OLLAMA_ORIGINS=*`. See [`health/README.md`](health/README.md#cors-configuration-required-when-serving-from-a-different-origin) for the per-OS recipe.

---

## Repo layout

```
apps-llima-harnesses/
├── index.html              ← Mission Control portal
├── style.css
├── app.js
├── README.md               ← this file
├── SiMaSentry.png          ← brand graphic (header brand-mark, value-prop panel, favicon)
├── favicon.ico             ← legacy favicon (unused; SiMaSentry.png is current)
├── .gitignore
├── health/                 ← SiMaSentry-Med
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── README.md
│   └── SiMaSentry-Med.png
├── safety/                 ← SiMaSentry-Safe
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── README.md
│   └── SiMaSentry-Safe.png
└── security/               ← SiMaSentry-Sec
    ├── index.html
    ├── style.css
    ├── app.js
    ├── README.md
    └── SiMaSentry-Sec.png
```

No `node_modules/`, no build artifacts, no lockfiles. Every file you see is what runs.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Header pill reads *Offline* | Browser sees no network interface. Check the workstation's network. On a true air-gapped LAN with the Modalix reachable, the pill should still read *Online* because the LAN counts as a network. |
| Mission Control opens but the iframe is blank | Harness folder missing or URL-param malformed. Open DevTools → Network and inspect the iframe request's HTTP status. |
| Harness inside Mission Control says *Cannot reach …* | The harness's chat request is being CORS-blocked. See the **CORS / cross-origin** section above; the fix is the reverse proxy. |
| Harness inside Mission Control says *HTTP 4xx/5xx* | llima rejected the request — typically the **Model** name doesn't match a loaded model. Open the harness's gear and copy the exact file name from `llima run`. |
| Microphone / camera disabled inside a harness | The portal is on `file://`. Use `python3 -m http.server` (or any static server) so the page is on `localhost`. |
| Mission Control's drawer Save does nothing visible | Browser is in private mode or has localStorage disabled. Open with a regular profile. |
| Harness gear shows old endpoint after Mission Control update | The harness owns its state. The portal seeds via URL parameters on each launch; changes inside the harness's own gear are local to that harness. Reopen via the portal to re-seed. |
| Edge Diagnostics dropdown won't expand | Browser without `<details>` support — extremely old. Upgrade to a modern browser. |

---

## License

MIT — do what you like, no warranty. Not affiliated with OpenAI or Ollama. Built for [SiMa.ai](https://sima.ai/) Llima.
