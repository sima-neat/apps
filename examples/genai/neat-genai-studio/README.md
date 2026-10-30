# Neat GenAI Studio

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | genai, vlm, asr, tts, japanese, multilingual, rag, model-switching, huggingface, openai-compatible, insight |
| Languages | Python |
| Status | stable |
| Binary Name | neat-genai-studio |
| Model | Loaded on demand (e.g. Qwen3-VL-4B-Instruct-Autoround-a16w4) + whisper-small-a16w8-layered-encoder |

## Concept
Runs language, vision-language, and speech models on a Modalix device as the board-side backend for the GenAI tab in Neat Insight, with an OpenAI-compatible API.

The Studio has no user interface of its own. Insight's **GenAI** tab is the UI:
it relays every call to this backend, which listens on `https://<board-ip>:5000`.
The backend starts without a chat model loaded. Through Insight, or directly
through the API, you can:

- find compatible models already on the device, and download models from the supported Hugging Face accounts when the device is online
- load one language or vision-language model at a time without restarting
- chat with text and pictures, transcribe or translate speech with Whisper, and synthesize speech with the installed voices
- answer from local Markdown documents with RAG

Once set up, the Studio runs without internet access. The first `setup.sh` run and searching for or downloading a model from Hugging Face need it (see [Prerequisites](#prerequisites)).

## Preview
![Neat GenAI Studio preview](../../../portal/assets/examples/genai/neat-genai-studio/image.png)

## Prerequisites
- A Modalix DevKit with `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)), the Neat Development Environment, and the Neat Library installed.
- The Neat Library's Python environment at `~/pyneat/bin/python`. If it is somewhere else, set `PYNEAT_PYTHON=/path/to/python-with-pyneat` before running the commands below.
- Neat Insight, to use the GenAI tab.
- Internet access on the board for the first `setup.sh` run and for Hugging Face downloads. Behind a firewall or proxy, allow `pypi.org` and `files.pythonhosted.org` (Python packages), `download.pytorch.org` (PyTorch for document search), `huggingface.co` and its download hosts (models and voices), `raw.githubusercontent.com` (English text-to-speech data for Piper, fetched with NLTK), and `docs.sima.ai` (Supertonic's compiled models, through `sima-cli`).

Run every command below in a shell on the board.

## Install Apps
If the Core and Insight selected by Apps are already installed, run the
install as `NEAT_APPS_SKIP_DEPENDENCIES=1 sima-cli neat install apps` to
skip reinstalling them. See
[Install Apps](https://github.com/sima-neat/apps/blob/main/README.md#install-apps).

Install the Apps bundle, enter it, and name the Studio's directory:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/genai/neat-genai-studio
```

Run the remaining commands from `prebuilt-apps/`. In a new shell, run the `cd`
and `APP_DIR=` lines again first.

## Prepare the Model
Install the Studio's environments, models, and voices once:

```bash
${APP_DIR}/setup.sh
```

`setup.sh` downloads the default speech-to-text model
(`florianvoss/whisper-small-a16w8-layered-encoder`) and the RAG embedding model
(`thenlper/gte-small`) into `/media/nvme/llima/models`, installs the
text-to-speech voices and the Supertonic engine, builds the default RAG
database, and writes `config.local.yaml`. When asked, pick the voice languages
to install. It finishes with `Install complete.`

No chat or vision-language model is installed by default; you load one from
Insight's GenAI tab (or the `/models` API) after startup. To download one during
setup instead:

```bash
CATALOG_MODEL_REPOS="simaai/Qwen3-VL-4B-Instruct-Autoround-a16w4" ${APP_DIR}/setup.sh
```

On a board without NVMe storage, choose another directory for the models:

```bash
LLIMA_MODELS_PATH=/path/to/models \
SUPERTONIC_MODELS_ROOT=/path/to/models/supertonic-tts/models \
${APP_DIR}/setup.sh
```

Other `setup.sh` settings (environment variables; `${APP_DIR}/setup.sh --help`
lists them all):

- `CHAT_MODEL_REPO`: also download **and preload** one chat/VLM model at startup.
- `ASR_MODEL_REPO`: the speech-to-text model installed and made active at
  startup (set to `""` to install none). Current LLiMa needs a layered-encoder
  Whisper build; on a runtime that needs a monolithic encoder pass
  `ASR_MODEL_REPO=simaai/whisper-small-a16w8`. Setup checks the model loads and
  leaves it out of the startup configuration if the runtime refuses it.
- `ASR_CATALOG_MODEL_REPOS`: extra speech-to-text repos to switch between at runtime.
- `ALLOW_HUB_DOWNLOAD`, `HUB_ORGS`: allow Hugging Face downloads through the API
  (default `true`) and which accounts are searched (default
  `simaai TDoSiMa florianvoss`); a repo from another account is refused.
- `TTS_LANGUAGES`, `TTS_OPTIONAL_VOICES`: catalogued voice languages and optional
  voices (for example `mera`). Non-interactive setup installs
  `en,de,es,fr,it,ja,pt,vi,zh`.
- `INSTALL_SUPERTONIC`, `SUPERTONIC_MODELS_ROOT`, `SUPERTONIC_VENV`: the
  MLA-accelerated Supertonic 3 engine (default on), its model files (default
  `/media/nvme/supertonic-tts/models`) and its runtime venv (default
  `./.venv-supertonic`).
- `WEB_HOST`, `WEB_PORT`, `WEB_HTTPS`, `WEB_CORS_ORIGINS`: the backend's address
  and CORS allowlist written to `app.web`.
- `SETUP_RAG`: install document search ([RAG](#rag)), default `1`. On a board
  short on disk space or memory, `SETUP_RAG=0` skips CPU PyTorch, the RAG
  packages, the embedding model and the default database, and writes
  `app.rag.enabled: false`; everything else works as usual.

Re-running `setup.sh` (for example to add a model or voice) regenerates
`config.local.yaml` but keeps your `app.web` settings, the Supertonic paths and
whether RAG is installed (`app.rag.enabled`); the previous file is saved as
`config.local.yaml.bak`.

## Configure
Most users change nothing. `setup.sh` writes `config.local.yaml` from the
packaged defaults in `src/common/config.yaml`, and any compatible model under
the model directory can be loaded without a restart. Edit `config.local.yaml`
only to change:

- `server.models.catalog_dir`: the directory scanned for models (default `/media/nvme/llima/models`).
- `server.models.chat`: chat or vision-language models to load at startup (none by default).
- `server.models.max_resident_chat_models`: how many chat or vision-language models stay loaded at once (default `1`; see [Keep several models loaded](#keep-several-models-loaded)).
- `server.models.asr`: the speech-to-text model active at startup. Models from Hugging Face accounts other than `simaai` are named `<org>@<name>`, for example `florianvoss@whisper-small-a16w8-layered-encoder`.
- `server.hub.allow_download`: whether models may be downloaded from Hugging Face.
- `app.request.max_tokens`: the default response length for chat requests that do not set `max_tokens`.
- `app.rag.enabled`: retrieval-augmented search. Turning it on after a
  `SETUP_RAG=0` install needs the RAG packages: run
  `SETUP_RAG=1 ${APP_DIR}/setup.sh` instead.
- `app.web`: the backend's host, port (default `5000`), HTTPS setting, and
  `cors_origins` (see [CORS](#cors-for-other-browser-front-ends)).

Restart the Studio after editing the file.

## Run
Start the Studio:

```bash
${APP_DIR}/run.sh
```

`run.sh` starts the model server (`${APP_DIR}/src/python/server/main.py`) and
the Studio backend (`${APP_DIR}/src/python/api/main.py`). When startup finishes
it prints the backend address, `https://<board-ip>:5000`, and keeps running
until you press `Ctrl+C`. If `setup.sh` has not been run yet, `run.sh` runs it
first.

Connect Insight:

1. Open Insight and select the **GenAI** tab. On the board itself Insight finds
   the backend at `https://127.0.0.1:5000`; from the Neat Development
   Environment it uses the paired DevKit. Otherwise enter the board address
   (`https://<board-ip>:5000`) in the tab's settings.
2. Load a model from the tab's model settings, or download one from Hugging
   Face there, then send a chat message.

Other commands:

```bash
${APP_DIR}/run.sh status   # running or not, and the backend address
${APP_DIR}/run.sh stop     # stop it from another shell (POST /shutdown does the same)
${APP_DIR}/run.sh --clean  # remove generated venvs, config, RAG db, voices and logs
```

## Expected Result
- `run.sh` prints `Neat GenAI Studio is starting up.` and the backend address.
- `curl -sk https://127.0.0.1:5000/health` answers JSON with `"ok": true` once
  the model server is reachable, and lists the active speech-to-text model and
  the loaded text-to-speech engines.
- In Insight's GenAI tab, after you load a model, a chat message returns a
  streamed reply, recorded speech is transcribed, and replies can be read aloud.

To confirm the model server from a second shell on the board:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

The output lists the speech-to-text model and any chat model you loaded.

## Troubleshooting
- **The model server does not start:** confirm `pyneat` imports in the environment at `PYNEAT_PYTHON` (default `~/pyneat/bin/python`).
- **Insight cannot reach the backend:** check `${APP_DIR}/run.sh status`, then `curl -sk https://<board-ip>:5000/health` from the machine running Insight. The backend uses a self-signed certificate generated on first start; `NEAT_TLS_CERT` and `NEAT_TLS_KEY` select your own.
- **`setup.sh` says the runtime refuses the speech-to-text model:** your Neat runtime needs a speech model with a different encoder layout. Install the older build instead with `ASR_MODEL_REPO=simaai/whisper-small-a16w8 ${APP_DIR}/setup.sh`.
- **`setup.sh` stops on a download (connection refused, timeout, `429`):** `setup.sh` already retries Hugging Face rate limits, server errors and dropped connections a few times, so this is a lasting problem. Check that the board reaches the hosts listed in [Prerequisites](#prerequisites); behind a proxy, export `HTTPS_PROXY` first. Run `${APP_DIR}/setup.sh` again: files already downloaded and verified are kept.
- **A model download fails:** the message says why. For a model that needs approval, request access on Hugging Face and set `HF_TOKEN`; for a model Hugging Face renamed, download the new id it names.
- **A model fails to load with an accelerator error:** restart the Studio (`${APP_DIR}/run.sh stop`, then `${APP_DIR}/run.sh`) to free the accelerator, then load the model again. The Studio does not reset or recover board services itself.
- **`/health` reports a voice engine with `"loaded": false`:** the engine failed to load (for example Supertonic while the accelerator was unavailable); the reason is `accelerator-unavailable` or `load-failed` and the details are in the Studio log. A speech request, or choosing the engine, retries it at most every 30 s.
- **Another instance is running:** `run.sh` refuses to start a second one; `${APP_DIR}/run.sh stop` stops it.
- **Start over:** `${APP_DIR}/run.sh --clean`, then `${APP_DIR}/setup.sh`. Downloaded models are kept; add `CLEAN_SUPERTONIC=1` to also remove the Supertonic model files.

## API
Everything is served on the backend port (HTTPS by default; set
`app.web.https: false` for plain HTTP behind another service). Any other path
answers 404. `GET /` describes the backend and `GET /health` reports readiness,
the API version Insight builds on (`api_version`), and optional features
(`features.rag`).

```text
GET  /health                       readiness: model server, active ASR model, loaded chat models, TTS engines
POST /v1/chat/completions          OpenAI chat (streaming), proxied to the loaded chat/VLM model; "neat_rag" adds document passages
POST /v1/audio/speech              text to speech (WAV)
GET  /v1/audio/voices              engines, voices and languages
POST /v1/audio/transcriptions      speech to text in the spoken language
POST /v1/audio/translations        speech to English text
     (the /audio/... paths without /v1 are the same routes)
GET  /models/status, /models/catalog, /models/card?name=, /models/logs, /models/logs/stream
POST /models/load, /models/unload, /models/asr, /models/delete            {"name": ...}
GET  /models/hub/search?q=, /models/hub/card?repoId=; POST /models/hub/download {"repoId": ...}
POST /benchmark/run, /benchmark/stop; GET /benchmark/status              TTFT / TPS benchmark
GET/POST /tts/engine; GET /supertonic/voices, /piperplus/voices, /voices; POST /supertonic/select, /piperplus/select, /voices/select
GET  /rag/status, /rag/inspect, /rag/search?query=...&k=3; POST /rag/upload (Markdown), /rag/reset, /rag/clear
GET  /board-camera/devices; /board-camera/snapshot?device=/dev/videoN   cameras on the board; a JPEG snapshot
POST /shutdown                     stop the Studio (not reachable cross-origin)
```

Quick checks from the board:

```bash
B=https://127.0.0.1:5000
curl -sk $B/health
curl -sk $B/models/status | python3 -m json.tool
curl -sk -X POST $B/models/load -H 'Content-Type: application/json' -d '{"name":"<catalog-model-name>"}'
curl -sk -N $B/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<catalog-model-name>","stream":true,"messages":[{"role":"user","content":"Say hello."}]}'
```

`src/python/api/apitest/audio_e2e.sh [host:port | base URL]` checks the workflow
end to end against a running backend: synthesized speech is transcribed back to
its words, German speech translates to English, the chat model answers, and,
when `/health` lists `features.rag`, document search returns passages and a
chat with `neat_rag` uses them, and the board camera lists its devices. With
`BOARD_CAMERA_DEVICE=/dev/videoN` set to a camera attached to the board, it
also checks that a snapshot returns a JPEG. It exits non-zero on any failure.
`src/python/api/apitest/backend_runtime_test.sh` exercises the model server and
the speech routes. The other scripts there (`speech.sh`, `transcriptions.sh`,
`translations.sh`, `chat.sh`, `completions.sh`) wrap single calls. They take a
full base URL (`http://…` when `app.web.https` is false) or `STUDIO_URL`.

### Models
`POST /models/load` loads a catalog model and warms it. `GET /models/logs/stream`
streams the load progress as server-sent events. If a load hits an accelerator
error, only the new model's registration is rolled back; the loaded models stay,
and the error is reported.

#### Keep several models loaded
By default one chat/VLM model is loaded at a time, and loading another replaces
it. To keep several loaded side by side, for example to compare them, raise
`server.models.max_resident_chat_models` in `config.local.yaml` (or set
`MAX_RESIDENT_CHAT_MODELS` before `setup.sh`), then restart the Studio. Each
loaded model answers on `/v1/chat/completions` under its own name, and clients
can use them at the same time. The Insight GenAI tab expects one loaded chat
model until it supports several, so keep the default for it.

Past the limit, the Studio never unloads a model on its own, because another
client may be using it. The load is refused with `409` and the loaded models,
and it succeeds once the request names the ones to unload:

```bash
curl -sk -X POST $B/models/load -H 'Content-Type: application/json' \
  -d '{"name":"<catalog-model-name>","unload":["<loaded-model-name>"]}'
```

`GET /models/status` lists the loaded models (`resident`, most recently loaded
first; loading a model that is loaded already moves it to the front) and the limit (`maxResident`). Each catalog entry reports the
accelerator memory the model takes (`acceleratorBytes`), the tokens it reads at
once (`contextTokens`), and its tokenizer (`tokenizerPath`).

#### Limits and what the Studio does at them
The board has fixed limits. The Studio checks them before it asks the board to
do something it can't, and answers with a reason instead of failing silently.

| Situation | What the Studio does |
| --- | --- |
| **More models than the limit** | Refused before anything changes (`409`, `"code": "resident_limit"`, with `resident` and how many to unload in `needed`). |
| **Accelerator memory full**: a model that won't fit beside the loaded ones | Refused before anything is loaded or unloaded (`409`, `"code": "no_room"`, with `needBytes`, `freeBytes` and the loaded models). The budget is `server.models.accelerator_memory_gib` (16 on a Modalix DevKit). |
| **A load fails anyway** (memory not handed back by earlier unloads) | Only the new model is rolled back; the others stay loaded. Restart the Studio to free the accelerator. |
| **Storage full**: a download bigger than the free space | Refused before it starts, keeping 1 GB free for the board. A disk that fills during a download is reported as such. |
| **Context full**: a conversation longer than the model reads at once | The oldest turns are left out so the model still answers, and `X-Context-Dropped` says how many (`X-Context-Window` gives the window). Tokens are counted with the model's own tokenizer when the Studio runs on the same board as the model server, and estimated otherwise. |
| **One message longer than the window** | Refused with the reason instead of an empty reply (`413`, `"code": "context_full"`). |
| **Accelerator busy** (Platform 3.0: `rc=-11`, kernel log `no free bank`, when several models run at once) | Retried once automatically, for streamed and whole replies, when no text had come back yet; otherwise the error is passed on. |

Speech-to-text has its own slot. `POST /models/asr` makes another ASR model
active (evicting the previous one) without touching the chat model. The switch
is not persistent: `server.models.asr` is what a restart re-selects. Set
`STUDIO_ASR_WARMUP=0` to skip the warm-up a switch performs.

`/models/hub/search` lists compatible models in the accounts in
`server.hub.orgs`, and `/models/hub/download` streams the download progress as
NDJSON. Downloads land under `catalog_dir`; a repo from an account other than
`simaai` is stored as `<org>@<name>`.

### Speech and transcription
```text
POST /v1/audio/speech            JSON
  input            required, up to 4096 characters
  model            default | supertonic | piper-plus | piper-tts   (a named engine is used or refused, never swapped;
                   OpenAI's tts-1 / tts-1-hd / gpt-4o-mini-tts mean default; any other name answers 400)
  voice            Supertonic speaker F1-F5 / M1-M5, per request. The Piper engines speak with the voice
                   selected for the language (POST /voices/select, /piperplus/select): omit voice or name
                   that one; any other Piper voice answers 400 rather than a different voice
  speed            0.25-4.0 (default 1.0); clamped to the engine's range, effective value in X-Speed
  response_format  wav (the only format produced; anything else answers 400)
  language         extension: ISO code the text is in (default en; auto means en)
  -> 200 audio/wav with X-Engine, X-Voice, X-Language, X-Speed, X-RTF, X-Audio-Duration, X-Elapsed-Time
     400 {error, param} | 503 {error, reason, engine} when the engine cannot serve the language

POST /v1/audio/transcriptions    multipart/form-data
  file             required (WAV, WebM/Opus, MP4/AAC, ... the server decodes with libavformat; up to 25 MiB)
  model            optional speech-to-text model; default: the active one
  language         optional ISO code or auto (default)
  response_format  json (default) -> {"text"} | verbose_json -> text plus language, language_detected,
                   tts_language, no_speech_prob, avg_logprob, ignored, reason, model | text -> text/plain
  -> headers X-ASR-Model, X-Elapsed-Time; 400 missing file | 413 too large | 503 no ASR active / server unreachable

POST /v1/audio/translations      multipart/form-data, same fields and limits as transcriptions
  -> the speech translated into English (Whisper's translate task); verbose_json adds
     task: "translate", language = the detected source language, tts_language: "en";
     headers X-ASR-Model, X-Task, X-Elapsed-Time

GET  /v1/audio/voices            extension: discovery
  -> {default_engine, languages, engines: [{key, label, loaded, languages, voices: [{id, label, ...}]}]}
```

`utterance_speed` / `utteranceSpeed` are still accepted on the speech route as
deprecated aliases of `speed`. In `verbose_json`, a recording is marked
`ignored` when `no_speech_prob > 0.6` or `avg_logprob <= -1.0`; override the
thresholds with `ASR_NO_SPEECH_THRESHOLD` and `ASR_LOGPROB_THRESHOLD` in the
environment of `run.sh`.

### Text-to-speech engines
| Engine | Licence | Runtime | Languages |
| --- | --- | --- | --- |
| **Supertonic 3** | OpenRAIL model terms (see notices) | PyNeat on the **MLA** (vector field + vocoder), onnxruntime (CPU text front end) | 30+ incl. English, German, Spanish, French, Italian, Portuguese, Japanese, **Korean**, Vietnamese (not Chinese) |
| **piper-plus** | MIT runtime; model-specific terms | onnxruntime (CPU) | Japanese, English, Chinese, Spanish, French, Portuguese |
| **piper-tts** | GPL-3.0 runtime; model-specific terms | onnxruntime (CPU) | English, Chinese, Spanish, French, Portuguese, German, Italian, Norwegian, Vietnamese |

- With `model: default`, a router picks the engine per language. The preferred
  engine is `supertonic` when its runtime loaded and `piper-tts` otherwise;
  `POST /tts/engine` changes the preference for languages more than one engine
  can speak.
- **Supertonic 3** runtime is vendored under `src/python/api/tts/supertonic_sima/`
  (see its README for provenance). It offers ten speakers (F1-F5, M1-M5).
  Each denoising step evaluates conditional and unconditional guidance together
  in one batch-two vector-estimator call; the vocoder remains batch one.
  `SUPERTONIC_VOICE` picks the startup speaker; `SUPERTONIC_STEPS` (5-12,
  default 8) trades quality for latency. It runs in its own venv
  (`./.venv-supertonic`, requirements in `src/python/requirements-supertonic.txt`
  plus the installed PyNeat) and worker process. `setup.sh` downloads its model
  files from [Supertone/supertonic-3](https://huggingface.co/Supertone/supertonic-3)
  and its runtime constants from
  [florianvoss/supertonic-3-sima](https://huggingface.co/florianvoss/supertonic-3-sima)
  at pinned revisions. Both compiled models come from the SiMa model zoo with
  `sima-cli download`; `src/python/api/tts/supertonic_sima/model_assets.json`
  records their addresses and checksums. Every file is
  verified by SHA-256, and each `setup.sh` run repairs missing or changed files.
  Supertonic needs Platform 3.0 or newer; when the runtime is missing the engine
  is not offered. The models root and venv are persisted under
  `app.tts.supertonic` in `config.local.yaml`.
- **CPU engines load on demand** when Supertonic is available; only languages
  Supertonic cannot speak (Chinese, Norwegian) get their Piper voice at startup.
- **piper-tts runs in its own venv** (`.venv-pipertts`) through a subprocess
  worker, because piper-tts and piper-plus both ship a top-level `piper`
  package. Its voices are split into encoder and decoder ONNX files during setup.
- **Japanese** defaults to Piper Plus CSS10; **MERA** is an optional Piper Plus
  model downloaded and verified when it is selected. **Chinese** defaults to the
  dedicated Huayan Piper voice. **Korean** needs Supertonic 3.
- The reviewed catalog is `src/python/api/tts/voice_catalog.json`: compact
  licence labels, pinned upstream revisions and SHA-256 checksums. Models under
  `CC-BY-NC-SA-4.0` are excluded. See
  [Third-party TTS models](https://github.com/sima-neat/apps/blob/main/examples/genai/neat-genai-studio/THIRD_PARTY_TTS_MODELS.md)
  for attribution notices.

### RAG
RAG is enabled by default after `setup.sh`, which builds
`src/python/api/milvus.db` (and `milvus.meta.json`) from the bundled
`src/common/rag/neat.md`. The backend starts the search service when the
database exists. A `SETUP_RAG=0` install leaves RAG off and the RAG requests
below answer 409; `SETUP_RAG=1 ${APP_DIR}/setup.sh` adds it later.

`POST /v1/chat/completions` answers from the documents when the request adds
`"neat_rag": true`, or `{"k": 5}` for the number of passages (1 to 10, default
3). The backend searches with the last user message, adds the matching passages
before it, and removes the field before the request reaches the model server.
The response headers `X-RAG-Hits` and `X-RAG-Sources` say which passages were
used. A request with `neat_rag` answers 409 when RAG is off or the documents
were cleared, 503 while the search service is starting, and 400 when the last
user message has no text. Every chat request is checked first: `messages` must
be a non-empty list of objects with a string `role` and a string or list
`content`; anything else answers 400 in the OpenAI error shape.

```bash
B=https://127.0.0.1:5000
curl -sk -F file=@/path/to/document.md $B/rag/upload   # replace the database with this document
curl -sk -X POST $B/rag/reset                           # back to the bundled document
curl -sk -X POST $B/rag/clear                           # remove all documents
curl -sk "$B/rag/inspect" | python3 -m json.tool        # what was ingested
```

`POST /rag/upload`, `/rag/reset` and `/rag/clear` answer `200` with
`text/plain` progress, one line per step; the last line starts with ✅ on
success or ❌ on failure, so check that line rather than the status code. They
answer JSON errors before the update starts: 409 while another update is
running or when RAG is off, 413 for a document larger than 10 MiB, and 400 for a
missing or non-Markdown file. An update replaces the whole database; if it
fails, or the client disconnects before it finishes, the previous database is
restored. `GET /rag/status` reports whether the search service is up
(`ok`, `starting`, `failed`, `no-database`, or `disabled`).

Passage text is added to the user's question, so the model follows any
instructions inside an uploaded document. Upload only documents you trust;
`/rag/upload` is reachable by any client that can reach the backend.

To use a different embedding model or disable RAG, edit `app.rag` in
`config.local.yaml`. To build the database from another Markdown file offline:

```bash
"${APP_DIR}/.venv/bin/python" "${APP_DIR}/src/python/rag/create_db.py" \
  --input /path/to/document.md \
  --output "${APP_DIR}/src/python/api/milvus.db" \
  --embedding-model "${LLIMA_MODELS_PATH:-/media/nvme/llima/models}/gte-small"
```

### CORS for other browser front ends
Insight relays every call, so it needs no CORS. Browser pages on other origins
that call the backend directly are refused by default; list the origins that
may call the API, or `*`:

```bash
BACKEND_CORS_ORIGINS="https://insight.local:8443,http://10.0.0.5:3000" ${APP_DIR}/run.sh
```

To keep the allowlist across launches, set `app.web.cors_origins` in
`config.local.yaml` (a comma-separated string or a YAML list); the environment
variable overrides it for one run, and setting it empty turns CORS off for that
run. Allowed origins get CORS headers (the `X-*` timing and engine headers are
exposed) and preflight answers on the API paths; `/shutdown` never does.

### Update and clean up
To update, install the newer Apps release with `sima-cli neat install apps` and
run `${APP_DIR}/setup.sh` again. (`${APP_DIR}/run.sh update` refreshes a source
checkout or a single-example download instead; see Development From Source.)

`${APP_DIR}/run.sh --clean` stops a running instance, lists what it will remove
(the venvs, `config.local.yaml`, the RAG database, downloaded TTS voices,
`__pycache__`, the pid file and `*.log`) with the total size, and asks to
confirm (`-y` or `CLEAN_YES=1` skips the prompt). Downloaded chat/VLM/ASR models
under `catalog_dir` are kept.

Shutdown is graceful: both processes handle `SIGTERM`, so the model server
releases its models from the MLA and the backend stops the RAG worker (a
force-kill follows only after `SHUTDOWN_GRACE_SECONDS`, default 10).

### Manual Process Start
Use this only when you want two explicit terminals. `run.sh` also sets up the
text-to-speech workers, which this manual start does not.

Terminal 1, model server + control API (from `prebuilt-apps/`). The server runs
in the Neat environment, which may not have PyYAML, so give it a JSON copy of the
config made with the app's own venv, as `run.sh` does:

```bash
APP_DIR=examples/genai/neat-genai-studio
"${APP_DIR}/.venv/bin/python" -c 'import json, sys, yaml; json.dump(yaml.safe_load(open(sys.argv[1])) or {}, open(sys.argv[2], "w"))' \
  "${APP_DIR}/config.local.yaml" "${APP_DIR}/.neat-genai-server-config.json"
~/pyneat/bin/python "${APP_DIR}/src/python/server/main.py" \
  --config "${APP_DIR}/.neat-genai-server-config.json"
```

Terminal 2, backend (from `prebuilt-apps/`):

```bash
APP_DIR=examples/genai/neat-genai-studio
"${APP_DIR}/.venv/bin/python" "${APP_DIR}/src/python/api/main.py" \
  --config "${APP_DIR}/config.local.yaml"
```

The model server's control API (`http://127.0.0.1:9997/control/status`,
`/control/load`, `/control/unload`, `/control/asr`) is local only; the backend's
`/models/*` routes proxy it.

## Source Files
- Launcher and setup: `run.sh`, `setup.sh`
- Model server and control API: `src/python/server/`
- Backend API, speech, and voices: `src/python/api/`
- Shared configuration and helpers: `src/common/`, `src/python/shared/`
- RAG tools and default document: `src/python/rag/`, `src/common/rag/`
- Manual API scripts: `src/python/api/apitest/`

## Development From Source
See the Apps repository [contributor guide](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md)
for contribution requirements. The repository (not the installed bundle) also
carries the host-runnable unit suites under `tests/python/`, collected by
`tests/python/test_unit.py`; `./tests/test.sh --unit` runs them and they need
only pytest and PyYAML.

Outside the installed bundle, the Studio can also be fetched on its own with
`curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- neat-genai-studio`.
`${APP_DIR}/run.sh update` refreshes such a download, or runs `git pull` in a checkout;
`NEAT_APPS_BRANCH=<branch>` picks the branch and `UPDATE_DEPS=1` also refreshes
the Python environments.

The backend folder was called ui before it became `src/python/api/`. After
`run.sh update` from such a version, the next `run.sh` start moves the
installed voices, RAG database and uploads into `src/python/api/`.

Do not commit the generated `config.local.yaml`, `src/python/api/milvus.db`, or
`src/python/api/milvus.meta.json`.
