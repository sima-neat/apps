# Neat GenAI Studio

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | genai, vlm, asr, tts, japanese, multilingual, rag, model-switching, huggingface, markdown, openai-compatible |
| Languages | Python |
| Status | experimental |
| Binary Name | neat-genai-studio |
| Model | Loaded on demand (e.g. Qwen3-VL-4B-Instruct-GPTQ-a16w4) + whisper-small-a16w8 |

## Concept
Neat GenAI Studio is an immersive multimodal assistant and model-management
workbench. It hosts SiMa-supported GenAI models through Neat's
OpenAI-compatible server and drives them from a polished Flask UI that adds:

- **Switch LLMs/VLMs on the fly** — pick any compatible model from an on-disk
  catalog and it loads at runtime, with no restart. Only **one chat/VLM model is
  resident at a time**: loading a new one clears all other chat/VLM models (the
  ASR model is always kept), so the MLA holds just the active model. The UI is
  decoupled from the models: the server and UI **start with no model resident**,
  and you load one on demand (or download one first). `run.sh` also puts the MLA
  in a clean slate on launch by stopping any stale model processes from a
  previous run.
- **Download compatible models from Hugging Face** — when the board is online,
  search the `simaai` (official) and `TDoSiMa` (community) model accounts and
  download new models straight into the studio, then load them like any other.
- **Live Markdown rendering** — assistant replies render Markdown (headings,
  lists, tables, bold, links) with syntax-highlighted code blocks and
  copy-to-clipboard buttons, instead of raw text.
- **Font customization** — choose a UI font family and size (or type any locally
  installed family); the choice persists in the browser.
- **Multilingual text-to-speech** — a small multi-engine router speaks replies
  in the selected language, including **Japanese** through a dedicated Piper
  voice or the optional piper-plus multilingual engine. See
  [Text-to-speech](#text-to-speech-voices--languages) below.
- **Board camera input** — besides the browser webcam, the backend can grab
  stills from a camera plugged into the devkit board itself (`/dev/video*`):
  the **⌾** button on the camera dock and the **Board Camera** button in the
  full-screen Vision view fetch a frame via `GET /board-camera/snapshot`
  (`GET /board-camera/devices` lists the nodes; `?device=` or
  `NEAT_CAMERA_DEVICE` picks one). API users can instead send
  `useBoardCamera=true` (optionally `boardCameraDevice`) with a `/upload` chat
  request to attach a fresh board frame server-side. Uses the same capture
  fallbacks as the CLI's `/camera` mode (`ffmpeg`, `fswebcam`,
  `libcamera`/`rpicam`, OpenCV).
- **Drag & drop images** — drop an image anywhere on the page to attach it to
  the chat (or onto the full-screen Vision view to ask about it).
- Image/text chat, audio transcription (ASR), RAG, system-prompt control, chat
  history, voice selection, and abort.

All UI assets — the Markdown renderer, sanitizer, syntax highlighter, and the
bundled fonts — are served locally, so the studio runs **fully offline** on the
board. Hugging Face is used only when you explicitly search or download.

The demo runs as two processes. `src/python/server/main.py` reads the `server`
section of the selected config, starts the Neat OpenAI-compatible server, and
also runs a small **model-management control API** (localhost only) that loads
and unloads models on the running server. `src/python/ui/main.py` reads the
`app` section and starts the Flask UI, which proxies to that control API.
`src/common/config.yaml` is the tracked template; `./setup.sh` writes the
runnable local config to `config.local.yaml`.

Runtime ownership is split deliberately:

- Neat hosts the OpenAI-compatible `/v1/chat/completions` endpoint for text and
  image chat, and `/v1/audio/transcriptions` for ASR.
- The model-management control API (`127.0.0.1:9997`) scans the model catalog
  and loads/unloads models at runtime via `pyneat`'s `add_model` / `remove_model`.
- The Flask app owns UI state, Markdown rendering, font settings, system prompt,
  chat history, abort, uploaded images, microphone recordings, Piper TTS, and the
  Hugging Face search/download (it writes new models into the catalog directory).
- RAG is app-side and optional, using the local VectorDB service and the active
  chat model.

## Preview
Neat GenAI Studio UI:

![Neat GenAI Studio preview](../../../portal/assets/examples/genai/neat-genai-studio/image.png)

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- The model server uses the Python environment where `pyneat` is available. The default scripts assume:

```text
~/pyneat/bin/python
```

Set `PYNEAT_PYTHON=/path/to/python-with-pyneat` if your Neat Library environment is
somewhere else.

## Install Apps
Fetch only Neat GenAI Studio and enter its directory. This avoids downloading
the complete Apps bundle:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- neat-genai-studio
cd neat-genai-studio
```

## Prepare the Model
Install the UI virtual environment, Whisper ASR model, GTE-small embedding
model, TTS voices (piper-tts + the piper-plus model), default RAG database, and
generated local config:

```bash
./setup.sh
```

At the end, `setup.sh` offers to create a **`neat-ai`** shell alias for `./run.sh`
so you can start the studio from any directory. On the eLxr board (interactive
SSH sessions are login shells) it is written to `~/.bash_profile`; for zsh it goes
to `~/.zshrc`. Answer the prompt, or set `CREATE_ALIAS=1`/`0` to skip it
non-interactively; after it's added, `source ~/.bash_profile` (or open a new
shell), then run `neat-ai`, `neat-ai --cli`, `neat-ai stop`, etc.

> You can skip running `setup.sh` yourself: **`./run.sh` runs it automatically on
> the first launch** if it hasn't completed. Opt out with `AUTO_SETUP=0` (it then
> errors with a hint instead of installing).

**No chat/VLM model is downloaded by default.** The UI starts decoupled with no
model resident — download one from the in-UI Hugging Face panel (or seed the
catalog at install time, below). By default, `setup.sh` downloads only:

- `simaai/whisper-small-a16w8` (ASR)
- `thenlper/gte-small` (RAG embedding)

Downloaded models are stored under `/media/nvme/llima/models` by default. That
directory is the **model catalog**: any compatible model directory under it can
be loaded on the fly from the UI. On a system without NVMe, set
`LLIMA_MODELS_PATH` to another writable location:

```bash
LLIMA_MODELS_PATH=/workspace/neat/models_genai ./setup.sh
```

Seed the catalog with one or more chat/VLM models at install time instead of
downloading them from the UI (space-separated Hugging Face repos):

```bash
CATALOG_MODEL_REPOS="simaai/<a-chat-or-vlm-repo> simaai/<another-repo>" ./setup.sh
```

Other useful environment variables:

- `CHAT_MODEL_REPO` — optionally download **and preload** one chat/VLM model at
  startup (empty by default, i.e. none).
- `MAX_RESIDENT_CHAT_MODELS` — kept for advanced use; by default only one
  chat/VLM model is resident and loading a new one clears the others.
- `ALLOW_HUB_DOWNLOAD` — `true`/`false` to enable/disable in-UI Hugging Face
  downloads (default `true`).
- `TTS_LANGUAGES` — comma- or space-separated catalogued server-TTS languages to
  install. Interactive setup prompts when this is unset; non-interactive setup
  defaults to `en,de,es,fr,it,ja,pt,vi,zh`.
- `TTS_OPTIONAL_VOICES` — optional voice ids to install, for example
  `mera,en_US-ljspeech-medium,zh_CN-chaowen-medium`.

The UI virtual environment is stored under `./.venv` unless `APP_VENV` is set.
The generated config is stored at `./config.local.yaml` unless `CONFIG_PATH` is
set. RAG is enabled by default and uses `src/python/ui/milvus.db`.

### Configure the model catalog
After install, edit `config.local.yaml` to change the catalog, memory budget, or
the models loaded at startup:

```yaml
server:
  models:
    catalog_dir: /media/nvme/llima/models   # scanned for loadable models
    max_resident_chat_models: 1             # chat/VLM models resident at once
    chat: []                                # optional: none preloaded by default.
      # To preload a model at startup instead, list it here, e.g.:
      # - name: Qwen3-VL-4B-Instruct-GPTQ-a16w4
      #   path: /media/nvme/llima/models/Qwen3-VL-4B-Instruct-GPTQ-a16w4
    asr:
      name: whisper-small-a16w8
      path: /media/nvme/llima/models/whisper-small-a16w8
  hub:
    allow_download: true
    org: simaai
```

Any compatible model directory (one containing `devkit/` with `vlm_config.json`
or `whisper_config.json`) placed under `catalog_dir` is discovered automatically
and can be loaded from the UI — no need to list it under `chat:` or restart.

Both `chat:` and `asr:` are **optional**. Use `chat: []` (and omit `asr:`) to
start the server and UI with no model resident; the UI comes up and prompts you
to load or download a model. This is the fully decoupled mode.

## Run
Start both the Neat OpenAI-compatible server (with the control API) and the Flask UI:

```bash
./run.sh          # or `neat-ai` if you created the alias
```

`run.sh` prints the web UI URL to open in a browser. Settings live behind the ⚙
icon; the **Models** tab lists your downloaded models (search, load/unload,
delete) and the **Add Model** tab lists models available to download from Hugging
Face with their download size and the NVMe free space remaining. When you're done,
either `Ctrl+C` the terminal or use the **⏻ Shutdown** button in the
sidebar — it stops both the UI and the model server gracefully (same as
`./run.sh stop`).

### Terminal chat (CLI)
Prefer the terminal? `--cli` starts the model server (same MLA reset/clean-slate
as usual) and drops you into an interactive chat instead of the web UI:

```bash
./run.sh --cli    # or `neat-ai --cli`
```

On an interactive start it first asks what you want to do — **Chat with a model**,
**Benchmark model(s)**, **Download a model** (when online), or **go straight to the
prompt**. Skip the menu by jumping straight to a mode:

```bash
./run.sh --chat [MODEL]        # chat now (load MODEL first if given)
./run.sh --download [REPO]      # download REPO (or prompt), then chat
./run.sh --benchmark [MODEL]    # benchmark MODEL (or prompt), then chat (alias --bench)
```

It then talks straight to the model server (the control API to list/load models
and the OpenAI endpoint to stream replies). Type a message to chat; commands:

```text
/models          list catalog models (● loaded, ○ not)
/load [name]     load a model — no name pops an arrow-key picker (↑/↓, Enter)
/download        browse Hugging Face — pick one, several, or all models to
                 download (Space to multi-select, 'a' for all), then load one
/unload [name]   unload a model — no name unloads the loaded LLM/VLM
/delete [name]   delete a model's weights from disk — no name pops a picker;
                 asks to confirm (irreversible; /rm, /remove)
/image [path]    attach an image to the next message (VLM only; no path prompts)
/camera [device] arm the board camera — every message then auto-sends a fresh
                 frame to the VLM (/camera off to stop; /dev/video16 — /cam, /webcam)
/benchmark …     TTFT/TPS benchmark — see the Benchmark section below
/system <text>   set a system prompt (empty clears it)
/new             clear the conversation
/export [file]   save this chat to a .log file (default neat-chat-<time>.log)
/reset           reset the accelerator (MLA) and restart the model server
/tokens <n>      set max response tokens
/rag [filter]    inspect the RAG database — list chunks (/docs; filter narrows)
/rag on|off      toggle RAG-augmented chat (top passages prepended to prompts)
/rag search <q>  semantic search — show top matches without asking the model
/rag db [path]   show, or switch to, which milvus.db is served ('default' reverts)
/rag status      show the RAG toggle, active database and service state
/rag reset|clear rebuild from the default document, or clear all RAG documents
/help  /quit     help / exit (aliases: /exit, /bye, /q, Ctrl+D)
```

Replies render live as Markdown, and LaTeX math is converted to Unicode for the
terminal (`$E = mc^2$` → `E = mc²`, `\frac`, `\sqrt`, Greek letters, `\sum`, …).

Ctrl+C stops the current reply; it prints per-response timing (tokens, TTFT,
tok/s). Exiting shuts the model server down. Use **↑/↓** at the prompt to recall
previous prompts (history persists across sessions in `~/.neat_ai_history`).

`/camera <index>` **arms** a camera attached **to the board** (the CLI runs on
the board, unlike the web UI, which uses the browser's camera). Once armed, every
message auto-grabs a fresh frame and sends it to the VLM — `/camera off` disarms,
and a `📷` in the prompt shows it's live. It shells out to whatever capture tool
the board has — `ffmpeg`, `fswebcam`, or `libcamera`/`rpicam` — so install one if
the board has none, or set `NEAT_CAMERA_DEVICE` to the right `/dev/video*` node.
An explicit `/image` still takes precedence for that one message.

### Stopping
Press `Ctrl+C` in the terminal running `run.sh`, or from another shell:

```bash
./run.sh stop      # cleanly stop a running instance
./run.sh status    # report whether it is running
```

### Update
Pull the latest version of the studio, then refresh dependencies. Your models,
`config.local.yaml`, both venvs, and the RAG database are **preserved**:

```bash
./run.sh update              # git pull in a checkout, else re-fetch the example
NEAT_APPS_BRANCH=develop ./run.sh update   # update from a specific branch
UPDATE_DEPS=0 ./run.sh update              # source only, skip the setup.sh refresh
```

In a full `apps` git checkout this runs `git pull`; if you fetched just this
example with `get-example.sh`, it re-downloads the release archive and overlays
the source (user data isn't in the archive, so it's left untouched).

### Clean up
Remove everything the app generated (both venvs, `config.local.yaml`, the RAG
database, downloaded TTS voices, `__pycache__`, the pid file, and any `*.log`)
to reclaim space or start fresh:

```bash
./run.sh --clean        # lists what will be removed, then asks to confirm
./run.sh --clean -y     # skip the prompt (or CLEAN_YES=1)
```

It stops a running instance first and lists each target with the total size
before deleting. Downloaded chat/VLM/ASR models under `catalog_dir` are **kept**
(they're large and shared); re-run `./setup.sh` afterwards to reinstall the venvs.

Shutdown is graceful: both processes handle `SIGTERM`, so the model server
releases its models from the MLA and the UI stops the RAG worker before exiting
(a force-kill only happens if they don't stop within `SHUTDOWN_GRACE_SECONDS`,
default 10). `run.sh` records its PID in `.neat-genai-studio.pid` (used by
`stop`/`status`) and refuses to start a second instance while one is running.

On launch, `run.sh` first puts the MLA in a clean slate. Models are loaded into
the MLA **shared-memory dispatcher** (`mlashmcomplex`, service
`simaai-appcomplex.service`), which persists across client processes — so a
stale or crashed run leaves models resident and the next load fails with
`MLA_LOAD_FAILED`. `run.sh` therefore:

1. stops any stale model-server/UI processes from a previous run,
2. resets the MLA dispatcher (releasing every model on the device), and
3. waits for the model-server port to free up.

Step 2 needs privileges. Run `run.sh` as a user with (passwordless) `sudo`, or
as root. It restarts only `simaai-appcomplex.service` and re-runs
`init_mla_memory.sh`; the application does not invoke the board-wide runtime
recovery script. Overrides:

- `MLA_RESET_CMD="my-reset-tool"` — run your own reset command instead.
- `MLA_DISPATCHER_SERVICE=<unit>` — use a different dispatcher service name.
- `MLA_RESET=0` — skip the MLA reset entirely.

If you hit `MLA_LOAD_FAILED` right now, reset the runtime once by hand:

```bash
sudo systemctl restart simaai-appcomplex.service && sudo /usr/bin/init_mla_memory.sh
```

Open the Flask UI:

```text
https://<target-ip>:5000
```

The Neat OpenAI-compatible server listens on `http://127.0.0.1:9998`, and the
model-management control API on `http://127.0.0.1:9997`.

Check that the startup models are hosted:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

### Switch models on the fly
In the UI's **Settings → Models** tab, your downloaded models are shown in a
searchable list (labelled *Downloaded — on this board*). Loaded models are marked
`● loaded`, on-disk ones `○ downloaded`; press **Load** on a not-yet-loaded model
to load it at runtime and unload all other chat/VLM models (whisper is always
kept), so the MLA holds just the active model. A **Load status** panel pins to the
top of the tab and shows the live progress bar while it loads. The studio cancels
the outgoing model's in-flight generation and waits for its memory to be released
before loading the new one, then warms it so your first message is instant.

If a switch still hits a busy/wedged accelerator, the studio automatically
requests a **supervised MLA reset**: the model server exits with a sentinel code,
`run.sh` resets the MLA dispatcher and relaunches it (the UI stays up and
reconnects). Disable this with `MLA_RESET_ON_SWITCH=0` (then a failed switch just
reports an error asking you to restart).

### Download models from Hugging Face
When the board is online, the **Settings → Add Model** tab appears (it's hidden
offline). It lists compatible `simaai` models *available to download*, each with a
**download size** badge (⬇ so you know how much space it needs) and a **`💾 NVMe
storage: … free`** readout so you can tell whether it will fit. Filter/search,
click **Download**, watch the progress bar, and the model moves to the **Models**
tab ready to load. Downloads land under `catalog_dir`. Set `HF_TOKEN` for gated
repos.

### Benchmark (TTFT / TPS)
The performance half of SiMa's **MoLE** (Modalix Language-model Evaluator) measures
**Time-To-First-Token (TTFT)** and **Tokens-Per-Second (TPS)** by streaming from
the on-device model. Open it with the speedometer icon in the header.

- **Pick one, several, or all models.** The *Models to benchmark* control is a
  multi-select (with **Select all downloaded**). Each selected model is loaded and
  benchmarked in turn.
- **Comparison + export.** With 2+ models you get a side-by-side bar chart and
  table (TPS mean/p90, TTFT, tokens, σ) with the best TPS/TTFT highlighted, plus
  **⤓ CSV / ⤓ JSON** export of the results.
- Configure **Runs** and **Output tokens**; each run streams live and the summary
  reports min/max/avg/median/σ/p90.

In the CLI, use `/benchmark`:

```text
/benchmark                 benchmark the active model (5 runs · 128 tokens)
/benchmark all             benchmark every downloaded LLM/VLM, then a comparison table
/benchmark m1,m2 5 128     specific models · 5 runs · 128 tokens each
```

After a CLI run it offers to export the results to a `.csv` or `.json` path.
Ctrl+C stops the current run. (Accuracy tasks like hellaswag/piqa still need the
host `llima-benchmark` CLI.)

### SiMaSentry Solutions (Med / Safe / Sec demo harnesses)
Three vertical AI harnesses — **SiMaSentry-Med** (clinical VLM chat + diagnostic
imaging workbench), **SiMaSentry-Safe** (PPE & hazard inspection with live camera
zones) and **SiMaSentry-Sec** (SOC threat analysis + change detection) — are
vendored into the Studio and open from the **shield icon** in the header.

- Picking a card launches the harness full-screen, **auto-wired to the currently
  loaded model** through a same-origin `/v1/chat/completions` proxy (the Studio
  page is HTTPS while the model server is HTTP, so the proxy avoids
  mixed-content/CORS blocks).
- The harnesses are **vision-centric** — load a VLM for the image features. A
  badge and confirmation warn when the loaded model has no vision support.
- The harness's ⌂ Home action returns to the launcher grid; ✕ (or Esc) closes it.
- The suite also works standalone at `https://<board>:5000/solutions/` (the
  SiMaSentry Mission Control portal), handy for kiosk setups.
- Safe's PPE Inspector uses the browser camera and all three support voice
  in/out — grant camera/microphone permission when prompted (HTTPS required,
  which the Studio already serves).

### Markdown & fonts
Assistant replies render Markdown live. Hover a reply to **copy the whole
response** (top-right button), and hover any code block to **copy the code**.
Under **Appearance**, pick a font family and size or type any locally installed
family; the choice is saved in the browser. The dark/light theme toggle is in the
Settings header.

### Text-to-speech (voices & languages)
Spoken replies use a **multi-engine router** that picks the best offline TTS
engine per language. The spoken language follows the **Transcription language**
selector in Settings.

| Engine | Licence | Runtime | Languages |
| --- | --- | --- | --- |
| **piper-plus** | MIT runtime; model-specific terms | onnxruntime (CPU) | Japanese, English, Chinese, Spanish, French, Portuguese |
| **piper-tts** | GPL-3.0 runtime; model-specific terms | onnxruntime (CPU) | English, Chinese, Spanish, French, Portuguese, German, Italian, Norwegian, Vietnamese |
| **Browser** (Web Speech API) | — | client-side (your browser / OS) | any language your device has a voice for |

- **Japanese** defaults to Piper Plus CSS10. CSS10 is declared public domain;
  the multilingual base model is CC BY 4.0 and its attribution is preserved in
  [the TTS notices](THIRD_PARTY_TTS_MODELS.md).
- **MERA** is an optional Piper Plus model published under Apache 2.0, with the
  CC BY 4.0 base-model attribution preserved. Selecting it downloads and
  verifies it automatically.
- **Chinese** defaults to the dedicated Huayan Piper voice. Chaowen is an
  optional second Chinese voice; Piper Plus CSS10 remains available as the
  multilingual alternative.
- **Korean has no server-side TTS model.** Use Browser TTS when the client has a
  Korean voice; otherwise replies remain text-only.
- **Voice engine** — a **Settings → Voice engine** dropdown chooses which engine
  is preferred for languages more than one can speak (piper-plus or piper-tts).
  Languages only one engine supports are unaffected.
- **Browser** — selecting the **Browser** engine speaks replies on the client with
  the Web Speech API instead of synthesizing on the board (no server compute). The
  server still cleans each sentence (Markdown/LaTeX stripped), so the browser
  utters clean text; pick a device voice under **Settings → Browser voice**. It
  works for any language your browser has a voice for.
- `setup.sh` asks which catalogued languages to install. In non-interactive use,
  set `TTS_LANGUAGES`, and use `TTS_OPTIONAL_VOICES=mera` to add MERA. The UI
  lists catalogued voices per language and downloads a missing voice when it is
  activated.
- CSS10 is the default Piper Plus model. It applies across all six Piper Plus
  languages; MERA is the only optional multilingual model.
- **piper-tts runs in its own venv** (`.venv-pipertts`). piper-tts and piper-plus
  both ship a top-level `piper` package and can't share one environment, so
  `setup.sh` installs piper-tts separately and the UI reaches it through a
  subprocess worker (`pipertts_worker.py`); `run.sh` exports `PIPERTTS_PYTHON`
  for this. If the venv is missing, dedicated voices are skipped and Piper Plus
  keeps working.
- The default router preference is `piper-tts`. Selecting `piper-plus` under
  **Settings → Voice engine** switches supported languages to the active
  multilingual voice.

The authoritative reviewed catalog is `src/python/ui/voice_catalog.json`. Each
entry has a compact licence label, pinned upstream repository revision, and
SHA-256 checksums. Models under `CC-BY-NC-SA-4.0` are excluded. Runtime
discovery ignores models outside the catalog. See
[Third-party TTS models](THIRD_PARTY_TTS_MODELS.md) for attribution notices.

### Manual Process Start
Use this only when you want two explicit terminals.

```bash
export EXAMPLE_DIR="${PWD}"
```

Terminal 1, model server + control API:

```bash
source ~/pyneat/bin/activate

python "${EXAMPLE_DIR}/src/python/server/main.py" \
  --config "${EXAMPLE_DIR}/config.local.yaml"
```

Terminal 2, Flask UI:

```bash
source .venv/bin/activate

python "${EXAMPLE_DIR}/src/python/ui/main.py" \
  --config "${EXAMPLE_DIR}/config.local.yaml"
```

The supported entrypoints are `src/python/server/main.py` for model hosting and
`src/python/ui/main.py` for the UI.

## RAG
RAG is enabled by default after `./setup.sh`.

The installer downloads `thenlper/gte-small`, stores it under the configured
models directory, and creates:

```text
src/python/ui/milvus.db
src/python/ui/milvus.meta.json
```

### Inspect the RAG database
See exactly what has been ingested — the source, chunk count, embedding model,
and every chunk (header breadcrumb + text):

- **Web UI**: **Settings → Knowledge (RAG) → Inspect RAG DB** opens a browser with
  a filter box.
- **CLI**: `/rag` lists the chunks, `/rag <filter>` narrows by a substring
  (alias `/docs`).

The UI reads through the running VectorDB service (which owns the DB file); the
CLI reads the service if it's up, otherwise the `milvus.db` file directly — so the
single-writer database is never opened twice.

### RAG-augmented chat (CLI)
The web UI has a **Search RAG Database** toggle; the CLI has the same, plus a way
to switch which database it searches:

- `/rag on` / `/rag off` — when on, each prompt is first used to retrieve the top
  passages from the database, which are prepended to that turn as context (your
  chat history keeps the clean prompt, so context isn't re-fed every turn).
- `/rag search <query>` — a one-off semantic search that prints the top matches
  without asking the model.
- `/rag db <path>` — point the CLI at a different `milvus.db` (`/rag db default`
  reverts). Inspection, search and augmentation then all use that file.
- `/rag status` — show the toggle, the active database and the service state.

RAG needs the VectorDB service (semantic search). In CLI mode the studio's web
service usually isn't running, so the CLI **starts its own** worker on first use
(loading the embedding model takes a moment) and stops it on exit. If the studio
*is* running, the CLI shares its service instead of starting a second one — and
`/rag status` shows which database that shared service actually serves (a pending
`/rag db` override only takes effect once the running service stops).

Retrieved context is size-capped before it's added to a prompt so it can't
overflow the on-board model's small context window. Because the single-writer
`milvus.db` is guarded by port reachability, avoid **cold-starting the CLI's RAG
and the web UI at the same instant** against the same database.

### Reset or clear the RAG database
- **Reset to Default** rebuilds RAG from the bundled `src/common/rag/neat.md`.
- **Clear** removes all ingested documents.

In the **Web UI**: **Settings → Knowledge (RAG)** → *Reset to Default* / *Clear RAG
DB* (both confirm first). In the **CLI**: `/rag reset` and `/rag clear`. The CLI
operations run only when the RAG service isn't holding the database file open
(otherwise it points you to the UI buttons).

To point RAG at a different local embedding model or disable it, edit
`config.local.yaml`:

```yaml
app:
  rag:
    enabled: true
    embedding_model_dir: /path/to/llima/models/gte-small
```

To rebuild the RAG database from another Markdown file:

```bash
export EXAMPLE_DIR="${PWD}"
source .venv/bin/activate

python src/python/rag/create_db.py \
  --input /path/to/document.md \
  --output src/python/ui/milvus.db \
  --embedding-model "${LLIMA_MODELS_PATH:-/media/nvme/llima/models}/gte-small"
```

Do not commit generated files:

```text
${EXAMPLE_DIR}/src/python/ui/milvus.db
${EXAMPLE_DIR}/src/python/ui/milvus.meta.json
${EXAMPLE_DIR}/config.local.yaml
```

## Verify
Use these checks after the model server and Flask UI are running.

Check hosted model names:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

Check the model-management control API:

```bash
# Catalog + loaded state + memory budget + Hugging Face availability
curl -s http://127.0.0.1:9997/control/status | python3 -m json.tool

# Load a catalog model at runtime (no restart), then confirm it via /v1/models
curl -s http://127.0.0.1:9997/control/load \
  -H 'Content-Type: application/json' \
  -d '{"name":"<catalog-model-name>"}' | python3 -m json.tool

# Unload it again
curl -s http://127.0.0.1:9997/control/unload \
  -H 'Content-Type: application/json' \
  -d '{"name":"<catalog-model-name>"}' | python3 -m json.tool
```

Check text chat:

```bash
CHAT_MODEL="<chat-model-name>"

curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${CHAT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Say hello in Markdown.\"}]}],\"max_tokens\":32}"
```

Check ASR with any short WAV file:

```bash
ASR_MODEL="<asr-model-name>"
AUDIO_FILE="/path/to/audio.wav"

curl -s http://127.0.0.1:9998/v1/audio/transcriptions \
  -F "model=${ASR_MODEL}" \
  -F "language=auto" \
  -F "file=@${AUDIO_FILE}"
```

The Studio defaults to Whisper's standard silence filter: a recording is
ignored when `no_speech_prob > 0.6`, unless `avg_logprob > -1.0` provides
strong evidence that speech was decoded. Automatic language detection also
routes the answer to the matching installed TTS voice. Override either
threshold when tuning for a microphone or environment:

```bash
ASR_NO_SPEECH_THRESHOLD=0.6 ASR_LOGPROB_THRESHOLD=-1.0 ./run.sh
```

If the deployed Whisper artifact does not provide `avg_logprob`, the Studio
uses `no_speech_prob` alone.

Check TTS through the Flask app. `language` selects the engine via the router
(English and Japanese below → piper-plus/piper-tts according to the selected
engine; dedicated piper-tts voices are the default):

```bash
# English
curl -k -s https://127.0.0.1:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper-tts","input":"Hello from Neat GenAI Studio.","language":"en"}' \
  --output /tmp/neat-genai-studio-tts-en.wav

# Japanese (dedicated piper-tts by default)
curl -k -s https://127.0.0.1:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper-tts","input":"こんにちは。Neat GenAI Studio です。","language":"ja"}' \
  --output /tmp/neat-genai-studio-tts-ja.wav
```

Then test the browser UI:

- send a text prompt and confirm the reply renders Markdown (try asking for a
  table or a code block, then use the code copy button)
- in **Settings → Models**, press **Load** on a different downloaded model and
  confirm it loads (watch the pinned Load-status bar) and answers without restarting
- if online, open **Settings → Add Model**, download a compatible model, and load it
- open the **Benchmark** (header speedometer), select two or more models, run it,
  and export the comparison as CSV/JSON
- under Appearance, change the font family/size and reload to confirm it persists
- enable `Include image in the prompt` and send an image prompt
- record audio and confirm transcription appears
- pick a **Transcription language** (e.g. Japanese) and confirm spoken replies
  play in that language; select a piper-tts voice and confirm playback
- change the system prompt, press abort during generation
- enable `Search RAG Database` and ask a question from `src/common/rag/neat.md`

## Source Files
- Run wrapper: `run.sh`
- Model hosting + control API: `src/python/server/main.py`, `src/python/server/model_manager.py`, `src/python/server/control_api.py`, `src/python/server/hub.py`
- UI: `src/python/ui/main.py`, `src/python/ui/flask_app.py`
- Terminal chat (CLI): `src/python/cli/main.py`
- TTS engines: `src/python/ui/piperplus_tts.py` (Piper Plus, main venv), `src/python/ui/pipertts.py` + `src/python/ui/pipertts_worker.py` (piper-tts, isolated venv)
- Voice/model install and policy: `src/python/voice_install.sh`, `src/python/ui/voice_catalog.py`, `src/python/ui/voice_catalog.json`, `THIRD_PARTY_TTS_MODELS.md`
- Shared config: `src/python/shared/config.py`, `src/common/config.yaml`
- Python dependencies: `src/python/requirements.txt` (main venv, Piper Plus), `src/python/requirements-pipertts.txt` (isolated piper-tts venv), `src/python/requirements-rag.txt`
- RAG helper: `src/python/rag/create_db.py`, `src/python/rag/vectordb.py`, `src/python/rag/vectordb_worker.py`
- RAG sample document: `src/common/rag/neat.md`
- UI assets: `src/python/ui/templates/`, `src/python/ui/static/` (including `static/vendor/` and `static/fonts/`), `src/python/ui/assets/`, `src/python/ui/certs/`
- SiMaSentry Solutions harnesses (vendored from `apps-llima-harnesses`): `src/python/ui/harnesses/`
- Manual API scripts: `src/python/ui/apitest/`
- Test scope: `tests/test-scope.yaml`

## Development From Source
See the Apps repository [contributor guide](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md)
for contribution requirements. The single-example download contains the Studio
source and can be edited directly; cloning the complete Apps repository is not
required to run or customize it.
