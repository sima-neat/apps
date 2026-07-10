# Neat GenAI Studio

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | genai, vlm, asr, tts, rag, model-switching, huggingface, markdown, openai-compatible |
| Languages | Python |
| Status | experimental |
| Binary Name | neat-genai-studio |
| Model | Loaded on demand (e.g. Qwen3-VL-4B-Instruct-GPTQ-a16w4) + whisper-small-a16w8 |

## Concept
Neat GenAI Studio is an improved, more immersive version of the multimodal assistant
runtime demo. It hosts SiMa-supported GenAI models through Neat's
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
- Everything the multimodal assistant already offered: image/text chat, audio
  transcription (ASR), Piper TTS, RAG, system-prompt control, chat history,
  voice selection, and abort.

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

![Neat GenAI Studio preview](../../../assets/portal/genai/neat-genai-studio/image.png)

## Prerequisites
- Installed Neat Development Environment + Neat Library.
- The model server uses the Python environment where `pyneat` is available. The default scripts assume:

```text
~/pyneat/bin/python
```

Set `PYNEAT_PYTHON=/path/to/python-with-pyneat` if your Neat Library environment is
somewhere else.

## Get The Apps Repo
Use the [Neat Development Environment](https://developer.sima.ai/software/getting-started/dev-environment/) with the [Neat Library](https://developer.sima.ai/software/getting-started/neat-library/) installed for setup and compilation.

Clone and build the apps repo inside the Neat Development Environment:

```bash
git clone https://github.com/sima-neat/apps.git
cd apps
./build.sh --clean
```

After building, run the example commands below on the Modalix/DevKit board.

## Install
Fetch only this example:

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- neat-genai-studio
```

Install the UI virtual environment, Whisper ASR model, GTE-small embedding
model, Piper TTS voices, default RAG database, and generated local config:

```bash
cd neat-genai-studio

./setup.sh
```

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
./run.sh
```

### Stopping
Press `Ctrl+C` in the terminal running `run.sh`, or from another shell:

```bash
./run.sh stop      # cleanly stop a running instance
./run.sh status    # report whether it is running
```

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
as root. By default it uses the SDK's `fix_devkit_runtime.sh` if present, else
restarts `simaai-appcomplex.service` and re-runs `init_mla_memory.sh`. Overrides:

- `MLA_RESET_CMD="my-reset-tool"` — run your own reset command instead.
- `MLA_DISPATCHER_SERVICE=<unit>` — use a different dispatcher service name.
- `MLA_RESET=0` — skip the MLA reset entirely.

```bash
# example: force a specific reset command
MLA_RESET_CMD="sudo fix_devkit_runtime.sh" ./run.sh
```

If you hit `MLA_LOAD_FAILED` right now, reset the runtime once by hand:

```bash
sudo fix_devkit_runtime.sh        # SDK recovery script (preferred)
# or, minimally:
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
In the UI settings panel, the **Active model** dropdown lists every model in the
catalog. Loaded models are marked with `●`; picking a not-yet-loaded model loads
it at runtime and unloads all other chat/VLM models (whisper is always kept), so
the MLA holds just the active model. The studio cancels the outgoing model's
in-flight generation and waits for its memory to be released before loading the
new one, then warms it so your first message is instant.

If a switch still hits a busy/wedged accelerator, the studio automatically
requests a **supervised MLA reset**: the model server exits with a sentinel code,
`run.sh` resets the MLA dispatcher and relaunches it (the UI stays up and
reconnects). Disable this with `MLA_RESET_ON_SWITCH=0` (then a failed switch just
reports an error asking you to restart).

### Download models from Hugging Face
When the board is online, the **Add model from Hugging Face** section appears in
the settings panel. Search the `simaai` model catalog, click **Download**, watch
the progress bar, and the new model appears in the Active model dropdown ready to
load. Downloads land under `catalog_dir`. Set `HF_TOKEN` for gated repos.

### Markdown & fonts
Assistant replies render Markdown live. Under **Appearance**, pick a font family
and size or type any locally installed family; the choice is saved in the
browser. The dark/light theme toggle is in the Settings header.

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
  -F "file=@${AUDIO_FILE}"
```

Check Piper TTS through the Flask app:

```bash
curl -k -s https://127.0.0.1:5000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"piper-tts","input":"Hello from Neat GenAI Studio."}' \
  --output /tmp/neat-genai-studio-tts.wav
```

Then test the browser UI:

- send a text prompt and confirm the reply renders Markdown (try asking for a
  table or a code block, then use the code copy button)
- open the Active model dropdown, pick a different catalog model, and confirm it
  loads and answers without restarting
- if online, search Hugging Face, download a compatible model, and load it
- under Appearance, change the font family/size and reload to confirm it persists
- enable `Include image in the prompt` and send an image prompt
- record audio and confirm transcription appears
- select a Piper voice and confirm playback
- change the system prompt, press abort during generation
- enable `Search RAG Database` and ask a question from `src/common/rag/neat.md`

## Source Files
- Run wrapper: `run.sh`
- Model hosting + control API: `src/python/server/main.py`, `src/python/server/model_manager.py`, `src/python/server/control_api.py`, `src/python/server/hub.py`
- UI: `src/python/ui/main.py`, `src/python/ui/flask_app.py`, `src/python/ui/pipertts.py`
- Shared config: `src/python/shared/config.py`, `src/common/config.yaml`
- Python dependencies: `src/python/requirements.txt`, `src/python/requirements-rag.txt`
- RAG helper: `src/python/rag/create_db.py`, `src/python/rag/vectordb.py`, `src/python/rag/vectordb_worker.py`
- RAG sample document: `src/common/rag/neat.md`
- UI assets: `src/python/ui/templates/`, `src/python/ui/static/` (including `static/vendor/` and `static/fonts/`), `src/python/ui/assets/`, `src/python/ui/certs/`
- Manual API scripts: `src/python/ui/apitest/`
- Test scope: `tests/test-scope.yaml`
```
