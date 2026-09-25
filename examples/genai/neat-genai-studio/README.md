# Neat GenAI Studio

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Advanced |
| Tags | genai, vlm, asr, tts, japanese, multilingual, rag, model-switching, huggingface, markdown, openai-compatible |
| Languages | Python |
| Status | stable |
| Binary Name | neat-genai-studio |
| Model | Loaded on demand (e.g. Qwen3-VL-4B-Instruct-GPTQ-a16w4) + whisper-small-a16w8 |

## Concept
Runs supported language and vision-language models on a Modalix device through a web interface for chat, image analysis, model setup, and diagnostics.

The Studio starts with no chat model loaded. From the web interface you can:

- find compatible models already on the device and load one language or vision-language model at a time without a restart
- download models from the supported Hugging Face accounts when the device is online
- chat with text, uploaded images, a browser camera, or a camera attached to the board
- transcribe speech with Whisper, speak replies with the installed voices, and search local documents with RAG

The interface and its fonts and JavaScript libraries run locally. Internet access is needed only when you search for or download a model from Hugging Face.

## Preview
![Neat GenAI Studio preview](../../../portal/assets/examples/genai/neat-genai-studio/image.png)

## Prerequisites
- `sima-cli` ([documentation](https://developer.sima.ai/software/tools/sima-cli/)) on a supported Modalix or DevKit target.
- Installed Neat Development Environment and Neat Library. The model server runs in the Python environment where `pyneat` is available; the scripts default to `~/pyneat/bin/python`. Set `PYNEAT_PYTHON=/path/to/python-with-pyneat` if it lives elsewhere.
- Internet access on the target only when you search for or download a model from Hugging Face.

## Install Apps
Install the latest Neat Apps runtime and enter the installed bundle:

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/genai/neat-genai-studio
```

Run the remaining commands from `prebuilt-apps/`.

## Prepare the Model
Neat GenAI Studio installs its models and dependencies with its own setup script, which downloads from Hugging Face. There is no `sima-cli` model-download step for this application.

```bash
${APP_DIR}/setup.sh
```

`setup.sh` creates the UI virtual environments, downloads the default speech and embedding models, installs the text-to-speech voices and the Supertonic engine, builds the default RAG database, and writes a generated `config.local.yaml`.

| Model | Role | Source |
| --- | --- | --- |
| `whisper-small-a16w8` | Default speech-to-text, installed by `setup.sh` | Hugging Face `simaai/whisper-small-a16w8` |
| `gte-small` | RAG embedding model, installed by `setup.sh` | Hugging Face `thenlper/gte-small` |
| `Qwen3-VL-4B-Instruct-GPTQ-a16w4` | Example chat / vision-language model, loaded on demand | Hugging Face (`simaai`, `TDoSiMa`) |

No chat or vision-language model is installed by default; load one from the web interface after startup, or fetch and preload one during setup:

```bash
CHAT_MODEL_REPO=simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4 ${APP_DIR}/setup.sh
```

Models are stored under `/media/nvme/llima/models` by default. Set `LLIMA_MODELS_PATH=/path/to/models` before `setup.sh` to use another directory.

## Configure
`setup.sh` writes `config.local.yaml`; the packaged defaults are in `${APP_DIR}/src/common/config.yaml`. Any compatible model directory under the catalog is discovered and loadable from the web interface without a restart, so most users change nothing. Edit `config.local.yaml` only to change:

- `server.models.catalog_dir` — the directory scanned for loadable models (default `/media/nvme/llima/models`).
- `server.models.chat` — chat or vision-language models to preload at startup (empty by default; models still load on demand).
- `server.models.asr` — the speech-to-text model active at startup (`whisper-small-a16w8` by default; switchable at runtime).
- `server.hub.allow_download` — whether the interface may download models from Hugging Face.
- `app.rag.enabled` and `app.rag.embedding_model_dir` — retrieval-augmented search and its embedding model.

## Run
Start the Studio. The launcher runs `setup.sh` automatically on first launch if it has not been run yet.

```bash
APP_DIR=examples/genai/neat-genai-studio
${APP_DIR}/run.sh
```

When it finishes starting, open the printed URL — `https://<target-ip>:5000` — in a browser. Load a model from **Settings → Models**, then chat.

Use the terminal chat instead of the web interface:

```bash
${APP_DIR}/run.sh --cli
```

Stop the Studio with `Ctrl+C`, or from another shell:

```bash
${APP_DIR}/run.sh stop
${APP_DIR}/run.sh status
```

To start the processes manually, run the packaged entrypoints under `${APP_DIR}/src/python/` in separate terminals — the OpenAI-compatible model server and the web interface:

```bash
${PYNEAT_PYTHON:-~/pyneat/bin/python} ${APP_DIR}/src/python/server/main.py --config ${APP_DIR}/config.local.yaml
${APP_DIR}/.venv/bin/python ${APP_DIR}/src/python/ui/main.py --config ${APP_DIR}/config.local.yaml
```

## Expected Result
The web interface loads at `https://<target-ip>:5000`. After loading a model from **Settings → Models**, a chat prompt returns a streamed reply. From a shell on the target you can confirm the model server and its loaded models:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

Speech transcription (Whisper), spoken replies (text-to-speech), image prompts, the board or browser camera, and RAG document search are available from the interface when their models and voices are installed.

## Troubleshooting
- If the model server does not start, confirm `pyneat` is importable in the environment at `PYNEAT_PYTHON` (default `~/pyneat/bin/python`).
- To download models from a gated Hugging Face repository, set `HF_TOKEN` before running `setup.sh` or the interface.
- The web interface uses HTTPS on port `5000`; the model server uses `127.0.0.1:9998` and its control API uses `127.0.0.1:9997`.
- Reinstall the environment and regenerated config with `${APP_DIR}/run.sh --clean` followed by `${APP_DIR}/setup.sh`.

## Source Files
- Launcher and setup: `run.sh`, `setup.sh`
- Model server: `src/python/server/`
- Web interface: `src/python/ui/`
- Terminal chat: `src/python/cli/`
- Shared configuration and runtime helpers: `src/common/`, `src/python/shared/`
- RAG database tools and default corpus: `src/python/rag/`, `src/common/rag/`
- Text-to-speech model attribution: `THIRD_PARTY_TTS_MODELS.md`

## Development From Source
To modify or test this example, use the [Apps contributor workflow](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
