# Multimodal Assistant

## Metadata
| Field | Value |
| --- | --- |
| Category | genai |
| Difficulty | Intermediate |
| Tags | genai, vlm, asr, tts, openai |
| Languages | Python |
| Status | experimental |
| Binary Name | multimodal-assistant |
| Model | LLM/VLM and Whisper ASR model directories |

## Concept
This example hosts SiMa-supported GenAI models through Neat's OpenAI-compatible server and uses the imported Flask demo UI as the interactive assistant surface.

The example supports one or more configured chat models, one configured ASR model, image/text chat, audio transcription, Piper TTS, system prompt control, chat history, voice selection, and abort.

The demo runs as two processes. `src/python/server/main.py` reads the `server`
section of `src/common/config.yaml` and starts the Neat OpenAI-compatible server.
`src/python/ui/main.py` reads the `app` section of `src/common/config.yaml` and starts
the existing Flask UI.

Runtime ownership is split deliberately:

- Neat hosts the OpenAI-compatible `/v1/chat/completions` endpoint for text and
  image chat.
- Neat hosts the OpenAI-compatible `/v1/audio/transcriptions` endpoint for ASR.
- The Flask app owns UI state, system prompt handling, chat history, abort
  requests, uploaded images, microphone recordings, and Piper TTS playback.
- Piper TTS is app-side. The default requirements install Piper; TTS produces
  audio when the configured voice assets are present under `src/python/ui/assets/`.
- RAG is app-side and optional. It uses the Flask UI, local VectorDB service,
  and the same chat model hosted by the Neat OpenAI-compatible server.

## Preview
Multimodal assistant UI:

![Multimodal Assistant preview](../../../assets/portal/genai/multimodal-assistant/image.png)

## Prerequisites
Run the commands from the `apps` repository root on the target system.

### 1. Install Neat
Install Neat before running the demo. The model server must use a Python
environment where `pyneat` is available. The examples below assume:

```text
~/pyneat/bin/python
```

### 2. Download Model Artifacts
Visit the [SiMa.ai Hugging Face page](https://huggingface.co/simaai) to see the
officially supported model artifacts. Download the chat/VLM and ASR artifacts
you want to run, then update `src/common/config.yaml` so each model path points to
the downloaded directory. Model paths can be absolute or relative to the `apps`
repository root.

```bash
python3 -m pip install -U "huggingface_hub[cli]"

MODEL_REPO="<model-repo>"
MODEL_DIR="/path/to/local/model-dir"
hf download "simaai/${MODEL_REPO}" --local-dir "${MODEL_DIR}"
```

Model directories must contain `devkit/` and `elf_files/`. Chat models use
`devkit/vlm_config.json`; ASR models use `devkit/whisper_config.json`.

### 3. Configure Models
Edit `examples/genai/multimodal-assistant/src/common/config.yaml`:

```yaml
server:
  openai:
    host: 0.0.0.0
    port: 9998

  models:
    chat:
      - name: <chat-model-name>
        path: <path-to-llm-or-vlm-model>
      - name: <another-chat-model-name>
        path: <path-to-another-llm-or-vlm-model>
    asr:
      name: <asr-model-name>
      path: <path-to-asr-model>
```

The first chat model in the list is selected by default. Additional chat models
appear in the UI model selector.

### 4. Create The UI Environment
Keep the Flask UI dependencies separate from the `pyneat` environment:

```bash
python3 -m venv ~/multimodal-assistant-app
source ~/multimodal-assistant-app/bin/activate
python3 -m pip install -r examples/genai/multimodal-assistant/src/python/requirements.txt
```

### 5. Install Piper Voices For TTS
If you need text-to-speech playback, install the Piper voice assets:

```bash
cd examples/genai/multimodal-assistant/src/python
bash voice_install.sh
cd /workspace/sima-neat/apps
```

The script downloads selected Piper `.onnx` voice models into
`src/python/ui/assets/`.
Each voice needs both files:

```text
src/python/ui/assets/<voice>.onnx
src/python/ui/assets/<voice>.onnx.json
```

## Run
For the general demo, start both processes with the wrapper script:

```bash
cd /workspace/sima-neat/apps

APP_PYTHON=~/multimodal-assistant-app/bin/python \
PYNEAT_PYTHON=~/pyneat/bin/python \
bash examples/genai/multimodal-assistant/run.sh
```

`run.sh` is a convenience wrapper. It starts:

- `src/python/server/main.py` with `PYNEAT_PYTHON`
- `src/python/ui/main.py` with `APP_PYTHON`

Set `PYNEAT_PYTHON` to the Python interpreter that has `pyneat` installed. Set
`APP_PYTHON` to the UI environment that has `src/python/requirements.txt`
installed. If `PYNEAT_PYTHON` is not set, the script uses `~/pyneat/bin/python`
when it exists. If `APP_PYTHON` is not set, it uses `python3`.

Open the Flask UI:

```text
https://<target-ip>:5000
```

The Neat OpenAI-compatible server listens on:

```text
http://127.0.0.1:9998
```

Check that the configured models are hosted:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

Wait until this command lists the configured chat and ASR models before testing
the browser UI.

### Manual Process Start
Use this only when you want two explicit terminals.

Terminal 1, model server:

```bash
cd /workspace/sima-neat/apps
source ~/pyneat/bin/activate

python /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/python/server/main.py \
  --config /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/common/config.yaml
```

Terminal 2, Flask UI:

```bash
cd /workspace/sima-neat/apps
source ~/multimodal-assistant-app/bin/activate

python /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/python/ui/main.py \
  --config /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/common/config.yaml
```

The supported entrypoints are `src/python/server/main.py` for model hosting and
`src/python/ui/main.py` for the UI.

## Optional RAG Setup
RAG is optional. If you do not need RAG, skip this section.

### 1. Enable RAG
Edit `examples/genai/multimodal-assistant/src/common/config.yaml`:

```yaml
app:
  rag:
    enabled: true
    embedding_model_dir: <path-to-embedding-model-dir>
```

### 2. Install RAG Dependencies
Install the RAG packages into the UI environment:

```bash
cd /workspace/sima-neat/apps
source ~/multimodal-assistant-app/bin/activate

python3 -m pip install -r examples/genai/multimodal-assistant/src/python/requirements-rag.txt
```

### 3. Download The Embedding Model
Place the local embedding model at the path configured by
`app.rag.embedding_model_dir`. The path can be absolute or relative to the
`apps` repository root. Use the full Hugging Face repository id.

```bash
EMBED_REPO="<embedding-model-repo>"
EMBED_DIR="/path/to/local/embedding-model-dir"
hf download "${EMBED_REPO}" --local-dir "${EMBED_DIR}"
```

### 4. Create A RAG Database From Markdown
The example includes a small Markdown document for smoke testing:

```text
examples/genai/multimodal-assistant/src/common/rag/neat.md
```

Create `milvus.db`:

```bash
cd /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/python
source ~/multimodal-assistant-app/bin/activate

EMBED_DIR="/path/to/local/embedding-model-dir"
python rag/create_db.py \
  --input /workspace/sima-neat/apps/examples/genai/multimodal-assistant/src/common/rag/neat.md \
  --output ui/milvus.db \
  --embedding-model "${EMBED_DIR}"
```

Do not commit generated files:

```text
examples/genai/multimodal-assistant/src/python/ui/milvus.db
examples/genai/multimodal-assistant/src/python/ui/milvus.meta.json
```

### 5. Run And Test RAG
Start the demo normally. The Flask app starts the local VectorDB service when
RAG is enabled.

```bash
cd /workspace/sima-neat/apps

APP_PYTHON=~/multimodal-assistant-app/bin/python \
PYNEAT_PYTHON=~/pyneat/bin/python \
bash examples/genai/multimodal-assistant/run.sh
```

Check VectorDB directly:

```bash
curl -sG http://127.0.0.1:9100/search \
  --data-urlencode "query=What is the canonical RAG validation phrase?" \
  --data "k=3" \
  --data "min_score=-1" | python3 -m json.tool
```

In the UI, enable `Search RAG Database` and ask:

```text
What is the canonical RAG validation phrase?
```

The RAG status area should report whether RAG was used and how many hits were
retrieved.

## Verify
Use these checks after the model server and Flask UI are running.

Check hosted model names:

```bash
curl -s http://127.0.0.1:9998/v1/models | python3 -m json.tool
```

Check text chat:

```bash
CHAT_MODEL="<chat-model-name>"

curl -s http://127.0.0.1:9998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${CHAT_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Say hello.\"}]}],\"max_tokens\":32}"
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
  -d '{"model":"piper-tts","input":"Hello from the Multimodal Assistant."}' \
  --output /tmp/multimodal-assistant-tts.wav
```

If RAG is enabled, check VectorDB:

```bash
curl -sG http://127.0.0.1:9100/search \
  --data-urlencode "query=What is Neat?" \
  --data "k=3" \
  --data "min_score=-1" | python3 -m json.tool
```

Then test the browser UI:

- send a text prompt
- select another hosted chat model
- enable `Include image in the prompt` and send an image prompt
- record audio and confirm transcription appears
- select a Piper voice and confirm playback
- change the system prompt and send a new prompt
- press abort during generation
- enable `Search RAG Database` and ask a question from `src/common/rag/neat.md`

## Source Files
- Run wrapper: `run.sh`
- Python source: `src/python/server/main.py`, `src/python/ui/main.py`, `src/python/ui/flask_app.py`, `src/python/shared/config.py`, `src/python/ui/pipertts.py`
- Python dependencies: `src/python/requirements.txt`, `src/python/requirements-rag.txt`
- RAG helper: `src/python/rag/create_db.py`, `src/python/rag/vectordb.py`, `src/python/rag/vectordb_worker.py`
- RAG sample document: `src/common/rag/neat.md`
- UI assets: `src/python/ui/templates/`, `src/python/ui/static/`, `src/python/ui/assets/`, `src/python/ui/certs/`
- Manual API scripts: `src/python/ui/apitest/`
- Shared config: `src/common/config.yaml`
- Test scope: `tests/test-scope.yaml`
