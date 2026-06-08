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
| Model | LLiMa VLM/LLM and Whisper ASR model directories |

## Concept
This example hosts SiMa-supported GenAI models through Neat's OpenAI-compatible server and uses the imported Flask demo UI as the interactive assistant surface.

The phase-1 target is one configured chat model, one configured ASR model, image/text chat, audio transcription, Piper TTS, system prompt control, chat history, voice selection, and abort.

`python/main.py` is the apps-native entrypoint. It reads `common/config.yaml`,
starts `pyneat.OpenAIServer`, registers the configured chat/VLM model and ASR
model, then starts the existing Flask UI.

Runtime ownership is split deliberately:

- Neat hosts the OpenAI-compatible `/v1/chat/completions` endpoint for text and
  image chat.
- Neat hosts the OpenAI-compatible `/v1/audio/transcriptions` endpoint for ASR.
- The Flask app owns UI state, system prompt handling, chat history, abort
  requests, uploaded images, microphone recordings, and Piper TTS playback.
- Piper TTS is app-side and part of phase 1. The default requirements install
  Piper; TTS produces audio when the configured voice assets are present under
  `python/assets/`.
- RAG source is preserved from the sandbox import, but RAG is disabled by
  default in phase 1 through `common/config.yaml`.

## Preview
Multimodal assistant web UI:

![Multimodal Assistant preview](../../../assets/portal/genai/multimodal-assistant/image.png)

## Prerequisites
- Installed Neat runtime with GenAI and `pyneat` support.
- A VLM or LLM model directory configured in `common/config.yaml`.
- A Whisper ASR model directory configured in `common/config.yaml`.
- Phase-1 Python packages from `python/requirements.txt`.

Model directories must contain `devkit/` and `elf_files/`. Chat models use `devkit/vlm_config.json`; ASR models use `devkit/whisper_config.json`.

The default `common/config.yaml` expects:

```text
assets/models/genai/gemma4-E2B-it
assets/models/genai/whisper-small-a16w8
```

Run from the Python environment created by the installed Neat runtime, then
install the app packages into that environment:

```bash
source ~/pyneat/bin/activate
python3 -m pip install -r examples/genai/multimodal-assistant/python/requirements.txt
```

## Run
From the `apps` repository root:

```bash
python3 examples/genai/multimodal-assistant/python/main.py \
  --config examples/genai/multimodal-assistant/common/config.yaml
```

The supported phase-1 entrypoint is `python/main.py`. The imported sandbox scripts under `python/` are kept as source provenance during migration.

By default, the Flask UI preserves the sandbox HTTPS behavior and listens on the
configured web port:

```text
https://<host>:5000
```

The Neat OpenAI-compatible server listens on the configured API port:

```text
http://127.0.0.1:9998
```

To start only the Neat OpenAI-compatible server and skip the Flask UI:

```bash
python3 examples/genai/multimodal-assistant/python/main.py \
  --config examples/genai/multimodal-assistant/common/config.yaml \
  --server-only
```

## Configuration
`common/config.yaml` controls the served model names, model directories, request
defaults, Flask binding, HTTPS behavior, and RAG gate.

```yaml
openai:
  client_host: 127.0.0.1
  port: 9998

models:
  chat:
    name: gemma4
    path: assets/models/genai/gemma4-E2B-it
  asr:
    name: whisper-small
    path: assets/models/genai/whisper-small-a16w8

request:
  max_tokens: 128
  system_prompt: Answer clearly and concisely.

web:
  port: 5000
  https: true

rag:
  enabled: false
```

The chat and ASR requests use the configured `models.*.name` values as their
OpenAI `model` fields. The paths are resolved relative to the apps repository
root unless they are absolute paths.

## Source Files
- Python source: `python/main.py`, `python/app.py`, `python/app_config.py`, `python/pipertts.py`
- Python dependencies: `python/requirements.txt`
- Web UI assets: `python/templates/`, `python/static/`, `python/assets/`, `python/certs/`
- Shared config: `common/config.yaml`
- Test scope: `test-scope.yaml`
