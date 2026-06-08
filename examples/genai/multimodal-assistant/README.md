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

## Preview
Multimodal assistant web UI:

![Multimodal Assistant preview](../../../assets/portal/genai/multimodal-assistant/image.png)

## Prerequisites
- Installed Neat runtime with GenAI and `pyneat` support.
- A VLM or LLM model directory configured in `common/config.yaml`.
- A Whisper ASR model directory configured in `common/config.yaml`.
- Python packages from `python/requirements.txt`.

Model directories must contain `devkit/` and `elf_files/`. Chat models use `devkit/vlm_config.json`; ASR models use `devkit/whisper_config.json`.

## Run
From the `apps` repository root:

```bash
python3 examples/genai/multimodal-assistant/python/main.py \
  --config examples/genai/multimodal-assistant/common/config.yaml
```

The supported phase-1 entrypoint is `python/main.py`. The imported sandbox scripts under `python/` are kept as source provenance during migration.

## Source Files
- Python source: `python/main.py`, `python/app.py`, `python/app_config.py`, `python/pipertts.py`
- Web UI assets: `python/templates/`, `python/static/`, `python/assets/`, `python/certs/`
- Shared config: `common/config.yaml`
