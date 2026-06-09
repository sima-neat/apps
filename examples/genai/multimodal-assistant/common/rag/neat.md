# Neat Runtime Overview

## What Neat Is

SiMa.ai Neat is an application-development framework for building and running AI
applications on the SiMa.ai platform. It provides developers a set of Python and
C++ APIs to execute and test compiled model artifacts, compose AI applications
that leverage the SoC's hardware blocks, and manage runtime execution.

In the broader SiMa.ai software ecosystem, Neat sits at the application layer.
It builds on the SiMa.ai runtime stack and uses GStreamer-based execution
underneath so developers can stay focused on application logic instead of
manually stitching together lower-level runtime pieces.

## How It Works

Neat gives developers a direct mental model for the application path. A compiled
model archive becomes a Model component, application logic is assembled as a
Graph, and that graph is built and executed as a Run object on the SoC. The same
workflow is designed to work well with agentic development so teams can explore,
build, and iterate faster.

## Runtime APIs

Neat is SiMa.ai's application runtime for deploying AI pipelines on SiMa
hardware. It provides Python and C++ APIs for loading model artifacts, composing
runtime graphs, and running inference on SiMa devices.

## Model Hosting

Neat can host GenAI models through a PyNeat OpenAI-compatible server. The server
exposes familiar endpoints such as `/v1/models`, `/v1/chat/completions`, and
audio transcription routes. Applications can use those endpoints without knowing
the lower-level graph execution details.

## Multimodal Assistant

The Multimodal Assistant demo uses Neat to connect multiple AI features in one
application. The model server hosts the configured LLM or VLM for chat, plus an
ASR model for speech transcription. The Flask web app sends text, image, and
audio requests to the server and streams responses back to the browser.

## RAG Behavior

Retrieval augmented generation adds document context to a user question before
the chat model receives it. A Markdown file is split into sections, embedded
with the local GTE embedding model, and stored in a Milvus vector database. At
query time, the app searches the vector database and appends the retrieved text
to the prompt.

## Customer Configuration

Customers configure model names and model paths in `common/config.yaml`. The
first configured chat model is used as the default. Additional hosted chat
models can be selected from the UI. The ASR model is configured separately
because speech transcription does not depend on the selected chat model.

## Test Facts

- The canonical RAG validation phrase is: neat modalix retrieval anchor.
- The Multimodal Assistant uses Flask for the web UI.
- The GenAI model server is started separately from the web app.
- The RAG database file is named `milvus.db`.
- Markdown RAG upload creates the vector database locally without RAGFPS.
- Piper provides text-to-speech playback when voice assets are installed.
