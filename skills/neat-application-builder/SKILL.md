---
name: neat-application-builder
description: Build Neat applications with public C++ or Python APIs in the Neat Development Environment or on DevKit. Use for classic apps that start from ONNX or compiled model archives, GenAI apps from deployed model directories, and camera, file, or RTSP pipelines. Choose Model, Graph, GenAIModel, or GenAIServer before consulting Apps examples. Model compilation and repository workflows use separate skills.
---

# Neat Application Builder

## Overview

Build applications against the installed Neat Library. Treat the current Neat
Development Environment's packaged core source, installed headers, and local
documentation as the source of truth. Use apps examples only as optional
reference implementations after the API shape is chosen.

## Workflow

1. Establish the environment and source of truth.
   - In the Neat Development Environment, read `/neat-resources/core-src` first.
   - Prefer installed public headers under the Neat Development Environment sysroot when checking the user-facing contract.
   - Read `references/source-of-truth.md`.
2. Choose the application API shape before opening an Apps example.
   - Read `references/api-decision-map.md`.
   - Continue after the runtime artifact, input owner, and API family are known.
3. If the request touches APIs outside the main Model/Graph/Run/GenAI path, read `references/api-surface-map.md` and inspect the referenced headers/docs.
4. For classic compiled model applications, read `references/model-graph-run.md`.
5. For LLM, VLM, ASR, or HTTP model serving applications, read `references/genai.md`.
6. Before claiming success, read `references/validation.md` and run the validation that is possible in the current environment.

## Defaults

- C++ applications should start with `#include <neat.h>` unless a narrower public include is clearly better.
- Python applications should use installed `pyneat` and `pyneat.genai`.
- Use only public APIs from installed headers and bindings.
- Prefer clear application endpoint names such as `image`, `detections`, `classes`, `preview`, `prompt`, and `tokens`.
- Keep generated application code runnable with explicit build and run commands.

## DevKit Display and Run

For applications that show results locally on a Modalix DevKit, or that run interactively until a human stops them:

- Inspect the target before choosing a local display path. Check installed GStreamer sinks, the active desktop session, the DRM owner and connector, and the OpenCV HighGUI backend instead of assuming one image's package or driver layout.
- Use Neat Insight when the result should be viewed remotely. `VideoSender` can encode compatible raw input as H.264 or forward an already encoded H.264 or H.265 stream with the matching codec; pair it with `MetadataSender` when the viewer needs structured overlays.
- Prefer `dk`/`devkit-run` for SDK-to-DevKit execution. The current helper streams output and attempts remote cleanup after interruption or common SSH and signal exits. Use `dk shell` when an interactive remote terminal is required, and fall back to direct SSH only when the helper cannot express the workflow.
- Have long-running applications handle `SIGINT` and `SIGTERM`, close their `Run` handles, and release model, codec, and streaming resources on every exit path.

## Boundaries

- Do not describe this as a repository maintenance skill.
- Do not add repository publication, release automation, review workflow, or contributor-process guidance.
- Use only public Neat Library APIs, packaged source, installed headers, official docs, and public examples.
- Do not guess API behavior from memory. Verify against current packaged core source or installed docs.

## References

- `references/source-of-truth.md`
- `references/api-decision-map.md`
- `references/api-surface-map.md`
- `references/model-graph-run.md`
- `references/genai.md`
- `references/validation.md`
