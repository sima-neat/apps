---
name: neat-application-builder
description: Build Neat applications with public C++ or Python APIs in the Neat Development Environment or on DevKit. Use for classic apps that start from ONNX or compiled model archives, GenAI apps from deployed model directories, and camera, file, or RTSP pipelines. Choose Model, Graph, GenAIModel, or GenAIServer, then inspect the closest Apps example before implementation. Model compilation and repository workflows use separate skills.
---

# Neat Application Builder

## Overview

Build applications against the installed Neat Library. Treat the current Neat
Development Environment's packaged core source, installed headers, and local
documentation as the API source of truth. Once the broad API family is known,
use the closest current Apps example as the implementation starting point.

## Workflow

1. Establish the environment and source of truth.
   - In the Neat Development Environment, read `/neat-resources/core-src` first.
   - Prefer installed public headers under the Neat Development Environment sysroot when checking the user-facing contract.
   - Read `references/source-of-truth.md`.
2. Determine the artifact, input owner, and broad API family.
   - Read `references/api-decision-map.md`.
   - Continue once all three are known.
3. Before writing application code, inspect the closest current example under `/neat-resources/apps-src/examples`.
   - Use it for project structure, configuration, graph composition, build commands, and runtime patterns.
   - Verify API details against the installed Core headers and docs.
4. If the request touches APIs outside the main Model/Graph/Run/GenAI path, read `references/api-surface-map.md` and inspect the referenced headers/docs.
5. For classic compiled model applications, read `references/model-graph-run.md`.
6. For LLM, VLM, ASR, or HTTP model serving applications, read `references/genai.md`.
7. Before claiming success, read `references/validation.md` and run the validation that is possible in the current environment.

## Defaults

- C++ applications should start with `#include <neat.h>` unless a narrower public include is clearly better.
- Python applications should use installed `pyneat` and `pyneat.genai`.
- Use only public APIs from installed headers and bindings.
- Prefer clear application endpoint names such as `image`, `detections`, `classes`, `preview`, `prompt`, and `tokens`.
- Keep generated application code runnable with explicit build and run commands.

## Neat Insight output

Prefer Neat Insight when results should be viewed remotely in a browser. This
keeps the DevKit desktop and display hardware available.

- Add the public `VideoSender` graph fragment for video. Use
  `VideoSenderOptions::H264RtpUdpFromRaw(...)` for raw frames or
  `VideoSenderOptions::Passthrough(codec)` for encoded H.264 or H.265.
- Pair it with `MetadataSender` when Insight needs JSON detections or other
  structured overlays.
- Inspect `include/nodes/groups/VideoSender.h` and
  `include/nodes/io/MetadataSender.h`. Read the packaged Core docs
  `advanced-concepts/application-design/video_sender.md` and
  `advanced-concepts/application-design/metadata_sender.md` before choosing
  ports, channels, codecs, or graph links.
- Use the Insight documentation for viewer setup and operations. This skill
  owns the application-side APIs only.

## DevKit local display and run

For applications that show results locally on a Modalix DevKit, or that run interactively until a human stops them:

- Inspect the target before choosing a local display path. Check the installed GStreamer sinks, active desktop session, DRM owner and connector, and OpenCV HighGUI backend.
- On the validated DevKit image, the X11 desktop is present but the GStreamer X-window sinks `ximagesink`, `xvimagesink`, and `glimagesink` are absent. Use one of these local paths:
  - For full-screen HDMI, use `kmssink`. Stop the desktop first with `sudo systemctl stop lightdm`, then restart it with `sudo systemctl start lightdm`. The `smifb` DRM driver needs `driver-name=smifb`, `BGRx` caps, `force-modesetting=true`, and `connector-id=<id>`. It accepts a full-screen primary-plane modeset, not a sub-CRTC overlay plane. `Could not open DRM module` means the driver name is missing. `drmModeSetPlane failed: Invalid argument` means the plane or dimensions are unsupported.
  - For a window on the existing desktop, use OpenCV `cv::imshow`. Launch with `DISPLAY=:0` and the desktop session's `XAUTHORITY`. Detect the close button with `getWindowProperty(name, WND_PROP_VISIBLE) < 1`; otherwise the next `imshow` call recreates the window. `cv::waitKey` reads the window's X events, so ESC, q, and close work only from the DevKit keyboard or mouse while the window has focus.
- Prefer `dk` or `devkit-run` for SDK-to-DevKit execution. The current helper streams output and cleans up the remote process after interruption and common SSH or signal exits. Use `dk shell` when the workflow needs an interactive PTY. Older helpers that lack this cleanup can orphan the application; use `ssh -tt` and an exit trap that stops the remote process in that environment.
- Have long-running applications handle `SIGINT`, `SIGTERM`, and `SIGHUP`, close their `Run` handles, and release model, codec, display, and streaming resources on every exit path.

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
