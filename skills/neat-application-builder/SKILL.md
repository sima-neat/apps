---
name: neat-application-builder
description: Build end-to-end Neat applications with public C++ or Python APIs. Use when a request starts from ONNX or a compiled model and targets a DevKit, including Model, Graph, GenAIModel, GenAIServer, camera, file, or RTSP pipelines. Choose the API shape before consulting Apps examples. Model compilation and repository workflows use separate skills.
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

## DevKit Local Display and Run

For applications that show results locally on a Modalix DevKit, or that run interactively until a human stops them:

- Choose a local display path. The DevKit runs an X11 desktop, but no GStreamer X-window video sink is installed (`ximagesink`, `xvimagesink`, and `glimagesink` are absent), so a GStreamer pipeline cannot render into a desktop window directly. Use one of:
  - HDMI full-screen via `kmssink`. The desktop holds the DRM master, so stop it first (`sudo systemctl stop lightdm`) and restart it after (`sudo systemctl start lightdm`). The DevKit DRM driver is `smifb`: it needs `driver-name=smifb`, an RGB (`BGRx`) caps, and a full-screen primary-plane modeset (`force-modesetting=true connector-id=<id>`); it rejects sub-CRTC overlay planes. Symptoms: "Could not open DRM module" means the driver name is missing; `drmModeSetPlane failed: Invalid argument` means a non-full-screen or overlay-plane config.
  - A window on the existing desktop via OpenCV `cv::imshow` (the installed OpenCV `highgui` uses the Qt backend), with no monitor takeover and no extra packages. Launch with `DISPLAY=:0` and the session's `XAUTHORITY`. Detect the window close ("X") button with `getWindowProperty(name, WND_PROP_VISIBLE) < 1`, otherwise `imshow` re-creates the closed window on the next call. `cv::waitKey` reads X events from the window, so ESC/q/close only work at the DevKit's own keyboard/mouse with the window focused.
  - Stream to Neat Insight with `VideoSender` (H.264 RTP/UDP) plus `MetadataSender` (detection JSON) and view the overlay in the browser — no local display.
- Run and stop over `dk`/`devkit-run`. These run the application over SSH without a pty, so a terminal Ctrl-C is not forwarded and the application can be left orphaned on the DevKit (still holding the MLA and streaming). For interactive runs, launch with `ssh -tt` so Ctrl-C and disconnects reach the application, have the application handle `SIGINT`/`SIGTERM`/`SIGHUP` for a clean `Run::close()` and teardown, and as a backup `trap` on exit to `pkill` the remote process. `dk`/`devkit-run` stays the right tool for short, bounded smoke runs that exit on their own.

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
