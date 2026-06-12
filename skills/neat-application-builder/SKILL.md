---
name: neat-application-builder
description: Use when building Neat Library applications with public C++ or Python APIs, including choosing between Model, Graph, GenAIModel, and GenAIServer; using Run handles returned by built graphs; reading packaged core headers/docs; composing application pipelines; and validating Neat Development Environment or DevKit behavior. Do not use for repository maintenance, release automation, review workflows, or work outside the public Neat Library application surface.
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
2. Choose the application API shape before writing code.
   - Read `references/api-decision-map.md`.
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

## API Selection

- Use `Model` for a single classic compiled model archive and direct request/response inference.
- Use `Graph` when the application needs multiple stages, named inputs or outputs, branching, fan-in, reusable fragments, or source/output nodes.
- Use `GenAIModel` for in-process GenAI calls against LLiMa model directories.
- Use `GenAIServer` when a browser, service, or remote client should call GenAI models over HTTP.

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
