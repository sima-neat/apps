---
name: neat-application-builder
description: Build or adapt Neat applications using public C++ or Python APIs. Use for application logic and pipelines consuming compiled model archives or deployed GenAI model directories.
---

# Neat Application Builder

Build against the installed Neat Library's public APIs. Installed headers and
bindings define the callable contract; matching packaged source and docs explain
behavior. Verify the APIs affected by the task against that installation.

## Choose the relevant guidance

- When locating the installation or a starting example, read
  [source of truth](references/source-of-truth.md).
- When choosing or changing the execution approach, read the
  [API decision map](references/api-decision-map.md). Identify the artifact,
  input owner, and API family before committing to a topology.
- For classic model execution or graph composition, read
  [Model, Graph, and Run](references/model-graph-run.md).
- For LLM, VLM, ASR, or HTTP model serving, read
  [GenAI APIs](references/genai.md).
- When locating an unfamiliar API, use the
  [API surface map](references/api-surface-map.md) to find the relevant header
  or binding. Inspect only the areas needed by the task.
- When adding or changing visualization, read
  [application output](references/application-output.md).

For a new application or execution approach, inspect the closest compatible Apps
example for structure, configuration, and runtime patterns. For a focused edit,
start with the existing application and consult other examples as needed.

## Implementation defaults

- C++ applications start with `#include <neat.h>` unless a narrower public
  include is appropriate. Python applications use installed `pyneat`, with
  `pyneat.genai` for GenAI APIs.
- Long-running applications handle `SIGINT`, `SIGTERM`, and `SIGHUP`, close
  their `Run` handles, and release model, codec, display, and streaming resources
  on exit.

## Completion

Within the approved scope, continue through implementation, the supported
build/run workflow, output inspection, and fixes to failures caused by the
change. Use [validation](references/validation.md) to select checks for the
changed behavior and rerun affected checks after fixes. Return runnable commands,
observed results, and explicit gaps when required evidence is unavailable.

## Related work

Model preparation belongs to the public Model Compiler or LLiMa documentation.
Viewer and device operations belong to Insight and Neat Development Environment
documentation. Use the corresponding installed skill when available. This skill
owns application code and application-side APIs.

For Apps contributions, follow the repository's `AGENTS.md` and
`CONTRIBUTING.md` for packaging, build/test, and publication workflows. Standalone
applications follow their own project's requirements.
