# API Decision Map

Choose the public API shape before writing code. Start with the smallest API that
matches the application boundary.

## Classic Model

Use `Model.run(...)` when:

- The artifact is a compiled model archive, usually `.tar.gz`.
- The app runs one model on one request or one frame at a time.
- Synchronous request/response behavior is acceptable.
- The app does not need custom input/output nodes, branching, fan-in, or a long-lived producer/consumer loop.

Use `Model.build(...)` when:

- The app still centers on one model.
- The app needs a long-lived runner with repeated `push` / `pull`.
- The model-owned `Runner` interface is enough.

## Graph And Run

Use `Graph` when:

- The app has multiple stages.
- The app composes a model with public nodes or node groups.
- The app needs named public inputs or outputs.
- The app needs branching, fan-in, reusable graph fragments, or source/output nodes.
- The app must expose an application boundary beyond a single model call.

## GenAI

Use `GenAIModel` when:

- The artifact is a deployed LLiMa model directory.
- The app calls an LLM, VLM, or ASR model in-process.
- The app needs `run(...)`, `stream(...)`, task detection, or model capability checks.

Use task-specific GenAI handles when the app knows the task:

- `VisionLanguageModel` for text-only LLMs and image-capable VLMs.
- `ASRModel` for speech-to-text.

Use `GenAIServer` when:

- The app boundary is HTTP.
- A browser, UI, companion service, or remote client should call one or more models.
- The server owns model registration and request routing.

Use GenAI graph fragments only when GenAI is one stage inside a larger Neat
`Graph`.

## Reference Examples

After choosing the API family, use public examples only as implementation
references. Do not start from an apps example and infer the API shape from it.
The API contract lives in the installed Neat Library.
