# API Decision Map

Choose the public API shape before writing code. Start with the artifact and
input boundary, then select the smallest API that owns the required topology.

## Artifact and input boundary

- A classic Neat application consumes a compiled model archive, commonly an MPK
  `.tar.gz`, rather than the source ONNX file. Compilation is a prerequisite
  outside this skill; return to application design when the archive exists.
- Application-owned input means the caller supplies decoded images or tensors
  and controls request cadence.
- Graph-owned input means the application topology owns a camera, file, RTSP
  stream, encoded media source, or another producer.
- Inspect the compiled package before adding preprocessing or postprocessing.
  Keep stages already owned by the model route instead of rebuilding them as
  application nodes.

## Classic Model

Use `Model.run(...)` when:

- The artifact is a compiled model archive, usually `.tar.gz`.
- Application code supplies decoded image or tensor input.
- The app runs one model on one request or one frame at a time.
- Synchronous request/response behavior is acceptable.
- The model-owned route is the complete pipeline.

Use `Model.build(...)` when:

- The app still centers on one model.
- Application code still owns input and request cadence.
- The app needs a long-lived runner with repeated `push` / `pull`.
- The model-owned `Runner` interface is enough.

## Graph

Use `Graph` when:

- The graph owns a camera, file, RTSP stream, encoded media source, or another
  producer.
- The application explicitly composes decoding, preprocessing, or output side
  paths around the model route.
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
