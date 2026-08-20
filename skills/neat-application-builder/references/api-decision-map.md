# API Decision Map

Choose the public API shape before writing code. Start with the artifact and
input owner, then select the smallest API that fits the application.

## Artifact and input boundary

- Classic applications consume a compiled model archive, not ONNX. Complete
  compilation before application design.
- Decide whether the caller supplies decoded inputs or the `Graph` owns a source
  and topology.
- Inspect the model package before adding preprocessing or postprocessing.

## Classic Model

Use `Model.run(...)` when:

- The caller supplies decoded images or tensors.
- Synchronous request/response behavior is acceptable.
- The model-owned route is the complete pipeline.

Use `Model.build(...)` when:

- The caller still owns input and request cadence.
- The app needs repeated `push` / `pull`.
- The model-owned `Runner` interface is enough.

## Graph

Use `Graph` when:

- The `Graph` owns a camera, file, RTSP stream, encoded source, or other
  producer.
- The application composes decoding, preprocessing, or output paths around the
  model route.
- The application needs multiple stages, named endpoints, branching, or fan-in.

With encoded RTSP input, decide whether to decode for inference, forward the
encoded stream to `VideoSender`, or branch to both. Use the same codec across
`RtspEncodedInput`, `SimaDecode`, and `VideoSenderOptions::Passthrough`.

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

After choosing the API family, inspect the closest current Apps example before
writing code. Use the example as the implementation starting point. The
installed Neat Library remains the API contract.
