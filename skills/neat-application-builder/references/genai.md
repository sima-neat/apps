# GenAI APIs

Use this reference for LLM, VLM, and ASR applications built with Neat GenAI APIs.
These APIs consume deployed LLiMa model directories, not classic MPK `.tar.gz`
model archives.

## Choose The Handle

Start with `GenAIModel` unless the application has a narrower known task.

- `genai::GenAIModel` / `pyneat.genai.GenAIModel`: auto-detect LLM, VLM, or ASR model directories.
- `genai::VisionLanguageModel` / `pyneat.genai.VisionLanguageModel`: text-only LLMs and image-capable VLMs.
- `genai::ASRModel` / `pyneat.genai.ASRModel`: speech-to-text models.
- `genai::GenAIServer` / `pyneat.genai.GenAIServer`: HTTP serving for one or more GenAI models.

## GenAIModel

Use `GenAIModel` for direct in-process application logic.

Relevant public methods:

- `task()`
- `accepts_text()`
- `accepts_image()`
- `accepts_audio()`
- `model_id()`
- `run(request)`
- `stream(request)`

`run(...)` waits for a full `GenerationResult`. `stream(...)` yields
`TokenSample` values and should be used when callers need incremental output or
cancellation.

## GenerationRequest

Keep requests explicit.

- Use `prompt` for a simple single-turn request.
- Use `messages` for chat history.
- Do not use `prompt` and `messages` together.
- Use `system_prompt` only with `prompt`.
- Use `images` for VLM prompt images.
- Use per-message images when using chat history.
- Use `audio` or `audio_file` for ASR.
- Do not mix direct images with cached images in the same request.
- Use `enable_thinking` only when the selected model supports reasoning. Read
  `GenerationResult.reasoning` or streamed `TokenSample.reasoning` separately
  from answer text.
- Use `tools` and `tool_choice` for supported tool-calling models. Read parsed
  calls from `tool_calls` instead of maintaining a model-family-specific parser.
- For ASR, use `language` when the source language is known or leave it as
  `auto`, and choose the `asr_task` required by the application. Inspect the
  returned language, `no_speech_prob`, and `avg_logprob` before deciding whether
  to keep or retry a transcription.

Capability-check the model before constructing task-specific requests when the
model directory is user-provided.

## GenAIServer

Use `GenAIServer` when the Neat application should expose HTTP endpoints.

Common steps:

1. Create `GenAIServerOptions` when the default host or port is not correct.
2. Construct `GenAIServer`.
3. Add one or more model directories with stable served names.
4. Use `serve()` for a blocking server or `start()` / `stop()` for managed lifetime.
5. Use `model_names()` and `remove_model(...)` for model management when needed.

`stop()` also removes the registered models. Add them again before restarting the
same server object.

`GenAIServer` does not provide authentication or TLS termination and allows CORS
requests from any origin. Bind it only to a trusted interface or put it behind a
network layer that provides access control and encryption.

Do not use a server when the app can call the model directly inside one process.
Direct APIs are the simpler starting point for embedded application logic and
tests.

## GenAI In Graphs

Direct GenAI calls are the default. Use public GenAI graph fragments only when
GenAI is one stage inside a larger Neat `Graph`.

Relevant public fragments:

- `genai::graphs::VisionLanguage(...)`
- `genai::graphs::SpeechTranscriber(...)`

When graph fragments are used, follow the same named endpoint rules as other
Neat graphs.

## Boundary

This skill does not cover compiling, quantizing, benchmarking, or selecting
GenAI models. Use the public LLiMa/model tooling documentation for model
preparation. This skill begins after a deployed model directory exists.
