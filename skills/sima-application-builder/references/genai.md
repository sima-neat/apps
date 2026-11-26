# GenAI

The GenAI APIs run LLM, VLM, and ASR models from a deployed LLiMa model directory, not from a
classic `*_mpk.tar.gz` archive. `include/neat/genai.h`, `include/genai/`, and Core's
`docs/develop-apps/development-workflow/genai-model/` define them.

## Choose the handle

| The application | Use |
|---|---|
| Calls a model in-process and may not know its task | `genai::GenAIModel` / `pyneat.genai.GenAIModel`, which detects LLM, VLM, or ASR |
| Calls a text-only LLM or an image-capable VLM | `genai::VisionLanguageModel` / `pyneat.genai.VisionLanguageModel` |
| Transcribes speech | `genai::ASRModel` / `pyneat.genai.ASRModel` |
| Serves one or more models over HTTP to a browser, UI, or remote client | `genai::GenAIServer` / `pyneat.genai.GenAIServer` |
| Runs GenAI as one stage inside a larger `Graph` | `genai::graphs::VisionLanguage(...)` or `genai::graphs::SpeechTranscriber(...)` |

Direct in-process calls are the simplest path for embedded logic and tests. Choose
`GenAIServer` only when the boundary is HTTP.

## GenAIModel

- `task()`, `accepts_text()`, `accepts_image()`, `accepts_audio()`, and `model_id()` describe
  the model. When the user supplies the model directory, check them before building a
  task-specific request.
- `run(request)` waits for a full `GenerationResult`. `stream(request)` yields `TokenSample`
  values. Use it for incremental output or cancellation.

## GenerationRequest

- Use `prompt` for a single-turn request, with `system_prompt` when needed. Use `messages` for
  chat history, with images attached per message. A request carries one of the two.
- Use `images` for VLM prompt images. A request carries direct images or cached images, not both.
  To reuse images across requests, call `VisionLanguageModel::encode(images)` once, then set
  `use_cached_images = true` on each request (or on the `ChatMessage`) instead of passing
  `images`. `cached_image_count()` reports how many are cached.
- Use `audio` or `audio_file` for ASR. Set `language` when the source language is known, or
  leave it `auto`, and choose the `asr_task` the application needs. Check the returned
  language, `no_speech_prob`, and `avg_logprob` before keeping or retrying a transcription.
- Set `enable_thinking` only for a model that supports reasoning. Read
  `GenerationResult.reasoning` or `TokenSample.reasoning` separately from the answer text.
  `reasoning_effort` (`low` by default, `medium`, or `high`) sets how much the model reasons,
  independent of `enable_thinking`, and only models whose chat template reads it respond.
- For tool-calling models, set `tools` and read the parsed calls from `tool_calls`.
  `tool_choice` accepts only `"auto"` or `"none"`, and any other value fails the request.

## GenAIServer

1. Create `GenAIServerOptions`. The defaults are `host = "0.0.0.0"`, every network interface,
   and `port = 9998`. Set `host = "127.0.0.1"` when only programs on the board call the server.
2. Construct `GenAIServer` and add one or more model directories with stable served names.
3. Call `start()` and `stop()` to manage its lifetime. C++ also has `serve()`, which blocks
   until `stop()`. Python has no `serve()`: call `start()`, then wait on the application's stop
   flag.
4. Manage models with `model_names()` and `remove_model(...)`.

`stop()` also removes the registered models, so add them again before restarting the same
server object.

`GenAIServer` has no authentication or TLS and allows CORS requests from any origin. Bind it
only to a trusted interface, or put it behind a network layer that controls access and
encrypts traffic.

## GenAI in a graph

GenAI graph fragments follow the same named-endpoint rules as other graphs:
[graph topologies](graph-topologies.md). In Python they are
`pyneat.genai.graphs.vision_language(...)` and `pyneat.genai.graphs.speech_transcriber(...)`.
