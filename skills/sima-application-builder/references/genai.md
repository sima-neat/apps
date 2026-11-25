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
- Use `audio` or `audio_file` for ASR. Set `language` when the source language is known, or
  leave it `auto`, and choose the `asr_task` the application needs. Check the returned
  language, `no_speech_prob`, and `avg_logprob` before keeping or retrying a transcription.
- Set `enable_thinking` only for a model that supports reasoning. Read
  `GenerationResult.reasoning` or `TokenSample.reasoning` separately from the answer text.
- For tool-calling models, set `tools` and `tool_choice` and read the parsed calls from
  `tool_calls`.

## GenAIServer

1. Create `GenAIServerOptions` when the default host or port does not fit.
2. Construct `GenAIServer` and add one or more model directories with stable served names.
3. Call `serve()` to block, or `start()` and `stop()` to manage its lifetime.
4. Manage models with `model_names()` and `remove_model(...)`.

`stop()` also removes the registered models, so add them again before restarting the same
server object.

`GenAIServer` has no authentication or TLS and allows CORS requests from any origin. Bind it
only to a trusted interface, or put it behind a network layer that controls access and
encrypts traffic.

## GenAI in a graph

GenAI graph fragments follow the same named-endpoint rules as other graphs:
[graph topologies](graph-topologies.md).
