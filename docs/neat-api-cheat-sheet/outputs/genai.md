# GenAI: GenAIServer, ASRModel and VisionLanguageModel

## What it is

The GenAI APIs run text, image-and-text, and speech models on the board. They
load a generative model from a model directory and let an application send it
requests and read the answers.

- **LLiMa** is the SiMa.ai tool that compiles a generative model into a *model
  directory* for the board. A model directory holds a `devkit/` folder, an
  `elf_files/` folder, and one config file: `devkit/vlm_config.json` (a text or
  image-and-text model) or `devkit/whisper_config.json` (a speech model).
  Core reads these folders. It does not compile models.
- **MLA** is the Machine Learning Accelerator, the chip block that runs the
  model. The board's CPU only drives it.
- **VLM** means vision-language model: it answers questions about images. A
  text-only model (an LLM) uses the same class.
- **ASR** means automatic speech recognition: it turns audio into text. The ASR
  models here are Whisper models.
- **Token** is the small piece of text a model produces one at a time.
- **Chat template** is the text format, shipped with the model, that turns
  the messages into the model's prompt.

Four types do the work:

| Type | Job |
|---|---|
| `VisionLanguageModel` | Runs a text or image-and-text model. `run()` returns the full answer. `stream()` returns it piece by piece. |
| `ASRModel` | Runs a Whisper model. `run()` and `stream()` take audio and return text. |
| `GenAIServer` | Serves one or more models over HTTP, with OpenAI-style and Ollama-style routes, so other programs can call them. |
| `GenerationRequest` | The one request object that all models take. It holds the prompt, images, audio and limits. |

`GenAIModel` is a fourth model handle. It reads the model directory and picks
`VisionLanguageModel` or `ASRModel` itself. `GenAIServer` uses it internally.

Core builds the GenAI classes only when the LLiMa package is present. Without
it, every constructor throws `NEAT GenAI/LLiMa support is not available in this
build.`

## When to use it

Use `GenAIServer` when other programs, a browser, or another device should call
the model over HTTP. Core's guide recommends it for most applications.

Use `VisionLanguageModel` or `ASRModel` directly when your own process owns the
model call, for example a script that answers one question about one image.

Use `genai::graphs::VisionLanguage(...)` or `genai::graphs::SpeechTranscriber(...)`
when the model is one stage in a larger `Graph`. See the language and graph
notes under Options.

Do not use these APIs for fixed-shape models such as detection or
classification. Use `Model` for those.

## Minimal example

Start a GenAI server with one model and serve it from a background thread.

C++

No shipped Apps example uses the GenAI classes from C++. Core's
[GenAI Server guide](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model/genai-server)
has a C++ server example.

Python

```python
server_options = pyneat.GenAIServerOptions()
server_options.host = host
server_options.port = port

genai_server = pyneat.GenAIServer(server_options)
served_name = genai_server.add_model(Path(model_path), model_name)
print(f"serving GenAI API on http://{host}:{port}", flush=True)
genai_server.start()
```

Complete flow: [detection-to-vlm-assistant](../../../examples/genai/detection-to-vlm-assistant/)

## Options

### GenAIServerOptions

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Host | `host` | `host` | `"0.0.0.0"` | Network address the server listens on. `0.0.0.0` means every network interface of the board. | The server has no login and allows requests from any web page origin. If only a program on the board calls it, set `"127.0.0.1"`. Symptom of leaving it open: any machine on the network can call `/v1/chat/completions`. |
| Port | `port` | `port` | `9998` | TCP port of the server. | Another program already uses 9998. With `start()` this ends the program, see Pitfalls. |

### GenerationRequest

One request type for every model. Each model rejects the fields that do not
belong to it, so set only the fields for your model kind.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Prompt | `prompt` | `prompt` | Not set | One user message. Use it or `messages`, not both. | Single question, no history. |
| System prompt | `system_prompt` | `system_prompt` | Not set | Instruction placed before the prompt. Valid only with `prompt`. | You want a fixed role, for example "Answer in one short sentence". With `messages`, add a `system` message instead. |
| Messages | `messages` | `messages` | Empty | Chat history as a list of `ChatMessage`. | Several turns, tool results, or an image on one turn only. |
| Images | `images` | `images` | Empty | Images for a VLM, attached to the prompt. Not allowed with `messages`. | The question is about a picture. See Language differences for how each language passes images. |
| Use cached images | `use_cached_images` | `use_cached_images` | `false` | Use the images stored earlier by `VisionLanguageModel::encode()` instead of new ones. | You ask many questions about the same image. Call `encode()` first, or the request fails. |
| Audio | `audio` | `audio` | Not set | Audio as a `Tensor`: 16 kHz, mono, `Float32`, one dimension. Needs audio metadata. | Audio is already in memory. Any other format fails. |
| Audio file | `audio_file` | `audio_file` | Not set | Path of an audio file. Use it or `audio`, not both. | Audio is in a file. |
| Language | `language` | `language` | `"auto"` | Spoken language of the audio, as a code or name. An empty string also means `auto`. | A short or noisy clip is detected as the wrong language and the text comes out in that language. Set `"en"` when you know it. |
| ASR task | `asr_task` | `asr_task` | `ASRTask::Transcribe` | `Transcribe` writes the speech in its own language. `Translate` writes English. | You need English text from foreign speech. Non-ASR models ignore it. |
| Max new tokens | `max_new_tokens` | `max_new_tokens` | `0` | Upper limit on generated tokens. `0` sets no limit of its own. | The answer rambles. With `0` the model runs until it stops or fills its context, and `finish_reason` is `length`. Prompt tokens plus this value can be at most 65535. |
| Enable thinking | `enable_thinking` | `enable_thinking` | `false` | Asks a reasoning model to return its reasoning, in `reasoning`. | You want the reasoning separate from the answer. A model that cannot reason throws. |
| Tools | `tools` | `tools` | Empty list | Function definitions the model may call, as JSON. Each needs `type` `"function"` and a `function.name`. | The model should return a call instead of text. |
| Tool choice | `tool_choice` | `tool_choice` | Not set (`null`, `None`) | `"auto"` or `"none"`. `"none"` turns tool calls off even when `tools` is set. | You keep `tools` in the request but want plain text this turn. |
| Reasoning effort | `reasoning_effort` | `reasoning_effort` | `"low"` | How much the model reasons: `low`, `medium`, `high`. Separate from `enable_thinking`. | A reasoning model gives shallow answers. Core's guides name GPT-OSS. Other models react only if their chat template uses it. |

### ChatMessage

One turn of a conversation, used in `GenerationRequest.messages`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Role | `role` | `role` | Empty | Who wrote the message, for example `system`, `user`, `assistant`, `tool`. | Always set it. Core copies it to the model's chat template without a check. |
| Content | `content` | `content` | Empty | Message text. | Always. |
| Images | `images` | `images` | Empty | Images attached to this message. | The image belongs to one turn of a longer chat. Top-level `images` is not allowed with `messages`. |
| Use cached images | `use_cached_images` | `use_cached_images` | `false` | Put the images stored by `encode()` at this message. | At most one message per request may set it. Two give `accepts at most one cached image insertion point`. |
| Tool calls | `tool_calls` | `tool_calls` | Empty list | Tool calls an `assistant` message made earlier, as JSON. | You send a tool result back and must replay the call that asked for it. |
| Tool call id | `tool_call_id` | `tool_call_id` | Not set | Id of the call a `tool` message answers. | Tool results, so the model can match result to call. |
| Name | `name` | `name` | Not set | Optional name of the tool or speaker. | Tool results that carry the tool name. |

### VisionLanguageOptions

For `genai::graphs::VisionLanguage(...)`. In Python these live in the
`pyneat.genai` submodule, not at the top of `pyneat`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| System prompt | `system_prompt` | `system_prompt` | `""` (none) | Sent with every prompt that arrives. | The stage should always answer in one role. |
| Max new tokens | `max_new_tokens` | `max_new_tokens` | `0` | Same as `GenerationRequest::max_new_tokens`. `0` sets no limit of its own. | A stage in a live loop must not run long. Set a cap such as `128`. |
| Streaming | `streaming` | `streaming` | `true` | `true`: one `tokens` sample per text piece. `false`: one `tokens` sample with the whole answer. | Set `false` when the next node needs the full answer in one sample. |
| Encode images on input | `encode_images_on_input` | `encode_images_on_input` | `true` | Runs the image through the vision encoder when it arrives and caches it. Later prompts reuse it. | Set `false` when every prompt comes with a new image anyway. The image is then kept as it is and sent with every later prompt until the next image arrives. |

### SpeechTranscriberOptions

For `genai::graphs::SpeechTranscriber(...)`. Python: `pyneat.genai.SpeechTranscriberOptions`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Language | `language` | `language` | `"auto"` | Spoken language. Empty also means `auto`. | Same as `GenerationRequest::language`. |
| Task | `task` | `task` | `ASRTask::Transcribe` | `Transcribe` or `Translate`. | Same as `GenerationRequest::asr_task`. |
| Streaming | `streaming` | `streaming` | `true` | `true`: text arrives in several `tokens` samples. `false`: one sample with the whole text. | Same as for `VisionLanguageOptions`. |

### ASRTask and GenAITask

`ASRTask`: C++ `ASRTask::Transcribe`, Python `pyneat.ASRTask.Transcribe`.

| Value | What it does |
|---|---|
| `Transcribe` | Writes the speech in the language that was spoken. |
| `Translate` | Writes the speech in English. |

`GenAITask` is what `GenAIModel::task()` returns: `VisionLanguage` or `ASR`.
Python: `pyneat.GenAITask`.

### GenAIServer methods

| Method | C++ | Python | What it does |
|---|---|---|---|
| Create | `GenAIServer(GenAIServerOptions = {})` | `pyneat.GenAIServer(options)` | Makes a server. Nothing is listening yet. |
| Add by path | `add_model(path)` | `add_model(model_dir)` | Reads the model directory, creates the model and returns the served name. A text or image model loads onto the MLA here, so its load or out-of-memory failure throws from `add_model()`. A speech model loads on its first request, see Pitfalls. The name comes from `model_type` in the config, in lower case without a leading `vlm-`, `llm-` or `asr-`. If that is empty, the directory name is used. |
| Add by path and name | `add_model(path, name)` | `add_model(model_dir, served_name)` | Same, with the name you choose. Clients put this name in the `model` field. |
| Add a loaded model | `add_model(name, shared_ptr<GenAIModel>)` | not available | Registers a model you already loaded. |
| Remove | `remove_model(name)` | `remove_model(served_name)` | Returns `true` if the name was registered. The model's memory is released once no request is still using it. |
| List | `model_names()` | `model_names()` | Names of the registered models. |
| Serve, blocking | `serve()` | not available | Warms the models, then listens until `stop()`. |
| Serve, background | `start()` | `start()` | Warms the models, then listens on a background thread and returns. |
| Stop | `stop()` | `stop()` | Stops listening and releases every registered model. The destructor does the same. |

The server's HTTP routes and request fields are in Core's
[GenAI Server guide](https://developer.sima.ai/software/develop-apps/development-workflow/genai-model/genai-server).

### VisionLanguageModel methods

| Method | C++ | Python | What it does |
|---|---|---|---|
| Create | `VisionLanguageModel(model_dir)` | `pyneat.VisionLanguageModel(model_dir)` | Reads the model directory and loads the model onto the MLA before it returns, so a load or out-of-memory failure throws here. This takes several seconds, and the first request is then as fast as the next. |
| Accepts images | `accepts_image()` | `accepts_image()` | `true` when the model has a vision part. |
| Model id | `model_id()` | `model_id()` | The model directory's name. |
| Run | `run(request)` | `run(request)` | Returns a full `GenerationResult`. |
| Stream | `stream(request)` | `stream(request)` | Returns a `GenerationStream`. |
| Encode images | `encode(Tensor)`, `encode(vector<Tensor>)`, `encode(cv::Mat)`, `encode(vector<cv::Mat>)` | `encode(images)` | Runs the vision encoder and stores the result in the model. Returns `true`. A new call replaces the stored images. |
| Cached count | `cached_image_count()` | `cached_image_count()` | Number of stored images. |
| Set LoRA | `set_lora(name)` | `set_lora(adapter_name)` | Switches to a LoRA adapter, a small set of extra weights that changes the model's behaviour. The name is one folder name from the model package. |
| Unset LoRA | `unset_lora()` | `unset_lora()` | Back to the base weights. |

Only one call runs on a model at a time. A second `run()`, `encode()` or
`set_lora()` from another thread waits for the first to finish. `stream()`
returns at once: its background thread does the waiting, so the first `next()`
waits instead.

### ASRModel methods

| Method | C++ | Python | What it does |
|---|---|---|---|
| Create | `ASRModel(model_dir)` | `pyneat.ASRModel(model_dir)` | Reads the directory. The model loads on the first `run()` or `stream()`. |
| Accepts audio | `accepts_audio()` | `accepts_audio()` | Always `true`. |
| Model id | `model_id()` | `model_id()` | The model directory's name. |
| Run | `run(request)` | `run(request)` | Returns a `GenerationResult` with text, language and the two Whisper confidence values. |
| Stream | `stream(request)` | `stream(request)` | Returns a `GenerationStream`. |

### GenAIModel methods

C++ `GenAIModel`, Python `pyneat.GenAIModel`. It has `task()`, `accepts_text()`,
`accepts_image()`, `accepts_audio()`, `model_id()`, `set_lora()`,
`unset_lora()`, `run()` and `stream()`, all bound in Python with the same names.
`set_lora()` on a speech model throws.

### GenerationStream

Returned by `stream()`. Work starts on a background thread as soon as the stream
exists.

| Method | C++ | Python | What it does |
|---|---|---|---|
| Next | `next()` | `next()` | Waits for the next `TokenSample`. Returns an empty `optional` (`None`) when the stream is finished. Rethrows an error from the background thread. |
| Cancel | `cancel()` | `cancel()` | Asks the model to stop. |
| Loop | `begin()` and `end()` | `for token in stream` | Loop over the samples until the stream ends. |

Dropping the stream cancels it and waits for the thread to finish.

### Result types

`GenerationResult` is returned by `run()`.

| Field | C++ and Python | What it holds |
|---|---|---|
| Text | `text` | The answer. Empty when the model made a tool call. |
| Reasoning | `reasoning` | The model's reasoning, when thinking was on and the model produced it. |
| Tool calls | `tool_calls` | JSON list of calls. Empty list when none. |
| Finish reason | `finish_reason` | `stop`, `length` (limit or full context), `tool_calls`, or `interrupted`. |
| Metrics | `metrics` | `GenerationMetrics`: `generated_tokens`, `time_to_first_token_s`, `tokens_per_second`. |
| Language | `language` | ASR only. Detected or chosen language. |
| No-speech probability | `no_speech_prob` | ASR only. How likely the audio holds no speech. Can be empty. |
| Average log probability | `avg_logprob` | ASR only. Whisper's average confidence in its tokens. Closer to 0 is better. Can be empty. |

`TokenSample` is what a stream returns. It has `text`, `reasoning`, `metrics`,
`is_final`, `finish_reason`, `language`, `no_speech_prob`, `avg_logprob` and
`tool_calls`. A normal sample carries either `text` or `reasoning`. The last
sample has `is_final` set and carries the finish reason, the ASR values and the
final metrics. It carries no text, so build the full text from the earlier samples.

### Graph fragments

`genai::graphs::VisionLanguage(model, options, name = "vision_language")` and
`genai::graphs::SpeechTranscriber(model, options, name = "speech_transcriber")`
return a `Graph` with named endpoints. The model is passed as a `shared_ptr`.
Python: `pyneat.genai.graphs.vision_language(model, options, label)` and
`pyneat.genai.graphs.speech_transcriber(model, options, label)`. The third
argument is called `label` in Python.

| Fragment | Inputs | Outputs |
|---|---|---|
| `VisionLanguage` | `prompt` (text), `image` (RGB `uint8` image tensor), `use_cached_image` (text `true` or `false`) | `tokens`, `done`, `encoded`, `error` |
| `SpeechTranscriber` | `audio` (16 kHz mono float tensor), `audio_path` (text path) | `tokens`, `done`, `error` |

`tokens` carries answer text. `done` carries a bundle with `text`,
`finish_reason`, `generated_tokens`, `time_to_first_token_s` and
`tokens_per_second`, plus `cached_image_count` (vision) or `language`,
`no_speech_prob` and `avg_logprob` (speech). `encoded` reports an image was
accepted. A failure inside the stage goes to `error` as text and the graph keeps
running.

## Language differences

- `GenAIServer::serve()` and `add_model(name, shared_ptr<GenAIModel>)` exist only
  in C++. In Python, use `start()` and keep the program alive yourself, as the
  example does.
- C++ takes images as `Tensor` or, with OpenCV, as `cv::Mat`. A 3-channel
  `cv::Mat` is read as BGR and converted to RGB. Python takes only a list or
  tuple of tensors or DLPack-compatible arrays. DLPack is the standard that
  lets NumPy, PyTorch and pyneat share an array without copying it. A single
  array, not in a list, raises an error. Python marks the images as RGB and
  the binding does no BGR conversion, so convert to RGB first.
- C++ `GenerationStream` is a range (`begin()`, `end()`). Python is an iterator
  and ends the loop with `StopIteration`. `next()` returns `None` at the end.
- Python has these types on `pyneat` and also under `pyneat.genai`. The two
  option structs `VisionLanguageOptions` and `SpeechTranscriberOptions` and the
  `graphs` functions exist only under `pyneat.genai`.
- Python names for the graph functions use `snake_case`, and the name argument
  is `label`.

## Pitfalls

- **`start()` with a busy port ends the program**, and no `try` block catches
  it. Pick a free port in `GenAIServerOptions`. Core issue
  [#1470](https://github.com/sima-neat/core/issues/1470).
- **One bad model stops the whole server.** `serve()` and `start()` run one
  small request on every registered model before they listen. If one fails,
  the call throws `GenAIServer warmup failed for model '<name>'` and nothing
  listens. Fix: remove that model with `remove_model()` and start again. The
  Neat GenAI Studio example does this for its speech model.
- **`stop()` removes your models.** After `stop()` the server holds no models.
  Fix: call `add_model()` again before the next `start()`.
- **A speech model added to a running server loads on its first request.**
  `start()` runs one small request on every model it holds, once. A speech
  model added after `start()` therefore loads only when the first user
  request arrives, which is then slow and is where a load or out-of-memory
  failure shows up. A text or image model has no such delay: it loads inside
  `add_model()`. Fix: after adding a speech model, send one short request
  yourself, as [neat-genai-studio](../../../examples/genai/neat-genai-studio/)
  does.
- **Same served name twice.** `add_model()` throws `OpenAI served model already
  exists`. Fix: pass a different name, or call `remove_model()` first. Two
  models whose config gives the same default name need explicit names.
- **Wrong class for the directory.** `VisionLanguageModel` on a Whisper
  directory, or `ASRModel` on a text or image model, throws. Fix: use the
  matching class, or `GenAIModel`.
- **Incomplete model directory.** A directory without `devkit/`, `elf_files/`
  or exactly one of the two config files throws. Fix: point at the directory
  LLiMa produced.
- **Speculative-decoding pair.** Speculative decoding speeds up a large model
  with a small draft model. Such a package has two model folders. Pointing at
  one of them throws `is part of a speculative-decoding pair`. Fix: pass the
  parent directory. Such a model does not support `set_lora()`.
- **A stream reports most errors late.** `stream()` checks the request shape
  and thinking support at once. Bad image or audio data fails on the background
  thread and appears when you call `next()`. Fix: wrap the loop, not only the
  `stream()` call, in error handling.
- **Image format.** A VLM image must be `uint8`, in HWC layout (height, width,
  channels, the layout OpenCV and NumPy images use), with 3 channels, tagged
  RGB. Any other image throws. A text-only model rejects images.
- **Prompt too long.** A prompt at or above the model's context size throws.
  Over HTTP this is status 400 with code `context_length_exceeded`. Fix:
  shorten the input or drop old messages.
- **`encode()` replaces earlier images.** Each call overwrites the stored set. A
  request with `use_cached_images` then sees only the latest call.
- **Cached images with `messages`.** Top-level `images` and `use_cached_images`
  are not allowed with `messages`; Core says to attach images to a
  `ChatMessage`. Fix: set `images` or `use_cached_images` on the message.
- **Speech results have no metrics.** `ASRModel::run()` leaves `metrics` at zero.
  A stream's final sample reports `generated_tokens` as `0`. Use your own timer
  for speech speed.
- **Graph fragment without an image.** On an image-capable model, a prompt that
  arrives before any image or cached image goes to `error`. Send the `image`
  input first.
- **`remove_model()` does not cancel a running request.** The request keeps its
  own reference and the model memory is released after it ends.

## Used in

- [detection-to-vlm-assistant](../../../examples/genai/detection-to-vlm-assistant/) (`GenAIServerOptions`, `GenAIServer`, `add_model`, `start`, `stop`; Python)
- [neat-genai-studio](../../../examples/genai/neat-genai-studio/) (`GenAIServer` with `add_model`, `remove_model`, `model_names`, `start`, `stop`; `ASRModel` and `GenerationRequest` with `audio_file` and `language` in `src/python/server/probe_asr.py`; Python)

No shipped example uses C++, `VisionLanguageModel`, `GenAIModel`,
`GenerationStream`, or the graph fragments.

## Source

- C++ header: `include/genai/GenAIServer.h`, `include/genai/ASRModel.h`, `include/genai/VisionLanguageModel.h`, `include/genai/GenAIModel.h`, `include/genai/GenAIOptions.h`, `include/genai/GenAITypes.h`, `include/genai/GenAIValueTypes.h`, `include/genai/GraphFragments.h`, `include/neat/genai.h`
- Python binding: `python/src/module.cpp`, `GenAIServer`, `GenAIServerOptions`, `GenAIModel`, `VisionLanguageModel`, `ASRModel`, `GenerationRequest`, `GenerationResult`, `GenerationStream`, `TokenSample`, `GenerationMetrics`, `ChatMessage`, `ImageList`, `GenAITask`, `ASRTask`, `genai.VisionLanguageOptions`, `genai.SpeechTranscriberOptions`, `genai.graphs.vision_language`, `genai.graphs.speech_transcriber`
