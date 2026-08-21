# API Surface Map

Use this file to discover important Neat Library API areas without turning the
skill into an API manual. Mention the area, then inspect the current packaged
source before writing code or explaining behavior.

## Contents

- Core Application Surface
- Data And Boundary Types
- Nodes And Graph Fragments
- Model Pre/Post And Decode Helpers
- GenAI Surface
- Diagnostics And Measurement
- Python Surface
- Tools Outside This Skill

## Core Application Surface

- `Model`, `Model::Options`, `Model::RouteOptions`, and `Model::Runner`
  - Inspect `include/model/Model.h`
  - Read `docs/develop-apps/development-workflow/model.mdx`
- `Graph`, `GraphOptions`, validation, save/load, describe helpers, and RTSP serving
  - Inspect `include/pipeline/Graph.h`
  - Inspect `include/pipeline/GraphOptions.h`
  - Read `docs/develop-apps/development-workflow/graph.mdx`
- `Run`, `RunOptions`, push/pull, endpoint names, measurement, and close/drain behavior
  - Inspect `include/pipeline/Run.h`
  - Read `docs/develop-apps/development-workflow/overview.mdx`

## Data And Boundary Types

- `Tensor`, `TensorList`, `TensorSpec`, dtype/layout/pixel-format helpers, and OpenCV/NumPy adapters
  - Inspect `include/pipeline/Tensor*.h`
  - Read `docs/develop-apps/development-workflow/core_types.mdx`
- `Sample`, bundles, frame IDs, timestamps, and metadata-carrying payloads
  - Inspect `include/pipeline/Tensor.h`
  - Read `docs/develop-apps/development-workflow/core_types.mdx`
- Per-frame string attributes that must stay attached through decode and graph branches
  - Inspect `Sample::attributes` in `include/pipeline/Tensor.h`
  - Read `docs/develop-apps/advanced-concepts/data-model-contracts/frame_attributes.md`
- Encoded media helpers and payload types
  - Inspect `include/pipeline/EncodedSampleUtil.h`
  - Inspect `include/pipeline/PayloadType.h`
  - Inspect `include/pipeline/FormatSpec.h`

## Nodes And Graph Fragments

- Public node umbrella includes
  - Inspect `include/neat/nodes.h`
  - Inspect `include/neat/node_groups.h`
- Boundary nodes and common graph nodes
  - Inspect `include/nodes/io/Input.h`
  - Inspect `include/nodes/common/Output.h`
  - Inspect `include/nodes/common/Queue.h`
  - Inspect `include/nodes/common/Caps.h`
- Image, video, RTSP, UDP, and metadata I/O
  - Inspect `include/nodes/io/CameraInput.h`
  - Inspect `include/nodes/io/RTSPInput.h`
  - Inspect `include/nodes/io/StillImageInput.h`
  - Inspect `include/nodes/io/UdpOutput.h`
  - Inspect `include/nodes/io/MetadataSender.h`
- Pre-built node groups for common application plumbing
  - Inspect `include/nodes/groups/*.h`
  - Start with `RtspEncodedInput.h`, `RtspDecodedInput.h`, `VideoSender.h`, `ImageInputGroup.h`, and `ModelGroups.h`

## Model Pre/Post And Decode Helpers

- Preprocess and model route options
  - Inspect `include/model/PreprocessPlan.h`
  - Inspect `include/nodes/sima/Preproc.h`
- Detection and segmentation decode helpers
  - Inspect `include/pipeline/DetectionTypes.h`
  - Inspect `include/pipeline/BoxDecodeType.h`
  - Inspect `include/pipeline/BoxDecodeOptions.h`
  - Inspect `include/pipeline/SuperPointTypes.h`
  - Inspect `include/nodes/sima/SimaBoxDecode.h`
  - Read `docs/reference/boxdecode_decode_types.md`
  - Treat `BoxDecodeType::Ssd` as a selector for the supported prepared SSD recipes, not as a generic shape-inferred SSD decoder
  - Inspect the SuperPoint profile and output format instead of decoding its result layout from tensor shape alone
- Native H.264, H.265/HEVC, JPEG, and MJPEG decoding
  - Inspect `include/nodes/sima/SimaDecode.h`
  - Prefer codec-neutral `payload_type` and `VideoSenderOptions::Passthrough(codec)` in new encoded-media code
- Fixed-function Sima nodes when composing lower-level graph routes
  - Inspect `include/nodes/sima/*.h`

## GenAI Surface

- GenAI umbrella include and task model APIs
  - Inspect `include/neat/genai.h`
  - Inspect `include/genai/GenAIModel.h`
  - Inspect `include/genai/VisionLanguageModel.h`
  - Inspect `include/genai/ASRModel.h`
  - Inspect `include/genai/GenAITypes.h`
- GenAI serving and graph fragments
  - Inspect `include/genai/GenAIServer.h`
  - Inspect `include/genai/GraphFragments.h`
  - Inspect `include/genai/GenAIOptions.h`
  - Read `docs/develop-apps/development-workflow/genai-model/index.mdx`
  - Read `docs/develop-apps/development-workflow/genai-model/direct-api.mdx`
  - Read `docs/develop-apps/development-workflow/genai-model/genai-server.mdx`

## Diagnostics And Measurement

- Structured errors, graph reports, and error code taxonomy
  - Inspect `include/pipeline/NeatError.h`
  - Inspect `include/pipeline/GraphReport.h`
  - Inspect `include/pipeline/ErrorCodes.h`
- Graph inspection helpers
  - Inspect `Graph::validate`, `Graph::describe`, `Graph::describe_backend`, `Graph::save`, and `Graph::load` in `include/pipeline/Graph.h`
- Measurement and runtime reports
  - Inspect `MeasureScope`, `MeasureReport`, and `Run::start_measurement` in `include/pipeline/Run.h`

## Python Surface

- Python bindings mirror the public C++ application concepts but names and helper methods can differ.
- Inspect `python/src/module.cpp` in packaged core source for binding truth.
- Read `docs/reference/pythonapi/` when present in the installed docs.
- For NumPy/image interop, inspect binding definitions for `Tensor`, `ModelOptions`, `RunOptions`, `GenAIModel`, and `GenAIServer`.

## Tools Outside This Skill

These are important Neat ecosystem areas, but this skill should only route to
their docs. Do not inline their workflows here.

- Model Compiler and model preparation
  - Read the Model Compiler and compile-a-model docs.
- LLiMa compile, test, and benchmark workflows
  - Read the LLiMa / GenAI tooling docs.
- Insight visualization or UI workflows
  - Read the Insight docs.
