# API Surface Map

Where each public API lives in Core source of the installed version. Paths are relative to the
source root. The installed headers are the callable contract and mark deprecated APIs.

## Application surface

- Entry point and development loop: `include/neat.h`, `docs/develop-apps/development-workflow/index.md`
- Generated C++ and Python reference: `docs/reference/`
- `Model`, `Model::Options`, `Model::RouteOptions`, `Model::Runner`: `include/model/Model.h`,
  `docs/develop-apps/development-workflow/model.mdx`
- `Graph`, `GraphOptions`, `GraphLinkOptions`, validation, save and load:
  `include/pipeline/Graph.h`, `include/pipeline/GraphOptions.h`,
  `docs/develop-apps/development-workflow/graph.mdx`, and
  `docs/develop-apps/advanced-concepts/application-design/graphs.md` for branching and joining
- `Run`, `RunOptions`, `OutputMemory`, push and pull, close and drain:
  `include/pipeline/Run.h`, `docs/develop-apps/development-workflow/overview.mdx`,
  `docs/develop-apps/development-workflow/pipeline.mdx`

## Data types

- `Tensor`, `TensorList`, `TensorSpec`, dtype, layout, pixel formats, OpenCV and NumPy
  adapters: `include/pipeline/Tensor.h`, `TensorCore.h`, `TensorSpec.h`, `TensorTypes.h`,
  `docs/develop-apps/development-workflow/core_types.mdx`,
  `docs/develop-apps/advanced-concepts/data-model-contracts/data_formats.md`
- `Sample`, frame identity, and `Sample::attributes`: `include/pipeline/GraphOptions.h`,
  `docs/develop-apps/advanced-concepts/data-model-contracts/frame_attributes.md`
- Memory placement between stages:
  `docs/develop-apps/advanced-concepts/data-model-contracts/memory_model.md`
- Encoded media and payload types: `include/pipeline/EncodedSampleUtil.h`,
  `include/pipeline/PayloadType.h`, `include/pipeline/FormatSpec.h`

## Nodes and node groups

- Umbrella includes: `include/neat/nodes.h`, `include/neat/node_groups.h`,
  `docs/develop-apps/development-workflow/node.mdx`
- `nodes::Input`, `nodes::Output`, `nodes::Queue`, and other common nodes:
  `include/nodes/io/Input.h`, `include/nodes/common/`
- Cameras, images, video, RTSP, HTTP, UDP, and `MetadataSender`: `include/nodes/io/`,
  `docs/reference/nodes/camera-input.mdx`, and
  `docs/develop-apps/advanced-concepts/application-design/`
- Node groups such as `RtspEncodedInput`, `RtspDecodedInput`, `VideoSender`, and
  `ImageInputGroup`: `include/nodes/groups/`
- `SimaDecode` and `SimaEncode` for H.264, H.265, JPEG, and MJPEG:
  `include/nodes/sima/SimaDecode.h`, `include/nodes/sima/SimaEncode.h`
- Other fixed-function nodes: `include/nodes/sima/`
- LiDAR voxelization and bird's-eye view: `docs/reference/lidar_bev_kernel_nodes.md`

## Preprocessing and decoding

- Preprocessing options: `include/model/PreprocessPlan.h`, `include/nodes/sima/Preproc.h`
- Box decoding and detection types: `include/pipeline/DetectionTypes.h`, `BoxDecodeType.h`,
  `BoxDecodeOptions.h`, `SuperPointTypes.h`, `include/nodes/sima/SimaBoxDecode.h`,
  `docs/reference/boxdecode_decode_types.md`

## GenAI

- `include/neat/genai.h`, `include/genai/`, and
  `docs/develop-apps/development-workflow/genai-model/` (`index.mdx`, `direct-api.mdx`,
  `genai-server.mdx`)

## Diagnostics and measurement

- Errors and graph reports: `include/pipeline/NeatError.h`, `GraphReport.h`, `ErrorCodes.h`,
  `docs/reference/error-codes.md`
- `Graph::validate`, `describe`, `describe_backend`, `save`, `load`: `include/pipeline/Graph.h`
- `Model::execution_plan`, `BenchmarkOptions`, `BenchmarkReport`: `include/model/Model.h`,
  `include/pipeline/ExecutionPlan.h`
- `Run::start_measurement`, `MeasureScope`, `MeasureReport`: `include/pipeline/Run.h`
- `TelemetryMonitor`: `include/pipeline/TelemetryMonitor.h`

## Python

Python bindings follow the C++ concepts, but names and helper methods can differ.
`python/src/module.cpp` is the binding source, and `docs/reference/pythonapi/` documents it
when the installed docs include it.
