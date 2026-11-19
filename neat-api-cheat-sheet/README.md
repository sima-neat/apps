# Neat API cheat sheet

This folder explains the Neat Core APIs that applications use, in plain English.
Each page covers one API in C++ and Python. It says what the API does, which
options it has, what each option defaults to, and why you would change it.

The cheat sheet does not replace the Core documentation or the Core headers.
It is a short guide to choosing options. When a page and Core disagree, Core is
right and the page needs a fix.

## Layout

There is one folder per area and one page per API. A page covers both
languages. There are no separate C++ and Python pages.

| Folder | What it covers |
|---|---|
| `graph/` | `Graph`, `GraphOptions`, links between nodes, link policies, `Branch` and `Combine`, queues and overflow policies |
| `run/` | `Run`, `RunOptions`, run presets, push and pull, samples, tensors, tensor memory |
| `model/` | `Model`, model options, route options, preprocessing, box decoding, benchmarking |
| `inputs/` | The `Input` node, RTSP and HTTP inputs, cameras, decoding, caps, PCIe |
| `outputs/` | The `Output` node, `VideoSender`, `MetadataSender`, GenAI, measuring |

Every page follows [TEMPLATE.md](TEMPLATE.md).

## Pages

### Run

- [Run](run/run.md): `Run`, `RunOptions`, `RunPreset`, `RunMode`
- [Push and pull](run/push-and-pull.md): `push`, `try_push`, `pull`, `pull_tensors`, `pull_samples`, `PullStatus`, `PullError`, `close_input`
- [Sample](run/samples.md): `Sample`, `SampleKind`, `tensors_from_sample`
- [Tensor](run/tensors.md): `Tensor`, `TensorList`, `TensorMemory`, `Mapping`, `OutputMemory`

### Model and decoding

- [Model](model/model.md): loading a compiled model, `Model::Options`, and running it with `run` or `build` and a runner
- [PreprocessOptions](model/preprocessing.md): turning an application's image into the model's input: resize, color, normalization
- [BoxDecodeType](model/decoding.md): turning a detector's raw outputs into boxes, and the `decode_*` helpers
- [Model::RouteOptions](model/routes.md): using a model as a stage of a larger `Graph`, and the hand-built route nodes
- [Model::benchmark](model/benchmark.md): measuring a model's latency, throughput and power on the board

## How C++ names map to Python names

The C++ API lives in the `simaai::neat` namespace. The Python API is the
`pyneat` module. Most names follow these rules, so pages do not repeat them.

| Kind | C++ | Python | Rule |
|---|---|---|---|
| Classes and option structs | `Graph`, `RunOptions` | `pyneat.Graph`, `pyneat.RunOptions` | Same name |
| Nested types | `Model::Options`, `Model::RouteOptions` | `pyneat.ModelOptions`, `pyneat.ModelRouteOptions` | Join the two names |
| Enum values | `RunPreset::Realtime` | `pyneat.RunPreset.Realtime` | Same name, `.` instead of `::` |
| Node factories | `nodes::Input(...)`, `nodes::SimaDecode(...)` | `pyneat.nodes.input(...)`, `pyneat.nodes.sima_decode(...)` | `snake_case` in `pyneat.nodes` |
| Node group factories | `nodes::groups::RtspDecodedInput(...)` | `pyneat.groups.rtsp_decoded_input(...)` | `snake_case` in `pyneat.groups` |
| Graph helpers | `graphs::Branch(...)` | `pyneat.graphs.branch(...)` | `snake_case` in `pyneat.graphs` |
| Static factory methods | `OutputOptions::EveryFrame()` | `pyneat.OutputOptions.every_frame()` | `snake_case` |
| Option fields | `options.queue_depth` | `options.queue_depth` | Same name, when Python has the field |

Some names do not follow the rules. For example, C++ `FormatTag` is
`pyneat.Format`, and C++ `ImageSpec::PixelFormat` is `pyneat.PixelFormat`. A
page names the Python spelling whenever the rules do not predict it.

## C++ and Python do not have the same options

Python does not expose every C++ option. For example, `GraphOptions` and
`Model::RouteOptions` have C++ fields with no Python counterpart. This is why
the options table on every page has one column per language. A cell says
`not available` when that language does not have the option.

## Where the facts come from

Read these in the Core repository, [sima-neat/core](https://github.com/sima-neat/core):

- `include/` has the public C++ headers. They hold the C++ options and defaults.
- `python/src/module.cpp` has the Python bindings. They decide which options
  Python has and what Python calls them.
- `docs/develop-apps/` and `docs/reference/` have the Core documentation.

The examples in this repository show how the APIs are used. They do not show
which options exist, because an example sets only the options it needs.

Each page records the Core commit it was written against. When Core changes an
API, update the page and its commit.

## Rules for a page

- Follow [TEMPLATE.md](TEMPLATE.md) and keep its section order.
- Write for an application developer who has not read the Core source.
- Give every option a default and a reason to change it.
- Take the C++ column from the header and the Python column from the bindings.
- Keep each code block to 10 lines or fewer.
- Link to a shipped example for the complete flow.
