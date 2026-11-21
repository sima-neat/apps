# PCIeSrc

## What it is

`PCIeSrc` is a node that receives samples over PCIe when the Modalix board is
a PCIe card inside a PC. A node is one processing step in a graph. The PC
(the header calls it the PCIe host) runs its own application that pushes
samples, for example encoded video, to the card.

The PC also sends the caps of each stream. Caps are the format, size
and frame rate of the stream. `PCIeSrc` allocates the received samples in
SiMa memory on the card. SiMa memory is buffer memory that the board's
processors can read directly.

## When to use it

Use `PCIeSrc` as the source of the graph that runs on a Modalix PCIe card,
when the PC sends the video. When the card reads a camera or a stream
by itself, use [RtspDecodedInput or RtspEncodedInput](rtsp.md) or
[a camera input](cameras.md) instead. For frames that an application on the
board produces itself, use [Input](input-node.md).

One `PCIeSrc` can feed several streams. Each stream leaves the node on its
own output named `src_N`, where `N` is the stream number, and is connected to
its own decoder graph.

## Minimal example

Create the source and connect stream output `src_N` to that stream's decoder
graph.

C++

```cpp
simaai::neat::PCIeSrcOptions options;
options.queue = config.queue;
options.buffer_size = config.pcie_buffer_size;
options.pool_size = config.pcie_pool_size;
const auto source = std::make_shared<simaai::neat::PCIeSrc>(std::move(options));
graph.connect(source, "src_" + std::to_string(stream), decoder);
```

Python

No shipped example uses `PCIeSrc` from Python. See the language differences.

Complete flow: [pcie-high-density-multi-stream-object-detector](../../../examples/object-detection/pcie-high-density-multi-stream-object-detector/)

## Options

`PCIeSrc` has one form: C++ `nodes::PCIeSrc(opt)` or
`std::make_shared<PCIeSrc>(opt)`, Python `pyneat.nodes.pcie_src(opt)`.

`PCIeSrcOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Queue | `queue` | `queue` | `0` | The PCIe data queue that the node listens on | The PC sends on another queue. The pcie example reads it from `card.queue` in its config and accepts `0` to `5` |
| Buffer size | `buffer_size` | `buffer_size` | `4194304` | Output buffer size in bytes when no size comes with the stream. Also the largest encoded sample | Keep the default. The pcie example reads it from its config and ships the default |
| Pool size | `pool_size` | `pool_size` | `0` | Output buffers per stream. `0` keeps the default of the PCIe element | The card runs out of memory: the pcie example's README says to lower `pcie.pool_size` if `CmaFree` on the card approaches zero. The example's configs set `2` or `4`, and it requires its decoder `input_buffers` to be at least `pool_size` |

## Language differences

- C++ connects a stream output to a graph with
  `Graph::connect(node, "src_N", graph)`. Python has no `connect` that takes
  an output name, so a Python graph cannot connect the outputs `src_N` of
  `PCIeSrc` one by one.

## Pitfalls

- **The PCIe element is not installed.** On a board without the PCIe
  element, building the graph fails with
  `PCIeSrc::backend_fragment: required GStreamer element not found: neatpciesrc`.
  Fix: use a card with a matching Neat Core installation, as the pcie
  example's README requires. Read in the source, not verified on a device.
- **Card memory runs out with many streams.** Each stream has its own output
  pool. The pcie example's README says to lower `input.decoder_buffers` or
  `pcie.pool_size` if `CmaFree` on the card approaches zero. Read in the
  example, not verified on a device.

## Used in

- [pcie-high-density-multi-stream-object-detector](../../../examples/object-detection/pcie-high-density-multi-stream-object-detector/) (`PCIeSrc` with one decoder graph per `src_N` output, C++ only)

## Source

- C++ header: `include/nodes/sima/PCIeSrc.h` (`PCIeSrcOptions`, `PCIeSrc`), `include/pipeline/Graph.h` (`Graph::connect` with an output name)
- Python binding: `python/src/module.cpp`, `PCIeSrcOptions`, `nodes.pcie_src`
- Core commit: `b1f9556c`
