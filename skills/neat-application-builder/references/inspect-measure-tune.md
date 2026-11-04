# Inspect, Measure, And Tune

Work in this order: see what the model executes, measure a default baseline,
then change one option at a time and measure again. Core's
`docs/develop-apps/development-workflow/model.mdx` and `pipeline.mdx` define
every option named here.

## Inspect

`Model::execution_plan(...)` / `model.execution_plan(...)` takes the same
arguments as `build(...)` and returns a plan whose `to_json()` lists each node
with the processor it runs on (A65, EV74, or MLA), its kernel, the resolved
Preproc and SimaBoxDecode settings, and the tensors between nodes. It builds
on the device. Read it before adding custom preprocessing or postprocessing or
moving a stage to another processor. Where buffers live between stages is described in Core's
`docs/develop-apps/advanced-concepts/data-model-contracts/memory_model.md`.

## Measure

- `Model::benchmark(BenchmarkOptions)` measures the model route alone, not the
  application. Pass real `inputs` when decode results depend on image content,
  report the latency percentiles and `telemetry.status`, and `save()` the report
  as evidence. `min_duration_ms = 0` gives a quick smoke run.
- For the application's own loop, wrap one window in
  `Run::start_measurement()` and a `TelemetryMonitor` started and stopped around
  the same window. `TelemetryMonitor` replaces the deprecated `PowerMonitor` and
  run power options.
- Telemetry comes from Sentinel on the DevKit and covers the whole board.
  Quote power only when `status` is `ok` and the metric reports samples.

## Tune memory and buffering

- Feed camera and decoder buffers straight into the model so they stay
  zero-copy.
- Choose output memory by how long the application keeps outputs:
  - Reads each output and releases it before the next pull: `Auto` or
    `ZeroCopy`. A held output or open `Mapping` keeps its runtime buffer, so
    holding several stalls the producer.
  - Keeps outputs (frame history, another thread, per-stream taps beside a
    shared model), or a stream stalls while the application holds frames:
    `RunOptions::output_memory = OutputMemory::Owned`. Each pull copies into
    independent system memory, which holds no runtime buffer. Owned outputs are
    CPU tensors: copy one with `Tensor::to(...)` before pushing it into a graph
    that starts on the device.
- Set `OutputOptions` per output: `Latest()` for live views, `EveryFrame(n)` for
  lossless processing.
- Change queue depths and inference buffer counts only after a measurement
  shows the need, and measure again after each change.
