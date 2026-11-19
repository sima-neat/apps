# Run

## What it is

A `Run` is the live handle to a graph that has been built and is executing. You
get one from `Graph::build(...)`, and it is the only object an application
needs once the graph is running: it reports whether the graph takes pushed
input, hands you output samples, and carries the last runtime error.

`RunOptions` is the struct you pass to `build()`. It decides how deep the
runtime queues are, what happens when they fill, and whether output tensors
share the producer's memory or are copied. `RunPreset` is a shorthand that sets
those fields together for one of three usage shapes.

A `Run` is move-only: it owns threads, queues and device memory. Hold it for as
long as you want the graph to run, and let it go out of scope (or call
`stop()`) to tear the graph down.

## When to use it

Every application that runs a graph uses `Run` — there is no alternative
surface. The choice you actually make is `RunOptions`.

Set only `preset` when your application fits one of the three shapes: a live
camera or RTSP feed (`Realtime`), a mixed workload (`Balanced`), or a file or
dataset where losing a frame is unacceptable (`Reliable`). Set the individual
fields when you need a depth or a policy the preset does not give you — but
read the pitfall below first, because an explicit value that happens to equal
the struct default is ignored.

For a one-shot "one frame in, one result out" call, use `Graph::run(...)`
instead of building a `Run` and pushing to it.

## Minimal example

C++

```cpp
simaai::neat::RunOptions run_options;
run_options.preset = simaai::neat::RunPreset::Realtime;
run_options.queue_depth = 3;
run_options.overflow_policy = simaai::neat::OverflowPolicy::KeepLatest;
run_options.output_memory = simaai::neat::OutputMemory::ZeroCopy;
auto run = graph.build(run_options);
```

Python

```python
run_opt = pyneat.RunOptions()
run_opt.queue_depth = 3
run_opt.overflow_policy = pyneat.OverflowPolicy.Block
run = graph.build(run_opt)
```

Complete flow: [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/)

## Options

### `RunOptions`

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Preset | `preset` | `preset` | `RunPreset::Balanced` | Sets `queue_depth` and `overflow_policy` together, and decides whether output storage is shared by default. | Set `Realtime` for a live camera or RTSP feed, where a late frame is worthless. Set `Reliable` for a file or dataset run, where dropping a frame corrupts the result. |
| Queue depth | `queue_depth` | `queue_depth` | `4` (then overridden by the preset — see Pitfalls) | Capacity of the ingress and internal runtime queues, and of framework-created output queues. | Deeper for a jittery source that arrives in bursts; shallower to cut latency. A public `Output` node's `OutputOptions::max_buffers` wins for its own terminal queue. |
| Overflow policy | `overflow_policy` | `overflow_policy` | `OverflowPolicy::Block` (then overridden by the preset — see Pitfalls) | What a full ingress queue does to a new frame. | `KeepLatest` for live video: drop the stale frame and keep running. `Block` when every frame must be processed. A public `Output` node uses its own `OutputOptions::drop` instead. |
| Output memory | `output_memory` | `output_memory` | `OutputMemory::Auto` | Whether pulled tensors share the producer's buffer or are copied into framework storage. | See [tensors.md](tensors.md); `ZeroCopy` is the fast path and the one that can stall the pipeline if you hold tensors too long. |
| Input timeout | `input_timeout_ms` | `input_timeout_ms` | `-1` (framework default) | Default pull timeout for `build()`/`run()` input-mode paths. | Leave it; pass an explicit `timeout_ms` to the `pull()` or `run()` call instead, which is what every shipped example does. |
| Startup preflight | `startup_preflight` | `startup_preflight` | `true` | For seeded `build(input, ...)` overloads only, pushes and pulls the seed once at build time so payload errors surface from `build()` rather than asynchronously. | Set `false` when the seed exists only to adapt caps or shape and you are latency-sensitive at startup; you then see first-sample failures through `pull()` or `last_error()`. Seedless and source builds ignore it. |
| Advanced knobs | `advanced` | `advanced` | `RunAdvancedOptions{}` | Nested struct: `copy_input`, `max_input_bytes`, `sync_num_buffers_override`, `prepare_output_cpu_visible`. | `copy_input = true` when the buffer you push is short-lived and you cannot keep it alive. `max_input_bytes` to reject oversized pushes instead of failing deeper in. The other two are compatibility fields. |
| Input-drop callback | `on_input_drop` | `on_input_drop` | empty | Called when the overflow policy drops a pushed frame, with an `InputDropInfo` record. | Add it when you need a drop counter or a log line for telemetry. It runs on the pushing thread, so keep it short. |
| Build-time JSON export | `run_export` | `run_export` | `RunAutoExportOptions{}` (disabled) | Writes a topology snapshot as soon as the `Run` is built. | Useful as a CI artifact or for graph visualisation before any frame has been pushed. For final throughput and latency numbers, export after the run has drained instead. |
| Legacy export spelling | `graph_run_export` | `graph_run_export` | `RunAutoExportOptions{}` | Backward-compatible alias for `run_export`. | Do not use in new code. |
| Board power monitor | `power_monitor` | `power_monitor` | `PowerMonitorOptions{}` | Deprecated board power telemetry. | Do not use. Use `TelemetryMonitor` around the measured work; it also records temperatures, CPU usage and memory. |

### `RunPreset`

The preset's effect is not documented in the header; these values come from
`preset_default_queue_depth`, `preset_default_overflow_policy` and
`preset_default_zero_copy` in `src/pipeline/graph/GraphBuildInput.cpp`.

| Value | Queue depth | Overflow policy | Shares output storage | Use it for |
|---|---|---|---|---|
| `Realtime` | 2 | `KeepLatest` | yes | A live camera or RTSP feed. Latency matters; a dropped stale frame is the right trade. |
| `Balanced` | 4 | `Block` | yes | The default. A mixed workload where neither latency nor completeness dominates. |
| `Reliable` | 8 | `Block` | no | A file, dataset or batch run. Every frame must be processed, and copied output is worth the cost. |

Every realtime example in this repository picks `Realtime` with `KeepLatest`
and `ZeroCopy`; `Reliable` is not used by any shipped example.

### `RunMode`

| Value | C++ | Python | What it does |
|---|---|---|---|
| Async | `RunMode::Async` | not available | Continuous pipeline; the application pushes and pulls asynchronously. |
| Sync | `RunMode::Sync` | not available | One frame in, one result out, synchronously. |

`RunMode` is not part of the surface an application configures. The header says
public `Graph` users should not pass it into `Graph::build(...)`: choose
`Graph::build(...)` for a reusable push/pull runner and `Graph::run(...)` for
one-shot execution, and the runtime selects the mode. It is listed here because
it appears in the public header and because sync mode changes the defaults you
get (see Pitfalls).

## Language differences

- `RunMode` has no Python binding at all.
- Python's `Run` exposes `json()`, `save_json()`, `graph_run_json()` and
  `save_graph_run_json()` as methods on the run; in C++ these are free
  functions (`run_to_json`, `save_run_json`, …) from `pipeline/RunExport.h`.
- Python's `Run.__bool__` wraps the C++ `explicit operator bool()`.
- The deprecated power helpers (`enable_board_power` and friends) emit a Python
  warning that points at `TelemetryMonitor`; the C++ versions are marked
  `[[deprecated]]` and warn at compile time instead.

## Pitfalls

**An explicit option that equals the struct default is treated as unset.**
`resolve_build_opt` in `src/pipeline/graph/GraphBuildInput.cpp:1383` fills in
preset-derived values with the test `if (out.queue_depth == defaults.queue_depth)`.
So writing `queue_depth = 4` — the struct default — next to
`RunPreset::Realtime` silently gives you **2**, not 4. Writing
`overflow_policy = OverflowPolicy::Block` next to `Realtime` silently gives you
`KeepLatest`. Symptom: the depth or policy you set is not the one the graph
runs with, and nothing warns you. Fix: pick a value that differs from the
struct default, or pick the preset that already means what you want. This is
why the realtime examples write `queue_depth = 3` rather than 4.

**Sync mode substitutes its own defaults before the preset is applied.** For
`RunMode::Sync`, any field still at its struct default is replaced from
`sync_run_defaults()` — `preset` becomes `Reliable` and `queue_depth` becomes
`1`. The same equality test applies, so the same surprise is available twice.

**`startup_preflight` does nothing on a seedless build.** It applies only to
`build(input, ...)` overloads that receive an initial `cv::Mat`, `TensorList`
or `Sample`. Setting it on a source-driven graph has no effect; first-frame
failures still surface through `pull()` or `last_error()`.

**A `Run` left in scope keeps the graph running.** It is move-only and its
destructor stops the runtime. Dropping it earlier than you meant to — for
example by letting a local go out of scope at the end of a setup function —
tears the pipeline down.

## Used in

- [high-density-multi-stream-object-detector](../../examples/object-detection/high-density-multi-stream-object-detector/) — `Realtime`, `KeepLatest`, `ZeroCopy`, `queue_depth = 3`
- [depth-estimator](../../examples/depth-estimation/depth-estimator/) — `Block`, explicit `queue_depth`, folder-of-images workload
- [fastflow-anomaly-detector](../../examples/anomaly-detection/fastflow-anomaly-detector/)
- [multi-stream-pose-estimator](../../examples/pose-estimation/multi-stream-pose-estimator/)
- [yolov8-instance-segmenter](../../examples/segmentation/yolov8-instance-segmenter/)

## Source

- C++ header: `include/pipeline/Run.h` (`Run`, `RunOptions`, `RunPreset`,
  `RunAdvancedOptions`, `RunAutoExportOptions`), `include/pipeline/GraphOptions.h`
  (`RunMode`)
- Preset values: `src/pipeline/graph/GraphBuildInput.cpp`
  (`preset_default_queue_depth`, `preset_default_overflow_policy`,
  `preset_default_zero_copy`, `resolve_build_opt`)
- Python binding: `python/src/module.cpp`, `nb::class_<RunOptions>`,
  `nb::class_<RunAdvancedOptions>`, `nb::enum_<RunPreset>`, `nb::class_<Run>`
- Core commit: `bce80cb6`
