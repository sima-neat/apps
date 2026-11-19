# MeasureOptions and TelemetryMonitor

## What it is

Neat has two measuring tools. They answer different questions and work
together.

`MeasureOptions` configures a measurement scope. You call
`Run::start_measurement(options)` before your own push and pull loop and
`MeasureScope::stop()` after it. `stop()` returns a `MeasureReport` with
throughput, latency, counters for pushed, pulled and dropped samples, and the
time each node spent. The scope only watches. It does not push, pull, or end
your loop.

`TelemetryMonitor` records what the board did during any window you choose:
power, temperatures, CPU use and memory. It reads them from Sentinel, the
daemon on the board that samples these values every 2 seconds. You call
`start()`, run your work, and call `stop()`. It returns a `Telemetry` summary.
All values are for the whole board, not for your application alone. The
"board" here is the SoM, the SiMa system-on-module on the DevKit.

Terms used on this page:

- **MLA**: the machine learning accelerator that runs the compiled model.
- **EV74**: the vector processor that does image preprocessing in front of
  the MLA.
- **Power rail**: one supply line of the board. Sentinel reports power per
  rail and for the whole SoM.
- **Die**: the silicon chip. Die sensors are temperature sensors on it.
- **DMA buffer**: memory that the accelerators read and write directly.
- **p50, p95, p99**: the value that 50%, 95% or 99% of the samples did not
  exceed. p95 shows the slow tail that an average hides.
- **LTTng**: a Linux tracing tool. Some detailed timing needs the `lttng`
  program on the board.

## When to use it

Use `MeasureOptions` and `start_measurement()` when you own the push and pull
loop and want to know how fast the pipeline runs in your application: frames
per second, how long a frame takes from graph entry to your `pull()`, and how
many frames were dropped.

Use `TelemetryMonitor` when the question is about the board: how many watts
the SoM drew, how hot the die got, how much memory the model holds.

Do not use them for these cases:

- You only want the speed of a compiled model package with synthetic input.
  Use `Model::benchmark()`. It runs its own latency pass and throughput pass
  and fills `BenchmarkReport.telemetry` with a `TelemetryMonitor` internally.
  See [model benchmarking (#747)](https://github.com/sima-neat/apps/issues/747).
- You need the numbers in a file for CI or a support ticket. Pass the
  `MeasureReport` to `save_run_json()` (C++) or `run.save_json()` (Python), and
  call `Telemetry::save()` for the board data.

Apps does not use these two APIs today. The benchmarking examples measure
without them. `model-benchmark` calls `Model.benchmark()`. `mipi-camera-capture`
times its own loop with `time.monotonic()`.

## Minimal example

Measure a push and pull loop and record board telemetry over the same window.
Start the monitor first, then the scope. Stop them in the reverse order.

C++

```cpp
TelemetryMonitor telemetry_monitor; // board power, temperatures, CPU usage, and memory
telemetry_monitor.start();
auto scope = run.start_measurement();
// run your normal push/pull loop here, then:
auto report = scope.stop();
Telemetry telemetry = telemetry_monitor.stop();
std::cout << report.to_text() << telemetry;
```

Python

```python
telemetry_monitor = pyneat.TelemetryMonitor()
telemetry_monitor.start()
scope = run.start_measurement()

# Push and pull here.

report = scope.stop()
telemetry = telemetry_monitor.stop()
```

Source: no shipped Apps example uses these APIs. The C++ block is from Core
`docs/reference/diagnostics.md` (lines 148-154, inside the "Customer graph
performance artifact" section). The Python block is from Core
`docs/develop-apps/development-workflow/pipeline.mdx` (lines 1092-1099, section
"Measure and export evidence"). To set options, build a `MeasureOptions`, set
its fields, and pass it: `run.start_measurement(options)`.

## Options

### MeasureOptions

All fields below sit on `MeasureOptions`. Python takes the same names on
`pyneat.MeasureOptions()`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Window length | `duration_ms` | `duration_ms` | `10000` | Copied into the report and the JSON export. Does not end the window. See pitfalls | Label an export with the window you meant to run. Must be above 0 |
| Warmup length | `warmup_ms` | `warmup_ms` | `1000` | Copied into the report. Does not skip any frames. See pitfalls | Same as above. Must be 0 or more |
| Pull timeout | `timeout_ms` | `timeout_ms` | `5000` | Copied into the report. The scope never pulls, so it has no effect | None. Must be above 0 |
| Plugin latency (deprecated) | `include_plugin_latency` | `include_plugin_latency` | `false` | Adds time per plugin and kernel through LTTng | Leave off. Core will remove it |
| Plugin latency source (deprecated) | `plugin_latency_source` | `pyneat.advanced.MetricsTraceSource` value in `plugin_latency_source` | `Lttng` | Chooses the tracer for plugin latency. See the `MetricsTraceSource` table | Leave alone. Core will remove it |
| Edge latency | `include_edge_latency` | `include_edge_latency` | `false` | Adds time that frames wait in queues and hand-offs between nodes | The end-to-end p95 is far above p50, or fps is low while each node looks fast, and you suspect frames wait between nodes |
| Message latency | `include_message_latency` | `include_message_latency` | `false` | Traces every frame through every node with LTTng and fills `path_timing` | You need to know which node and which hand-off adds the delay for each frame. Expect more overhead |
| Message latency source | `message_latency_source` | `pyneat.advanced.MetricsTraceSource` value in `message_latency_source` | `Lttng` | Chooses the tracer for message latency | Set `Auto` when the board may not have `lttng`. See pitfalls |
| Keep trace | `retain_metrics_trace` | `retain_metrics_trace` | `false` | Keeps the LTTng trace folder after Core parses it | A support engineer asks for the raw trace. The folder is then in `report.metrics_trace_dir` |
| Trace folder | `metrics_trace_dir` | `metrics_trace_dir` | `""` (system temp folder) | Parent folder for the trace | The temp folder is small or read-only on your board |
| Power (deprecated) | `include_power` | `include_power` | `false` | Adds the old per-run power summary to `report.power` | Never. Use `TelemetryMonitor`. See pitfalls |
| Title | `title` | `title` | `"NEAT measurement"` | Heading of `to_text()` and stored in the report | Several scopes in one program and you need to tell the reports apart |
| Model | `model` | `model` | `""` | Label printed in the report | Same as title. Core does not read it |
| Input | `input` | `input` | `""` | Label printed in the report | Same as title |
| Placement | `placement` | `placement` | `""` | Label printed in the report | Same as title, for example which processor ran the model |
| Logical batch size | `logical_batch_size` | `logical_batch_size` | `1` | Inferences in one output. `throughput_inferences_per_s` is batches per second times this. A value of 0 or less becomes 1 | One pulled output holds several inferences, for example a model compiled with batch 4. Without it, fps reads four times too low |

`MetricsTraceSource` (Python: `pyneat.advanced.MetricsTraceSource`)

| Value | C++ | Python | What it does | When to use it |
|---|---|---|---|---|
| Auto | `MetricsTraceSource::Auto` | `pyneat.advanced.MetricsTraceSource.Auto` | Uses LTTng. If LTTng cannot start, the scope still starts and the report says "unavailable" | The same code runs on boards that may lack `lttng` |
| Off | `MetricsTraceSource::Off` | `pyneat.advanced.MetricsTraceSource.Off` | No tracing for that metric | Switch one metric off while keeping the other |
| Lttng | `MetricsTraceSource::Lttng` | `pyneat.advanced.MetricsTraceSource.Lttng` | Needs LTTng. `start_measurement()` throws if it cannot start | You want a hard error instead of a report with a gap |

`start_measurement()` forms

| Form | C++ | Python | What it does |
|---|---|---|---|
| With options | `run.start_measurement(opt)` | `run.start_measurement(opt)` | Starts a scope. `opt` defaults to `MeasureOptions{}` |
| Model runner | `runner.start_measurement(opt)` | `runner.start_measurement(opt)` | Same, on a `Model::Runner` (`pyneat.ModelRunner`) |
| Graph run | `GraphRun::start_measurement(opt)` | not available | Same, on a `GraphRun` |
| Plugin flag (deprecated) | `start_measurement(bool)` | `start_measurement(bool)` | Same as setting `include_plugin_latency` |

### MeasureReport

`MeasureScope::stop()` returns it. Calling `stop()` again returns the same
report. `stopped()` tells whether `stop()` ran. In Python the fields are
readable and writable. `to_text()` prints a table and `to_json(indent)` returns
JSON. Python also has `text()`, the same as `to_text()`.

| Field | What it holds |
|---|---|
| `options` | The options used, with `logical_batch_size` fixed to 1 or more |
| `outputs` | Outputs pulled through the run during the window |
| `elapsed_s` | Seconds from `start_measurement()` to `stop()` |
| `throughput_batches_per_s` | `outputs` divided by `elapsed_s` |
| `throughput_inferences_per_s` | Batches per second times `logical_batch_size` |
| `end_to_end` | Latency from graph entry to your `pull()`, as `MeasureLatencyStats`. It includes queue wait |
| `frame_gap` | Time between two consecutive outputs, as `MeasureLatencyStats` |
| `latency_samples_collected` | `true` when `end_to_end` or `frame_gap` has samples |
| `end_to_end_semantics`, `end_to_end_interpretation` | Text that says how to read `end_to_end` |
| `counters` | `MeasureCounters`: `inputs_enqueued`, `inputs_pushed`, `outputs_ready`, `outputs_pulled`, `inputs_dropped`, `outputs_dropped`, all counted inside the window |
| `input` | `MeasureInputStats`: push, pull, poll, drop, renegotiation and allocation counts, and average microseconds for alloc, map, copy, push, pull wait and decode |
| `node_metrics` | Time per node in the window. Always filled. Minimum and maximum per node are not available for a window |
| `plugin_latency`, `plugin_latency_unattributed` | `MeasurePluginLatency` rows. Empty unless plugin latency is on |
| `edge_latency`, `edge_latency_unattributed` | `MeasureEdgeLatency` rows. Filled when edge or message latency is on. Do not add them up |
| `path_timing` | `MeasurePathTiming`: `available`, `status`, `source`, `reason`, `warnings`, and per-node arrival, gap and output-tail rows. Needs `include_message_latency` |
| `plugin_latency_status`, `message_latency_status` | `off`, `collected`, `unavailable` or `failed` |
| `plugin_latency_source`, `message_latency_source` | Where the rows came from: `lttng`, `diagnostics` or `none` |
| `metrics_trace_dir` | Trace folder when it was kept, otherwise empty |
| `warnings`, `trace_loss_detected` | Messages from the scope, and whether LTTng lost events |
| `graph_sample_timing_unkeyed`, `graph_sample_timing_misses` | Outputs whose graph-entry time Core could not match. High values mean `end_to_end` is missing samples |
| `power` | Old power summary. Deprecated |
| `warmup_iterations` | Always `0`. Core never sets it |

`MeasureLatencyStats` has `count`, `avg_ms`, `p50_ms`, `p90_ms`, `p95_ms`,
`p99_ms` and `max_ms`.

The row types are for diagnosis. `MeasurePluginLatency` has name, backend,
phase, kernel, stage, node ids, call count, and total, average, minimum and
maximum milliseconds. `MeasureEdgeLatency` has the two ends of the edge,
sample count, total, average, minimum, maximum, p50 and p95 milliseconds, and
`timing_semantics`, for example `queue_residence`. `MeasurePathTiming` rows
use `MeasurePathStat` (samples, average, p50, p95, maximum, `reliable`). Rows
with `reliable` false lost samples and need care.

### TelemetryMonitor

| Member | C++ | Python | What it does |
|---|---|---|---|
| Create | `TelemetryMonitor()` | `pyneat.TelemetryMonitor()` | A stopped monitor. It contacts nothing until `start()` |
| Create with socket | `TelemetryMonitor(socket_path)` | `pyneat.TelemetryMonitor(socket_path)` | Reads Sentinel from another socket. Default: its standard socket |
| Start | `start()` | `start()` | Starts a new window from zero. Does nothing while running |
| Read | `read()` | `read()` | The summary so far. Does not stop the monitor |
| Stop | `stop()` | `stop()` | Ends the window and returns the final summary. When already stopped, returns the last one |
| Running | `running()` | `running()` | `true` between `start()` and `stop()` |

`Telemetry`

| Field | What it holds |
|---|---|
| `status` | `"ok"` once one sample counted. Before that, the reason there is none. `"not started"` before the first `start()` |
| `sentinel_version` | Version of the Sentinel daemon |
| `sample_interval_ms` | Time between Sentinel samples. 2000 until Core reads the real value |
| `samples` | Sentinel samples counted in the window |
| `duration_ms` | Window length, up to `stop()` or up to now while running |
| `power_som_watts` | Estimated power of the whole SoM |
| `power_mla_watts` | MLA rail. `samples == 0` on boards without this rail |
| `power_rails_watts` | Every rail the board reports |
| `die_temperatures_c` | The 7 die sensors, in degrees Celsius |
| `die_temperature_max_c` | The hottest die sensor in each sample, summarized |
| `board_temperatures_c` | The SoM top-side and bottom-side sensors |
| `cpu_usage_pct` | CPU use of the board's CPU. Each sample averages the interval before it, so `maximum` is the busiest interval |
| `cpu_cores_usage_pct` | The same per CPU core. One busy core shows here when the average stays low |
| `mla_memory_mib` | DMA buffers the MLA holds, in MiB. With nothing else running this is the model's MLA memory |
| `ev74_memory_mib` | DMA buffers EV74 holds, in MiB. A buffer shared by EV74 and the MLA counts here only |
| `linux_memory_used_mib` | Memory the whole board uses, in MiB. It includes every process and EV74's buffers. Do not add it to the two above |

Each entry above except `status`, the version and the counters is a
`TelemetryStats` with `key`, `label`, `unit`, `samples`, `mean`, `minimum`,
`maximum` and `last`. `samples == 0` means the board did not report this value
and the statistics are zero. A reported zero still counts as a sample.

`Telemetry::save(path)` writes YAML, or JSON when the name ends in `.json`. It
creates missing folders, replaces an existing file, and returns the path. It
throws on a write failure. Printing a `Telemetry` gives the same YAML.

## Language differences

- `MetricsTraceSource` and the edge and path row types
  (`MeasureEdgeLatency`, `MeasurePathTiming`, `MeasurePathStat`,
  `MeasurePathIdentity`, `MeasurePathNodeArrival`, `MeasurePathInterPluginGap`,
  `MeasurePathOutputTail`) live in `pyneat.advanced`. `MeasurePluginLatency`
  and the other report types are in `pyneat`.
- `GraphRun::start_measurement()` is C++ only. `module.cpp` has no `GraphRun`
  class.
- Python `Telemetry` and `TelemetryStats` fields are read-only. Python
  `MeasureReport` fields can be written. Python `MeasureEdgeLatency` fields
  are read-only.
- To print a `Telemetry` in Python, use `print(telemetry)`. It calls the same
  YAML writer as C++ `std::cout << telemetry`.
- `to_text()` is `to_text()` in both languages. Python also has `text()`.
- `MeasureScope` in Python has `stop()` and `stopped()` only. It is not a
  context manager, so call `stop()` yourself.
- The C++ `MeasureReportWithPluginPercentiles` and `stop_with_plugin_percentiles()`
  exist only in Core's internal build. Applications do not see them.
- Errors: `start_measurement()` throws `std::runtime_error` in C++. Python
  raises `RuntimeError`. Both for bad option values, no live run, or a failed
  LTTng start.

## Pitfalls

- **The window is not `duration_ms`.** The scope runs from
  `start_measurement()` to the moment you call `stop()`. `duration_ms`,
  `warmup_ms` and `timeout_ms` are only checked and copied into the report.
  Symptom: you set `duration_ms = 30000` and `elapsed_s` is 4.2. Fix: time the
  window yourself and call `stop()` when it ends. The header comments for
  these fields say otherwise (see the note below).
- **Warmup frames are counted.** `warmup_ms` skips nothing, and
  `warmup_iterations` stays 0. Symptom: the first frames raise `end_to_end`
  `max_ms` and lower fps. Fix: push and pull your warmup frames first, then
  call `start_measurement()`.
- **`start_measurement()` throws with `plugin_latency_source` or
  `message_latency_source` at the default `Lttng`.** With plugin or message
  latency on, and `lttng` missing or not startable, the call throws
  `LTTng metrics start failed: ...`. Fix: set the source to `Auto`. Then the
  scope starts and the report shows `unavailable` and a warning.
- **Message latency stays empty.** Symptom: `path_timing.available` is false
  and `status` is `unavailable_message_trace_disabled`, with the warning
  "Exact path timing requires MeasureOptions.include_message_latency=true".
  Fix: set `include_message_latency = true`. `include_edge_latency` alone gives
  only diagnostic totals.
- **`end_to_end` is queue time, not model time.** It runs from graph entry to
  your `pull()`. If you push many frames before you pull, it includes the wait
  in the queues. Fix: for per-frame latency, push one frame and pull it before
  the next. For fps, read `throughput_batches_per_s` instead.
- **Edge latency does not add up.** `MeasureEdgeLatency` rows have
  `non_additive` true. Symptom: the sum of edge times is larger than
  `end_to_end`. Fix: read each edge alone.
- **Deprecated power switches.** `MeasureOptions::include_power`,
  `RunOptions::power_monitor`, `RunOptions::enable_board_power()`,
  `enable_modalix_som_power()`, `enable_modalix_dvt_power()` and the same on
  `GraphRunOptions` are deprecated and point to `TelemetryMonitor`. In C++ the
  compiler prints a deprecation warning. In Python they emit a
  `DeprecationWarning`. `include_power` fills `report.power` only when the
  run was built with a power monitor on. Symptom: `report.power.enabled` is
  false. Fix: use `TelemetryMonitor` and drop these.
- **No telemetry samples in a short window.** Sentinel takes a sample about
  every 2 seconds, and samples taken before `start()` are not counted.
  Symptom: `status` reads "Sentinel took no new sample during the window; make
  the window longer than the sample interval", and every `samples` is 0. Fix:
  run the work for several sample intervals.
- **Sentinel is not reachable.** `TelemetryMonitor` does not throw. It prints
  one warning per distinct message on stderr, puts the reason in `status`,
  and keeps trying. Symptom: `status` holds an error text and the values are
  0. Fix: start Sentinel on the board, then run the window again. Check
  `status == "ok"` before you trust a value.
- **A power spike is missed.** Each value is one reading per Sentinel sample.
  Symptom: `maximum` is lower than the peak you see on a bench meter. Fix: use
  the numbers as a trend over a long window. Do not use them as a peak.
- **The numbers include other work.** Every value is for the whole board.
  Symptom: `linux_memory_used_mib` or `cpu_usage_pct` is high with a small
  model. Fix: stop other processes before the window, and do not add
  `linux_memory_used_mib` to the two accelerator memory values.
- **`start()` on a running monitor does nothing.** Symptom: you call
  `start()` to reset and the old window continues. Fix: call `stop()`, then
  `start()`.

Header comments that do not match Core's behavior:

- `Run.h` says `duration_ms` is the "timed measurement window", `warmup_ms` a
  "warmup window excluded from latency/metrics results", and `timeout_ms` a
  "per-output pull timeout". Core only validates them and copies them into the
  report.
- `TelemetryMonitor.h` says the monitor "warns once on stderr". Core warns
  once per distinct message in each window.

## Used in

Not used in Apps. No shipped example uses `MeasureOptions`, `MeasureScope`,
`start_measurement()` or `TelemetryMonitor`. The closest real usage measures
without them:

- [model-benchmark](../../examples/benchmarking/model-benchmark/) calls
  `Model.benchmark()` and reads `report.telemetry`.
- [mipi-camera-capture](../../examples/benchmarking/mipi-camera-capture/) times
  its own capture loop with `time.monotonic()`.

## Source

- C++ header: `include/pipeline/Run.h` (`MeasureOptions`, `MetricsTraceSource`,
  `MeasureReport`, `MeasureScope`, `Run::start_measurement`),
  `include/pipeline/TelemetryMonitor.h`, `include/graph/GraphRun.h`
  (`GraphRun::start_measurement`, deprecated power members),
  `include/model/Model.h` (`Model::Runner::start_measurement`)
- Python binding: `python/src/module.cpp`, `pyneat.MeasureOptions`,
  `pyneat.MeasureScope`, `pyneat.MeasureReport`, `pyneat.MeasureLatencyStats`,
  `pyneat.MeasureCounters`, `pyneat.MeasureInputStats`,
  `pyneat.MeasurePluginLatency`, `pyneat.advanced.MetricsTraceSource`,
  `pyneat.advanced.MeasureEdgeLatency`, `pyneat.advanced.MeasurePathTiming`,
  `pyneat.TelemetryMonitor`, `pyneat.Telemetry`, `pyneat.TelemetryStats`
- Core commit: `7e5317be`
