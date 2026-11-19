# Model::benchmark

## What it is

`Model::benchmark` measures how fast a loaded model runs on the board. A
model here is a compiled model package (an MPK, `*_mpk.tar.gz`) loaded with
`simaai::neat::Model`, and it runs on the MLA, the board's machine-learning
accelerator. The call reports latency, throughput (FPS), power, energy and
memory, and returns them as a `BenchmarkReport`.

It runs two passes over the model's route, the chain of preprocessing,
inference and postprocessing stages the `Model` resolved from the package.
The latency pass sends one input and waits for its result before it sends the
next. The throughput pass keeps several inputs in flight at once. Power and
board telemetry come from Sentinel, the board's telemetry daemon, which
samples power, temperatures, CPU and memory every 2 s during the throughput
pass.

## When to use it

Use it to answer "how fast is this model on this board?": to compare two
models or two quantized variants, to check that a new Neat release did not
slow a model down, or to estimate the power a model draws. By default it feeds
synthetic inputs generated from `Model::input_specs()`, so it needs no camera,
video or image files.

It does not measure an application. Input decoding, video and metadata output,
overlays and your own postprocessing are not in the numbers. To measure a
complete pipeline, measure the running `Graph` (the application's pipeline of
nodes) or the model runner instead (`Model::Runner::start_measurement`, see the
`outputs/` pages). Do not quote
`benchmark` FPS as the throughput of a finished application.

## Minimal example

No shipped example calls `benchmark` from C++. C++ uses the same call,
`Model::benchmark(BenchmarkOptions)`, which returns a `BenchmarkReport` with
the same fields.

Python, from
[`model-benchmark/src/python/main.py`](../../examples/benchmarking/model-benchmark/src/python/main.py)
(lines 156, 162–165, 172–174, de-indented):

```python
model = pyneat.Model(str(model_path))
benchmark_options = pyneat.BenchmarkOptions()
benchmark_options.num_samples = frames
benchmark_options.min_duration_ms = min_duration_ms
report = model.benchmark(benchmark_options)
print(f"latency_ms={report.latency_ms}")
print(f"latency_p99_ms={report.latency_p99_ms}")
print(f"fps={report.fps}")
```

Complete flow: [Model Benchmark](../../examples/benchmarking/model-benchmark/)

## Options

`BenchmarkOptions` holds every option. In Python it is
`pyneat.BenchmarkOptions`, and an unset optional field is `None`.

| Option | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Minimum samples | `num_samples` | `num_samples` | `100` | The least number of inputs each pass measures. Must be greater than 0. A pass stops only after it has measured this many inputs **and** run for `min_duration_ms`. `energy_joules` is reported for this many inputs. | Raise it when you want the percentiles (`latency_p99_ms`) to come from more inputs on a slow model that measures few inputs in the time window. Set it to the number of inputs your application handles per job when you want `energy_joules` to answer "how much energy does one job cost". |
| Source width | `original_width` | `original_width` | Not set (`None`) | Width, in pixels, of the image your application would feed. Set it together with `original_height`. | The route takes images and resizes them itself: set the camera size, for example 1920 × 1080, so the synthetic image has that size and the resize cost is in the numbers. The route ends in BoxDecode (the Neat stage that turns detection outputs into boxes): set it so boxes are mapped to that source size. On other routes it has no effect. |
| Source height | `original_height` | `original_height` | Not set (`None`) | Height, in pixels, of the source image. Set it together with `original_width`. | Same cases as `original_width`. |
| Resize policy | `resize_mode` | `resize_mode` | Not set (`None`): the route's own resize policy, or, when the route does not resize, the decoder's default: `Stretch` for `Ssd`, `Detr`, `DetrFocal` and the Segment Anything decoders, `Letterbox` for the others | How the source image maps onto the model input in the metadata BoxDecode receives. Values are `ResizeMode::Stretch`, `Letterbox` and `Crop` (Python `pyneat.ResizeMode.*`). | Only for BoxDecode routes, when your application letterboxes or crops but the route would assume another policy. It does not change what the benchmark measures on other routes. |
| Plugin latency (deprecated) | `include_plugin_latency` | `include_plugin_latency` | `false` / `False` | Prints per-plugin latency for both passes. The report has no field for it. | Do not use it in new code. It is deprecated and will be removed. |
| Minimum pass length | `min_duration_ms` | `min_duration_ms` | `10000` (10 s) | The least time each pass, latency and throughput, runs, in milliseconds. Must be 0 or greater. `0` stops each pass after `num_samples` inputs. | Keep the default, or raise it to 30000 as the `model-benchmark` example does, when you need power and memory: Sentinel samples every 2 s, so a short window has no power sample. Set `0` for a quick smoke test on a development board when you only need latency and FPS. |
| MLA layer timings | `include_mla_layer_stats` | `include_mla_layer_stats` | `false` / `False` | After the benchmark, times each compiled MLA layer with the `mla-rt` tool and fills `mla_layer_stats`. | When a model is slower than expected and you want to see which layers take the time, for example before asking for a different compilation. Needs no other MLA workload on the board. |
| MLA stage timings | `include_mla_stage_stats` | `include_mla_stage_stats` | `false` / `False` | After the benchmark, times each compiled MLA stage run on its own (mean of 200 runs) and fills `mla_stage_stats`. | When you want the pure accelerator time of the model, without host overhead, to compare with `latency_ms`. Same requirements as layer timings. |
| Your own inputs | `inputs` | `inputs` | Empty (`[]`): synthetic inputs | Tensors to push for every sample, one per model input, in `input_specs()` order, as `run()` takes them. | When the model's cost depends on the data, such as a detector whose decode time grows with the number of objects, or a segmentation decoder. Pass a real frame, for example an image read with OpenCV. |

`benchmark` has overloads that take no `BenchmarkOptions`:

| C++ | Python | Same as | Status |
|---|---|---|---|
| `benchmark(int num_samples = 100)` | `benchmark(num_samples=100)` | Default options with `num_samples` set | Current |
| `benchmark(int num_samples, bool include_plugin_latency)` | `benchmark(num_samples=100, include_plugin_latency=False)` | Default options with both fields set | Deprecated |
| `benchmark(bool include_plugin_latency)` | not available | Default options with `include_plugin_latency` set | Deprecated |

Python binds the first two rows as one function, so `model.benchmark()` and
`model.benchmark(num_samples=500)` behave like the current C++ overload.

### What the report contains

`BenchmarkReport` (Python `pyneat.BenchmarkReport`) holds these fields. Power
and energy are zero when Sentinel took no sample; `telemetry.status` then says
why.

| Field | Meaning | Unit |
|---|---|---|
| `latency_ms` | Mean time from pushing one input to pulling its result, from the latency pass | ms |
| `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` | Percentiles of the same latency pass | ms |
| `fps` | Inferences per second from the throughput pass. For a model compiled with a batch size above 1, batches per second times the batch size | inferences/s |
| `throughput_samples`, `throughput_duration_ms` | How many inputs the throughput pass measured, and for how long | count, ms |
| `avg_power_watts`, `peak_power_watts` | Estimated mean and highest power of the whole SoM (system on module) during the throughput pass | W |
| `energy_joules` | `avg_power_watts` multiplied by the time the measured throughput needs for `num_samples` inputs. It is not the energy of the whole run | J |
| `telemetry` | Everything Sentinel recorded: `status` (`"ok"` or the reason there is no data), `samples`, temperatures, CPU usage, and memory such as `mla_memory_mib` (DMA buffers the MLA holds) and `ev74_memory_mib` (buffers of the EV74 vision processor) | per field |
| `mla_layer_stats`, `mla_stage_stats` | `status` (`"not requested"`, `"ok"` or the failure) and the per-layer or per-stage `latency_us` | µs |
| `input` | `"synthetic"` or `"provided"` (you set `inputs`) | — |

`benchmark` prints a summary to standard output but does not write a file.
`report.save(path)` writes the whole report: JSON for a `.json` path, YAML for
any other ending. It creates missing directories and replaces an existing file.

## Language differences

- C++ has `benchmark(bool include_plugin_latency)`; Python does not. In Python,
  `model.benchmark(True)` matches `num_samples` and runs with
  `num_samples = 1` and no plugin latency.
- The deprecated overloads give a C++ compiler warning (`[[deprecated]]`).
  Python gives no warning; only the docstring says `include_plugin_latency` is
  deprecated.
- Python releases the GIL while the benchmark runs, so other Python threads
  keep running during the 20 s or more it takes.
- Invalid options throw `std::runtime_error` in C++ and `RuntimeError` in
  Python. Provided `inputs` that do not match the model throw `NeatError` with
  error code `misconfig.input_shape` in C++, and `pyneat.NeatError` with an
  `error_code` attribute in Python.
- In Python the headline report fields (`latency_ms` to `input`) can be
  assigned; `telemetry`, `mla_layer_stats` and `mla_stage_stats` are
  read-only.

## Pitfalls

- **Power is zero with a short window.** With `min_duration_ms = 0` and
  100 samples, the throughput pass lasts about 0.1 s, shorter than Sentinel's
  2 s sample interval. `avg_power_watts` and `energy_joules` are `0`, and
  `telemetry.status` is `"Sentinel took no new sample during the window; make
  the window longer than the sample interval"`. Fix: keep `min_duration_ms`
  at several seconds. The 10 s default gave 5 Sentinel samples on a DevKit.
- **`num_samples` is a minimum, not the run length.** With the defaults
  (100 samples, 10 s), one run on a DevKit measured 2375 inputs in the latency
  pass and 12080 in the throughput pass. A whole call takes at least
  `2 × min_duration_ms` plus warm-up (100 inputs per pass) and model setup. Set
  `min_duration_ms = 0` when you need exactly `num_samples` inputs.
- **The `model-benchmark` example uses a 30 s window.** Its `config.yaml` and
  `main.py` default `min_duration_ms` to 30000, not Core's 10000. A plain
  `model.benchmark()` therefore runs a shorter window than the example, and
  its power average comes from fewer Sentinel samples.
- **`energy_joules` is per `num_samples`, not per run.** It answers "energy for
  `num_samples` inputs at the measured throughput". Raising `min_duration_ms`
  does not change what it covers; changing `num_samples` does.
- **Invalid options fail before anything runs.** `num_samples <= 0` throws
  `Model::benchmark: num_samples must be > 0`, a negative `min_duration_ms`
  throws `min_duration_ms must be >= 0`, and setting only one of
  `original_width` and `original_height` throws `must be provided together`.
- **Source size must match provided images.** When `inputs` holds an image the
  route resizes, `original_width` and `original_height` must be omitted or
  equal the image size, or the call throws. The image supplies its own size.
- **Source size does nothing on a tensor route.** A model whose route takes
  a model-shaped tensor and does not end in BoxDecode, such as the YOLO26n
  package's default route, accepts `original_width` and `original_height` and
  ignores them.
- **MLA timings need an idle board.** `include_mla_layer_stats` and
  `include_mla_stage_stats` need no other MLA workload. If collection fails,
  the benchmark numbers stay valid, a warning is printed, and the reason is in
  `mla_layer_stats.status` or `mla_stage_stats.status`. Check
  `status == "ok"` before reading the list.
- **Telemetry is board-wide.** Power, CPU and memory include every process on
  the board. Stop other applications before you compare power numbers.

## Used in

- [Model Benchmark](../../examples/benchmarking/model-benchmark/) (Python)

## Source

- C++ header: `include/model/Model.h`, `BenchmarkOptions`, `BenchmarkReport`,
  `MlaLayerStats`, `MlaStageStats` and `Model::benchmark`;
  `include/pipeline/TelemetryMonitor.h`, `Telemetry`
- Python binding: `python/src/module.cpp`, `BenchmarkOptions`,
  `BenchmarkReport`, `MlaLayer`, `MlaLayerStats`, `MlaStage`, `MlaStageStats`,
  `Telemetry` and `Model.benchmark`
- Core commit: `6ed16525`
