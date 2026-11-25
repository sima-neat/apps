---
name: sima-application-optimizer
description: Speed up a working Neat application. Use to measure its throughput or latency, raise frames per second or the stream count, lower latency, or remove copies in a C++ or Python application built with the Neat APIs.
---

# SiMa Application Optimizer

Start from an application that produces correct output, and keep it correct. A faster run with
wrong boxes, labels, video, or metadata is a regression. Building or fixing the application
belongs to the `sima-application-builder` skill.

## Steps

1. **Record the baseline** on the target board. Read the MLA clock first:
   `cat /sys/class/devfreq/5000000.mla/cur_freq`. Throughput depends on it, so compare runs only
   at the same clock. Count results per stream and in total where they arrive, after a warm-up,
   over a fixed window. Without a project rule, discard the first 10 seconds and count for the
   next 60 seconds, and use the same two values for every later run. In Python, count each
   result pulled from the `Run` by its `sample.stream_id` (a `collections.Counter`), and divide
   by the window measured with `time.monotonic()`. Done when the baseline names the stream
   count, the per-stream and total frames per second, the warm-up and window, and the MLA
   clock.
   - `Run::start_measurement()` returns a `MeasureScope` for one window of the application's own
     loop. `MeasureOptions` `duration_ms`, `warmup_ms` and `timeout_ms` have no effect
     (core#1471), so push the warm-up frames before you start it and call `stop()` yourself
     when the window ends. For board power, start a `TelemetryMonitor` and stop it around the
     same window.
     `TelemetryMonitor` replaces the deprecated `PowerMonitor` and `RunOptions::power_monitor`.
   - Telemetry comes from Sentinel and covers the whole board. Quote power only when `status`
     is `ok` and the metric reports samples.

2. **Find the limit.** Done when you can name the stage or resource that limits the run and
   the measurement that shows it.
   - **Model or application:** run `Model::benchmark(BenchmarkOptions)` (`model.benchmark(...)`
     in Python) with the application's `Model::Options` and real `inputs`, such as decoded
     frames from the stream, and read `BenchmarkReport::fps`. When the application's total rate
     is close to that `fps`, the model is the limit, and only a smaller model or more boards
     raise it. When the total is well below it, the application is the limit: decoding, copies,
     queues, CPU work, or the senders. `min_duration_ms = 0` gives a quick smoke run, and
     `save()` keeps the report.
   - A source that sends fewer frames per second than the target caps the result. Check the
     source rate first.
   - Keeping up with the source is not a capacity measurement. To find capacity, feed more than
     the application can process and count what arrives. The same URL repeated does not prove
     capacity for that many distinct streams, so state which one you measured.
   - One model has one set of hardware lanes. More model graphs or more processes add queues,
     not capacity. Divide the measured total rate by the stream count to size the application,
     and split streams across boards when the total is too low.
   - Count results where they arrive. A count of results the application sent overstates the
     rate when a nonblocking `MetadataSender` drops messages.
   - `Model::execution_plan(...)` takes the same arguments as `build(...)` and builds on the
     device. Its `to_json()` lists each node with the processor it runs on (A65, EV74, or MLA),
     its kernel, the resolved Preproc and SimaBoxDecode settings, and the tensors between
     nodes. Look for work on the A65 that the EV74 could do and for CPU conversions between
     stages. Core's `docs/develop-apps/advanced-concepts/data-model-contracts/memory_model.md`
     describes where buffers live between stages.
   - Read per-thread CPU use with `top -H` to find CPU work on every frame.

3. **Change one thing** from [tuning](references/tuning.md) that addresses that limit, then
   measure again with the same inputs, warm-up, window, and MLA clock. Done when the new
   measurement exists.

4. **Keep or revert.** Keep the change only when the measurement improves and the output is
   still correct. Check correctness on a recorded file or one repeated source, so every run
   sees the same frames: compare the boxes per frame and their labels with the baseline run
   over the same frames, and look at the overlays in Insight on every stream. State each result
   with what it measured: `20 streams, 27.4 FPS per stream, MLA 1 GHz`. Done when every kept
   change has its before and after measurement and its correctness comparison, and every other
   change is undone.
