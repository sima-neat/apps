# Push and pull

## What it is

Push and pull are the two halves of moving data through a running graph. An
application pushes frames into the graph's input and pulls results out of its
outputs.

Not every graph takes pushed input. A graph whose first node is a video source
— an RTSP feed, a camera, a file reader — feeds itself, and the application
only pulls. On a graph with one input and one output, `Run::can_push()` and
`Run::can_pull()` tell you which kind of graph you built.

`pull()` has to report four different things, not two: a sample arrived, the
wait timed out, the pipeline reached end of stream, or the runtime failed.
`PullStatus` is that answer and `PullError` carries the detail when the answer
is `Error`. Python has no `PullStatus` form. On a graph with one output, a
Python application separates the middle two with `can_pull()`, which is
`false` once that output is at end of stream.

## When to use it

Pull in every application. Push only when the graph has a push input — when
the application decodes or produces frames itself, for example reading from
OpenCV or bridging two graphs together.

Use the `PullStatus` form of `pull()` whenever the graph can end or fail while
you are reading it, which is every streaming application. Use the
`std::optional<Sample>` form only for a short, bounded read where you do not
need to distinguish a timeout from end of stream.

Call `try_push()` when stale frames should be dropped. `push()` always waits,
see Pitfalls.

Use `Graph::run(...)` instead of either when you want one frame in and one
result out.

## Minimal example

C++

```cpp
simaai::neat::Sample detections;
simaai::neat::PullError err;
const auto status = run.pull("detections", /*timeout_ms=*/0, detections, &err);
if (status == simaai::neat::PullStatus::Closed) {
  throw std::runtime_error("output closed: " + run.last_error());
}
if (status == simaai::neat::PullStatus::Timeout) { return; }
if (status == simaai::neat::PullStatus::Error) {
  throw std::runtime_error(err.message);
}
```

Python

```python
sample = run.pull("detections", 50)
if sample is None:
    # None means timed out or closed. With one output, can_pull() tells which.
    if not run.can_pull():
        raise RuntimeError(f"output closed: {run.last_error()}")
    return
```

Complete flow: [high-density-multi-stream-object-detector](../../../examples/object-detection/high-density-multi-stream-object-detector/)

## Options

These are calls rather than an option struct, so the table lists the calls and
their arguments. `RunOptions` is on [run.md](run.md).

### Pushing

| Call | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Push a tensor list | `push(const TensorList&)` | `push(input)` / `push_tensors(inputs)` | — | Pushes one frame as a list of tensors to the default input port. **Waits** for queue space under every overflow policy. Returns `false` when the input is closed, the run has stopped, or a graph input timed out; work already accepted stays available to drain. See Pitfalls. | The usual push for an application that builds its own tensor, for example from a `cv::Mat`, and can tolerate waiting when the graph falls behind. |
| Push to a named port | `push(std::string_view, const TensorList&)` | `push(name, input)` | — | Same, to a specific named input port. | Multi-input models, where each port takes a different tensor. |
| Push images | `push(const std::vector<cv::Mat>&)` | not available (pass arrays to `push`) | — | C++ convenience overload that wraps `cv::Mat`s as tensors. | Saves building the `TensorList` yourself when you already hold `cv::Mat`s. |
| Push a sample | `push(const Sample&)` | `push_samples(inputs)` | — | Pushes a typed `Sample`, which can carry metadata and a bundle of payloads. | When you need to set `port_name`, timestamps or attributes on the frame, or when bridging a pulled sample into another graph. |
| Push without blocking | `try_push(...)` | `try_push(...)` / `try_push_tensors` / `try_push_samples` | — | Same payload forms, but applies the resolved `overflow_policy` instead of waiting: `Block` returns `false`, `DropIncoming` drops your frame and returns `false`, `KeepLatest` evicts the oldest queued frame and returns `true`. `Auto` follows the preset, see [run.md](run.md). | **The only call that reads `overflow_policy`.** Use it for a live source where the calling thread must not stall. No callback reports these drops. |
| Copy on push | not an argument (`RunAdvancedOptions::copy_input`) | `copy=False` keyword | `false` | Copies the pushed buffer instead of referencing it. | Set it when the buffer you push does not outlive the call. In Python it is a per-call keyword; in C++ it is a run-wide option. |
| Declare layout / format | not available | `layout=None`, `image_format=None` keywords | `None` (inferred) | Tells the binding how to interpret the array you pass. | Set `image_format` when the array is an image whose pixel format cannot be inferred, for example BGR vs RGB. |
| Close the input | `close_input()` | `close_input()` | — | Signals end of input; the graph drains and then its outputs report `Closed`. | Call it at shutdown so pullers see a clean end of stream rather than a timeout. |

`push_holder` and `try_push_holder` are marked internal in the header and are
not public API; no shipped example uses them.

### Port names and readiness

| Call | C++ | Python | What it does | Why and when to use it |
|---|---|---|---|---|
| Can the run take pushes | `can_push()` | `can_push()` | `true` while the input side accepts pushes, `false` once it is closed. It checks only the default input, so on a graph with more than one input it is always `false`. | On a graph with one input, branch on it once after `build()` to tell a push-driven graph from a source-driven one. |
| Can the run yield samples | `can_pull()` | `can_pull()` | `true` while the output side may still produce a sample, `false` at end of stream. It checks only the default output, so on a graph with more than one output it is always `false`. | **The end-of-stream check on a graph with one output.** In Python it is then the only way to tell a `None` from `pull()` apart from a timeout. |
| Input port names | `input_names()` | `input_names()` | The names accepted by `push(name, ...)`. Empty for an unnamed or linear run. | Call it once after `build()` to learn the ports a multi-input graph exposes, instead of hard-coding the strings. |
| Output port names | `output_names()` | `output_names()` | The names accepted by `pull(name, ...)`. Empty for an unnamed or linear run. | The same for outputs, and the natural way to drive a drain loop over every output of a multi-stream graph. |

### Pulling

| Call | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Pull with a status | `pull(int, Sample&, PullError*)` and `pull(name, int, Sample&, PullError*)` | not available | `timeout_ms` required | Fills `out` and returns `PullStatus`. | The only form that distinguishes a timeout from end of stream and from a runtime error. Use it in any loop that must shut down cleanly. |
| Pull a sample | `pull(int timeout_ms = -1)` → `std::optional<Sample>` | `pull(timeout_ms=-1)` → `Sample \| None` | `-1` | Returns the sample, or nothing on timeout. | The short form. In Python it is the only form. |
| Pull from a named output | `pull(name, timeout_ms = -1)` | `pull(name, timeout_ms=-1)` | `-1` | Reads one named output. | Any graph with more than one output, which includes every multi-stream application. |
| Pull tensors | `pull_tensors(timeout_ms = -1)` and the named variant | `pull_tensors(...)` | `-1` | Pulls and flattens straight to a `TensorList`, skipping the `Sample`. | When you do not need the frame metadata — stream id, timestamps, attributes. No shipped example uses it, because multi-stream applications all need the metadata. |
| Pull samples | `pull_samples(timeout_ms = -1)` and the named variant | `pull_samples(...)` | `-1` | Like `pull()` but returns a `Sample` directly instead of an optional. | When an empty result is already an error for your caller. |
| One-shot | `run(inputs, timeout_ms = -1)` | `run(input, timeout_ms=-1, copy=False, ...)` | `-1` | Pushes and pulls in one call. | A single inference where you do not want to manage a loop. |
| Blocking forever | `timeout_ms = -1` | `timeout_ms=-1` | `-1` | A negative timeout blocks until a sample arrives. | Use `0` to poll without waiting, as the multi-stream detector does when draining several outputs in turn. |

### `PullStatus`

| Value | C++ | Python | What it means |
|---|---|---|---|
| Ok | `PullStatus::Ok` | `pyneat.PullStatus.Ok` | A sample is in the output parameter. |
| Timeout | `PullStatus::Timeout` | `pyneat.PullStatus.Timeout` | The wait elapsed and no sample arrived. The graph is still running; poll again. |
| Closed | `PullStatus::Closed` | `pyneat.PullStatus.Closed` | The pipeline reached end of stream. No further sample will ever arrive; stop reading this output. |
| Error | `PullStatus::Error` | `pyneat.PullStatus.Error` | A runtime error occurred. Read the `PullError`. |

The enum is bound in Python, but no Python method returns it — see Language
differences.

### `PullError`

| Field | C++ | Python | Default | What it does |
|---|---|---|---|---|
| Message | `message` | `message` | empty | Human-readable error, usually prefixed with `[code]`. |
| Code | `code` | `code` | empty | Canonical machine-triage code from `pipeline/ErrorCodes.h`. Use this for branching, not the message text. |
| Report | `report` | `report` | unset | Full structured `GraphReport` for severe runtime or plugin failures. |

## Language differences

**Python's `pull()` does not return a status.** The `PullStatus`-returning
overload is not bound; Python's `pull()` returns `Sample | None` and `None`
means both "nothing arrived yet" and "the pipeline has ended". On a graph
with one output, call `run.can_pull()` after a `None` to tell them apart:
`false` means end of stream, so stop reading that output. On a graph with more
than one output, `can_pull()` is always `false`, and Python has no call that
tells a timeout from end of stream for one output. `run.running()` does not
fill that gap: it reports the pipeline state, which can still be playing while
an output is drained. The shipped Python examples use `running()` plus
`last_error()`; see `pull_result_has_sample` in
[fastflow-anomaly-detector](../../../examples/anomaly-detection/fastflow-anomaly-detector/src/python/main.py).
A C++ application should use the `PullStatus` form rather than either.

**Python rejects a bare `Tensor` or `Sample` in `push()`.** The binding calls
`reject_single_tensor_or_sample`, so `run.push(tensor)` raises; pass a sequence,
`run.push([tensor])`. C++ has a single-`Sample` overload and accepts it
directly.

**Python has typed push variants that C++ gets by overload.** `push_tensors`,
`push_samples`, `try_push_tensors` and `try_push_samples` exist because Python
cannot overload on argument type; the generic `push` sniffs the argument
instead.

**Python's `push` takes per-call keywords that C++ does not have.** `copy`,
`layout` and `image_format` are arguments in Python. In C++ the equivalent of
`copy` is the run-wide `RunAdvancedOptions::copy_input`.

**`push_holder` and `try_push_holder` are C++ only**, and internal in both.

## Pitfalls

**Treating `Closed` as a timeout loops forever.** Once an output reports
`Closed` it will never produce another sample. A loop that only breaks on
`Timeout` spins at full speed after end of stream. Symptom: a process that
pegs a core and never exits after the video ends. Fix: break or throw on
`Closed`, and read `run.last_error()` to see whether the close was clean or the
consequence of a failure.

**In Python, a `None` from `pull()` at shutdown looks like an ordinary
timeout.** The same bug as above, but the type system does not help you.
Symptom: a Python application that polls forever after the video ends. Fix:
on a graph with one output, check `run.can_pull()` after every `None` and stop
when it is `false`. On a graph with more than one output, `can_pull()` is
always `false`; see Language differences.

**`pull()` with the default `timeout_ms = -1` blocks forever.** In a loop that
drains several outputs, one quiet output blocks the others. Fix: poll with
`timeout_ms = 0` per output and sleep once per round, which is what the
high-density detector does.

**Forgetting `close_input()` turns a clean shutdown into a timeout.** Without
it, pullers cannot tell the difference between "the producer is done" and "the
producer is slow". Call it before joining the threads that pull.

**`overflow_policy` has no effect on `push()`.** Every `push()` overload waits
for queue space; the policy is read only on the `try_push()` path. Symptom: an
application sets `KeepLatest` for a live camera, calls `push()`, and gets
ingress backpressure instead of frame dropping — no error, no drop counter, and
the latency appears upstream of Neat in the capture or network stack, where it
cannot be attributed to this setting. Fix: call `try_push()`.

**`push()` returning `false` does not mean the queue was full.** Because it
waits, `false` means the input has been closed, the run has stopped, or a graph
input timed out; work already accepted is still there to drain. The shipped
examples throw on `false`, which is correct for their case. A `false` from
`try_push()` is the different, ordinary case: the queue was full under `Block`
or `DropIncoming`, and the frame was not accepted.

## Used in

- [high-density-multi-stream-object-detector](../../../examples/object-detection/high-density-multi-stream-object-detector/) — the four-way `PullStatus` loop over many named outputs
- [multi-stream-pose-estimator](../../../examples/pose-estimation/multi-stream-pose-estimator/) — Python `pull(name, 0)` drain loop plus `pull(name, 50)`
- [superpoint-feature-extractor](../../../examples/feature-extraction/superpoint-feature-extractor/) — pulls NV12 frames from a named `Output` and runs each one through a `Model::Runner`
- [fastflow-anomaly-detector](../../../examples/anomaly-detection/fastflow-anomaly-detector/) — the Python `pull_result_has_sample` workaround
- [rfdetr-detection-segmentation](../../../examples/object-detection/rfdetr-detection-segmentation/) — `close_input()` at shutdown, both languages

## Source

- C++ header: `include/pipeline/Run.h` (`Run::push`, `try_push`, `pull`,
  `pull_tensors`, `pull_samples`, `run`, `close_input`, `can_push`, `can_pull`,
  `input_names`, `output_names`), `include/pipeline/GraphOptions.h`
  (`PullStatus`, `PullError`)
- Python binding: `python/src/module.cpp`, `nb::class_<Run>`,
  `nb::enum_<PullStatus>`, `nb::class_<PullError>`
