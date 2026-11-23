# Validation, performance and troubleshooting

## Scale validation to the change

A build verifies packaging, not useful inference. A unit test verifies its stated
contract, not hardware throughput. A copied source tree is not proof that the
running installed node contains those files. Record revision/hashes, config,
model identity and target runtime for hardware claims.

| Change | Focused checks |
| --- | --- |
| Config/resource paths | Load each affected profile from the documented directory and installed layout |
| Node/package layout | Independent colcon build, installed imports/resources, documented launch |
| Packing/ownership | Exact pixels/headers; padded planes, supported formats, reuse/lifetime checks; real decoder frames |
| Preprocessing/model route | Actual input tensors and outputs versus a matched reference; non-square and oversized inputs |
| Scheduling/identity | Alternating distinct frames, bounded overload, empty results, drops, duplicate/reordered IDs |
| Controls/lifecycle | Each supported mode, rejected atomic updates, standby/resume, model failure and shutdown/restart |
| Visualization/GenAI | Matched visible output; receiving Insight video; streamed chunks/final answer and errors |
| Performance | Comparable repeated before/after runs at the boundary being claimed |

Use existing tests where they cover the contract. When identical app-local
helpers are changed, apply the same focused test to every copy rather than
assuming one tested file proves all packages. Test padding and ownership behavior,
not just whether a particular implementation string appears. Keep test-only data
and instrumentation out of production node flows.

## Correctness before speed

Compare the same model/variant, inputs, preprocessing, thresholds and decoding.
For a transport-only change, require byte-identical input pixels and, where
deterministic, numerical outputs. If a tolerance is needed, establish and explain
it before accepting the result. Do not widen tolerances to hide an unexplained
mismatch. Separate model stochasticity from transport corruption.

Include positive and empty scenes, boundary geometry, supported formats and
retained-frame reuse. Check finite outputs, valid boxes/masks/depth, labels,
units and correlation. Relative monocular depth is not calibrated metric depth.
Numerical equality establishes equivalence to the reference; visual plausibility
or an unlabeled video does not establish task accuracy. Use labeled evaluation
when claiming an accuracy metric. If the reference is incorrect, report that
separately instead of silently changing normalization during a parity fix.

## Name the timing boundary

| Metric | Meaning |
| --- | --- |
| Source FPS | Actual frames produced/published per elapsed monotonic time |
| Received FPS | Callbacks observed by that subscriber, with its QoS/load |
| Completed output FPS | Completed predictions/publications per wall-time interval |
| Input preparation time | Conversion/packing/tensor construction included in that timer |
| Async result latency | Admission or source receipt to a matched completion/publication |
| Insight FPS | Frames received/decoded/displayed at the measured receiver boundary |
| VLM first-chunk / final latency | Request acceptance to first emitted text / completed answer |

The reciprocal of asynchronous per-frame latency is not pipeline throughput.
Neither a faster helper nor fewer copies proves lower end-to-end latency. A
source-capped run does not establish maximum inference capacity. An observer
subscribing to several full-resolution images can itself consume CPU/bandwidth
and miss best-effort samples; distinguish its rate from the application's counters.

Keep model, precision, source geometry/content/rate, queue limits, QoS, executor,
rendering, power state and competing workload comparable. Separate build/load,
warmup, steady-state and drain time. Alternate before/after order where useful,
repeat enough windows to expose variability, and report median/spread and counts.
Do not invent an acceptable slowdown. If the user requires no measurable slowdown,
use repeated measurements at the agreed boundary and report uncertainty honestly.

For copy changes, measure allocation and conversion together with the removed
copy. Preserve an owned-buffer control and compare outputs independently of the
timed loop. Avoid claiming universal savings from one resolution, codec or model.
Use representative real inputs after synthetic layout tests.

## Locate the first failed boundary

| Observation | First checks |
| --- | --- |
| `ros2` or `colcon` missing | Sourced installation, PATH, tool packaging and interpreter; not model code |
| Launch cannot find file/helper | Working directory, package share, installed resources, selected overlay |
| No image callbacks | Publisher health, topic/remap, QoS compatibility, domain and payload limits |
| File input rejected on first frame | Actual published encoding versus inference contract; reachable source path |
| RTSP stops after changing video | Stream codec/geometry/rate, decoder status and negotiated caps; not just configured FPS |
| Input rate low | Source capacity, decode/packing cost, DDS/observer loss, callback backlog |
| Inputs admitted but no completions | First runtime error, graph identity, queue deadlock, incompatible tensor/layout |
| Depth/masks look wrong | Preprocessing, normalization, output orientation, aspect ratio and display range |
| VLM ready but answer seems absent | Correct topic/schema, request ID, retained QoS, CLI truncation, chunks and error state |
| VLM slow only with other models | Shared-resource contention; honor the selected concurrency policy |
| Model load fails | First error, artifact compatibility, free space/extraction directory and actual API support |
| CI red after an app edit | Exact failing job/step/suite and branch diff; distinguish build, native runtime, ROS and publish jobs |

A wrapper traceback propagating a failed test is not the root cause. Inspect its
preceding assertion and artifacts. A native test failure does not establish a ROS
SDK mismatch. Do not bypass a failing check or change runtime pins on speculation.
If logs need authorization, request the failing excerpt rather than fabricating a
cause. A rerun can test intermittency; it does not explain or permanently fix it.

## Handoff and limits

Report what changed, why it is safe, the exact checks performed, reproducible
commands and remaining gaps. Distinguish simulated NV16 packing from a physical
USB camera run, RTSP from MIPI, sender output from browser verification, and a
normal stop from interruption during initialization. Do not mark an app ready
solely because the process starts or the latest test subset passes.

Keep bounded test logs/snapshots outside the deliverable unless they are intended
public fixtures. Stop only test-owned processes. Preserve failed-run evidence
before a rerun overwrites it. Do not claim a staged change has been committed,
published or integrated into the SDK installer unless that action was performed.
