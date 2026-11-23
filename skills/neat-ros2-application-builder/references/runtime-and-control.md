# Runtime, QoS and controls

## Admission and overlap

For live input, the subscription callback should retain a message reference,
update bounded state and wake a worker. Keep decoding/conversion, graph building,
model waits and rendering outside the callback and outside the queue lock.

Choose the processing contract explicitly:

| Contract | Admission policy | What to count |
| --- | --- | --- |
| Fresh live output | Bounded latest pending frame(s) | Received, replaced before admission, admitted, completed |
| Every frame | Bounded FIFO plus source pacing/backpressure | Sent, acknowledged, admitted, completed, failures |
| On-demand request | Bounded request slot/queue; explicit busy response | Accepted, rejected, completed, failed/cancelled |

Bound pending work, admitted work, output queues, frame caches and model-stage
handoffs. Per-camera bounds alone do not bound the total for many cameras. Once
admitted, preserve outputs needed for correlation or explicitly account for their
cancellation. Do not silently drop a completed result and shift FIFO identities.

Use the installed Neat push/pull or equivalent asynchronous API to overlap work
when required. A submit-then-blocking-wait loop measures serial round trips.
Separate submission and completion workers can be appropriate if one coordinator
would block on a full queue that only it can drain. Keep worker counts small and
measured; a MultiThreadedExecutor or more Python threads is not an automatic fix.
Idle workers should wait on conditions/events. Bound native waits and preserve
the first real error; do not spin in retry loops on a failed runner.

Warmup must finish before measured result accounting starts. Build once for a
compatible input contract, not once per frame. Geometry/format changes require a
supported rebuild or a clear rejection. If a split model passes intermediate
tensors between stages, keep their owners and frame context alive and bound both
stages; do not round-trip features through CPU memory without a need.

## Correlate before publishing

An admitted context should identify the source camera, timestamp/request ID,
visible geometry, inverse preprocessing transform, generation and backing owners.
Check the built graph's actual identity propagation; do not assume arbitrary
attributes, frame IDs or starting sequence numbers survive every route.

Use a supported explicit correlation key where possible. FIFO requires a proven
one-input/one-output ordered, lossless admitted path, including intermediate
queues. `Block` overflow alone is not proof of that contract. Treat unknown,
duplicate or mismatched results as errors rather than silently relabelling them.
Publish empty detections with the correct source header too.

A header timestamp is not always unique: bag loops, clock resets and repeated
stamps need a sequence/replay epoch or an explicit drain policy. Different ROS
topics can arrive in a different order; an observer must join using identity with
a bounded wait/cache, not insist that its source callback ran first. A best-effort
observer may miss source images that the inference node received; account for
observer coverage instead of manufacturing a model-correlation failure.

## QoS and readiness

Inspect `ros2 topic info --verbose <topic>` before changing queue sizes. A
best-effort publisher cannot satisfy a reliable subscription; a best-effort
subscription can receive from either reliability kind. Durability must also
match. Late subscribers need compatible transient-local settings to receive
retained state. See the [ROS QoS compatibility tables](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst).

Reliable transport does not provide application-level completion acknowledgement,
undo latest-frame replacement, or guarantee every offline sample is processed.
Wait for required endpoints and model/source readiness before finite replay.
Keep startup waits bounded and report which prerequisite is missing. Inspect the
installed parameter client API instead of guessing its readiness method names.

## Runtime parameters and model modes

Declare types, ranges and defaults. Mark artifact paths, input geometry and graph
options read-only at runtime unless a safe rebuild transition is implemented.
Keep user-facing YAML, launch overrides and actual live parameters consistent.

Parameter validation must be side-effect-free: a later validator can reject the
same request. Apply committed updates in the supported post-set callback or an
equivalent serialized control path. For coupled changes, use an atomic request
and validate the proposed combined state. A post-set handler cannot roll back a
committed parameter by returning an error. See the [ROS parameter callback contract](https://github.com/ros2/ros2_documentation/blob/jazzy/source/Concepts/Basic/About-Parameters.rst).

Define `enabled`, standby and overlay mode separately:

- Disabled inference stops admission/submission and clears or accounts for pending
  work; it must not merely suppress publication. Document whether loaded models
  remain resident and what happens to already-admitted frames.
- Display-only detection/segmentation controls can reuse one model that produces
  both outputs. Changing an overlay need not reload or disable model inference.
- Concurrent models share accelerator/CPU/memory resources. Do not pause another
  pipeline for a VLM request unless the user selected that scheduling policy.

For enable/model transitions, use a generation counter or equivalent mechanism
that prevents obsolete results from appearing after the transition. Serialize
committed state changes with the final publication decision; checking generation
and then publishing without synchronization leaves a race. Avoid holding that
lock during model building or blocking inference. Report loading, ready/standby,
active and error states accurately; keep desired state distinct from readiness.

## Shutdown and recovery

Stop admission; wake idle workers; bound or interrupt waits using supported APIs;
close owned runners; join workers; then destroy ROS entities. Resolve outstanding
requests with documented cancellation/failure semantics. Launch event handlers
must recognize shutdown already in progress instead of triggering repeated stops.

Exercise shutdown while idle, running, loading a model and disconnected from the
source when those paths change. A normal post-initialization Ctrl+C test does not
prove startup cancellation. A Python join timeout cannot interrupt a native call
by itself; diagnose forced termination rather than reporting it as clean shutdown.

After a source outage, preserve model state where supported and implement only
the promised reconnect policy. A new source epoch must not mix old cached frames
with new results. Bound validation retries and stop on repeated runtime failure;
do not change firmware, kill unrelated processes or delete caches as app recovery.
