---
name: neat-ros2-application-builder
description: Build, integrate, debug and validate ROS 2 applications using the public SiMa Neat Library APIs on Modalix. Use for ROS node and package design, camera-to-tensor transport, asynchronous inference, model controls, Insight output and performance comparisons. Model compilation and unrelated ROS administration are outside this skill.
---

# Neat ROS 2 Application Builder

Create understandable ROS nodes around a verified Neat model pipeline. Keep model
initialization, input preparation, inference, subscriptions and publication easy
to locate. Preserve the user's architecture and message contracts; an existing
ROS camera or application does not need to be replaced by a new demo.

This skill is self-contained. It assumes no particular checkout, board address,
model, credentials or installed companion skill. Use `neat-application-builder`
for additional Neat API guidance when available. Follow the installed SDK and
ROS contracts rather than assuming an API exists because a newer example uses it.

## Start with the requested change

Inspect the affected package, launch file, config and nearest compatible native
example. Establish only the facts needed for the task:

- Target Neat/ROS versions, interpreter ABI or C++ toolchain, and build location.
- Compiled model or GenAI artifact, preprocessing, output layout and precision.
- Source owner, encoding, geometry, timestamps and topic QoS.
- Complete-frame processing versus live freshness; bounded pending/in-flight work.
- Required outputs and the numerical, latency or throughput acceptance criteria.

For a bug report, trace the reachable path at the reviewed revision before editing.
For optimization, identify the measured boundary and keep a before/after baseline.
Infer available details from files and tools before asking the user. Do not expand
a small fix into a framework rewrite or require a complete board sweep for it.

## Read only the relevant reference

| Task | Reference |
| --- | --- |
| Install/setup, colcon, SDK versus board, copied packages | [Environment and build](references/environment-and-build.md) |
| New package, node roles, topics, YAML, source modes | [Package and node design](references/package-and-nodes.md) |
| Decoder planes, tensors, zero-copy claims, image conversions | [Images and memory](references/images-and-memory.md) |
| Async workers, correlation, QoS, parameters, standby and shutdown | [Runtime and control](references/runtime-and-control.md) |
| Insight, result sinks, on-demand/periodic VLM, HTTP/TLS | [Insight and GenAI](references/insight-and-genai.md) |
| Accuracy, FPS/latency, regression checks, CI diagnosis | [Validation and troubleshooting](references/validation-and-troubleshooting.md) |

## Decisions that preserve correctness

1. **Keep the model contract.** Match color order, resize/interpolation, bounds,
   normalization, layout, decoding and thresholds. Similar-looking overlays do
   not establish numerical parity. If the native reference is itself wrong,
   separate its correction from a transport optimization.
2. **Make node boundaries deliberate.** Image-topic composition is useful for
   reuse. A combined decoder/inference Neat graph can avoid image transport in
   a high-throughput mode. Choose from requirements and measurements; do not
   equate one node per model stage with better performance.
3. **Retain each admitted frame.** Keep its owners, identity, geometry and model
   generation until completion. Validate correlation before publishing; never
   attach the latest image's header to an older inference result.
4. **Bound work and waiting.** Keep expensive operations off image callbacks.
   Specify admission, overflow, timeout and failure behavior. Latest-frame
   dropping before admission is different from dropping admitted results.
5. **Remove only demonstrated redundant work.** A NumPy view, Neat zero-copy
   output, DDS transport and accelerator memory are different boundaries. Keep
   required ownership/transfer copies; measure before claiming zero-copy or FPS.
6. **Make controls truthful.** Disabling inference must stop new inference work,
   not merely hide overlays. Validate parameter changes before applying them;
   invalidate stale results when a model or enabled state changes.

## Finish with evidence and a usable example

For new or substantially changed apps, provide an app-local README with a node
map, topic/parameter tables and exact build, launch, source-mode and Insight
steps. State which settings are startup-only, how paths resolve, how to see an
answer/result, and how to stop the app. Preserve existing documented commands
when changing an established package.

Test the behavior changed. Use an authorized target for hardware/runtime claims;
use isolated workspaces and bounded test processes. Report code revision/hashes,
what passed, measured boundaries and what remains unverified. A local unit test,
a build, a source benchmark and an end-to-end run prove different things.

Do not install/upgrade a board runtime, stop other workloads, change SDK pins,
commit or publish merely because this skill was invoked. Complete authorized
application work and identify any remaining platform or release-integration step.
