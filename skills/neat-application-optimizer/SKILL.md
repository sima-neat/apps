---
name: neat-application-optimizer
description: Make a working Neat application faster. Use to raise frames per second, serve more streams, lower latency, or remove copies in a C++ or Python application built with the Neat APIs.
---

# Neat Application Optimizer

Start from an application that produces correct output. Faster output never excuses wrong
boxes, labels, video, or metadata.

## Steps

1. Record the baseline on the target board: frames per second per stream and in total,
   counted where the results arrive, over a fixed number of frames after a warm-up. Record the
   MLA clock with it: `cat /sys/class/devfreq/5000000.mla/cur_freq`. Throughput depends on that
   clock, so compare runs only at the same clock.
2. Find what limits the run. Read where each model stage runs with `execution_plan(...)`, as
   the `neat-application-builder` skill's inspect, measure, and tune reference describes, and
   look for CPU work on every frame.
3. Change one thing from [tuning](references/tuning.md), then measure again with the same
   inputs and frame count.
4. Keep the change only when the measurement improves and the output is still correct. State
   each result with what it measured: `20 streams, 27.4 FPS per stream, MLA 1 GHz`.
