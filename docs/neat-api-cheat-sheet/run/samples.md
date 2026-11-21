# Sample

## What it is

A `Sample` is one frame's worth of data as it crosses the boundary between the
graph and the application. It is what `Run::pull()` returns and what
`Run::push()` accepts.

A `Sample` is a tagged union. `kind` says which payload field is meaningful:
one tensor, a flat list of tensors, or a recursive bundle of samples. The shape
you get is decided by the model's output topology, not by anything the
application sets.

Alongside the payload, a `Sample` carries the per-frame metadata an application
needs to make sense of it: which stream it came from, which logical output it
is, its frame id and timestamps, and any attributes a source captured.

## When to use it

You use `Sample` whenever you pull, unless you pull straight to tensors with
`pull_tensors()` and genuinely do not need the metadata. Every multi-stream
application needs it, because `stream_id` and `stream_label` are the only way
to know which camera a detection belongs to.

Push a `Sample` rather than a `TensorList` when you need to set `port_name`,
timestamps, or attributes on the frame you are pushing, or when you are
forwarding a sample you pulled from another graph.

Use `tensors_from_sample()` in C++ when you want the numbers and do not care
about the shape they arrived in — it flattens all three kinds, recursively.

## Minimal example

C++

```cpp
const auto status = run.pull("detections", 0, sample, &err);
if (status != simaai::neat::PullStatus::Ok) { return {}; } // nothing to read yet
// Bundles carry one field per logical output; pick yours by label.
for (const simaai::neat::Sample& field : sample.fields) {
  if (field.stream_label == "frame") {
    return simaai::neat::tensors_from_sample(field).front();
  }
}
return simaai::neat::tensors_from_sample(sample).front();
```

Python

```python
sample = run.pull("detections", 0)
if sample is None:  # a timeout or end of stream
    return None
# tensors_from_sample is not bound; walk the union yourself.
if sample.kind == pyneat.SampleKind.Bundle:
    sample = next(f for f in sample.fields if f.stream_label == "frame")
if sample.kind == pyneat.SampleKind.TensorSet:
    tensor = sample.tensors[0]
else:
    tensor = sample.tensor
```

Complete flow: [fastflow-anomaly-detector](../../../examples/anomaly-detection/fastflow-anomaly-detector/)

## Options

`Sample` is a payload, not an option struct, so the table lists its fields.

### `SampleKind`

| Value | C++ | Python | Which field is set | When you get it |
|---|---|---|---|---|
| Tensor | `SampleKind::Tensor` | `pyneat.SampleKind.Tensor` | `tensor` | A model with one output that is a single tensor. |
| TensorSet | `SampleKind::TensorSet` | `pyneat.SampleKind.TensorSet` | `tensors` | Several tensors at one logical output — the common detector case, where boxes, scores and classes arrive together. |
| Bundle | `SampleKind::Bundle` | `pyneat.SampleKind.Bundle` | `fields` | A model with several logical outputs, or a graph that combines branches. Recursive: a field can itself be a Bundle. |
| Unknown | `SampleKind::Unknown` | `pyneat.SampleKind.Unknown` | none | Default-constructed or uninitialised. Treat it as empty. |

### Payload fields

| Field | C++ | Python | Default | What it does | Why and when to use it |
|---|---|---|---|---|---|
| Kind | `kind` | `kind` | `SampleKind::Unknown` | Discriminator for the union. | Always branch on this before reading a payload field. |
| Single tensor | `tensor` | `tensor` | unset | Set when `kind == Tensor`. | — |
| Tensor list | `tensors` | `tensors` | empty | Set when `kind == TensorSet`. | — |
| Bundle fields | `fields` | `fields` | empty | Set when `kind == Bundle`; each entry is a full `Sample`. | Walk it to find the logical output you want, usually by `stream_label`. |
| Ownership | `owned` | `owned` | `true` | `false` means the framework holds a borrowed reference to the producer's buffer. | Read it, do not set it. A borrowed sample must not outlive the producer's pool — see [tensors.md](tensors.md). |

### Metadata fields

| Field | C++ | Python | Default | What it does | Why and when to use it |
|---|---|---|---|---|---|
| Stream id | `stream_id` | `stream_id` | empty | Stream identifier in a multi-stream pipeline. | The routing key in any application with more than one camera. |
| Stream label | `stream_label` | `stream_label` | empty | Human-readable stream or field label. | How the shipped examples pick a bundle field — `"frame"`, `"segments"`. |
| Port name | `port_name` | `port_name` | empty | Ingress port name for multi-input models. | Set it when pushing to a named port. |
| Frame id | `frame_id` | `frame_id` | `-1` | Source-assigned frame id, when the source carries one. | Correlating a result with the source frame. |
| Logical output index | `logical_output_index` | `logical_output_index` | `-1` | Which logical model output this sample is. | An alternative to `stream_label` for picking a bundle field. |
| Input sequence | `input_seq` | `input_seq` | `-1` | Sequence number assigned at push time. | The way to match a pulled result back to the frame you pushed in an async run. |
| Original input sequence | `orig_input_seq` | `orig_input_seq` | `-1` | The pre-renumbering sequence when the frame passed through a sub-pipeline. | Tracing a frame across a nested graph. |
| Presentation timestamp | `pts_ns` | `pts_ns` | `-1` | Presentation timestamp in nanoseconds. | Latency measurement and A/V sync. `-1` means absent. |
| Decode timestamp | `dts_ns` | `dts_ns` | `-1` | Decode timestamp in nanoseconds. | Rarely needed by applications. |
| Duration | `duration_ns` | `duration_ns` | `-1` | Sample duration in nanoseconds. | Rate calculations on encoded sources. |
| Caps string | `caps_string` | `caps_string` | empty | The GStreamer caps string of the source buffer. | Diagnostics when a payload is not the shape you expected. |
| Payload type | `payload_type` | `payload_type` | `PayloadType::Auto` | Semantic payload family: `Auto`, `Image`, `Tensor`, `Encoded`. | Set it when pushing an encoded or image payload whose family cannot be inferred. |
| Media type | `media_type` | `media_type` | empty | MIME-style type, for example `"video/x-raw"`. | Diagnostics, and routing encoded payloads. |
| Payload tag | `payload_tag` | `payload_tag` | empty | Subformat tag, for example `"NV12"`, `"FP32"`, `"INT8"`. | Checking what you actually received before decoding it. |
| Legacy format tag | `format` | `format` | empty | Deprecated spelling of `payload_tag`. | Do not use in new code. |
| Legacy output index | `output_index` | `output_index` | `-1` | Deprecated alias for `logical_output_index`. | Do not use in new code. |
| Attributes | `attributes` | `attributes` | empty | Per-frame string-to-string map captured by a source. | Carrying request metadata through the pipeline. Keys are lowercased by capturing producers, ordered lexicographically, and must be valid UTF-8 with no NUL bytes. The framework copies them and never parses or merges them. |
| Memory index | `memory_index` | `memory_index` | `-1` | Underlying memory segment index. | Advanced zero-copy routing. |
| Route slot | `route_slot` | `route_slot` | `-1` | Route-graph slot identifier. | Advanced. |
| Segment name | `segment_name` | `segment_name` | empty | Memory segment name. | Advanced. |

### Methods and free helpers

| Helper | C++ | Python | What it does | Why and when to use it |
|---|---|---|---|---|
| Is it empty | `empty()` | `bool(sample)`, `False` when empty | True when the sample carries no payload. | Guard before reading. |
| Logical count | `size()` | `len(sample)` | Bundle field count; `1` for any non-empty non-Bundle sample; `0` when empty. | Lets one loop handle all three kinds. |
| Uniform access | `front()`, `back()`, `operator[]`, `begin()`/`end()` | `sample[i]`, iteration | On a Bundle these address the fields; on a non-Bundle they return the sample itself. Python's `sample[i]` checks the index and raises `IndexError` when it is out of range. | Write code that does not branch on `kind`: iterating a single-tensor sample yields that one sample. |
| Build a Bundle | `reserve(n)`, `push_back(Sample)` | `append(sample)` | Appending to an empty sample turns it into a Bundle. | Composing a multi-output sample to push. |
| Flatten to tensors | `tensors_from_sample(sample, require_nonempty = true)` | not available | Extracts every tensor from any of the three shapes, recursively. | The one-line way to get the numbers. Pass `false` when an empty sample is acceptable rather than an error. |
| Build from tensors | `sample_from_tensors(tensors)` | not available | Makes a `TensorSet` sample. | Pushing a tensor list as a sample. |
| Single tensor or throw | `require_single_tensor(sample)` | not available | Returns the one tensor; throws `NeatError` if there are zero or more than one. | Asserting a single-output contract instead of silently taking `front()`. |
| Tensor list or throw | `sample_tensor_list(sample)` | not available | Reference to `tensors`; throws if `kind != TensorSet`. | Same, for the list case. |
| Multi-output test | `sample_is_multi_output(sample)` | not available | True for a Bundle, or a `TensorSet` with more than one tensor. | Branching between single- and multi-output handling. |
| Has a tensor list | `sample_has_tensor_list(sample)` | not available | True for a non-empty `TensorSet`. | Guard before `sample_tensor_list`. |
| Make a tensor sample | `make_tensor_sample(port_name, tensor)` | `pyneat.make_tensor_sample(port_name, tensor)` | Builds a single-tensor sample bound to a port. | Pushing to a named port. |
| Make an image sample | `make_image_sample(mat, ...)` | not available | Builds an image sample from a `cv::Mat`. | C++ applications pushing frames from OpenCV. |
| Make a bundle | `make_bundle_sample({a, b})` | not available | Builds a Bundle from a list of samples. | Composing multi-output input. |
| Make an encoded sample | `make_encoded_sample(bytes, caps, ...)` | `pyneat.make_encoded_sample(...)` or `pyneat.Sample.from_encoded(...)` | Wraps an encoded bitstream with its caps. | Pushing H.264/H.265 data you read yourself. |
| Make a text sample | not available under this name | `pyneat.make_text_sample(port_name, text)` | Builds a text-payload sample. | GenAI prompts and other text input. |

## Language differences

**Most of the `Sample` free helpers are not bound in Python.**
`tensors_from_sample`, `sample_from_tensors`, `require_single_tensor`,
`sample_tensor_list`, `sample_is_multi_output`, `sample_has_tensor_list`,
`make_image_sample` and `make_bundle_sample` are C++ only. Only
`make_tensor_sample`, `make_text_sample` and `make_encoded_sample` are
exposed, plus the `Sample.from_encoded` classmethod added in
`python/pyneat/_wrappers.py`.

This is why every Python example defines its own walker —
`first_tensor_from_sample` in the pose estimator,
`segment_tensors_from_sample` in the instance segmenter. A Python application
has to branch on `kind` by hand.

**`make_text_sample` is Python-only** under that name.

**`attributes` is a live map, not a dict snapshot.** The binding exposes it as
its own `SampleAttributes` type by reference, specifically so that
`sample.attributes["k"] = "v"` reaches the C++ `Sample`. A plain conversion to
a Python dict would have dropped item assignment silently.

## Pitfalls

**Reading the wrong union field gives you nothing, not an error.** If `kind` is
`TensorSet` and you read `tensor`, you get an unset optional; if `kind` is
`Bundle` and you read `tensors`, you get an empty list. Symptom: zero
detections, no exception. Fix: branch on `kind`, or use
`tensors_from_sample()` in C++, which handles all three.

**A Bundle's field order is not a contract.** The shipped examples find their
field by comparing `stream_label` rather than indexing `fields[0]`. Symptom: a
model or graph change silently swaps your frame and your detections. Fix:
select by `stream_label` or `logical_output_index`.

**In C++, `front()` and `operator[]` do not fail on a non-Bundle.** They return the
sample itself. That is deliberate — it lets one loop handle every kind — but it
means an index error does not look like one. Symptom: `sample[3]` on a
single-tensor sample returns the sample rather than throwing. Fix: check
`size()` when the index actually matters. Python's `sample[i]` raises
`IndexError` instead.

**A borrowed sample (`owned == false`) must not be kept.** It holds a loan on
the producer's buffer. Copy the data out before the sample goes out of scope.
The superpoint example notes this explicitly where it converts to a `cv::Mat`:
the CPU copy outlives the decoder loan held by the sample.

**`pts_ns`, `frame_id` and friends default to `-1`, not `0`.** A source that
does not carry timestamps leaves them at `-1`. Symptom: nonsensical latency
numbers if you treat `-1` as a real timestamp. Fix: check for `-1` before
arithmetic.

## Used in

- [fastflow-anomaly-detector](../../../examples/anomaly-detection/fastflow-anomaly-detector/) — walks `fields` by `stream_label == "frame"`, then `tensors_from_sample`
- [superpoint-feature-extractor](../../../examples/feature-extraction/superpoint-feature-extractor/) — picks a bundle field, then flattens; notes the decoder loan
- [single-stream-instance-segmenter](../../../examples/segmentation/single-stream-instance-segmenter/) — resolves the `"segments"` field out of a Bundle in both languages
- [multi-stream-pose-estimator](../../../examples/pose-estimation/multi-stream-pose-estimator/) — uses the sample's stream identity to route results per camera
- [high-density-multi-stream-object-detector](../../../examples/object-detection/high-density-multi-stream-object-detector/) — derives a stream index from the pulled sample

## Source

- C++ header: `include/pipeline/GraphOptions.h` (`Sample`, `SampleKind`,
  `SampleAttributes`, `tensors_from_sample`, `sample_from_tensors`,
  `make_tensor_sample`, `make_image_sample`, `make_bundle_sample`,
  `require_single_tensor`, `sample_tensor_list`)
- Python binding: `python/src/module.cpp`, `nb::class_<Sample>`,
  `nb::enum_<SampleKind>`, `nb::bind_map<SampleAttributes>`,
  `make_tensor_sample`, `make_text_sample`, `make_encoded_sample`;
  `python/pyneat/_wrappers.py`, `Sample.from_encoded`
- Core commit: `c68a5241`
