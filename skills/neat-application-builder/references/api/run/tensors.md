# Tensor

## What it is

A `Tensor` is a block of numbers plus everything needed to interpret it:
element type, shape, strides, where it lives, and what the numbers mean. It is
the payload inside a `Sample`, and `TensorList` is just
`std::vector<Tensor>` for the multi-input and multi-output cases.

A `Tensor` does not own its bytes directly. `storage` is a shared handle to a
`TensorBuffer`, and several tensors can view the same storage at different
offsets. That is what makes zero-copy possible, and it is also the source of
the one pitfall in this area that will stall your pipeline.

`TensorMemory` says where storage is allocated when you create a tensor.
`OutputMemory` — a `RunOptions` field — says whether tensors you pull share the
producer's storage or are copied. `Mapping` is the scoped window through which
C++ reads or writes the bytes.

## When to use it

Every application touches tensors: you build them to push, and you read them
after you pull.

To create one, pick the call that matches what you already hold:
`Tensor::from_cv_mat` in C++, `Tensor.from_numpy` or `Tensor.from_torch` in
Python. To read one, use a `Mapping` in C++ when you want to avoid a copy, or
`copy_dense_bytes_tight()` / `copy_payload_bytes()` when a copy is fine and
simpler. In Python there is no mapping — use `to_numpy()` or one of the copy
helpers.

Set `output_memory` on `RunOptions` rather than converting tensors after the
fact. `ZeroCopy` is the fast path and what the realtime examples use; `Owned`
is the one to pick when you need pulled tensors to outlive the frame they came
from.

## Minimal example

C++

```cpp
// Zero-copy read: the Mapping holds the producer's buffer loan.
if (tensor.is_contiguous() && tensor.dense_bytes_tight() > 0) {
  const auto mapping = tensor.map_read();
  const cv::Mat patches(rows, cols, CV_32F, mapping.data);
  return score(patches);  // Mapping releases the loan at end of scope.
}
return score_from_copy(tensor.copy_dense_bytes_tight());
```

Python

```python
tensor = pyneat.Tensor.from_numpy(
    frame, copy=True,
    image_format=pyneat.PixelFormat.BGR,
    memory=pyneat.TensorMemory.CPU,
)
run.push([tensor])
# Reading back: no Mapping in Python, so copy out.
arr = np.frombuffer(out_tensor.copy_dense_bytes_tight(), dtype=np.float32)
```

Complete flow: [patchcore-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector)

## Options

### `Tensor` fields

| Field | C++ | Python | Default | What it does | Why and when to use it |
|---|---|---|---|---|---|
| Storage | `storage` | `storage` (read-only) | null | Shared handle to the `TensorBuffer` holding the bytes. | In C++ you map through it. Treat it as opaque otherwise; several tensors may share one. |
| Element type | `dtype` | `dtype` | `TensorDType::UInt8` | Element type. | Always check it before reinterpreting bytes. A model output is often `Float32` while an image input is `UInt8`. |
| Shape | `shape` | `shape` | empty | Per-dimension sizes. | The first thing to read on a model output. |
| Strides | `strides_bytes` | `strides_bytes` | empty | Per-dimension strides in bytes. **Empty means compact row-major derived from shape and dtype**, not "no strides". | Only set it when you are describing a padded or non-contiguous buffer. |
| Byte offset | `byte_offset` | `byte_offset` | `0` | Offset from the start of `storage` to this tensor's first element. | How two tensors view one buffer. The mapping applies it for you, once. |
| Axis roles | `axis_semantics` | `axis_semantics` | empty | Per-axis role tags (N/D/H/W/C/Unknown). | The modern way to say which axis is which. Prefer it over `layout`. |
| Legacy layout | `layout` | `layout` | `TensorLayout::Unknown` | Coarse layout token. Transitional. | Prefer `axis_semantics`. Still needed when a Python creation call wants a layout hint. |
| Device | `device` | `device` | default | Where the tensor lives. | Check it before assuming a CPU pointer is valid. |
| Semantic | `semantic` | `semantic` | default | What the numbers represent — image, audio, tessellated, and so on. | Distinguishing an image tensor from a plain feature tensor. |
| Planes | `planes` | `planes` | empty | Per-plane records for composite formats. Empty for dense tensors. | NV12 has Y + UV; I420 has Y + U + V. Use the plane helpers rather than assuming one block. |
| Read-only | `read_only` | `read_only` | `true` | When true the framework refuses mutable data access. | **Defaults to true.** Set it false deliberately if you intend to write into the tensor. |
| Route metadata | `route` | `route` | default | Output identity in multi-output models. | Matching a tensor to a model output. |

### `TensorMemory`

Placement for tensors you create that own their storage. C++ alias: `Memory`.

| Value | C++ | Python | What it does | Why and when to change it |
|---|---|---|---|---|
| EV74 | `TensorMemory::EV74` | `pyneat.TensorMemory.EV74` | DDR from the SiMa allocator in the EV74-addressable window. The default for new owning creators. | The right default for data headed into the accelerator. |
| CPU | `TensorMemory::CPU` | `pyneat.TensorMemory.CPU` | Ordinary malloc-backed host memory. | A frame you are about to manipulate on the host, or a short-lived push buffer. The shipped examples use this for `cv::Mat`-derived input. |
| MLA | `TensorMemory::MLA` | `pyneat.TensorMemory.MLA` | DMS0-backed, when the runtime allocator is available. | Advanced placement for MLA-adjacent work. |
| A65 | `TensorMemory::A65` | `pyneat.TensorMemory.A65` | A65-addressable placement. | Advanced. |
| Auto | `TensorMemory::Auto` | `pyneat.TensorMemory.Auto` | Let the runtime choose. | When you have no reason to care. |

Note the enum order differs between the two languages' declarations — `EV74`
is first in C++ and the numeric zero, while the Python binding lists `Auto`
first. The names and their meanings are identical; do not rely on the integer
value.

### `OutputMemory` (a `RunOptions` field)

| Value | C++ | Python | Default | What it does | Why and when to change it |
|---|---|---|---|---|---|
| Auto | `OutputMemory::Auto` | `pyneat.OutputMemory.Auto` | the default | Retains standard DMA-BUF output storage and certified CPU-only terminal pools; other storage follows the preset's policy. | Leave it unless you know which of the two you want. The preset decides: `Realtime` and `Balanced` default to sharing storage, `Reliable` does not. |
| ZeroCopy | `OutputMemory::ZeroCopy` | `pyneat.OutputMemory.ZeroCopy` | — | Pulled tensors share the producer's storage. Retained tensors and their mappings hold producer-buffer loans. | The fast path, and what every realtime example picks. Choose it when you consume each frame promptly and release it. |
| Owned | `OutputMemory::Owned` | `pyneat.OutputMemory.Owned` | — | Copies into independent framework-owned system memory, holding no producer loan and using no device (CMA) memory. | Choose it when a pulled tensor must outlive its frame — buffering results, handing them to another thread, or keeping a history. Pay one copy to stop worrying about loans. |

### Reading and writing

| Call | C++ | Python | Default | What it does | Why and when to use it |
|---|---|---|---|---|---|
| Map for access | `map(MapMode)` | not available | — | Opens a scoped CPU window over the bytes. | The zero-copy read path in C++. |
| Map for reading | `map_read()` | not available | — | `map(MapMode::Read)`. | The common case; the framework may skip cache invalidation when the buffer is known clean. |
| View | `view(MapMode = Read)` | not available | `Read` | Mapping variant respecting named-segment view bounds. | Segmented buffers. |
| Copy dense bytes | `copy_dense_bytes_tight()` | `copy_dense_bytes_tight()` | — | Returns a tight copy of a dense tensor's bytes. | The simple read path, and the only one in Python. Pair with `dense_bytes_tight()` to size a buffer. |
| Copy payload bytes | `copy_payload_bytes()` | `copy_payload_bytes()` | — | Copies the payload including composite planes. | NV12 or I420 output you want as one block. |
| Copy into your buffer | `copy_dense_bytes_tight_to(dst, n)`, `copy_payload_bytes_to(dst, n)` | not available | — | Same, into memory you own. | Avoiding an allocation per frame in C++. |
| To numpy | not available | `to_numpy(copy=False)` | `copy=False` | Views the tensor as a numpy array through DLPack. | The idiomatic Python read. Pass `copy=True` when the array must outlive the tensor. |
| From numpy | not available | `Tensor.from_numpy(array, copy=False, layout=None, image_format=None, memory=None, byte_format=None)` | `copy=False` | Builds a tensor from an array. | The idiomatic Python push path. `copy=True` when the array is transient. |
| From torch | not available | `Tensor.from_torch(...)` / `from_pytorch(...)`, `to_torch(...)` | `copy=False` | PyTorch interop through DLPack; a non-CPU torch tensor is moved to CPU first. | Feeding a model from an existing torch pipeline. |
| From a Python list | not available | `Tensor.from_list(data, shape=None, dtype=None, ...)` | copies | Builds from a sequence, routed through `from_numpy`. Int lists become `int32`, since `int64` is not a device dtype. | Small hand-written inputs and tests. |
| From a `cv::Mat` | `Tensor::from_cv_mat(mat, fmt, memory)` | not available | — | Wraps an OpenCV matrix. | The C++ push path in the shipped examples. |
| `cv::Mat` views | `map_cv_mat_view(fmt)`, `to_cv_mat_copy(fmt)` | not available | — | Mapping bundled with a non-owning `cv::Mat`, or a copy. | C++ post-processing with OpenCV. |
| Move across devices | `to(Device)`, `cpu()`, `cvu()`, `mla(force)`, `to_cpu_if_needed()` | `cpu()`, `cvu()`, `mla(force=False)`, `to_cpu_if_needed()` | `force=False` | Returns the tensor on another device. | `cpu()` is the explicit conversion out of device storage. `to(Device)` is C++ only. |
| Normalise layout | `contiguous()`, `clone()` | `contiguous()`, `clone()` | — | Compact copy, or an independent copy. | `contiguous()` before handing bytes to code that assumes row-major; `clone()` to break a loan. |
| Shape questions | `is_dense()`, `is_composite()`, `is_contiguous()`, `has_axis_semantics()`, `axis_semantics_match_shape()` | same | — | Predicates on the layout. | Guard a mapping with `is_contiguous()`, as the examples do. |
| Image questions | `width()`, `height()`, `channels()`, `image_format()`, `is_nv12()`, `is_i420()` | same | — | Image-specific accessors. | Deciding how to convert a decoded frame. |
| Plane access | `plane(role)`, `try_plane(role)`, `has_plane(role)` | `has_plane(role)` | — | Per-plane records for composite formats. | NV12 and I420 handling. `plane()` throws when absent; `try_plane()` returns null. |
| NV12 / I420 helpers | `map_nv12_read()`, `nv12_required_bytes()`, `copy_nv12_contiguous[_to]()`, and the I420 equivalents | not available | — | Format-specific read helpers. | Converting a decoded frame without writing the plane arithmetic yourself. |
| Validate | `validate(std::string*)` | `validate()` | — | Checks internal consistency. | Debugging a tensor you built by hand. |
| Debug | `debug_string()`, `to_text()`, `from_text()` | `debug_string()`, `to_text()`, `from_text()`, `__repr__` | — | Human-readable forms. | Logging what you actually received. |

### `MapMode`

| Value | C++ | Python | What it does | Why and when to change it |
|---|---|---|---|---|
| Read | `MapMode::Read` | not available | Read-only; the framework may skip cache invalidation when the buffer is known clean. | The default for consuming output. |
| Write | `MapMode::Write` | not available | Write-only; skips read-back and flushes cache on unmap. | Filling a tensor you created. |
| ReadWrite | `MapMode::ReadWrite` | not available | Both; treated as full ownership for the mapping's lifetime. | In-place modification. |

### `Mapping`

| Field | C++ | Python | Default | What it does |
|---|---|---|---|---|
| Data pointer | `data` | not available | null | CPU-readable pointer, valid for the `Mapping`'s lifetime. |
| Size | `size_bytes` | not available | `0` | Size of the mapped region. May include trailing backing bytes, so compare against `dense_bytes_tight()` rather than trusting it as the payload size. |
| Unmap callback | `unmap` | not available | unset | Runs in the destructor: cache flush and refcount decrement. |
| Keepalive | `keepalive` | not available | unset | Lifetime guard ensuring the storage outlives the mapping. |

`Mapping` is move-only; copying is deleted, and the destructor releases the
mapping. There is no Python equivalent — Python reads through DLPack or the
copy helpers.

## Language differences

**There is no `Mapping` or `MapMode` in Python.** Zero-copy reads are a C++
capability. Python reads either through DLPack (`to_numpy`, `to_torch`,
`__dlpack__`) or by copying (`copy_dense_bytes_tight`, `copy_payload_bytes`).

**`__dlpack__` only supports dense tensors** and raises for composite ones, so
an NV12 output cannot go straight to numpy. Use `copy_payload_bytes()` and do
the plane arithmetic, as the fastflow example does.

**`__dlpack_device__` always reports CPU**, whatever `device` says. Call
`cpu()` or `to_cpu_if_needed()` before handing a device tensor to numpy or
torch rather than trusting the DLPack device.

**Python adds framework interop that C++ does not have**: `from_numpy`,
`to_numpy`, `from_torch`/`from_pytorch`, `to_torch`/`to_pytorch`,
`from_dlpack` and `from_list`, all defined in `python/pyneat/_wrappers.py` on
top of the bound `_from_dlpack_capsule`.

**C++ has OpenCV and format helpers that Python does not**: `from_cv_mat`,
`map_cv_mat_view`, `to_cv_mat_copy`, the NV12/I420 map and copy helpers, the
`_to(dst, n)` copy-into-your-buffer variants, and `to(Device)`.

**Python's creation calls take a `layout` hint** and infer one when you omit
it; passing `byte_format` forces `TensorLayout::Unknown`. C++ sets the field
directly.

## Pitfalls

**A retained zero-copy tensor stalls the pipeline.** Under
`OutputMemory::ZeroCopy`, a pulled tensor and any `Mapping` over it hold a loan
on the producer's buffer pool. Hold several frames' worth and the pool runs
out, the producer blocks, and throughput collapses — with no error message.
Symptom: frame rate drops to near zero after a few seconds, and nothing in the
logs. Fix: scope the mapping tightly and copy out what you need to keep. Both
shipped examples say so in a comment where they do it: patchcore scores while
the read mapping owns the loan and releases it before drawing; superpoint notes
that the CPU copy outlives the decoder loan held by the sample. If you must
keep tensors, use `OutputMemory::Owned`, or `clone()` them.

**`strides_bytes` being empty does not mean "contiguous unknown".** It means
compact row-major derived from `shape` and `dtype`. Code that checks
`strides_bytes.empty()` to decide whether strides matter will read a padded
tensor wrongly in the other direction. Use `is_contiguous()`, which handles
both cases.

**`mapping.size_bytes` can be larger than the payload.** The mapped extent may
include trailing backing bytes. Both examples guard with
`mapping.size_bytes >= tensor.dense_bytes_tight()` and then use the tight size.
Symptom: garbage at the end of a buffer you sized from `size_bytes`.

**`read_only` defaults to `true`.** A tensor you construct and expect to fill
will refuse mutable access until you clear it. Symptom: a write path that
fails on a tensor that looks fine.

**Mapping a non-contiguous tensor and treating it as a flat block is wrong.**
Both examples check `is_contiguous()` first and fall back to a copy otherwise.
`contiguous()` gives you a compact copy when you need one.

**`int64` numpy arrays are not a device dtype.** `from_list` maps Python int
lists to `int32` for you, but `from_numpy` on an explicitly `int64` array does
not. Cast before pushing.

## Used in

- [patchcore-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/patchcore-anomaly-detector) — `map_read()` guarded by `is_contiguous()` and `dense_bytes_tight()`, loan released before drawing
- [high-density-multi-stream-object-detector](https://github.com/sima-neat/apps/tree/main/examples/object-detection/high-density-multi-stream-object-detector) — `map_read()` with a copy fallback, `ZeroCopy` output
- [superpoint-feature-extractor](https://github.com/sima-neat/apps/tree/main/examples/feature-extraction/superpoint-feature-extractor) — `Tensor::from_cv_mat` with `TensorMemory::CPU`; Python `from_numpy`
- [fastflow-anomaly-detector](https://github.com/sima-neat/apps/tree/main/examples/anomaly-detection/fastflow-anomaly-detector) — Python `copy_payload_bytes()` into numpy, `TensorMemory.EV74`
- [image-classification-explorer](https://github.com/sima-neat/apps/tree/main/examples/classification/image-classification-explorer) — Python `copy_dense_bytes_tight()` with an explicit dtype

## Source

- C++ header: `include/pipeline/Tensor.h` (`TensorList`),
  `include/pipeline/TensorCore.h` (`Tensor`, `TensorMemory`, `Memory`,
  `MapMode`, `Mapping`, `PlaneRole`, `TensorBuffer`),
  `include/pipeline/TensorTypes.h` (`TensorDType`, `TensorLayout`,
  `TensorAxisSemantic`), `include/pipeline/Run.h` (`OutputMemory`)
- Python binding: `python/src/module.cpp`, `nb::class_<Tensor>`,
  `nb::enum_<TensorMemory>`, `nb::enum_<OutputMemory>`;
  `python/pyneat/_wrappers.py` for `from_numpy`, `to_numpy`, `from_torch`,
  `to_torch`, `from_dlpack`, `from_list`
- Core commit: `c68a5241`
