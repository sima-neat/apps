"""Decoded 4:2:0 frames laid out the way the DevKit decoder hands them to an application.

The decoder pads its frames. Measured on a DevKit for H.264 RTSP streams, a 1080p NV12 frame
has 2048-byte rows and its UV plane starts after 1088 rows, and a 720p frame has 1280-byte rows
and its UV plane starts after 768 rows. Neat reports this in ``tensor.planes``. Code that reads
such a frame on the host has to pack it with ``tensor.contiguous()`` or follow each plane's
``byte_offset`` and ``strides_bytes``: slicing ``copy_payload_bytes()`` to
``width * height * 3 / 2`` shears a 1080p frame into stripes and moves a 720p frame's colors
down with a green band on top.

The tensors built here are real ``pyneat`` CPU tensors with that plane layout, so a test can
pass one to an application's frame conversion and compare the result with the packed picture.
"""

from __future__ import annotations

import numpy as np

#: (width, height, row stride in bytes, buffer row where the chroma starts), per measured layout.
DECODER_LAYOUTS = [(1920, 1080, 2048, 1088), (1280, 720, 1280, 768)]
DECODER_LAYOUT_IDS = ["1080p", "720p"]

#: Padding bytes are white so that reading them as pixels changes the picture.
PADDING = 255


def yuv420_pattern(width: int, height: int) -> np.ndarray:
    """A packed 4:2:0 picture, ``height * 3 / 2`` rows of ``width`` bytes, that varies everywhere.

    Read as NV12, the chroma rows interleave U and V; read as I420, the same bytes hold the U
    plane and then the V plane. Either way, a shifted row or a padding byte changes the result.
    """
    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    luma = ((rows * 3 + cols * 5) % 220 + 16).astype(np.uint8)
    chroma = np.empty((height // 2, width), dtype=np.uint8)
    chroma[:, 0::2] = ((rows[: height // 2] * 7) % 200 + 28).astype(np.uint8)
    chroma[:, 1::2] = ((cols[:, : width // 2] * 3) % 200 + 28).astype(np.uint8)
    return np.vstack([luma, chroma])


def _plane(pyneat, role, shape, stride: int, offset: int):
    plane = pyneat.Plane()
    plane.role, plane.shape = role, list(shape)
    plane.strides_bytes, plane.byte_offset = [stride, 1], offset
    return plane


def _planar_tensor(pyneat, buffer, image_format, width: int, height: int, stride: int, planes):
    tensor = pyneat.Tensor.from_numpy(
        buffer.reshape(-1, stride), copy=True, image_format=image_format,
        memory=pyneat.TensorMemory.CPU,
    )
    tensor.shape, tensor.strides_bytes = [height, width], [stride, 1]
    tensor.axis_semantics = [pyneat.TensorAxisSemantic.H, pyneat.TensorAxisSemantic.W]
    tensor.planes = planes
    return tensor


def padded_nv12_tensor(packed: np.ndarray, width: int, height: int, stride: int, uv_row: int):
    """``packed`` NV12 with ``stride``-byte rows and its UV plane at buffer row ``uv_row``."""
    import pyneat

    buffer = np.full((uv_row + height // 2, stride), PADDING, dtype=np.uint8)
    buffer[:height, :width] = packed[:height]
    buffer[uv_row:, :width] = packed[height:]
    planes = [
        _plane(pyneat, pyneat.PlaneRole.Y, (height, width), stride, 0),
        _plane(pyneat, pyneat.PlaneRole.UV, (height // 2, width), stride, uv_row * stride),
    ]
    return _planar_tensor(pyneat, buffer, pyneat.PixelFormat.NV12, width, height, stride, planes)


def padded_i420_tensor(packed: np.ndarray, width: int, height: int, stride: int, uv_row: int):
    """``packed`` I420 in the same padded geometry as :func:`padded_nv12_tensor`.

    The Y plane has ``stride``-byte rows and ``uv_row`` rows; the U and V planes follow it, each
    with ``stride / 2``-byte rows and ``uv_row / 2`` rows.
    """
    import pyneat

    chroma_w, chroma_h, chroma_stride = width // 2, height // 2, stride // 2
    y_bytes = uv_row * stride
    chroma_bytes = (uv_row // 2) * chroma_stride
    buffer = np.full(y_bytes + 2 * chroma_bytes, PADDING, dtype=np.uint8)
    buffer[:y_bytes].reshape(uv_row, stride)[:height, :width] = packed[:height]
    chroma = packed[height:].reshape(2, chroma_h, chroma_w)
    for index, start in enumerate((y_bytes, y_bytes + chroma_bytes)):
        rows = buffer[start:start + chroma_bytes].reshape(uv_row // 2, chroma_stride)
        rows[:chroma_h, :chroma_w] = chroma[index]
    planes = [
        _plane(pyneat, pyneat.PlaneRole.Y, (height, width), stride, 0),
        _plane(pyneat, pyneat.PlaneRole.U, (chroma_h, chroma_w), chroma_stride, y_bytes),
        _plane(
            pyneat, pyneat.PlaneRole.V, (chroma_h, chroma_w), chroma_stride,
            y_bytes + chroma_bytes,
        ),
    ]
    return _planar_tensor(pyneat, buffer, pyneat.PixelFormat.I420, width, height, stride, planes)
