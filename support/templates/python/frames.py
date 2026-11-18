"""Decoded frames: tensors in a sample, and conversion to BGR images for OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import pyneat


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor_from_sample(sample):
    """The first tensor in a sample, searching bundle fields in order; None when there is none."""
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field in sample.fields:
        tensor = first_tensor_from_sample(field)
        if tensor is not None:
            return tensor
    return None


def find_field(sample, label: str):
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    """The field of a combined output with this stream label, else the bundle field at the index."""
    field = find_field(sample, label)
    fields = list(getattr(sample, "fields", []))
    if field is not None:
        return field
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output missing {label} field")


def packed_yuv420(tensor, name: str):
    """An NV12 or I420 frame as one width x (height * 3 / 2) image with its planes back to back.

    The decoder pads its frames: a 1080p frame has 2048-byte rows and its UV plane starts after
    1088 rows, and a 720p UV plane starts after 768 rows. contiguous() packs the planes, so the
    bytes can be read as one image; slicing the raw payload instead shears or recolors it.
    """
    width = tensor_dim(tensor, "width")
    height = tensor_dim(tensor, "height")
    payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size < expected:
        raise RuntimeError(f"{name} payload too small: {payload.size} < {expected}")
    return payload[:expected].reshape((height * 3 // 2, width))


def tensor_bgr_from_decoded(tensor):
    """A decoded NV16, NV12, I420 or packed frame as a contiguous BGR uint8 image."""
    if tensor.image_format() == pyneat.PixelFormat.NV16:
        # A 4:2:2 USB camera frame. The payload is the whole mapped buffer, so the Y and UV
        # planes are read at their own offsets and strides; each row keeps its own chroma by
        # interleaving Y and UV into packed 4:2:2 (Y0 U0 Y1 V0).
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        planes = {plane.role: plane for plane in tensor.planes}

        def plane_rows(role, rows):
            plane = planes[role]
            stride = plane.strides_bytes[0]
            end = plane.byte_offset + stride * rows
            if end > payload.size:
                raise RuntimeError(f"NV16 payload smaller than its {role} plane")
            return payload[plane.byte_offset:end].reshape((rows, stride))[:, :width]

        y = plane_rows(pyneat.PlaneRole.Y, height)
        uv = plane_rows(pyneat.PlaneRole.UV, height)
        return np.ascontiguousarray(cv2.cvtColor(np.dstack([y, uv]), cv2.COLOR_YUV2BGR_YUY2))

    if tensor.is_nv12():
        nv12 = packed_yuv420(tensor, "NV12")
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    if tensor.is_i420():
        i420 = packed_yuv420(tensor, "I420")
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)
