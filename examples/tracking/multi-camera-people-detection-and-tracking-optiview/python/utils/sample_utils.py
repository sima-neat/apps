"""Sample and payload helpers for the Python multi-camera example."""

from __future__ import annotations

import struct
from typing import Any

from .tracker import TrackedDetection


def filter_person_detections(boxes: list[dict], person_class_id: int = 0) -> list[dict]:
    return [box for box in boxes if int(box.get("class_id", -1)) == int(person_class_id)]


def make_optiview_tracking_detection(
    pyneat: Any,
    tracked: list[TrackedDetection],
) -> tuple[list[Any], list[str]]:
    objects: list[Any] = []
    labels: list[str] = []
    for index, det in enumerate(tracked):
        obj = pyneat.OptiViewObject()
        obj.x = int(det.x1)
        obj.y = int(det.y1)
        obj.w = int(max(0.0, det.x2 - det.x1))
        obj.h = int(max(0.0, det.y2 - det.y1))
        obj.score = float(det.score)
        obj.class_id = index
        objects.append(obj)
        labels.append(f"Track ID: {det.track_id}")
    return objects, labels


def iter_tensors(pyneat: Any, sample: Any):
    if getattr(sample, "kind", None) == pyneat.SampleKind.Tensor and getattr(sample, "tensor", None) is not None:
        yield sample.tensor
    for tensor in getattr(sample, "tensors", []):
        yield tensor
    for field in getattr(sample, "fields", []):
        yield from iter_tensors(pyneat, field)


def first_tensor(pyneat: Any, sample: Any) -> Any:
    if sample is None:
        return None
    if getattr(sample, "kind", None) == pyneat.SampleKind.Tensor and getattr(sample, "tensor", None) is not None:
        return sample.tensor
    for tensor in iter_tensors(pyneat, sample):
        return tensor
    return None


def extract_bbox_payload(pyneat: Any, sample: Any) -> bytes | None:
    for tensor in iter_tensors(pyneat, sample):
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def tensor_bgr_from_sample(runtime, sample: Any):
    tensor = first_tensor(runtime.pyneat, sample)
    if tensor is None:
        raise RuntimeError("no tensor payload found in decoded RTSP sample")

    if tensor.is_nv12():
        width = int(tensor.width() if callable(tensor.width) else tensor.width)
        height = int(tensor.height() if callable(tensor.height) else tensor.height)
        payload = runtime.np.frombuffer(tensor.copy_payload_bytes(), dtype=runtime.np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = runtime.cv2.cvtColor(nv12, runtime.cv2.COLOR_YUV2BGR_NV12)
        rgb = runtime.cv2.cvtColor(bgr, runtime.cv2.COLOR_BGR2RGB)
        return runtime.np.ascontiguousarray(rgb)

    if tensor.is_i420():
        width = int(tensor.width() if callable(tensor.width) else tensor.width)
        height = int(tensor.height() if callable(tensor.height) else tensor.height)
        payload = runtime.np.frombuffer(tensor.copy_payload_bytes(), dtype=runtime.np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = runtime.cv2.cvtColor(i420, runtime.cv2.COLOR_YUV2BGR_I420)
        rgb = runtime.cv2.cvtColor(bgr, runtime.cv2.COLOR_BGR2RGB)
        return runtime.np.ascontiguousarray(rgb)

    if hasattr(tensor, "to_numpy"):
        try:
            arr = runtime.np.asarray(tensor.to_numpy(copy=True))
        except Exception:
            arr = None
        if arr is not None:
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim != 3:
                raise RuntimeError(f"unexpected decoded tensor shape: {arr.shape}")
            if arr.dtype != runtime.np.uint8:
                arr = runtime.np.clip(arr, 0, 255).astype(runtime.np.uint8)
            return runtime.np.ascontiguousarray(arr)

    if hasattr(tensor, "to_cv_mat_copy"):
        return tensor.to_cv_mat_copy(runtime.pyneat.PixelFormat.RGB)

    raise RuntimeError("decoded sample tensor does not expose a supported conversion API")


def parse_bbox_payload(payload: bytes | None, img_w: int, img_h: int) -> list[dict]:
    if not payload or len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    boxes: list[dict] = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes
