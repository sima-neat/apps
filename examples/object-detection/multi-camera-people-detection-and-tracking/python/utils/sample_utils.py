"""Sample and payload helpers for the Python multi-camera example."""

from __future__ import annotations

import json
import struct
from typing import Any

from .tracker import TrackedDetection


def filter_person_detections(boxes: list[dict], person_class_id: int = 0) -> list[dict]:
    return [box for box in boxes if int(box.get("class_id", -1)) == int(person_class_id)]


def make_optiview_tracking_json(
    timestamp_ms: int,
    frame_id: str,
    tracked: list[TrackedDetection],
) -> str:
    objects = []
    for det in tracked:
        objects.append(
            {
                "id": f"track_{det.track_id}",
                "track_id": int(det.track_id),
                "label": "person",
                "confidence": float(det.score),
                "bbox": [
                    float(det.x1),
                    float(det.y1),
                    float(max(0.0, det.x2 - det.x1)),
                    float(max(0.0, det.y2 - det.y1)),
                ],
            }
        )
    return json.dumps(
        {
            "type": "object-detection",
            "timestamp": int(timestamp_ms),
            "frame_id": str(frame_id),
            "data": {"objects": objects},
        }
    )


def iter_tensors(pyneat: Any, sample: Any):
    if getattr(sample, "kind", None) == pyneat.SampleKind.Tensor and getattr(sample, "tensor", None) is not None:
        yield sample.tensor
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
    stack = [sample]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(getattr(current, "fields", []))))
        if getattr(current, "kind", None) != pyneat.SampleKind.Tensor:
            continue
        tensor = getattr(current, "tensor", None)
        if tensor is None:
            continue
        fmt = (getattr(current, "payload_tag", "") or getattr(current, "format", "") or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


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


def tensor_bgr_from_sample(runtime, sample: Any):
    tensor = first_tensor(runtime.pyneat, sample)
    if tensor is None:
        raise RuntimeError("no tensor payload found in decoded RTSP sample")
    arr = runtime.np.asarray(tensor.to_numpy(copy=True))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {arr.shape}")
    if arr.dtype != runtime.np.uint8:
        arr = runtime.np.clip(arr, 0, 255).astype(runtime.np.uint8)
    return runtime.np.ascontiguousarray(arr)
