"""Sample helpers for family-specific detector outputs."""

from __future__ import annotations

import struct
from typing import Any

from .model_family import YOLOX_NOT_SUPPORTED_MESSAGE


def class_label(class_labels: list[str], class_id: int) -> str:
    if 0 <= int(class_id) < len(class_labels):
        return class_labels[int(class_id)]
    return str(int(class_id))


def iter_tensors(pyneat: Any, sample: Any):
    if getattr(sample, "kind", None) == pyneat.SampleKind.Tensor and getattr(sample, "tensor", None) is not None:
        yield sample.tensor
    for field in getattr(sample, "fields", []):
        yield from iter_tensors(pyneat, field)


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


def sample_payload_tag(pyneat: Any, sample: Any) -> str:
    if sample is None:
        return ""
    tag = (getattr(sample, "payload_tag", "") or getattr(sample, "format", "") or "").upper()
    if tag:
        return tag
    for field in getattr(sample, "fields", []):
        child = sample_payload_tag(pyneat, field)
        if child:
            return child
    return ""


def optiview_frame_id(sample: Any, fallback_frame_index: int) -> str:
    frame_id = getattr(sample, "frame_id", -1)
    try:
        if int(frame_id) >= 0:
            return str(int(frame_id))
    except Exception:
        pass
    return str(int(fallback_frame_index))


def optiview_timestamp_ms(publish_time_s: float, offset_ms: float) -> int:
    return int(round(float(publish_time_s) * 1000.0 + float(offset_ms)))


def require_detector_output_kind(pyneat: Any, family: str, sample: Any) -> str:
    lowered = str(family).strip().lower()
    if lowered == "yolov8":
        expected = "BBOX"
    elif lowered == "yolox":
        raise ValueError(YOLOX_NOT_SUPPORTED_MESSAGE)
    else:
        raise ValueError(f"unsupported model family: {family}")

    actual = sample_payload_tag(pyneat, sample)
    if actual != expected:
        raise RuntimeError(
            f"unsupported detector output: family={lowered} expected={expected} actual={actual or '<empty>'}"
        )
    return expected


def detections_from_detector_sample(
    pyneat: Any,
    family: str,
    sample: Any,
    img_w: int,
    img_h: int,
) -> list[dict]:
    kind = require_detector_output_kind(pyneat, family, sample)
    payload = extract_bbox_payload(pyneat, sample)
    if payload:
        return parse_bbox_payload(payload, img_w, img_h)
    if kind != "BBOX":
        raise RuntimeError(f"unsupported detector output kind: {kind}")
    return []


def make_optiview_detection_payload(
    pyneat: Any,
    detections: list[dict],
    *,
    img_w: int,
    img_h: int,
    class_labels: list[str],
) -> tuple[list[Any], list[str]]:
    objects: list[Any] = []
    for det in detections:
        x1 = max(0.0, min(float(img_w), float(det.get("x1", 0.0))))
        y1 = max(0.0, min(float(img_h), float(det.get("y1", 0.0))))
        x2 = max(0.0, min(float(img_w), float(det.get("x2", 0.0))))
        y2 = max(0.0, min(float(img_h), float(det.get("y2", 0.0))))
        if x2 <= x1 or y2 <= y1:
            continue

        obj = pyneat.OptiViewObject()
        obj.x = int(x1)
        obj.y = int(y1)
        obj.w = int(max(0.0, x2 - x1))
        obj.h = int(max(0.0, y2 - y1))
        obj.score = float(det.get("score", 0.0))
        obj.class_id = int(det.get("class_id", -1))
        objects.append(obj)
    # OptiView resolves object labels by indexing this full label table with object.class_id.
    return objects, list(class_labels)


def first_tensor(pyneat: Any, sample: Any) -> Any:
    if sample is None:
        return None
    if getattr(sample, "kind", None) == pyneat.SampleKind.Tensor and getattr(sample, "tensor", None) is not None:
        return sample.tensor
    for tensor in iter_tensors(pyneat, sample):
        return tensor
    return None


def tensor_rgb_from_sample(runtime, sample: Any):
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
