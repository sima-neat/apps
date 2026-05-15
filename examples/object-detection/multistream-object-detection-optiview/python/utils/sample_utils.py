"""Sample and detection helpers for the Python multistream object detection example."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any

from .model_family import ModelFamily, to_string as model_family_to_string

_BBOX_PAYLOAD_TAG = "BBOX"


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class OptiViewDetectionPayload:
    objects: list[dict[str, int | float]]
    labels: list[str]

def _sample_payload_tag_upper(sample: Any) -> str:
    tag = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    tag = str(tag).upper()
    if tag:
        return tag
    if getattr(sample, "tensors", []):
        return _BBOX_PAYLOAD_TAG
    for field in getattr(sample, "fields", []):
        nested = _sample_payload_tag_upper(field)
        if nested:
            return nested
    return ""


def iter_tensors(sample: Any):
    if getattr(sample, "tensor", None) is not None:
        yield sample.tensor
    for tensor in getattr(sample, "tensors", []):
        yield tensor
    for field in getattr(sample, "fields", []):
        yield from iter_tensors(field)


def first_tensor(sample: Any) -> Any | None:
    for tensor in iter_tensors(sample):
        return tensor
    return None


def extract_bbox_payload(sample: Any) -> bytes:
    for tensor in iter_tensors(sample):
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int) -> list[dict[str, float | int]]:
    if not payload or len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    offset = 4
    boxes: list[dict[str, float | int]] = []
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


def require_detector_output_kind(family: ModelFamily, sample: Any) -> str:
    if family is not ModelFamily.YOLOV8:
        raise ValueError("unsupported model family")
    actual = _sample_payload_tag_upper(sample)
    if actual != _BBOX_PAYLOAD_TAG:
        raise RuntimeError(
            "unsupported detector output: "
            f"family={model_family_to_string(family)} expected={_BBOX_PAYLOAD_TAG} "
            f"actual={actual or '<empty>'}"
        )
    return _BBOX_PAYLOAD_TAG


def detections_from_detector_sample(
    family: ModelFamily,
    sample: Any,
    img_w: int,
    img_h: int,
) -> list[Detection]:
    require_detector_output_kind(family, sample)
    payload = extract_bbox_payload(sample)
    boxes = parse_bbox_payload(payload, img_w, img_h)
    return [
        Detection(
            float(box["x1"]),
            float(box["y1"]),
            float(box["x2"]),
            float(box["y2"]),
            float(box["score"]),
            int(box["class_id"]),
        )
        for box in boxes
    ]


def optiview_frame_id(sample: Any, fallback_frame_index: int) -> str:
    frame_id = getattr(sample, "frame_id", -1)
    if isinstance(frame_id, int) and frame_id >= 0:
        return str(frame_id)
    return str(fallback_frame_index)


def optiview_timestamp_ms(publish_time_s: float, offset_ms: float) -> int:
    return int(round(publish_time_s * 1000.0 + offset_ms))


def build_optiview_detection_payload(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    class_labels: list[str],
) -> OptiViewDetectionPayload:
    objects: list[dict[str, int | float]] = []
    for det in detections:
        x1 = max(0, min(img_w, int(det.x1)))
        y1 = max(0, min(img_h, int(det.y1)))
        x2 = max(0, min(img_w, int(det.x2)))
        y2 = max(0, min(img_h, int(det.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append(
            {
                "x": x1,
                "y": y1,
                "w": max(0, x2 - x1),
                "h": max(0, y2 - y1),
                "score": float(det.score),
                "class_id": int(det.class_id),
            }
        )
    return OptiViewDetectionPayload(objects=objects, labels=list(class_labels))


def tensor_rgb_from_sample(runtime: Any, sample: Any):
    tensor = first_tensor(sample)
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
        arr = runtime.np.asarray(tensor.to_numpy(copy=True))
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
