"""Data helpers for the object-triggered VLM example."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from urllib import request

import cv2
import numpy as np
import pyneat
import yaml


@dataclass
class Config:
    rtsp_url: str
    model_path: str
    labels_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    classes: tuple[str, ...]
    trigger_class: str
    trigger_classes: list[str]
    timeout_ms: int
    debug: bool
    insight_host: str
    video_port: int
    metadata_port: int
    channel: int
    vlm_enabled: bool
    vlm_host: str
    vlm_port: int
    vlm_model: str
    vlm_max_tokens: int
    vlm_interval_seconds: float
    vlm_timeout_seconds: float
    vlm_max_pending_requests: int
    vlm_system_prompt: str
    memory_sample_interval_seconds: float
    memory_retention_seconds: float
    memory_jpeg_quality: int
    tracker_iou_threshold: float
    tracker_max_missing_seconds: float
    qa_enabled: bool
    qa_host: str
    qa_port: int
    qa_max_tokens: int
    qa_past_tolerance_seconds: float
    qa_default_past_seconds: float
    qa_crop_padding_ratio: float


def section(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = section(raw, "source")
    model = section(raw, "model")
    trigger = section(raw, "trigger")
    insight = section(raw, "insight")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    vlm = section(raw, "vlm")
    memory = section(raw, "memory")
    tracker = section(raw, "tracker")
    qa = section(raw, "qa")
    trigger_classes = list(class_filter(trigger.get("classes")))
    if not trigger_classes:
        trigger_class = str(trigger.get("class", "person")).strip().lower()
        trigger_classes = [trigger_class]
    else:
        trigger_class = trigger_classes[0]
    return Config(
        rtsp_url=source.get("rtsp_url", ""),
        model_path=model.get("path", ""),
        labels_path=model.get("labels", ""),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference.get("min_score", 0.55)),
        nms_iou=float(inference.get("nms_iou", 0.50)),
        max_detections=int(inference.get("max_detections", 24)),
        classes=class_filter(inference.get("classes")),
        trigger_class=trigger_class,
        trigger_classes=trigger_classes,
        timeout_ms=int(runtime.get("timeout_ms", 20000)),
        debug=bool(runtime.get("debug", False)),
        insight_host=insight.get("host", "127.0.0.1") or "127.0.0.1",
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
        channel=int(insight.get("channel", 0)),
        vlm_enabled=bool(vlm.get("enabled", False)),
        vlm_host=vlm.get("host", "127.0.0.1") or "127.0.0.1",
        vlm_port=int(vlm.get("port", 9998)),
        vlm_model=vlm.get("model", "gemma4-E4B-it") or "gemma4-E4B-it",
        vlm_max_tokens=int(vlm.get("max_tokens", 24)),
        vlm_interval_seconds=float(vlm.get("interval_seconds", 3.0)),
        vlm_timeout_seconds=float(vlm.get("timeout_seconds", 30.0)),
        vlm_max_pending_requests=max(1, int(vlm.get("max_pending_requests", 1))),
        vlm_system_prompt=vlm.get(
            "system_prompt",
            'Reply exactly: "{object_detector_class} of {c} color was just seen".',
        ),
        memory_sample_interval_seconds=float(memory.get("sample_interval_seconds", 0.5)),
        memory_retention_seconds=float(memory.get("retention_seconds", 10.0)),
        memory_jpeg_quality=int(memory.get("jpeg_quality", 85)),
        tracker_iou_threshold=float(tracker.get("iou_threshold", 0.40)),
        tracker_max_missing_seconds=float(tracker.get("max_missing_seconds", 1.0)),
        qa_enabled=bool(qa.get("enabled", True)),
        qa_host=qa.get("host", "0.0.0.0") or "0.0.0.0",
        qa_port=int(qa.get("port", 8088)),
        qa_max_tokens=int(qa.get("max_tokens", 64)),
        qa_past_tolerance_seconds=float(qa.get("past_tolerance_seconds", 1.0)),
        qa_default_past_seconds=float(qa.get("default_past_seconds", 3.0)),
        qa_crop_padding_ratio=float(qa.get("crop_padding_ratio", 0.75)),
    )


def class_filter(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    return tuple(str(item).strip().lower() for item in values if str(item).strip())


def load_labels(path: str) -> list[str]:
    label_path = Path(path)
    if not path or not label_path.is_file():
        return []
    return [
        line.strip()
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open RTSP source: {url}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0)) or 30
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP dimensions")
    return width, height, fps


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def decoded_tensor_to_rgb(tensor) -> np.ndarray:
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def bbox_payload_from_tensors(tensors) -> bytes:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def bbox_payload(result) -> bytes:
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return bbox_payload_from_tensors(result)

    if not is_sample_like(result):
        return b""

    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            payload = bbox_payload_from_tensors(current.tensors)
            if payload:
                return payload
            continue
        if current.kind != pyneat.SampleKind.Tensor or current.tensor is None:
            continue
        fmt = (current.payload_tag or current.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = current.tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return b""


def parse_boxes(result) -> list[dict]:
    payload = bbox_payload(result)
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    boxes = []
    for idx in range(count):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, 4 + idx * 24)
        if w > 0 and h > 0:
            boxes.append(
                {
                    "bbox": [x, y, w, h],
                    "score": score,
                    "class_id": class_id,
                }
            )
    return boxes


def label_for_box(box: dict, labels: list[str]) -> str:
    class_id = int(box["class_id"])
    return labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"


def metadata_json(boxes: list[dict], labels: list[str], classes: tuple[str, ...] = ()) -> str:
    objects = []
    allowed = set(classes)
    for box in boxes:
        label = label_for_box(box, labels)
        if allowed and label.lower() not in allowed:
            continue
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": label,
                "confidence": box["score"],
                "bbox": box["bbox"],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def best_box_for_label(boxes: list[dict], labels: list[str], wanted: str):
    wanted = wanted.lower()
    matches = [
        box
        for box in boxes
        if label_for_box(box, labels).lower() == wanted
    ]
    return max(matches, key=lambda box: box["score"], default=None)


def crop_box(frame: np.ndarray, box: dict, padding_ratio: float = 0.0) -> np.ndarray:
    x, y, w, h = [int(value) for value in box["bbox"]]
    height, width = frame.shape[:2]
    pad_x = int(max(0.0, padding_ratio) * w)
    pad_y = int(max(0.0, padding_ratio) * h)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid object crop: {box['bbox']}")
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def image_data_uri(crop_rgb: np.ndarray) -> str:
    crop = rgb_to_bgr(crop_rgb)
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("failed to encode object crop")
    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image}"


def preview_image_data_uri(image_rgb: np.ndarray, max_width: int = 480, quality: int = 75) -> str:
    height, width = image_rgb.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        image_rgb = cv2.resize(
            image_rgb,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = rgb_to_bgr(image_rgb)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode preview image")
    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image}"


def request_vlm_answer(
    image_rgb: np.ndarray,
    cfg: Config,
    question: str,
    system_prompt: str,
    metadata_text: str | None = None,
    max_tokens: int | None = None,
) -> str:
    user_content = [
        {"type": "image", "image": image_data_uri(image_rgb)},
    ]
    if metadata_text:
        user_content.append({"type": "text", "text": f"Detector metadata: {metadata_text}"})
    user_content.append({"type": "text", "text": f"Question: {question}"})
    payload = {
        "model": cfg.vlm_model,
        "stream": True,
        "max_tokens": max_tokens if max_tokens is not None else cfg.vlm_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    }

    url = f"http://{cfg.vlm_host}:{cfg.vlm_port}/v1/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    with request.urlopen(req, timeout=cfg.vlm_timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
            response_text += delta.get("content") or ""
    return response_text.strip()


def request_object_color_sentence(crop_rgb: np.ndarray, detected_class: str, cfg: Config) -> str:
    payload = {
        "model": cfg.vlm_model,
        "stream": True,
        "max_tokens": cfg.vlm_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": cfg.vlm_system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data_uri(crop_rgb)},
                    {
                        "type": "text",
                        "text": (
                            f'object_detector_class: "{detected_class}". '
                            "Identify the dominant visible color of this detected object. "
                            "Answer only in the required format."
                        ),
                    },
                ],
            },
        ],
    }

    url = f"http://{cfg.vlm_host}:{cfg.vlm_port}/v1/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    with request.urlopen(req, timeout=cfg.vlm_timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
            response_text += delta.get("content") or ""
    return response_text.strip()
