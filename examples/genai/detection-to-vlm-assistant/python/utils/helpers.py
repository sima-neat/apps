"""Small data helpers for the YOLOv8 Insight example."""

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


@dataclass(frozen=True)
class Config:
    rtsp_url: str
    model_path: str
    labels_path: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    classes: tuple[str, ...]
    timeout_ms: int
    debug: bool
    insight_host: str
    video_port: int
    metadata_port: int
    channel: int
    openai_enabled: bool
    openai_host: str
    openai_port: int
    openai_model: str
    openai_max_tokens: int
    openai_interval_seconds: float
    openai_timeout_seconds: float
    openai_max_pending_requests: int
    openai_system_prompt: str


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source = raw.get("source", {})
    model = raw.get("model", {})
    insight = raw.get("insight", {})
    inference = raw.get("inference", {})
    runtime = raw.get("runtime", {})
    openai = raw.get("openai", {})
    return Config(
        rtsp_url=source.get("rtsp_url", ""),
        model_path=model.get("path", ""),
        labels_path=model.get("labels", ""),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference.get("min_score", 0.55)),
        nms_iou=float(inference.get("nms_iou", 0.50)),
        max_detections=int(inference.get("max_detections", 24)),
        classes=class_filter(inference.get("classes")),
        timeout_ms=int(runtime.get("timeout_ms", 5000)),
        debug=bool(runtime.get("debug", False)),
        insight_host=insight.get("host", "127.0.0.1") or "127.0.0.1",
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
        channel=int(insight.get("channel", 0)),
        openai_enabled=bool(openai.get("enabled", False)),
        openai_host=openai.get("host", "127.0.0.1") or "127.0.0.1",
        openai_port=int(openai.get("port", 9998)),
        openai_model=openai.get("model", "gemma3") or "gemma3",
        openai_max_tokens=int(openai.get("max_tokens", 128)),
        openai_interval_seconds=float(openai.get("interval_seconds", 5.0)),
        openai_timeout_seconds=float(openai.get("timeout_seconds", 30.0)),
        openai_max_pending_requests=max(1, int(openai.get("max_pending_requests", 1))),
        openai_system_prompt=openai.get(
            "system_prompt",
            "Write one sarcastic scene caption. Comment on the visible situation, "
            "not identity or appearance. Use 10 words or fewer. Avoid: darling, "
            "ordinary, riveting, seriously, wow, posed. No explanation.",
        ),
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


def ev74_rgb_tensor(frame: np.ndarray):
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def iter_tensors(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)
        return
    if getattr(value, "tensor", None) is not None:
        yield value.tensor
    yield from getattr(value, "tensors", [])
    for field in getattr(value, "fields", []):
        yield from iter_tensors(field)


def bbox_payload(result) -> bytes:
    for tensor in iter_tensors(result):
        try:
            payload = tensor.copy_payload_bytes()
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


def metadata_json(boxes: list[dict], labels: list[str], classes: tuple[str, ...] = ()) -> str:
    objects = []
    allowed = set(classes)
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
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
    matches = []
    for box in boxes:
        class_id = int(box["class_id"])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        if label.lower() == wanted:
            matches.append(box)
    return max(matches, key=lambda box: box["score"], default=None)


def crop_box(frame: np.ndarray, box: dict) -> np.ndarray:
    x, y, w, h = [int(value) for value in box["bbox"]]
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"invalid person crop: {box['bbox']}")
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def request_person_crop_comment(crop_rgb: np.ndarray, cfg: Config) -> str:
    crop = rgb_to_bgr(crop_rgb)
    ok, encoded = cv2.imencode(".jpg", crop)
    if not ok:
        raise RuntimeError("failed to encode person crop")

    image = base64.b64encode(encoded.tobytes()).decode("ascii")
    payload = {
        "model": cfg.openai_model,
        "stream": True,
        "max_tokens": cfg.openai_max_tokens,
        "messages": [
            {"role": "system", "content": cfg.openai_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This image crop came from a YOLO person detection. "
                            "Comment on the visible situation."
                        ),
                    },
                ],
            },
        ],
    }

    url = f"http://{cfg.openai_host}:{cfg.openai_port}/v1/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response_text = ""
    with request.urlopen(req, timeout=cfg.openai_timeout_seconds) as response:
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
