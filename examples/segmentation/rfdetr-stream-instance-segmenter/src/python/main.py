#!/usr/bin/env python3
"""RF-DETR-Seg (432) single-camera RTSP Insight example using pyneat.

The model ships as a two-stage split -- INT8 backbone (MLA) and a BF16
transformer + seg-head (MLA) -- with a host-side top-k+gather hop between
them. The backbone runs embedded in the same async graph as the RTSP decode
and the (passthrough) video sender; a bridge thread does the top-k+gather and
feeds the transformer, which runs as a separately-built Runner. This mirrors
the object-detection RF-DETR example in this repo
(examples/object-detection/rfdetr-object-detector), adapted for the
segmentation model's extra mask output and third (now-removed) compiled
top-k stage: the top-k .so this model shipped with was verified bit-exact
against a plain stable argsort-by-score + gather on captured device tensors,
so it is not needed (and dropping it also drops the subprocess this port
used to keep TVM out of the same process as pyneat).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_LABELS = DEFAULT_CONFIG.parent / "coco_label.txt"

# ── model constants (fixed by the compiled RF-DETR-Seg 432 backbone/transformer) ──
IMAGE_H = IMAGE_W = 432
NUM_QUERIES = 200
NUM_CLASSES = 91
MASK_HW = 108
COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

BACKBONE_NAME = "rfdetr_seg_432_simplified_backbone_before_topk_base_mpk"
TRANSFORMER_NAME = "rfdetr_seg_432_simplified_transformer_after_gather_base_mpk"

# MetadataSender rejects a payload above 65507 bytes, and pyneat raises on the rejection. Half of
# that leaves room for the envelope and keeps the datagram count low enough for Insight to
# reassemble within its 250 ms window.
METADATA_BYTE_BUDGET = 32768

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class OutputConfig:
    save_dir: str
    save_every: int
    mask_alpha: float
    mask_threshold: float
    draw_boxes: bool


@dataclass(frozen=True)
class AppConfig:
    model_root: str
    labels_path: Path
    source_url: str
    tcp: bool = True
    latency_ms: int = 200
    source_fps: int = 0
    frames: int = 0
    score_threshold: float = 0.30
    max_detections: int = 24
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = ""
    video_port: int = 9000
    metadata_port: int = 9100
    # Foreground cutoff on the sigmoid-activated mask. The device seg-head's BF16
    # quantization affine-shifts the mask logits (empirically device_logit ~=
    # 0.59*onnx_logit - 3.07), so the ONNX true foreground boundary (logit > 0)
    # maps to a *raw* device logit around -2.4 to -3.0, i.e. sigmoid(-2.4..-3.0)
    # ~= 0.05-0.08 -- not the 0.5 a plain sigmoid boundary would suggest. Verify
    # against your own model export before trusting this default.
    output: OutputConfig = OutputConfig("", 0, 0.55, 0.08, True)


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return
    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR-Seg (432) single-camera RTSP Insight example"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def string_or(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return bool(value)


def validate_config(cfg: AppConfig) -> None:
    if not cfg.source_url:
        raise ValueError("source.url must be set")
    if not cfg.model_root:
        raise ValueError("model.path must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError("source.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.score_threshold <= 1.0:
        raise ValueError("inference.score_threshold must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.output.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.output.mask_alpha <= 1.0:
        raise ValueError("output.mask_alpha must be between 0 and 1")
    if not 0.0 <= cfg.output.mask_threshold <= 1.0:
        raise ValueError("output.mask_threshold must be between 0 and 1")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")

    cfg = AppConfig(
        model_root=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(DEFAULT_LABELS))),
        source_url=string_or(source, "url"),
        tcp=bool_or(source, "tcp", True),
        latency_ms=int_or(source, "latency_ms", 200),
        source_fps=int_or(source, "fps", 0),
        frames=int_or(inference, "frames", 0),
        score_threshold=float_or(inference, "score_threshold", 0.30),
        max_detections=int_or(inference, "max_detections", 24),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        output=OutputConfig(
            save_dir=string_or(output, "save_dir"),
            save_every=int_or(output, "save_every", 0),
            mask_alpha=float_or(output, "mask_alpha", 0.55),
            mask_threshold=float_or(output, "mask_threshold", 0.08),
            draw_boxes=bool_or(output, "draw_boxes", True),
        ),
    )
    validate_config(cfg)
    return cfg


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise RuntimeError(f"labels file does not exist: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {labels_path}")
    return labels


def sigmoid(x):
    # np.exp(-x) overflows for very negative logits (masks_dev holds raw pre-sigmoid values, which
    # run well past -700 in the background); clip first since sigmoid saturates long before that.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ── model options ────────────────────────────────────────────────────────
# The backbone consumes the decoder's NV12 output directly (fed by the graph, not the host), and
# EV74 resizes from the live stream resolution down to the compiled 432x432 input with a plain
# stretch (independent x/y scale, no letterbox padding) -- confirmed on real hardware to work at
# native stream resolution as well as at a pre-resized 432x432 input.
def backbone_options(stream_w: int, stream_h: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = stream_w
    opt.preprocess.input_max_height = stream_h
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.width = IMAGE_W
    opt.preprocess.resize.height = IMAGE_H
    opt.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    opt.preprocess.resize.scaling_type = "BILINEAR"
    opt.preprocess.normalize.enable = pyneat.AutoFlag.On
    opt.preprocess.normalize.mean = list(COCO_MEAN)
    opt.preprocess.normalize.stddev = list(COCO_STD)
    opt.preprocess.normalize.has_explicit_stats = True
    opt.processcvu.pre_run_target = "EV74"
    opt.processcvu.post_run_target = "A65"
    return opt


def transformer_options():
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.enable = pyneat.AutoFlag.Off
    opt.processcvu.pre_run_target = "A65"
    opt.processcvu.post_run_target = "A65"
    return opt


# ── host-side top-k + gather ────────────────────────────────────────────────
# Verified bit-exact against the model's compiled top-k .so on captured device tensors (stable
# argsort by score, descending, then gather the matching boxes) -- see the PR description for the
# comparison. Removing the compiled stage also removes the subprocess this port used to keep TVM
# out of the same process as pyneat (they conflict when imported together -- confirmed by an
# in-process crash during that verification).
def stable_topk_gather(reduce_scores, concat_boxes):
    flat_scores = np.asarray(reduce_scores, dtype=np.float32).reshape(-1)
    flat_boxes = np.asarray(concat_boxes, dtype=np.float32).reshape(-1, 4)
    if flat_scores.size < NUM_QUERIES or flat_boxes.shape[0] != flat_scores.size:
        raise RuntimeError("backbone score and box shapes do not match")
    indices = np.argsort(-flat_scores, kind="stable")[:NUM_QUERIES]
    return np.ascontiguousarray(flat_boxes[indices][None, ...])


def collect_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for field in getattr(sample, "fields", []):
        tensors.extend(collect_tensors(field))
    return tensors


def tensor_to_f32(tensor):
    return np.asarray(tensor.to_numpy(copy=True), dtype=np.float32)


def tensor_shape(tensor) -> tuple:
    return tuple(int(d) for d in tensor.shape)


class FrameOut:
    __slots__ = ("boxes", "logits", "masks_dev")

    def __init__(self, boxes, logits, masks_dev) -> None:
        self.boxes = boxes  # (200,4) normalized cx,cy,w,h in [0,1]
        self.logits = logits  # (200,91) raw (pre-sigmoid)
        self.masks_dev = masks_dev  # (108,108,200) raw (pre-sigmoid)


def split_backbone(sample):
    feature = reduce_ = concat = None
    for tensor in collect_tensors(sample):
        shape = tensor_shape(tensor)
        n = 1
        for d in shape:
            n *= d
        if shape == (1, 36, 36, 256):
            feature = tensor_to_f32(tensor).reshape(36, 36, 256)
        elif shape == (1, 1296, 4):
            concat = tensor_to_f32(tensor).reshape(1296, 4)
        elif n == 1296:
            reduce_ = tensor_to_f32(tensor).reshape(1296)
    if feature is None or reduce_ is None or concat is None:
        raise RuntimeError("backbone did not produce feature, reduce, and concat tensors")
    return feature, reduce_, concat


def split_transformer(sample) -> FrameOut:
    boxes = logits = masks_dev = None
    for tensor in collect_tensors(sample):
        shape = tensor_shape(tensor)
        n = 1
        for d in shape:
            n *= d
        if shape == (1, 200, 4):
            boxes = tensor_to_f32(tensor).reshape(200, 4)
        elif shape == (1, 200, 91):
            logits = tensor_to_f32(tensor).reshape(200, 91)
        elif shape == (108, 108, 200) or n == NUM_QUERIES * MASK_HW * MASK_HW:
            masks_dev = tensor_to_f32(tensor).reshape(108, 108, 200)
    if boxes is None or logits is None or masks_dev is None:
        raise RuntimeError("transformer did not produce box, class, and mask tensors")
    return FrameOut(boxes, logits, masks_dev)


def copy_identity(source, target) -> None:
    for name in (
        "frame_id", "stream_id", "stream_label", "input_seq", "orig_input_seq",
        "pts_ns", "dts_ns", "duration_ns",
    ):
        setattr(target, name, getattr(source, name))
    target.attributes = source.attributes


def identity_key(sample) -> int:
    return sample.frame_id if sample.frame_id >= 0 else sample.input_seq


# ── per-query select + stretch-space box/mask projection ───────────────────
# The compiled graph resizes with a plain stretch (independent x/y scale, no
# letterbox padding), so, unlike the letterboxed examples in this category,
# mapping a normalized box or the 108x108 mask grid back to frame pixels is a
# plain per-axis scale.
def select_dets(fo: FrameOut, score_threshold: float, max_detections: int) -> list[dict]:
    scores = sigmoid(fo.logits)
    best_class = np.argmax(scores, axis=1)
    best_score = scores[np.arange(NUM_QUERIES), best_class]
    keep = [
        {"query": q, "score": float(best_score[q]), "class_id": int(best_class[q])}
        for q in range(NUM_QUERIES)
        if best_score[q] >= score_threshold
    ]
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep[:max_detections]


def frame_rect_for_query(fo: FrameOut, query: int, frame_shape: tuple[int, ...]) -> tuple:
    frame_h, frame_w = frame_shape[:2]
    cx, cy, bw, bh = (float(v) for v in fo.boxes[query])
    x0 = max(0, min(frame_w - 1, int(round((cx - bw / 2) * frame_w))))
    y0 = max(0, min(frame_h - 1, int(round((cy - bh / 2) * frame_h))))
    x1 = max(x0 + 1, min(frame_w, int(round((cx + bw / 2) * frame_w))))
    y1 = max(y0 + 1, min(frame_h, int(round((cy + bh / 2) * frame_h))))
    return x0, y0, x1, y1


def mask_for_query(masks_dev, query: int):
    """Sigmoid-activated 8-bit mask for `query`, gathered directly from the raw device
    (108,108,200) buffer so only the handful of kept queries are ever touched."""
    return (sigmoid(masks_dev[:, :, query]) * 255.0).clip(0, 255).astype(np.uint8)


def mask_rect_for_frame_rect(frame_rect: tuple, frame_shape: tuple[int, ...]) -> tuple:
    """The 108x108 mask grid covers the whole stretched 432x432 model input with no
    padding, so a frame rectangle reaches it through a plain per-axis scale."""
    frame_h, frame_w = frame_shape[:2]
    sx, sy = MASK_HW / frame_w, MASK_HW / frame_h
    fx0, fy0, fx1, fy1 = frame_rect
    x0 = max(0, min(MASK_HW - 1, int(np.floor(fx0 * sx))))
    y0 = max(0, min(MASK_HW - 1, int(np.floor(fy0 * sy))))
    x1 = max(x0 + 1, min(MASK_HW, int(np.ceil(fx1 * sx))))
    y1 = max(y0 + 1, min(MASK_HW, int(np.ceil(fy1 * sy))))
    return x0, y0, x1, y1


def project_mask_roi(mask108, frame_rect: tuple, frame_shape: tuple[int, ...]):
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape)
    return cv2.resize(mask108[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
        (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
        (52, 147, 26), (187, 212, 0), (255, 194, 0), (168, 153, 44),
    ]
    return palette[max(0, class_id) % len(palette)]


def draw_box(frame, rect: tuple, class_id: int, score: float, labels: list[str]) -> None:
    x0, y0, x1, y1 = rect
    color = class_color(class_id)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        frame, f"{class_name(labels, class_id)} {score:.2f}", (x0, max(0, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )


def overlay_segmentation(frame, fo: FrameOut, detections: list[dict], labels: list[str],
                         cfg: OutputConfig):
    annotated = frame.copy()
    for det in detections:
        rect = frame_rect_for_query(fo, det["query"], annotated.shape)
        mask108 = mask_for_query(fo.masks_dev, det["query"])
        roi_mask = project_mask_roi(mask108, rect, annotated.shape)
        _, binary_mask = cv2.threshold(roi_mask, cfg.mask_threshold * 255.0, 255, cv2.THRESH_BINARY)
        x0, y0, x1, y1 = rect
        if cv2.countNonZero(binary_mask) > 0:
            color = class_color(det["class_id"])
            roi = annotated[y0:y1, x0:x1]
            mask_color = np.full(roi.shape, color, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0)
            cv2.copyTo(blended, binary_mask, roi)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, color, 2, cv2.LINE_8)
        if cfg.draw_boxes:
            draw_box(annotated, rect, det["class_id"], det["score"], labels)
    return annotated


def mask_polygon(mask108, frame_rect: tuple, frame_shape: tuple[int, ...],
                 threshold: float) -> list[list[int]]:
    roi = project_mask_roi(mask108, frame_rect, frame_shape)
    _, binary = cv2.threshold(roi, threshold * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    x0, y0 = frame_rect[0], frame_rect[1]
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]


def build_metadata_segments(fo: FrameOut, detections: list[dict], labels: list[str],
                            frame_shape: tuple[int, ...], mask_threshold: float) -> list[dict]:
    segments = []
    for det in detections:
        rect = frame_rect_for_query(fo, det["query"], frame_shape)
        mask108 = mask_for_query(fo.masks_dev, det["query"])
        polygon = mask_polygon(mask108, rect, frame_shape, mask_threshold)
        if not polygon:
            continue
        x0, y0, x1, y1 = rect
        segments.append(
            {
                "id": f"seg_{len(segments) + 1}",
                "label": class_name(labels, det["class_id"]),
                "confidence": det["score"],
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "mask_format": "polygon",
                "mask": polygon,
            }
        )
    return segments


def encode_segments(segments: list[dict]) -> tuple[str, int]:
    ordered = sorted(segments, key=lambda segment: segment["confidence"], reverse=True)
    kept = []
    total = len('{"segments":[]}')
    for segment in ordered:
        entry_bytes = len(json.dumps(segment, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":")), len(ordered) - len(kept)


# ── RTSP source (encoded passthrough + separate decode) + Insight video sender ──
def probe_ffprobe(cfg: AppConfig) -> tuple[int, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-rw_timeout", "5000000", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate", "-of", "default=nw=1",
    ]
    if cfg.tcp:
        cmd.extend(["-rtsp_transport", "tcp"])
    cmd.append(cfg.source_url)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0, 0
    if result.returncode != 0:
        return 0, 0, 0
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    def fps_from_rate(value: str) -> int:
        if not value or value in {"0/0", "0/1"}:
            return 0
        num, _, den = value.partition("/")
        try:
            fps = float(num) / float(den) if den else float(num)
        except (ValueError, ZeroDivisionError):
            return 0
        return int(round(fps)) if fps > 0 else 0

    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )
    try:
        width = int(values.get("width") or 0)
        height = int(values.get("height") or 0)
    except ValueError:
        width = height = 0
    return width, height, fps


def probe_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    width, height, fps = probe_ffprobe(cfg)
    if cfg.source_fps > 0:
        fps = cfg.source_fps
    if width <= 0 or height <= 0 or fps <= 0:
        cap = cv2.VideoCapture(cfg.source_url)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open RTSP source for probing: {cfg.source_url}")
        width = width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = fps or int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
        cap.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError("failed to resolve RTSP width, height, and FPS; check source.url")
    return width, height, fps


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width, height = tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.instances = 0
        self.dropped_segments = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.metadata_ms = 0.0

    def add(self, pull_ms: float, decode_ms: float, metadata_ms: float, instance_count: int,
           dropped: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.instances += instance_count
        self.dropped_segments += dropped
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.metadata_ms += metadata_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed_ms = max(time_ms() - self.start_ms, 1e-6)
        frames = float(self.frames)
        print(
            f"[profile] frames={self.frames} output_fps={self.frames * 1000.0 / elapsed_ms} "
            f"avg_pull_ms={self.pull_ms / frames} avg_decode_ms={self.decode_ms / frames} "
            f"avg_metadata_ms={self.metadata_ms / frames} "
            f"avg_instances={self.instances / frames} dropped_segments={self.dropped_segments}",
            flush=True,
        )
        self.reset()


def run(cfg: AppConfig) -> int:
    width, height, fps = probe_source_geometry(cfg)
    labels = load_labels(cfg.labels_path)
    save_frames = bool(cfg.output.save_dir)
    if save_frames:
        Path(cfg.output.save_dir).mkdir(parents=True, exist_ok=True)

    backbone = pyneat.Model(f"{cfg.model_root}/{BACKBONE_NAME}", backbone_options(width, height))
    transformer = pyneat.Model(f"{cfg.model_root}/{TRANSFORMER_NAME}", transformer_options())

    # ── RTSP source: keep the encoded bitstream for a true passthrough to Insight (no
    # decode-then-re-encode round trip), and decode once, separately, for the model. ──
    encoded_options = pyneat.RtspEncodedInputOptions()
    encoded_options.url = cfg.source_url
    encoded_options.codec = pyneat.RtspCodec.H264
    encoded_options.latency_ms = cfg.latency_ms
    encoded_options.tcp = cfg.tcp
    encoded_options.source_fps = fps
    encoded_options.fallback_h264_width = width
    encoded_options.fallback_h264_height = height
    source = pyneat.groups.rtsp_encoded_input(encoded_options)

    decode_options = pyneat.SimaDecodeOptions()
    decode_options.type = pyneat.SimaDecodeType.H264
    decode_options.out_format = pyneat.Format.NV12
    decode_options.raw_output = True
    decode_options.dec_width = width
    decode_options.dec_height = height
    decode_options.dec_fps = fps
    decoder = pyneat.Graph("decoder")
    decoder.add(pyneat.nodes.sima_decode(decode_options))

    video_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H264)
    video_options.host = cfg.insight_host
    video_options.video_port_base = cfg.video_port
    video_options.channel = 0
    video_options.async_ = True
    video = pyneat.groups.video_sender(video_options)

    # ── backbone runs embedded in the same async graph as decode/video, pipelined two
    # frames deep, instead of behind a manual push/pull call in the main loop. ──
    backbone_graph = backbone.graph()
    backbone_output = pyneat.Graph("backbone_output")
    backbone_output.add(pyneat.nodes.output("backbone", pyneat.OutputOptions.every_frame(2)))

    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.max_inflight_per_stream = 2
    link.stream_id = "stream0"

    graph = pyneat.Graph("rfdetr_seg")
    graph.connect(source, decoder)
    graph.connect(source, video)
    graph.connect(decoder, backbone_graph, link)
    graph.connect(backbone_graph, backbone_output)

    frame_output = pyneat.Graph("frame_output")
    if save_frames:
        frame_output.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(2)))
        graph.connect(decoder, frame_output)

    if cfg.profile:
        print(f"Backend:\n{graph.describe_backend()}")

    transformer_run_options = pyneat.RunOptions()
    transformer_run_options.preset = pyneat.RunPreset.Realtime
    transformer_run_options.queue_depth = 2
    transformer_run_options.overflow_policy = pyneat.OverflowPolicy.Block
    transformer_run_options.output_memory = pyneat.OutputMemory.Owned
    dummy_inputs = [
        pyneat.Tensor.from_numpy(np.zeros((36, 36, 256), dtype=np.float32), copy=True),
        pyneat.Tensor.from_numpy(np.zeros((1, NUM_QUERIES, 4), dtype=np.float32), copy=True),
    ]
    transformer_runner = transformer.build(
        dummy_inputs, route_options=pyneat.ModelRouteOptions(), run_options=transformer_run_options
    )

    source_run_options = pyneat.RunOptions()
    source_run_options.preset = pyneat.RunPreset.Realtime
    source_run_options.queue_depth = 3
    source_run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    source_run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    source_run_options.advanced.prepare_output_cpu_visible = True
    source_run = graph.build(source_run_options)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = 0
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"source={cfg.source_url} stream={width}x{height}@{fps} insight={cfg.insight_host} "
        f"video={video_options.video_port} metadata={metadata_sender.metadata_port()} channel=0",
        flush=True,
    )

    # ── bridge thread: pulls the backbone's output, does the (now host-side) top-k +
    # gather, and feeds the transformer. Runs concurrently with the main loop pulling
    # transformer output below, so the three stages overlap instead of running one
    # frame fully to completion before the next starts. ──
    stop = threading.Event()
    bridge_error: list[BaseException] = []
    identity_lock = threading.Lock()
    source_pts: dict[int, int] = {}
    pending_frames: dict[int, object] = {}

    def bridge() -> None:
        try:
            while not stop.is_set():
                sample = source_run.pull("backbone", 500)
                if sample is None:
                    continue
                feature, reduce_, concat = split_backbone(sample)
                gathered = stable_topk_gather(reduce_, concat)

                # Fresh CPU tensors, not the live device-backed backbone output: pushing the
                # backbone's own output tensor straight into the transformer's 2-input stage has
                # been observed to scramble that stage's packed multi-input buffer on this NEAT
                # build.
                feat_in = pyneat.Tensor.from_numpy(feature, copy=True)
                gath_in = pyneat.Tensor.from_numpy(gathered, copy=True)
                transformer_sample = pyneat.Sample()
                transformer_sample.kind = pyneat.SampleKind.TensorSet
                transformer_sample.tensors = [feat_in, gath_in]
                copy_identity(sample, transformer_sample)

                key = identity_key(sample)
                with identity_lock:
                    source_pts[key] = sample.pts_ns
                    if len(source_pts) > 8:
                        source_pts.pop(next(iter(source_pts)))

                if save_frames:
                    frame_sample = source_run.pull("frame", 0)
                    if frame_sample is not None:
                        tensors = collect_tensors(frame_sample)
                        if tensors:
                            with identity_lock:
                                pending_frames[identity_key(frame_sample)] = tensor_bgr_from_decoded(
                                    tensors[0]
                                )
                                if len(pending_frames) > 8:
                                    pending_frames.pop(next(iter(pending_frames)))

                if not transformer_runner.push_samples(transformer_sample):
                    if not stop.is_set():
                        raise RuntimeError("transformer input closed")
                    break
        except BaseException as exc:  # noqa: BLE001 - surface to main thread, then stop
            if not stop.is_set():
                bridge_error.append(exc)
            stop.set()

    previous_handlers = {
        signum: signal.signal(signum, lambda *_: stop.set())
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    worker = threading.Thread(target=bridge, name="rfdetr-seg-bridge", daemon=True)
    worker.start()

    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    dropped_total = 0
    try:
        while not stop.is_set() and (cfg.frames <= 0 or processed < cfg.frames):
            pull_start = time_ms()
            sample = transformer_runner.pull(timeout_ms=500)
            pull_end = time_ms()
            if not collect_tensors(sample):
                continue

            decode_start = time_ms()
            fo = split_transformer(sample)
            detections = select_dets(fo, cfg.score_threshold, cfg.max_detections)
            decode_end = time_ms()

            metadata_start = time_ms()
            data_json, dropped = encode_segments(
                build_metadata_segments(fo, detections, labels, (height, width),
                                        cfg.output.mask_threshold)
            )
            with identity_lock:
                pts_ns = source_pts.pop(identity_key(sample), sample.pts_ns)
            timestamp_ms = pts_ns // 1_000_000 if pts_ns >= 0 else -1
            frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
            if not metadata_sender.send_metadata("segmentation", data_json, timestamp_ms, frame_id):
                print("[warn] insight metadata send failed", file=sys.stderr)
            metadata_end = time_ms()
            if dropped > 0 and dropped_total == 0:
                print(f"[warn] metadata byte budget exceeded, dropped {dropped} segments",
                     file=sys.stderr)
            dropped_total += dropped

            processed += 1
            if save_frames and cfg.output.save_every > 0 and processed % cfg.output.save_every == 0:
                with identity_lock:
                    frame = pending_frames.pop(identity_key(sample), None)
                if frame is not None:
                    annotated = overlay_segmentation(frame, fo, detections, labels, cfg.output)
                    out_path = Path(cfg.output.save_dir) / f"frame_{processed}.jpg"
                    if not cv2.imwrite(str(out_path), annotated):
                        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)
            profile.add(pull_end - pull_start, decode_end - decode_start,
                       metadata_end - metadata_start, len(detections), dropped)
        if bridge_error:
            raise bridge_error[0]
    finally:
        stop.set()
        source_run.stop()
        transformer_runner.close()
        worker.join(timeout=5)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _ = (backbone, graph)

    profile.flush()
    print(
        f"processed={processed} dropped_segments={dropped_total} "
        f"video_sender={cfg.insight_host}:{video_options.video_port}"
    )
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config}")
            return 0

        load_runtime_dependencies()
        return 0 if run(cfg) > 0 else 3
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
