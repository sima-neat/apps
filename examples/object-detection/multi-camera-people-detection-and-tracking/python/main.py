#!/usr/bin/env python3
"""Python-first multi-camera people detection and tracking example.

Architecture (per stream):
  [producer thread]  source_run.pull() -> frame_queue
  [infer thread]     frame_queue -> cpu_preproc -> detect_run.push/pull (pipelined) -> result_queue
  [publish thread]   result_queue -> parse_bbox -> filter -> tracker -> OptiView video/json + optional save
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import glob
import json
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from typing import Any

from tracker import PeopleTracker, TrackedDetection


@dataclass(frozen=True)
class AppConfig:
    model: str
    rtsp_urls: list[str]
    output_dir: str | None
    frames: int
    optiview_host: str
    optiview_video_port_base: int
    optiview_json_port_base: int
    fps: int
    bitrate_kbps: int
    save_every: int
    profile: bool
    person_class_id: int
    detection_threshold: float | None
    nms_iou_threshold: float | None
    top_k: int | None
    tracker_iou_threshold: float
    tracker_max_missing: int
    latency_ms: int
    tcp: bool


@dataclass(frozen=True)
class RtspProbe:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class RuntimeModules:
    cv2: Any
    np: Any
    pyneat: Any


@dataclass(frozen=True)
class QuantTessCpuPreproc:
    width: int
    height: int
    aspect_ratio: bool
    padding_type: str


@dataclass
class StreamMetrics:
    pulled: int = 0
    processed: int = 0
    detections: int = 0
    saved: int = 0
    source_time_s: float = 0.0
    preproc_time_s: float = 0.0
    push_time_s: float = 0.0
    pull_wait_s: float = 0.0
    track_time_s: float = 0.0
    overlay_time_s: float = 0.0
    write_time_s: float = 0.0
    total_loop_time_s: float = 0.0
    wall_started_at_s: float | None = None
    wall_last_processed_at_s: float | None = None
    _interval_source_s: float = 0.0
    _interval_preproc_s: float = 0.0
    _interval_push_s: float = 0.0
    _interval_pull_s: float = 0.0
    _interval_output_s: float = 0.0
    _interval_loop_s: float = 0.0
    _interval_frames: int = 0
    _interval_wall_started_at_s: float | None = None


@dataclass
class FramePacket:
    """Carries a decoded BGR frame between producer and infer threads."""
    frame: Any
    frame_index: int
    source_time_s: float


@dataclass
class ResultPacket:
    """Carries infer results + original frame between infer and overlay threads."""
    frame: Any
    frame_index: int
    bbox_payload: bytes | None
    source_time_s: float
    preproc_time_s: float
    push_time_s: float
    pull_wait_s: float


@dataclass
class StreamRuntime:
    index: int
    url: str
    probe: RtspProbe
    runtime: RuntimeModules
    model: Any
    quant_preproc: QuantTessCpuPreproc
    source_session: Any
    source_run: Any
    detect_session: Any
    detect_run: Any
    video_session: Any
    video_run: Any
    json_sender: Any
    tracker: PeopleTracker
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    error: Exception | None = None


_RUNTIME_MODULES: RuntimeModules | None = None


def optiview_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def optiview_json_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def filter_person_detections(boxes: list[dict], person_class_id: int = 0) -> list[dict]:
    return [box for box in boxes if int(box.get("class_id", -1)) == int(person_class_id)]


def sample_output_path(output_dir: Path, stream_index: int, frame_index: int) -> Path:
    return output_dir / f"stream_{stream_index}" / f"frame_{frame_index:06d}.jpg"


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


def effective_writer_fps(cfg: AppConfig, probe: RtspProbe) -> int:
    return cfg.fps if cfg.fps > 0 else max(probe.fps, 10)


def load_runtime_modules() -> RuntimeModules:
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is not None:
        return _RUNTIME_MODULES

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import cv2
        import numpy as np
        import pyneat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "runtime dependencies are missing; install numpy, OpenCV, and pyneat "
            "or run inside the NEAT Python environment"
        ) from exc

    _RUNTIME_MODULES = RuntimeModules(cv2=cv2, np=np, pyneat=pyneat)
    return _RUNTIME_MODULES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-camera people detection and tracking example."
    )
    parser.add_argument("--model", required=True, help="Path to YOLOv8 model pack.")
    parser.add_argument(
        "--rtsp",
        action="append",
        required=True,
        dest="rtsp",
        help="RTSP input URL. Repeat for each camera.",
    )
    parser.add_argument("--output", help="Optional output directory for sampled overlay frames.")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Frames to process per stream. Default 0 means unlimited.",
    )
    parser.add_argument(
        "--optiview-host",
        default="127.0.0.1",
        help="OptiView host for live video and JSON metadata.",
    )
    parser.add_argument(
        "--optiview-video-port-base",
        type=int,
        default=9000,
        help="OptiView UDP video port base; stream i uses video_port_base + i.",
    )
    parser.add_argument(
        "--optiview-json-port-base",
        type=int,
        default=9100,
        help="OptiView UDP JSON port base; stream i uses json_port_base + i.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Output FPS override. Default uses the probed stream FPS or 10.",
    )
    parser.add_argument(
        "--bitrate-kbps",
        type=int,
        default=2500,
        help="H264 encoder bitrate for OptiView live video output.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="Save every Nth overlay frame when --output is set.",
    )
    parser.add_argument("--profile", action="store_true", help="Print timing summary.")
    parser.add_argument(
        "--person-class-id",
        type=int,
        default=0,
        help="Class id to track as people.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=None,
        help="Detection score threshold for box decode. Default preserves the model-pack value.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=None,
        help="NMS IoU threshold for box decode. Default preserves the model-pack value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Maximum detections per frame. Default preserves the model-pack value.",
    )
    parser.add_argument(
        "--tracker-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracker association.",
    )
    parser.add_argument(
        "--tracker-max-missing",
        type=int,
        default=15,
        help="Frames to keep a missing track alive.",
    )
    parser.add_argument("--latency-ms", type=int, default=200, help="RTSP receive latency in ms.")
    parser.add_argument("--tcp", action="store_true", help="Use TCP RTSP transport.")
    return parser


def parse_config(argv: list[str] | None = None) -> AppConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.optiview_video_port_base <= 0:
        parser.error("--optiview-video-port-base must be > 0")
    if args.optiview_json_port_base <= 0:
        parser.error("--optiview-json-port-base must be > 0")
    if args.output and args.save_every <= 0:
        parser.error("--save-every must be > 0 when --output is set")
    if args.frames < 0:
        parser.error("--frames must be >= 0")
    if args.fps < 0:
        parser.error("--fps must be >= 0")
    if args.bitrate_kbps <= 0:
        parser.error("--bitrate-kbps must be > 0")
    if args.detection_threshold is not None and not (0.0 <= args.detection_threshold <= 1.0):
        parser.error("--detection-threshold must be between 0 and 1")
    if args.nms_iou_threshold is not None and not (0.0 <= args.nms_iou_threshold <= 1.0):
        parser.error("--nms-iou-threshold must be between 0 and 1")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be > 0")
    if not (0.0 <= args.tracker_iou_threshold <= 1.0):
        parser.error("--tracker-iou-threshold must be between 0 and 1")
    if args.tracker_max_missing < 0:
        parser.error("--tracker-max-missing must be >= 0")
    if args.latency_ms < 0:
        parser.error("--latency-ms must be >= 0")

    output_dir = str(Path(args.output)) if args.output else None
    return AppConfig(
        model=args.model,
        rtsp_urls=list(args.rtsp),
        output_dir=output_dir,
        frames=args.frames,
        optiview_host=args.optiview_host,
        optiview_video_port_base=args.optiview_video_port_base,
        optiview_json_port_base=args.optiview_json_port_base,
        fps=args.fps,
        bitrate_kbps=args.bitrate_kbps,
        save_every=args.save_every,
        profile=bool(args.profile),
        person_class_id=args.person_class_id,
        detection_threshold=args.detection_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        top_k=args.top_k,
        tracker_iou_threshold=args.tracker_iou_threshold,
        tracker_max_missing=args.tracker_max_missing,
        latency_ms=args.latency_ms,
        tcp=bool(args.tcp),
    )


# ---------------------------------------------------------------------------
# Sample / tensor helpers
# ---------------------------------------------------------------------------


def iter_tensors(sample: Any):
    if getattr(sample, "kind", None) == load_runtime_modules().pyneat.SampleKind.Tensor and getattr(
        sample, "tensor", None
    ) is not None:
        yield sample.tensor
    for field in getattr(sample, "fields", []):
        yield from iter_tensors(field)


def first_tensor(sample: Any) -> Any:
    if sample is None:
        return None
    if getattr(sample, "kind", None) == load_runtime_modules().pyneat.SampleKind.Tensor and getattr(
        sample, "tensor", None
    ) is not None:
        return sample.tensor
    for tensor in iter_tensors(sample):
        return tensor
    return None


def extract_bbox_payload(sample: Any) -> bytes | None:
    stack = [sample]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(getattr(current, "fields", []))))
        if getattr(current, "kind", None) != load_runtime_modules().pyneat.SampleKind.Tensor:
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


_DFL_BINS_16 = None


def _get_dfl_bins():
    global _DFL_BINS_16
    if _DFL_BINS_16 is None:
        _DFL_BINS_16 = load_runtime_modules().np.arange(16, dtype=load_runtime_modules().np.float32)
    return _DFL_BINS_16


def _tensor_to_hwc_f32(tensor: Any) -> Any:
    np = load_runtime_modules().np
    arr = np.asarray(tensor.to_numpy(copy=True)).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def _iou_xyxy(a, b) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def decode_yolov8_boxes(
    sample: Any,
    infer_size: int,
    img_w: int,
    img_h: int,
    min_score: float = 0.5,
    nms_iou: float = 0.5,
    max_det: int = 100,
) -> list[dict]:
    """Decode YOLOv8 bounding boxes from DetessDequant output (6 tensors)."""
    np = load_runtime_modules().np
    dfl_bins = _get_dfl_bins()

    tensors = list(iter_tensors(sample))
    if len(tensors) < 6:
        return []
    regs = [_tensor_to_hwc_f32(tensors[i]) for i in range(3)]
    clss = [_tensor_to_hwc_f32(tensors[i]) for i in range(3, 6)]

    # Letterbox scale and padding.
    scale = min(infer_size / img_w, infer_size / img_h)
    pad_x = (infer_size - img_w * scale) / 2.0
    pad_y = (infer_size - img_h * scale) / 2.0

    # Sigmoid threshold in logit domain.
    min_logit = -np.log(max(1.0 / min_score - 1.0, 1e-9)) if min_score > 0 else -60.0

    boxes_all: list = []
    scores_all: list = []
    class_all: list = []
    for reg, cls in zip(regs, clss):
        h, w, c = reg.shape
        if c < 64:
            continue
        stride = infer_size / float(h)

        cls_flat = cls.reshape(-1, cls.shape[2])
        best_class = np.argmax(cls_flat, axis=1).astype(np.int32)
        flat_idx = np.arange(cls_flat.shape[0], dtype=np.int32)
        best_logit = cls_flat[flat_idx, best_class]

        keep = best_logit >= min_logit
        if not np.any(keep):
            continue

        kept_idx = flat_idx[keep]
        kept_class = best_class[keep]
        kept_logit = best_logit[keep]
        kept_score = 1.0 / (1.0 + np.exp(-np.clip(kept_logit, -60.0, 60.0)))

        reg_flat = reg.reshape(-1, 4, 16)[kept_idx]
        reg_flat = reg_flat - np.max(reg_flat, axis=2, keepdims=True)
        reg_exp = np.exp(reg_flat)
        reg_prob = reg_exp / np.sum(reg_exp, axis=2, keepdims=True)
        dists = np.sum(reg_prob * dfl_bins[None, None, :], axis=2) * stride

        ys = kept_idx // w
        xs = kept_idx % w
        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride

        x1 = np.clip((cx - dists[:, 0] - pad_x) / scale, 0.0, float(img_w))
        y1 = np.clip((cy - dists[:, 1] - pad_y) / scale, 0.0, float(img_h))
        x2 = np.clip((cx + dists[:, 2] - pad_x) / scale, 0.0, float(img_w))
        y2 = np.clip((cy + dists[:, 3] - pad_y) / scale, 0.0, float(img_h))

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            continue

        boxes_all.append(np.stack((x1[valid], y1[valid], x2[valid], y2[valid]), axis=1))
        scores_all.append(kept_score[valid].astype(np.float32))
        class_all.append(kept_class[valid].astype(np.int32))

    if not boxes_all:
        return []

    boxes = np.concatenate(boxes_all, axis=0)
    scores = np.concatenate(scores_all, axis=0)
    class_ids = np.concatenate(class_all, axis=0)

    # NMS per class.
    order = np.argsort(-scores, kind="stable")
    keep_idx: list[int] = []
    keep_by_class: dict[int, list[int]] = {}
    for i in order:
        ci = int(class_ids[i])
        suppressed = False
        for k in keep_by_class.get(ci, []):
            if _iou_xyxy(boxes[k], boxes[i]) > nms_iou:
                suppressed = True
                break
        if suppressed:
            continue
        ii = int(i)
        keep_idx.append(ii)
        keep_by_class.setdefault(ci, []).append(ii)
        if len(keep_idx) >= max_det:
            break

    return [
        dict(
            x1=float(boxes[i, 0]),
            y1=float(boxes[i, 1]),
            x2=float(boxes[i, 2]),
            y2=float(boxes[i, 3]),
            score=float(scores[i]),
            class_id=int(class_ids[i]),
        )
        for i in keep_idx
    ]


def tensor_bgr_from_sample(runtime: RuntimeModules, sample: Any):
    tensor = first_tensor(sample)
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


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def class_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id + 17) % 256),
        int((97 * track_id + 73) % 256),
        int((53 * track_id + 191) % 256),
    )


def draw_tracked_people(runtime: RuntimeModules, frame, tracked: list[TrackedDetection]):
    cv2 = runtime.cv2
    for det in tracked:
        color = class_color(det.track_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        text = f"person #{det.track_id} {det.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return frame


def save_overlay_frame(
    runtime: RuntimeModules,
    output_dir: Path | None,
    stream_index: int,
    frame_index: int,
    frame,
    save_every: int,
) -> bool:
    if output_dir is None:
        return False
    if save_every <= 0 or frame_index % save_every != 0:
        return False
    out_path = sample_output_path(output_dir, stream_index, frame_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(runtime.cv2.imwrite(str(out_path), frame))


# ---------------------------------------------------------------------------
# RTSP probe
# ---------------------------------------------------------------------------


def probe_rtsp(url: str) -> RtspProbe:
    runtime = load_runtime_modules()
    cap = runtime.cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(runtime.cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(runtime.cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(runtime.cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to probe RTSP frame size: {url}")
    return RtspProbe(width=width, height=height, fps=max(0, fps))


# ---------------------------------------------------------------------------
# Model (built once, shared across streams)
# ---------------------------------------------------------------------------


def build_model(runtime: RuntimeModules, cfg: AppConfig):
    pyneat = runtime.pyneat
    opt = pyneat.ModelOptions()
    opt.media_type = "application/vnd.simaai.tensor"
    opt.format = ""
    return pyneat.Model(cfg.model, opt)


def build_quanttess_cpu_preproc(runtime: RuntimeModules, model: Any) -> QuantTessCpuPreproc:
    pyneat = runtime.pyneat
    pre = pyneat.PreprocOptions(model)
    cfg_json = dict(getattr(pre, "config_json", None) or {})
    width = int(cfg_json.get("output_width") or cfg_json.get("input_width") or 640)
    height = int(cfg_json.get("output_height") or cfg_json.get("input_height") or 640)
    return QuantTessCpuPreproc(
        width=width,
        height=height,
        aspect_ratio=bool(cfg_json.get("aspect_ratio", False)),
        padding_type=str(cfg_json.get("padding_type", "CENTER")).upper(),
    )


def cpu_quanttess_input(runtime: RuntimeModules, frame_bgr, contract: QuantTessCpuPreproc):
    cv2 = runtime.cv2
    np = runtime.np
    src_h, src_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if contract.aspect_ratio:
        scale = min(contract.width / src_w, contract.height / src_h)
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
    else:
        scaled_w = contract.width
        scaled_h = contract.height

    pad_x = 0
    pad_y = 0
    if contract.padding_type == "CENTER":
        pad_x = (contract.width - scaled_w) // 2
        pad_y = (contract.height - scaled_h) // 2

    resized = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    quant_input = np.zeros((contract.height, contract.width, 3), dtype=np.float32)
    quant_input[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = (
        resized.astype(np.float32) / 255.0
    )
    return np.ascontiguousarray(quant_input)


# ---------------------------------------------------------------------------
# Session builders
# ---------------------------------------------------------------------------


def build_source_run(runtime: RuntimeModules, cfg: AppConfig, url: str, probe: RtspProbe):
    pyneat = runtime.pyneat
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = cfg.latency_ms
    ro.tcp = cfg.tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.out_format = "BGR"
    ro.decoder_raw_output = False
    ro.auto_caps_from_stream = True
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.fallback_h264_width = probe.width
    ro.fallback_h264_height = probe.height
    if probe.fps > 0:
        ro.fallback_h264_fps = probe.fps
    ro.output_caps.enable = True
    ro.output_caps.format = "BGR"
    ro.output_caps.width = probe.width
    ro.output_caps.height = probe.height
    if probe.fps > 0:
        ro.output_caps.fps = probe.fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    session = pyneat.Session()
    session.add(pyneat.groups.rtsp_decoded_input(ro))
    session.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = session.build(run_opt)
    return session, run


_YOLOV8_BOXDECODE_DEFAULTS = {
    "detection_threshold": 0.6,
    "nms_iou_threshold": 0.5,
    "topk": 24,
}

_SOURCE_STARTUP_PULL_TIMEOUT_MS = 50000
_SOURCE_PULL_TIMEOUT_MS = 10000
_SOURCE_STARTUP_STAGGER_S = 0.5


def build_detect_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    model: Any,
    probe: RtspProbe,
    quant_preproc: QuantTessCpuPreproc,
):
    pyneat = runtime.pyneat
    np = runtime.np

    # QuantTess + MLA + BoxDecode (hardware box decode on CVU).
    # SimaBoxDecode now emits num-buffers matching the model's CVU config.
    session = pyneat.Session()
    session.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    session.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    session.add(pyneat.groups.mla(model))
    session.add(
        pyneat.nodes.sima_box_decode(
            model,
            decode_type="yolov8",
            original_width=probe.width,
            original_height=probe.height,
            detection_threshold=(
                cfg.detection_threshold
                if cfg.detection_threshold is not None
                else _YOLOV8_BOXDECODE_DEFAULTS["detection_threshold"]
            ),
            nms_iou_threshold=(
                cfg.nms_iou_threshold
                if cfg.nms_iou_threshold is not None
                else _YOLOV8_BOXDECODE_DEFAULTS["nms_iou_threshold"]
            ),
            top_k=cfg.top_k if cfg.top_k is not None else _YOLOV8_BOXDECODE_DEFAULTS["topk"],
        )
    )
    session.add(pyneat.nodes.output())

    dummy = np.zeros((quant_preproc.height, quant_preproc.width, 3), dtype=np.float32)
    run = session.build(dummy)
    return session, run


def build_optiview_video_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
    stream_index: int,
):
    pyneat = runtime.pyneat
    np = runtime.np

    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "BGR"
    input_opt.use_simaai_pool = False

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(pyneat.nodes.video_convert())
    session.add(
        pyneat.nodes.h264_encode_sima(
            probe.width,
            probe.height,
            effective_writer_fps(cfg, probe),
            bitrate_kbps=cfg.bitrate_kbps,
            profile="baseline",
            level="4.1",
        )
    )
    udp_opt = pyneat.UdpH264OutputGroupOptions()
    udp_opt.payload_type = 96
    udp_opt.config_interval = 1
    udp_opt.udp_host = cfg.optiview_host
    udp_opt.udp_port = optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index)
    udp_opt.udp_sync = False
    udp_opt.udp_async = False
    session.add(pyneat.groups.udp_h264_output_group(udp_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 2
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return session, run


def build_optiview_json_output(
    runtime: RuntimeModules,
    cfg: AppConfig,
    stream_index: int,
):
    pyneat = runtime.pyneat

    channel = pyneat.OptiViewChannelOptions()
    channel.host = cfg.optiview_host
    channel.channel = stream_index
    channel.video_port_base = cfg.optiview_video_port_base
    channel.json_port_base = cfg.optiview_json_port_base
    return pyneat.OptiViewJsonOutput(channel)


# ---------------------------------------------------------------------------
# Stream setup (model shared, sessions built sequentially)
# ---------------------------------------------------------------------------


def create_stream_runtime(
    index: int,
    url: str,
    cfg: AppConfig,
    model: Any,
    quant_preproc: QuantTessCpuPreproc,
) -> StreamRuntime:
    runtime = load_runtime_modules()
    probe = probe_rtsp(url)
    source_session, source_run = build_source_run(runtime, cfg, url, probe)
    detect_session, detect_run = build_detect_run(runtime, cfg, model, probe, quant_preproc)
    video_session, video_run = build_optiview_video_run(runtime, cfg, probe, index)
    json_sender = build_optiview_json_output(runtime, cfg, index)
    tracker = PeopleTracker(
        iou_threshold=cfg.tracker_iou_threshold,
        max_missing_frames=cfg.tracker_max_missing,
    )
    return StreamRuntime(
        index=index,
        url=url,
        probe=probe,
        runtime=runtime,
        model=model,
        quant_preproc=quant_preproc,
        source_session=source_session,
        source_run=source_run,
        detect_session=detect_session,
        detect_run=detect_run,
        video_session=video_session,
        video_run=video_run,
        json_sender=json_sender,
        tracker=tracker,
    )


def close_stream_runtime(stream: StreamRuntime) -> None:
    for run in (stream.video_run, stream.detect_run, stream.source_run):
        try:
            if run is not None:
                run.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Queue helper
# ---------------------------------------------------------------------------


def put_keep_latest(q: queue.Queue, item: Any) -> None:
    """Non-blocking put; if full, drop oldest and retry."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


# ---------------------------------------------------------------------------
# Per-stream worker threads
# ---------------------------------------------------------------------------


def producer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    frame_q: queue.Queue,
    stop_event: threading.Event,
    startup_ready: threading.Event | None = None,
) -> None:
    """Pulls decoded BGR frames from RTSP source and enqueues them."""
    frame_index = 0
    empty_pulls = 0
    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and frame_index >= cfg.frames:
                return
            t0 = time.perf_counter()
            pull_timeout_ms = (
                _SOURCE_STARTUP_PULL_TIMEOUT_MS if frame_index == 0 else _SOURCE_PULL_TIMEOUT_MS
            )
            sample = stream.source_run.pull(timeout_ms=pull_timeout_ms)
            elapsed = time.perf_counter() - t0
            if sample is None:
                empty_pulls += 1
                if cfg.frames > 0 and empty_pulls >= 20:
                    raise RuntimeError(f"stream {stream.index} timed out waiting for RTSP frames")
                continue
            empty_pulls = 0
            frame = tensor_bgr_from_sample(stream.runtime, sample)
            put_keep_latest(frame_q, FramePacket(frame=frame, frame_index=frame_index, source_time_s=elapsed))
            if startup_ready is not None and frame_index == 0:
                startup_ready.set()
            frame_index += 1
    except Exception as exc:
        stream.error = exc
        stop_event.set()
        if startup_ready is not None:
            startup_ready.set()


def start_producer_threads_sequentially(
    producer_threads: list[threading.Thread],
    startup_events: list[threading.Event],
    stop_event: threading.Event,
    startup_timeout_ms: int = _SOURCE_STARTUP_PULL_TIMEOUT_MS,
    startup_stagger_s: float = _SOURCE_STARTUP_STAGGER_S,
) -> list[threading.Thread]:
    started_threads: list[threading.Thread] = []
    timeout_s = startup_timeout_ms / 1000.0

    for index, thread in enumerate(producer_threads):
        if stop_event.is_set():
            break
        thread.start()
        started_threads.append(thread)

        if not startup_events[index].wait(timeout_s):
            stop_event.set()
            break
        if stop_event.is_set():
            break
        if index + 1 < len(producer_threads) and startup_stagger_s > 0:
            time.sleep(startup_stagger_s)

    return started_threads


def infer_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    frame_q: queue.Queue,
    result_q: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """CPU preproc + push_and_pull on the detect session (GIL-released)."""
    runtime = stream.runtime
    detect_run = stream.detect_run

    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and stream.metrics.processed >= cfg.frames:
                return

            try:
                pkt = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            preproc_t0 = time.perf_counter()
            quant_input = cpu_quanttess_input(runtime, pkt.frame, stream.quant_preproc)
            preproc_elapsed = time.perf_counter() - preproc_t0

            # Run.run() releases GIL for the entire round-trip.
            roundtrip_t0 = time.perf_counter()
            det_sample = detect_run.run(quant_input, timeout_ms=50000)
            roundtrip_elapsed = time.perf_counter() - roundtrip_t0

            if det_sample is None:
                raise RuntimeError(f"stream {stream.index} detect run timed out")

            bbox_payload = extract_bbox_payload(det_sample)
            put_keep_latest(
                result_q,
                ResultPacket(
                    frame=pkt.frame,
                    frame_index=pkt.frame_index,
                    bbox_payload=bbox_payload,
                    source_time_s=pkt.source_time_s,
                    preproc_time_s=preproc_elapsed,
                    push_time_s=0.0,
                    pull_wait_s=roundtrip_elapsed,
                ),
            )

    except Exception as exc:
        stream.error = exc
        stop_event.set()


def publish_thread(
    stream: StreamRuntime,
    cfg: AppConfig,
    result_q: queue.Queue,
    stop_event: threading.Event,
) -> None:
    """Track + OptiView publish + optional overlay save."""
    runtime = stream.runtime
    output_dir = Path(cfg.output_dir) if cfg.output_dir else None
    profile_every = cfg.save_every if cfg.save_every > 0 else 10

    try:
        while not stop_event.is_set():
            if cfg.frames > 0 and stream.metrics.processed >= cfg.frames:
                return
            try:
                pkt = result_q.get(timeout=0.1)
            except queue.Empty:
                continue

            loop_start = time.perf_counter()
            if stream.metrics.wall_started_at_s is None:
                stream.metrics.wall_started_at_s = loop_start
            if stream.metrics._interval_wall_started_at_s is None:
                stream.metrics._interval_wall_started_at_s = loop_start

            # Accumulate timing from upstream threads.
            stream.metrics.source_time_s += pkt.source_time_s
            stream.metrics._interval_source_s += pkt.source_time_s
            stream.metrics.preproc_time_s += pkt.preproc_time_s
            stream.metrics.push_time_s += pkt.push_time_s
            stream.metrics.pull_wait_s += pkt.pull_wait_s
            stream.metrics._interval_preproc_s += pkt.preproc_time_s
            stream.metrics._interval_push_s += pkt.push_time_s
            stream.metrics._interval_pull_s += pkt.pull_wait_s
            stream.metrics.pulled += 1

            # Parse bboxes from hardware BoxDecode output.
            boxes = parse_bbox_payload(pkt.bbox_payload, pkt.frame.shape[1], pkt.frame.shape[0])
            boxes = filter_person_detections(boxes, cfg.person_class_id)

            # Track.
            track_t0 = time.perf_counter()
            tracked = stream.tracker.update(boxes, frame_index=pkt.frame_index)
            stream.metrics.track_time_s += time.perf_counter() - track_t0
            stream.metrics.detections += len(tracked)

            # Publish clean video first, then matching OptiView JSON.
            write_t0 = time.perf_counter()
            pyneat = runtime.pyneat
            if not stream.video_run.push(pkt.frame, copy=True, image_format=pyneat.PixelFormat.BGR):
                raise RuntimeError(f"stream {stream.index} OptiView video push failed")

            frame_id = str(stream.metrics.processed)
            payload_json = make_optiview_tracking_json(
                int(time.time() * 1000),
                frame_id,
                tracked,
            )
            if not stream.json_sender.send_json(payload_json):
                raise RuntimeError(f"stream {stream.index} OptiView JSON send failed")

            # Render overlay only when a saved debug frame is actually needed.
            if (
                output_dir is not None
                and cfg.save_every > 0
                and pkt.frame_index % cfg.save_every == 0
            ):
                overlay_t0 = time.perf_counter()
                overlay = draw_tracked_people(runtime, pkt.frame.copy(), tracked)
                stream.metrics.overlay_time_s += time.perf_counter() - overlay_t0
                if save_overlay_frame(
                    runtime,
                    output_dir,
                    stream.index,
                    pkt.frame_index,
                    overlay,
                    cfg.save_every,
                ):
                    stream.metrics.saved += 1
            stream.metrics.write_time_s += time.perf_counter() - write_t0

            completed_at = time.perf_counter()
            output_elapsed = completed_at - loop_start
            stream.metrics._interval_output_s += output_elapsed
            stream.metrics.processed += 1
            per_frame = pkt.source_time_s + pkt.preproc_time_s + pkt.push_time_s + pkt.pull_wait_s + output_elapsed
            stream.metrics.total_loop_time_s += per_frame
            stream.metrics._interval_loop_s += per_frame
            stream.metrics._interval_frames += 1
            stream.metrics.wall_last_processed_at_s = completed_at

            if cfg.profile and stream.metrics._interval_frames >= profile_every:
                print_interval_profile(stream, profile_every)

    except Exception as exc:
        stream.error = exc
        stop_event.set()


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def print_interval_profile(stream: StreamRuntime, profile_every: int) -> None:
    m = stream.metrics
    n = m._interval_frames
    if n <= 0:
        return
    src_ms = m._interval_source_s * 1000.0 / n
    pre_ms = m._interval_preproc_s * 1000.0 / n
    push_ms = m._interval_push_s * 1000.0 / n
    pull_ms = m._interval_pull_s * 1000.0 / n
    out_ms = m._interval_output_s * 1000.0 / n
    loop_ms = m._interval_loop_s * 1000.0 / n
    throughput_fps = wall_clock_fps(
        n,
        m._interval_wall_started_at_s,
        m.wall_last_processed_at_s,
    )
    print(
        f"  [stream {stream.index}] frames {m.processed - n}-{m.processed - 1} | "
        f"src={src_ms:.1f}ms  preproc={pre_ms:.1f}ms  push={push_ms:.1f}ms  "
        f"pull_wait={pull_ms:.1f}ms  output={out_ms:.1f}ms  "
        f"loop={loop_ms:.1f}ms  throughput_fps={throughput_fps:.1f}"
    )
    m._interval_source_s = 0.0
    m._interval_preproc_s = 0.0
    m._interval_push_s = 0.0
    m._interval_pull_s = 0.0
    m._interval_output_s = 0.0
    m._interval_loop_s = 0.0
    m._interval_frames = 0
    m._interval_wall_started_at_s = m.wall_last_processed_at_s


def print_profile_summary(streams: list[StreamRuntime]) -> None:
    print("\nProfile summary (averages per frame):")
    for stream in streams:
        m = stream.metrics
        n = max(m.processed, 1)
        src = m.source_time_s * 1000.0 / n
        pre = m.preproc_time_s * 1000.0 / n
        psh = m.push_time_s * 1000.0 / n
        pll = m.pull_wait_s * 1000.0 / n
        trk = m.track_time_s * 1000.0 / n
        ovl = m.overlay_time_s * 1000.0 / n
        wrt = m.write_time_s * 1000.0 / n
        out = trk + ovl + wrt
        loop = m.total_loop_time_s * 1000.0 / n
        throughput_fps = wall_clock_fps(
            m.processed,
            m.wall_started_at_s,
            m.wall_last_processed_at_s,
        )
        print(
            f"  [stream {stream.index}] {m.processed} frames | "
            f"src={src:.1f}ms  preproc={pre:.1f}ms  push={psh:.1f}ms  "
            f"pull_wait={pll:.1f}ms  output={out:.1f}ms "
            f"(track={trk:.1f} overlay={ovl:.1f} write={wrt:.1f})  "
            f"loop={loop:.1f}ms  throughput_fps={throughput_fps:.1f}"
        )


def wall_clock_fps(frame_count: int, started_at_s: float | None, ended_at_s: float | None) -> float:
    if frame_count <= 0 or started_at_s is None or ended_at_s is None:
        return 0.0
    elapsed_s = ended_at_s - started_at_s
    if elapsed_s <= 0:
        return 0.0
    return frame_count / elapsed_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = parse_config(argv)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Build model once, shared across all streams.
    runtime = load_runtime_modules()
    try:
        model = build_model(runtime, cfg)
        quant_preproc = build_quanttess_cpu_preproc(runtime, model)
    except Exception as exc:
        print(f"Error: failed to build model: {exc}", file=sys.stderr)
        return 3

    # Build sessions sequentially, one stream at a time.
    streams: list[StreamRuntime] = []
    try:
        for index, url in enumerate(cfg.rtsp_urls):
            stream = create_stream_runtime(index, url, cfg, model, quant_preproc)
            streams.append(stream)
    except Exception as exc:
        print(f"Error: failed to set up stream runtimes: {exc}", file=sys.stderr)
        for s in streams:
            close_stream_runtime(s)
        return 4

    for stream in streams:
        print(
            f"[stream {stream.index}] {stream.probe.width}x{stream.probe.height} "
            f"@{effective_writer_fps(cfg, stream.probe)}fps "
            f"{stream.url} -> optiview://{cfg.optiview_host} "
            f"video={optiview_video_port_for_stream(cfg.optiview_video_port_base, stream.index)} "
            f"json={optiview_json_port_for_stream(cfg.optiview_json_port_base, stream.index)}"
        )

    # Launch infer/publish threads first, then bring producer threads up one by one
    # so RTSP startup does not stampede all cameras at once.
    stop_event = threading.Event()
    worker_threads: list[threading.Thread] = []
    producer_threads: list[threading.Thread] = []
    producer_ready_events: list[threading.Event] = []
    for stream in streams:
        frame_q: queue.Queue[FramePacket] = queue.Queue(maxsize=4)
        result_q: queue.Queue[ResultPacket] = queue.Queue(maxsize=4)
        producer_ready = threading.Event()
        producer_ready_events.append(producer_ready)
        producer_threads.append(
            threading.Thread(
                target=producer_thread,
                args=(stream, cfg, frame_q, stop_event, producer_ready),
                name=f"producer-{stream.index}",
                daemon=True,
            )
        )
        worker_threads.append(
            threading.Thread(
                target=infer_thread,
                args=(stream, cfg, frame_q, result_q, stop_event),
                name=f"infer-{stream.index}",
                daemon=True,
            )
        )
        worker_threads.append(
            threading.Thread(
                target=publish_thread,
                args=(stream, cfg, result_q, stop_event),
                name=f"publish-{stream.index}",
                daemon=True,
            )
        )

    try:
        for thread in worker_threads:
            thread.start()
        started_threads = list(worker_threads)
        started_threads.extend(
            start_producer_threads_sequentially(
                producer_threads,
                producer_ready_events,
                stop_event,
            )
        )
        for thread in started_threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for thread in worker_threads + producer_threads:
            thread.join(timeout=5)
    finally:
        for stream in streams:
            close_stream_runtime(stream)

    failed = [stream for stream in streams if stream.error is not None]
    if failed:
        for stream in failed:
            print(f"[stream {stream.index}] error: {stream.error}", file=sys.stderr)
        return 5

    if cfg.profile:
        print_profile_summary(streams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
