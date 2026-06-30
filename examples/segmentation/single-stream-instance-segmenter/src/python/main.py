"""Single-camera RTSP YOLO26 segmentation Insight example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import sys
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_LABELS = DEFAULT_CONFIG.parent / "coco_label.txt"
DEFAULT_FPS = 30
MASK_ALPHA = 0.55
MASK_THRESHOLD = 0.50

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
    model_path: str
    labels_path: Path
    rtsp_url: str
    latency_ms: int = 200
    tcp: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    output: OutputConfig = OutputConfig("", 0, MASK_ALPHA, MASK_THRESHOLD, True)


@dataclass
class PipelineRuntime:
    model: object
    graph: object
    run: object
    video_graph: object
    video_run: object
    metadata_sender: object
    labels: list[str]
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.instances = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.overlay_ms = 0.0
        self.video_push_ms = 0.0
        self.metadata_ms = 0.0

    def add(
        self,
        pull_ms: float,
        decode_ms: float,
        overlay_ms: float,
        video_push_ms: float,
        metadata_ms: float,
        instance_count: int,
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.instances += instance_count
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.overlay_ms += overlay_ms
        self.video_push_ms += video_push_ms
        self.metadata_ms += metadata_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed_ms = max(time_ms() - self.start_ms, 1e-6)
        frames = float(self.frames)
        print(
            f"[profile] frames={self.frames} "
            f"output_fps={self.frames * 1000.0 / elapsed_ms} "
            f"avg_pull_ms={self.pull_ms / frames} "
            f"avg_decode_ms={self.decode_ms / frames} "
            f"avg_overlay_ms={self.overlay_ms / frames} "
            f"avg_video_push_ms={self.video_push_ms / frames} "
            f"avg_metadata_ms={self.metadata_ms / frames} "
            f"avg_instances={self.instances / frames}",
            flush=True,
        )
        self.reset()


def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

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
        description="Single-camera RTSP YOLO26 segmentation Insight example"
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
    if not cfg.rtsp_url:
        raise ValueError("source.rtsp_url must be set")
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
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
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(DEFAULT_LABELS))),
        rtsp_url=string_or(source, "rtsp_url"),
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        output=OutputConfig(
            save_dir=string_or(output, "save_dir"),
            save_every=int_or(output, "save_every", 0),
            mask_alpha=float_or(output, "mask_alpha", MASK_ALPHA),
            mask_threshold=float_or(output, "mask_threshold", MASK_THRESHOLD),
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


def tensor_to_numpy(tensor) -> object:
    return np.asarray(tensor.to_numpy(copy=True))


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

    frame = tensor_to_numpy(tensor)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def tensor_from_rgb_frame(frame):
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def extract_tensors(sample) -> list:
    if isinstance(sample, (list, tuple)) and all(hasattr(item, "to_numpy") for item in sample):
        return list(sample)
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)

    tensors = []
    for field in sample.fields:
        tensors.extend(extract_tensors(field))
    return tensors


def find_field(sample, label: str):
    if sample is None:
        return None
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    field = find_field(sample, label)
    fields = list(getattr(sample, "fields", []))
    if field is not None:
        return field
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output missing {label} field")


def frame_tensor_from_sample(sample):
    tensors = extract_tensors(joined_field(sample, "frame", 0))
    if not tensors:
        raise RuntimeError("joined frame field has no tensor")
    return tensors[0]


def segment_tensors_from_sample(sample) -> list:
    tensors = extract_tensors(joined_field(sample, "segments", 1))
    if not tensors:
        raise RuntimeError("joined segments field has no tensors")
    return tensors


def decode_segmentation_output(tensors: list, frame_w: int, frame_h: int, max_detections: int):
    decoded = pyneat.decode_segmentation(
        tensors,
        clamp_to=(frame_w, frame_h),
        top_k=max_detections,
        strict=False,
    )
    detections = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32)
        masks = tensor_to_numpy(item.masks).astype(np.uint8)
        for row, mask in zip(boxes.reshape((-1, 6)), masks.reshape((-1, 160, 160))):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class_id": int(class_id),
                    "mask": mask,
                }
            )
            if len(detections) >= max_detections:
                return detections
    return detections


def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return width, height, fps if fps > 0 else DEFAULT_FPS


def make_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.fallback_h264_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def make_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26Seg
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_metadata_sender(cfg: AppConfig):
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.channel = 0
    options.metadata_port_base = cfg.metadata_port
    return pyneat.MetadataSender(options)


def build_video_graph(cfg: AppConfig, width: int, height: int, fps: int):
    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = pyneat.Format.RGB
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight_host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.video_port
    sender_opt.encoder.bitrate_kbps = 1000

    graph = pyneat.Graph("video")
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(pyneat.groups.video_sender(sender_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    return graph, graph.build([seed]), sender_opt.video_port


def build_pipeline(cfg: AppConfig) -> PipelineRuntime:
    frame_w, frame_h, fps = probe_rtsp(cfg.rtsp_url)
    model = make_model(cfg)
    labels = load_labels(cfg.labels_path)

    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, fps, frame_w, frame_h))
    branch = pyneat.graphs.branch("source", ["frame", "model"])

    frame_graph = pyneat.Graph("frame")
    frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    segments_graph = pyneat.Graph("segments")
    segments_graph.add(pyneat.nodes.output("segments", pyneat.OutputOptions.every_frame(4)))

    joined = pyneat.graphs.combine(
        ["frame", "segments"], "segmentation_output", pyneat.CombinePolicy.ByFrame
    )

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, frame_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, segments_graph)
    graph.connect(frame_graph, joined)
    graph.connect(segments_graph, joined)
    if cfg.profile:
        print(f"Backend:\n{graph.describe_backend()}")

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)

    video_graph, video_run, video_port = build_video_graph(cfg, frame_w, frame_h, fps)
    metadata_sender = build_metadata_sender(cfg)
    print(
        f"rtsp={cfg.rtsp_url} stream={frame_w}x{frame_h}@{fps} "
        f"insight={cfg.insight_host} video={video_port} "
        f"metadata={metadata_sender.metadata_port()} channel=0"
    )
    return PipelineRuntime(
        model=model,
        graph=graph,
        run=run,
        video_graph=video_graph,
        video_run=video_run,
        metadata_sender=metadata_sender,
        labels=labels,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=fps,
        video_port=video_port,
    )


def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (255, 194, 0),
        (168, 153, 44),
    ]
    return palette[max(0, class_id) % len(palette)]


def frame_rect_for_detection(det: dict, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x0 = max(0, min(frame_w - 1, int(np.floor(float(det["x1"])))))
    y0 = max(0, min(frame_h - 1, int(np.floor(float(det["y1"])))))
    x1 = max(x0 + 1, min(frame_w, int(np.ceil(float(det["x2"])))))
    y1 = max(y0 + 1, min(frame_h, int(np.ceil(float(det["y2"])))))
    return x0, y0, x1, y1


def mask_rect_for_frame_rect(
    frame_rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    mask_h, mask_w = mask_shape[:2]
    model_w = mask_w * 4
    model_h = mask_h * 4
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) * 0.5
    pad_y = (model_h - frame_h * scale) * 0.5

    def to_mask_x(frame_x: float) -> float:
        return (frame_x * scale + pad_x) * mask_w / model_w

    def to_mask_y(frame_y: float) -> float:
        return (frame_y * scale + pad_y) * mask_h / model_h

    fx0, fy0, fx1, fy1 = frame_rect
    x0 = max(0, min(mask_w - 1, int(np.floor(to_mask_x(fx0)))))
    y0 = max(0, min(mask_h - 1, int(np.floor(to_mask_y(fy0)))))
    x1 = max(x0 + 1, min(mask_w, int(np.ceil(to_mask_x(fx1)))))
    y1 = max(y0 + 1, min(mask_h, int(np.ceil(to_mask_y(fy1)))))
    return x0, y0, x1, y1


def project_letterbox_mask_roi(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
):
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape, mask.shape)
    return cv2.resize(mask[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def draw_box(frame, det: dict, labels: list[str]) -> None:
    cls_id = int(det["class_id"])
    color = class_color(cls_id)
    x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        frame,
        f"{class_name(labels, cls_id)} {float(det['score']):.2f}",
        (x0, max(0, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def overlay_segmentation(
    frame,
    detections: list[dict],
    min_score: float,
    cfg: OutputConfig,
    labels: list[str],
):
    annotated = frame.copy()
    for det in detections:
        if float(det["score"]) < min_score:
            continue
        mask = det.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        threshold = cfg.mask_threshold * (255.0 if int(mask.max()) > 1 else 1.0)
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
        roi_mask = project_letterbox_mask_roi(mask, frame.shape, (x0, y0, x1, y1))
        _, binary_mask = cv2.threshold(roi_mask, threshold, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary_mask) > 0:
            color = class_color(int(det["class_id"]))
            roi = annotated[y0:y1, x0:x1]
            mask_color = np.full(roi.shape, color, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0)
            cv2.copyTo(blended, binary_mask, roi)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, color, 2, cv2.LINE_8)
        if cfg.draw_boxes:
            draw_box(annotated, det, labels)
    return annotated


def metadata_boxes(detections: list[dict], labels: list[str], frame_w: int, frame_h: int):
    objects = []
    for index, det in enumerate(detections, start=1):
        x0, y0, x1, y1 = frame_rect_for_detection(det, (frame_h, frame_w, 3))
        objects.append(
            {
                "id": f"obj_{index}",
                "label": class_name(labels, int(det["class_id"])),
                "confidence": float(det["score"]),
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            }
        )
    return objects


def send_metadata(runtime: PipelineRuntime, sample, detections: list[dict]) -> None:
    frame_id = getattr(sample, "frame_id", -1)
    if frame_id is None or frame_id < 0:
        frame_id = 0
    runtime.metadata_sender.send_metadata(
        "instance-segmentation",
        json.dumps(
            {"objects": metadata_boxes(detections, runtime.labels, runtime.frame_w, runtime.frame_h)},
            separators=(",", ":"),
        ),
        int(time.time() * 1000),
        str(frame_id),
    )


def push_annotated_video(runtime: PipelineRuntime, annotated_bgr) -> None:
    rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    if not runtime.video_run.push([tensor_from_rgb_frame(rgb)]):
        raise RuntimeError("Insight video push failed")


def maybe_save_frame(cfg: AppConfig, processed: int, annotated_bgr) -> None:
    if not cfg.output.save_dir or cfg.output.save_every <= 0:
        return
    if processed % cfg.output.save_every != 0:
        return
    out_path = Path(cfg.output.save_dir) / f"frame_{processed}.jpg"
    if not cv2.imwrite(str(out_path), annotated_bgr):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    while cfg.frames <= 0 or processed < cfg.frames:
        pull_start = time_ms()
        sample = runtime.run.pull("segmentation_output", 20000)
        pull_end = time_ms()
        if sample is None:
            print("[warn] timed out waiting for segmentation output", file=sys.stderr)
            continue

        decode_start = time_ms()
        frame = tensor_bgr_from_decoded(frame_tensor_from_sample(sample))
        detections = decode_segmentation_output(
            segment_tensors_from_sample(sample),
            runtime.frame_w,
            runtime.frame_h,
            cfg.max_detections,
        )
        decode_end = time_ms()

        overlay_start = time_ms()
        annotated = overlay_segmentation(frame, detections, cfg.min_score, cfg.output, runtime.labels)
        overlay_end = time_ms()

        video_start = time_ms()
        push_annotated_video(runtime, annotated)
        video_end = time_ms()

        metadata_start = time_ms()
        send_metadata(runtime, sample, detections)
        metadata_end = time_ms()

        processed += 1
        maybe_save_frame(cfg, processed, annotated)
        profile.add(
            pull_end - pull_start,
            decode_end - decode_start,
            overlay_end - overlay_start,
            video_end - video_start,
            metadata_end - metadata_start,
            len(detections),
        )

    profile.flush()
    print(f"processed={processed} video_sender={cfg.insight_host}:{runtime.video_port}")
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config}")
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
            os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
        if cfg.output.save_dir:
            Path(cfg.output.save_dir).mkdir(parents=True, exist_ok=True)

        runtime = build_pipeline(cfg)
        try:
            return 0 if run_pipeline(runtime, cfg) > 0 else 3
        finally:
            runtime.video_run.close()
            runtime.run.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
