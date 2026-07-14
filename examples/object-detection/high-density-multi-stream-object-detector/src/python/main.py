"""Graph-native multi-stream RTSP YOLO26 Insight example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
from pathlib import Path
import signal
import struct
import sys
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
MAX_STREAMS = 80
DEFAULT_INITIAL_DETECTION_TIMEOUT_MS = 30_000
DETECTOR_RESULT_TIMEOUT_MS = 5_000
DEFAULT_QUEUE_DEPTH = 4
DEFAULT_INTERNAL_QUEUE_DEPTH = 1
DEFAULT_MAX_INFLIGHT_PER_STREAM = 4
# Complete MLA work asynchronously so a busy model stage cannot hold the
# fused fan-in callback and starve a subset of decoder streams.
INFERENCE_ASYNC = True
DEFAULT_FAN_IN_POLICY = "latest"
DEFAULT_DECODER_BUFFERS = 16
DEFAULT_DECODER_INPUT_BUFFERS = 2
ALL_INSIGHT_STREAMS = -1
MAX_ENCODED_AU_BYTES = 16 * 1024 * 1024

cv2 = None
pyneat = None
_STOP_REQUESTED = False


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_urls: list[str]
    decode_type: str = "yolo26"
    workers: int = 1
    queue_depth: int = DEFAULT_QUEUE_DEPTH
    internal_queue_depth: int = DEFAULT_INTERNAL_QUEUE_DEPTH
    max_inflight_per_stream: int = DEFAULT_MAX_INFLIGHT_PER_STREAM
    fan_in_policy: str = DEFAULT_FAN_IN_POLICY
    decoder_buffers: int = DEFAULT_DECODER_BUFFERS
    decoder_input_buffers: int = DEFAULT_DECODER_INPUT_BUFFERS
    decoder_tuning: str = "auto"
    input_width: int = 0
    input_height: int = 0
    input_fps: int = 0
    latency_ms: int = 100
    tcp: bool = True
    skip_rtsp_probe: bool = False
    rtsp_drop_on_latency: bool = False
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    initial_detection_timeout_ms: int = DEFAULT_INITIAL_DETECTION_TIMEOUT_MS
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    insight_visible_streams: int = ALL_INSIGHT_STREAMS
    video_enabled: bool = True


BOX_DECODE_TYPE_NAMES = {
    "yolo26": "YoloV26",
    "yolov26": "YoloV26",
    "yolov8": "YoloV8",
}


@dataclass
class StreamProfile:
    enabled: bool
    stream_index: int
    interval: int = 100
    frames: int = 0
    boxes: int = 0
    start_ms: float = 0.0
    parse_ms: float = 0.0
    metadata_send_ms: float = 0.0

    def add(self, parse_ms: float, metadata_send_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.parse_ms += parse_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        metadata_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile stream={self.stream_index}] frames={self.frames} "
            f"metadata_fps={metadata_fps} "
            f"avg_parse_ms={self.parse_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_boxes={self.boxes / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.parse_ms = 0.0
        self.metadata_send_ms = 0.0


@dataclass
class AggregateProfile:
    enabled: bool = False
    stream_count: int = 0
    interval_frames: int = 400
    frames: int = 0
    start_ms: float = 0.0

    def add(self) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        if self.frames >= self.interval_frames:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed_s = max(0.001, (time_ms() - self.start_ms) / 1000.0)
        aggregate_fps = self.frames / elapsed_s
        per_stream = aggregate_fps / self.stream_count if self.stream_count > 0 else 0.0
        print(
            f"[profile aggregate] frames={self.frames} elapsed_s={elapsed_s} "
            f"aggregate_fps={aggregate_fps} avg_per_stream_fps={per_stream}",
            flush=True,
        )
        self.frames = 0
        self.start_ms = 0.0


@dataclass
class SourceRuntime:
    index: int
    url: str
    metadata_sender: object | None
    labels: list[str]
    source_options: object
    profile: StreamProfile
    frame_w: int
    frame_h: int
    source_fps: int
    video_port: int = 0
    processed: int = 0


@dataclass
class AppRuntime:
    model: object
    graph: object
    run: object
    sources: list[SourceRuntime]


def load_runtime_dependencies() -> None:
    global pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import pyneat as pyneat_module

    pyneat = pyneat_module


def load_cv_dependency() -> None:
    global cv2
    if cv2 is not None:
        return
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cv2 as cv2_module

    cv2 = cv2_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-native multi-stream RTSP YOLO Insight example")
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
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def parse_streams(raw: dict) -> list[str]:
    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")

    rtsp_urls = []
    for index, value in enumerate(streams):
        if not isinstance(value, str):
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_url = value.strip()
        if not rtsp_url:
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(rtsp_url)
    return rtsp_urls


def normalize_box_decode_type(token: str) -> str:
    decode_type = token.strip().lower()
    if decode_type not in BOX_DECODE_TYPE_NAMES:
        raise ValueError("model.decode_type must be one of: yolo26, yolov8")
    return decode_type


def normalize_decoder_tuning(token: str) -> str:
    decoder_tuning = token.strip().lower().replace("_", "-")
    if not decoder_tuning or decoder_tuning == "auto":
        return "auto"
    if decoder_tuning == "default":
        return "default"
    if decoder_tuning in ("low-memory", "memory"):
        return "low-memory"
    if decoder_tuning in (
        "throughput-low-latency",
        "throughput",
        "low-latency-throughput",
    ):
        return "throughput-low-latency"
    raise ValueError(
        "input.decoder_tuning must be one of: auto, default, low-memory, throughput-low-latency"
    )


def normalize_fan_in_policy(token: str) -> str:
    if token in ("latest", "every_frame"):
        return token
    raise ValueError("inference.fan_in_policy must be one of: latest, every_frame")


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    normalize_box_decode_type(cfg.decode_type)
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.rtsp_urls:
        raise ValueError("streams must be set")
    if len(cfg.rtsp_urls) > MAX_STREAMS:
        raise ValueError("this example supports up to 80 streams")
    if cfg.workers != 1:
        raise ValueError(
            "graph-native multi-stream detector uses one shared worker; set inference.workers to 1"
        )
    if cfg.queue_depth <= 0:
        raise ValueError("inference.queue_depth must be > 0")
    if cfg.queue_depth > 32:
        raise ValueError("inference.queue_depth must be <= 32")
    if cfg.internal_queue_depth < 0:
        raise ValueError("inference.internal_queue_depth must be >= 0")
    if cfg.internal_queue_depth > 32:
        raise ValueError("inference.internal_queue_depth must be <= 32")
    if cfg.max_inflight_per_stream <= 0:
        raise ValueError("inference.max_inflight_per_stream must be > 0")
    if cfg.max_inflight_per_stream > 32:
        raise ValueError("inference.max_inflight_per_stream must be <= 32")
    normalize_fan_in_policy(cfg.fan_in_policy)
    if cfg.decoder_buffers <= 0:
        raise ValueError("input.decoder_buffers must be > 0")
    if cfg.decoder_buffers > 64:
        raise ValueError("input.decoder_buffers must be <= 64")
    if cfg.decoder_input_buffers <= 0:
        raise ValueError("input.decoder_input_buffers must be > 0")
    normalize_decoder_tuning(cfg.decoder_tuning)
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.input_width < 0:
        raise ValueError("input.width must be >= 0")
    if cfg.input_height < 0:
        raise ValueError("input.height must be >= 0")
    if cfg.input_fps < 0:
        raise ValueError("input.fps must be >= 0")
    if (cfg.input_width == 0) != (cfg.input_height == 0):
        raise ValueError("input.width and input.height must be set together")
    if cfg.skip_rtsp_probe and not (
        cfg.input_width > 0 and cfg.input_height > 0 and cfg.input_fps > 0
    ):
        raise ValueError("input.skip_rtsp_probe requires input.width, input.height, and input.fps")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.initial_detection_timeout_ms <= 0:
        raise ValueError("runtime.initial_detection_timeout_ms must be > 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.video_port_base > 65535:
        raise ValueError("output.insight.video_port_base must be <= 65535")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.metadata_port_base > 65535:
        raise ValueError("output.insight.metadata_port_base must be <= 65535")
    if cfg.insight_visible_streams < ALL_INSIGHT_STREAMS:
        raise ValueError("output.insight.max_visible_streams must be >= -1 (-1 means all)")
    if cfg.insight_visible_streams >= 0 and cfg.insight_visible_streams > len(cfg.rtsp_urls):
        raise ValueError("output.insight.max_visible_streams cannot exceed stream count")
    visible_streams = (
        len(cfg.rtsp_urls)
        if cfg.insight_visible_streams < 0
        else cfg.insight_visible_streams
    )
    if visible_streams > 0:
        video_last = cfg.video_port_base + visible_streams - 1
        metadata_last = cfg.metadata_port_base + visible_streams - 1
        if video_last > 65535:
            raise ValueError("output.insight video port range exceeds 65535")
        if metadata_last > 65535:
            raise ValueError("output.insight metadata port range exceeds 65535")
        if not (
            video_last < cfg.metadata_port_base
            or metadata_last < cfg.video_port_base
        ):
            raise ValueError("output.insight video and metadata port ranges overlap")


def resolve_config_relative_path(config_path: Path, value: str) -> Path:
    if not value:
        return Path()
    path = Path(value).expanduser()
    return path if path.is_absolute() else config_path.resolve().parent / path


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    inference = section(raw, "inference")
    if "fps" in inference:
        raise ValueError("inference.fps is not supported; set stream FPS at the RTSP source")
    if "target_fps" in inference:
        raise ValueError(
            "inference.target_fps is not supported; set stream FPS at the RTSP source"
        )

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    if "hidden_streams" in output:
        raise ValueError(
            "output.hidden_streams was removed; set output.insight.max_visible_streams explicitly"
        )
    if "debug_dir" in output or "save_every" in output:
        raise ValueError(
            "output.debug_dir/output.save_every were removed from the realtime application"
        )
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"
    rtsp_urls = parse_streams(raw)

    queue_depth = int_or(inference, "queue_depth", DEFAULT_QUEUE_DEPTH)
    if "queue_depth" not in inference:
        queue_depth = int_or(
            inference,
            "inflight",
            int_or(inference, "detector_queue_depth", DEFAULT_QUEUE_DEPTH),
        )

    model_path_value = string_or(model, "path")
    labels_path_value = string_or(model, "labels", str(default_labels))

    cfg = AppConfig(
        model_path=(
            str(resolve_config_relative_path(config_path, model_path_value))
            if model_path_value
            else ""
        ),
        decode_type=normalize_box_decode_type(string_or(model, "decode_type", "yolo26")),
        labels_path=resolve_config_relative_path(config_path, labels_path_value),
        rtsp_urls=rtsp_urls,
        workers=int_or(inference, "workers", 1),
        queue_depth=queue_depth,
        internal_queue_depth=int_or(
            inference, "internal_queue_depth", DEFAULT_INTERNAL_QUEUE_DEPTH
        ),
        max_inflight_per_stream=int_or(
            inference, "max_inflight_per_stream", DEFAULT_MAX_INFLIGHT_PER_STREAM
        ),
        fan_in_policy=normalize_fan_in_policy(
            string_or(inference, "fan_in_policy", DEFAULT_FAN_IN_POLICY)
        ),
        decoder_buffers=int_or(input_cfg, "decoder_buffers", DEFAULT_DECODER_BUFFERS),
        decoder_input_buffers=int_or(
            input_cfg, "decoder_input_buffers", DEFAULT_DECODER_INPUT_BUFFERS
        ),
        decoder_tuning=normalize_decoder_tuning(string_or(input_cfg, "decoder_tuning", "auto")),
        input_width=int_or(input_cfg, "width", 0),
        input_height=int_or(input_cfg, "height", 0),
        input_fps=int_or(input_cfg, "fps", 0),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        skip_rtsp_probe=bool_or(input_cfg, "skip_rtsp_probe", False),
        rtsp_drop_on_latency=bool_or(input_cfg, "drop_on_latency", False),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        initial_detection_timeout_ms=int_or(
            runtime, "initial_detection_timeout_ms", DEFAULT_INITIAL_DETECTION_TIMEOUT_MS
        ),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        insight_visible_streams=int_or(insight, "max_visible_streams", ALL_INSIGHT_STREAMS),
        video_enabled=bool_or(output, "video_enabled", True),
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


def extract_tensor_bbox_payload(sample, tensor=None) -> bytes:
    tensor = tensor if tensor is not None else getattr(sample, "tensor", None)
    if tensor is None:
        raise RuntimeError("capture_missing_tensor")
    fmt = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    semantic = getattr(tensor, "semantic", None)
    tess = getattr(semantic, "tess", None)
    if not fmt and tess is not None:
        fmt = getattr(tess, "format", "")
    fmt = str(fmt).upper()
    if fmt and fmt != "BBOX":
        raise RuntimeError(f"capture_expected_bbox format={fmt}")
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("capture_empty_payload")
    return payload


def extract_bbox_payload(sample) -> bytes:
    if sample.kind == pyneat.SampleKind.Bundle:
        for field in sample.fields:
            try:
                return extract_bbox_payload(field)
            except RuntimeError:
                continue
        raise RuntimeError("bundle missing BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError("capture_expected_tensor")
    return extract_tensor_bbox_payload(sample)


def parse_boxes_strict(payload: bytes, img_w: int, img_h: int, expected_topk: int) -> list[dict]:
    if len(payload) < 4:
        raise RuntimeError("bbox buffer too small")

    header = struct.unpack_from("<I", payload, 0)[0]
    max_boxes = (len(payload) - 4) // 24
    if header > max_boxes:
        raise RuntimeError("bbox header exceeds payload count")
    if expected_topk > 0 and header > expected_topk:
        raise RuntimeError("bbox header exceeds expected topk")

    boxes = []
    offset = 4
    for _ in range(header):
        x, y, w, h, score, class_id = struct.unpack_from("<iiiifi", payload, offset)
        offset += 24
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(img_w))),
                "y1": max(0.0, min(float(y), float(img_h))),
                "x2": max(0.0, min(float(x + w), float(img_w))),
                "y2": max(0.0, min(float(y + h), float(img_h))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes


def build_metadata_boxes(
    boxes: list[dict], labels: list[str], frame_w: int, frame_h: int
) -> list[dict]:
    metadata_boxes = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        class_id = int(box["class_id"])
        metadata_boxes.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_boxes


def send_metadata(source: SourceRuntime, frame, boxes: list[dict]) -> None:
    if source.metadata_sender is None:
        return
    metadata_boxes = build_metadata_boxes(boxes, source.labels, source.frame_w, source.frame_h)
    timestamp_ms = int(frame.pts_ns // 1_000_000) if frame.pts_ns >= 0 else -1
    frame_id = str(frame.frame_id) if frame.frame_id >= 0 else ""
    source.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": metadata_boxes}, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    )


def env_bool(key: str, fallback: bool = False) -> bool:
    value = os.environ.get(key)
    if not value:
        return fallback
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_int(key: str, fallback: int) -> int:
    value = os.environ.get(key)
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def app_verbose() -> bool:
    return env_bool("APP16_VERBOSE", False)


def app_print_backend() -> bool:
    return app_verbose() or env_bool("APP16_PRINT_BACKEND", False)


def stream_id_for(stream_index: int) -> str:
    return f"stream{stream_index}"


def effective_insight_visible_streams(cfg: AppConfig) -> int:
    if cfg.insight_visible_streams < 0:
        return len(cfg.rtsp_urls)
    return cfg.insight_visible_streams


def is_insight_visible_stream(cfg: AppConfig, stream_index: int) -> bool:
    return 0 <= stream_index < effective_insight_visible_streams(cfg)


def should_send_metadata(cfg: AppConfig, stream_index: int) -> bool:
    return is_insight_visible_stream(cfg, stream_index)


def stream_index_from_detection(sample, stream_count: int) -> int:
    stream_id = str(getattr(sample, "stream_id", ""))
    prefix = "stream"
    if not stream_id.startswith(prefix):
        if stream_count == 1:
            return 0
        raise RuntimeError(f"detection sample missing stream id: {stream_id}")
    suffix = stream_id[len(prefix) :]
    if not suffix or not suffix.isdigit():
        raise RuntimeError(f"invalid detection stream id: {stream_id}")
    index = int(suffix)
    if index < 0 or index >= stream_count:
        raise RuntimeError(f"detection stream id out of range: {stream_id}")
    return index


def target_frames_per_stream() -> int:
    return max(0, env_int("APP16_FRAMES_PER_STREAM", 0))


def target_reached(sources: list[SourceRuntime]) -> bool:
    target = target_frames_per_stream()
    if target <= 0 or not sources:
        return False
    return all(source.processed >= target for source in sources)


def realtime_options(queue_depth: int = 3, overflow_policy=None):
    overflow_policy = overflow_policy or pyneat.OverflowPolicy.KeepLatest
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = queue_depth
    run_options.overflow_policy = overflow_policy
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def graph_options(internal_queue_depth: int):
    if not hasattr(pyneat, "GraphOptions"):
        raise RuntimeError(
            "installed pyneat does not expose GraphOptions; "
            "install the matching Neat package before running this graph-native app"
        )
    options = pyneat.GraphOptions()
    advanced = getattr(options, "advanced_execution", None)
    if advanced is None or not all(
        hasattr(advanced, field) for field in ("internal_queue_depth", "inference_async")
    ):
        raise RuntimeError(
            "installed pyneat does not expose "
            "GraphOptions.advanced_execution.internal_queue_depth and "
            "GraphOptions.advanced_execution.inference_async; "
            "install the matching Neat package before running this graph-native app"
        )
    advanced.internal_queue_depth = internal_queue_depth
    advanced.inference_async = INFERENCE_ASYNC
    return options


def graph_realtime_link(
    queue_depth: int,
    stream_id: str,
    max_inflight_per_stream: int = DEFAULT_MAX_INFLIGHT_PER_STREAM,
    fan_in_policy: str = DEFAULT_FAN_IN_POLICY,
):
    if not hasattr(pyneat, "GraphLinkOptions") or not hasattr(pyneat, "GraphLinkPolicy"):
        raise RuntimeError(
            "installed pyneat does not expose GraphLinkOptions/GraphLinkPolicy; "
            "install the matching Neat package before running this graph-native app"
        )
    link = pyneat.GraphLinkOptions()
    fan_in_policy = normalize_fan_in_policy(fan_in_policy)
    if fan_in_policy == "every_frame":
        if not hasattr(pyneat.GraphLinkPolicy, "RealtimeEveryFrameByStream"):
            raise RuntimeError(
                "installed pyneat does not expose RealtimeEveryFrameByStream; "
                "install the matching Neat package before using every_frame fan-in"
            )
        link.policy = pyneat.GraphLinkPolicy.RealtimeEveryFrameByStream
    else:
        link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.stream_id = stream_id
    link.max_inflight_per_stream = max_inflight_per_stream
    return link


def apply_h264_caps(opt, width: int, height: int, fps: int) -> tuple[int, int, int]:
    width_out = 0
    height_out = 0
    fps_out = 0
    if width > 0 and height > 0:
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
        opt.output_caps.width = width
        opt.output_caps.height = height
        width_out = width
        height_out = height
    if fps > 0:
        opt.fallback_h264_fps = fps
        opt.output_caps.fps = fps
        fps_out = fps
    return width_out, height_out, fps_out


def probe_rtsp(url: str) -> tuple[int, int, int]:
    load_cv_dependency()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")
    return width, height, fps


def make_source_options(
    cfg: AppConfig,
    url: str,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
):
    width_out = 0
    height_out = 0
    fps_out = 0

    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.drop_on_latency = cfg.rtsp_drop_on_latency
    opt.payload_type = 96
    opt.insert_queue = True
    opt.out_format = pyneat.Format.NV12
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.decoder_next_element = "CVU"
    opt.auto_caps_from_stream = not cfg.skip_rtsp_probe
    opt.num_buffers = cfg.decoder_buffers

    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.memory = pyneat.CapsMemory.Any

    if not cfg.skip_rtsp_probe:
        if width is None or height is None or fps is None:
            width, height, fps = probe_rtsp(url)
        w, h, f = apply_h264_caps(opt, int(width or 0), int(height or 0), int(fps or 0))
        width_out = w or width_out
        height_out = h or height_out
        fps_out = f or fps_out

    w, h, f = apply_h264_caps(opt, cfg.input_width, cfg.input_height, cfg.input_fps)
    width_out = w or width_out
    height_out = h or height_out
    fps_out = f or fps_out
    return opt, fps_out, width_out, height_out



def use_system_memory_for_encoded_input(options) -> None:
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(options, "memory_policy"):
        options.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    elif hasattr(options, "use_simaai_pool"):
        options.use_simaai_pool = False


def make_encoded_h264_input_options(block: bool):
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Encoded
    input_options.format = pyneat.Format.H264
    input_options.block = block
    input_options.max_bytes = MAX_ENCODED_AU_BYTES
    use_system_memory_for_encoded_input(input_options)
    return input_options


def make_rtsp_h264_input(opt):
    encoded = pyneat.RtspEncodedInputOptions()
    encoded.url = opt.url
    encoded.codec = pyneat.RtspCodec.H264
    encoded.latency_ms = opt.latency_ms
    encoded.tcp = opt.tcp
    encoded.drop_on_latency = opt.drop_on_latency
    encoded.buffer_mode = opt.buffer_mode
    encoded.insert_queue = opt.insert_queue
    encoded.sync_mode = opt.sync_mode
    encoded.h264_payload_type = opt.payload_type
    encoded.h264_parse_config_interval = opt.h264_parse_config_interval
    encoded.h264_fps = opt.h264_fps
    encoded.h264_width = opt.h264_width
    encoded.h264_height = opt.h264_height
    encoded.auto_caps_from_stream = opt.auto_caps_from_stream
    encoded.fallback_h264_fps = opt.fallback_h264_fps
    encoded.fallback_h264_width = opt.fallback_h264_width
    encoded.fallback_h264_height = opt.fallback_h264_height
    return pyneat.groups.rtsp_encoded_input(encoded)


def make_rtsp_decoded_input(
    opt,
    decoder_buffers: int,
    input_name: str = "",
    decoder_input_buffers: int = DEFAULT_DECODER_INPUT_BUFFERS,
    decoder_tuning: str = "auto",
):
    dec_w = opt.h264_width if opt.h264_width > 0 else opt.fallback_h264_width
    dec_h = opt.h264_height if opt.h264_height > 0 else opt.fallback_h264_height
    dec_fps = opt.h264_fps if opt.h264_fps > 0 else opt.fallback_h264_fps

    if input_name:
        graph = pyneat.Graph()
        graph.add(pyneat.nodes.input(input_name, make_encoded_h264_input_options(True)))
    else:
        graph = make_rtsp_h264_input(opt)

    decode = pyneat.SimaDecodeOptions()
    decode.type = pyneat.SimaDecodeType.H264
    decode.sima_allocator_type = opt.sima_allocator_type
    decode.out_format = pyneat.Format.NV12
    decode.decoder_name = opt.decoder_name
    decode.raw_output = opt.decoder_raw_output
    decode.next_element = opt.decoder_next_element
    decode.dec_width = dec_w
    decode.dec_height = dec_h
    decode.dec_fps = dec_fps
    decode.num_buffers = decoder_buffers
    decode.input_buffers = decoder_input_buffers
    decode.decoder_tuning = decoder_tuning
    graph.add(pyneat.nodes.sima_decode(decode))

    if opt.output_caps.enable:
        graph.add(
            pyneat.nodes.caps_raw(
                "NV12",
                opt.output_caps.width,
                opt.output_caps.height,
                opt.output_caps.fps,
                opt.output_caps.memory,
            )
        )
    return graph


def make_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    if hasattr(opt, "verbose") and not app_verbose():
        # C++ sets quiet verbose options. pyneat versions without VerboseOptions simply ignore this.
        verbose_options = getattr(pyneat, "VerboseOptions", None)
        if verbose_options is not None and hasattr(verbose_options, "quiet"):
            opt.verbose = verbose_options.quiet()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = getattr(pyneat.BoxDecodeType, BOX_DECODE_TYPE_NAMES[cfg.decode_type])
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def make_video_options(cfg: AppConfig, source: SourceRuntime):
    video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
    video_options.host = cfg.insight_host
    video_options.channel = source.index
    video_options.video_port_base = cfg.video_port_base
    video_options.async_ = True
    return video_options


def make_source_runtime(cfg: AppConfig, stream_index: int, labels: list[str]) -> SourceRuntime:
    url = cfg.rtsp_urls[stream_index]
    source_options, fps, frame_w, frame_h = make_source_options(cfg, url)
    if frame_w <= 0 or frame_h <= 0:
        raise RuntimeError("failed to probe RTSP frame dimensions")
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")

    metadata_sender = None
    if should_send_metadata(cfg, stream_index):
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = stream_index
        metadata_options.metadata_port_base = cfg.metadata_port_base
        metadata_sender = pyneat.MetadataSender(metadata_options)

    return SourceRuntime(
        index=stream_index,
        url=url,
        metadata_sender=metadata_sender,
        labels=labels,
        source_options=source_options,
        profile=StreamProfile(cfg.profile, stream_index),
        frame_w=frame_w,
        frame_h=frame_h,
        source_fps=fps,
    )


def build_detector_graph(cfg: AppConfig):
    model = make_model(cfg)

    input_graph = pyneat.Graph()
    detector_input_options = model.input_appsrc_options(False)
    detector_input_buffers = max(1, int(getattr(detector_input_options, "pool_max_buffers", 1)))
    if cfg.input_width > 0 and cfg.input_height > 0:
        detector_input_options.max_bytes = (
            cfg.input_width * cfg.input_height * 3 // 2 * detector_input_buffers
        )
        detector_input_options.pool_max_buffers = detector_input_buffers
    detector_input_options.block = True
    input_graph.add(pyneat.nodes.input("detector_frame", detector_input_options))

    model_graph = model.graph()

    detections_graph = pyneat.Graph()
    detections_graph.add(
        pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(cfg.queue_depth))
    )

    graph = pyneat.Graph()
    graph.connect(input_graph, model_graph)
    graph.connect(model_graph, detections_graph)
    return model, graph


def video_route_label(cfg: AppConfig, source: SourceRuntime) -> str:
    if not cfg.video_enabled:
        return "disabled"
    if is_insight_visible_stream(cfg, source.index):
        return str(source.video_port)
    return "not-published"


def metadata_route_label(source: SourceRuntime) -> str:
    if source.metadata_sender is None:
        return "disabled"
    return str(source.metadata_sender.metadata_port())


def connect_source_graph(
    app: AppRuntime,
    cfg: AppConfig,
    source: SourceRuntime,
    detector_graph,
) -> None:
    branch = pyneat.graphs.branch("source", ["detector_frame"])
    detector_link = graph_realtime_link(
        cfg.queue_depth,
        stream_id_for(source.index),
        cfg.max_inflight_per_stream,
        cfg.fan_in_policy,
    )

    if cfg.video_enabled and is_insight_visible_stream(cfg, source.index):
        h264 = make_rtsp_h264_input(source.source_options)
        h264_branch = pyneat.graphs.branch("h264_source", ["detector_h264", "video_h264"])
        decoder = make_rtsp_decoded_input(
            source.source_options,
            cfg.decoder_buffers,
            input_name="detector_h264",
            decoder_input_buffers=cfg.decoder_input_buffers,
            decoder_tuning=cfg.decoder_tuning,
        )

        app.graph.connect(h264, h264_branch)
        app.graph.connect(h264_branch, decoder, graph_realtime_link(3, stream_id_for(source.index)))
        app.graph.connect(decoder, branch, graph_realtime_link(1, stream_id_for(source.index)))

        video_options = make_video_options(cfg, source)
        source.video_port = video_options.video_port
        video_graph = pyneat.Graph("video_h264")
        video_graph.connect(
            pyneat.nodes.input("video_h264", make_encoded_h264_input_options(False)),
            pyneat.groups.video_sender(video_options),
        )
        app.graph.connect(h264_branch, video_graph, graph_realtime_link(3, stream_id_for(source.index)))
    else:
        rtsp = make_rtsp_decoded_input(
            source.source_options,
            cfg.decoder_buffers,
            decoder_input_buffers=cfg.decoder_input_buffers,
            decoder_tuning=cfg.decoder_tuning,
        )
        app.graph.connect(rtsp, branch)

    app.graph.connect(branch, detector_graph, detector_link)

    print(
        f"[stream {source.index}] rtsp={source.url} "
        f"stream={source.frame_w}x{source.frame_h}@{source.source_fps} "
        f"insight={cfg.insight_host} "
        f"video={video_route_label(cfg, source)} "
        f"metadata={metadata_route_label(source)}"
    )


def validate_worker_pool_geometry(sources: list[SourceRuntime]) -> None:
    if not sources:
        return
    frame_w = sources[0].frame_w
    frame_h = sources[0].frame_h
    for source in sources:
        if source.frame_w != frame_w or source.frame_h != frame_h:
            raise RuntimeError("detector graph requires all streams to share frame size")


def complete_detection(
    source: SourceRuntime,
    cfg: AppConfig,
    aggregate_profile: AggregateProfile,
    detections,
) -> None:
    parse_start = time_ms()
    payload = extract_bbox_payload(detections)
    boxes = parse_boxes_strict(payload, source.frame_w, source.frame_h, cfg.max_detections)
    parse_end = time_ms()

    source.processed += 1
    warming_up = source.processed <= cfg.warmup_frames
    if not warming_up:
        metadata_start = time_ms()
        send_metadata(source, detections, boxes)
        metadata_end = time_ms()
        source.profile.add(parse_end - parse_start, metadata_end - metadata_start, len(boxes))
        aggregate_profile.add()


def pull_detections(app: AppRuntime, cfg: AppConfig, aggregate_profile: AggregateProfile) -> None:
    saw_detection = False
    while not _STOP_REQUESTED:
        timeout_ms = DETECTOR_RESULT_TIMEOUT_MS if saw_detection else cfg.initial_detection_timeout_ms
        detections = app.run.pull("detections", timeout_ms)
        if detections is None:
            if hasattr(app.run, "can_pull") and not app.run.can_pull():
                break
            raise RuntimeError(
                "timed out waiting for detections"
                if saw_detection
                else "timed out waiting for initial detection"
            )

        saw_detection = True
        stream_index = stream_index_from_detection(detections, len(app.sources))
        complete_detection(app.sources[stream_index], cfg, aggregate_profile, detections)
        if target_reached(app.sources):
            break


def _request_stop(_signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def run_app(cfg: AppConfig) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _request_stop)

    labels = load_labels(cfg.labels_path)
    aggregate_profile = AggregateProfile(cfg.profile, len(cfg.rtsp_urls))

    model, detector_graph = build_detector_graph(cfg)
    app = AppRuntime(
        model=model,
        graph=pyneat.Graph(options=graph_options(cfg.internal_queue_depth)),
        run=None,
        sources=[],
    )
    try:
        for index in range(len(cfg.rtsp_urls)):
            source = make_source_runtime(cfg, index, labels)
            app.sources.append(source)
            connect_source_graph(app, cfg, source, detector_graph)
        validate_worker_pool_geometry(app.sources)

        if cfg.profile and app_print_backend():
            print(f"Application backend:\n{app.graph.describe_backend()}")

        run_options = realtime_options(cfg.queue_depth)
        if cfg.fan_in_policy == "every_frame":
            if not hasattr(app.graph, "build_fused_realtime_source"):
                raise RuntimeError(
                    "installed pyneat does not expose build_fused_realtime_source; "
                    "install the matching Neat package before using every_frame fan-in"
                )
            app.run = app.graph.build_fused_realtime_source(run_options)
        else:
            app.run = app.graph.build(run_options)
        pull_detections(app, cfg, aggregate_profile)
    finally:
        if app.run is not None:
            app.run.close()
        aggregate_profile.flush()
        for source in app.sources:
            source.profile.flush()
            print(f"[stream {source.index}] processed={source.processed}")
        signal.signal(signal.SIGINT, previous_sigint)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(
                f"Config validated: {args.config} "
                f"(streams={len(cfg.rtsp_urls)}, workers={cfg.workers}, "
                f"queue_depth={cfg.queue_depth}, "
                f"internal_queue_depth={cfg.internal_queue_depth}, "
                f"inference_async={str(INFERENCE_ASYNC).lower()}, "
                f"max_inflight_per_stream={cfg.max_inflight_per_stream}, "
                f"fan_in_policy={cfg.fan_in_policy}, "
                f"insight_visible_streams={effective_insight_visible_streams(cfg)}, "
                "decoder_admission=core)"
            )
            return 0

        load_runtime_dependencies()
        run_app(cfg)
        return 0
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
