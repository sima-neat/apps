from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
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
DEFAULT_NO_DETECTION_TIMEOUT_MS = 30_000
DEFAULT_MAX_MISSED_DETECTION_ROUNDS = 2
DETECTION_PRIMING_OBSERVATIONS = 2
WATCHDOG_CHECK_INTERVAL_S = 0.1
DEFAULT_QUEUE_DEPTH = 4
DEFAULT_INTERNAL_QUEUE_DEPTH = 1
DEFAULT_MAX_INFLIGHT_PER_STREAM = 4
DEFAULT_MAX_INFLIGHT_TOTAL = 8
# Complete MLA work asynchronously so a busy model stage cannot hold the
# shared graph and starve a subset of decoder streams.
INFERENCE_ASYNC = True
DEFAULT_DECODER_BUFFERS = 16
DEFAULT_DECODER_INPUT_BUFFERS = 2
ALL_INSIGHT_STREAMS = -1
MAX_DETECTION_PULLS_PER_ROUND = 64

cv2 = None
pyneat = None
_STOP_REQUESTED = False


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_urls: list[str]
    # Encoded RTSP path used for every stream: "h264" or "h265". Kept as a
    # string because load_app_config runs before the lazy pyneat import.
    codec: str = "h264"
    decode_type: str = "yolo26"
    workers: int = 1
    queue_depth: int = DEFAULT_QUEUE_DEPTH
    internal_queue_depth: int = DEFAULT_INTERNAL_QUEUE_DEPTH
    max_inflight_per_stream: int = DEFAULT_MAX_INFLIGHT_PER_STREAM
    max_inflight_total: int = DEFAULT_MAX_INFLIGHT_TOTAL
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
    no_detection_timeout_ms: int = DEFAULT_NO_DETECTION_TIMEOUT_MS
    max_missed_detection_rounds: int = DEFAULT_MAX_MISSED_DETECTION_ROUNDS
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
    metadata_send_ok: int = 0
    metadata_send_fail: int = 0


@dataclass
class AppRuntime:
    model: object
    graph: object
    run: object
    sources: list[SourceRuntime]


class DetectionFailureKind(Enum):
    NONE = auto()
    STARTUP = auto()
    STREAM_STARVATION = auto()
    GLOBAL_STALL = auto()


@dataclass(frozen=True)
class DetectionFailure:
    kind: DetectionFailureKind = DetectionFailureKind.NONE
    streams: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return self.kind is not DetectionFailureKind.NONE


class DetectionWatchdog:
    """Check detector progress without assuming per-stream wall-clock cadence."""

    def __init__(
        self,
        stream_count: int,
        priming_observations: int,
        startup_timeout_s: float,
        no_progress_timeout_s: float,
        max_missed_completions: int,
        start: float | None = None,
    ) -> None:
        if stream_count <= 0:
            raise ValueError("detection watchdog requires at least one stream")
        if priming_observations <= 0:
            raise ValueError("detection watchdog requires at least one priming observation")
        if startup_timeout_s <= 0 or no_progress_timeout_s <= 0:
            raise ValueError("detection watchdog timeouts must be positive")
        if max_missed_completions <= 0:
            raise ValueError("detection watchdog progress budget must be positive")
        start = time.monotonic() if start is None else start
        self._priming_counts = [0] * stream_count
        self._last_seen_sequence = [0] * stream_count
        self._starvation_latched = [False] * stream_count
        self._priming_observations = priming_observations
        self._primed_streams = 0
        self._total_observations = 0
        self._startup_deadline = start + startup_timeout_s
        self._no_progress_timeout_s = no_progress_timeout_s
        self._max_missed_completions = max_missed_completions
        self._last_any_seen = start
        self._running = False

    def observe(self, stream_index: int, now: float | None = None) -> None:
        if stream_index < 0 or stream_index >= len(self._last_seen_sequence):
            raise IndexError("detection watchdog stream index is out of range")
        now = time.monotonic() if now is None else now
        self._total_observations += 1
        self._last_any_seen = now
        if self._running:
            self._last_seen_sequence[stream_index] = self._total_observations
            # Preserve the first violated progress boundary even if the
            # affected stream recovers later in the same drain batch.
            for index, sequence in enumerate(self._last_seen_sequence):
                if (
                    self._total_observations - sequence
                    > self._max_missed_completions
                ):
                    self._starvation_latched[index] = True
            return

        if self._priming_counts[stream_index] < self._priming_observations:
            self._priming_counts[stream_index] += 1
            if self._priming_counts[stream_index] == self._priming_observations:
                self._primed_streams += 1
        if self._primed_streams == len(self._priming_counts):
            self._running = True
            # Exclude staggered startup work from steady-state starvation
            # accounting by giving every stream the same sequence baseline.
            self._last_seen_sequence = [self._total_observations] * len(
                self._last_seen_sequence
            )

    def startup_complete(self) -> bool:
        return self._running

    def check(self, now: float | None = None) -> DetectionFailure:
        now = time.monotonic() if now is None else now
        if now >= self._last_any_seen + self._no_progress_timeout_s:
            return DetectionFailure(DetectionFailureKind.GLOBAL_STALL)

        if not self._running:
            if now < self._startup_deadline:
                return DetectionFailure()
            return DetectionFailure(
                DetectionFailureKind.STARTUP,
                tuple(
                    index
                    for index, count in enumerate(self._priming_counts)
                    if count < self._priming_observations
                ),
            )

        streams = tuple(
            index
            for index, starved in enumerate(self._starvation_latched)
            if starved
        )
        return (
            DetectionFailure(DetectionFailureKind.STREAM_STARVATION, streams)
            if streams
            else DetectionFailure()
        )


def load_runtime_dependencies() -> None:
    global pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import pyneat as pyneat_module

    pyneat = pyneat_module


def detection_progress_budget(
    stream_count: int, missed_rounds: int, max_inflight_total: int
) -> int:
    return missed_rounds * stream_count + max_inflight_total


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
    parser = argparse.ArgumentParser(
        description="Graph-native multi-stream RTSP YOLO Insight example"
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


def parse_input_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    raise ValueError("input.codec must be h264/avc or h265/hevc")


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
    if cfg.max_inflight_total <= 0:
        raise ValueError("inference.max_inflight_total must be > 0")
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
    if cfg.no_detection_timeout_ms <= 0:
        raise ValueError("runtime.no_detection_timeout_ms must be > 0")
    if cfg.max_missed_detection_rounds <= 0:
        raise ValueError("runtime.max_missed_detection_rounds must be > 0")
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
    if "fan_in_policy" in inference:
        raise ValueError(
            "inference.fan_in_policy was removed; remove it because ordinary "
            "connect()/build() now selects realtime fan-in automatically"
        )
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
    rtsp_urls = parse_streams(raw)

    queue_depth = int_or(inference, "queue_depth", DEFAULT_QUEUE_DEPTH)
    if "queue_depth" not in inference:
        queue_depth = int_or(
            inference,
            "inflight",
            int_or(inference, "detector_queue_depth", DEFAULT_QUEUE_DEPTH),
        )

    model_path_value = string_or(model, "path")
    labels_path_value = string_or(model, "labels", "coco_label.txt")

    cfg = AppConfig(
        model_path=(
            str(resolve_config_relative_path(config_path, model_path_value))
            if model_path_value
            else ""
        ),
        decode_type=normalize_box_decode_type(string_or(model, "decode_type", "yolo26")),
        labels_path=Path(labels_path_value).expanduser(),
        rtsp_urls=rtsp_urls,
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        workers=int_or(inference, "workers", 1),
        queue_depth=queue_depth,
        internal_queue_depth=int_or(
            inference, "internal_queue_depth", DEFAULT_INTERNAL_QUEUE_DEPTH
        ),
        max_inflight_per_stream=int_or(
            inference, "max_inflight_per_stream", DEFAULT_MAX_INFLIGHT_PER_STREAM
        ),
        max_inflight_total=int_or(inference, "max_inflight_total", DEFAULT_MAX_INFLIGHT_TOTAL),
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
        no_detection_timeout_ms=int_or(
            runtime, "no_detection_timeout_ms", DEFAULT_NO_DETECTION_TIMEOUT_MS
        ),
        max_missed_detection_rounds=int_or(
            runtime,
            "max_missed_detection_rounds",
            DEFAULT_MAX_MISSED_DETECTION_ROUNDS,
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


def rtp_timestamp_from_pts_ns(pts_ns: int) -> int:
    if pts_ns < 0:
        return 0
    return ((pts_ns * 90_000) // 1_000_000_000) & 0xFFFFFFFF


def build_metadata_payload(source: SourceRuntime, frame, boxes: list[dict]) -> str:
    pts_ns = int(getattr(frame, "pts_ns", -1))
    frame_id = int(getattr(frame, "frame_id", -1))
    stream_id = str(getattr(frame, "stream_id", "")) or stream_id_for(source.index)
    payload = {
        "type": "object-detection",
        "data": {
            "objects": build_metadata_boxes(
                boxes, source.labels, source.frame_w, source.frame_h
            )
        },
        "timestamp": pts_ns // 1_000_000 if pts_ns >= 0 else -1,
        "frame_id": str(frame_id) if frame_id >= 0 else "",
        "stream_id": stream_id,
        "stream_index": source.index,
        "pts_ns": pts_ns,
        "dts_ns": int(getattr(frame, "dts_ns", -1)),
        "duration_ns": int(getattr(frame, "duration_ns", -1)),
        "input_seq": int(getattr(frame, "input_seq", -1)),
        "orig_input_seq": int(getattr(frame, "orig_input_seq", -1)),
    }
    if pts_ns >= 0:
        payload["rtp_timestamp"] = rtp_timestamp_from_pts_ns(pts_ns)
    return json.dumps(payload, separators=(",", ":"))


def should_log_failure_count(count: int) -> bool:
    return count > 0 and count & (count - 1) == 0


def send_metadata_nonblocking(source: SourceRuntime, payload: str) -> None:
    if source.metadata_sender is None:
        return
    error = ""
    try:
        sent = source.metadata_sender.send_raw_json(payload)
    except Exception as exc:  # Nonblocking UDP failures are observable drops, not pipeline errors.
        sent = False
        error = str(exc)
    if sent:
        source.metadata_send_ok += 1
        return
    source.metadata_send_fail += 1
    failures = source.metadata_send_fail
    if should_log_failure_count(failures):
        print(
            f"[warn] stream {source.index} nonblocking metadata send failed "
            f"(count={failures})" + (f": {error}" if error else ""),
            file=sys.stderr,
            flush=True,
        )


def send_metadata(source: SourceRuntime, frame, boxes: list[dict]) -> None:
    if source.metadata_sender is None:
        return
    send_metadata_nonblocking(source, build_metadata_payload(source, frame, boxes))


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
    return env_bool("HIGH_DENSITY_DETECTOR_VERBOSE", False)


def app_print_backend() -> bool:
    return app_verbose() or env_bool("HIGH_DENSITY_DETECTOR_PRINT_BACKEND", False)


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
    return max(0, env_int("HIGH_DENSITY_DETECTOR_FRAMES_PER_STREAM", 0))


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
    max_inflight_total: int = DEFAULT_MAX_INFLIGHT_TOTAL,
):
    if not hasattr(pyneat, "GraphLinkOptions") or not hasattr(pyneat, "GraphLinkPolicy"):
        raise RuntimeError(
            "installed pyneat does not expose GraphLinkOptions/GraphLinkPolicy; "
            "install the matching Neat package before running this graph-native app"
        )
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.stream_id = stream_id
    link.max_inflight_per_stream = max_inflight_per_stream
    link.max_inflight_total = max_inflight_total
    return link


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


def rtsp_codec(codec: str):
    """Map the parsed `input.codec` config token onto the Core RTSP codec selector."""
    return pyneat.RtspCodec.H265 if codec == "h265" else pyneat.RtspCodec.H264


def make_source_options(
    cfg: AppConfig,
    url: str,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
):
    needs_probe = not cfg.skip_rtsp_probe and (
        cfg.input_width <= 0 or cfg.input_height <= 0 or cfg.input_fps <= 0
    )
    if needs_probe and (width is None or height is None or fps is None):
        width, height, fps = probe_rtsp(url)

    width_out = cfg.input_width if cfg.input_width > 0 else int(width or 0)
    height_out = cfg.input_height if cfg.input_height > 0 else int(height or 0)
    fps_out = cfg.input_fps if cfg.input_fps > 0 else int(fps or 0)

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
    opt.codec = rtsp_codec(cfg.codec)

    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.memory = pyneat.CapsMemory.Any

    if width_out > 0 and height_out > 0:
        opt.dec_width = width_out
        opt.dec_height = height_out
        if cfg.codec == "h264":
            opt.fallback_h264_width = width_out
            opt.fallback_h264_height = height_out
        opt.output_caps.width = width_out
        opt.output_caps.height = height_out
    if fps_out > 0:
        opt.source_fps = fps_out
        opt.output_caps.fps = fps_out
    return opt, fps_out, width_out, height_out



def make_rtsp_encoded_input(opt):
    encoded = pyneat.RtspEncodedInputOptions()
    encoded.url = opt.url
    encoded.codec = opt.codec
    encoded.latency_ms = opt.latency_ms
    encoded.tcp = opt.tcp
    encoded.drop_on_latency = opt.drop_on_latency
    encoded.buffer_mode = opt.buffer_mode
    encoded.insert_queue = opt.insert_queue
    encoded.sync_mode = opt.sync_mode
    encoded.auto_caps_from_stream = opt.auto_caps_from_stream
    encoded.source_fps = opt.source_fps
    encoded.payload_type = opt.payload_type
    if opt.codec != pyneat.RtspCodec.H265:
        encoded.h264_parse_config_interval = opt.h264_parse_config_interval
        encoded.fallback_h264_width = opt.fallback_h264_width
        encoded.fallback_h264_height = opt.fallback_h264_height
    return pyneat.groups.rtsp_encoded_input(encoded)


def append_decoder(
    graph,
    opt,
    decoder_buffers: int,
    decoder_input_buffers: int,
    decoder_tuning: str,
) -> None:
    use_h265 = opt.codec == pyneat.RtspCodec.H265

    decode = pyneat.SimaDecodeOptions()
    decode.type = pyneat.SimaDecodeType.H265 if use_h265 else pyneat.SimaDecodeType.H264
    decode.sima_allocator_type = opt.sima_allocator_type
    decode.out_format = pyneat.Format.NV12
    decode.decoder_name = opt.decoder_name
    decode.raw_output = opt.decoder_raw_output
    decode.next_element = opt.decoder_next_element
    decode.dec_width = opt.dec_width
    decode.dec_height = opt.dec_height
    decode.dec_fps = opt.source_fps
    decode.num_buffers = decoder_buffers
    decode.input_buffers = decoder_input_buffers
    decode.decoder_tuning = decoder_tuning
    decode.memory_opt = decoder_tuning in ("low-memory", "throughput-low-latency")
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


def make_decoder(
    opt,
    decoder_buffers: int,
    decoder_input_buffers: int = DEFAULT_DECODER_INPUT_BUFFERS,
    decoder_tuning: str = "auto",
):
    graph = pyneat.Graph("decoder")
    append_decoder(
        graph, opt, decoder_buffers, decoder_input_buffers, decoder_tuning
    )
    graph.add(pyneat.nodes.output("detector_frame"))
    return graph


def make_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    if hasattr(opt, "verbose") and not app_verbose():
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
    video_options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg.codec))
    video_options.host = cfg.insight_host
    video_options.channel = source.index
    video_options.video_port_base = cfg.video_port_base
    # This sender is fused into the same live pipeline as the decoder fan-in.
    # Do not let every UDP sink participate in the shared preroll barrier.
    video_options.async_ = False
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
        send_options = pyneat.MetadataSenderSendOptions()
        send_options.nonblocking = True
        metadata_sender = pyneat.MetadataSender(metadata_options, send_options)

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
    detector_link = graph_realtime_link(
        cfg.queue_depth,
        stream_id_for(source.index),
        cfg.max_inflight_per_stream,
        cfg.max_inflight_total,
    )

    # Keep the encoded producer explicit. Core internally fuses this ordinary
    # fan-out so VideoSender consumes each read-only encoded AU before decoding,
    # without retaining decoded EV buffers in the application.
    rtsp = make_rtsp_encoded_input(source.source_options)
    decoder = make_decoder(
        source.source_options,
        cfg.decoder_buffers,
        cfg.decoder_input_buffers,
        cfg.decoder_tuning,
    )
    app.graph.connect(rtsp, decoder)
    app.graph.connect(decoder, detector_graph, detector_link)

    if cfg.video_enabled and is_insight_visible_stream(cfg, source.index):
        video_options = make_video_options(cfg, source)
        source.video_port = video_options.video_port
        video_sender = pyneat.groups.video_sender(video_options)
        video_sender.set_name(f"encoded_insight_video_sender_{source.index}")
        video_link = pyneat.GraphLinkOptions()
        video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
        app.graph.connect(rtsp, video_sender, video_link)

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
    max_missed_completions = detection_progress_budget(
        len(app.sources), cfg.max_missed_detection_rounds, cfg.max_inflight_total
    )
    watchdog = DetectionWatchdog(
        len(app.sources),
        DETECTION_PRIMING_OBSERVATIONS,
        cfg.initial_detection_timeout_ms / 1000.0,
        cfg.no_detection_timeout_ms / 1000.0,
        max_missed_completions,
    )
    next_watchdog_check = time.monotonic()
    while not _STOP_REQUESTED:
        did_work = False
        for _ in range(MAX_DETECTION_PULLS_PER_ROUND):
            detections = app.run.pull("detections", 0)
            if detections is None:
                raw_error = app.run.last_error() if hasattr(app.run, "last_error") else ""
                error = str(raw_error or "")
                running = bool(app.run.running()) if hasattr(app.run, "running") else True
                if error:
                    raise RuntimeError(f"failed to pull detections: {error}")
                if not running:
                    raise RuntimeError("detections output closed unexpectedly")
                break

            did_work = True
            stream_index = stream_index_from_detection(detections, len(app.sources))
            watchdog.observe(stream_index)
            complete_detection(app.sources[stream_index], cfg, aggregate_profile, detections)
            if target_reached(app.sources):
                return

        now = time.monotonic()
        check_watchdog = did_work or now >= next_watchdog_check
        failure = watchdog.check(now) if check_watchdog else DetectionFailure()
        if check_watchdog:
            next_watchdog_check = now + WATCHDOG_CHECK_INTERVAL_S
        if failure:
            if failure.kind is DetectionFailureKind.GLOBAL_STALL:
                raise RuntimeError("timed out waiting for any detector progress")
            stream_list = ",".join(str(index) for index in failure.streams)
            if failure.kind is DetectionFailureKind.STARTUP:
                raise RuntimeError(
                    f"timed out waiting for two initial detections from streams: {stream_list}"
                )
            raise RuntimeError(
                f"detector progress budget exceeded for streams: {stream_list}"
            )
        if not did_work:
            time.sleep(0.001)


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
        app.run = app.graph.build(run_options)
        pull_detections(app, cfg, aggregate_profile)
    finally:
        if app.run is not None:
            app.run.close()
        aggregate_profile.flush()
        for source in app.sources:
            source.profile.flush()
            sender_stats = (
                source.metadata_sender.stats()
                if source.metadata_sender is not None
                and hasattr(source.metadata_sender, "stats")
                else None
            )
            print(
                f"[stream {source.index}] processed={source.processed} "
                f"metadata_send_ok={source.metadata_send_ok} "
                f"metadata_send_fail={source.metadata_send_fail} "
                f"metadata_would_block={getattr(sender_stats, 'would_block', 0)} "
                f"metadata_no_buffer_space={getattr(sender_stats, 'no_buffer_space', 0)} "
                f"metadata_send_max_ns={getattr(sender_stats, 'max_send_duration_ns', 0)}",
                flush=True,
            )
        signal.signal(signal.SIGINT, previous_sigint)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            progress_budget = detection_progress_budget(
                len(cfg.rtsp_urls),
                cfg.max_missed_detection_rounds,
                cfg.max_inflight_total,
            )
            print(
                f"Config validated: {args.config} "
                f"(streams={len(cfg.rtsp_urls)}, workers={cfg.workers}, "
                f"queue_depth={cfg.queue_depth}, "
                f"internal_queue_depth={cfg.internal_queue_depth}, "
                f"inference_async={str(INFERENCE_ASYNC).lower()}, "
                f"max_inflight_per_stream={cfg.max_inflight_per_stream}, "
                f"max_inflight_total={cfg.max_inflight_total}, "
                f"max_missed_detection_rounds={cfg.max_missed_detection_rounds}, "
                f"detection_progress_budget={progress_budget}, "
                f"no_detection_timeout_ms={cfg.no_detection_timeout_ms}, "
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
