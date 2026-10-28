"""USB (UVC) camera YOLO26 object detection with Insight output, using pyneat.

UsbCameraInput captures the camera's MJPEG. Each JPEG is sent to Insight
unchanged and decoded once for the model, in the camera's own sampling (NV12 for
4:2:0, NV16 for 4:2:2), which the model preprocessing reads directly:

    UsbCameraInput -> branch -+-> video_sender (MJPEG passthrough, RTP/JPEG)
                              `-> SimaDecode -> model

Both branches stay inside one Run so the video and the detections share a
GStreamer timeline; Insight correlates the RTP timestamp with the metadata
timestamp and cannot render overlays if they drift apart.

Exit codes: 0 success | 1 invalid configuration or runtime error | 130 interrupted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import struct
import sys
import time
import signal

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

#: One BBOX record: x, y, w, h (int32), score (float32), class_id (int32).
BBOX_RECORD_FORMAT = "<iiiifi"
BBOX_RECORD_SIZE = struct.calcsize(BBOX_RECORD_FORMAT)

#: RTP/JPEG (RFC 2435) stores each side / 8 in one byte.
MAX_RTP_JPEG_SIDE = 2040

np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    """Validated runtime settings resolved from config.yaml."""

    model_path: str
    labels_path: Path
    camera: str  # Stable USB camera ID from usb_camera_ids().
    width: int
    height: int
    fps: int
    override_fragment: str
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int
    profile: bool
    profile_interval: int
    queue_depth: int
    insight_host: str
    video_port: int
    metadata_port: int
    override_format: str = "NV12"  # Decoded format of the override's MJPEG frames.


@dataclass
class PipelineRuntime:
    """Handles the run loop needs, kept alive for the lifetime of the Run."""

    model: object
    graph: object
    run: object
    metadata_sender: object
    labels: list[str]
    video_port: int


class ProfileWindow:
    """Windowed timing. Every figure is a mean over THIS window, not since start:
    a cumulative mean hides a pipeline that degrades halfway through a run."""

    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(self, detection_pull_ms: float, metadata_send_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.detection_pull_ms += detection_pull_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile] frames={self.frames} "
            f"output_fps={output_fps:.2f} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames:.2f} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames:.2f} "
            f"avg_boxes={self.boxes / self.frames:.2f}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0


def load_runtime_dependencies() -> None:
    """Import the on-device modules. Deferred so --help and config validation
    run on a host without pyneat installed."""
    global np, pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import numpy as np_module
    import pyneat as pyneat_module

    np = np_module
    pyneat = pyneat_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB camera YOLO26 Insight example")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration"
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the configuration, print the capture/decode route, and exit",
    )
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
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def validate_config(cfg: AppConfig) -> None:
    """Raise ValueError when a resolved configuration cannot be run."""
    if not cfg.model_path:
        raise ValueError("model.path must be set to a compiled model package")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must point to a labels file")
    if not cfg.camera and not cfg.override_fragment:
        raise ValueError("source.camera must be set")
    if cfg.override_format not in ("NV12", "NV16"):
        raise ValueError("source.override_format must be NV12 or NV16")
    if cfg.width <= 0:
        raise ValueError(f"source.width must be > 0, got {cfg.width}")
    if cfg.height <= 0:
        raise ValueError(f"source.height must be > 0, got {cfg.height}")
    if (cfg.width % 8 or cfg.height % 8
            or cfg.width > MAX_RTP_JPEG_SIDE or cfg.height > MAX_RTP_JPEG_SIDE):
        raise ValueError(
            "source.width and source.height must be multiples of 8 and at most 2040 for RTP/JPEG")
    if cfg.fps <= 0:
        raise ValueError(f"source.fps must be > 0, got {cfg.fps}")
    if cfg.frames < 0:
        raise ValueError(f"inference.frames must be >= 0, got {cfg.frames}")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError(f"inference.min_score must be in [0.0, 1.0], got {cfg.min_score}")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError(f"inference.nms_iou must be in [0.0, 1.0], got {cfg.nms_iou}")
    if cfg.max_detections <= 0:
        raise ValueError(f"inference.max_detections must be > 0, got {cfg.max_detections}")
    if cfg.profile_interval <= 0:
        raise ValueError(f"runtime.profile_interval must be > 0, got {cfg.profile_interval}")
    if cfg.queue_depth <= 0:
        raise ValueError(f"runtime.queue_depth must be > 0, got {cfg.queue_depth}")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not 1 <= cfg.video_port <= 65535:
        raise ValueError(f"output.insight.video_port must be in [1, 65535], got {cfg.video_port}")
    if not 1 <= cfg.metadata_port <= 65535:
        raise ValueError(f"output.insight.metadata_port must be in [1, 65535], got {cfg.metadata_port}")


def build_app_config(raw: dict) -> AppConfig:
    """Map a parsed config mapping onto AppConfig without validating it."""
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    return AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        camera=string_or(source, "camera"),
        width=int_or(source, "width", 1920),
        height=int_or(source, "height", 1080),
        fps=int_or(source, "fps", 30),
        override_fragment=string_or(source, "override_fragment"),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.30),
        nms_iou=float_or(inference, "nms_iou", 0.50),
        max_detections=int_or(inference, "max_detections", 100),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        queue_depth=int_or(runtime, "queue_depth", 3),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        override_format=string_or(source, "override_format", "NV12"),
    )


def load_app_config(config_path: Path) -> AppConfig:
    """Read, resolve, and validate the example configuration."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = build_app_config(raw)
    validate_config(cfg)
    return cfg


def load_labels(labels_path: Path) -> list[str]:
    if not labels_path.is_file():
        raise ValueError(f"labels file does not exist: {labels_path}")
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"labels file is empty: {labels_path}")
    return labels


def camera_caps(cfg: AppConfig) -> str:
    return f"image/jpeg,width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"


def source_description(cfg: AppConfig) -> str:
    if cfg.override_fragment:
        return cfg.override_fragment
    return (f"UsbCameraInput camera={cfg.camera} caps={camera_caps(cfg)} "
            "-> {VideoSender(MJPEG passthrough), SimaDecode(MJPEG) -> model}")


def make_usb_camera_options(cfg: AppConfig):
    opt = pyneat.UsbCameraInputOptions()
    opt.camera_name = cfg.camera
    opt.width = cfg.width
    opt.height = cfg.height
    opt.framerate_num = cfg.fps
    return opt


def parse_bbox_payload(
    payload: bytes, img_w: int, img_h: int, max_detections: int, min_score: float = 0.0
) -> list[dict]:
    """Parse a BBOX payload into detections clamped to the frame."""
    if not payload or len(payload) < 4:
        return []

    declared = struct.unpack_from("<I", payload, 0)[0]
    count = min(declared, (len(payload) - 4) // BBOX_RECORD_SIZE)
    if max_detections > 0:
        count = min(count, max_detections)

    detections: list[dict] = []
    offset = 4
    for _ in range(count):
        x, y, w, h, score, class_id = struct.unpack_from(BBOX_RECORD_FORMAT, payload, offset)
        offset += BBOX_RECORD_SIZE
        if not min_score <= score <= 1 or class_id < 0:
            continue
        x1, y1 = max(0, min(x, img_w)), max(0, min(y, img_h))
        x2, y2 = max(0, min(x + w, img_w)), max(0, min(y + h, img_h))
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
            }
        )
    return detections


def extract_tensor_bbox_payload(sample, tensor) -> bytes:
    if tensor is None:
        raise RuntimeError("detection sample has no tensor")
    payload = tensor.copy_payload_bytes()
    if not payload:
        raise RuntimeError("detection sample has an empty payload")
    return payload


def extract_bbox_payload(sample) -> bytes:
    """Pull the BBOX bytes out of whichever Sample shape the Run returns."""
    if sample.kind == pyneat.SampleKind.Bundle:
        for field in sample.fields:
            try:
                return extract_bbox_payload(field)
            except RuntimeError:
                continue
        raise RuntimeError("bundle has no BBOX field")

    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])

    if sample.kind != pyneat.SampleKind.Tensor:
        raise RuntimeError("expected a tensor sample")
    return extract_tensor_bbox_payload(sample, sample.tensor)


def class_label(class_id: int, labels: list[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return "unknown"


def build_metadata_boxes(
    boxes: list[dict], labels: list[str], frame_w: int, frame_h: int
) -> list[dict]:
    """Convert detections to Insight's object-detection contract (xywh, clamped)."""
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
        metadata_boxes.append(
            {
                "id": f"obj_{index}",
                "label": class_label(int(box["class_id"]), labels),
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_boxes


def make_model(cfg: AppConfig, fmt):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = (
        pyneat.PreprocessColorFormat.NV16 if fmt == pyneat.Format.NV16
        else pyneat.PreprocessColorFormat.NV12)
    opt.preprocess.input_max_width = cfg.width
    opt.preprocess.input_max_height = cfg.height
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_video_graph(cfg: AppConfig):
    sender_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
    sender_options.host = cfg.insight_host
    sender_options.channel = 0
    sender_options.video_port_base = cfg.video_port

    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_options))
    return graph, sender_options.video_port


def build_pipeline(cfg: AppConfig) -> PipelineRuntime:
    # The camera is checked once; its info gives the decoded layout (NV12 for 4:2:0 JPEGs,
    # NV16 for 4:2:2) and builds the node.
    camera_options = make_usb_camera_options(cfg)
    camera = None if cfg.override_fragment else pyneat.usb_camera_info(camera_options)
    fmt = (camera.decoded_format if camera is not None
           else getattr(pyneat.Format, cfg.override_format))
    labels = load_labels(cfg.labels_path)
    model = make_model(cfg, fmt)
    video_graph, video_port = build_video_graph(cfg)

    # The decoder writes the camera's own sampling and the model preprocessing reads it
    # directly, so a 4:2:2 camera's NV16 reaches the EV74 with no conversion stage or copy.
    decode = pyneat.SimaDecodeOptions()
    decode.type = pyneat.SimaDecodeType.MJPEG
    decode.raw_output = True
    decode.out_format = fmt
    decode.dec_width, decode.dec_height = cfg.width, cfg.height
    decode.dec_fps = cfg.fps
    model_graph = pyneat.Graph("model")
    model_graph.add(pyneat.nodes.input("model"))
    model_graph.add(pyneat.nodes.sima_decode(decode))
    model_graph.add(model)

    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    # RealtimeLatestByStream: if one branch falls behind, drop its stale frames rather
    # than back-pressuring the camera. The video branch must never stall the MLA.
    live = pyneat.GraphLinkOptions()
    live.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

    source_graph = pyneat.Graph("capture")
    if camera is None:
        source_graph.add(pyneat.nodes.custom(cfg.override_fragment, pyneat.InputRole.Source))
    else:
        source_graph.add(pyneat.nodes.usb_camera_input(camera_options, camera))
    # The override fragment supplies MJPEG too, so both sources take the same branches.

    branch = pyneat.graphs.branch("camera", ["video", "model"])
    graph = pyneat.Graph("usb_camera_object_detector")
    graph.connect(source_graph, branch)
    graph.connect(branch, video_graph, live)
    graph.connect(branch, model_graph, live)
    graph.connect(model_graph, detections_graph)

    if cfg.profile:
        print(f"Backend:\n{graph.describe_backend()}", flush=True)

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = 0
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata_sender = pyneat.MetadataSender(metadata_options)

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = cfg.queue_depth
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run_options.input_timeout_ms = 20000
    run = graph.build(run_options)

    source_label = "override" if cfg.override_fragment else cfg.camera
    print(
        f"source={source_label} stream={cfg.width}x{cfg.height}@{cfg.fps} "
        f"model={cfg.model_path} insight={cfg.insight_host} video={video_port} "
        f"metadata={metadata_sender.metadata_port()} channel=0",
        flush=True,
    )
    return PipelineRuntime(
        model=model,
        graph=graph,
        run=run,
        metadata_sender=metadata_sender,
        labels=labels,
        video_port=video_port,
    )


def send_metadata(runtime: PipelineRuntime, cfg: AppConfig, sample, boxes: list[dict]) -> None:
    metadata_boxes = build_metadata_boxes(boxes, runtime.labels, cfg.width, cfg.height)
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    if not runtime.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": metadata_boxes}, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    ):
        raise RuntimeError("metadata send failed")


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    detections = 0
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            pull_start = time_ms()
            sample = runtime.run.pull("detections", 20000)
            pull_end = time_ms()
            if sample is None:
                raise RuntimeError("timed out waiting for detections")

            boxes = parse_bbox_payload(
                extract_bbox_payload(sample), cfg.width, cfg.height, cfg.max_detections, cfg.min_score
            )

            metadata_start = time_ms()
            send_metadata(runtime, cfg, sample, boxes)
            metadata_end = time_ms()

            processed += 1
            detections += len(boxes)
            profile.add(pull_end - pull_start, metadata_end - metadata_start, len(boxes))

    finally:
        runtime.run.close()

    profile.flush()
    print(
        f"processed={processed} detections={detections} "
        f"video_sender={cfg.insight_host}:{runtime.video_port}",
        flush=True,
    )
    return processed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.config.is_file():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        cfg = load_app_config(args.config)
        labels = load_labels(cfg.labels_path)
    except (ValueError, TypeError, yaml.YAMLError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.validate_config_only:
        source_label = "override" if cfg.override_fragment else cfg.camera
        print(
            f"[validate] model={cfg.model_path} classes={len(labels)} "
            f"source={source_label} stream={cfg.width}x{cfg.height}@{cfg.fps} "
            f"min_score={cfg.min_score:.2f} nms_iou={cfg.nms_iou:.2f} "
            f"max_detections={cfg.max_detections} queue_depth={cfg.queue_depth} "
            f"insight={cfg.insight_host}:{cfg.video_port}/{cfg.metadata_port}"
        )
        print(f"[validate] source_pipeline={source_description(cfg)}")
        print("[validate] configuration OK")
        return 0

    def request_stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    try:
        load_runtime_dependencies()
        runtime = build_pipeline(cfg)
        run_pipeline(runtime, cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # noqa: BLE001 - report any runtime failure to the caller
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
