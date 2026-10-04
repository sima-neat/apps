"""Multi-stream RTSP YOLO26-to-BlazePose Insight application."""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

MAX_DETECTIONS = 100
MAX_INFLIGHT_PER_STREAM = 4

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class StreamConfig:
    id: str
    url: str
    codec: str
    insight_channel: int
    width: int = 0
    height: int = 0
    fps: int = 0


@dataclass(frozen=True)
class AppConfig:
    detector_model_path: str
    pose_model_path: str
    streams: list[StreamConfig]
    tcp: bool = True
    latency_ms: int = 100
    detector_min_score: float = 0.30
    detector_nms_iou: float = 0.60
    max_people_per_frame: int = 4
    roi_scale: float = 1.65
    pose_presence_threshold: float = 0.50
    pose_temporal_filter_enabled: bool = True
    frame_limit: int = 0
    insight_host: str = ""
    video_port_base: int = 9000
    metadata_port_base: int = 9100


@dataclass(frozen=True)
class FrameIdentity:
    stream_id: str
    frame_id: int
    pts_ns: int


@dataclass
class FrameJob:
    job_id: int
    stream_index: int
    rgb: Any
    identity: FrameIdentity
    people: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PoseInputContext:
    job_id: int
    roi_index: int
    box: dict[str, Any]
    affine: tuple[float, float, float, float, float, float]


@dataclass
class PoseAggregate:
    stream_index: int
    expected: int
    identity: FrameIdentity
    completed: int = 0
    poses: list[dict[str, Any]] = field(default_factory=list)


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersection_width = max(0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"]))
    intersection_height = max(0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left["x2"] - left["x1"]) * max(0.0, left["y2"] - left["y1"])
    right_area = max(0.0, right["x2"] - right["x1"]) * max(0.0, right["y2"] - right["y1"])
    union_area = left_area + right_area - intersection
    return intersection / union_area if union_area > 0.0 else 0.0


class PoseSmoother:
    """Per-stream exponential smoothing of 2D and world landmarks.

    Each pose is matched to the previous frame's pose with the highest box IoU;
    matched landmarks move ALPHA of the way toward the new estimate, unmatched
    poses pass through unchanged.
    """

    ALPHA = 0.5
    MINIMUM_MATCH_IOU = 0.15

    def __init__(self) -> None:
        self.previous: list[dict[str, Any]] = []

    def filter(self, poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        used: set[int] = set()
        for current in poses:
            match, best_iou = -1, self.MINIMUM_MATCH_IOU
            for index, previous in enumerate(self.previous):
                overlap = 0.0 if index in used else box_iou(current["box"], previous["box"])
                if overlap >= best_iou:
                    match, best_iou = index, overlap
            if match < 0:
                continue
            used.add(match)
            for key, coordinates in (("keypoints", "xy"), ("world_keypoints", "xyz")):
                for point, old in zip(current[key], self.previous[match][key], strict=True):
                    for axis in coordinates:
                        point[axis] = old[axis] + self.ALPHA * (point[axis] - old[axis])
        self.previous = poses
        return poses


def keep_latest(mailboxes: list, index: int, job) -> bool:
    """Store `job` as the newest work for one stream; True when it replaced older
    work, which is dropped so a slow stage never accumulates stale frames."""
    replaced = mailboxes[index] is not None
    mailboxes[index] = job
    return replaced


def send_metadata_pair(send: Callable[[str], bool]) -> bool:
    """Attempt both messages of a frame; the pair counts only if both were sent."""
    overlay_sent = send("pose-estimation")
    world_sent = send("auxiliary-visualization")
    return overlay_sent and world_sent


@dataclass
class StreamRuntime:
    index: int
    config: StreamConfig
    source_options: Any
    metadata_sender: Any
    width: int
    height: int
    fps: int
    pose_temporal_filter_enabled: bool = True
    source_graph: Any = None
    source_run: Any = None
    metadata_lock: threading.Lock = field(default_factory=threading.Lock)
    pose_smoother: PoseSmoother = field(default_factory=PoseSmoother)
    frames_in: int = 0
    frames_out: int = 0
    # Admitted frames that are still queued or inside a model.
    outstanding_frames: int = 0
    closed: bool = False


class SharedState:
    def __init__(self, stream_count: int) -> None:
        self.condition = threading.Condition()
        # Latest-only work per stream; newer frames replace queued ones.
        self.detector_mailboxes: list[FrameJob | None] = [None] * stream_count
        self.pose_mailboxes: list[FrameJob | None] = [None] * stream_count
        # FIFO context for each input pushed to a shared model, in output order.
        self.pending_detector_outputs: deque[FrameJob] = deque()
        self.pending_pose_outputs: deque[PoseInputContext] = deque()
        self.aggregates: dict[int, PoseAggregate] = {}
        self.next_detector_stream = 0
        self.next_pose_stream = 0
        self.stopping = False
        self.error: BaseException | None = None


@dataclass
class AppRuntime:
    detector_graph: Any
    detector_run: Any
    pose_graph: Any
    pose_run: Any
    detector_model: Any
    pose_model: Any
    streams: list[StreamRuntime]
    state: SharedState
    next_job_id: int = 1


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-stream RTSP YOLO26-to-BlazePose Insight application"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def string_or(raw: dict[str, Any], key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def int_or(raw: dict[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def float_or(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be true or false")
    return value


def parse_codec(value: str) -> str:
    if value not in {"h264", "h265"}:
        raise ValueError("stream codec must be h264 or h265")
    return value


def validate_config(cfg: AppConfig) -> None:
    if not cfg.detector_model_path:
        raise ValueError("models.detector_path must be set")
    if not cfg.pose_model_path:
        raise ValueError("models.pose_path must be set")
    if not 1 <= len(cfg.streams) <= 4:
        raise ValueError("streams must contain between 1 and 4 entries")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if not 0.0 <= cfg.detector_min_score <= 1.0:
        raise ValueError("detector.min_score must be between 0 and 1")
    if not 0.0 <= cfg.detector_nms_iou <= 1.0:
        raise ValueError("detector.nms_iou must be between 0 and 1")
    if not 1 <= cfg.max_people_per_frame <= 10:
        raise ValueError("pose.max_people_per_frame must be between 1 and 10")
    if not math.isfinite(cfg.roi_scale) or cfg.roi_scale <= 0.0:
        raise ValueError("pose.roi_scale must be finite and > 0")
    if not 0.0 <= cfg.pose_presence_threshold <= 1.0:
        raise ValueError("pose.presence_threshold must be between 0 and 1")
    if cfg.frame_limit < 0:
        raise ValueError("runtime.frames must be >= 0")
    if not 1 <= cfg.video_port_base <= 65535 or not 1 <= cfg.metadata_port_base <= 65535:
        raise ValueError("Insight port bases must be between 1 and 65535")
    for stream in cfg.streams:
        if not stream.id:
            raise ValueError("stream id must be set")
        if not stream.url:
            raise ValueError("stream url must be set")
        if stream.insight_channel < 0:
            raise ValueError("stream insight_channel must be >= 0")
        no_caps = stream.width == stream.height == stream.fps == 0
        all_caps = stream.width > 0 and stream.height > 0 and stream.fps > 0
        if not (no_caps or all_caps):
            raise ValueError(
                "stream width, height, and fps must either all be omitted or all be > 0"
            )
        if cfg.video_port_base + stream.insight_channel > 65535:
            raise ValueError("stream video port must be <= 65535")
        if cfg.metadata_port_base + stream.insight_channel > 65535:
            raise ValueError("stream metadata port must be <= 65535")
    if len({stream.id for stream in cfg.streams}) != len(cfg.streams):
        raise ValueError("stream ids must be unique")
    channels = {stream.insight_channel for stream in cfg.streams}
    if len(channels) != len(cfg.streams):
        raise ValueError("stream insight channels must be unique")
    video_ports = {cfg.video_port_base + channel for channel in channels}
    if video_ports & {cfg.metadata_port_base + channel for channel in channels}:
        raise ValueError("Insight video and metadata ports must not overlap")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    models = section(raw, "models")
    input_cfg = section(raw, "input")
    detector = section(raw, "detector")
    pose = section(raw, "pose")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    raw_streams = raw.get("streams") or []
    if not isinstance(raw_streams, list) or not all(isinstance(item, dict) for item in raw_streams):
        raise ValueError("streams must be a list of 'key: value' mappings")
    streams = [
        StreamConfig(
            string_or(item, "id"),
            string_or(item, "url"),
            parse_codec(string_or(item, "codec", "h264")),
            int_or(item, "insight_channel", -1),
            int_or(item, "width", 0),
            int_or(item, "height", 0),
            int_or(item, "fps", 0),
        )
        for item in raw_streams
    ]
    cfg = AppConfig(
        detector_model_path=string_or(models, "detector_path"),
        pose_model_path=string_or(models, "pose_path"),
        streams=streams,
        tcp=bool_or(input_cfg, "tcp", True),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        detector_min_score=float_or(detector, "min_score", 0.30),
        detector_nms_iou=float_or(detector, "nms_iou", 0.60),
        max_people_per_frame=int_or(pose, "max_people_per_frame", 4),
        roi_scale=float_or(pose, "roi_scale", 1.65),
        pose_presence_threshold=float_or(pose, "presence_threshold", 0.50),
        pose_temporal_filter_enabled=bool_or(pose, "temporal_filter_enabled", True),
        frame_limit=int_or(runtime, "frames", 0),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
    )
    validate_config(cfg)
    return cfg


def round_half_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def square_roi(box: dict[str, Any], scale: float) -> tuple[int, int, int, int]:
    width = max(0.0, float(box["x2"]) - float(box["x1"]))
    height = max(0.0, float(box["y2"]) - float(box["y1"]))
    side = max(1, round_half_away_from_zero(max(width, height) * scale))
    center_x = (float(box["x1"]) + float(box["x2"])) * 0.5
    center_y = (float(box["y1"]) + float(box["y2"])) * 0.5
    return (
        round_half_away_from_zero(center_x - side * 0.5),
        round_half_away_from_zero(center_y - side * 0.5),
        side,
        side,
    )


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def decode_pose(
    raw_landmarks: Any,
    raw_world_landmarks: Any,
    affine: tuple[float, float, float, float, float, float],
    box: dict[str, Any],
    presence: float,
    roi_index: int,
) -> dict[str, Any]:
    values = np.asarray(raw_landmarks, dtype=np.float32).reshape(39, 5)
    world_values = np.asarray(raw_world_landmarks, dtype=np.float32).reshape(39, 3)
    m00, m01, m02, m10, m11, m12 = affine
    keypoints = []
    for index, raw in enumerate(values[:33]):
        keypoints.append(
            {
                "name": LANDMARK_NAMES[index],
                "x": float(m00 * raw[0] + m01 * raw[1] + m02),
                "y": float(m10 * raw[0] + m11 * raw[1] + m12),
                "confidence": min(sigmoid(float(raw[3])), sigmoid(float(raw[4]))),
            }
        )
    world_keypoints = [
        {
            "name": LANDMARK_NAMES[index],
            "x": float(raw[0]),
            "y": float(raw[1]),
            "z": float(raw[2]),
            "confidence": keypoints[index]["confidence"],
        }
        for index, raw in enumerate(world_values[:33])
    ]
    return {
        "roi_index": roi_index,
        "presence": presence,
        "box": box,
        "keypoints": keypoints,
        "world_keypoints": world_keypoints,
    }


def poses_data(poses: list[dict[str, Any]], stream_id: str) -> dict[str, Any]:
    published = []
    for pose in sorted(poses, key=lambda item: int(item["roi_index"])):
        box = pose["box"]
        published.append(
            {
                "id": f"pose_{int(pose['roi_index']) + 1}",
                "label": "person",
                "presence": round(float(pose["presence"]), 3),
                "confidence": round(float(box["score"]), 3),
                "bbox": [
                    round_half_away_from_zero(float(box["x1"])),
                    round_half_away_from_zero(float(box["y1"])),
                    round_half_away_from_zero(
                        max(0.0, float(box["x2"]) - float(box["x1"]))
                    ),
                    round_half_away_from_zero(
                        max(0.0, float(box["y2"]) - float(box["y1"]))
                    ),
                ],
                "keypoints": [
                    {
                        "name": point["name"],
                        "x": round_half_away_from_zero(float(point["x"])),
                        "y": round_half_away_from_zero(float(point["y"])),
                        "confidence": round(float(point["confidence"]), 3),
                    }
                    for point in pose["keypoints"]
                ],
                "world_keypoints": [
                    {
                        "name": point["name"],
                        "x": round(float(point["x"]), 6),
                        "y": round(float(point["y"]), 6),
                        "z": round(float(point["z"]), 6),
                        "confidence": round(float(point["confidence"]), 3),
                    }
                    for point in pose["world_keypoints"]
                ],
            }
        )
    return {"stream_id": stream_id, "poses": published}


def auxiliary_visualization_data(
    view_id: str,
    renderer: str,
    payload: dict[str, Any],
    title: str | None = None,
) -> dict[str, Any]:
    data = {
        "schema_version": 1,
        "id": view_id,
        "renderer": renderer,
        "payload": payload,
    }
    if title is not None:
        data["title"] = title
    return data


def world_pose_auxiliary_from_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    """Build the 3D view from the 2D message's world keypoints, so both carry
    the same rounded values without decoding them twice."""
    world_poses = [
        {"id": pose["id"], "presence": pose["presence"], "keypoints": pose["world_keypoints"]}
        for pose in overlay["poses"]
    ]
    data = auxiliary_visualization_data(
        "world-pose", "blazepose-3d", {"poses": world_poses}, "3D Pose"
    )
    data["stream_id"] = overlay["stream_id"]
    return data


def rtsp_codec(codec: str):
    return pyneat.RtspCodec.H265 if codec == "h265" else pyneat.RtspCodec.H264


def probe_rtsp(url: str, tcp: bool) -> tuple[int, int, int]:
    if tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = round(capture.get(cv2.CAP_PROP_FPS) or 0)
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"RTSP probe must resolve width, height, and FPS for {url}")
    return width, height, fps


def build_source_options(
    cfg: AppConfig, stream: StreamConfig, width: int, height: int, fps: int
):
    options = pyneat.RtspDecodedInputOptions()
    options.url = stream.url
    options.codec = rtsp_codec(stream.codec)
    options.latency_ms = cfg.latency_ms
    options.tcp = cfg.tcp
    options.payload_type = 96
    # The integer FPS (probed and rounded, or configured) is only the decoder's
    # rate hint. Pinning it into caps would reject NTSC-rate cameras: a 29.97 fps
    # stream negotiates 30000/1001, which a 30/1 caps filter cannot accept.
    options.dec_fps = fps
    options.insert_queue = True
    options.decoder_name = f"decoder_{stream.id}"
    options.decoder_raw_output = True
    options.auto_caps_from_stream = True
    options.dec_width = width
    options.dec_height = height
    if stream.codec == "h264":
        options.fallback_h264_width = width
        options.fallback_h264_height = height
        options.fallback_h264_fps = fps
    return options


def encoded_input_options(codec, memory_policy):
    options = pyneat.InputOptions()
    options.payload_type = pyneat.PayloadType.Encoded
    options.format = pyneat.Format.H265 if codec == pyneat.RtspCodec.H265 else pyneat.Format.H264
    options.memory_policy = memory_policy
    return options


def make_encoded_source(options):
    encoded = pyneat.RtspEncodedInputOptions()
    encoded.url = options.url
    encoded.codec = options.codec
    encoded.latency_ms = options.latency_ms
    encoded.tcp = options.tcp
    encoded.source_fps = options.source_fps
    encoded.payload_type = options.payload_type
    encoded.insert_queue = options.insert_queue
    encoded.auto_caps_from_stream = options.auto_caps_from_stream
    encoded.fallback_h264_width = options.fallback_h264_width
    encoded.fallback_h264_height = options.fallback_h264_height
    encoded.fallback_h264_fps = options.fallback_h264_fps
    graph = pyneat.Graph("encoded_source")
    graph.add(pyneat.groups.rtsp_encoded_input(encoded))
    return graph


def make_decoder(options):
    decode = pyneat.SimaDecodeOptions()
    decode.type = (
        pyneat.SimaDecodeType.H265
        if options.codec == pyneat.RtspCodec.H265
        else pyneat.SimaDecodeType.H264
    )
    decode.sima_allocator_type = options.sima_allocator_type
    decode.out_format = pyneat.Format.NV12
    decode.decoder_name = options.decoder_name
    decode.raw_output = options.decoder_raw_output
    decode.next_element = options.decoder_next_element
    decode.dec_width = options.dec_width
    decode.dec_height = options.dec_height
    decode.dec_fps = options.dec_fps
    decode.num_buffers = options.num_buffers
    graph = pyneat.Graph(f"decoder_{options.decoder_name}")
    graph.connect(
        pyneat.nodes.input(
            "encoded",
            encoded_input_options(options.codec, pyneat.InputMemoryPolicy.Ev74),
        ),
        pyneat.nodes.sima_decode(decode),
    )
    graph.add(
        pyneat.nodes.caps_raw(
            "NV12",
            options.dec_width,
            options.dec_height,
            options.output_caps.fps,
            pyneat.CapsMemory.Any,
        )
    )
    graph.add(pyneat.nodes.output("analytics_frame"))
    return graph


def make_video_sender(cfg: AppConfig, stream: StreamRuntime):
    options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(stream.config.codec))
    options.host = cfg.insight_host
    options.channel = stream.config.insight_channel
    options.video_port_base = cfg.video_port_base
    options.async_ = False
    graph = pyneat.Graph(f"video_{stream.config.id}")
    graph.connect(
        pyneat.nodes.input(
            "encoded",
            encoded_input_options(
                rtsp_codec(stream.config.codec), pyneat.InputMemoryPolicy.SystemMemory
            ),
        ),
        pyneat.groups.video_sender(options),
    )
    return graph


def make_detector_model(cfg: AppConfig, max_width: int, max_height: int):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.input_max_width = max_width
    options.preprocess.input_max_height = max_height
    options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    options.decode_type = pyneat.BoxDecodeType.YoloV26
    options.score_threshold = cfg.detector_min_score
    options.nms_iou_threshold = cfg.detector_nms_iou
    options.top_k = MAX_DETECTIONS
    return pyneat.Model(cfg.detector_model_path, options)


def make_pose_model(cfg: AppConfig):
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.enable = pyneat.AutoFlag.On
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    options.preprocess.resize.enable = pyneat.AutoFlag.On
    options.preprocess.resize.width = 256
    options.preprocess.resize.height = 256
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    options.preprocess.normalize.enable = pyneat.AutoFlag.On
    return pyneat.Model(cfg.pose_model_path, options)


def copy_identity(identity: FrameIdentity, sample) -> None:
    sample.stream_id = identity.stream_id
    sample.frame_id = identity.frame_id
    sample.pts_ns = identity.pts_ns


def reliable_model_run_options():
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Reliable
    options.overflow_policy = pyneat.OverflowPolicy.Block
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    options.input_timeout_ms = 30000
    options.startup_preflight = False
    return options


def pose_input_sample(tensor, identity: FrameIdentity | None):
    sample = pyneat.make_tensor_sample("pose_input", tensor)
    sample.payload_type = pyneat.PayloadType.Tensor
    sample.media_type = "application/vnd.simaai.tensor"
    sample.format = tensor.semantic.tess.format
    sample.payload_tag = sample.format
    if identity is not None:
        copy_identity(identity, sample)
    return sample


def build_pose_run(model):
    inputs = model.input_specs()
    outputs = model.output_specs()
    if len(inputs) != 1 or list(inputs[0].shape) != [-1, -1, 3]:
        raise RuntimeError("BlazePose public input must be dynamic HWC RGB")
    if [list(item.shape) for item in outputs] != [
        [1, 195],
        [1, 1],
        [1, 117],
    ]:
        raise RuntimeError(
            "BlazePose contract must be one input and [1,195], [1,1], [1,117]"
        )
    seed_image = np.zeros((256, 256, 3), dtype=np.uint8)
    seed_roi = pyneat.PreprocessRoi(0, 0, 0, 256, 256)
    seed_tensor = pyneat.stages.preproc(
        [seed_image],
        model,
        rois=[seed_roi],
        image_format=pyneat.PixelFormat.RGB,
        copy=False,
    )[0]
    graph = pyneat.Graph("blazepose_runner")
    graph.add(pyneat.nodes.input("pose_input"))
    graph.add(model.inference())
    graph.add(model.postprocess())
    graph.add(pyneat.nodes.output("pose_output", pyneat.OutputOptions.every_frame(4)))
    return graph, graph.build(
        [pose_input_sample(seed_tensor, None)], reliable_model_run_options()
    )


def image_input_sample(name: str, tensor, identity: FrameIdentity | None):
    sample = pyneat.make_tensor_sample(name, tensor)
    sample.payload_type = pyneat.PayloadType.Image
    sample.media_type = "video/x-raw"
    sample.format = "RGB"
    sample.payload_tag = sample.format
    if identity is not None:
        copy_identity(identity, sample)
    return sample


def build_detector_run(model, max_width: int, max_height: int):
    inputs = model.input_specs()
    outputs = model.output_specs()
    if len(inputs) != 1 or list(inputs[0].shape) != [-1, -1, 3]:
        raise RuntimeError("YOLO26 public input must be dynamic HWC RGB")
    if len(outputs) != 1:
        raise RuntimeError("YOLO26 must expose one decoded BBOX output")
    seed_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    seed_tensor = pyneat.Tensor.from_numpy(
        seed_image,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    graph = pyneat.Graph("yolo26_runner")
    input_options = model.input_appsrc_options(False)
    input_options.block = True
    input_graph = pyneat.Graph()
    input_graph.add(pyneat.nodes.input("detector_input", input_options))
    model_graph = model.graph()
    output_graph = pyneat.Graph()
    output_graph.add(
        pyneat.nodes.output("detector_output", pyneat.OutputOptions.every_frame(4))
    )
    graph.connect(input_graph, model_graph)
    graph.connect(model_graph, output_graph)
    return graph, graph.build(
        [image_input_sample("detector_input", seed_tensor, None)],
        reliable_model_run_options(),
    )


def frame_output_name(stream_index: int) -> str:
    return f"frame_{stream_index}"


def make_rgb_output(stream: StreamRuntime):
    graph = pyneat.Graph(f"rgb_{stream.index}")
    graph.add(pyneat.nodes.input("analytics_frame"))
    graph.add(pyneat.nodes.video_convert())
    graph.add(pyneat.nodes.caps_raw("RGB", stream.width, stream.height))
    graph.add(
        pyneat.nodes.output(
            frame_output_name(stream.index), pyneat.OutputOptions.latest()
        )
    )
    return graph


def realtime_link(stream: StreamRuntime):
    options = pyneat.GraphLinkOptions()
    options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    options.stream_id = stream.config.id
    options.max_inflight_per_stream = MAX_INFLIGHT_PER_STREAM
    options.max_inflight_total = MAX_INFLIGHT_PER_STREAM
    return options


def build_runtime(cfg: AppConfig) -> AppRuntime:
    streams: list[StreamRuntime] = []
    for index, stream_cfg in enumerate(cfg.streams):
        if stream_cfg.width > 0:
            width, height, fps = stream_cfg.width, stream_cfg.height, stream_cfg.fps
        else:
            try:
                width, height, fps = probe_rtsp(stream_cfg.url, cfg.tcp)
            except RuntimeError as error:
                raise RuntimeError(
                    f"{error}; configure width, height, and fps to start while it is offline"
                ) from error
        source_options = build_source_options(cfg, stream_cfg, width, height, fps)
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = stream_cfg.insight_channel
        metadata_options.metadata_port_base = cfg.metadata_port_base
        send_options = pyneat.MetadataSenderSendOptions()
        send_options.nonblocking = True
        metadata_sender = pyneat.MetadataSender(metadata_options, send_options)
        streams.append(
            StreamRuntime(
                index,
                stream_cfg,
                source_options,
                metadata_sender,
                width,
                height,
                fps,
                pose_temporal_filter_enabled=cfg.pose_temporal_filter_enabled,
            )
        )

    for stream in streams:
        source_graph = pyneat.Graph(f"blazepose3d_source_{stream.index}")
        source = make_encoded_source(stream.source_options)
        decoder = make_decoder(stream.source_options)
        source_graph.connect(source, decoder)
        source_graph.connect(decoder, make_rgb_output(stream), realtime_link(stream))
        source_graph.connect(source, make_video_sender(cfg, stream), realtime_link(stream))
        stream.source_graph = source_graph
        print(
            f"[stream {stream.config.id}] codec={stream.config.codec} "
            f"source={stream.width}x{stream.height}@{stream.fps} "
            f"channel={stream.config.insight_channel} "
            f"video={cfg.video_port_base + stream.config.insight_channel} "
            f"metadata={stream.metadata_sender.metadata_port()}",
            flush=True,
        )

    max_width = max(stream.width for stream in streams)
    max_height = max(stream.height for stream in streams)
    detector_model = make_detector_model(cfg, max_width, max_height)
    detector_graph, detector_run = build_detector_run(
        detector_model, max_width, max_height
    )
    pose_model = make_pose_model(cfg)
    pose_graph, pose_run = build_pose_run(pose_model)
    return AppRuntime(
        detector_graph,
        detector_run,
        pose_graph,
        pose_run,
        detector_model,
        pose_model,
        streams,
        SharedState(len(streams)),
    )


def tensors_from_sample(sample) -> list[Any]:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    if sample.kind == pyneat.SampleKind.Bundle:
        tensors = []
        for field_sample in sample.fields:
            tensors.extend(tensors_from_sample(field_sample))
        return tensors
    return []


def extract_tensor_bbox_payload(sample, tensor) -> bytes:
    fmt = getattr(sample, "payload_tag", "") or getattr(sample, "format", "")
    if not fmt and tensor.semantic.tess is not None:
        fmt = tensor.semantic.tess.format
    if fmt and str(fmt).upper() != "BBOX":
        raise RuntimeError(f"capture_expected_bbox format={fmt}")
    payload = bytes(tensor.copy_payload_bytes())
    if not payload:
        raise RuntimeError("capture_empty_payload")
    return payload


def extract_bbox_payload(sample) -> bytes:
    if sample.kind == pyneat.SampleKind.Bundle:
        for field_sample in sample.fields:
            try:
                return extract_bbox_payload(field_sample)
            except RuntimeError:
                continue
        raise RuntimeError("bundle missing BBOX field")
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return extract_tensor_bbox_payload(sample, sample.tensors[0])
    if sample.kind != pyneat.SampleKind.Tensor or sample.tensor is None:
        raise RuntimeError("capture_expected_tensor")
    return extract_tensor_bbox_payload(sample, sample.tensor)


def parse_boxes_strict(
    payload: bytes, width: int, height: int, top_k: int
) -> list[dict[str, Any]]:
    if len(payload) < 4:
        raise RuntimeError("bbox buffer too small")
    count = struct.unpack_from("<I", payload, 0)[0]
    if count > (len(payload) - 4) // 24 or count > top_k:
        raise RuntimeError("bbox header exceeds payload count or configured top-k")
    boxes = []
    for index in range(count):
        x, y, box_width, box_height, score, class_id = struct.unpack_from(
            "<iiiifi", payload, 4 + index * 24
        )
        boxes.append(
            {
                "x1": max(0.0, min(float(x), float(width))),
                "y1": max(0.0, min(float(y), float(height))),
                "x2": max(0.0, min(float(x + box_width), float(width))),
                "y2": max(0.0, min(float(y + box_height), float(height))),
                "score": float(score),
                "class_id": int(class_id),
            }
        )
    return boxes


def select_people(
    sample, stream: StreamRuntime, cfg: AppConfig
) -> list[dict[str, Any]]:
    boxes = parse_boxes_strict(
        extract_bbox_payload(sample), stream.width, stream.height, MAX_DETECTIONS
    )
    people = sorted(
        (box for box in boxes if int(box["class_id"]) == 0),
        key=lambda box: float(box["score"]),
        reverse=True,
    )
    return people[: cfg.max_people_per_frame]


def require_rgb_tensor(sample):
    tensors = tensors_from_sample(sample)
    if len(tensors) != 1:
        raise RuntimeError("RGB frame output must contain one tensor")
    image = tensors[0].semantic.image
    if image is None or image.format != pyneat.PixelFormat.RGB:
        raise RuntimeError("VideoConvert output is not RGB")
    return tensors[0]


def finish_frame(runtime: AppRuntime, stream: StreamRuntime) -> None:
    """Mark one admitted frame finished: published or replaced by newer work."""
    with runtime.state.condition:
        stream.outstanding_frames -= 1
        runtime.state.condition.notify_all()


def publish_frame(
    stream: StreamRuntime, identity: FrameIdentity, poses: list[dict[str, Any]]
) -> None:
    """Send one frame's 2D and 3D metadata. The per-stream lock keeps the two
    messages of one frame from interleaving with another frame's."""
    with stream.metadata_lock:
        if stream.pose_temporal_filter_enabled:
            poses = stream.pose_smoother.filter(poses)
        overlay = poses_data(poses, identity.stream_id)
        payloads = {
            "pose-estimation": json.dumps(overlay, separators=(",", ":")),
            "auxiliary-visualization": json.dumps(
                world_pose_auxiliary_from_overlay(overlay), separators=(",", ":")
            ),
        }
        timestamp_ms = identity.pts_ns // 1_000_000 if identity.pts_ns >= 0 else -1

        def send(metadata_type: str) -> bool:
            try:
                sent = bool(
                    stream.metadata_sender.send_metadata(
                        metadata_type, payloads[metadata_type], timestamp_ms, str(identity.frame_id)
                    )
                )
                error = ""
            except RuntimeError as exc:  # pyneat raises when the sender reports an error.
                sent, error = False, str(exc)
            if not sent:
                print(
                    f"[warn] stream {stream.config.id} {metadata_type} "
                    f"metadata send failed: {error}",
                    file=sys.stderr,
                )
            return sent

        if send_metadata_pair(send):
            stream.frames_out += 1


def affine_from_tensor(tensor) -> tuple[float, float, float, float, float, float]:
    meta = tensor.semantic.preprocess
    if meta is None:
        raise RuntimeError("BlazePose Preproc output is missing affine metadata")
    return (
        float(meta.affine_m00),
        float(meta.affine_m01),
        float(meta.affine_m02),
        float(meta.affine_m10),
        float(meta.affine_m11),
        float(meta.affine_m12),
    )


def set_error(runtime: AppRuntime, error: BaseException) -> None:
    with runtime.state.condition:
        if runtime.state.error is None:
            runtime.state.error = error
        runtime.state.stopping = True
        runtime.state.condition.notify_all()


def take_next_job(
    runtime: AppRuntime, mailboxes: list[FrameJob | None], next_stream_attr: str
) -> FrameJob | None:
    """Wait for queued work and take it round-robin across streams."""
    state = runtime.state
    with state.condition:
        state.condition.wait_for(lambda: state.stopping or any(mailboxes))
        if state.stopping:
            return None
        next_stream = getattr(state, next_stream_attr)
        for offset in range(len(mailboxes)):
            index = (next_stream + offset) % len(mailboxes)
            if mailboxes[index] is not None:
                job = mailboxes[index]
                mailboxes[index] = None
                setattr(state, next_stream_attr, (index + 1) % len(mailboxes))
                return job
    return None


def push_with_context(
    runtime: AppRuntime, run, input_name: str, sample, pending: deque, context, rejection: str
) -> bool:
    """Push one model input without the blocking push binding. Its FIFO context
    is queued first so the output puller can always correlate the result.
    Returns False when the application is stopping."""
    state = runtime.state
    while True:
        with state.condition:
            if state.stopping:
                return False
            pending.append(context)
        if run.try_push(input_name, [sample]):
            return True
        with state.condition:
            if state.stopping:
                return False
            pending.pop()
            if not run.can_push():
                raise RuntimeError(rejection)
            state.condition.wait(0.001)


def close_source_stream(runtime: AppRuntime, stream: StreamRuntime, reason: str) -> None:
    if not stream.closed:
        stream.closed = True
        print(
            f"[warn] stream {stream.config.id} stopped: {reason}",
            file=sys.stderr,
            flush=True,
        )
    with runtime.state.condition:
        runtime.state.condition.notify_all()


def pull_source_frames(runtime: AppRuntime, cfg: AppConfig, stream: StreamRuntime) -> None:
    state = runtime.state
    output = frame_output_name(stream.index)
    while True:
        with state.condition:
            if state.stopping:
                return
        if cfg.frame_limit > 0 and stream.frames_in >= cfg.frame_limit:
            close_source_stream(runtime, stream, "runtime frame limit reached")
            return
        try:
            # This Run has one output and no competing consumer. With an infinite
            # timeout, None unambiguously means closure rather than a timeout.
            sample = stream.source_run.pull(output, -1)
            if sample is None:
                close_source_stream(runtime, stream, "source reached end of stream")
                return
            rgb = require_rgb_tensor(sample)
            with state.condition:
                if state.stopping:
                    return
                stream.frames_in += 1
                identity = FrameIdentity(stream.config.id, stream.frames_in, int(sample.pts_ns))
                job = FrameJob(runtime.next_job_id, stream.index, rgb, identity)
                runtime.next_job_id += 1
                if not keep_latest(state.detector_mailboxes, stream.index, job):
                    stream.outstanding_frames += 1
                state.condition.notify_all()
        except Exception as error:  # noqa: BLE001 - isolate a failed source.
            close_source_stream(runtime, stream, str(error))
            return


def run_source_stream(runtime: AppRuntime, cfg: AppConfig, stream: StreamRuntime) -> None:
    try:
        options = pyneat.RunOptions()
        options.preset = pyneat.RunPreset.Realtime
        options.output_memory = pyneat.OutputMemory.ZeroCopy
        source_run = stream.source_graph.build(options)
        with runtime.state.condition:
            stopping = runtime.state.stopping
            if stopping:
                stream.closed = True
            else:
                stream.source_run = source_run
        if stopping:
            source_run.close()
            return
        pull_source_frames(runtime, cfg, stream)
    except Exception as error:  # noqa: BLE001 - isolate a failed source.
        close_source_stream(runtime, stream, str(error))


def pull_model_output(runtime: AppRuntime, run, output: str, model: str):
    """Pull one output of a shared model; None when the application is stopping.
    Raises when the Run closed on its own."""
    while True:
        sample = run.pull(output, 20)
        if sample is not None:
            return sample
        with runtime.state.condition:
            if runtime.state.stopping:
                return None
        if not run.can_pull():
            detail = run.last_error()
            raise RuntimeError(
                f"{model} output closed unexpectedly" + (f": {detail}" if detail else "")
            )


def dispatch_detector_jobs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while job := take_next_job(runtime, state.detector_mailboxes, "next_detector_stream"):
            sample = image_input_sample("detector_input", job.rgb.cvu(), job.identity)
            if not push_with_context(
                runtime,
                runtime.detector_run,
                "detector_input",
                sample,
                state.pending_detector_outputs,
                job,
                "YOLO26 Run rejected a frame input",
            ):
                return
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def pull_detector_outputs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while (
            sample := pull_model_output(runtime, runtime.detector_run, "detector_output", "YOLO26")
        ) is not None:
            with state.condition:
                if not state.pending_detector_outputs:
                    if state.stopping:
                        return
                    raise RuntimeError("YOLO26 output arrived without pending frame context")
                job = state.pending_detector_outputs.popleft()
            stream = runtime.streams[job.stream_index]
            job.people = select_people(sample, stream, cfg)
            if not job.people:
                # An empty pair clears the stream's previous poses in Insight.
                publish_frame(stream, job.identity, [])
                finish_frame(runtime, stream)
                continue
            with state.condition:
                if state.stopping:
                    return
                dropped = keep_latest(state.pose_mailboxes, job.stream_index, job)
                state.condition.notify_all()
            if dropped:
                finish_frame(runtime, stream)
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def writable_rgb_view(tensor):
    rgb_view = np.asarray(tensor.to_numpy(copy=False))
    if rgb_view.ndim != 3 or rgb_view.shape[2] != 3:
        raise RuntimeError("RGB frame must be a packed HWC image")
    # stages.preproc uses DLPack, which cannot export a read-only NumPy array.
    return rgb_view if rgb_view.flags.writeable else rgb_view.copy()


def dispatch_pose_jobs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while job := take_next_job(runtime, state.pose_mailboxes, "next_pose_stream"):
            rois = [pyneat.PreprocessRoi(0, *square_roi(box, cfg.roi_scale)) for box in job.people]
            output = pyneat.stages.preproc(
                [writable_rgb_view(job.rgb)],
                runtime.pose_model,
                rois=rois,
                image_format=pyneat.PixelFormat.RGB,
                copy=False,
            )
            if len(output) != len(rois):
                raise RuntimeError("BlazePose Preproc output count does not match ROI count")
            # Detached asynchronous Runs may retain their input after push().
            # Give each ROI independent EV74 storage before enqueueing.
            inputs = [
                (
                    PoseInputContext(job.job_id, index, box, affine_from_tensor(tensor)),
                    tensor.clone().cvu(),
                )
                for index, (box, tensor) in enumerate(zip(job.people, output, strict=True))
            ]
            with state.condition:
                state.aggregates[job.job_id] = PoseAggregate(
                    job.stream_index, len(inputs), job.identity
                )
            for context, tensor in inputs:
                if not push_with_context(
                    runtime,
                    runtime.pose_run,
                    "pose_input",
                    pose_input_sample(tensor, job.identity),
                    state.pending_pose_outputs,
                    context,
                    "BlazePose Run rejected an ROI input",
                ):
                    return
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def parse_pose_output(sample, context: PoseInputContext, cfg: AppConfig):
    tensors = tensors_from_sample(sample)
    if len(tensors) != 3:
        raise RuntimeError("BlazePose output must contain three tensors")
    presence = np.asarray(tensors[1].to_numpy(copy=True), dtype=np.float32).reshape(-1)
    if presence.size != 1:
        raise RuntimeError("BlazePose presence output must contain one float")
    if not math.isfinite(float(presence[0])):
        return None
    presence_probability = sigmoid(float(presence[0]))
    if presence_probability < cfg.pose_presence_threshold:
        return None
    landmarks = np.asarray(tensors[0].to_numpy(copy=True), dtype=np.float32).reshape(-1)
    if landmarks.size != 195:
        raise RuntimeError("BlazePose screen-landmark output must contain 195 floats")
    world_landmarks = np.asarray(
        tensors[2].to_numpy(copy=True), dtype=np.float32
    ).reshape(-1)
    if world_landmarks.size != 117:
        raise RuntimeError("BlazePose world-landmark output must contain 117 floats")
    # A non-finite landmark would fail integer rounding when the frame is
    # published, so discard only this ROI's pose; the frame still publishes.
    if not (np.isfinite(landmarks).all() and np.isfinite(world_landmarks).all()):
        return None
    return decode_pose(
        landmarks,
        world_landmarks,
        context.affine,
        context.box,
        presence_probability,
        context.roi_index,
    )


def pull_pose_outputs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while (
            sample := pull_model_output(runtime, runtime.pose_run, "pose_output", "BlazePose")
        ) is not None:
            with state.condition:
                if not state.pending_pose_outputs:
                    if state.stopping:
                        return
                    raise RuntimeError("BlazePose output arrived without pending ROI context")
                context = state.pending_pose_outputs.popleft()
            pose = parse_pose_output(sample, context, cfg)
            with state.condition:
                aggregate = state.aggregates.get(context.job_id)
                if aggregate is None:
                    return  # stop_runtime() cleared the in-flight frames.
                if pose is not None:
                    aggregate.poses.append(pose)
                aggregate.completed += 1
                completed = aggregate.completed == aggregate.expected
                if completed:
                    del state.aggregates[context.job_id]
            if completed:
                stream = runtime.streams[aggregate.stream_index]
                publish_frame(stream, aggregate.identity, aggregate.poses)
                finish_frame(runtime, stream)
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def all_streams_done(runtime: AppRuntime) -> bool:
    return all(stream.closed and stream.outstanding_frames == 0 for stream in runtime.streams)


def require_successful_completion(runtime: AppRuntime, frame_limit: int) -> None:
    if frame_limit == 0:
        raise RuntimeError("all source streams stopped")
    incomplete = [stream.config.id for stream in runtime.streams if stream.frames_in < frame_limit]
    if incomplete:
        raise RuntimeError(
            f"source streams stopped before reaching runtime.frames={frame_limit}: "
            + ", ".join(incomplete)
        )


def stop_runtime(runtime: AppRuntime) -> None:
    with runtime.state.condition:
        runtime.state.stopping = True
        runtime.state.detector_mailboxes = [None] * len(
            runtime.state.detector_mailboxes
        )
        runtime.state.pose_mailboxes = [None] * len(runtime.state.pose_mailboxes)
        runtime.state.pending_detector_outputs.clear()
        runtime.state.pending_pose_outputs.clear()
        runtime.state.aggregates.clear()
        source_runs = [
            stream.source_run
            for stream in runtime.streams
            if stream.source_run is not None
        ]
        runtime.state.condition.notify_all()
    for source_run in source_runs:
        source_run.close()
    runtime.detector_run.close()
    runtime.pose_run.close()


def print_summary(runtime: AppRuntime) -> None:
    for stream in runtime.streams:
        print(
            f"[summary stream={stream.config.id}] frames_in={stream.frames_in} "
            f"frames_out={stream.frames_out}",
            flush=True,
        )


def release_runtime_objects(runtime: AppRuntime) -> None:
    """Release Python-owned graph cycles before nanobind module teardown."""
    runtime.detector_run = None
    runtime.pose_run = None
    runtime.detector_graph = None
    runtime.pose_graph = None
    runtime.detector_model = None
    runtime.pose_model = None
    for stream in runtime.streams:
        stream.source_run = None
        stream.source_graph = None
    runtime.streams.clear()
    gc.collect()


def join_source_workers(
    runtime: AppRuntime,
    source_workers: list[threading.Thread],
    timeout_s: float = 0.5,
) -> list[str]:
    deadline = time.monotonic() + timeout_s
    for worker in source_workers:
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
    return [
        runtime.streams[index].config.id
        for index, worker in enumerate(source_workers)
        if worker.is_alive()
    ]


def run_app(cfg: AppConfig) -> None:
    if not Path(cfg.detector_model_path).is_file():
        raise RuntimeError(f"detector model not found: {cfg.detector_model_path}")
    if not Path(cfg.pose_model_path).is_file():
        raise RuntimeError(f"pose model not found: {cfg.pose_model_path}")
    load_runtime_dependencies()
    runtime = build_runtime(cfg)
    # Each source run starts and pulls on its own thread, so an offline source
    # cannot delay the others; the shared models have dedicated workers.
    source_pullers = [
        threading.Thread(target=run_source_stream, args=(runtime, cfg, stream), daemon=True)
        for stream in runtime.streams
    ]
    model_workers = [
        threading.Thread(target=worker, args=(runtime, cfg), daemon=True)
        for worker in (
            dispatch_detector_jobs,
            pull_detector_outputs,
            dispatch_pose_jobs,
            pull_pose_outputs,
        )
    ]
    for thread in source_pullers + model_workers:
        thread.start()
    try:
        with runtime.state.condition:
            while runtime.state.error is None and not all_streams_done(runtime):
                runtime.state.condition.wait(0.05)
            if runtime.state.error is not None:
                raise runtime.state.error
        require_successful_completion(runtime, cfg.frame_limit)
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 - central owner boundary records shutdown cause.
        set_error(runtime, error)
    finally:
        stop_runtime(runtime)
        unfinished_sources = join_source_workers(runtime, source_pullers)
        for worker in model_workers:
            worker.join()
        print_summary(runtime)
    error = runtime.state.error
    if unfinished_sources:
        print(
            "[warn] not waiting for source startup timeout during shutdown: "
            + ", ".join(unfinished_sources),
            file=sys.stderr,
            flush=True,
        )
    else:
        release_runtime_objects(runtime)
    del runtime
    gc.collect()
    if error is not None:
        raise error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2
    try:
        run_app(load_app_config(args.config))
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to exit status.
        print(f"[ERR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
