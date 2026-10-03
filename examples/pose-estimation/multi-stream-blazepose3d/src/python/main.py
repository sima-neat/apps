"""Multi-stream RTSP YOLO26-to-BlazePose Insight application."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import re
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

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
    pose_job_timeout_ms: int = 1000
    frame_limit: int = 0
    insight_host: str = ""
    video_port_base: int = 9000
    metadata_port_base: int = 9100


@dataclass
class FrameJob:
    stream_index: int
    rgb: Any
    people: list[dict[str, Any]]
    frame_id: int
    pts_ns: int
    deadline: float


@dataclass(frozen=True)
class PoseInputContext:
    box: dict[str, Any]
    affine: tuple[float, float, float, float, float, float]


def box_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersection_width = max(
        0.0, min(left["x2"], right["x2"]) - max(left["x1"], right["x1"])
    )
    intersection_height = max(
        0.0, min(left["y2"], right["y2"]) - max(left["y1"], right["y1"])
    )
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left["x2"] - left["x1"]) * max(0.0, left["y2"] - left["y1"])
    right_area = max(0.0, right["x2"] - right["x1"]) * max(
        0.0, right["y2"] - right["y1"]
    )
    union_area = left_area + right_area - intersection
    return intersection / union_area if union_area > 0.0 else 0.0


class PoseSmoother:
    POSITION_ALPHA = 0.45
    CONFIDENCE_ALPHA = 0.20
    FAST_MOTION_ALPHA = 0.90
    FAST_MOTION_THRESHOLD = 0.08
    MINIMUM_MATCH_IOU = 0.15
    RESET_AFTER_NS = 250_000_000
    MAX_COAST_FRAMES = 2
    COAST_CONFIDENCE_DECAY = 0.85

    def __init__(self) -> None:
        self.previous: list[dict[str, Any]] = []
        self.last_pts_ns = -1
        self.missing_frames = 0

    @staticmethod
    def _blend(previous: float, current: float, alpha: float) -> float:
        return previous + alpha * (current - previous)

    def filter(self, poses: list[dict[str, Any]], pts_ns: int) -> list[dict[str, Any]]:
        if (
            pts_ns >= 0
            and self.last_pts_ns >= 0
            and (
                pts_ns <= self.last_pts_ns
                or pts_ns - self.last_pts_ns > self.RESET_AFTER_NS
            )
        ):
            self.previous = []
            self.missing_frames = 0
        if not poses:
            if self.previous and self.missing_frames < self.MAX_COAST_FRAMES:
                self.missing_frames += 1
                coasted = copy.deepcopy(self.previous)
                decay = self.COAST_CONFIDENCE_DECAY**self.missing_frames
                for pose in coasted:
                    pose["presence"] *= decay
                    pose["box"]["score"] *= decay
                    for point, world in zip(
                        pose["keypoints"], pose["world_keypoints"], strict=True
                    ):
                        point["confidence"] *= decay
                        world["confidence"] = point["confidence"]
                return coasted
            return poses
        self.missing_frames = 0
        used: set[int] = set()
        for current in poses:
            matches = [
                (box_iou(current["box"], previous["box"]), index, previous)
                for index, previous in enumerate(self.previous)
                if index not in used
            ]
            if not matches:
                continue
            overlap, previous_index, previous = max(matches, key=lambda match: match[0])
            if overlap < self.MINIMUM_MATCH_IOU:
                continue
            used.add(previous_index)
            box = current["box"]
            previous_box = previous["box"]
            scale = max(1.0, math.hypot(box["x2"] - box["x1"], box["y2"] - box["y1"]))
            center_motion = (
                math.hypot(
                    (box["x1"] + box["x2"] - previous_box["x1"] - previous_box["x2"])
                    * 0.5,
                    (box["y1"] + box["y2"] - previous_box["y1"] - previous_box["y2"])
                    * 0.5,
                )
                / scale
            )
            box_alpha = (
                self.FAST_MOTION_ALPHA
                if center_motion >= self.FAST_MOTION_THRESHOLD
                else self.POSITION_ALPHA
            )
            for coordinate in ("x1", "y1", "x2", "y2"):
                box[coordinate] = self._blend(
                    previous_box[coordinate], box[coordinate], box_alpha
                )
            current["presence"] = self._blend(
                previous["presence"], current["presence"], self.CONFIDENCE_ALPHA
            )
            box["score"] = self._blend(
                previous_box["score"], box["score"], self.CONFIDENCE_ALPHA
            )

            for point, old_point, world, old_world in zip(
                current["keypoints"],
                previous["keypoints"],
                current["world_keypoints"],
                previous["world_keypoints"],
                strict=True,
            ):
                motion = (
                    math.hypot(point["x"] - old_point["x"], point["y"] - old_point["y"])
                    / scale
                )
                alpha = (
                    self.FAST_MOTION_ALPHA
                    if motion >= self.FAST_MOTION_THRESHOLD
                    else self.POSITION_ALPHA
                )
                point["x"] = self._blend(old_point["x"], point["x"], alpha)
                point["y"] = self._blend(old_point["y"], point["y"], alpha)
                point["confidence"] = self._blend(
                    old_point["confidence"], point["confidence"], self.CONFIDENCE_ALPHA
                )
                for coordinate in ("x", "y", "z"):
                    world[coordinate] = self._blend(
                        old_world[coordinate], world[coordinate], alpha
                    )
                world["confidence"] = point["confidence"]

        self.previous = poses
        if pts_ns >= 0:
            self.last_pts_ns = pts_ns
        return poses


@dataclass
class StreamRuntime:
    index: int
    config: StreamConfig
    source_options: Any
    metadata_sender: Any
    width: int
    height: int
    fps: int
    source_graph: Any = None
    source_run: Any = None
    metadata_lock: threading.Lock = field(default_factory=threading.Lock)
    pose_smoother: PoseSmoother = field(default_factory=PoseSmoother)
    metadata_frames: int = 0
    source_frames: int = 0
    completed_rois: int = 0
    outstanding_frames: int = 0
    closed: bool = False


class SharedState:
    def __init__(self, stream_count: int) -> None:
        self.condition = threading.Condition()
        self.detector_mailboxes: list[FrameJob | None] = [None] * stream_count
        self.pose_mailboxes: list[FrameJob | None] = [None] * stream_count
        self.prepared_frames = deque()
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
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return value


def scalar(raw: dict[str, Any], key: str, default):
    value = raw.get(key, default)
    kind = type(default)
    if kind in {str, bool}:
        if not isinstance(value, kind):
            raise TypeError(f"{key} must be a {kind.__name__}")
        return value
    text = str(value)
    pattern = (
        r"[+-]?[0-9]+"
        if kind is int
        else r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    )
    if not re.fullmatch(pattern, text):
        raise ValueError(f"{key} has the wrong numeric type")
    parsed = kind(text)
    if kind is float and not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite")
    return parsed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_codec(value: str) -> str:
    if value not in {"h264", "h265"}:
        raise ValueError("stream codec must be h264 or h265")
    return value


def validate_config(cfg: AppConfig) -> None:
    require(bool(cfg.detector_model_path), "models.detector_path must be set")
    require(bool(cfg.pose_model_path), "models.pose_path must be set")
    require(1 <= len(cfg.streams) <= 4, "streams must contain between 1 and 4 entries")
    require(bool(cfg.insight_host), "output.insight.host must be set")
    require(cfg.latency_ms >= 0, "input.latency_ms must be >= 0")
    require(
        0.0 <= cfg.detector_min_score <= 1.0,
        "detector.min_score must be between 0 and 1",
    )
    require(
        0.0 <= cfg.detector_nms_iou <= 1.0, "detector.nms_iou must be between 0 and 1"
    )
    require(
        1 <= cfg.max_people_per_frame <= 10,
        "pose.max_people_per_frame must be between 1 and 10",
    )
    require(cfg.roi_scale > 0.0, "pose.roi_scale must be > 0")
    require(
        0.0 <= cfg.pose_presence_threshold <= 1.0,
        "pose.presence_threshold must be between 0 and 1",
    )
    require(cfg.pose_job_timeout_ms > 0, "pose.job_timeout_ms must be > 0")
    require(cfg.frame_limit >= 0, "runtime.frames must be >= 0")
    require(
        1 <= cfg.video_port_base <= 65535 and 1 <= cfg.metadata_port_base <= 65535,
        "Insight port bases must be between 1 and 65535",
    )
    ids = [stream.id for stream in cfg.streams]
    channels = [stream.insight_channel for stream in cfg.streams]
    require(len(ids) == len(set(ids)), "stream ids must be unique")
    require(
        len(channels) == len(set(channels)), "stream insight channels must be unique"
    )
    for stream in cfg.streams:
        no_explicit_caps = stream.width == stream.height == stream.fps == 0
        complete_explicit_caps = (
            stream.width > 0 and stream.height > 0 and stream.fps > 0
        )
        require(
            no_explicit_caps or complete_explicit_caps,
            "stream width, height, and fps must either all be omitted or all be > 0",
        )
        require(
            cfg.video_port_base + stream.insight_channel <= 65535,
            "stream video port must be <= 65535",
        )
        require(
            cfg.metadata_port_base + stream.insight_channel <= 65535,
            "stream metadata port must be <= 65535",
        )
    video_ports = {
        cfg.video_port_base + stream.insight_channel for stream in cfg.streams
    }
    metadata_ports = {
        cfg.metadata_port_base + stream.insight_channel for stream in cfg.streams
    }
    require(
        not video_ports & metadata_ports,
        "Insight video and metadata ports must not overlap",
    )


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

    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("streams must be a non-empty list")
    streams: list[StreamConfig] = []
    for index, value in enumerate(raw_streams):
        if not isinstance(value, dict):
            raise TypeError(f"streams[{index}] must be a mapping")
        stream_id = scalar(value, "id", "")
        url = scalar(value, "url", "")
        channel = scalar(value, "insight_channel", -1)
        if not stream_id:
            raise ValueError(f"streams[{index}].id must be set")
        if not url:
            raise ValueError(f"streams[{index}].url must be set")
        if channel < 0:
            raise ValueError(f"streams[{index}].insight_channel must be >= 0")
        width = scalar(value, "width", 0)
        height = scalar(value, "height", 0)
        fps = scalar(value, "fps", 0)
        streams.append(
            StreamConfig(
                stream_id,
                url,
                parse_codec(scalar(value, "codec", "h264")),
                channel,
                width,
                height,
                fps,
            )
        )

    cfg = AppConfig(
        detector_model_path=scalar(models, "detector_path", ""),
        pose_model_path=scalar(models, "pose_path", ""),
        streams=streams,
        tcp=scalar(input_cfg, "tcp", True),
        latency_ms=scalar(input_cfg, "latency_ms", 100),
        detector_min_score=scalar(detector, "min_score", 0.30),
        detector_nms_iou=scalar(detector, "nms_iou", 0.60),
        max_people_per_frame=scalar(pose, "max_people_per_frame", 4),
        roi_scale=scalar(pose, "roi_scale", 1.65),
        pose_presence_threshold=scalar(pose, "presence_threshold", 0.50),
        pose_job_timeout_ms=scalar(pose, "job_timeout_ms", 1000),
        frame_limit=scalar(runtime, "frames", 0),
        insight_host=scalar(insight, "host", ""),
        video_port_base=scalar(insight, "video_port_base", 9000),
        metadata_port_base=scalar(insight, "metadata_port_base", 9100),
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
        "presence": presence,
        "box": box,
        "keypoints": keypoints,
        "world_keypoints": world_keypoints,
    }


def poses_data(poses: list[dict[str, Any]], stream_id: str) -> dict[str, Any]:
    published = []
    for pose_index, pose in enumerate(poses, 1):
        box = pose["box"]
        published.append(
            {
                "id": f"pose_{pose_index}",
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


def world_pose_auxiliary_data(overlay: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "world-pose",
        "renderer": "blazepose-3d",
        "title": "3D Pose",
        "stream_id": overlay["stream_id"],
        "payload": {
            "poses": [
                {
                    "id": pose["id"],
                    "presence": pose["presence"],
                    "keypoints": pose["world_keypoints"],
                }
                for pose in overlay["poses"]
            ]
        },
    }


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
    options.source_fps = fps
    options.insert_queue = True
    options.out_format = pyneat.Format.NV12
    options.decoder_name = f"decoder_{stream.id}"
    options.decoder_raw_output = True
    options.auto_caps_from_stream = True
    options.dec_width = width
    options.dec_height = height
    if stream.codec == "h264":
        options.fallback_h264_width = width
        options.fallback_h264_height = height
    options.output_caps.enable = True
    options.output_caps.format = pyneat.Format.NV12
    options.output_caps.width = width
    options.output_caps.height = height
    options.output_caps.fps = fps
    options.output_caps.memory = pyneat.CapsMemory.Any
    return options


def encoded_format(codec):
    return pyneat.Format.H265 if codec == pyneat.RtspCodec.H265 else pyneat.Format.H264


def encoded_input_options(codec, memory_policy):
    options = pyneat.InputOptions()
    options.payload_type = pyneat.PayloadType.Encoded
    options.format = encoded_format(codec)
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
    decode.dec_fps = options.source_fps
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
            options.source_fps,
            options.output_caps.memory,
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
    options.top_k = 100
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


def pose_input_sample(tensor, pts_ns: int = -1):
    sample = pyneat.make_tensor_sample("pose_input", tensor)
    sample.payload_type = pyneat.PayloadType.Tensor
    sample.media_type = "application/vnd.simaai.tensor"
    sample.format = tensor.semantic.tess.format
    sample.payload_tag = sample.format
    sample.pts_ns = pts_ns
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
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Reliable
    options.overflow_policy = pyneat.OverflowPolicy.Block
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    options.input_timeout_ms = 30000
    options.startup_preflight = False
    return graph, graph.build([pose_input_sample(seed_tensor)], options)


def image_input_sample(name: str, tensor, pts_ns: int = -1):
    sample = pyneat.make_tensor_sample(name, tensor)
    sample.payload_type = pyneat.PayloadType.Image
    sample.media_type = "video/x-raw"
    sample.format = "RGB"
    sample.payload_tag = sample.format
    sample.pts_ns = pts_ns
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
    options = pyneat.RunOptions()
    options.preset = pyneat.RunPreset.Reliable
    options.overflow_policy = pyneat.OverflowPolicy.Block
    options.output_memory = pyneat.OutputMemory.ZeroCopy
    options.input_timeout_ms = 30000
    options.startup_preflight = False
    return graph, graph.build(
        [image_input_sample("detector_input", seed_tensor)], options
    )


def make_rgb_output(stream: StreamRuntime):
    graph = pyneat.Graph(f"rgb_{stream.index}")
    graph.add(pyneat.nodes.input("analytics_frame"))
    graph.add(pyneat.nodes.video_convert())
    graph.add(pyneat.nodes.caps_raw("RGB", stream.width, stream.height, stream.fps))
    graph.add(
        pyneat.nodes.output(f"frame_{stream.index}", pyneat.OutputOptions.latest())
    )
    return graph


def realtime_link(stream: StreamRuntime):
    options = pyneat.GraphLinkOptions()
    options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    options.stream_id = stream.config.id
    options.max_inflight_per_stream = 4
    options.max_inflight_total = 4
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
            )
        )

    for stream in streams:
        source_graph = pyneat.Graph(f"blazepose3d_source_{stream.index}")
        source = make_encoded_source(stream.source_options)
        decoder = make_decoder(stream.source_options)
        source_graph.connect(source, decoder)
        source_graph.connect(decoder, make_rgb_output(stream), realtime_link(stream))
        source_graph.connect(
            source, make_video_sender(cfg, stream), realtime_link(stream)
        )
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
        extract_bbox_payload(sample), stream.width, stream.height, 100
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


def publish_metadata_locked(stream: StreamRuntime, job: FrameJob, poses) -> None:
    timestamp_ms = job.pts_ns // 1_000_000 if job.pts_ns >= 0 else -1
    frame_id = str(job.frame_id) if job.frame_id >= 0 else ""
    poses = stream.pose_smoother.filter(poses, job.pts_ns)
    overlay = poses_data(poses, stream.config.id)
    overlay_data = json.dumps(overlay, separators=(",", ":"))
    auxiliary_data = json.dumps(
        world_pose_auxiliary_data(overlay),
        separators=(",", ":"),
    )
    stream.metadata_sender.send_metadata(
        "pose-estimation", overlay_data, timestamp_ms, frame_id
    )
    stream.metadata_sender.send_metadata(
        "auxiliary-visualization", auxiliary_data, timestamp_ms, frame_id
    )
    stream.metadata_frames += 1


def complete_frame(stream: StreamRuntime, job: FrameJob, poses=None) -> None:
    # Only the pose worker publishes. Dropped frames need no completion queue.
    with stream.metadata_lock:
        if poses is not None:
            publish_metadata_locked(stream, job, poses)
        stream.outstanding_frames -= 1


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
    runtime: AppRuntime,
    mailboxes: list[FrameJob | None],
    next_stream_attr: str,
    wait: bool = True,
) -> FrameJob | None:
    state = runtime.state
    with state.condition:
        if wait:
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


def pump_requests(runtime, run, input_name, output_name, depth, next_request, consume):
    """One owner pipelines bounded requests and consumes results in FIFO order."""
    pending = deque()
    current = None
    while not runtime.state.stopping:
        if current is None and len(pending) < depth:
            current = next_request()
        sent = False
        if current is not None:
            context, sample, deadline = current
            if sample is not None and time.monotonic() >= deadline:
                sample = None  # Unaccepted stale work occupies only an ordering marker.
                current = (context, None, deadline)
            if sample is None or run.try_push(input_name, [sample]):
                pending.append(current)
                current = None
                sent = True
            elif not run.can_push():
                raise RuntimeError(f"{input_name} closed: {run.last_error()}")
        if pending:
            context, sample, deadline = pending[0]
            output = None if sample is None else run.pull(output_name, 0 if sent else 2)
            if sample is None or output is not None:
                pending.popleft()
                consume(context, output)
            elif not run.can_pull():
                raise RuntimeError(f"{output_name} closed: {run.last_error()}")
            elif time.monotonic() >= deadline:
                # Stop after an accepted request times out: its late output must
                # never be mistaken for a subsequent frame's result.
                raise RuntimeError(f"{output_name} inference timed out")
        elif not sent:
            with runtime.state.condition:
                runtime.state.condition.wait(0.001 if current is not None else 0.02)


def close_source_stream(
    runtime: AppRuntime, stream: StreamRuntime, reason: str
) -> None:
    if not stream.closed:
        stream.closed = True
        print(
            f"[warn] stream {stream.config.id} stopped: {reason}",
            file=sys.stderr,
            flush=True,
        )
    with runtime.state.condition:
        runtime.state.condition.notify_all()


def pull_source_frames(runtime: AppRuntime, cfg: AppConfig, stream_index: int) -> None:
    state = runtime.state
    stream = runtime.streams[stream_index]
    output = f"frame_{stream_index}"
    while True:
        with state.condition:
            if state.stopping:
                return
        with stream.metadata_lock:
            frame_limit_reached = (
                cfg.frame_limit > 0 and stream.metadata_frames >= cfg.frame_limit
            )
        if frame_limit_reached:
            close_source_stream(runtime, stream, "runtime frame limit reached")
            return
        try:
            # This Run has one output and no competing consumer. With an infinite
            # timeout, None unambiguously means closure rather than a timeout.
            sample = stream.source_run.pull(output, -1)
            if sample is None:
                close_source_stream(runtime, stream, "source reached end of stream")
                return
            dropped = None
            with state.condition:
                if state.stopping:
                    return
                if cfg.frame_limit > 0:
                    with stream.metadata_lock:
                        remaining = cfg.frame_limit - stream.metadata_frames
                        if remaining <= 0 or stream.outstanding_frames >= remaining:
                            continue
                stream.source_frames += 1
                candidates = (sample.frame_id, sample.orig_input_seq, sample.input_seq)
                valid_ids = (value for value in candidates if value >= 0)
                frame_id = next(valid_ids, stream.source_frames)
                job = FrameJob(
                    stream_index,
                    require_rgb_tensor(sample),
                    [],
                    frame_id,
                    int(sample.pts_ns),
                    time.monotonic() + cfg.pose_job_timeout_ms / 1000.0,
                )
                with stream.metadata_lock:
                    stream.outstanding_frames += 1
                if state.detector_mailboxes[stream_index] is not None:
                    dropped = state.detector_mailboxes[stream_index]
                state.detector_mailboxes[stream_index] = job
                state.condition.notify_all()
            if dropped is not None:
                complete_frame(stream, dropped)
        except Exception as error:  # noqa: BLE001 - isolate a failed source.
            close_source_stream(runtime, stream, str(error))
            return


def run_source_stream(runtime: AppRuntime, cfg: AppConfig, stream_index: int) -> None:
    stream = runtime.streams[stream_index]
    try:
        options = pyneat.RunOptions()
        options.preset = pyneat.RunPreset.Realtime
        options.output_memory = pyneat.OutputMemory.ZeroCopy
        source_run = stream.source_graph.build(options)
        stopping = False
        with runtime.state.condition:
            if runtime.state.stopping:
                stream.closed = True
                stopping = True
                runtime.state.condition.notify_all()
            else:
                stream.source_run = source_run
        if stopping:
            source_run.close()
            return
        pull_source_frames(runtime, cfg, stream_index)
    except Exception as error:  # noqa: BLE001 - isolate a failed source.
        close_source_stream(runtime, stream, str(error))


def detector_worker(runtime: AppRuntime, cfg: AppConfig) -> None:
    state = runtime.state

    def next_request():
        job = take_next_job(
            runtime, state.detector_mailboxes, "next_detector_stream", wait=False
        )
        if job is None:
            return None
        return (
            job,
            image_input_sample("detector_input", job.rgb.cvu(), job.pts_ns),
            job.deadline,
        )

    def consume(job, sample):
        stream = runtime.streams[job.stream_index]
        if sample is not None:
            job.people = select_people(sample, stream, cfg)
        with state.condition:
            dropped = state.pose_mailboxes[job.stream_index]
            state.pose_mailboxes[job.stream_index] = job
            state.condition.notify_all()
        if dropped is not None:
            complete_frame(stream, dropped)

    try:
        pump_requests(
            runtime,
            runtime.detector_run,
            "detector_input",
            "detector_output",
            4,
            next_request,
            consume,
        )
    except Exception as error:  # noqa: BLE001 - worker errors propagate to the owner.
        set_error(runtime, error)


def writable_rgb_view(tensor):
    rgb_view = np.asarray(tensor.to_numpy(copy=False))
    if rgb_view.ndim != 3 or rgb_view.shape[2] != 3:
        raise RuntimeError("RGB frame must be a packed HWC image")
    # stages.preproc uses DLPack, which cannot export a read-only NumPy array.
    return rgb_view if rgb_view.flags.writeable else rgb_view.copy()


def prepare_pose_worker(runtime: AppRuntime, cfg: AppConfig) -> None:
    state = runtime.state
    try:
        while (
            job := take_next_job(runtime, state.pose_mailboxes, "next_pose_stream")
        ) is not None:
            requests = []
            if job.people and time.monotonic() < job.deadline:
                output = pyneat.stages.preproc(
                    [writable_rgb_view(job.rgb)],
                    runtime.pose_model,
                    rois=[
                        pyneat.PreprocessRoi(0, *square_roi(box, cfg.roi_scale))
                        for box in job.people
                    ],
                    image_format=pyneat.PixelFormat.RGB,
                    copy=False,
                )
                if len(output) != len(job.people):
                    raise RuntimeError(
                        "BlazePose Preproc output count does not match ROI count"
                    )
                for index, tensor in enumerate(output):
                    context = PoseInputContext(
                        job.people[index],
                        affine_from_tensor(tensor),
                    )
                    requests.append(
                        (
                            (job, context, index == len(output) - 1),
                            pose_input_sample(tensor.clone().cvu(), job.pts_ns),
                            job.deadline,
                        )
                    )
            else:
                requests.append(((job, None, True), None, job.deadline))
            with state.condition:
                state.condition.wait_for(
                    lambda: state.stopping or not state.prepared_frames
                )
                if state.stopping:
                    return
                state.prepared_frames.append(requests)
                state.condition.notify_all()
    except Exception as error:  # noqa: BLE001 - worker errors propagate to the owner.
        set_error(runtime, error)


def pose_worker(runtime: AppRuntime, cfg: AppConfig) -> None:
    state = runtime.state
    prepared = deque()
    poses = []

    def next_request():
        if not prepared:
            with state.condition:
                if not state.prepared_frames:
                    return None
                prepared.extend(state.prepared_frames.popleft())
                state.condition.notify_all()
        return prepared.popleft()

    def consume(context, sample):
        job, roi, last = context
        stream = runtime.streams[job.stream_index]
        if sample is not None:
            stream.completed_rois += 1
            pose = parse_pose_output(sample, roi, cfg)
            if pose is not None:
                poses.append(pose)
        if last:
            # Requests for a frame are consecutive, so one local accumulator
            # replaces the aggregate map and per-stream completion queues.
            complete_frame(stream, job, list(poses))
            poses.clear()

    try:
        pump_requests(
            runtime,
            runtime.pose_run,
            "pose_input",
            "pose_output",
            4,
            next_request,
            consume,
        )
    except Exception as error:  # noqa: BLE001 - worker errors propagate to the owner.
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
    return decode_pose(
        landmarks,
        world_landmarks,
        context.affine,
        context.box,
        presence_probability,
    )


def all_streams_done(runtime: AppRuntime, frame_limit: int) -> bool:
    return all(
        (frame_limit > 0 and stream.metadata_frames >= frame_limit)
        or (stream.closed and stream.outstanding_frames == 0)
        for stream in runtime.streams
    )


def require_successful_completion(runtime: AppRuntime, frame_limit: int) -> None:
    incomplete = [
        stream.config.id
        for stream in runtime.streams
        if frame_limit > 0 and stream.metadata_frames < frame_limit
    ]
    if incomplete:
        raise RuntimeError(
            f"source streams stopped before reaching runtime.frames={frame_limit}: "
            + ", ".join(incomplete)
        )
    if frame_limit == 0 and all(stream.closed for stream in runtime.streams):
        raise RuntimeError("all source streams stopped")


def stop_runtime(runtime: AppRuntime) -> None:
    with runtime.state.condition:
        runtime.state.stopping = True
        runtime.state.detector_mailboxes = [None] * len(
            runtime.state.detector_mailboxes
        )
        runtime.state.pose_mailboxes = [None] * len(runtime.state.pose_mailboxes)
        runtime.state.prepared_frames.clear()
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


def print_summary(runtime: AppRuntime, elapsed: float) -> None:
    total_frames = sum(stream.metadata_frames for stream in runtime.streams)
    total_rois = sum(stream.completed_rois for stream in runtime.streams)
    print(
        f"[summary aggregate] elapsed_s={elapsed:.3f} "
        f"metadata_fps={total_frames / elapsed if elapsed > 0 else 0.0:.3f} "
        f"pose_fps={total_rois / elapsed if elapsed > 0 else 0.0:.3f}",
        flush=True,
    )


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
    started = time.monotonic()
    source_pullers = [
        threading.Thread(
            target=run_source_stream,
            args=(runtime, cfg, stream.index),
            daemon=True,
        )
        for stream in runtime.streams
    ]
    workers = [
        threading.Thread(target=target, args=(runtime, cfg), daemon=True)
        for target in (detector_worker, prepare_pose_worker, pose_worker)
    ]
    for worker in source_pullers + workers:
        worker.start()
    try:
        while not all_streams_done(runtime, cfg.frame_limit):
            with runtime.state.condition:
                if runtime.state.error is not None:
                    raise runtime.state.error
                runtime.state.condition.wait(0.05)
        require_successful_completion(runtime, cfg.frame_limit)
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 - central owner boundary records shutdown cause.
        set_error(runtime, error)
    finally:
        stop_runtime(runtime)
        unfinished_sources = join_source_workers(runtime, source_pullers)
        for worker in workers:
            worker.join()
        print_summary(runtime, time.monotonic() - started)
    error = runtime.state.error
    if unfinished_sources:
        print(
            "[warn] not waiting for source startup timeout during shutdown: "
            + ", ".join(unfinished_sources),
            file=sys.stderr,
            flush=True,
        )
    if error is not None:
        raise error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        return 2
    try:
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(
                f"Config validated: {args.config} (streams={len(cfg.streams)}, "
                f"max_people_per_frame={cfg.max_people_per_frame})"
            )
            return 0
        run_app(cfg)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary converts failures to exit status.
        print(f"[ERR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
