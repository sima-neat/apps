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


@dataclass(frozen=True)
class AppConfig:
    detector_model_path: str
    pose_model_path: str
    streams: list[StreamConfig]
    tcp: bool = True
    latency_ms: int = 100
    detector_min_score: float = 0.30
    detector_nms_iou: float = 0.60
    max_detections: int = 100
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    max_people_per_frame: int = 4
    roi_scale: float = 1.65
    pose_presence_threshold: float = 0.50
    pose_job_timeout_ms: int = 1000
    max_pending_jobs: int = 64
    frame_limit: int = 0
    insight_host: str = ""
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True


@dataclass(frozen=True)
class FrameIdentity:
    stream_id: str
    frame_id: int
    pts_ns: int
    dts_ns: int
    duration_ns: int
    input_seq: int
    orig_input_seq: int


@dataclass
class FrameJob:
    job_id: int
    stream_index: int
    rgb: Any
    people: list[dict[str, Any]]
    identity: FrameIdentity
    deadline: float


@dataclass(frozen=True)
class PoseInputContext:
    job_id: int
    stream_index: int
    roi_index: int
    roi_count: int
    box: dict[str, Any]
    affine: tuple[float, float, float, float, float, float]
    identity: FrameIdentity


@dataclass(frozen=True)
class PreparedPoseInput:
    roi_index: int
    box: dict[str, Any]
    affine: tuple[float, float, float, float, float, float]
    tensor: Any


@dataclass
class PoseAggregate:
    stream_index: int
    expected: int
    identity: FrameIdentity
    deadline: float
    completed: int = 0
    poses: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamRuntime:
    index: int
    config: StreamConfig
    source_options: Any
    metadata_sender: Any
    width: int
    height: int
    fps: int
    metadata_lock: threading.Lock = field(default_factory=threading.Lock)
    metadata_frames: int = 0
    source_frames: int = 0
    detector_frames: int = 0
    selected_rois: int = 0
    completed_rois: int = 0
    detector_mailbox_drops: int = 0
    pose_mailbox_drops: int = 0
    timed_out_jobs: int = 0


class SharedState:
    def __init__(self, stream_count: int) -> None:
        self.condition = threading.Condition()
        self.detector_mailboxes: list[FrameJob | None] = [None] * stream_count
        self.pose_mailboxes: list[FrameJob | None] = [None] * stream_count
        self.pending_detector_outputs: deque[FrameJob] = deque()
        self.pending_pose_outputs: deque[PoseInputContext] = deque()
        self.aggregates: dict[int, PoseAggregate] = {}
        self.next_detector_stream = 0
        self.next_pose_stream = 0
        self.stopping = False
        self.error: BaseException | None = None


@dataclass
class AppRuntime:
    source_graph: Any
    source_run: Any
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
    parser.add_argument("--validate-config-only", action="store_true")
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
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    raise ValueError("stream codec must be h264/avc or h265/hevc")


def validate_config(cfg: AppConfig) -> None:
    if not cfg.detector_model_path:
        raise ValueError("models.detector_path must be set")
    if not cfg.pose_model_path:
        raise ValueError("models.pose_path must be set")
    if not cfg.streams:
        raise ValueError("streams must be a non-empty list")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if not 0.0 <= cfg.detector_min_score <= 1.0:
        raise ValueError("detector.min_score must be between 0 and 1")
    if not 0.0 <= cfg.detector_nms_iou <= 1.0:
        raise ValueError("detector.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("detector.max_detections must be > 0")
    if cfg.max_inflight_per_stream != -1 and cfg.max_inflight_per_stream <= 0:
        raise ValueError("detector.max_inflight_per_stream must be -1 or > 0")
    if cfg.max_inflight_total != -1 and cfg.max_inflight_total <= 0:
        raise ValueError("detector.max_inflight_total must be -1 or > 0")
    if cfg.max_people_per_frame <= 0:
        raise ValueError("pose.max_people_per_frame must be > 0")
    if cfg.roi_scale <= 0.0:
        raise ValueError("pose.roi_scale must be > 0")
    if not 0.0 <= cfg.pose_presence_threshold <= 1.0:
        raise ValueError("pose.presence_threshold must be between 0 and 1")
    if cfg.pose_job_timeout_ms <= 0:
        raise ValueError("pose.job_timeout_ms must be > 0")
    if cfg.max_pending_jobs <= 0:
        raise ValueError("pose.max_pending_jobs must be > 0")
    if cfg.frame_limit < 0:
        raise ValueError("runtime.frames must be >= 0")
    if cfg.video_port_base <= 0 or cfg.metadata_port_base <= 0:
        raise ValueError("Insight port bases must be > 0")
    ids = [stream.id for stream in cfg.streams]
    channels = [stream.insight_channel for stream in cfg.streams]
    if len(ids) != len(set(ids)):
        raise ValueError("stream ids must be unique")
    if len(channels) != len(set(channels)):
        raise ValueError("stream insight channels must be unique")


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
        stream_id = string_or(value, "id")
        url = string_or(value, "url")
        channel = int_or(value, "insight_channel", -1)
        if not stream_id:
            raise ValueError(f"streams[{index}].id must be set")
        if not url:
            raise ValueError(f"streams[{index}].url must be set")
        if channel < 0:
            raise ValueError(f"streams[{index}].insight_channel must be >= 0")
        streams.append(
            StreamConfig(
                stream_id, url, parse_codec(string_or(value, "codec", "h264")), channel
            )
        )

    cfg = AppConfig(
        detector_model_path=string_or(models, "detector_path"),
        pose_model_path=string_or(models, "pose_path"),
        streams=streams,
        tcp=bool_or(input_cfg, "tcp", True),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        detector_min_score=float_or(detector, "min_score", 0.30),
        detector_nms_iou=float_or(detector, "nms_iou", 0.60),
        max_detections=int_or(detector, "max_detections", 100),
        max_inflight_per_stream=int_or(detector, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(detector, "max_inflight_total", 16),
        max_people_per_frame=int_or(pose, "max_people_per_frame", 4),
        roi_scale=float_or(pose, "roi_scale", 1.65),
        pose_presence_threshold=float_or(pose, "presence_threshold", 0.50),
        pose_job_timeout_ms=int_or(pose, "job_timeout_ms", 1000),
        max_pending_jobs=int_or(pose, "max_pending_jobs", 64),
        frame_limit=int_or(runtime, "frames", 0),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
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


def crop_plan(
    roi: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    x, y, width, height = roi
    if frame_width <= 0 or frame_height <= 0 or width <= 0 or height <= 0:
        return None
    left = max(0, x)
    top = max(0, y)
    right = min(frame_width, x + width)
    bottom = min(frame_height, y + height)
    if right <= left or bottom <= top:
        return None
    image = (left, top, right - left, bottom - top)
    return image, (x - left, y - top, width, height)


def batch_crop_plan(
    rois: list[tuple[int, int, int, int]], frame_width: int, frame_height: int
) -> tuple[
    tuple[int, int, int, int], list[tuple[int, tuple[int, int, int, int]]]
] | None:
    valid = []
    for index, roi in enumerate(rois):
        crop = crop_plan(roi, frame_width, frame_height)
        if crop is not None:
            valid.append((index, roi, crop[0]))
    if not valid:
        return None
    left = min(crop[0] for _, _, crop in valid)
    top = min(crop[1] for _, _, crop in valid)
    right = max(crop[0] + crop[2] for _, _, crop in valid)
    bottom = max(crop[1] + crop[3] for _, _, crop in valid)
    image = (left, top, right - left, bottom - top)
    relative_rois = [
        (index, (roi[0] - left, roi[1] - top, roi[2], roi[3]))
        for index, roi, _ in valid
    ]
    return image, relative_rois


def offset_affine(
    affine: tuple[float, float, float, float, float, float], x: int, y: int
) -> tuple[float, float, float, float, float, float]:
    m00, m01, m02, m10, m11, m12 = affine
    return m00, m01, m02 + x, m10, m11, m12 + y


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def decode_pose(
    raw_landmarks: Any,
    affine: tuple[float, float, float, float, float, float],
    box: dict[str, Any],
    roi_index: int,
) -> dict[str, Any]:
    values = np.asarray(raw_landmarks, dtype=np.float32).reshape(39, 5)
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
    return {"roi_index": roi_index, "box": box, "keypoints": keypoints}


def poses_data(poses: list[dict[str, Any]]) -> dict[str, Any]:
    published = []
    for pose in sorted(poses, key=lambda item: int(item["roi_index"])):
        box = pose["box"]
        published.append(
            {
                "id": f"pose_{int(pose['roi_index']) + 1}",
                "label": "person",
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
            }
        )
    return {"poses": published}


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
    options.top_k = cfg.max_detections
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


def pose_input_sample(tensor, context: PoseInputContext | None):
    sample = pyneat.make_tensor_sample("pose_input", tensor)
    sample.payload_type = pyneat.PayloadType.Tensor
    sample.media_type = "application/vnd.simaai.tensor"
    sample.format = tensor.semantic.tess.format
    sample.payload_tag = sample.format
    if context is not None:
        identity = context.identity
        sample.stream_id = identity.stream_id
        sample.frame_id = identity.frame_id
        sample.pts_ns = identity.pts_ns
        sample.dts_ns = identity.dts_ns
        sample.duration_ns = identity.duration_ns
        sample.input_seq = identity.input_seq
        sample.orig_input_seq = identity.orig_input_seq
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
    return graph, graph.build([pose_input_sample(seed_tensor, None)], options)


def image_input_sample(name: str, tensor, identity: FrameIdentity | None):
    sample = pyneat.make_tensor_sample(name, tensor)
    sample.payload_type = pyneat.PayloadType.Image
    sample.media_type = "video/x-raw"
    sample.format = "RGB"
    sample.payload_tag = sample.format
    if identity is not None:
        sample.stream_id = identity.stream_id
        sample.frame_id = identity.frame_id
        sample.pts_ns = identity.pts_ns
        sample.dts_ns = identity.dts_ns
        sample.duration_ns = identity.duration_ns
        sample.input_seq = identity.input_seq
        sample.orig_input_seq = identity.orig_input_seq
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
        [image_input_sample("detector_input", seed_tensor, None)], options
    )


def frame_output_name(stream_index: int) -> str:
    return f"frame_{stream_index}"


def make_rgb_output(stream: StreamRuntime):
    graph = pyneat.Graph(f"rgb_{stream.index}")
    graph.add(pyneat.nodes.input("analytics_frame"))
    graph.add(pyneat.nodes.video_convert())
    graph.add(pyneat.nodes.caps_raw("RGB", stream.width, stream.height, stream.fps))
    graph.add(
        pyneat.nodes.output(
            frame_output_name(stream.index), pyneat.OutputOptions.latest()
        )
    )
    return graph


def realtime_link(cfg: AppConfig, stream: StreamRuntime):
    options = pyneat.GraphLinkOptions()
    options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    options.stream_id = stream.config.id
    options.max_inflight_per_stream = cfg.max_inflight_per_stream
    options.max_inflight_total = cfg.max_inflight_total
    return options


def build_runtime(cfg: AppConfig) -> AppRuntime:
    streams: list[StreamRuntime] = []
    for index, stream_cfg in enumerate(cfg.streams):
        width, height, fps = probe_rtsp(stream_cfg.url, cfg.tcp)
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
                index, stream_cfg, source_options, metadata_sender, width, height, fps
            )
        )

    source_graph = pyneat.Graph("multistream_blazepose3d_sources")
    for stream in streams:
        source = make_encoded_source(stream.source_options)
        decoder = make_decoder(stream.source_options)
        source_graph.connect(source, decoder)
        source_graph.connect(
            decoder, make_rgb_output(stream), realtime_link(cfg, stream)
        )
        if cfg.video_enabled:
            source_graph.connect(source, make_video_sender(cfg, stream))
        print(
            f"[stream {stream.config.id}] codec={stream.config.codec} "
            f"source={stream.width}x{stream.height}@{stream.fps} "
            f"channel={stream.config.insight_channel} "
            f"video={cfg.video_port_base + stream.config.insight_channel if cfg.video_enabled else 'disabled'} "
            f"metadata={stream.metadata_sender.metadata_port()}",
            flush=True,
        )
    source_options = pyneat.RunOptions()
    source_options.preset = pyneat.RunPreset.Realtime
    source_options.output_memory = pyneat.OutputMemory.ZeroCopy
    source_run = source_graph.build(source_options)

    max_width = max(stream.width for stream in streams)
    max_height = max(stream.height for stream in streams)
    detector_model = make_detector_model(cfg, max_width, max_height)
    detector_graph, detector_run = build_detector_run(
        detector_model, max_width, max_height
    )
    pose_model = make_pose_model(cfg)
    pose_graph, pose_run = build_pose_run(pose_model)
    return AppRuntime(
        source_graph,
        source_run,
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
        extract_bbox_payload(sample), stream.width, stream.height, cfg.max_detections
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


def identity_from_sample(sample) -> FrameIdentity:
    return FrameIdentity(
        str(sample.stream_id),
        int(sample.frame_id),
        int(sample.pts_ns),
        int(sample.dts_ns),
        int(sample.duration_ns),
        int(sample.input_seq),
        int(sample.orig_input_seq),
    )


def select_frame_id(
    frame_id: int, orig_input_seq: int, input_seq: int, pull_sequence: int
) -> int:
    if frame_id >= 0:
        return frame_id
    if orig_input_seq >= 0:
        return orig_input_seq
    if input_seq >= 0:
        return input_seq
    return pull_sequence


def publish_metadata(
    stream: StreamRuntime, identity: FrameIdentity, poses: list[dict[str, Any]]
) -> None:
    timestamp_ms = identity.pts_ns // 1_000_000 if identity.pts_ns >= 0 else -1
    frame_id = str(identity.frame_id) if identity.frame_id >= 0 else ""
    data = json.dumps(poses_data(poses), separators=(",", ":"))
    with stream.metadata_lock:
        stream.metadata_sender.send_metadata(
            "pose-estimation", data, timestamp_ms, frame_id
        )
        stream.metadata_frames += 1


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


def pull_source_frames(runtime: AppRuntime, cfg: AppConfig, stream_index: int) -> None:
    try:
        state = runtime.state
        stream = runtime.streams[stream_index]
        output = frame_output_name(stream_index)
        while True:
            with state.condition:
                if state.stopping:
                    return
            sample = runtime.source_run.pull(output, 50)
            if sample is None:
                continue
            source_identity = identity_from_sample(sample)
            stream.source_frames += 1
            identity = FrameIdentity(
                stream.config.id,
                select_frame_id(
                    source_identity.frame_id,
                    source_identity.orig_input_seq,
                    source_identity.input_seq,
                    stream.source_frames,
                ),
                source_identity.pts_ns,
                source_identity.dts_ns,
                source_identity.duration_ns,
                source_identity.input_seq,
                source_identity.orig_input_seq,
            )
            rgb = require_rgb_tensor(sample)
            with state.condition:
                if state.stopping:
                    return
                job = FrameJob(
                    runtime.next_job_id,
                    stream_index,
                    rgb,
                    [],
                    identity,
                    time.monotonic() + cfg.pose_job_timeout_ms / 1000.0,
                )
                runtime.next_job_id += 1
                if state.detector_mailboxes[stream_index] is not None:
                    stream.detector_mailbox_drops += 1
                state.detector_mailboxes[stream_index] = job
                state.condition.notify_all()
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def dispatch_detector_jobs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while True:
            job = take_next_job(
                runtime, state.detector_mailboxes, "next_detector_stream"
            )
            if job is None:
                return
            stream = runtime.streams[job.stream_index]
            if time.monotonic() >= job.deadline:
                stream.timed_out_jobs += 1
                publish_metadata(stream, job.identity, [])
                continue
            detector_frame = job.rgb.cvu()
            sample = image_input_sample("detector_input", detector_frame, job.identity)
            with state.condition:
                state.condition.wait_for(
                    lambda: (
                        state.stopping
                        or len(state.pending_detector_outputs) < cfg.max_pending_jobs
                    )
                )
                if state.stopping:
                    return
                state.pending_detector_outputs.append(job)
            if not runtime.detector_run.push("detector_input", [sample]):
                with state.condition:
                    if state.stopping:
                        return
                raise RuntimeError("YOLO26 Run rejected a frame input")
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def pull_detector_outputs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while True:
            with state.condition:
                if state.stopping and not state.pending_detector_outputs:
                    return
            sample = runtime.detector_run.pull("detector_output", 20)
            if sample is None:
                continue
            with state.condition:
                if not state.pending_detector_outputs:
                    if state.stopping:
                        return
                    raise RuntimeError(
                        "YOLO26 output arrived without pending frame context"
                    )
                job = state.pending_detector_outputs.popleft()
                state.condition.notify_all()

            stream = runtime.streams[job.stream_index]
            stream.detector_frames += 1
            if time.monotonic() >= job.deadline:
                stream.timed_out_jobs += 1
                publish_metadata(stream, job.identity, [])
                continue
            job.people = select_people(sample, stream, cfg)
            stream.selected_rois += len(job.people)
            if not job.people:
                publish_metadata(stream, job.identity, [])
                continue

            with state.condition:
                if state.stopping:
                    return
                if state.pose_mailboxes[job.stream_index] is not None:
                    stream.pose_mailbox_drops += 1
                state.pose_mailboxes[job.stream_index] = job
                state.condition.notify_all()
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def dispatch_pose_jobs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        while True:
            job = take_next_job(
                runtime, runtime.state.pose_mailboxes, "next_pose_stream"
            )
            if job is None:
                return
            stream = runtime.streams[job.stream_index]
            if time.monotonic() >= job.deadline:
                stream.timed_out_jobs += 1
                publish_metadata(stream, job.identity, [])
                continue
            if not job.people:
                publish_metadata(stream, job.identity, [])
                continue

            rgb_view = np.asarray(job.rgb.to_numpy(copy=False))
            if rgb_view.ndim != 3 or rgb_view.shape[2] != 3:
                raise RuntimeError("RGB frame must be a packed HWC image")
            requested_rois = [
                square_roi(box, cfg.roi_scale) for box in job.people
            ]
            plan = batch_crop_plan(
                requested_rois,
                int(rgb_view.shape[1]),
                int(rgb_view.shape[0]),
            )
            if plan is None:
                publish_metadata(stream, job.identity, [])
                continue
            image, planned_rois = plan
            x, y, width, height = image
            crop = np.array(
                rgb_view[y : y + height, x : x + width],
                dtype=np.uint8,
                copy=True,
                order="C",
            )
            output = pyneat.stages.preproc(
                [crop],
                runtime.pose_model,
                rois=[
                    pyneat.PreprocessRoi(0, *relative_roi)
                    for _, relative_roi in planned_rois
                ],
                image_format=pyneat.PixelFormat.RGB,
                copy=False,
            )
            if len(output) != len(planned_rois):
                raise RuntimeError(
                    "BlazePose Preproc output count does not match ROI count"
                )
            prepared_inputs = []
            for tensor, (person_index, _) in zip(output, planned_rois, strict=True):
                affine = offset_affine(affine_from_tensor(tensor), x, y)
                # Detached asynchronous Runs may retain their input after push().
                # Give each ROI independent EV74 storage before enqueueing.
                prepared_inputs.append(
                    PreparedPoseInput(
                        person_index,
                        job.people[person_index],
                        affine,
                        tensor.clone().cvu(),
                    )
                )

            state = runtime.state
            with state.condition:
                state.condition.wait_for(
                    lambda state=state: (
                        state.stopping or len(state.aggregates) < cfg.max_pending_jobs
                    )
                )
                if state.stopping:
                    return
                state.aggregates[job.job_id] = PoseAggregate(
                    job.stream_index, len(prepared_inputs), job.identity, job.deadline
                )

            for prepared in prepared_inputs:
                context = PoseInputContext(
                    job.job_id,
                    job.stream_index,
                    prepared.roi_index,
                    len(prepared_inputs),
                    prepared.box,
                    prepared.affine,
                    job.identity,
                )
                sample = pose_input_sample(prepared.tensor, context)
                with state.condition:
                    if state.stopping:
                        return
                    state.pending_pose_outputs.append(context)
                if not runtime.pose_run.push("pose_input", [sample]):
                    with state.condition:
                        if state.stopping:
                            return
                    raise RuntimeError("BlazePose Run rejected an ROI input")
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def publish_aggregate(runtime: AppRuntime, job_id: int) -> None:
    state = runtime.state
    with state.condition:
        aggregate = state.aggregates.pop(job_id, None)
        state.condition.notify_all()
    if aggregate is not None:
        publish_metadata(
            runtime.streams[aggregate.stream_index], aggregate.identity, aggregate.poses
        )


def expire_pose_jobs(runtime: AppRuntime) -> None:
    now = time.monotonic()
    with runtime.state.condition:
        expired = [
            (job_id, aggregate.stream_index)
            for job_id, aggregate in runtime.state.aggregates.items()
            if now >= aggregate.deadline
        ]
    for job_id, stream_index in expired:
        runtime.streams[stream_index].timed_out_jobs += 1
        publish_aggregate(runtime, job_id)


def parse_pose_output(sample, context: PoseInputContext, cfg: AppConfig):
    tensors = tensors_from_sample(sample)
    if len(tensors) != 3:
        raise RuntimeError("BlazePose output must contain three tensors")
    presence = np.asarray(tensors[1].to_numpy(copy=True), dtype=np.float32).reshape(-1)
    if presence.size != 1 or float(presence[0]) < cfg.pose_presence_threshold:
        return None
    landmarks = np.asarray(tensors[0].to_numpy(copy=True), dtype=np.float32).reshape(-1)
    if landmarks.size != 195:
        raise RuntimeError("BlazePose screen-landmark output must contain 195 floats")
    return decode_pose(landmarks, context.affine, context.box, context.roi_index)


def pull_pose_outputs(runtime: AppRuntime, cfg: AppConfig) -> None:
    try:
        state = runtime.state
        while True:
            with state.condition:
                if state.stopping:
                    return
            sample = runtime.pose_run.pull("pose_output", 20)
            if sample is None:
                expire_pose_jobs(runtime)
                continue
            with state.condition:
                if not state.pending_pose_outputs:
                    if state.stopping:
                        return
                    raise RuntimeError(
                        "BlazePose output arrived without pending ROI context"
                    )
                context = state.pending_pose_outputs.popleft()
            pose = parse_pose_output(sample, context, cfg)
            complete = False
            with state.condition:
                aggregate = state.aggregates.get(context.job_id)
                if aggregate is not None:
                    aggregate.completed += 1
                    if pose is not None:
                        aggregate.poses.append(pose)
                    complete = aggregate.completed == aggregate.expected
            runtime.streams[context.stream_index].completed_rois += 1
            if complete:
                publish_aggregate(runtime, context.job_id)
            expire_pose_jobs(runtime)
    except Exception as error:  # noqa: BLE001 - propagate worker failures to the owner thread.
        set_error(runtime, error)


def all_streams_done(runtime: AppRuntime, frame_limit: int) -> bool:
    return frame_limit > 0 and all(
        stream.metadata_frames >= frame_limit for stream in runtime.streams
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
        runtime.state.condition.notify_all()
    runtime.source_run.close()
    runtime.detector_run.close()
    runtime.pose_run.close()


def print_summary(runtime: AppRuntime, elapsed: float) -> None:
    for stream in runtime.streams:
        print(
            f"[summary stream={stream.config.id}] source_frames={stream.source_frames} "
            f"detector_frames={stream.detector_frames} "
            f"metadata_frames={stream.metadata_frames} selected_rois={stream.selected_rois} "
            f"completed_rois={stream.completed_rois} "
            f"detector_mailbox_drops={stream.detector_mailbox_drops} "
            f"pose_mailbox_drops={stream.pose_mailbox_drops} "
            f"timed_out_jobs={stream.timed_out_jobs}",
            flush=True,
        )
    total_frames = sum(stream.metadata_frames for stream in runtime.streams)
    total_rois = sum(stream.completed_rois for stream in runtime.streams)
    print(
        f"[summary aggregate] elapsed_s={elapsed:.3f} "
        f"metadata_fps={total_frames / elapsed if elapsed > 0 else 0.0:.3f} "
        f"pose_fps={total_rois / elapsed if elapsed > 0 else 0.0:.3f}",
        flush=True,
    )


def release_runtime_objects(runtime: AppRuntime) -> None:
    """Release Python-owned graph cycles before nanobind module teardown."""
    runtime.source_run = None
    runtime.detector_run = None
    runtime.pose_run = None
    runtime.source_graph = None
    runtime.detector_graph = None
    runtime.pose_graph = None
    runtime.detector_model = None
    runtime.pose_model = None
    runtime.streams.clear()
    gc.collect()


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
            target=pull_source_frames,
            args=(runtime, cfg, stream.index),
            daemon=True,
        )
        for stream in runtime.streams
    ]
    detector_dispatcher = threading.Thread(
        target=dispatch_detector_jobs, args=(runtime, cfg), daemon=True
    )
    detector_puller = threading.Thread(
        target=pull_detector_outputs, args=(runtime, cfg), daemon=True
    )
    pose_dispatcher = threading.Thread(
        target=dispatch_pose_jobs, args=(runtime, cfg), daemon=True
    )
    pose_puller = threading.Thread(
        target=pull_pose_outputs, args=(runtime, cfg), daemon=True
    )
    for puller in source_pullers:
        puller.start()
    detector_dispatcher.start()
    detector_puller.start()
    pose_dispatcher.start()
    pose_puller.start()
    try:
        while not all_streams_done(runtime, cfg.frame_limit):
            with runtime.state.condition:
                if runtime.state.error is not None:
                    raise runtime.state.error
                runtime.state.condition.wait(0.05)
    except KeyboardInterrupt:
        pass
    except Exception as error:  # noqa: BLE001 - central owner boundary records shutdown cause.
        set_error(runtime, error)
    finally:
        stop_runtime(runtime)
        for puller in source_pullers:
            puller.join()
        detector_dispatcher.join()
        detector_puller.join()
        pose_dispatcher.join()
        pose_puller.join()
        print_summary(runtime, time.monotonic() - started)
    error = runtime.state.error
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
