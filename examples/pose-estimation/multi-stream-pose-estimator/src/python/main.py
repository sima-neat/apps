"""Multi-camera RTSP YOLO26 pose Insight example using pyneat."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    rtsp_urls: list[str]
    # Encoded RTSP path used for every stream: "h264" or "h265". Kept as a
    # string because load_app_config runs before the lazy pyneat import.
    codec: str = "h264"
    latency_ms: int = 100
    tcp: bool = True
    input_max_width: int = 1920
    input_max_height: int = 1080
    frames: int = 0
    fps: int = 0
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_poses: int = 50
    min_keypoint_visibility: float = 0.30
    profile: bool = False
    warmup_frames: int = 30
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    save_dir: str = ""
    save_every: int = 0


@dataclass
class StreamRuntime:
    index: int
    url: str
    source_options: object
    metadata_sender: object
    profile: ProfileWindow
    latest_debug_frame: object | None
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int
    processed: int = 0
    closed: bool = False


@dataclass
class AppRuntime:
    graph: object
    run: object | None
    model: object | None
    streams: list[StreamRuntime]


class ProfileWindow:
    def __init__(self, enabled: bool, stream_index: int, interval: int = 100) -> None:
        self.enabled = enabled
        self.stream_index = stream_index
        self.interval = interval
        self.frames = 0
        self.poses = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(
        self, detection_pull_ms: float, metadata_send_ms: float, pose_count: int
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.poses += pose_count
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
            f"[profile stream={self.stream_index}] frames={self.frames} "
            f"output_fps={output_fps} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_poses={self.poses / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.poses = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0


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
        description="Multi-camera RTSP YOLO26 pose Insight example"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")  # noqa: TRY004
    return value


def string_or(raw: dict, key: str, default: str = "") -> str:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")  # noqa: TRY004
    return value


def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")  # noqa: TRY004
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")  # noqa: TRY004
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")  # noqa: TRY004
    return value


def parse_input_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    raise ValueError("input.codec must be h264/avc or h265/hevc")


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not cfg.rtsp_urls:
        raise ValueError("streams must be set")
    if len(cfg.rtsp_urls) > 4:
        raise ValueError("this phase supports up to four streams")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
    if cfg.input_max_width <= 0:
        raise ValueError("input.max_width must be > 0")
    if cfg.input_max_height <= 0:
        raise ValueError("input.max_height must be > 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if cfg.max_inflight_per_stream != -1 and cfg.max_inflight_per_stream <= 0:
        raise ValueError("inference.max_inflight_per_stream must be -1 or > 0")
    if cfg.max_inflight_total != -1 and cfg.max_inflight_total <= 0:
        raise ValueError("inference.max_inflight_total must be -1 or > 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_poses <= 0:
        raise ValueError("inference.max_poses must be > 0")
    if not 0.0 <= cfg.min_keypoint_visibility <= 1.0:
        raise ValueError("output.min_keypoint_visibility must be between 0 and 1")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")  # noqa: TRY004

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")

    streams = raw.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("streams must be a non-empty list")
    rtsp_urls = []
    for index, value in enumerate(streams):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"streams[{index}] must be a non-empty string")
        rtsp_urls.append(value)

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        rtsp_urls=rtsp_urls,
        codec=parse_input_codec(string_or(input_cfg, "codec", "h264")),
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        input_max_width=int_or(input_cfg, "max_width", 1920),
        input_max_height=int_or(input_cfg, "max_height", 1080),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 16),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_poses=int_or(inference, "max_poses", 50),
        min_keypoint_visibility=float_or(output, "min_keypoint_visibility", 0.30),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""COCO keypoint names in the order the BoxDecode pose payload emits them.

The order is fixed by the wire format, not by configuration: ``decode_pose`` returns an
``[N, 17, 3]`` array whose second axis is positional. Insight joins skeleton edges by name,
so these strings are part of the published metadata contract.
"""

COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
"""Skeleton edges as index pairs into ``COCO_KEYPOINT_NAMES``, matching Insight's topology."""


def tensor_to_numpy(tensor) -> object:
    # A frame with no person decodes to a zero-row tensor. Copying one throws, because a
    # zero-byte payload has nothing to map, so treat "no rows" as an empty result.
    shape = tuple(getattr(tensor, "shape", ()) or ())
    if not shape or 0 in shape:
        return np.empty(shape or (0,), dtype=np.float32)
    return np.asarray(tensor.to_numpy(copy=True))


def extract_tensors(sample) -> list:
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


def decode_poses(sample, frame_w: int, frame_h: int, max_poses: int) -> list[dict]:
    """Decode one pose sample into pose records in source-frame pixel space.

    ``decode_pose`` returns boxes as ``[N, 6]`` (x1, y1, x2, y2, score, class_id) and
    keypoints as ``[N, 17, 3]`` (x, y, visibility), positionally aligned. Clamping applies to
    box coordinates only; keypoints are emitted unclamped, so the visibility floor and the
    drawing code are what keep stray joints off the overlay.
    """
    tensors = extract_tensors(sample)
    if not tensors:
        raise RuntimeError("pose sample carried no tensors")

    decoded = pyneat.decode_pose(tensors, clamp_to=(frame_w, frame_h), top_k=max_poses)
    poses: list[dict] = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).reshape((-1, 6))
        keypoints = tensor_to_numpy(item.keypoints).reshape((-1, 17, 3))
        if boxes.shape[0] != keypoints.shape[0]:
            raise RuntimeError(
                f"pose decode returned {boxes.shape[0]} boxes but {keypoints.shape[0]} keypoint sets"
            )
        for box, points in zip(boxes, keypoints):
            if len(poses) >= max_poses:
                return poses
            poses.append(
                {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "score": float(box[4]),
                    "keypoints": [
                        {"x": float(x), "y": float(y), "visibility": float(v)}
                        for x, y, v in points
                    ],
                }
            )
    return poses


def pose_metadata_data(poses: list[dict], min_visibility: float) -> dict:
    """Build the ``data`` object Insight's ``pose-estimation`` overlay consumes.

    Insight joins skeleton edges by keypoint name and independently hides joints at or below
    0.3 confidence, so publishing below ``min_visibility`` only inflates the datagram.
    """
    published = []
    for index, pose in enumerate(poses, start=1):
        keypoints = [
            {
                "name": COCO_KEYPOINT_NAMES[k],
                "x": point["x"],
                "y": point["y"],
                "confidence": point["visibility"],
            }
            for k, point in enumerate(pose["keypoints"])
            if point["visibility"] >= min_visibility
        ]
        published.append(
            {
                "id": f"pose_{index}",
                "label": "person",
                "confidence": pose["score"],
                "bbox": [
                    pose["x1"],
                    pose["y1"],
                    max(0.0, pose["x2"] - pose["x1"]),
                    max(0.0, pose["y2"] - pose["y1"]),
                ],
                "keypoints": keypoints,
            }
        )
    return {"poses": published}


def rtsp_codec(codec: str):
    """Map the parsed `input.codec` config token onto the Core RTSP codec selector."""
    return pyneat.RtspCodec.H265 if codec == "h265" else pyneat.RtspCodec.H264


def probe_rtsp(url: str, tcp: bool) -> tuple[int, int, int]:
    if tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = round(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")
    return width, height, fps


def build_source_options(cfg: AppConfig, url: str, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.codec = rtsp_codec(cfg.codec)
    opt.dec_width = width
    opt.dec_height = height
    if cfg.codec == "h264":
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
    opt.source_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def output_caps_enabled(caps) -> bool:
    return caps.enable or caps.width > 0 or caps.height > 0 or caps.fps > 0


def build_encoded_source_graph(opt) -> pyneat.Graph:
    source = pyneat.Graph("rtsp_encoded_source")

    encoded_opt = pyneat.RtspEncodedInputOptions()
    encoded_opt.url = opt.url
    encoded_opt.codec = opt.codec
    encoded_opt.latency_ms = opt.latency_ms
    encoded_opt.tcp = opt.tcp
    encoded_opt.source_fps = opt.source_fps
    if opt.codec == pyneat.RtspCodec.H264:
        encoded_opt.fallback_h264_width = opt.fallback_h264_width
        encoded_opt.fallback_h264_height = opt.fallback_h264_height
    source.add(pyneat.groups.rtsp_encoded_input(encoded_opt))
    return source


def encoded_format_tag(codec):
    return pyneat.Format.H265 if codec == pyneat.RtspCodec.H265 else pyneat.Format.H264


def encoded_decode_input_options(codec):
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = encoded_format_tag(codec)
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
    return opt


def encoded_video_input_options(codec):
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = encoded_format_tag(codec)
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    elif hasattr(opt, "use_simaai_pool"):
        opt.use_simaai_pool = False
    return opt


def build_decode_graph(input_name: str, opt) -> pyneat.Graph:
    decode = pyneat.Graph("decode")
    use_h265 = opt.codec == pyneat.RtspCodec.H265

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H265 if use_h265 else pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = opt.sima_allocator_type
    dec.out_format = pyneat.Format.NV12
    dec.decoder_name = opt.decoder_name
    dec.raw_output = opt.decoder_raw_output
    dec.next_element = opt.decoder_next_element
    dec.dec_width = opt.dec_width
    dec.dec_height = opt.dec_height
    dec.dec_fps = opt.source_fps
    dec.num_buffers = opt.num_buffers
    decode.connect(
        pyneat.nodes.input(input_name, encoded_decode_input_options(opt.codec)),
        pyneat.nodes.sima_decode(dec),
    )
    if opt.use_videoconvert:
        decode.add(pyneat.nodes.video_convert())
    if opt.use_videoscale:
        decode.add(pyneat.nodes.video_scale())
    if output_caps_enabled(opt.output_caps):
        decode.add(
            pyneat.nodes.caps_raw(
                "NV12",
                opt.output_caps.width,
                opt.output_caps.height,
                opt.output_caps.fps,
                opt.output_caps.memory,
            )
        )
    if opt.extra_fragment:
        decode.add(pyneat.nodes.custom(opt.extra_fragment))
    return decode


def build_video_sender_graph(input_name: str, codec, video_options) -> pyneat.Graph:
    video = pyneat.Graph("video_sender")
    video.connect(
        pyneat.nodes.input(input_name, encoded_video_input_options(codec)),
        pyneat.groups.video_sender(video_options),
    )
    return video


def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width = cfg.input_max_width
    opt.preprocess.input_max_height = cfg.input_max_height
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26Pose
    # YOLO26 pose ships single-class ("person") score heads. The packaged MPK still declares
    # the 80-class detector metadata, so state the real class count rather than inheriting it.
    opt.num_classes = 1
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_poses
    return pyneat.Model(cfg.model_path, opt)


def build_run_options() -> pyneat.RunOptions:
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def save_frames_enabled(cfg: AppConfig) -> bool:
    return bool(cfg.save_dir) and cfg.save_every > 0


def stream_id_for(stream_index: int) -> str:
    return f"stream{stream_index}"


def stream_index_from_sample(sample, stream_count: int) -> int:
    stream_id = str(getattr(sample, "stream_id", ""))
    prefix = "stream"
    if not stream_id.startswith(prefix):
        if stream_count == 1:
            return 0
        raise RuntimeError(f"pose sample missing stream id: {stream_id}")
    suffix = stream_id[len(prefix) :]
    if not suffix.isdigit():
        raise RuntimeError(f"invalid pose stream id: {stream_id}")
    index = int(suffix)
    if index < 0 or index >= stream_count:
        raise RuntimeError(f"pose stream id out of range: {stream_id}")
    return index


def realtime_link(
    stream_index: int,
    queue_depth: int,
    max_inflight_per_stream: int = -1,
    max_inflight_total: int = -1,
):
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.max_inflight_per_stream = max_inflight_per_stream
    link.max_inflight_total = max_inflight_total
    link.stream_id = stream_id_for(stream_index)
    return link


def build_estimator_graph(cfg: AppConfig):
    model = build_model(cfg)
    input_options = model.input_appsrc_options(False)
    input_options.block = True

    estimator = pyneat.Graph("estimator")
    estimator.connect(pyneat.nodes.input("estimator_frame", input_options), model)
    return model, estimator


def build_poses_graph() -> pyneat.Graph:
    poses = pyneat.Graph("poses")
    poses.add(pyneat.nodes.output("poses", pyneat.OutputOptions.every_frame(4)))
    return poses


def build_debug_frame_graph(stream_index: int) -> pyneat.Graph:
    frames = pyneat.Graph("debug_frame")
    frames.connect(
        pyneat.nodes.input("debug_frame"),
        pyneat.nodes.output(
            f"debug_frame_{stream_index}", pyneat.OutputOptions.every_frame(4)
        ),
    )
    return frames


def make_video_options(cfg: AppConfig, stream_index: int):
    video_options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg.codec))
    video_options.host = cfg.insight_host
    video_options.channel = stream_index
    video_options.video_port_base = cfg.video_port_base
    video_options.async_ = True
    return video_options


def build_stream_runtime(cfg: AppConfig, stream_index: int, url: str) -> StreamRuntime:
    frame_w, frame_h, fps = probe_rtsp(url, cfg.tcp)
    output_fps = cfg.fps if cfg.fps > 0 else fps

    source_options = build_source_options(cfg, url, fps, frame_w, frame_h)

    video_port = 0
    if cfg.video_enabled:
        video_port = make_video_options(cfg, stream_index).video_port

    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = stream_index
    metadata_options.metadata_port_base = cfg.metadata_port_base
    metadata_sender = pyneat.MetadataSender(metadata_options)

    print(
        f"[stream {stream_index}] rtsp={url} stream={frame_w}x{frame_h}@{output_fps} "
        f"insight={cfg.insight_host} "
        f"video={video_port if cfg.video_enabled else 'disabled'} "
        f"metadata={metadata_sender.metadata_port()}"
    )
    return StreamRuntime(
        index=stream_index,
        url=url,
        source_options=source_options,
        metadata_sender=metadata_sender,
        profile=ProfileWindow(cfg.profile, stream_index),
        latest_debug_frame=None,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=output_fps,
        video_port=video_port,
    )


def connect_stream_graph(
    app: AppRuntime, cfg: AppConfig, stream: StreamRuntime, estimator_graph
) -> None:
    source = build_encoded_source_graph(stream.source_options)
    decoder = build_decode_graph("decode_h264", stream.source_options)

    if cfg.video_enabled:
        encoded_branch = pyneat.graphs.branch("encoded", ["decode_h264", "video_h264"])
        app.graph.connect(source, encoded_branch)
        app.graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3))

        video_options = make_video_options(cfg, stream.index)
        app.graph.connect(
            encoded_branch,
            build_video_sender_graph(
                "video_h264", rtsp_codec(cfg.codec), video_options
            ),
            realtime_link(stream.index, 3),
        )
    else:
        app.graph.connect(source, decoder, realtime_link(stream.index, 3))

    save_debug_frames = save_frames_enabled(cfg)
    decoded_outputs = (
        ["estimator_frame", "debug_frame"] if save_debug_frames else ["estimator_frame"]
    )
    decoded_branch = pyneat.graphs.branch("decoded", decoded_outputs)
    app.graph.connect(decoder, decoded_branch)
    app.graph.connect(
        decoded_branch,
        estimator_graph,
        realtime_link(
            stream.index,
            4,
            cfg.max_inflight_per_stream,
            cfg.max_inflight_total,
        ),
    )
    if save_debug_frames:
        app.graph.connect(
            decoded_branch,
            build_debug_frame_graph(stream.index),
            realtime_link(stream.index, 4),
        )


def send_metadata(
    stream: StreamRuntime, cfg: AppConfig, sample, poses: list[dict]
) -> None:
    data = pose_metadata_data(poses, cfg.min_keypoint_visibility)
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    stream.metadata_sender.send_metadata(
        "pose-estimation",
        json.dumps(data, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    )


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor_from_sample(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field in sample.fields:
        tensor = first_tensor_from_sample(field)
        if tensor is not None:
            return tensor
    return None


def tensor_bgr_from_decoded(tensor):
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(
            tensor.contiguous().copy_payload_bytes(), dtype=np.uint8
        )
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    frame = np.asarray(tensor.to_numpy(copy=True))
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def draw_poses(frame, poses: list[dict], min_visibility: float) -> None:
    color = (0, 255, 0)
    height, width = frame.shape[0], frame.shape[1]

    def visible(point: dict) -> bool:
        return (
            point["visibility"] >= min_visibility
            and 0.0 <= point["x"] < width
            and 0.0 <= point["y"] < height
        )

    for pose in poses:
        x1 = max(0, round(pose["x1"]))
        y1 = max(0, round(pose["y1"]))
        x2 = min(width - 1, round(pose["x2"]))
        y2 = min(height - 1, round(pose["y2"]))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        keypoints = pose["keypoints"]
        for start, end in COCO_SKELETON:
            a, b = keypoints[start], keypoints[end]
            if visible(a) and visible(b):
                cv2.line(
                    frame,
                    (round(a["x"]), round(a["y"])),
                    (round(b["x"]), round(b["y"])),
                    color,
                    2,
                )
        for point in keypoints:
            if visible(point):
                cv2.circle(
                    frame,
                    (round(point["x"]), round(point["y"])),
                    3,
                    color,
                    -1,
                )


def maybe_save_debug_frame(
    cfg: AppConfig, stream: StreamRuntime, frame, poses: list[dict]
) -> None:
    if (
        not cfg.save_dir
        or cfg.save_every <= 0
        or stream.processed % cfg.save_every != 0
    ):
        return
    if frame is None:
        return

    frame = frame.copy()
    draw_poses(frame, poses, cfg.min_keypoint_visibility)
    out_path = (
        Path(cfg.save_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    )
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def all_streams_done(streams: list[StreamRuntime], frame_limit: int) -> bool:
    if frame_limit <= 0:
        return False
    return all(stream.processed >= frame_limit or stream.closed for stream in streams)


def process_output_sample(
    stream: StreamRuntime, cfg: AppConfig, sample, detection_pull_ms: float
) -> None:
    if cfg.frames > 0 and stream.processed >= cfg.frames:
        return

    poses = decode_poses(sample, stream.frame_w, stream.frame_h, cfg.max_poses)

    stream.processed += 1
    warming_up = stream.processed <= cfg.warmup_frames
    if not warming_up:
        metadata_start = time_ms()
        send_metadata(stream, cfg, sample, poses)
        metadata_end = time_ms()
        if save_frames_enabled(cfg):
            maybe_save_debug_frame(cfg, stream, stream.latest_debug_frame, poses)
        stream.profile.add(detection_pull_ms, metadata_end - metadata_start, len(poses))


def debug_frame_output_name(stream_index: int) -> str:
    return f"debug_frame_{stream_index}"


def drain_debug_frames(app: AppRuntime, cfg: AppConfig) -> None:
    if not save_frames_enabled(cfg):
        return
    for stream in app.streams:
        output_name = debug_frame_output_name(stream.index)
        while True:
            sample = app.run.pull(output_name, 0)
            if sample is None:
                break
            tensor = first_tensor_from_sample(sample)
            if tensor is not None:
                stream.latest_debug_frame = tensor_bgr_from_decoded(tensor)


def process_run_once(app: AppRuntime, cfg: AppConfig, output_name: str) -> bool:
    drain_debug_frames(app, cfg)
    pull_start = time_ms()
    sample = app.run.pull(output_name, 50)
    pull_end = time_ms()
    if sample is None:
        last_error_fn = getattr(app.run, "last_error", None)
        last_error = last_error_fn() if callable(last_error_fn) else ""
        if last_error:
            raise RuntimeError(f"runtime error: {last_error}")
        return False
    stream_index = stream_index_from_sample(sample, len(app.streams))
    process_output_sample(app.streams[stream_index], cfg, sample, pull_end - pull_start)
    drain_debug_frames(app, cfg)
    return True


def run_app(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
    if save_frames_enabled(cfg):
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # One model, shared by every stream: the estimator graph is built once here and each
    # stream's decoded branch links into it below.
    model, estimator_graph = build_estimator_graph(cfg)
    poses_graph = build_poses_graph()
    app = AppRuntime(graph=pyneat.Graph(), run=None, model=model, streams=[])
    for index, url in enumerate(cfg.rtsp_urls):
        stream = build_stream_runtime(cfg, index, url)
        app.streams.append(stream)
        connect_stream_graph(app, cfg, stream, estimator_graph)
    app.graph.connect(estimator_graph, poses_graph)

    try:
        if cfg.profile:
            print(f"Backend:\n{app.graph.describe_backend()}")
        app.run = app.graph.build(build_run_options())
        while not all_streams_done(app.streams, cfg.frames):
            process_run_once(app, cfg, "poses")
    finally:
        if app.run is not None:
            app.run.close()
        for stream in app.streams:
            stream.profile.flush()
            print(f"[stream {stream.index}] processed={stream.processed}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(
                f"Config validated: {args.config} (streams={len(cfg.rtsp_urls)}, "
                f"max_inflight_per_stream={cfg.max_inflight_per_stream}, "
                f"max_inflight_total={cfg.max_inflight_total})"
            )
            return 0

        load_runtime_dependencies()
        run_app(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports runtime failures.
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
