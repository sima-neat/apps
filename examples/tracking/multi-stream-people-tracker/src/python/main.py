"""Multi-camera RTSP YOLO26 Insight example using pyneat."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import glob
import json
import os
import struct
import sys
import time

import yaml

from utils.tracker import PeopleTracker, TrackedDetection

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
    frames: int = 0
    fps: int = 0
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    person_class_id: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    tracker_iou_threshold: float = 0.3
    tracker_max_missing: int = 15
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
    source_fps: int
    metadata_sender: object
    tracker: PeopleTracker
    profile: "ProfileWindow"
    latest_debug_frame: object | None
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int
    processed: int = 0


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
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.tracker_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(
        self, detection_pull_ms: float, tracker_ms: float, metadata_send_ms: float, track_count: int
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += track_count
        self.detection_pull_ms += detection_pull_ms
        self.tracker_ms += tracker_ms
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
            f"avg_tracker_ms={self.tracker_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_tracks={self.boxes / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.tracker_ms = 0.0
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
    parser = argparse.ArgumentParser(description="Multi-camera RTSP YOLO26 Insight example")
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
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if cfg.max_inflight_per_stream != -1 and cfg.max_inflight_per_stream <= 0:
        raise ValueError("inference.max_inflight_per_stream must be -1 or > 0")
    if cfg.max_inflight_total != -1 and cfg.max_inflight_total <= 0:
        raise ValueError("inference.max_inflight_total must be -1 or > 0")
    if cfg.person_class_id < 0:
        raise ValueError("inference.person_class_id must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if not 0.0 <= cfg.tracker_iou_threshold <= 1.0:
        raise ValueError("tracking.iou_threshold must be between 0 and 1")
    if cfg.tracker_max_missing < 0:
        raise ValueError("tracking.max_missing_frames must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    tracking = section(raw, "tracking")
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
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 16),
        person_class_id=int_or(inference, "person_class_id", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        tracker_iou_threshold=float_or(tracking, "iou_threshold", 0.3),
        tracker_max_missing=int_or(tracking, "max_missing_frames", 15),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


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


def filter_people(boxes: list[dict], person_class_id: int) -> list[dict]:
    return [box for box in boxes if int(box["class_id"]) == person_class_id]


def build_metadata_tracks(
    tracks: list[TrackedDetection], frame_w: int, frame_h: int
) -> list[dict]:
    metadata_tracks = []
    for track in tracks:
        x = max(0, int(track.x1))
        y = max(0, int(track.y1))
        w = max(0, int(track.x2 - track.x1))
        h = max(0, int(track.y2 - track.y1))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        metadata_tracks.append(
            {
                "id": str(track.track_id),
                "label": "person",
                "confidence": float(track.score),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_tracks


def rtsp_codec(codec: str):
    """Map the parsed `input.codec` config token onto the Core RTSP codec selector."""
    return pyneat.RtspCodec.H265 if codec == "h265" else pyneat.RtspCodec.H264


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
    if fps <= 0:
        raise RuntimeError("failed to probe RTSP frame rate")
    return width, height, fps


def source_options(cfg: AppConfig, stream: StreamRuntime):
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = stream.url
    opt.codec = rtsp_codec(cfg.codec)
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.source_fps = stream.source_fps
    if cfg.codec == "h264":
        opt.fallback_h264_width = stream.frame_w
        opt.fallback_h264_height = stream.frame_h
    return opt


def decoder_options(cfg: AppConfig, stream: StreamRuntime):
    opt = pyneat.SimaDecodeOptions()
    opt.type = pyneat.SimaDecodeType.H265 if cfg.codec == "h265" else pyneat.SimaDecodeType.H264
    opt.out_format = pyneat.Format.NV12
    opt.decoder_name = "decoder"
    opt.raw_output = True
    opt.dec_width = stream.frame_w
    opt.dec_height = stream.frame_h
    opt.dec_fps = stream.source_fps
    return opt


def build_model(cfg: AppConfig):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
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
        raise RuntimeError(f"detection sample missing stream id: {stream_id}")
    suffix = stream_id[len(prefix) :]
    if not suffix.isdigit():
        raise RuntimeError(f"invalid detection stream id: {stream_id}")
    index = int(suffix)
    if index < 0 or index >= stream_count:
        raise RuntimeError(f"detection stream id out of range: {stream_id}")
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


def latest_link():
    """Drops the oldest encoded frame instead of slowing the source when Insight lags."""
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    return link


def make_video_options(cfg: AppConfig, stream_index: int):
    video_options = pyneat.VideoSenderOptions.passthrough(rtsp_codec(cfg.codec))
    video_options.host = cfg.insight_host
    video_options.channel = stream_index
    video_options.video_port_base = cfg.video_port_base
    video_options.async_ = False
    return video_options


def build_stream_runtime(cfg: AppConfig, stream_index: int, url: str) -> StreamRuntime:
    frame_w, frame_h, fps = probe_rtsp(url)
    output_fps = cfg.fps if cfg.fps > 0 else fps

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
        source_fps=fps,
        metadata_sender=metadata_sender,
        tracker=PeopleTracker(cfg.tracker_iou_threshold, cfg.tracker_max_missing),
        profile=ProfileWindow(cfg.profile, stream_index),
        latest_debug_frame=None,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=output_fps,
        video_port=video_port,
    )


def send_metadata(stream: StreamRuntime, sample, tracks: list[TrackedDetection]) -> None:
    metadata_tracks = build_metadata_tracks(tracks, stream.frame_w, stream.frame_h)
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    stream.metadata_sender.send_metadata(
        "tracking",
        json.dumps({"tracks": metadata_tracks}, separators=(",", ":")),
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
        payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
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


def draw_tracks(frame, tracks: list[TrackedDetection], min_score: float) -> None:
    for track in tracks:
        score = float(track.score)
        if score < min_score:
            continue
        x1 = max(0, int(round(track.x1)))
        y1 = max(0, int(round(track.y1)))
        x2 = min(frame.shape[1] - 1, int(round(track.x2)))
        y2 = min(frame.shape[0] - 1, int(round(track.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"track id={track.track_id} score={score:.6f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )


def maybe_save_debug_frame(
    cfg: AppConfig, stream: StreamRuntime, frame, tracks: list[TrackedDetection]
) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or stream.processed % cfg.save_every != 0:
        return
    if frame is None:
        return

    frame = frame.copy()
    draw_tracks(frame, tracks, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def all_streams_done(streams: list[StreamRuntime], frame_limit: int) -> bool:
    if frame_limit <= 0:
        return False
    return all(stream.processed >= frame_limit for stream in streams)


def process_output_sample(stream: StreamRuntime, cfg: AppConfig, sample, detection_pull_ms: float) -> None:
    if cfg.frames > 0 and stream.processed >= cfg.frames:
        return

    payload = extract_bbox_payload(sample)
    boxes = parse_boxes_strict(payload, stream.frame_w, stream.frame_h, cfg.max_detections)
    people = filter_people(boxes, cfg.person_class_id)
    tracker_start = time_ms()
    tracks = stream.tracker.update(people, stream.processed)
    tracker_end = time_ms()

    stream.processed += 1
    warming_up = stream.processed <= cfg.warmup_frames
    if not warming_up:
        metadata_start = time_ms()
        send_metadata(stream, sample, tracks)
        metadata_end = time_ms()
        if save_frames_enabled(cfg):
            maybe_save_debug_frame(cfg, stream, stream.latest_debug_frame, tracks)
        stream.profile.add(
            detection_pull_ms,
            tracker_end - tracker_start,
            metadata_end - metadata_start,
            len(tracks),
        )


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


def pull_result_has_sample(run, sample, output_name: str) -> bool:
    if sample is not None:
        return True
    last_error_fn = getattr(run, "last_error", None)
    last_error = last_error_fn() if callable(last_error_fn) else ""
    running_fn = getattr(run, "running", None)
    running = running_fn() if callable(running_fn) else True
    if not running:
        message = f"{output_name} output closed unexpectedly"
        if last_error:
            message += f": {last_error}"
        raise RuntimeError(message)
    if last_error:
        raise RuntimeError(f"runtime error: {last_error}")
    return False


def process_run_once(app: AppRuntime, cfg: AppConfig, output_name: str) -> bool:
    drain_debug_frames(app, cfg)
    pull_start = time_ms()
    sample = app.run.pull(output_name, 50)
    pull_end = time_ms()
    if not pull_result_has_sample(app.run, sample, output_name):
        return False
    stream_index = stream_index_from_sample(sample, len(app.streams))
    process_output_sample(app.streams[stream_index], cfg, sample, pull_end - pull_start)
    drain_debug_frames(app, cfg)
    return True


def build_graph(cfg: AppConfig, streams: list[StreamRuntime], model):
    """Feed every stream into one shared detector; Core renders a single pipeline."""
    detector_input = model.input_appsrc_options(False)
    detector_input.block = True
    detector = pyneat.Graph("detector")
    detector.connect(pyneat.nodes.input("detector_frame", detector_input), model)

    graph = pyneat.Graph()
    for stream in streams:
        source = pyneat.Graph("rtsp_encoded_source")
        source.add(pyneat.groups.rtsp_encoded_input(source_options(cfg, stream)))

        decoder = pyneat.Graph("decode")
        decoder.add(pyneat.nodes.sima_decode(decoder_options(cfg, stream)))
        decoder.add(
            pyneat.nodes.caps_raw(
                "NV12", stream.frame_w, stream.frame_h, stream.source_fps, pyneat.CapsMemory.Any
            )
        )
        # Named so connect() can match the shared model's "detector_frame" input.
        decoder.add(pyneat.nodes.output("detector_frame"))

        # A plain source link lets Core tee the encoded stream to the decoder and the sender.
        graph.connect(source, decoder)
        graph.connect(
            decoder,
            detector,
            realtime_link(stream.index, 4, cfg.max_inflight_per_stream, cfg.max_inflight_total),
        )
        if cfg.video_enabled:
            video = pyneat.Graph("video_sender")
            video.add(pyneat.groups.video_sender(make_video_options(cfg, stream.index)))
            graph.connect(source, video, latest_link())
        if save_frames_enabled(cfg):
            frames = pyneat.Graph("debug_frame")
            frames.connect(
                pyneat.nodes.input("debug_frame"),
                pyneat.nodes.output(
                    debug_frame_output_name(stream.index), pyneat.OutputOptions.every_frame(4)
                ),
            )
            graph.connect(decoder, frames, realtime_link(stream.index, 4))

    detections = pyneat.Graph("detections")
    detections.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    graph.connect(detector, detections)
    return graph


def run_app(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
    if save_frames_enabled(cfg):
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(cfg)
    streams = [build_stream_runtime(cfg, i, url) for i, url in enumerate(cfg.rtsp_urls)]
    app = AppRuntime(graph=build_graph(cfg, streams, model), run=None, model=model, streams=streams)

    try:
        if cfg.profile:
            print(f"Backend:\n{app.graph.describe_backend()}")
        app.run = app.graph.build(build_run_options())
        while not all_streams_done(app.streams, cfg.frames):
            process_run_once(app, cfg, "detections")
    except KeyboardInterrupt:
        raise
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
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
