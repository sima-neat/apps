# Copyright 2026 SiMa Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-stream RTSP YOLO26 detector for Insight, with three topologies.

Every stream is detected at its own resolution and frame rate, so 720p, 1080p,
and 4K sources can run side by side. `mode` in the config picks the topology:

    live   One graph per stream. Edit streams.sources while the application
           runs and the change is applied on the next poll; the other streams
           keep running. Each stream re-encodes its video for Insight.

    scale  One graph for every stream, fanning into a single shared detector,
           with each source's H.264 passed through to Insight untouched.
           Highest stream count; changing the stream set needs a restart.

    group  The scale topology run as several independent processes, each
           owning `group.size` streams. A failure stops only its own group.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import glob
import json
import multiprocessing
import os
from pathlib import Path
import signal
import struct
import sys
import threading
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

MODES = ("live", "scale", "group")

cv2 = None
np = None
pyneat = None


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StreamSource:
    id: str
    url: str


@dataclass(frozen=True)
class AppConfig:
    mode: str
    model_path: str
    labels_path: Path
    sources: list[StreamSource]
    config_path: Path
    max_streams: int = 16
    group_size: int = 4
    source_type: str = "rtsp"
    source_codec: str = "h264"
    tcp: bool = True
    latency_ms: int = 100
    source_fps: int = 0
    frames: int = 0
    fps: int = 0
    min_score: float = 0.30
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    config_watch_seconds: float = 1.0
    insight_host: str = ""
    video_port: int = 9000
    metadata_port: int = 9100
    video_enabled: bool = True
    save_dir: str = ""
    save_every: int = 0
    # Insight channel of this process's first stream. Not a config key: group
    # mode sets it per worker so every group publishes on its own channel range.
    channel_base: int = 0


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
    return int(value)


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


def parse_mode(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in MODES:
        return lowered
    if not lowered:
        raise ValueError(f"mode must be set to one of: {', '.join(MODES)}")
    raise ValueError(f"mode must be one of: {', '.join(MODES)} (got '{value}')")


def parse_source_type(value: str) -> str:
    if value.lower() == "rtsp":
        return "rtsp"
    raise ValueError("source.type must be rtsp")


def parse_source_codec(value: str) -> str:
    # The scale and group topologies forward each source's H.264 to Insight
    # untouched, so H.264 is the only codec every mode can carry.
    if value.lower() in {"h264", "avc", "h.264"}:
        return "h264"
    raise ValueError("source.codec must be h264")


def parse_streams(raw: dict) -> tuple[list[StreamSource], int]:
    """streams: {max_streams: N, sources: [{id, url}, ...]}"""
    streams = section(raw, "streams")
    max_streams = int_or(streams, "max_streams", 16)
    items = streams.get("sources")
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ValueError("streams.sources must be a list")

    sources: list[StreamSource] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"streams.sources[{index}] must be a mapping with id and url")
        # The id is what live mode diffs a config edit by, so it cannot be
        # derived from the position: removing the first entry would otherwise
        # rename every stream after it and restart them all.
        stream_id = string_or(item, "id").strip()
        url = string_or(item, "url").strip()
        if not stream_id:
            raise ValueError(f"streams.sources[{index}].id must be set")
        if not url:
            raise ValueError(f"streams.sources[{index}].url must be set")
        sources.append(StreamSource(id=stream_id, url=url))
    return sources, max_streams


def validate_sources(sources: list[StreamSource], max_streams: int) -> None:
    if max_streams < 1:
        raise ValueError("streams.max_streams must be >= 1")
    if len(sources) > max_streams:
        raise ValueError(
            f"streams.sources has {len(sources)} entries, more than streams.max_streams "
            f"({max_streams})"
        )
    ids = [s.id for s in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("streams.sources ids must be unique")


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    # Path("") is ".", so an empty value in the file arrives here as ".".
    if str(cfg.labels_path) in ("", "."):
        raise ValueError("model.labels must be set")
    validate_sources(cfg.sources, cfg.max_streams)
    # Live mode may start empty and wait for a config edit to add streams; the
    # other modes build their graph from the startup list and need one.
    if cfg.mode != "live" and not cfg.sources:
        raise ValueError(f"streams.sources must not be empty in {cfg.mode} mode")
    if cfg.group_size < 1:
        raise ValueError("group.size must be >= 1")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError("source.fps must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.fps < 0:
        raise ValueError("inference.fps must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.config_watch_seconds <= 0:
        raise ValueError("runtime.config_watch_seconds must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")


def load_raw_config(config_path: Path) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    return raw


def load_app_config(config_path: Path) -> AppConfig:
    raw = load_raw_config(config_path)

    model = section(raw, "model")
    source = section(raw, "source")
    group = section(raw, "group")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    sources, max_streams = parse_streams(raw)

    cfg = AppConfig(
        mode=parse_mode(string_or(raw, "mode")),
        model_path=string_or(model, "path"),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        sources=sources,
        config_path=config_path,
        max_streams=max_streams,
        group_size=int_or(group, "size", 4),
        source_type=parse_source_type(string_or(source, "type", "rtsp")),
        source_codec=parse_source_codec(string_or(source, "codec", "h264")),
        tcp=bool_or(source, "tcp", True),
        latency_ms=int_or(source, "latency_ms", 100),
        source_fps=int_or(source, "fps", 0),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        min_score=float_or(inference, "min_score", 0.30),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        config_watch_seconds=float_or(runtime, "config_watch_seconds", 1.0),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        save_dir=string_or(output, "save_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


def reload_sources(config_path: Path, capacity: int) -> list[StreamSource]:
    """Parse a live config edit with the same checks as startup.

    `capacity` is the channel count fixed at startup from the ORIGINAL
    max_streams. Raising max_streams in a live edit cannot add channels, so the
    reload is checked against both. Raising here makes run_live ignore the whole
    edit instead of applying the part that fits.
    """
    sources, file_max = parse_streams(load_raw_config(config_path))
    validate_sources(sources, file_max)
    if len(sources) > capacity:
        raise ValueError(
            f"streams.sources has {len(sources)} entries, more than the {capacity} channels "
            f"this run started with; restart to raise streams.max_streams"
        )
    return sources


def split_groups(cfg: AppConfig) -> list[AppConfig]:
    """One scale config per group of `group_size` streams, in config order.

    Group g owns Insight channels g*size .. g*size+size-1, so its streams land on
    the same channels they would have in a single scale run.
    """
    groups = []
    for start in range(0, len(cfg.sources), cfg.group_size):
        groups.append(
            replace(cfg, sources=cfg.sources[start : start + cfg.group_size], channel_base=start)
        )
    return groups


# ── shared detection helpers ──────────────────────────────────────────────────


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


def enable_profile_env(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")


def save_frames_enabled(cfg: AppConfig) -> bool:
    return bool(cfg.save_dir) and cfg.save_every > 0


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
        for field_sample in sample.fields:
            try:
                return extract_bbox_payload(field_sample)
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


def metadata_objects(boxes: list[dict], labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
    """Compact object list: only the fields Insight's overlay reads.

    Insight drops metadata datagrams larger than the ~1500-byte MTU instead of
    reassembling them. Leaving out a per-box id, rounding confidence, and using
    integer pixels takes a box from ~95 to ~59 bytes, so ~25 detections fit one
    datagram instead of ~15.
    """
    objects = []
    for box in boxes:
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        class_id = int(box["class_id"])
        objects.append(
            {
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": round(float(box["score"]), 2),
                "bbox": [x, y, max(0, w), max(0, h)],
            }
        )
    return objects


def probe_rtsp(url: str) -> tuple[int, int, int]:
    """(width, height, fps) of an RTSP source; fps is 0 when it is not reported."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to probe RTSP frame size: {url}")
    return width, height, max(0, fps)


def make_model(cfg: AppConfig, input_max_w: int, input_max_h: int, name_suffix: str = ""):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # The preprocess envelope defaults to 1920x1080, so a larger frame (a 4K
    # source) fails the graph compile with "input width N exceeds configured
    # capacity". Size it to the frame this model actually receives.
    opt.preprocess.input_max_width = input_max_w
    opt.preprocess.input_max_height = input_max_h
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    if name_suffix and hasattr(opt, "name_suffix"):
        opt.name_suffix = name_suffix
    return pyneat.Model(cfg.model_path, opt)


class ProfileWindow:
    def __init__(self, enabled: bool, stream_label: str, interval: int = 100) -> None:
        self.enabled = enabled
        self.stream_label = stream_label
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
            f"[profile stream={self.stream_label}] frames={self.frames} "
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


def first_tensor_from_sample(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for field_sample in sample.fields:
        tensor = first_tensor_from_sample(field_sample)
        if tensor is not None:
            return tensor
    return None


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


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


def draw_boxes(frame, boxes: list[dict], min_score: float) -> None:
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        x1 = max(0, int(round(box["x1"])))
        y1 = max(0, int(round(box["y1"])))
        x2 = min(frame.shape[1] - 1, int(round(box["x2"])))
        y2 = min(frame.shape[0] - 1, int(round(box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={int(box['class_id'])} score={score:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


def save_debug_frame(cfg: AppConfig, channel: int, processed: int, frame, boxes: list[dict]) -> None:
    if frame is None or processed % cfg.save_every != 0:
        return
    draw_boxes(frame, boxes, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"stream_{channel}_frame_{processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


def rtp_timestamp_from_pts_ns(pts_ns: int) -> int:
    """PTS -> the 32-bit 90 kHz RTP timestamp Insight matches metadata on."""
    if pts_ns < 0:
        return 0
    return ((pts_ns * 90_000) // 1_000_000_000) & 0xFFFFFFFF


# ── live mode: one graph per stream ───────────────────────────────────────────

# Pull errors that a rebuild of the affected stream recovers from.
TRANSIENT_ACCELERATOR_ERRORS = (
    "Unable to load model",
    "misconfig.caps",
    "accelerator_execution_failed",
)

# Serializes MLA model loads across stream threads. Concurrent pipeline builds
# (initial builds and reconnects) otherwise race on MLA/RPMsg setup.
_BUILD_LOCK = threading.Lock()


@dataclass
class LiveStream:
    channel: int
    id: str
    url: str
    labels: list[str]
    width: int = 0
    height: int = 0
    output_fps: int = 0
    metadata_sender: object = None
    model: object = None
    graph: object = None
    run: object = None
    output_name: str = "detections"
    video_port: int = 0
    processed: int = 0
    profile: ProfileWindow = None


def make_live_source_options(cfg: AppConfig, stream: LiveStream):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = stream.url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = stream.width
    opt.fallback_h264_height = stream.height
    # Leave the decoder fps unspecified (-1) rather than pinning it: a fixed
    # dec-fps rejects streams whose real rate differs (high/variable-rate or
    # mis-probed sources) with a framerate mismatch. inference.fps still caps
    # how often detections are processed.
    opt.fallback_h264_fps = -1
    # No output caps: the stream is detected and delivered at its native size,
    # which keeps the zero-copy DMA path.
    opt.output_caps.enable = False
    return opt


def build_live_pipeline_once(cfg: AppConfig, stream: LiveStream) -> None:
    """RTSP -> decode -> {model, re-encoded video, optional debug frames}.

    Video and detections hang off the same decoded branch, so both carry the
    same restamped PTS and the metadata's rtp_timestamp matches the video
    Insight receives (measured: 100% of boxes rendered up to 6 streams).
    """
    save_frames = save_frames_enabled(cfg)
    model = make_model(cfg, stream.width, stream.height, f"_s{stream.channel}")

    model_name = "model"
    video_name = "video"
    debug_frame_name = "debug_frame"
    detections_name = "detections"
    debug_output_name = "debug_output"

    graph = pyneat.Graph()
    live_link = pyneat.GraphLinkOptions()
    live_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

    source = pyneat.groups.rtsp_decoded_input(make_live_source_options(cfg, stream))
    outputs = [model_name]
    if cfg.video_enabled:
        outputs.append(video_name)
    if save_frames:
        outputs.append(debug_frame_name)
    branch = pyneat.graphs.branch("source", outputs)
    graph.connect(source, branch)

    video_port = 0
    if cfg.video_enabled:
        video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
            stream.width, stream.height, stream.output_fps
        )
        video_options.encoder.bitrate_kbps = 1000
        video_options.host = cfg.insight_host
        video_options.channel = stream.channel
        video_options.video_port_base = cfg.video_port
        video_port = video_options.video_port
        video_graph = pyneat.Graph(video_name)
        video_graph.connect(pyneat.nodes.input(video_name), pyneat.groups.video_sender(video_options))
        graph.connect(branch, video_graph, live_link)

    model_graph = pyneat.Graph(model_name)
    model_graph.connect(pyneat.nodes.input(model_name), model)
    graph.connect(branch, model_graph, live_link)

    detections_graph = pyneat.Graph(detections_name)
    detections_graph.add(pyneat.nodes.output(detections_name, pyneat.OutputOptions.every_frame(4)))
    graph.connect(model_graph, detections_graph)

    if save_frames:
        frames = pyneat.Graph(debug_frame_name)
        frames.add(pyneat.nodes.output(debug_frame_name, pyneat.OutputOptions.every_frame(4)))
        debug_join = pyneat.graphs.combine(
            [debug_frame_name, detections_name], debug_output_name, pyneat.CombinePolicy.ByFrame
        )
        graph.connect(branch, frames, live_link)
        graph.connect(frames, debug_join)
        graph.connect(detections_graph, debug_join)

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy

    stream.model = model
    stream.graph = graph
    stream.run = graph.build(run_options)
    stream.output_name = debug_output_name if save_frames else detections_name
    stream.video_port = video_port


def build_live_pipeline(cfg: AppConfig, stream: LiveStream) -> None:
    """(Re)build one stream's graph. The metadata sender and probe persist.

    Model loads are serialized across streams and retried: concurrent rebuilds
    otherwise race on MLA/RPMsg setup ("Unable to load model").
    """
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with _BUILD_LOCK:
                # Tear down (run -> graph -> model) and rebuild under the same
                # lock, so two streams never tear down and load MLA pipelines
                # at the same time.
                rebuilding = stream.model is not None
                if stream.run is not None:
                    try:
                        stream.run.close()
                    except Exception:  # noqa: BLE001 - teardown is best-effort
                        pass
                stream.run = None
                stream.graph = None
                stream.model = None
                # The MLA pipeline tears down asynchronously; building the next
                # one before it is released can segfault when several streams
                # rebuild at once.
                if rebuilding:
                    time.sleep(0.8)
                build_live_pipeline_once(cfg, stream)
            return
        except Exception as exc:  # noqa: BLE001 - retry transient MLA build failures
            stream.run = None
            stream.graph = None
            stream.model = None
            if attempt >= attempts:
                raise
            print(
                f"[warn] stream {stream.id} pipeline build attempt {attempt} failed ({exc}); retrying",
                file=sys.stderr,
            )
            time.sleep(0.75)


def init_live_stream(cfg: AppConfig, channel: int, source: StreamSource, labels: list[str]) -> LiveStream:
    width, height, probed_fps = probe_rtsp(source.url)
    # Not every source reports its rate (high or variable-rate streams); the
    # live topology only needs one for the video encoder, so fall back to 30.
    source_fps = cfg.source_fps or probed_fps or 30
    stream = LiveStream(
        channel=channel,
        id=source.id,
        url=source.url,
        labels=labels,
        width=width,
        height=height,
        output_fps=cfg.fps if cfg.fps > 0 else source_fps,
        profile=ProfileWindow(cfg.profile, source.id),
    )
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = channel
    metadata_options.metadata_port_base = cfg.metadata_port
    stream.metadata_sender = pyneat.MetadataSender(metadata_options)

    build_live_pipeline(cfg, stream)
    print(
        f"[stream {source.id}] channel={channel} rtsp={source.url} "
        f"size={width}x{height}@{stream.output_fps} "
        f"video={stream.video_port if cfg.video_enabled else 'disabled'} "
        f"metadata={stream.metadata_sender.metadata_port()}",
        flush=True,
    )
    return stream


def send_live_metadata(stream: LiveStream, sample, boxes: list[dict], stream_count: int) -> None:
    """Publish detections keyed to the frame's RTP timestamp.

    Insight matches metadata to a held video frame by an exact 32-bit
    `rtp_timestamp`. send_metadata() cannot carry that field, so the full
    envelope is built here and sent with send_raw_json().
    """
    pts_ns = getattr(sample, "pts_ns", -1)
    pts_ns = int(pts_ns) if pts_ns is not None else -1
    frame_id = getattr(sample, "frame_id", -1)
    frame_id = int(frame_id) if frame_id is not None and frame_id >= 0 else -1

    def _attr(name: str) -> int:
        value = getattr(sample, name, -1)
        return int(value) if value is not None else -1

    payload = {
        "type": "object-detection",
        "data": {
            "objects": metadata_objects(boxes, stream.labels, stream.width, stream.height),
            "stream_count": int(stream_count),
            "stream_id": stream.id,
        },
        "timestamp": pts_ns // 1_000_000 if pts_ns >= 0 else int(time.time() * 1000),
        "frame_id": str(frame_id) if frame_id >= 0 else "",
        "stream_id": stream.id,
        "stream_index": stream.channel,
        "pts_ns": pts_ns,
        "dts_ns": _attr("dts_ns"),
        "duration_ns": _attr("duration_ns"),
        "input_seq": _attr("input_seq"),
    }
    if pts_ns >= 0:
        payload["rtp_timestamp"] = rtp_timestamp_from_pts_ns(pts_ns)
    try:
        stream.metadata_sender.send_raw_json(json.dumps(payload, separators=(",", ":")))
    except Exception:  # noqa: BLE001 - a dropped datagram must never stall detection
        pass


@dataclass
class ManagedStream:
    id: str
    url: str
    channel: int
    stop_event: threading.Event
    thread: object = None


class StreamManager:
    """Owns the live set of streams, channel allocation, and shared stop state.

    A stream failure is isolated to that stream: its worker retires its own
    slot and the rest keep running. stop_event means the whole application is
    stopping, never that one stream broke.
    """

    def __init__(self, cfg: AppConfig, labels: list[str]) -> None:
        self.cfg = cfg
        self.labels = labels
        self.lock = threading.Lock()
        self.streams: dict[str, ManagedStream] = {}
        self.free_channels = list(range(cfg.max_streams))
        self.stop_event = threading.Event()
        self.errors: list[Exception] = []
        # id -> reason for streams that died on their own. Kept after the slot
        # is released so the run can report them, and so all_failed() can tell
        # "every stream broke" apart from "the config lists no streams".
        self.failed: dict[str, str] = {}
        # Sources a reload asked for while no channel was free. A removed stream
        # holds its channel until its worker exits, so at capacity a
        # replace-one-stream edit can arrive before the channel is back.
        self.pending_adds: list[StreamSource] = []

    def active_count(self) -> int:
        with self.lock:
            return len(self.streams)

    def add(self, source: StreamSource) -> bool:
        with self.lock:
            if source.id in self.streams:
                return False
            if not self.free_channels:
                print(f"[warn] cannot add stream {source.id}: no free channel", file=sys.stderr)
                return False
            channel = self.free_channels.pop(0)
            managed = ManagedStream(source.id, source.url, channel, threading.Event())
            self.streams[source.id] = managed
            self.failed.pop(source.id, None)
            self.pending_adds = [p for p in self.pending_adds if p.id != source.id]
        managed.thread = threading.Thread(target=self._consume, args=(managed,), name=f"stream-{source.id}")
        managed.thread.start()
        return True

    def _release_channel(self, managed: ManagedStream) -> None:
        with self.lock:
            if managed.channel not in self.free_channels:
                self.free_channels.append(managed.channel)
                self.free_channels.sort()
        print(f"[stream {managed.id}] removed (channel {managed.channel} released)", flush=True)
        self._drain_pending_adds()

    def _drain_pending_adds(self) -> None:
        """Start queued sources that now fit; stop at the first that does not."""
        while True:
            with self.lock:
                if not self.pending_adds or not self.free_channels:
                    return
                source = self.pending_adds.pop(0)
            print(f"[stream {source.id}] retrying deferred add", flush=True)
            if not self.add(source):
                with self.lock:
                    self.pending_adds.insert(0, source)
                return

    def remove(self, stream_id: str, join_timeout: float = 15.0) -> None:
        with self.lock:
            managed = self.streams.pop(stream_id, None)
        if managed is None:
            return
        managed.stop_event.set()
        if managed.thread is not None:
            managed.thread.join(timeout=join_timeout)
        # A worker mid-build does not see stop_event until the build returns.
        # Releasing its channel early would let the next add bind the same
        # Insight ports while the old worker still runs, so a reaper releases it
        # once the worker is really gone.
        if managed.thread is not None and managed.thread.is_alive():
            print(
                f"[stream {stream_id}] still shutting down; holding channel "
                f"{managed.channel} until it exits",
                file=sys.stderr,
            )

            def reap() -> None:
                managed.thread.join()
                self._release_channel(managed)

            threading.Thread(target=reap, name=f"reap-{stream_id}", daemon=True).start()
            return
        self._release_channel(managed)

    def apply_sources(self, sources: list[StreamSource]) -> None:
        with self.lock:
            current = {sid: m.url for sid, m in self.streams.items()}
        wanted = {s.id: s for s in sources}
        # A changed URL under the same id is a replacement, not a no-op.
        to_remove = [sid for sid, url in current.items() if sid not in wanted or wanted[sid].url != url]
        to_add = [s for s in sources if s.id not in current or s.id in to_remove]
        for stream_id in to_remove:
            self.remove(stream_id)
        with self.lock:
            # A source dropped from the config stops counting as failed.
            for stream_id in [sid for sid in self.failed if sid not in wanted]:
                del self.failed[stream_id]
            self.pending_adds = [s for s in self.pending_adds if s.id in wanted]
        for source in to_add:
            if not self.add(source):
                with self.lock:
                    if all(p.id != source.id for p in self.pending_adds):
                        self.pending_adds.append(source)
        self._drain_pending_adds()

    def all_failed(self) -> bool:
        """True when every stream is gone and each one left because it failed."""
        with self.lock:
            return bool(self.failed) and not self.streams

    def all_done(self) -> bool:
        with self.lock:
            managed = list(self.streams.values())
        if not managed:
            return False
        return all(m.thread is not None and not m.thread.is_alive() for m in managed)

    def _retire(self, managed: ManagedStream, reason: str) -> None:
        """Release a failed stream's slot from inside its own worker.

        Cannot go through remove(), which joins the thread we are running on.
        The identity check keeps a late failure from evicting a same-id stream
        that has since been re-added.
        """
        with self.lock:
            if self.streams.get(managed.id) is managed:
                del self.streams[managed.id]
                self.free_channels.append(managed.channel)
                self.free_channels.sort()
            self.failed[managed.id] = reason
        print(
            f"[warn] stream {managed.id} failed ({reason}); channel {managed.channel} "
            f"released, other streams keep running",
            file=sys.stderr,
        )

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.lock:
            managed = list(self.streams.values())
        for m in managed:
            m.stop_event.set()
        for m in managed:
            if m.thread is not None:
                m.thread.join(timeout=15.0)

    def _stopping(self, managed: ManagedStream) -> bool:
        return managed.stop_event.is_set() or self.stop_event.is_set()

    def _reconnect(self, managed: ManagedStream, stream: LiveStream) -> bool:
        """Rebuild a stream after its source closed (RTSP drop, EOS, timeout).

        Returns True once reconnected, False when the stream should stop.
        """
        max_reconnects = 30
        for attempt in range(1, max_reconnects + 1):
            if self._stopping(managed):
                return False
            backoff_ms = min(500 * attempt, 5000)
            print(
                f"[warn] stream {managed.id} source closed; reconnecting "
                f"(attempt {attempt}/{max_reconnects} in {backoff_ms}ms)",
                file=sys.stderr,
            )
            time.sleep(backoff_ms / 1000.0)
            if self._stopping(managed):
                return False
            try:
                build_live_pipeline(self.cfg, stream)
                print(f"[stream {managed.id}] reconnected", flush=True)
                return True
            except Exception as exc:  # noqa: BLE001 - keep retrying with backoff
                print(f"[warn] stream {managed.id} reconnect attempt {attempt} failed ({exc})", file=sys.stderr)
        print(f"[warn] stream {managed.id} reconnect budget exhausted; stopping stream", file=sys.stderr)
        return False

    def _consume(self, managed: ManagedStream) -> None:
        cfg = self.cfg
        stream = None
        # Set when this stream ends because it broke, rather than because it
        # was asked to stop or finished its frame budget.
        failure = None
        # Desynchronize identical streams so a fleet-wide recovery rolls
        # through instead of every stream rebuilding at once.
        stagger_ms = float(managed.channel) * 350.0
        transient_recoveries = 0
        max_transient_recoveries = 10
        consecutive_transient = 0
        transient_retries = 8
        # inference.fps caps how often this stream processes and publishes a
        # detection. The runtime already keeps only the newest frame.
        min_interval_ms = 1000.0 / cfg.fps if cfg.fps > 0 else 0.0
        last_process_ms = -1e12
        try:
            stream = init_live_stream(cfg, managed.channel, StreamSource(managed.id, managed.url), self.labels)
            while not self._stopping(managed) and (cfg.frames <= 0 or stream.processed < cfg.frames):
                if min_interval_ms > 0.0:
                    now = time_ms()
                    due = last_process_ms + min_interval_ms
                    if now < due:
                        time.sleep(min(due - now, 15.0) / 1000.0)
                        continue
                pull_start = time_ms()
                try:
                    sample = stream.run.pull(stream.output_name, 50)
                    pull_error = ""
                except Exception as exc:  # noqa: BLE001 - pyneat.NeatError is not a RuntimeError
                    # pyneat raises a pull error instead of returning it.
                    sample = None
                    pull_error = str(exc)
                pull_end = time_ms()
                if sample is None:
                    last_error_fn = getattr(stream.run, "last_error", None)
                    last_error = pull_error or (last_error_fn() if callable(last_error_fn) else "")
                    stopping = self._stopping(managed)
                    if last_error:
                        # Accelerator errors under churn are transient: the MLA
                        # model load is deferred to run start, so a rebuild can
                        # surface here as a caps or load error, and closing
                        # another stream's run (a live edit) can fail this
                        # stream's accelerator stage. Rebuilding recovers it.
                        transient = any(
                            marker in last_error
                            for marker in TRANSIENT_ACCELERATOR_ERRORS
                        )
                        if transient and not stopping:
                            # Most blips clear on their own; rebuilding at once
                            # cascades into a rebuild storm.
                            consecutive_transient += 1
                            if consecutive_transient < transient_retries:
                                time.sleep(0.025)
                                continue
                            if transient_recoveries < max_transient_recoveries:
                                transient_recoveries += 1
                                consecutive_transient = 0
                                print(
                                    f"[warn] stream {managed.id} persistent MLA error; rebuilding "
                                    f"pipeline (recovery {transient_recoveries}/{max_transient_recoveries})",
                                    file=sys.stderr,
                                )
                                time.sleep(0.3 + stagger_ms / 1000.0)
                                build_live_pipeline(cfg, stream)
                                last_process_ms = -1e12
                                continue
                        # Closing the other streams' runs during shutdown can fail
                        # this one's accelerator stage; that is not a stream failure.
                        if stopping:
                            break
                        raise RuntimeError(f"stream {managed.id} runtime error: {last_error}")
                    running_fn = getattr(stream.run, "running", None)
                    if callable(running_fn) and not running_fn():
                        if stopping:
                            break
                        if not self._reconnect(managed, stream):
                            if not self._stopping(managed):
                                failure = "source closed and the reconnect budget ran out"
                            break
                        last_process_ms = -1e12
                        consecutive_transient = 0
                    continue

                boxes = parse_boxes_strict(
                    extract_bbox_payload(sample), stream.width, stream.height, cfg.max_detections
                )
                stream.processed += 1
                transient_recoveries = 0
                consecutive_transient = 0
                last_process_ms = time_ms()
                if stream.processed <= cfg.warmup_frames:
                    continue

                stream_count = self.active_count()
                metadata_start = time_ms()
                send_live_metadata(stream, sample, boxes, stream_count)
                metadata_end = time_ms()
                if save_frames_enabled(cfg) and stream.processed % cfg.save_every == 0:
                    tensor = first_tensor_from_sample(sample)
                    frame = tensor_bgr_from_decoded(tensor) if tensor is not None else None
                    save_debug_frame(cfg, stream.channel, stream.processed, frame, boxes)
                stream.profile.add(pull_end - pull_start, metadata_end - metadata_start, len(boxes))
        except Exception as exc:  # noqa: BLE001 - isolate the failure to this stream
            self.errors.append(exc)
            failure = str(exc) or exc.__class__.__name__
        finally:
            if stream is not None:
                stream.profile.flush()
                if stream.run is not None:
                    try:
                        stream.run.close()
                    except Exception:  # noqa: BLE001 - teardown is best-effort
                        pass
                print(f"[stream {managed.id}] processed={stream.processed}", flush=True)
            if failure is not None:
                self._retire(managed, failure)


def run_live(cfg: AppConfig) -> None:
    labels = load_labels(cfg.labels_path)
    manager = StreamManager(cfg, labels)
    try:
        last_mtime = cfg.config_path.stat().st_mtime
    except OSError:
        last_mtime = 0.0

    # SIGTERM must run the same shutdown as Ctrl+C: without a handler the
    # process dies before manager.shutdown() closes each run, and the decoder
    # and CVU pools it holds stay allocated.
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: manager.stop_event.set())
    try:
        for source in cfg.sources:
            manager.add(source)
        print(f"[app] live mode: {len(cfg.sources)} stream(s); watching {cfg.config_path}", flush=True)
        while not manager.stop_event.is_set():
            if manager.all_done():
                break
            # A finite run has nothing left to wait for once every stream has
            # failed. A continuous run keeps going: the config may sit at zero
            # streams until an edit adds one.
            if cfg.frames > 0 and manager.all_failed():
                break
            time.sleep(cfg.config_watch_seconds)
            try:
                mtime = cfg.config_path.stat().st_mtime
            except OSError:
                continue
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            try:
                sources = reload_sources(cfg.config_path, cfg.max_streams)
            except Exception as exc:  # noqa: BLE001 - a bad edit must not stop the run
                print(f"[warn] ignoring invalid config reload: {exc}", file=sys.stderr)
                continue
            print(f"[config] reload: {[s.id for s in sources]}", flush=True)
            manager.apply_sources(sources)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        manager.shutdown()

    for stream_id, reason in sorted(manager.failed.items()):
        print(f"[warn] stream {stream_id} did not run: {reason}", file=sys.stderr)
    # Individual stream failures are reported, not fatal. The run fails only
    # when no stream survived.
    if manager.errors and manager.all_failed():
        raise manager.errors[0]


# ── scale mode: one fused graph, one shared detector ──────────────────────────

# Largest upscale the hardware scaler performs. Past it the scaler emits
# nothing, with no error, so the shared detector geometry is clamped against it.
# Measured on Modalix: 2x works, 3x does not.
MAX_UPSCALE = 2
# The detector's own input size; clamping below it would discard detail the
# model uses anyway.
MODEL_INPUT = 640

# Decoder profile the scale and group pipelines were measured with at 16
# streams. Lower decoder_buffers only past ~16 streams, where the memory saving
# is worth the risk of stutter under network jitter.
DECODER_BUFFERS = 8
DECODER_INPUT_BUFFERS = 2
DECODER_TUNING = "auto"
# Declared decode rate cap for admission. A very high source rate times many
# streams exceeds the decoder core and the whole graph is rejected before it
# starts ("decoder processing capacity unavailable"). Above the cap the decode
# rate is left unpinned so admission passes and the decoder drops what it
# cannot process. Video to Insight is the untouched passthrough either way.
DECODER_FPS_CAP = 30


@dataclass(frozen=True)
class FusedProfile:
    """Shared-detector admission limits, as measured per mode."""

    max_inflight_per_stream: int
    max_inflight_total: int


SCALE_PROFILE = FusedProfile(max_inflight_per_stream=4, max_inflight_total=16)
# A group owns a few streams on one detector input port, so queue depth buys
# staleness rather than throughput.
GROUP_PROFILE = FusedProfile(max_inflight_per_stream=1, max_inflight_total=8)


@dataclass
class FusedStream:
    index: int
    channel: int
    id: str
    url: str
    metadata_sender: object
    labels: list[str]
    profile: ProfileWindow
    frame_w: int
    frame_h: int
    # Geometry the detector leg was normalized to. Detections come back in
    # this space and are scaled to frame_w/frame_h before publishing.
    det_w: int
    det_h: int
    source_fps: int
    video_port: int
    # Decoded frames awaiting their detection, keyed by frame id.
    debug_frames: dict = field(default_factory=dict)
    processed: int = 0
    closed: bool = False
    debug_pairing_warned: bool = False
    last_process_ms: float = -1e12


def detector_geometry(probes: list[tuple[int, int, int]]) -> tuple[int, int]:
    """One input geometry for the shared detector, from every source's size.

    A shared model has one input port, so sources of different sizes are scaled
    to one geometry in front of it. The largest source keeps the detail small
    objects need, clamped so the smallest source is never upscaled past
    MAX_UPSCALE, and never below MODEL_INPUT unless every source is smaller.
    """
    max_w = max(p[0] for p in probes)
    max_h = max(p[1] for p in probes)
    min_w = min(p[0] for p in probes)
    min_h = min(p[1] for p in probes)
    det_w = min(max_w, max(MODEL_INPUT, min(max_w, MAX_UPSCALE * min_w)))
    det_h = min(max_h, max(MODEL_INPUT, min(max_h, MAX_UPSCALE * min_h)))
    # NV12 needs even dimensions.
    return det_w - (det_w % 2), det_h - (det_h % 2)


def stream_link_id(stream_index: int) -> str:
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


def realtime_link(stream_index: int, queue_depth: int, max_inflight_per_stream: int = -1,
                  max_inflight_total: int = -1):
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.max_inflight_per_stream = max_inflight_per_stream
    link.max_inflight_total = max_inflight_total
    link.stream_id = stream_link_id(stream_index)
    return link


def encoded_input_options(memory_policy: str):
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = pyneat.Format.H264
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = getattr(pyneat.InputMemoryPolicy, memory_policy)
    elif memory_policy == "SystemMemory" and hasattr(opt, "use_simaai_pool"):
        opt.use_simaai_pool = False
    return opt


def build_fused_source_graph(cfg: AppConfig, stream: FusedStream):
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = stream.url
    opt.codec = pyneat.RtspCodec.H264
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    if hasattr(opt, "drop_on_latency"):
        opt.drop_on_latency = True
    opt.source_fps = stream.source_fps
    opt.fallback_h264_width = stream.frame_w
    opt.fallback_h264_height = stream.frame_h
    source = pyneat.Graph("rtsp_encoded_source")
    source.add(pyneat.groups.rtsp_encoded_input(opt))
    return source


def build_fused_decode_graph(stream: FusedStream):
    """Decode, then scale to the shared detector geometry when it differs."""
    # The decoded RTSP group's defaults are the decoder settings Core expects
    # in front of a model; reuse them instead of restating them here.
    defaults = pyneat.RtspDecodedInputOptions()
    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = defaults.sima_allocator_type
    dec.out_format = pyneat.Format.NV12
    dec.decoder_name = "decoder"
    dec.raw_output = True
    dec.next_element = defaults.decoder_next_element
    dec.dec_width = stream.frame_w
    dec.dec_height = stream.frame_h
    # Pinning a high source rate makes admission reject the graph, and pinning
    # a lower rate than the stream fails caps negotiation ("framerate
    # mismatch"). Above the cap the rate is left unspecified (-1).
    capped = stream.source_fps > DECODER_FPS_CAP
    dec.dec_fps = -1 if capped else stream.source_fps
    dec.num_buffers = DECODER_BUFFERS
    dec.input_buffers = DECODER_INPUT_BUFFERS
    dec.decoder_tuning = DECODER_TUNING
    dec.memory_opt = DECODER_TUNING in ("low-memory", "throughput-low-latency")

    decode = pyneat.Graph("decode")
    decode.connect(
        pyneat.nodes.input("decode_h264", encoded_input_options("Ev74")),
        pyneat.nodes.sima_decode(dec),
    )
    if (stream.det_w, stream.det_h) != (stream.frame_w, stream.frame_h):
        decode.add(pyneat.nodes.video_scale())
    decode.add(
        pyneat.nodes.caps_raw(
            "NV12", stream.det_w, stream.det_h, 0 if capped else stream.source_fps, pyneat.CapsMemory.Any
        )
    )
    return decode


def connect_fused_stream(graph, cfg: AppConfig, profile: FusedProfile, stream: FusedStream,
                         detector_graph) -> None:
    source = build_fused_source_graph(cfg, stream)
    decoder = build_fused_decode_graph(stream)

    if cfg.video_enabled:
        # The encoded access unit feeds both the decoder and Insight, so the
        # video reaches Insight without a re-encode. Hardware encoders are what
        # run out first at high stream counts.
        encoded_branch = pyneat.graphs.branch("encoded", ["decode_h264", "video_h264"])
        graph.connect(source, encoded_branch)
        graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3))

        video_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H264)
        video_options.host = cfg.insight_host
        video_options.channel = stream.channel
        video_options.video_port_base = cfg.video_port
        video_options.async_ = True
        video = pyneat.Graph("video_sender")
        video.connect(
            pyneat.nodes.input("video_h264", encoded_input_options("SystemMemory")),
            pyneat.groups.video_sender(video_options),
        )
        graph.connect(encoded_branch, video, realtime_link(stream.index, 3))
    else:
        graph.connect(source, decoder, realtime_link(stream.index, 3))

    save_frames = save_frames_enabled(cfg)
    decoded_branch = pyneat.graphs.branch(
        "decoded", ["detector_frame", "debug_frame"] if save_frames else ["detector_frame"]
    )
    graph.connect(decoder, decoded_branch)
    graph.connect(
        decoded_branch,
        detector_graph,
        realtime_link(stream.index, 4, profile.max_inflight_per_stream, profile.max_inflight_total),
    )
    if save_frames:
        frames = pyneat.Graph("debug_frame")
        frames.connect(
            pyneat.nodes.input("debug_frame"),
            pyneat.nodes.output(debug_frame_output_name(stream.index), pyneat.OutputOptions.every_frame(4)),
        )
        graph.connect(decoded_branch, frames, realtime_link(stream.index, 4))


def fused_run_options(stream_count: int):
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    # One ingress queue slot per stream, floor 4, as the 16-stream high-density
    # profile does.
    run_options.queue_depth = max(4, min(32, stream_count))
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def send_fused_metadata(stream: FusedStream, sample, boxes: list[dict]) -> None:
    # Arrival-order metadata, without rtp_timestamp. Exact matching needs the
    # passthrough payloader pinned to a known RTP base, which
    # VideoSenderOptions does not expose, so a key derived from PTS would match
    # nothing and Insight would render no boxes. Measured: 99% forwarded at
    # 16 streams x 25 fps.
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    objects = metadata_objects(boxes, stream.labels, stream.frame_w, stream.frame_h)
    stream.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": objects}, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    )


def debug_frame_output_name(stream_index: int) -> str:
    return f"debug_frame_{stream_index}"


# Decoded frames held per stream while their detection catches up. Inference is
# a few frames behind the decoder; past that the detection is never coming.
DEBUG_FRAME_CACHE = 16


def drain_debug_frames(run, cfg: AppConfig, streams: list[FusedStream]) -> None:
    """Collect decoded frames keyed by frame id.

    The decoded and detector branches run on independent queues, so the newest
    decoded frame is not the one the next detection describes.
    """
    if not save_frames_enabled(cfg):
        return
    for stream in streams:
        while True:
            sample = run.pull(debug_frame_output_name(stream.index), 0)
            if sample is None:
                break
            tensor = first_tensor_from_sample(sample)
            if tensor is None:
                continue
            frame_id = int(getattr(sample, "frame_id", -1) or -1)
            stream.debug_frames[frame_id] = tensor_bgr_from_decoded(tensor)
            while len(stream.debug_frames) > DEBUG_FRAME_CACHE:
                stream.debug_frames.pop(next(iter(stream.debug_frames)))


def take_debug_frame(stream: FusedStream, frame_id: int):
    """The decoded frame this detection came from, or None if nothing is held."""
    frame = stream.debug_frames.pop(frame_id, None)
    if frame is not None:
        for stale in [fid for fid in stream.debug_frames if fid < frame_id]:
            del stream.debug_frames[stale]
        return frame
    if not stream.debug_frames:
        return None
    if not stream.debug_pairing_warned:
        stream.debug_pairing_warned = True
        print(
            f"[warn] stream {stream.id} debug frames carry no id matching their detections; "
            f"saved images fall back to the newest frame",
            file=sys.stderr,
        )
    return stream.debug_frames.pop(next(reversed(stream.debug_frames)))


def process_fused_sample(stream: FusedStream, cfg: AppConfig, sample, detection_pull_ms: float) -> None:
    if cfg.frames > 0 and stream.processed >= cfg.frames:
        return
    # inference.fps caps how often this stream publishes; the shared detector
    # has already run on the sample, and the video stays at the source rate.
    now = time_ms()
    if cfg.fps > 0 and now < stream.last_process_ms + 1000.0 / cfg.fps:
        return
    stream.last_process_ms = now

    # Boxes come back in the shared detector geometry. Clamp there, then scale
    # to the stream's native size so they line up with the passthrough video.
    boxes = parse_boxes_strict(extract_bbox_payload(sample), stream.det_w, stream.det_h, cfg.max_detections)
    # Debug frames are tapped after the scaler, so they need the boxes in the
    # detector geometry.
    debug_boxes = [dict(b) for b in boxes] if save_frames_enabled(cfg) else boxes
    if (stream.det_w, stream.det_h) != (stream.frame_w, stream.frame_h):
        # The scale to the detector geometry is a stretch, not a letterbox: the
        # output caps pin width and height and leave the pixel aspect ratio
        # free, so a plain per-axis ratio maps the boxes back.
        sx = stream.frame_w / float(stream.det_w)
        sy = stream.frame_h / float(stream.det_h)
        for b in boxes:
            b["x1"] *= sx
            b["x2"] *= sx
            b["y1"] *= sy
            b["y2"] *= sy

    stream.processed += 1
    if stream.processed <= cfg.warmup_frames:
        return
    metadata_start = time_ms()
    send_fused_metadata(stream, sample, boxes)
    metadata_end = time_ms()
    if save_frames_enabled(cfg) and stream.processed % cfg.save_every == 0:
        frame = take_debug_frame(stream, int(getattr(sample, "frame_id", -1) or -1))
        save_debug_frame(cfg, stream.channel, stream.processed, frame, debug_boxes)
    stream.profile.add(detection_pull_ms, metadata_end - metadata_start, len(boxes))


def process_fused_run_once(run, cfg: AppConfig, streams: list[FusedStream]) -> bool:
    """Pull and handle one detection. Returns False once the shared output has closed."""
    drain_debug_frames(run, cfg, streams)
    pull_start = time_ms()
    sample = run.pull("detections", 50)
    pull_end = time_ms()
    if sample is None:
        last_error_fn = getattr(run, "last_error", None)
        last_error = last_error_fn() if callable(last_error_fn) else ""
        if last_error:
            raise RuntimeError(f"runtime error: {last_error}")
        running_fn = getattr(run, "running", None)
        if callable(running_fn) and not running_fn():
            # The shared graph is gone; no later pull can succeed.
            print("[warn] detection output closed; ending run", file=sys.stderr)
            for stream in streams:
                stream.closed = True
            return False
        return True
    stream = streams[stream_index_from_sample(sample, len(streams))]
    process_fused_sample(stream, cfg, sample, pull_end - pull_start)
    return True


def all_streams_done(streams: list[FusedStream], frame_limit: int) -> bool:
    if not streams:
        return True
    return all(stream.closed or (frame_limit > 0 and stream.processed >= frame_limit) for stream in streams)


def run_scale(cfg: AppConfig, profile: FusedProfile = SCALE_PROFILE, ready=None) -> None:
    """Run every stream of `cfg` through one graph and one shared detector.

    `ready`, when given, is set once the graph is running; group mode uses it
    to start its workers one at a time.
    """
    labels = load_labels(cfg.labels_path)
    # One at a time: OpenCV's GStreamer capture crashes the process when
    # several failing probes run at once, instead of reporting the bad source.
    probes = [probe_rtsp(source.url) for source in cfg.sources]
    for source, (_, _, fps) in zip(cfg.sources, probes):
        if cfg.source_fps <= 0 and fps <= 0:
            raise RuntimeError(f"stream {source.id} did not report a frame rate; set source.fps")

    det_w, det_h = detector_geometry(probes)
    # Raising the clamp to MODEL_INPUT can re-cross the upscale limit (320x180
    # beside 1080p), and no geometry satisfies both. Name the stream that will
    # deliver video but no detections rather than let it go quiet.
    worst = max(max(det_w / p[0], det_h / p[1]) for p in probes)
    if worst > MAX_UPSCALE:
        smallest = min(probes, key=lambda p: p[0] * p[1])
        print(
            f"[warn] detector geometry {det_w}x{det_h} needs a {worst:.1f}x upscale from the "
            f"smallest source ({smallest[0]}x{smallest[1]}), past the {MAX_UPSCALE}x the scaler "
            f"performs - that stream will deliver video but no detections. Use sources within "
            f"{MAX_UPSCALE}x of each other.",
            file=sys.stderr,
        )
    if len({(p[0], p[1]) for p in probes}) > 1:
        print(
            f"[detector] mixed source sizes {sorted({(p[0], p[1]) for p in probes})} -> detector "
            f"input {det_w}x{det_h} (video stays native per stream)",
            flush=True,
        )

    model = make_model(cfg, det_w, det_h)
    detector_input = model.input_appsrc_options(False)
    detector_input.block = True
    detector_graph = pyneat.Graph("detector")
    detector_graph.connect(pyneat.nodes.input("detector_frame", detector_input), model)
    detections_graph = pyneat.Graph("detections")
    detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

    graph = pyneat.Graph()
    streams: list[FusedStream] = []
    for index, (source, (width, height, fps)) in enumerate(zip(cfg.sources, probes)):
        channel = cfg.channel_base + index
        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = channel
        metadata_options.metadata_port_base = cfg.metadata_port
        stream = FusedStream(
            index=index,
            channel=channel,
            id=source.id,
            url=source.url,
            metadata_sender=pyneat.MetadataSender(metadata_options),
            labels=labels,
            profile=ProfileWindow(cfg.profile, source.id),
            frame_w=width,
            frame_h=height,
            det_w=det_w,
            det_h=det_h,
            source_fps=cfg.source_fps or fps,
            video_port=cfg.video_port + channel if cfg.video_enabled else 0,
        )
        streams.append(stream)
        connect_fused_stream(graph, cfg, profile, stream, detector_graph)
        print(
            f"[stream {source.id}] channel={channel} rtsp={source.url} "
            f"size={width}x{height}@{stream.source_fps} "
            f"video={stream.video_port if cfg.video_enabled else 'disabled'} "
            f"metadata={stream.metadata_sender.metadata_port()}",
            flush=True,
        )
    graph.connect(detector_graph, detections_graph)

    stop_requested = threading.Event()
    output_closed = False
    # SIGTERM must run the same shutdown as Ctrl+C, or run.close() never
    # releases the decoder and CVU pools.
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    run = None
    try:
        if cfg.profile:
            print(f"Backend:\n{graph.describe_backend()}")
        run = graph.build(fused_run_options(len(streams)))
        print(f"[app] graph running: {len(streams)} stream(s)", flush=True)
        if ready is not None:
            ready.set()
        while not stop_requested.is_set() and not all_streams_done(streams, cfg.frames):
            if not process_fused_run_once(run, cfg, streams):
                output_closed = True
        # A continuous run has no natural end, so a closed output is always a
        # failure; a finite run fails only if it closed short of its budget.
        if output_closed and not stop_requested.is_set():
            if cfg.frames <= 0:
                raise RuntimeError("detection output closed unexpectedly; the run produced no further metadata")
            short = [s for s in streams if s.processed < cfg.frames]
            if short:
                detail = ", ".join(f"stream {s.id} got {s.processed}/{cfg.frames}" for s in short)
                raise RuntimeError(f"detection output closed before every stream reached inference.frames ({detail})")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if run is not None:
            run.close()
        for stream in streams:
            stream.profile.flush()
            print(f"[stream {stream.id}] processed={stream.processed}", flush=True)


# ── group mode: several independent scale processes ───────────────────────────


def _group_worker(cfg: AppConfig, group_index: int, ready) -> None:
    """Entry point of one group's process: a scale run over its own streams."""
    try:
        load_runtime_dependencies()
        run_scale(cfg, GROUP_PROFILE, ready)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - report and exit nonzero for this group only
        print(f"[ERR] group {group_index}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


def group_streams(cfg: AppConfig) -> tuple:
    """What a group runs; a group restarts only when this changes."""
    return tuple((s.id, s.url) for s in cfg.sources)


def changed_groups(running: dict[int, AppConfig], wanted: list[AppConfig]) -> list[int]:
    """Indexes of groups to stop, start, or restart to go from `running` to `wanted`."""
    indexes = set(running) | set(range(len(wanted)))
    return sorted(
        index
        for index in indexes
        if index not in running
        or index >= len(wanted)
        or group_streams(running[index]) != group_streams(wanted[index])
    )


@dataclass
class GroupWorker:
    index: int
    cfg: AppConfig
    process: object
    # Stopped by the supervisor (a config edit or shutdown), so its exit is not
    # a group failure.
    retired: bool = False


class GroupSupervisor:
    """Runs one scale process per group and applies config edits group by group.

    A group whose streams change is restarted alone; the other groups keep
    running. Groups start one at a time because concurrent model loads race on
    the MLA, and a group that fails to start does not block the next one.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        # Spawn, not fork: each group loads its own runtime from a clean process.
        self.context = multiprocessing.get_context("spawn")
        self.stop_requested = threading.Event()
        self.workers: dict[int, GroupWorker] = {}
        # Groups that exited non-zero on their own, by index: the exit code.
        self.failed: dict[int, int] = {}

    def start(self, index: int, group_cfg: AppConfig) -> None:
        ready = self.context.Event()
        process = self.context.Process(
            target=_group_worker, args=(group_cfg, index, ready), name=f"group-{index}"
        )
        process.start()
        self.workers[index] = GroupWorker(index, group_cfg, process)
        self.failed.pop(index, None)
        last_channel = group_cfg.channel_base + len(group_cfg.sources) - 1
        print(
            f"[group {index}] pid={process.pid} "
            f"streams={','.join(s.id for s in group_cfg.sources)} "
            f"channels={group_cfg.channel_base}-{last_channel}",
            flush=True,
        )
        while not ready.wait(0.5):
            if not process.is_alive() or self.stop_requested.is_set():
                break

    def stop(self, index: int) -> None:
        worker = self.workers.pop(index)
        worker.retired = True
        process = worker.process
        # SIGTERM runs the group's normal shutdown, which releases its pools.
        if process.is_alive():
            process.terminate()
        process.join(timeout=20.0)
        if process.is_alive():
            print(f"[warn] group {index} did not stop after SIGTERM; killing it", file=sys.stderr)
            process.kill()
            process.join()

    def record_exits(self) -> None:
        for index, worker in self.workers.items():
            code = worker.process.exitcode
            if code not in (None, 0) and index not in self.failed:
                self.failed[index] = code
                print(f"[warn] group {index} exited with code {code}; other groups keep running",
                      file=sys.stderr)

    def any_running(self) -> bool:
        return any(w.process.is_alive() for w in self.workers.values())

    def apply(self, groups: list[AppConfig]) -> None:
        running = {index: worker.cfg for index, worker in self.workers.items()}
        changed = changed_groups(running, groups)
        for index in changed:
            if index in self.workers:
                verb = "streams changed; restarting" if index < len(groups) else "removed"
                print(f"[group {index}] {verb}", flush=True)
                self.stop(index)
        for index in changed:
            if index < len(groups) and not self.stop_requested.is_set():
                self.start(index, groups[index])

    def shutdown(self) -> None:
        for index in list(self.workers):
            self.stop(index)


def run_group(cfg: AppConfig) -> None:
    groups = split_groups(cfg)
    print(
        f"[app] group mode: {len(cfg.sources)} stream(s) in {len(groups)} group(s) of up to "
        f"{cfg.group_size}; watching {cfg.config_path}",
        flush=True,
    )
    supervisor = GroupSupervisor(cfg)
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: supervisor.stop_requested.set())
    try:
        last_mtime = cfg.config_path.stat().st_mtime
    except OSError:
        last_mtime = 0.0
    try:
        supervisor.apply(groups)
        while not supervisor.stop_requested.is_set():
            supervisor.record_exits()
            # A finite run ends once every group has finished. A continuous run
            # keeps going: a config edit can still restart a failed group.
            if cfg.frames > 0 and not supervisor.any_running():
                break
            time.sleep(cfg.config_watch_seconds)
            try:
                mtime = cfg.config_path.stat().st_mtime
            except OSError:
                continue
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            try:
                sources = reload_sources(cfg.config_path, cfg.max_streams)
            except Exception as exc:  # noqa: BLE001 - a bad edit must not stop the run
                print(f"[warn] ignoring invalid config reload: {exc}", file=sys.stderr)
                continue
            print(f"[config] reload: {[s.id for s in sources]}", flush=True)
            supervisor.apply(split_groups(replace(cfg, sources=sources)))
    except KeyboardInterrupt:
        supervisor.stop_requested.set()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        supervisor.record_exits()
        stopped = supervisor.stop_requested.is_set()
        supervisor.shutdown()

    if stopped:
        return
    if supervisor.failed:
        detail = ", ".join(f"{i} (exit {code})" for i, code in sorted(supervisor.failed.items()))
        raise RuntimeError(f"group(s) {detail} failed; see the [ERR] lines above")


# ── entry point ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-stream RTSP YOLO26 detector for Insight. Set `mode` (live, scale, "
        "or group) in the config file."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config} (mode={cfg.mode}, streams={len(cfg.sources)})")
            return 0
        enable_profile_env(cfg)
        if save_frames_enabled(cfg):
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        if cfg.mode == "group":
            # Each group process loads the runtime itself.
            run_group(cfg)
            return 0
        load_runtime_dependencies()
        if cfg.mode == "scale":
            run_scale(cfg)
        else:
            run_live(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level error boundary
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
