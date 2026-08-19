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

"""Adaptive-resolution multi-stream YOLO26 detector for Insight (pyneat).

Runs YOLO26 across N RTSP streams where the stream count and each stream's input
resolution adapt at runtime. Streams are added/removed live by editing
streams.sources in the config; each stream picks a resolution tier from scene
content (object size, density, confidence) under a shared compute budget, and
publishes H.264 video + detection metadata (with active tier + stream count) to
Insight per stream.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import glob
import json
import os
import struct
import sys
import threading
import time

import yaml

from adaptive_policy import (
    OutputPolicyConfig,
    PolicyConfig,
    PolicyState,
    effective_tier,
    frame_stats,
    output_candidates,
    select_output_index,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

cv2 = None
np = None
pyneat = None

# Serializes MLA model loads across stream threads. Concurrent pipeline builds
# (initial or tier-switch rebuilds) otherwise race on MLA/RPMsg setup.
_BUILD_LOCK = threading.Lock()


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass
class StreamSource:
    id: str
    rtsp_url: str


@dataclass
class AppConfig:
    model_path: str
    tier_paths: dict[int, str]
    labels_path: Path
    policy: PolicyConfig
    budget_units: float
    output_policy: OutputPolicyConfig
    max_streams: int
    sources: list[StreamSource]
    config_path: Path
    latency_ms: int = 100
    tcp: bool = True
    frames: int = 0
    fps: int = 0
    min_score: float = 0.30
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    warmup_frames: int = 30
    config_watch_seconds: float = 1.0
    insight_host: str = "127.0.0.1"
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    # Send the source H.264 to Insight untouched instead of re-encoding it.
    # Removes one hardware encoder per stream - the allocation that limits
    # stream count. Costs: Insight receives the source bitrate as-is.
    encoded_passthrough: bool = True
    decoder_buffers: int = 4        # decoder output pool frames (untuned default: 14)
    decoder_input_buffers: int = 2  # decoder input pool frames (untuned default: 8)
    decoder_tuning: str = "throughput-low-latency"
    metadata_rtp_timestamp: str = "auto"   # "auto" | "on" | "off"
    save_dir: str = ""
    save_every: int = 0

    @property
    def resolutions(self) -> list[int]:
        return self.policy.resolutions

    def tier_model_path(self, size: int) -> str:
        """Archive for a tier: the tier-specific one if it exists, else model.path."""
        path = self.tier_paths.get(size)
        if path and Path(path).is_file():
            return path
        return self.model_path


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


def parse_sources_from_raw(raw: dict) -> tuple[list[StreamSource], int]:
    """Parse streams from config, accepting both the rich and bare-list forms.

    Rich (checked-in):  streams: {max_streams: N, sources: [{id, rtsp_url}, ...]}
    Bare  (harness):    streams: [rtsp_url, rtsp_url, ...]
    """
    streams = raw.get("streams")
    max_streams = 8
    sources: list[StreamSource] = []

    if isinstance(streams, list):
        items = streams
    elif isinstance(streams, dict):
        max_streams = int_or(streams, "max_streams", 8)
        items = streams.get("sources")
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError("streams.sources must be a list")
    else:
        raise ValueError("streams must be a list or a mapping with sources")

    for index, item in enumerate(items):
        if isinstance(item, str):
            url = item.strip()
            source = StreamSource(id=f"cam-{index + 1}", rtsp_url=url)
        elif isinstance(item, dict):
            url = string_or(item, "rtsp_url").strip()
            source = StreamSource(id=string_or(item, "id") or f"cam-{index + 1}", rtsp_url=url)
        else:
            raise ValueError(f"streams[{index}] must be a string or a mapping")
        if not source.rtsp_url:
            raise ValueError(f"streams[{index}] has an empty rtsp_url")
        sources.append(source)

    return sources, max_streams


def build_policy(adaptive: dict) -> tuple[PolicyConfig, float]:
    resolutions = adaptive.get("resolutions") or [320, 640, 960]
    if not isinstance(resolutions, list) or not resolutions:
        raise ValueError("adaptive.resolutions must be a non-empty list")
    resolutions = [int(r) for r in resolutions]
    if resolutions != sorted(resolutions):
        raise ValueError("adaptive.resolutions must be ascending")
    policy = PolicyConfig(
        resolutions=resolutions,
        confidence_low=float_or(adaptive, "confidence_low", 0.40),
        min_object_px=float_or(adaptive, "min_object_px", 24.0),
        hysteresis_frames=int_or(adaptive, "hysteresis_frames", 15),
        density_high=int_or(adaptive, "density_high", 20),
    )
    budget_units = float_or(adaptive, "budget_units", 12.0)
    return policy, budget_units


def build_output_policy(output: dict) -> OutputPolicyConfig:
    """Parse the output.adaptive block: delivered-video bandwidth policy.

    Absent block => defaults (heights [2160,1080,720,480], budget 280 MP/s), so
    existing configs keep working (a single stream just delivers its native size).
    """
    adaptive = section(output, "adaptive")
    defaults = OutputPolicyConfig()
    heights = adaptive.get("heights") or defaults.heights
    if not isinstance(heights, list) or not heights:
        raise ValueError("output.adaptive.heights must be a non-empty list")
    heights = [int(h) for h in heights]
    if any(h <= 0 for h in heights):
        raise ValueError("output.adaptive.heights must be positive")
    budget = float_or(adaptive, "budget_megapixels_per_s", defaults.budget_megapixels_per_s)
    return OutputPolicyConfig(heights=sorted(set(heights), reverse=True), budget_megapixels_per_s=budget)


def resolve_tier_paths(model: dict, resolutions: list[int]) -> dict[int, str]:
    tiers = model.get("tiers") or {}
    if not isinstance(tiers, dict):
        raise ValueError("model.tiers must be a mapping of size -> path")
    resolved: dict[int, str] = {}
    for size in resolutions:
        for key in (size, str(size)):
            if key in tiers and tiers[key]:
                resolved[size] = str(tiers[key])
                break
    return resolved


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path and not cfg.tier_paths:
        raise ValueError("model.path or model.tiers must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.sources:
        raise ValueError("streams must be a non-empty list")
    if cfg.max_streams < 1:
        raise ValueError("streams.max_streams must be >= 1")
    if len(cfg.sources) > cfg.max_streams:
        raise ValueError("streams count exceeds streams.max_streams")
    ids = [s.id for s in cfg.sources]
    if len(set(ids)) != len(ids):
        raise ValueError("streams.sources ids must be unique")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("input.latency_ms must be >= 0")
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
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if cfg.budget_units <= 0:
        raise ValueError("adaptive.budget_units must be > 0")
    if not cfg.output_policy.heights:
        raise ValueError("output.adaptive.heights must be a non-empty list")
    if cfg.output_policy.budget_megapixels_per_s <= 0:
        raise ValueError("output.adaptive.budget_megapixels_per_s must be > 0")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    adaptive = section(raw, "adaptive")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    policy, budget_units = build_policy(adaptive)
    output_policy = build_output_policy(output)
    sources, max_streams = parse_sources_from_raw(raw)

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        tier_paths=resolve_tier_paths(model, policy.resolutions),
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        policy=policy,
        budget_units=budget_units,
        output_policy=output_policy,
        max_streams=max_streams,
        sources=sources,
        config_path=config_path,
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        min_score=float_or(inference, "min_score", 0.30),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        warmup_frames=int_or(runtime, "warmup_frames", 30),
        config_watch_seconds=float_or(runtime, "config_watch_seconds", 1.0),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", int_or(insight, "video_port", 9000)),
        metadata_port_base=int_or(
            insight, "metadata_port_base", int_or(insight, "metadata_port", 9100)
        ),
        video_enabled=bool_or(output, "video_enabled", True),
        encoded_passthrough=bool_or(output, "encoded_passthrough", True),
        decoder_buffers=int_or(input_cfg, "decoder_buffers", 4),
        decoder_input_buffers=int_or(input_cfg, "decoder_input_buffers", 2),
        decoder_tuning=string_or(input_cfg, "decoder_tuning", "throughput-low-latency"),
        metadata_rtp_timestamp=string_or(output, "metadata_rtp_timestamp", "auto"),
        save_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
    )
    validate_config(cfg)
    return cfg


def reload_sources(config_path: Path) -> list[StreamSource]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    sources, _ = parse_sources_from_raw(raw)
    return sources


# ── detections / metadata ─────────────────────────────────────────────────────


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


def build_metadata_boxes(boxes: list[dict], labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
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


def metadata_payload(
    boxes: list[dict], labels: list[str], frame_w: int, frame_h: int, active_tier: int,
    stream_count: int, stream_id: str,
) -> str:
    """Compact object-detection metadata plus the active tier and stream count.

    Only the fields Insight's overlay reads (label, confidence, bbox); the id is
    dropped, confidence is rounded to 2 decimals, and bboxes are integer pixels, so
    ~2x more detections fit one UDP datagram (Insight drops fragmented metadata
    packets larger than the ~1500-byte MTU — it does not reassemble them).
    """
    objects = [
        {
            "label": b["label"],
            "confidence": round(float(b["confidence"]), 2),
            "bbox": [int(v) for v in b["bbox"]],
        }
        for b in build_metadata_boxes(boxes, labels, frame_w, frame_h)
    ]
    return json.dumps(
        {
            "objects": objects,
            "active_tier": int(active_tier),
            "stream_count": int(stream_count),
            "stream_id": stream_id,
        },
        separators=(",", ":"),
    )


# ── pipeline ──────────────────────────────────────────────────────────────────


class ProfileWindow:
    def __init__(self, enabled: bool, stream_id: str, interval: int = 100) -> None:
        self.enabled = enabled
        self.stream_id = stream_id
        self.interval = interval
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0

    def add(self, detection_pull_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.detection_pull_ms += detection_pull_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile stream={self.stream_id}] frames={self.frames} output_fps={output_fps:.2f} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames:.2f} "
            f"avg_boxes={self.boxes / self.frames:.2f}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0


@dataclass
class StreamRuntime:
    channel: int
    id: str
    url: str
    labels: list[str]
    # frame_w/frame_h are the CURRENT delivered (output) resolution: detection runs
    # on the post-videoscale frame, so boxes, metadata, and video all share these.
    frame_w: int = 0
    frame_h: int = 0
    # native_w/native_h are the source's decoded size (parser fallback caps + the
    # ceiling that output candidates are clamped to; the source is never upscaled).
    native_w: int = 0
    native_h: int = 0
    output_fps: int = 0
    # False when the lowered graph restamps PTS: an rtp_timestamp that cannot
    # match makes Insight discard 100% of metadata silently, whereas omitting it
    # lets the browser fall back to arrival order and boxes still render.
    emit_rtp_timestamp: bool = True
    metadata_sender: object = None
    model: object = None
    graph: object = None
    run: object = None
    output_name: str = "detections"
    video_port: int = 0
    tier_index: int = 0
    # Delivered-resolution candidates (highest-area first) and the active index,
    # chosen from the shared output-bandwidth budget and current stream count.
    candidates: list = field(default_factory=list)
    out_index: int = 0
    processed: int = 0
    profile: ProfileWindow = None


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
        fps = 30  # not always probeable (high/variable-rate sources); fall back
    return width, height, fps


def make_source_options(
    cfg: AppConfig, url: str, fps: int, native_w: int, native_h: int, out_w: int, out_h: int
):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    # Parser fallback caps describe the INCOMING (native) stream.
    opt.fallback_h264_width = native_w
    opt.fallback_h264_height = native_h
    # Leave the decoder fps unspecified (-1) rather than pinning it: a fixed
    # dec-fps rejects streams whose real rate differs (high/variable-rate or
    # mis-probed sources) with a framerate mismatch. auto_caps_from_stream emits
    # the native rate; inference.fps still caps the output at the VideoSender.
    opt.fallback_h264_fps = -1
    # Delivered-resolution control: when the chosen output size is below native,
    # scale the decoded frame down once at the decode tail (videoscale) so model,
    # video, and metadata all share one resolution and overlays stay aligned. When
    # output == native, leave caps off to keep the proven zero-copy DMA path.
    if out_w > 0 and out_h > 0 and (out_w != native_w or out_h != native_h):
        opt.use_videoscale = True
        opt.output_caps.enable = True
        opt.output_caps.width = out_w
        opt.output_caps.height = out_h
    else:
        opt.output_caps.enable = False
    return opt


def make_encoded_source_options(cfg: AppConfig, url: str, native_w: int, native_h: int):
    """RTSP input that stops at the H.264 parser instead of decoding.

    The encoded access unit is what both Insight and the decoder consume, so the
    stream reaches Insight without a re-encode. That removes one hardware encoder
    per stream - which is the allocation that actually runs out first at high
    stream counts (measured: the 9th 720p stream died on
    `n4_encoder_76: Allocate output buffers failed`, not on decoder memory).
    Note: this alone does NOT preserve source timestamps end-to-end - rtspsrc
    converts RTP into pipeline running time and the payloader re-derives the wire
    timestamp from PTS. What matters for metadata matching is that BOTH legs
    carry the same GST_BUFFER_PTS, which is why every bridge input below sets
    do_timestamp=False.
    """
    opt = pyneat.RtspEncodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.h264_payload_type = 96
    opt.insert_queue = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = native_w
    opt.fallback_h264_height = native_h
    opt.fallback_h264_fps = -1  # never pin the rate; sources vary and may be very high
    opt.drop_on_latency = True
    return opt


def make_decoder_graph(cfg: AppConfig, runtime: StreamRuntime, output_name: str):
    """Decode-only leg: the sole consumer that pays for a decoder.

    `caps_raw` after the decoder is required, not decorative - without it the
    downstream preprocess has no geometry and the graph fails to start with
    "Preproc: missing input w/h/format from upstream input contract".

    The buffer counts are the point of this path: the RtspDecodedInput group
    hides them, so the untuned defaults (input 8 frames, output 14) are what make
    each stream expensive. SimaDecodeOptions exposes them, and dropping to
    input 2 / output 4 with the low-latency tuning is how the reference example
    fits far more streams into the same 1.8 GB decoder budget.
    """
    decode = pyneat.SimaDecodeOptions()
    decode.type = pyneat.SimaDecodeType.H264
    decode.sima_allocator_type = 2
    decode.out_format = pyneat.Format.NV12
    decode.decoder_name = f"decoder{runtime.channel}"
    decode.raw_output = True
    decode.next_element = "CVU"
    decode.dec_width = runtime.native_w
    decode.dec_height = runtime.native_h
    decode.dec_fps = -1  # never pin the rate; sources vary and may be very high
    decode.num_buffers = cfg.decoder_buffers
    decode.input_buffers = cfg.decoder_input_buffers
    decode.decoder_tuning = cfg.decoder_tuning
    decode.memory_opt = cfg.decoder_tuning in ("low-memory", "throughput-low-latency")

    # The rtsp->decoder edge is lowered as appsink->appsrc unless Core fuses the
    # fan-out. Starting this graph at sima_decode leaves the boundary undeclared,
    # so Core synthesizes it with stock InputOptions - and the stock default
    # do_timestamp=True makes that appsrc overwrite every buffer PTS with THIS
    # pipeline's running time. That is what breaks Insight's exact rtp_timestamp
    # match under passthrough: the video leg keeps the encoded AU's PTS, the
    # metadata leg gets a different origin, and the two drift apart per stream.
    # Declaring the input node makes the options ours.
    bridge_in = pyneat.InputOptions()
    bridge_in.do_timestamp = False
    bridge_in.is_live = True
    bridge_in.block = True

    graph = pyneat.Graph("h264_decoder")
    graph.add(pyneat.nodes.input(f"h264_au_{runtime.channel}", bridge_in))
    graph.add(pyneat.nodes.sima_decode(decode))
    graph.add(
        pyneat.nodes.caps_raw(
            "NV12", runtime.native_w, runtime.native_h, 0, pyneat.CapsMemory.Any
        )
    )
    graph.add(pyneat.nodes.output(output_name))
    return graph


def make_model(cfg: AppConfig, model_path: str, name_suffix: str,
               input_max_w: int = 0, input_max_h: int = 0):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # Preprocess input capacity defaults to 1920x1080, so anything larger (a 4K
    # delivered tier) fails the graph compile with "input width N exceeds
    # configured capacity". Size the envelope to the frame this model will
    # actually receive - the delivered resolution, since videoscale has already
    # run at the decode tail - rather than a blanket 4K, so the allocation stays
    # proportional to the tier in use.
    if input_max_w > 0 and input_max_h > 0:
        opt.preprocess.input_max_width = input_max_w
        opt.preprocess.input_max_height = input_max_h
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    if hasattr(opt, "name_suffix"):
        opt.name_suffix = name_suffix
    return pyneat.Model(model_path, opt)


def build_pipeline_once(cfg: AppConfig, runtime: StreamRuntime, tier_index: int) -> None:
    """One pipeline build attempt. Assumes run/graph/model have been cleared.

    The delivered output resolution is runtime.candidates[runtime.out_index]; the
    caller sets out_index (from the shared output-bandwidth budget) before building.
    """
    size = cfg.resolutions[tier_index]
    model_path = cfg.tier_model_path(size)
    out_w, out_h, _rate = runtime.candidates[runtime.out_index]
    suffix = f"_s{runtime.channel}_t{size}_o{out_h}"

    # Decided up front: it determines the geometry the model will actually see.
    # Under passthrough nothing rescales, so the model receives NATIVE frames and
    # the preprocess envelope must be sized for those - sizing it to the smaller
    # delivered size would fail the graph compile with "input width N exceeds
    # configured capacity" the moment the output policy picks a lower tier.
    save_debug_frames = bool(cfg.save_dir and cfg.save_every > 0)
    passthrough = cfg.encoded_passthrough and not save_debug_frames
    model_w = runtime.native_w if passthrough else out_w
    model_h = runtime.native_h if passthrough else out_h
    model = make_model(cfg, model_path, suffix, model_w, model_h)

    model_name = "model"
    video_name = "video"
    debug_frame_name = "debug_frame"
    detections_name = "detections"
    debug_output_name = "debug_output"
    output_name = debug_output_name if save_debug_frames else detections_name

    graph = pyneat.Graph()
    live_link_options = pyneat.GraphLinkOptions()
    live_link_options.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream

    model_graph = pyneat.Graph(model_name)
    # do_timestamp MUST be False. The default (True) makes the bridging appsrc
    # restamp every buffer with pipeline running-time at push, discarding the
    # source PTS. Under passthrough the video keeps the SOURCE RTP timestamps, so
    # a restamped metadata clock can never match Insight's exact 32-bit key lookup
    # and no box ever renders. (This was invisible on the old re-encode topology
    # because BOTH legs came off that same restamping appsrc, so they agreed.)
    # The reference high-density pipeline has no appsrc in this path at all.
    # Only meaningful under passthrough, where the video keeps the SOURCE PTS and
    # the metadata must too. In the re-encode topology BOTH legs hang off the same
    # branch and are restamped identically, so they already agree - disabling it
    # there would desynchronise the very thing it is meant to fix.
    model_input_options = pyneat.InputOptions()
    model_input_options.do_timestamp = not passthrough
    model_graph.connect(pyneat.nodes.input(model_name, model_input_options), model)
    detections_graph = pyneat.Graph(detections_name)
    detections_graph.add(pyneat.nodes.output(detections_name, pyneat.OutputOptions.every_frame(4)))

    video_port = 0
    branch = None

    if passthrough:
        # One encoded producer feeding two consumers. Core fuses this ordinary
        # fan-out so VideoSender takes the read-only H.264 access unit before the
        # decoder - no branch node, no re-encode, no second decoded copy.
        rtsp = pyneat.groups.rtsp_encoded_input(
            make_encoded_source_options(cfg, runtime.url, runtime.native_w, runtime.native_h)
        )
        decoder = make_decoder_graph(cfg, runtime, model_name)
        graph.connect(rtsp, decoder)
        graph.connect(decoder, model_graph, live_link_options)

        if cfg.video_enabled:
            video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
            video_options.host = cfg.insight_host
            video_options.channel = runtime.channel
            video_options.video_port_base = cfg.video_port_base
            # A synchronising sink would hold the shared live pipeline in PAUSED
            # waiting for preroll from every stream.
            if hasattr(video_options, "async_"):
                video_options.async_ = False
            video_port = video_options.video_port
            sender = pyneat.groups.video_sender(video_options)
            sender.set_name(f"encoded_insight_video_sender_{runtime.channel}")
            # Both legs must carry the SAME GST_BUFFER_PTS, so this bridge needs
            # the same treatment as the decoder one. block=False here: the RTP
            # leg must never back-pressure detection.
            video_in = pyneat.InputOptions()
            video_in.do_timestamp = False
            video_in.is_live = True
            video_in.block = False
            video_graph = pyneat.Graph(video_name)
            video_graph.connect(pyneat.nodes.input(video_name, video_in), sender)
            graph.connect(rtsp, video_graph, live_link_options)
    else:
        source = pyneat.groups.rtsp_decoded_input(
            make_source_options(
                cfg, runtime.url, runtime.output_fps,
                runtime.native_w, runtime.native_h, out_w, out_h,
            )
        )
        outputs = [model_name]
        if cfg.video_enabled:
            outputs.append(video_name)
        if save_debug_frames:
            outputs.append(debug_frame_name)
        branch = pyneat.graphs.branch("source", outputs)
        graph.connect(source, branch)

        if cfg.video_enabled:
            video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
                out_w, out_h, runtime.output_fps
            )
            video_options.encoder.bitrate_kbps = 1000
            video_options.host = cfg.insight_host
            video_options.channel = runtime.channel
            video_options.video_port_base = cfg.video_port_base
            video_port = video_options.video_port
            video_graph = pyneat.Graph(video_name)
            video_graph.connect(
                pyneat.nodes.input(video_name), pyneat.groups.video_sender(video_options)
            )
            graph.connect(branch, video_graph, live_link_options)

        graph.connect(branch, model_graph, live_link_options)

    graph.connect(model_graph, detections_graph)

    if save_debug_frames:
        frames = pyneat.Graph(debug_frame_name)
        frames.add(pyneat.nodes.output(debug_frame_name, pyneat.OutputOptions.every_frame(4)))
        debug_join = pyneat.graphs.combine(
            [debug_frame_name, detections_name], debug_output_name, pyneat.CombinePolicy.ByFrame
        )
        graph.connect(branch, frames, live_link_options)
        graph.connect(frames, debug_join)
        graph.connect(detections_graph, debug_join)

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy

    runtime.model = model
    runtime.graph = graph
    runtime.run = graph.build(run_options)
    runtime.output_name = output_name
    runtime.video_port = video_port
    runtime.tier_index = tier_index

    # Decide rtp_timestamp from what Core ACTUALLY lowered, not from hope. If a
    # bridging appsrc still restamps PTS, an rtp_timestamp can never match and
    # Insight discards every message silently; omitting the field instead lets the
    # browser fall back to arrival order, so boxes render with small bounded lag.
    backend = ""
    try:
        backend = graph.describe_backend()
    except Exception:  # noqa: BLE001 - introspection is best-effort
        pass
    restamped = "do-timestamp=true" in backend
    if cfg.metadata_rtp_timestamp == "on":
        runtime.emit_rtp_timestamp = True
    elif cfg.metadata_rtp_timestamp == "off":
        runtime.emit_rtp_timestamp = False
    else:
        runtime.emit_rtp_timestamp = not restamped
    if not runtime.emit_rtp_timestamp:
        print(f"[stream {runtime.id}] metadata rtp_timestamp DISABLED "
              f"(graph restamps PTS; Insight will match by arrival order)", flush=True)

    # Detection runs on whatever frame the model actually receives. Under
    # passthrough nothing rescales, so that is the NATIVE size; using out_w/out_h
    # there would put boxes in the wrong coordinate space the moment the output
    # policy picks a smaller tier.
    if passthrough:
        runtime.frame_w = runtime.native_w
        runtime.frame_h = runtime.native_h
    else:
        runtime.frame_w = out_w
        runtime.frame_h = out_h


def build_pipeline(cfg: AppConfig, runtime: StreamRuntime, tier_index: int) -> None:
    """(Re)build the RTSP -> {video, model} graph for a stream at a resolution tier.

    The MetadataSender and probed geometry persist; only model/graph/run rebuild.
    Model loads are serialized across streams and retried: concurrent rebuilds
    otherwise race on MLA/RPMsg setup ("Unable to load model" under heavy
    tier-switch thrash). The steady-state per-frame pull loop is unaffected.
    """
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            with _BUILD_LOCK:
                # Tear down (run -> graph -> model) AND rebuild under the same lock,
                # so two streams never tear down / load MLA pipelines concurrently.
                rebuilding = runtime.model is not None
                if runtime.run is not None:
                    try:
                        runtime.run.close()
                    except Exception:
                        pass
                runtime.run = None
                runtime.graph = None
                runtime.model = None
                # The Neat MLA pipeline tears down asynchronously; give it time to
                # fully release before building the next one, or concurrent
                # teardown/build in the runtime can segfault when several streams
                # switch tiers at once.
                if rebuilding:
                    time.sleep(0.8)
                build_pipeline_once(cfg, runtime, tier_index)
            return
        except Exception as exc:  # noqa: BLE001 - retry transient MLA build failures
            runtime.run = None
            runtime.graph = None
            runtime.model = None
            if attempt >= attempts:
                raise
            print(
                f"[warn] stream {runtime.id} pipeline build attempt {attempt} "
                f"failed ({exc}); retrying",
                file=sys.stderr,
            )
            time.sleep(0.75)


def init_stream_runtime(cfg: AppConfig, channel: int, source: StreamSource, labels: list[str],
                        initial_tier: int, active_count: int) -> StreamRuntime:
    native_w, native_h, fps = probe_rtsp(source.rtsp_url)
    output_fps = cfg.fps if cfg.fps > 0 else fps
    candidates = output_candidates(native_w, native_h, output_fps, cfg.output_policy.heights)
    out_index = select_output_index(
        active_count, candidates, cfg.output_policy.budget_megapixels_per_s
    )
    runtime = StreamRuntime(
        channel=channel,
        id=source.id,
        url=source.rtsp_url,
        labels=labels,
        native_w=native_w,
        native_h=native_h,
        output_fps=output_fps,
        candidates=candidates,
        out_index=out_index,
        profile=ProfileWindow(cfg.profile, source.id),
    )
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.channel = channel
    metadata_options.metadata_port_base = cfg.metadata_port_base
    # Nonblocking UDP: at high source rates a congested Insight must show up as
    # dropped datagrams, never as back-pressure that stalls the detector thread.
    if hasattr(metadata_options, "nonblocking"):
        metadata_options.nonblocking = True
    runtime.metadata_sender = pyneat.MetadataSender(metadata_options)

    build_pipeline(cfg, runtime, initial_tier)  # sets frame_w/frame_h from out_index
    out_w, out_h, _rate = runtime.candidates[runtime.out_index]
    print(
        f"[stream {source.id}] channel={channel} rtsp={source.rtsp_url} "
        f"native={native_w}x{native_h}@{output_fps} output={out_w}x{out_h} "
        f"tier={cfg.resolutions[initial_tier]} "
        f"video={runtime.video_port if cfg.video_enabled else 'disabled'} "
        f"metadata={runtime.metadata_sender.metadata_port()}",
        flush=True,
    )
    return runtime


def switch_tier(cfg: AppConfig, runtime: StreamRuntime, new_tier: int) -> None:
    old_size = cfg.resolutions[runtime.tier_index]
    new_size = cfg.resolutions[new_tier]
    print(f"[stream {runtime.id}] tier {old_size} -> {new_size} (rebuilding pipeline)", flush=True)
    build_pipeline(cfg, runtime, new_tier)  # handles teardown (run -> graph -> model)


def switch_output(cfg: AppConfig, runtime: StreamRuntime, new_out_index: int) -> None:
    old_w, old_h, _ = runtime.candidates[runtime.out_index]
    new_w, new_h, _ = runtime.candidates[new_out_index]
    print(
        f"[stream {runtime.id}] output {old_w}x{old_h} -> {new_w}x{new_h} (rebuilding pipeline)",
        flush=True,
    )
    runtime.out_index = new_out_index  # build_pipeline_once reads out_index for geometry
    build_pipeline(cfg, runtime, runtime.tier_index)  # handles teardown (run -> graph -> model)


def rtp_timestamp_from_pts_ns(pts_ns: int) -> int:
    """Source PTS -> the 32-bit 90 kHz RTP timestamp Insight matches on."""
    if pts_ns < 0:
        return 0
    return ((pts_ns * 90_000) // 1_000_000_000) & 0xFFFFFFFF


def send_metadata(runtime: StreamRuntime, sample, boxes: list[dict], active_tier: int,
                  stream_count: int) -> None:
    """Publish detections on the frame's own source clock.

    Insight matches metadata to a held video frame by an EXACT 32-bit
    `rtp_timestamp` key, with no arrival-order fallback once the field is
    present - so a payload without it never renders a box, however healthy the
    stream is. The convenience `send_metadata()` API cannot carry that field
    (MetadataSender.h has no such parameter), which is why this builds the full
    envelope and ships it with `send_raw_json`, exactly as the high-density
    example does.

    This is not a frame-rate problem, but frame rate amplifies it: Insight's
    pending-match map holds ~300 entries, which is a 12 s window at 25 fps but
    only ~2 s at 150 messages/s, so unmatched metadata is evicted before its
    frame arrives. Nothing here caps or paces anything - detection still runs at
    full hardware speed.
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
        "data": json.loads(
            metadata_payload(
                boxes, runtime.labels, runtime.frame_w, runtime.frame_h, active_tier,
                stream_count, runtime.id,
            )
        ),
        "timestamp": pts_ns // 1_000_000 if pts_ns >= 0 else int(time.time() * 1000),
        "frame_id": str(frame_id) if frame_id >= 0 else "",
        "stream_id": runtime.id,
        "stream_index": runtime.channel,
        "pts_ns": pts_ns,
        "dts_ns": _attr("dts_ns"),
        "duration_ns": _attr("duration_ns"),
        "input_seq": _attr("input_seq"),
    }
    if pts_ns >= 0 and runtime.emit_rtp_timestamp:
        payload["rtp_timestamp"] = rtp_timestamp_from_pts_ns(pts_ns)

    try:
        runtime.metadata_sender.send_raw_json(json.dumps(payload, separators=(",", ":")))
    except Exception:  # noqa: BLE001 - a dropped datagram must never stall detection
        pass


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
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
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


def draw_boxes_and_banner(frame, boxes: list[dict], min_score: float, banner: str) -> None:
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
    cv2.putText(frame, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def maybe_save_debug_frame(cfg: AppConfig, runtime: StreamRuntime, sample, boxes: list[dict],
                           active_tier: int, stream_count: int) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or runtime.processed % cfg.save_every != 0:
        return
    frame_tensor = first_tensor_from_sample(sample)
    if frame_tensor is None:
        return
    frame = tensor_bgr_from_decoded(frame_tensor)
    banner = f"{runtime.id}  tier={active_tier}  streams={stream_count}"
    draw_boxes_and_banner(frame, boxes, cfg.min_score, banner)
    out_path = Path(cfg.save_dir) / f"stream_{runtime.channel}_frame_{runtime.processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


# ── stream registry (runtime add/remove) ──────────────────────────────────────


@dataclass
class ManagedStream:
    id: str
    url: str
    channel: int
    stop_event: threading.Event
    thread: object = None


class StreamManager:
    """Owns the live set of streams, channel allocation, and shared stop/error state."""

    def __init__(self, cfg: AppConfig, labels: list[str]) -> None:
        self.cfg = cfg
        self.labels = labels
        self.lock = threading.Lock()
        self.streams: dict[str, ManagedStream] = {}
        self.free_channels = list(range(cfg.max_streams))
        self.stop_event = threading.Event()
        self.errors: list[Exception] = []
        self.initial_tier = len(cfg.resolutions) // 2

    def active_count(self) -> int:
        with self.lock:
            return len(self.streams)

    def add(self, source: StreamSource) -> bool:
        with self.lock:
            if source.id in self.streams:
                return False
            if len(self.streams) >= self.cfg.max_streams or not self.free_channels:
                print(f"[warn] cannot add stream {source.id}: at max_streams", file=sys.stderr)
                return False
            channel = self.free_channels.pop(0)
            managed = ManagedStream(source.id, source.rtsp_url, channel, threading.Event())
            self.streams[source.id] = managed
        managed.thread = threading.Thread(
            target=self._consume, args=(managed,), name=f"stream-{source.id}"
        )
        managed.thread.start()
        return True

    def remove(self, stream_id: str) -> None:
        with self.lock:
            managed = self.streams.pop(stream_id, None)
        if managed is None:
            return
        managed.stop_event.set()
        if managed.thread is not None:
            managed.thread.join(timeout=15.0)
        with self.lock:
            self.free_channels.append(managed.channel)
            self.free_channels.sort()
        print(f"[stream {stream_id}] removed (channel {managed.channel} released)", flush=True)

    def apply_sources(self, sources: list[StreamSource]) -> None:
        with self.lock:
            current_ids = set(self.streams.keys())
        new_by_id = {s.id: s for s in sources}
        to_remove = [sid for sid in current_ids if sid not in new_by_id]
        to_add = [s for sid, s in new_by_id.items() if sid not in current_ids]
        for stream_id in to_remove:
            self.remove(stream_id)
        for source in to_add:
            self.add(source)

    def all_done(self) -> bool:
        with self.lock:
            managed = list(self.streams.values())
        if not managed:
            return False
        return all(m.thread is not None and not m.thread.is_alive() for m in managed)

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.lock:
            managed = list(self.streams.values())
        for m in managed:
            m.stop_event.set()
        for m in managed:
            if m.thread is not None:
                m.thread.join(timeout=15.0)

    def _reconnect_stream(self, managed: "ManagedStream", runtime: "StreamRuntime") -> bool:
        """Rebuild a stream at its current tier after a source-side close.

        A source drop (RTSP disconnect, EOS at a loop boundary, RTCP timeout) is
        not fatal. Returns True once reconnected, or False if we should stop the
        stream (shutdown, stream removed, or the reconnect budget is exhausted).
        """
        max_reconnects = 30
        for attempt in range(1, max_reconnects + 1):
            if managed.stop_event.is_set() or self.stop_event.is_set():
                return False
            backoff_ms = min(500 * attempt, 5000)
            print(
                f"[warn] stream {managed.id} source closed; reconnecting "
                f"(attempt {attempt}/{max_reconnects} in {backoff_ms}ms)",
                file=sys.stderr,
            )
            time.sleep(backoff_ms / 1000.0)
            if managed.stop_event.is_set() or self.stop_event.is_set():
                return False
            try:
                build_pipeline(self.cfg, runtime, runtime.tier_index)
                print(
                    f"[stream {managed.id}] reconnected at tier "
                    f"{self.cfg.resolutions[runtime.tier_index]}",
                    flush=True,
                )
                return True
            except Exception as exc:  # noqa: BLE001 - keep retrying with backoff
                print(
                    f"[warn] stream {managed.id} reconnect attempt {attempt} failed ({exc})",
                    file=sys.stderr,
                )
        print(
            f"[warn] stream {managed.id} reconnect budget exhausted; stopping stream",
            file=sys.stderr,
        )
        return False

    def _consume(self, managed: ManagedStream) -> None:
        cfg = self.cfg
        runtime = None
        policy_state = PolicyState(tier_index=self.initial_tier, pending_index=self.initial_tier)
        transient_recoveries = 0
        max_transient_recoveries = 10
        last_switch_ms = -1e12  # allow the first switch immediately
        min_switch_interval_ms = 2500.0
        # Throttle: inference.fps caps the rate at which we PROCESS + emit. <=0 is
        # uncapped. The runtime already drops stale decoded frames (KeepLatest), so
        # this bounds MLA-output/network/CPU work, not the hardware decode rate.
        min_process_interval_ms = 1000.0 / cfg.fps if cfg.fps > 0 else 0.0
        last_process_ms = -1e12
        # Desync identical streams so a fleet-wide change rolls through, not storms.
        stagger_ms = float(managed.channel) * 350.0
        # Retry transient MLA blips before paying for a rebuild (avoids storms).
        consecutive_transient = 0
        transient_retries = 8
        try:
            source = StreamSource(managed.id, managed.url)
            runtime = init_stream_runtime(
                cfg, managed.channel, source, self.labels, self.initial_tier, self.active_count()
            )
            while (
                not managed.stop_event.is_set()
                and not self.stop_event.is_set()
                and (cfg.frames <= 0 or runtime.processed < cfg.frames)
            ):
                # Throttle: pace processing to inference.fps. We keep pulling the
                # freshest frame (KeepLatest) but skip frames arriving faster than
                # the target, capping metadata/video/tier work per stream.
                if min_process_interval_ms > 0.0:
                    now = time_ms()
                    due = last_process_ms + min_process_interval_ms
                    if now < due:
                        time.sleep(min(due - now, 15.0) / 1000.0)
                        continue
                pull_start = time_ms()
                sample = runtime.run.pull(runtime.output_name, 50)
                pull_end = time_ms()
                if sample is None:
                    last_error_fn = getattr(runtime.run, "last_error", None)
                    last_error = last_error_fn() if callable(last_error_fn) else ""
                    stopping = managed.stop_event.is_set() or self.stop_event.is_set()
                    if last_error:
                        # MLA model load is deferred to run start, so an over-churned
                        # rebuild can surface here as a transient caps/load error.
                        transient = "Unable to load model" in last_error or "misconfig.caps" in last_error
                        if transient and not stopping:
                            # Retry the pull a few times before rebuilding: most blips
                            # clear on their own; an immediate rebuild cascades into a storm.
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
                                build_pipeline(cfg, runtime, runtime.tier_index)
                                last_process_ms = -1e12
                                continue
                        raise RuntimeError(f"stream {managed.id} runtime error: {last_error}")
                    running_fn = getattr(runtime.run, "running", None)
                    if callable(running_fn) and not running_fn():
                        # Source ended: clean shutdown -> exit; otherwise reconnect.
                        if stopping:
                            break
                        if not self._reconnect_stream(managed, runtime):
                            break
                        last_process_ms = -1e12
                        consecutive_transient = 0
                        continue
                    continue

                payload = extract_bbox_payload(sample)
                boxes = parse_boxes_strict(payload, runtime.frame_w, runtime.frame_h, cfg.max_detections)
                runtime.processed += 1
                transient_recoveries = 0  # a successful pull clears the transient-error budget
                consecutive_transient = 0
                last_process_ms = time_ms()
                if runtime.processed <= cfg.warmup_frames:
                    continue

                active = self.active_count()
                stats = frame_stats(boxes, cfg.min_score)
                target_tier = effective_tier(
                    policy_state, stats, cfg.policy, active, cfg.budget_units
                )
                if (
                    target_tier != runtime.tier_index
                    and pull_end - last_switch_ms >= min_switch_interval_ms + stagger_ms
                ):
                    # Rate-limit rebuilds and stagger by channel so a budget change
                    # doesn't switch every identical stream in lockstep.
                    switch_tier(cfg, runtime, target_tier)
                    last_switch_ms = time_ms()
                    last_process_ms = -1e12
                    continue

                # Output-resolution axis: the shared bandwidth budget's fair share
                # depends only on the active stream count, so this changes just on
                # add/remove. Reuse the same rate-limit + per-channel stagger so a
                # fleet-wide step (e.g. the 16th stream joins) rolls through rather
                # than rebuilding every stream in lockstep.
                target_out = select_output_index(
                    active, runtime.candidates, cfg.output_policy.budget_megapixels_per_s
                )
                if (
                    target_out != runtime.out_index
                    and pull_end - last_switch_ms >= min_switch_interval_ms + stagger_ms
                ):
                    switch_output(cfg, runtime, target_out)
                    last_switch_ms = time_ms()
                    last_process_ms = -1e12
                    continue

                active_tier = cfg.resolutions[runtime.tier_index]
                send_metadata(runtime, sample, boxes, active_tier, self.active_count())
                maybe_save_debug_frame(cfg, runtime, sample, boxes, active_tier, self.active_count())
                runtime.profile.add(pull_end - pull_start, len(boxes))
        except Exception as exc:  # noqa: BLE001 - surface to main thread
            self.errors.append(exc)
            self.stop_event.set()
        finally:
            if runtime is not None:
                if runtime.profile is not None:
                    runtime.profile.flush()
                try:
                    runtime.run.close()
                except Exception:
                    pass
                print(f"[stream {managed.id}] processed={runtime.processed}", flush=True)


def run_app(cfg: AppConfig) -> None:
    if cfg.profile:
        os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
        os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
        os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
    if cfg.save_dir:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_path)
    manager = StreamManager(cfg, labels)
    try:
        last_mtime = cfg.config_path.stat().st_mtime
    except OSError:
        last_mtime = 0.0

    for source in cfg.sources:
        manager.add(source)

    try:
        while not manager.stop_event.is_set():
            if manager.all_done():
                break
            time.sleep(cfg.config_watch_seconds)
            try:
                mtime = cfg.config_path.stat().st_mtime
            except OSError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    sources = reload_sources(cfg.config_path)
                except Exception as exc:  # noqa: BLE001 - a bad edit shouldn't kill the app
                    print(f"[warn] ignoring invalid config reload: {exc}", file=sys.stderr)
                    continue
                print(f"[config] reload: {[s.id for s in sources]}", flush=True)
                manager.apply_sources(sources)
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()

    if manager.errors:
        raise manager.errors[0]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive-resolution multi-stream YOLO26 Insight detector"
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
            print(f"Config validated: {args.config} (streams={len(cfg.sources)})")
            return 0
        load_runtime_dependencies()
        run_app(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level error boundary
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
