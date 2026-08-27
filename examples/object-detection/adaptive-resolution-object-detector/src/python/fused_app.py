"""Fused multi-stream RTSP YOLO26 detector: ONE graph, ONE shared detector.

The companion to `main.py` in this example, and the opposite trade-off.

    main.py        one graph PER stream. A stream can be built or torn down
                   while the others keep running, at the cost of per-stream
                   bridges that cap reliable metadata at ~6 streams.

    fused_main.py  ONE graph for every stream, fanning into a single shared
                   detector, with the source H.264 passed through to Insight
                   untouched (no re-encode). Boxes stay correct at high stream
                   counts, but adding a stream rebuilds the whole graph.

The `pipelines/` bundle drives this file as both the `scale` pipeline (one
process) and the `group` pipeline (several independent processes, each owning a
subset of the streams).

Both detectors live in this example so the bundle is standalone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import glob
import json
import os
import struct
import signal
import sys
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# Largest upscale the hardware scaler will actually perform. Past this it emits
# nothing at all - silently, with no error - so the shared detector geometry is
# clamped against it (see run_app). Measured on Modalix: 2x works, 3x does not.
MAX_UPSCALE = 2
# The detector's own input size; clamping below this would throw away detail the
# model is going to use anyway.
MODEL_INPUT = 640

cv2 = None
np = None
pyneat = None


# Decoder profile shared by the dataclass defaults and load_app_config()'s
# fallbacks, so the two cannot drift again - they disagreed before (18/auto
# declared, 4/throughput-low-latency actually applied). Matches what
# pipelines/pipeline-{scale,group}/pipeline.py generate.
DEFAULT_DECODER_BUFFERS = 8
DEFAULT_DECODER_INPUT_BUFFERS = 2
DEFAULT_DECODER_TUNING = "auto"


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_urls: list[str]
    latency_ms: int = 100
    tcp: bool = True
    frames: int = 0
    fps: int = 0
    max_inflight_per_stream: int = 4
    max_inflight_total: int = 16
    # Decoder pool sizing. num_buffers=4 with throughput-low-latency (which
    # turns memory_opt ON) starves the decoder output pool under network jitter
    # and produces stutter/freezes - that was the old fallback here, and nothing
    # ever got the value declared on this line because load_app_config() always
    # passed its own literal. 8/auto is what pipelines/ generates and runs at 16
    # streams, so the loader fallback below matches it exactly. Lower only when
    # pushing past ~16 streams, where the memory trade becomes worth it.
    decoder_buffers: int = DEFAULT_DECODER_BUFFERS
    decoder_input_buffers: int = DEFAULT_DECODER_INPUT_BUFFERS
    decoder_tuning: str = DEFAULT_DECODER_TUNING
    # DECODER-side fps cap for admission. A source's native rate is declared to
    # the decoder for capacity admission; a very high rate (e.g. 500 fps) x many
    # streams exceeds the decoder core and the whole graph is REJECTED before it
    # starts ("decoder processing capacity unavailable"). Capping the DECLARED
    # decode rate lets admission pass; the decoder then processes up to the cap
    # and drops the rest (drop-on-latency). This does NOT cap the video sent to
    # Insight - passthrough forwards the full-rate source untouched. 0 = uncapped.
    decoder_fps_cap: int = 30
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
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
    labels: list[str]
    profile: "ProfileWindow"
    # Decoded frames awaiting their detection, keyed by frame id. See
    # drain_debug_frames(): the two branches are queued independently, so the
    # newest decoded frame is not the one the next detection describes.
    debug_frames: dict
    frame_w: int
    frame_h: int
    # Geometry the detector leg was normalised to. Detections come back in
    # THIS space, so they must be scaled to frame_w/frame_h before publishing.
    det_w: int
    det_h: int
    output_fps: int
    video_port: int
    processed: int = 0
    closed: bool = False
    debug_pairing_warned: bool = False
    # Wall-clock throttle state for inference.fps. See process_output_sample():
    # this used to be set from the probed rate and then never consulted again,
    # so the cap only ever changed the startup banner.
    last_process_ms: float = -1e12


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
            f"[profile stream={self.stream_index}] frames={self.frames} "
            f"output_fps={output_fps} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_boxes={self.boxes / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
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


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if not str(cfg.labels_path):
        raise ValueError("model.labels must be set")
    if not cfg.rtsp_urls:
        raise ValueError("streams must be set")
    # The original limit was a development-phase scope guard ("this phase supports
    # up to four streams"), not a hardware one: this app's architecture - one fused
    # graph, one shared detector, encoded passthrough - is the same one the
    # high-density example uses to run 16/24/48 streams. Raised so the real ceiling
    # is decided by measured decoder/pool capacity instead of a placeholder.
    if len(cfg.rtsp_urls) > 64:
        raise ValueError("streams must not exceed 64")
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
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
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
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    input_cfg = section(raw, "input")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

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
        labels_path=Path(string_or(model, "labels", str(default_labels))),
        rtsp_urls=rtsp_urls,
        latency_ms=int_or(input_cfg, "latency_ms", 100),
        decoder_buffers=int_or(input_cfg, "decoder_buffers", DEFAULT_DECODER_BUFFERS),
        decoder_input_buffers=int_or(
            input_cfg, "decoder_input_buffers", DEFAULT_DECODER_INPUT_BUFFERS
        ),
        decoder_tuning=string_or(input_cfg, "decoder_tuning", DEFAULT_DECODER_TUNING),
        tcp=bool_or(input_cfg, "tcp", True),
        frames=int_or(inference, "frames", 0),
        fps=int_or(inference, "fps", 0),
        max_inflight_per_stream=int_or(inference, "max_inflight_per_stream", 4),
        max_inflight_total=int_or(inference, "max_inflight_total", 16),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
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


def build_source_options(cfg: AppConfig, url: str, fps: int, width: int, height: int,
                         det_w: int = 0, det_h: int = 0):
    """Source options for one stream.

    det_w/det_h are the geometry the DETECTOR leg must present. A single shared
    model has a single input port, so streams of different native sizes must be
    normalised to one geometry before it ("input spec mismatch for port 'in'").
    Only the detector leg is scaled - the video sent to Insight is the encoded
    passthrough, which keeps each camera's native resolution. Box coordinates stay
    correct because metadata is emitted against the NATIVE frame size, and the
    model letterboxes either way.
    """
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.payload_type = 96
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.auto_caps_from_stream = True
    opt.fallback_h264_width = width
    opt.fallback_h264_height = height
    opt.source_fps = fps
    opt.output_caps.enable = True
    opt.output_caps.format = pyneat.Format.NV12
    scale_w = det_w if det_w > 0 else width
    scale_h = det_h if det_h > 0 else height
    if (scale_w, scale_h) != (width, height):
        opt.use_videoscale = True
    opt.output_caps.width = scale_w
    opt.output_caps.height = scale_h
    # Do NOT pin the decoded framerate when the source will be admission-capped:
    # a caps framerate that differs from the real stream fails negotiation
    # ("framerate mismatch"). 0 leaves it unconstrained so the decoder emits
    # whatever rate it achieves. Only pin fps for normal (uncapped) sources.
    capped = cfg.decoder_fps_cap > 0 and fps > cfg.decoder_fps_cap
    opt.output_caps.fps = 0 if capped else fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    opt.num_buffers = cfg.decoder_buffers
    return opt


def output_caps_enabled(caps) -> bool:
    return caps.enable or caps.width > 0 or caps.height > 0 or caps.fps > 0


def build_encoded_source_graph(opt) -> pyneat.Graph:
    source = pyneat.Graph("rtsp_encoded_source")

    encoded_opt = pyneat.RtspEncodedInputOptions()
    encoded_opt.url = opt.url
    encoded_opt.codec = pyneat.RtspCodec.H264
    encoded_opt.latency_ms = opt.latency_ms
    encoded_opt.tcp = opt.tcp
    if hasattr(encoded_opt, 'drop_on_latency'):
        encoded_opt.drop_on_latency = True
    encoded_opt.source_fps = opt.source_fps
    encoded_opt.fallback_h264_width = opt.fallback_h264_width
    encoded_opt.fallback_h264_height = opt.fallback_h264_height
    source.add(pyneat.groups.rtsp_encoded_input(encoded_opt))
    return source


def h264_decode_input_options():
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = pyneat.Format.H264
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
    return opt


def h264_video_input_options():
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = pyneat.Format.H264
    if hasattr(pyneat, "InputMemoryPolicy") and hasattr(opt, "memory_policy"):
        opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    elif hasattr(opt, "use_simaai_pool"):
        opt.use_simaai_pool = False
    return opt


def build_decode_graph(input_name: str, opt, cfg=None) -> pyneat.Graph:
    decode = pyneat.Graph("decode")
    dec_w = opt.h264_width if opt.h264_width > 0 else opt.fallback_h264_width
    dec_h = opt.h264_height if opt.h264_height > 0 else opt.fallback_h264_height

    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = opt.sima_allocator_type
    dec.out_format = pyneat.Format.NV12
    dec.decoder_name = opt.decoder_name
    dec.raw_output = opt.decoder_raw_output
    dec.next_element = opt.decoder_next_element
    dec.dec_width = dec_w
    dec.dec_height = dec_h
    # High-fps sources: leave dec_fps UNSPECIFIED (-1) rather than pinning a
    # value. Pinning the source rate (e.g. 500) makes admission reject the graph
    # ("processing capacity unavailable"); pinning a lower cap (e.g. 30) fails
    # caps negotiation ("framerate mismatch"). -1 lets admission use its default
    # (~30 fps) AND leaves caps unpinned, so the decoder admits and then processes
    # what it can, dropping the rest. Normal sources keep their exact rate.
    #
    # inference.fps deliberately does NOT feed this. Pinning a rate BELOW the
    # real stream is exactly what the paragraph above rules out, so there is no
    # safe way to make the decoder emit at a requested lower rate here - only
    # the source's true rate, or unpinned. inference.fps is applied after the
    # pull instead; see should_throttle_fps().
    cap = cfg.decoder_fps_cap if cfg is not None else 0
    dec.dec_fps = -1 if (cap > 0 and opt.source_fps > cap) else opt.source_fps
    dec.num_buffers = opt.num_buffers
    if cfg is not None:
        dec.input_buffers = cfg.decoder_input_buffers
        dec.decoder_tuning = cfg.decoder_tuning
        dec.memory_opt = cfg.decoder_tuning in ('low-memory', 'throughput-low-latency')
    decode.connect(
        pyneat.nodes.input(input_name, h264_decode_input_options()),
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


def build_video_sender_graph(input_name: str, video_options) -> pyneat.Graph:
    video = pyneat.Graph("video_sender")
    video.connect(
        pyneat.nodes.input(input_name, h264_video_input_options()),
        pyneat.groups.video_sender(video_options),
    )
    return video


def build_model(cfg: AppConfig, det_w: int = 0, det_h: int = 0):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    # The preprocess envelope defaults to 1920x1080; anything larger (a 4K source)
    # fails the graph compile with "input width N exceeds configured capacity".
    if det_w > 0 and det_h > 0:
        opt.preprocess.input_max_width = det_w
        opt.preprocess.input_max_height = det_h
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = cfg.min_score
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return pyneat.Model(cfg.model_path, opt)


def build_run_options(stream_count: int = 1) -> pyneat.RunOptions:
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    # One ingress/internal queue slot per stream, floor 4, as the proven
    # 16-stream high-density profile does (it sets queue_depth to its stream
    # count). This sizes the runtime's own ingress and internal queues only.
    #
    # It does NOT size the shared detector stages: the `num-buffers=4` on
    # neatprocesscvu/neatprocessmla in the pipeline dump comes from
    # GraphOptions.advanced_execution.inference_output_buffers ->
    # processmla.output_pool_buffers, whose framework default is 4, and which
    # is set on the Graph constructor rather than here. Measured: 5 streams
    # detect at full rate with those stages still at 4.
    run_options.queue_depth = max(4, min(32, stream_count))
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


def build_detector_graph(cfg: AppConfig, det_w: int = 0, det_h: int = 0):
    model = build_model(cfg, det_w, det_h)
    input_options = model.input_appsrc_options(False)
    input_options.block = True

    detector = pyneat.Graph("detector")
    detector.connect(pyneat.nodes.input("detector_frame", input_options), model)
    return model, detector


def build_detections_graph() -> pyneat.Graph:
    detections = pyneat.Graph("detections")
    detections.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))
    return detections


def build_debug_frame_graph(stream_index: int) -> pyneat.Graph:
    frames = pyneat.Graph("debug_frame")
    frames.connect(
        pyneat.nodes.input("debug_frame"),
        pyneat.nodes.output(f"debug_frame_{stream_index}", pyneat.OutputOptions.every_frame(4)),
    )
    return frames


def build_stream_runtime(
    cfg: AppConfig, stream_index: int, url: str, labels: list[str],
    det_w: int = 0, det_h: int = 0, probe: tuple[int, int, int] | None = None
) -> StreamRuntime:
    # Reuse the probe from run_app instead of hitting the RTSP source a second
    # time - a duplicate probe per stream is N extra serial handshakes on every
    # (re)build, which dominates rebuild latency at high stream counts.
    frame_w, frame_h, fps = probe if probe is not None else probe_rtsp(url)
    output_fps = cfg.fps if cfg.fps > 0 else fps

    source_options = build_source_options(cfg, url, fps, frame_w, frame_h, det_w, det_h)

    video_port = 0
    if cfg.video_enabled:
        video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
        video_options.host = cfg.insight_host
        video_options.channel = stream_index
        video_options.video_port_base = cfg.video_port_base
        video_port = video_options.video_port

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
        labels=labels,
        profile=ProfileWindow(cfg.profile, stream_index),
        debug_frames={},
        frame_w=frame_w,
        frame_h=frame_h,
        det_w=det_w if det_w > 0 else frame_w,
        det_h=det_h if det_h > 0 else frame_h,
        output_fps=output_fps,
        video_port=video_port,
    )


def connect_stream_graph(
    app: AppRuntime, cfg: AppConfig, stream: StreamRuntime, detector_graph
) -> None:
    source = build_encoded_source_graph(stream.source_options)
    decoder = build_decode_graph("decode_h264", stream.source_options, cfg)

    if cfg.video_enabled:
        encoded_branch = pyneat.graphs.branch("encoded", ["decode_h264", "video_h264"])
        app.graph.connect(source, encoded_branch)
        app.graph.connect(encoded_branch, decoder, realtime_link(stream.index, 3))

        video_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_encoded()
        video_options.host = cfg.insight_host
        video_options.channel = stream.index
        video_options.video_port_base = cfg.video_port_base
        video_options.async_ = True
        app.graph.connect(
            encoded_branch,
            build_video_sender_graph("video_h264", video_options),
            realtime_link(stream.index, 3),
        )
    else:
        app.graph.connect(source, decoder, realtime_link(stream.index, 3))

    save_debug_frames = save_frames_enabled(cfg)
    decoded_outputs = ["detector_frame", "debug_frame"] if save_debug_frames else ["detector_frame"]
    decoded_branch = pyneat.graphs.branch("decoded", decoded_outputs)
    app.graph.connect(decoder, decoded_branch)
    app.graph.connect(
        decoded_branch,
        detector_graph,
        realtime_link(
            stream.index,
            4,
            cfg.max_inflight_per_stream,
            cfg.max_inflight_total,
        ),
    )
    if save_debug_frames:
        app.graph.connect(
            decoded_branch, build_debug_frame_graph(stream.index), realtime_link(stream.index, 4)
        )


def send_metadata(stream: StreamRuntime, sample, boxes: list[dict]) -> None:
    # Arrival-order metadata (no rtp_timestamp). This forwards reliably at normal
    # frame rates (measured 99% at 16x25 fps). Exact rtp_timestamp matching is NOT
    # usable here: it needs the video payloader pinned to timestamp-offset=0 (as
    # the high-density C++ app does), but VideoSenderOptions.rtp exposes no such
    # field in pyneat, so the passthrough video keeps a random RTP base the
    # metadata key can never match.
    metadata_boxes = build_metadata_boxes(boxes, stream.labels, stream.frame_w, stream.frame_h)
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    stream.metadata_sender.send_metadata(
        "object-detection",
        json.dumps({"objects": metadata_boxes}, separators=(",", ":")),
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
        label = f"id={int(box['class_id'])} score={score:.6f}"
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
    cfg: AppConfig, stream: StreamRuntime, frame, boxes: list[dict]
) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or stream.processed % cfg.save_every != 0:
        return
    if frame is None:
        return

    frame = frame.copy()
    draw_boxes(frame, boxes, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)


# Set by the SIGTERM/SIGINT handler installed in run_app(). Mirrors
# g_stop_requested in src/cpp/fused_app.h: the fused topology has no per-stream
# manager to signal, so the run loop polls this instead.
_stop_requested = False


def _request_stop(*_args) -> None:
    global _stop_requested
    _stop_requested = True


# Set when the shared detection output closes without anyone asking it to. A
# continuous run (frames == 0) that loses its output produces no further
# metadata at all, so finishing normally would hand supervisors a zero exit for
# a detector that had silently stopped detecting.
_output_closed_unexpectedly = False


def all_streams_done(streams: list[StreamRuntime], frame_limit: int) -> bool:
    if not streams:
        return True
    # A closed stream is done at any frame limit. Returning False outright for
    # frame_limit <= 0 meant a continuous run could never finish, so once the
    # shared output closed the loop spun on it forever.
    return all(
        stream.closed or (frame_limit > 0 and stream.processed >= frame_limit)
        for stream in streams
    )


def should_throttle_fps(cfg: AppConfig, stream: StreamRuntime, now: float) -> bool:
    """True when inference.fps says this stream must wait before processing again.

    Scope, stated plainly: this caps the rate at which THIS STREAM parses boxes
    and emits metadata and debug frames. It does NOT reduce MLA work - the
    shared detector has already inferred the sample by the time it is pulled,
    and the decoder cannot be asked for a lower rate (pinning one below the real
    stream fails caps negotiation; see build_decode_graph). Nor does it change
    the video Insight receives, which is the encoded passthrough at the source's
    own rate. Setting output_fps from cfg.fps at build time changed only the
    startup banner and nothing ever consulted it, so before this the setting did
    nothing at all (mirrors src/cpp/fused_app.h).
    """
    if cfg.fps <= 0:
        return False
    min_interval_ms = 1000.0 / cfg.fps
    return now < stream.last_process_ms + min_interval_ms


def process_output_sample(stream: StreamRuntime, cfg: AppConfig, sample, detection_pull_ms: float) -> None:
    if cfg.frames > 0 and stream.processed >= cfg.frames:
        return

    now = time_ms()
    if should_throttle_fps(cfg, stream, now):
        return
    stream.last_process_ms = now

    payload = extract_bbox_payload(sample)
    # Detections are emitted in the detector's input geometry, which is shared
    # across streams. Clamp in that space, then scale to this stream's native size
    # so boxes line up with the natively-delivered video.
    boxes = parse_boxes_strict(payload, stream.det_w, stream.det_h, cfg.max_detections)
    # The debug frame is tapped off the decoded branch, which build_source_options
    # normalised to the shared detector geometry - so it needs the boxes in THAT
    # space. Copy them before the rescale below, or every non-native leg saves
    # images whose overlays are scaled wrong against a correct frame.
    rescale_to_native = (stream.det_w, stream.det_h) != (stream.frame_w, stream.frame_h)
    debug_boxes = ([dict(b) for b in boxes]
                   if rescale_to_native and save_frames_enabled(cfg) else boxes)
    if rescale_to_native:
        # Independent per-axis ratios, because the scale to the shared detector
        # geometry is a STRETCH, not a letterbox. That is worth stating: the
        # videoscale element defaults to add-borders=true, so letterboxing looks
        # like the safe assumption - but the output caps here pin only width and
        # height and leave pixel-aspect-ratio free, so videoscale distorts the
        # pixels instead of adding bars. Verified by dumping a detector-input
        # frame for a 640x480 source in a 1280x720 detector: it filled the full
        # 1280 width with no black columns at all. Correcting for bars that do
        # not exist shifts every box sideways, so this stays a plain ratio.
        sx = stream.frame_w / float(stream.det_w)
        sy = stream.frame_h / float(stream.det_h)
        for b in boxes:
            b["x1"] *= sx; b["x2"] *= sx
            b["y1"] *= sy; b["y2"] *= sy

    stream.processed += 1
    warming_up = stream.processed <= cfg.warmup_frames
    if not warming_up:
        metadata_start = time_ms()
        send_metadata(stream, sample, boxes)
        metadata_end = time_ms()
        if save_frames_enabled(cfg):
            frame = take_debug_frame(stream, int(getattr(sample, "frame_id", -1) or -1))
            if frame is not None:
                maybe_save_debug_frame(cfg, stream, frame, debug_boxes)
        stream.profile.add(detection_pull_ms, metadata_end - metadata_start, len(boxes))


def debug_frame_output_name(stream_index: int) -> str:
    return f"debug_frame_{stream_index}"


# Decoded frames held per stream while their detection catches up. Inference is
# a few frames behind the decoder; past that the detection is never coming.
DEBUG_FRAME_CACHE = 16


def drain_debug_frames(app: AppRuntime, cfg: AppConfig) -> None:
    """Collect decoded frames KEYED BY FRAME ID.

    The decoded and detector branches run on independent realtime queues, so the
    newest decoded frame is not the one the next detection describes. Keeping only
    the newest paired every saved image with whatever had arrived by then: correct
    geometry, wrong moment, which reads as drift on anything moving. Frames are
    held by id until their detection turns up or ages out.
    """
    if not save_frames_enabled(cfg):
        return
    for stream in app.streams:
        output_name = debug_frame_output_name(stream.index)
        while True:
            sample = app.run.pull(output_name, 0)
            if sample is None:
                break
            tensor = first_tensor_from_sample(sample)
            if tensor is None:
                continue
            frame_id = int(getattr(sample, "frame_id", -1) or -1)
            stream.debug_frames[frame_id] = tensor_bgr_from_decoded(tensor)
            while len(stream.debug_frames) > DEBUG_FRAME_CACHE:
                stream.debug_frames.pop(next(iter(stream.debug_frames)))


def take_debug_frame(stream: StreamRuntime, frame_id: int):
    """The decoded frame this detection came from, or None if nothing is held.

    A match also clears everything older: those frames' detections have already
    gone past, so they can never be claimed.
    """
    frame = stream.debug_frames.pop(frame_id, None)
    if frame is not None:
        for stale in [fid for fid in stream.debug_frames if fid < frame_id]:
            del stream.debug_frames[stale]
        return frame
    if not stream.debug_frames:
        return None
    # Nothing carries this id. Rather than stop saving images altogether, fall
    # back to the newest frame - the old behaviour - but say so once, because
    # those images are the approximate pairing this function exists to avoid.
    if not stream.debug_pairing_warned:
        stream.debug_pairing_warned = True
        print(f"[warn] stream {stream.index} debug frames carry no id matching "
              f"their detections; saved images fall back to the newest frame",
              file=sys.stderr)
    return stream.debug_frames.pop(next(reversed(stream.debug_frames)))


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
        running_fn = getattr(app.run, "running", None)
        if callable(running_fn) and not running_fn():
            # The shared graph is gone, so no later pull can succeed. Mark every
            # stream closed and let the run loop end: an empty pull was otherwise
            # indistinguishable from a timeout, leaving the process alive and
            # spinning on an output that was never coming back.
            print("[warn] detection output closed; ending run", file=sys.stderr)
            global _output_closed_unexpectedly
            _output_closed_unexpectedly = True
            for stream in app.streams:
                stream.closed = True
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

    labels = load_labels(cfg.labels_path)
    # Probe every source first: the shared detector needs ONE input geometry, so
    # pick the largest native size and scale the other detector legs to it.
    # Largest (rather than smallest) avoids discarding detail small objects need.
    # Video delivery is unaffected - it stays native per stream.
    # Probe concurrently: probes are I/O-bound RTSP handshakes, so N of them
    # collapse from N*t to ~t instead of running serially.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(16, len(cfg.rtsp_urls))) as pool:
        probes = list(pool.map(probe_rtsp, cfg.rtsp_urls))
    det_w = max(p[0] for p in probes)
    det_h = max(p[1] for p in probes)
    # ...but never ask the scaler for more than MAX_UPSCALE on the SMALLEST
    # source. The hardware scaler silently produces nothing past ~2x upscale:
    # measured with a 640x480 source, scaling to 1280x720 (2x) detected at the
    # full 30/s, while 1920x1080 (3x) and 2560x1440 (4x) both yielded exactly
    # zero detections - no error, no warning, just a dead detector leg while
    # video kept flowing. Clamping to what the scaler can actually do is what
    # lets a small source (640x480) share a detector with a large one (2K/4K).
    # Downscaling is unconstrained, so the large sources simply scale down.
    max_w, max_h = det_w, det_h
    min_w = min(p[0] for p in probes)
    min_h = min(p[1] for p in probes)
    capped_w = min(max_w, MAX_UPSCALE * min_w)
    capped_h = min(max_h, MAX_UPSCALE * min_h)
    # Keep the clamp at or above the model's own input so it never becomes the
    # accuracy bottleneck - but never above the largest source either, or a
    # single small stream would be upscaled for no reason (a lone 640x480 must
    # stay 640x480 and skip the scaler entirely). NV12 needs even dimensions.
    capped_w = min(max_w, max(MODEL_INPUT, capped_w))
    capped_h = min(max_h, max(MODEL_INPUT, capped_h))
    det_w = capped_w - (capped_w % 2)
    det_h = capped_h - (capped_h % 2)
    if len({(p[0], p[1]) for p in probes}) > 1:
        note = ""
        if (det_w, det_h) != (max(p[0] for p in probes), max(p[1] for p in probes)):
            note = f" (clamped to <={MAX_UPSCALE}x the smallest source, {min_w}x{min_h})"
        print(f"[detector] mixed source sizes {sorted({(p[0], p[1]) for p in probes})} "
              f"-> normalising detector input to {det_w}x{det_h}{note} "
              f"(video stays native per stream)", flush=True)

    model, detector_graph = build_detector_graph(cfg, det_w, det_h)
    detections_graph = build_detections_graph()
    app = AppRuntime(graph=pyneat.Graph(), run=None, model=model, streams=[])

    for index, url in enumerate(cfg.rtsp_urls):
        stream = build_stream_runtime(cfg, index, url, labels, det_w, det_h, probes[index])
        app.streams.append(stream)
        connect_stream_graph(app, cfg, stream, detector_graph)
    app.graph.connect(detector_graph, detections_graph)

    # SIGTERM is the NORMAL stop signal here: every pipelines/ panel and CLI
    # `down` sends it (see stop_app/stop_group). Without a handler the default
    # disposition terminates the process outright, so app.run.close() never runs
    # and the decoder/CVU pools it would have released stay allocated in the
    # reserved region - the exact failure stop_app's docstring warns about,
    # while the caller sees the PID vanish and reports a clean stop.
    global _stop_requested, _output_closed_unexpectedly
    _stop_requested = False
    _output_closed_unexpectedly = False
    previous_sigterm = signal.signal(signal.SIGTERM, _request_stop)

    try:
        if cfg.profile:
            print(f"Backend:\n{app.graph.describe_backend()}")
        app.run = app.graph.build(build_run_options(len(app.streams)))
        # Printed ONLY after the shared graph builds. Each stream's "] rtsp="
        # banner is emitted by build_stream_runtime() well before this, so a
        # build failure left every banner on disk and pipelines/'s
        # wait_for_streams() reported a successful run over a detector that had
        # already exited. That check now requires this line too.
        print(f"[app] graph running: {len(app.streams)} stream(s)", flush=True)
        while not _stop_requested and not all_streams_done(app.streams, cfg.frames):
            process_run_once(app, cfg, "detections")
        # A continuous run has no natural end, so reaching here without a stop
        # request means the output went away underneath us. Exiting 0 would tell
        # a supervisor the experiment succeeded.
        if _output_closed_unexpectedly and not _stop_requested and cfg.frames <= 0:
            raise RuntimeError(
                "detection output closed unexpectedly; the run produced no further "
                "metadata"
            )
    except KeyboardInterrupt:
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
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
