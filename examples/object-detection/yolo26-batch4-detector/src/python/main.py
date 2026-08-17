"""Batch-4 YOLO26 object detection over four Insight RTSP streams.

One frame is taken from each stream and the four are submitted to the MLA as a
single `[4, 640, 640, 3]` batch, so four streams cost one dispatch instead of
four. The six YOLO26 heads come back with a leading batch axis, are decoded per
lane on the CPU, and each stream's detections are published to Insight on its
own channel.

Flow: RTSP ingest -> decode to NV12 -> letterbox into a batch lane ->
one MLA dispatch -> per-lane head decode -> Insight video + metadata.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import glob
import json
from pathlib import Path
import sys
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
REPO_ROOT = Path(__file__).resolve().parents[5]

# YOLO26 emits three levels, each as a bbox head and a class head.
LEVELS = 3
EXPECTED_OUTPUTS = 2 * LEVELS
BBOX_CHANNELS = 4
CLASS_CHANNELS = 80
PAD_VALUE = 114
MAX_STREAMS = 4

cv2 = None
np = None
pyneat = None


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    rtsp_urls: list[str]
    tcp: bool = True
    latency_ms: int = 100
    frames: int = 0
    score_threshold: float = 0.35
    max_detections: int = 100
    insight_host: str = ""
    video_port_base: int = 9000
    metadata_port_base: int = 9100
    video_enabled: bool = True
    debug_dir: str = ""
    save_every: int = 0
    timeout_ms: int = 20000
    warmup_frames: int = 10
    profile: bool = False
    profile_interval: int = 50


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


def resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (REPO_ROOT / path)


def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path or cfg.model_path.startswith("<"):
        raise ValueError("model.path must be set and must not be a placeholder")
    if not cfg.rtsp_urls:
        raise ValueError("streams must list at least one RTSP URL")
    if len(cfg.rtsp_urls) > MAX_STREAMS:
        raise ValueError(f"this example supports up to {MAX_STREAMS} streams")
    if any(not url or url.startswith("<") for url in cfg.rtsp_urls):
        raise ValueError("streams still contains a placeholder URL")
    if not cfg.insight_host or cfg.insight_host.startswith("<"):
        raise ValueError("output.insight.host must be set")
    if not 0.0 < cfg.score_threshold < 1.0:
        raise ValueError("inference.score_threshold must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.timeout_ms <= 0:
        raise ValueError("runtime.timeout_ms must be > 0")
    if cfg.warmup_frames < 0:
        raise ValueError("runtime.warmup_frames must be >= 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if cfg.video_port_base <= 0:
        raise ValueError("output.insight.video_port_base must be > 0")
    if cfg.metadata_port_base <= 0:
        raise ValueError("output.insight.metadata_port_base must be > 0")


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "input")
    inference = section(raw, "inference")
    output = section(raw, "output")
    insight = section(output, "insight")
    runtime = section(raw, "runtime")

    streams = raw.get("streams") or []
    if not isinstance(streams, list) or not all(isinstance(url, str) for url in streams):
        raise ValueError("streams must be a list of RTSP URLs")

    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"
    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=resolve(string_or(model, "labels", str(default_labels))),
        rtsp_urls=list(streams),
        tcp=bool_or(source, "tcp", True),
        latency_ms=int_or(source, "latency_ms", 100),
        frames=int_or(inference, "frames", 0),
        score_threshold=float_or(inference, "score_threshold", 0.35),
        max_detections=int_or(inference, "max_detections", 100),
        insight_host=string_or(insight, "host"),
        video_port_base=int_or(insight, "video_port_base", 9000),
        metadata_port_base=int_or(insight, "metadata_port_base", 9100),
        video_enabled=bool_or(output, "video_enabled", True),
        debug_dir=string_or(output, "debug_dir"),
        save_every=int_or(output, "save_every", 0),
        timeout_ms=int_or(runtime, "timeout_ms", 20000),
        warmup_frames=int_or(runtime, "warmup_frames", 10),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 50),
    )
    validate_config(cfg)
    return cfg


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def nv12_to_rgb(tensor, out):
    """Decoded NV12 straight to RGB in a caller-owned buffer.

    RGB is what both the MLA input and the Insight video sender want, so the
    frame is converted once here and never touched again.
    """
    width = tensor_dim(tensor, "width")
    height = tensor_dim(tensor, "height")
    payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size < expected:
        raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
    nv12 = payload[:expected].reshape((height * 3 // 2, width))
    cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12, dst=out)
    return out


def letterbox_into(rgb, lane, net: int):
    """Aspect-preserving resize into one batch lane, pad 114, centered.

    The scaled pixels are written directly into the lane as normalized float32,
    so the batch tensor is filled in place instead of being stacked from four
    freshly allocated frames.
    """
    height, width = rgb.shape[:2]
    scale = min(net / width, net / height)
    new_w, new_h = round(width * scale), round(height * scale)
    dx, dy = (net - new_w) // 2, (net - new_h) // 2
    lane.fill(PAD_VALUE / 255.0)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    np.multiply(
        resized,
        np.float32(1.0 / 255.0),
        out=lane[dy:dy + new_h, dx:dx + new_w],
        dtype=np.float32,
    )
    return scale, dx, dy


# --------------------------------------------------------------------------
# postprocessing
# --------------------------------------------------------------------------
def heads_from_outputs(arrays: list, lane: int) -> dict:
    """Six `[N, H, W, C]` outputs -> `{head_name: [H, W, C]}` for one lane.

    Heads are matched by shape, not by output order: 4 channels is a bbox head,
    80 a class head, and the grid size orders the three levels. Sorted by
    descending grid so level 0 is the 80x80 map, matching the strides
    `decode_heads` derives from the grid size.
    """
    bbox, cls = [], []
    for array in arrays:
        if array.ndim != 4:
            raise RuntimeError(f"expected [N,H,W,C] outputs, got {array.shape}")
        if lane >= array.shape[0]:
            raise RuntimeError(f"lane {lane} out of range for output {array.shape}")
        plane = array[lane]
        channels = plane.shape[2]
        if channels == BBOX_CHANNELS:
            bbox.append(plane)
        elif channels == CLASS_CHANNELS:
            cls.append(plane)
        else:
            raise RuntimeError(
                f"output with {channels} channels is neither a bbox head "
                f"({BBOX_CHANNELS}) nor a class head ({CLASS_CHANNELS})"
            )
    if len(bbox) != LEVELS or len(cls) != LEVELS:
        raise RuntimeError(
            f"expected {LEVELS} bbox and {LEVELS} class heads, got {len(bbox)} and {len(cls)}"
        )
    bbox.sort(key=lambda plane: -plane.shape[0])
    cls.sort(key=lambda plane: -plane.shape[0])

    heads = {}
    for level in range(LEVELS):
        if bbox[level].shape[:2] != cls[level].shape[:2]:
            raise RuntimeError(
                f"level {level} bbox grid {bbox[level].shape[:2]} does not match "
                f"class grid {cls[level].shape[:2]}"
            )
        heads[f"bbox_{level}"] = bbox[level]
        heads[f"class_logit_{level}"] = cls[level]
    return heads


def decode_heads(heads: dict, net: int, score_threshold: float, max_detections: int) -> list[dict]:
    """YOLO26 reg_max=1: raw l/t/r/b distances plus class logits. No NMS.

    The one-to-one head is already deduplicated, so detections are only ranked
    by score and capped.

    The class heads hold 672 000 logits per frame, so the decode is dominated by
    whatever runs across all of them. Sigmoid is monotonic, which means a cell
    clears a probability threshold exactly when its logit clears the matching
    logit threshold. Thresholding the raw logits keeps the exponential and the
    per-cell argmax on the handful of surviving cells instead of the whole
    tensor, and the box arithmetic is done once per level as array operations.
    """
    logit_threshold = float(np.log(score_threshold / (1.0 - score_threshold)))
    detections = []
    for level in range(LEVELS):
        box = heads[f"bbox_{level}"]
        cls = heads[f"class_logit_{level}"]
        stride = net / box.shape[0]

        best_logit = cls.max(-1)
        rows, cols = np.nonzero(best_logit >= logit_threshold)
        if rows.size == 0:
            continue

        scores = 1.0 / (1.0 + np.exp(-np.clip(best_logit[rows, cols], -30.0, 30.0)))
        class_ids = cls[rows, cols].argmax(-1)
        ltrb = box[rows, cols]
        anchor_x = cols + 0.5
        anchor_y = rows + 0.5
        x1 = (anchor_x - ltrb[:, 0]) * stride
        y1 = (anchor_y - ltrb[:, 1]) * stride
        x2 = (anchor_x + ltrb[:, 2]) * stride
        y2 = (anchor_y + ltrb[:, 3]) * stride

        detections.extend(
            {
                "score": float(s),
                "class_id": int(c),
                "x1": float(a),
                "y1": float(b),
                "x2": float(cc),
                "y2": float(dd),
            }
            for s, c, a, b, cc, dd in zip(
                scores.tolist(), class_ids.tolist(),
                x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(),
            )
        )
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections[:max_detections]


def to_original(detections: list[dict], scale: float, dx: int, dy: int, width: int, height: int):
    """Undo the letterbox so boxes are in source-frame coordinates."""
    mapped = []
    for det in detections:
        x1 = min(max((det["x1"] - dx) / scale, 0.0), width - 1.0)
        y1 = min(max((det["y1"] - dy) / scale, 0.0), height - 1.0)
        x2 = min(max((det["x2"] - dx) / scale, 0.0), width - 1.0)
        y2 = min(max((det["y2"] - dy) / scale, 0.0), height - 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        mapped.append({**det, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return mapped


# --------------------------------------------------------------------------
# runtime plumbing
# --------------------------------------------------------------------------
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

    cv2, np, pyneat = cv2_module, np_module, pyneat_module


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def first_tensor(sample):
    if sample is None:
        return None
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.kind == pyneat.SampleKind.TensorSet and sample.tensors:
        return sample.tensors[0]
    for nested in sample.fields:
        tensor = first_tensor(nested)
        if tensor is not None:
            return tensor
    return None


def pull_latest(run, output_name: str, timeout_ms: int):
    """Block for one frame, then drain so the caller always gets the newest."""
    sample = run.pull(output_name, timeout_ms)
    if sample is None:
        return None
    while True:
        newer = run.pull(output_name, 0)
        if newer is None:
            return sample
        sample = newer


def probe_rtsp(url: str) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(capture.get(cv2.CAP_PROP_FPS) or 0))
    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to probe RTSP frame size: {url}")
    return width, height, fps if fps > 0 else 30


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------
def build_source_options(cfg: AppConfig, url: str, width: int, height: int, fps: int):
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
    opt.output_caps.width = width
    opt.output_caps.height = height
    opt.output_caps.fps = fps
    opt.output_caps.memory = pyneat.CapsMemory.Any
    return opt


def build_encoded_source_graph(index: int, opt):
    """RTSP pulled as encoded H.264, decoded by the graph below."""
    encoded = pyneat.RtspEncodedInputOptions()
    encoded.url = opt.url
    encoded.codec = pyneat.RtspCodec.H264
    encoded.latency_ms = opt.latency_ms
    encoded.tcp = opt.tcp
    encoded.source_fps = opt.source_fps
    encoded.fallback_h264_width = opt.fallback_h264_width
    encoded.fallback_h264_height = opt.fallback_h264_height

    source = pyneat.Graph(f"rtsp_source_{index}")
    source.add(pyneat.groups.rtsp_encoded_input(encoded))
    return source


def encoded_input_options():
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = pyneat.Format.H264
    opt.memory_policy = pyneat.InputMemoryPolicy.Ev74
    return opt


def build_decode_graph(index: int, opt):
    """Hardware decode to NV12, published as this stream's `frame_N` output.

    `OutputOptions.latest()` keeps the newest frame rather than a backlog, so a
    dispatch always batches current pictures.
    """
    decode = pyneat.Graph(f"decode_{index}")
    dec = pyneat.SimaDecodeOptions()
    dec.type = pyneat.SimaDecodeType.H264
    dec.sima_allocator_type = opt.sima_allocator_type
    dec.out_format = pyneat.Format.NV12
    dec.decoder_name = opt.decoder_name
    dec.raw_output = opt.decoder_raw_output
    dec.dec_width = opt.fallback_h264_width
    dec.dec_height = opt.fallback_h264_height
    dec.dec_fps = opt.source_fps
    decode.connect(
        pyneat.nodes.input("decode_h264", encoded_input_options()),
        pyneat.nodes.sima_decode(dec),
    )
    decode.add(
        pyneat.nodes.caps_raw(
            "NV12",
            opt.output_caps.width,
            opt.output_caps.height,
            opt.output_caps.fps,
            opt.output_caps.memory,
        )
    )
    decode.add(pyneat.nodes.output(f"frame_{index}", pyneat.OutputOptions.latest()))
    return decode


def realtime_link(index: int, queue_depth: int = 3):
    link = pyneat.GraphLinkOptions()
    link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    link.queue_depth = queue_depth
    link.stream_id = f"stream{index}"
    return link


def build_run_options():
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return run_options


def h264_video_input_options():
    opt = pyneat.InputOptions()
    opt.payload_type = pyneat.PayloadType.Encoded
    opt.format = pyneat.Format.H264
    opt.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
    return opt


def build_video_sender_graph(cfg: AppConfig, index: int):
    """Forward the source H.264 to Insight without re-encoding it.

    The stream arrives already compressed, so the cheap thing to do is hand
    those bytes straight to the sender. Decoding, converting to RGB and running
    a hardware encoder per channel to rebuild an equivalent stream costs four
    encoders and a full-frame copy per channel, and it was the largest single
    item in the publish path.

    The trade is that the picture reaches the viewer as soon as it arrives while
    its detections only exist an inference later, so overlays lag the video
    slightly and refresh at the dispatch rate rather than the source rate. The
    metadata carries the source `pts_ns`/`frame_id`, so a viewer that aligns on
    those can still pair them up.
    """
    options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.H264)
    options.host = cfg.insight_host
    options.channel = index
    options.video_port_base = cfg.video_port_base
    options.async_ = True

    video = pyneat.Graph(f"video_sender_{index}")
    video.connect(
        pyneat.nodes.input("video_h264", h264_video_input_options()),
        pyneat.groups.video_sender(options),
    )
    return video, options.video_port


# --------------------------------------------------------------------------
# streams and batching
# --------------------------------------------------------------------------
@dataclass
class Stream:
    index: int
    url: str
    frame_w: int
    frame_h: int
    fps: int
    metadata_sender: object
    video_port: int = 0
    processed: int = 0


@dataclass(frozen=True)
class FrameRef:
    """Metadata retained after the decoder-backed Sample is released."""
    pts_ns: int
    frame_id: int


@dataclass
class BatchSlot:
    """One reusable batch: the MLA input plus what the lanes came from.

    Two of these rotate between ingest and synchronous MLA/decode/publish.
    Keeping those phases on separate slots lets ingest overlap processing.
    """
    tensor: object
    rgb: list = field(default_factory=list)
    geometry: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    ready: bool = False


def make_batch_slot(batch_size: int, net: int, streams: list[Stream]) -> BatchSlot:
    return BatchSlot(
        tensor=np.empty((batch_size, net, net, 3), dtype=np.float32),
        rgb=[np.empty((s.frame_h, s.frame_w, 3), dtype=np.uint8) for s in streams],
        geometry=[(1.0, 0, 0)] * len(streams),
        samples=[None] * len(streams),
    )


class BatchPrefetcher:
    """Keeps one batch in flight so ingest overlaps inference.

    A frame pull blocks until the source produces its next picture — about
    33 ms at 30 fps — and the letterbox costs another few ms per lane. Doing
    that between dispatches would serialize it with the MLA and the decode.
    Instead the lanes for the next batch are filled on worker threads while the
    caller works on the current one, and both cv2 and the pull release the GIL.
    """

    def __init__(self, run, streams: list[Stream], net: int, batch_size: int, timeout_ms: int):
        self.run = run
        self.streams = streams
        self.net = net
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.closed = False
        self.lane_pool = ThreadPoolExecutor(max_workers=max(1, len(streams)))
        self.fill_pool = ThreadPoolExecutor(max_workers=1)
        self.slots = [make_batch_slot(batch_size, net, streams) for _ in range(2)]
        self.slot = 0
        self.pending = self.fill_pool.submit(self._fill, self.slots[0])

    def next(self) -> BatchSlot | None:
        """Hand back the filled batch and immediately start filling the other."""
        slot = self.pending.result()
        self.slot = (self.slot + 1) % len(self.slots)
        self.pending = self.fill_pool.submit(self._fill, self.slots[self.slot])
        return slot if slot.ready else None

    def _fill(self, slot: BatchSlot) -> BatchSlot:
        results = list(self.lane_pool.map(lambda s: self._fill_lane(slot, s), self.streams))
        slot.ready = all(results)
        # Fewer streams than the compiled batch: repeat the last lane so the MLA
        # still runs exactly one dispatch.
        for lane in range(len(self.streams), self.batch_size):
            slot.tensor[lane] = slot.tensor[len(self.streams) - 1]
        return slot

    def _fill_lane(self, slot: BatchSlot, stream: Stream) -> bool:
        if self.closed:
            return False
        sample = pull_latest(self.run, f"frame_{stream.index}", self.timeout_ms)
        if sample is None:
            return False
        tensor = first_tensor(sample)
        if tensor is None:
            return False
        rgb = nv12_to_rgb(tensor, slot.rgb[stream.index])
        slot.geometry[stream.index] = letterbox_into(rgb, slot.tensor[stream.index], self.net)
        # Do not retain the zero-copy decoder Sample across inference. Four
        # pipeline slots would otherwise pin enough decoder buffers to starve
        # the live graph. Insight publication only needs these scalar fields.
        slot.samples[stream.index] = FrameRef(
            pts_ns=int(getattr(sample, "pts_ns", -1)),
            frame_id=int(getattr(sample, "frame_id", -1)),
        )
        return True

    def close(self) -> None:
        """Let the in-flight fill finish before the caller closes the Run.

        Its lane threads are inside `run.pull`, so tearing the Run down first
        would pull the runtime out from under them.
        """
        self.closed = True
        try:
            self.pending.result(timeout=self.timeout_ms / 1000.0 + 1.0)
        except Exception:
            pass
        self.fill_pool.shutdown(wait=True, cancel_futures=True)
        self.lane_pool.shutdown(wait=True, cancel_futures=True)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def send_metadata(stream: Stream, sample, detections: list[dict], labels: list[str]) -> None:
    objects = []
    for index, det in enumerate(detections, start=1):
        class_id = det["class_id"]
        objects.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": det["score"],
                "bbox": [
                    det["x1"],
                    det["y1"],
                    det["x2"] - det["x1"],
                    det["y2"] - det["y1"],
                ],
            }
        )
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""

    message = {
        "type": "object-detection",
        "data": {"objects": objects},
        "timestamp": timestamp_ms,
        "frame_id": frame_id,
    }
    if sample.pts_ns >= 0:
        message["_insight"] = {
            "rtp_timestamp": ((sample.pts_ns * 90_000) // 1_000_000_000)
            & 0xFFFFFFFF
        }
    stream.metadata_sender.send_raw_json(json.dumps(message, separators=(",", ":")))


def save_debug_frame(cfg: AppConfig, stream: Stream, rgb, detections, labels) -> None:
    if not cfg.debug_dir or cfg.save_every <= 0 or stream.processed % cfg.save_every != 0:
        return
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for det in detections:
        p1 = (int(round(det["x1"])), int(round(det["y1"])))
        p2 = (int(round(det["x2"])), int(round(det["y2"])))
        cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
        class_id = det["class_id"]
        label = labels[class_id] if 0 <= class_id < len(labels) else "unknown"
        cv2.putText(
            annotated,
            f"{label} {det['score']:.2f}",
            (p1[0], max(0, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    out_path = Path(cfg.debug_dir) / f"stream_{stream.index}_frame_{stream.processed}.jpg"
    if not cv2.imwrite(str(out_path), annotated):
        print(f"[warn] failed to write {out_path}", file=sys.stderr)


def publish_stream(cfg: AppConfig, slot: BatchSlot, labels: list[str],
                   stream: Stream, detections: list[dict]) -> None:
    """Send one stream's detections for the exact analysed frame."""
    sample = slot.samples[stream.index]
    rgb = slot.rgb[stream.index]
    send_metadata(stream, sample, detections, labels)
    save_debug_frame(cfg, stream, rgb, detections, labels)


class ProfileWindow:
    """Throughput and a phase breakdown, printed every `interval` dispatches."""

    def __init__(self, cfg: AppConfig, streams: int) -> None:
        self.enabled = cfg.profile
        self.interval = cfg.profile_interval
        self.streams = streams
        self.reset()

    def reset(self) -> None:
        self.dispatches = 0
        self.batch_ms = 0.0
        self.infer_ms = 0.0
        self.decode_ms = 0.0
        self.publish_ms = 0.0
        self.detections = 0
        self.start_ms = 0.0

    def add(self, batch_ms, infer_ms, decode_ms, publish_ms, detections) -> None:
        if not self.enabled:
            return
        if self.dispatches == 0:
            self.start_ms = time_ms()
        self.dispatches += 1
        self.batch_ms += batch_ms
        self.infer_ms += infer_ms
        self.decode_ms += decode_ms
        self.publish_ms += publish_ms
        self.detections += detections
        if self.dispatches >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.dispatches == 0:
            return
        elapsed = time_ms() - self.start_ms
        dispatch_fps = self.dispatches * 1000.0 / elapsed if elapsed > 0 else 0.0
        n = self.dispatches
        print(
            f"[profile] dispatch_fps={dispatch_fps:.2f} "
            f"per_stream_fps={dispatch_fps:.2f} "
            f"aggregate_fps={dispatch_fps * self.streams:.2f} | "
            f"avg_batch_ms={self.batch_ms / n:.1f} "
            f"avg_infer_ms={self.infer_ms / n:.1f} "
            f"avg_decode_ms={self.decode_ms / n:.1f} "
            f"avg_publish_ms={self.publish_ms / n:.1f} | "
            f"avg_detections={self.detections / n:.1f}",
            flush=True,
        )
        self.reset()


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
def load_model(cfg: AppConfig) -> tuple[object, int, int]:
    """Load the batched pack and refuse anything that is not a six-head batch model."""
    model = pyneat.Model(str(resolve(cfg.model_path)))
    shape = [int(x) for x in model.input_specs()[0].shape]
    if len(shape) != 4:
        raise RuntimeError(
            f"this example needs a batched model, got input shape {shape}. "
            "Compile the detector with --batch_size 4."
        )
    batch_size, net = shape[0], shape[1]
    if batch_size != MAX_STREAMS:
        raise RuntimeError(
            f"model batch size is {batch_size}; this example requires batch size {MAX_STREAMS}"
        )
    outputs = model.output_specs()
    if len(outputs) != EXPECTED_OUTPUTS:
        raise RuntimeError(
            f"expected {EXPECTED_OUTPUTS} YOLO26 head tensors, model has {len(outputs)}"
        )
    print(
        f"Model loaded: batch={batch_size} net={net} "
        f"heads={[list(spec.shape) for spec in outputs]}",
        flush=True,
    )
    return model, batch_size, net


def build_streams(cfg: AppConfig, graph) -> list[Stream]:
    streams: list[Stream] = []
    for index, url in enumerate(cfg.rtsp_urls):
        width, height, fps = probe_rtsp(url)

        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = index
        metadata_options.metadata_port_base = cfg.metadata_port_base
        metadata_sender = pyneat.MetadataSender(metadata_options)

        options = build_source_options(cfg, url, width, height, fps)
        source = build_encoded_source_graph(index, options)
        decoder = build_decode_graph(index, options)

        video_port = 0
        if cfg.video_enabled:
            # Split the encoded stream: one copy is decoded for detection, the
            # other goes to Insight untouched.
            branch = pyneat.graphs.branch(f"encoded_{index}", ["decode_h264", "video_h264"])
            graph.connect(source, branch)
            graph.connect(branch, decoder, realtime_link(index))
            sender, video_port = build_video_sender_graph(cfg, index)
            graph.connect(branch, sender, realtime_link(index))
        else:
            graph.connect(source, decoder, realtime_link(index))

        stream = Stream(
            index=index,
            url=url,
            frame_w=width,
            frame_h=height,
            fps=fps,
            metadata_sender=metadata_sender,
            video_port=video_port,
        )
        print(
            f"[stream {index}] rtsp={url} {width}x{height}@{fps} "
            f"video={stream.video_port if cfg.video_enabled else 'disabled'} "
            f"metadata={metadata_sender.metadata_port()}",
            flush=True,
        )
        streams.append(stream)
    return streams


def run_app(cfg: AppConfig) -> None:
    labels = load_labels(cfg.labels_path)
    if cfg.debug_dir:
        Path(cfg.debug_dir).mkdir(parents=True, exist_ok=True)

    model, batch_size, net = load_model(cfg)

    graph = pyneat.Graph()
    streams = build_streams(cfg, graph)
    run = graph.build(build_run_options())
    profile = ProfileWindow(cfg, len(streams))
    prefetcher = BatchPrefetcher(run, streams, net, batch_size, cfg.timeout_ms)
    decode_pool = ThreadPoolExecutor(max_workers=len(streams))

    def decode_lane(stream: Stream, head_arrays: list, slot: BatchSlot):
        """One lane's heads -> detections in that stream's source coordinates."""
        heads = heads_from_outputs(head_arrays, stream.index)
        detections = decode_heads(heads, net, cfg.score_threshold, cfg.max_detections)
        scale, dx, dy = slot.geometry[stream.index]
        return to_original(detections, scale, dx, dy, stream.frame_w, stream.frame_h)

    try:
        while cfg.frames <= 0 or min(s.processed for s in streams) < cfg.frames:
            batch_started = time_ms()
            batch = prefetcher.next()
            batch_ms = time_ms() - batch_started
            if batch is None:
                last_error = run.last_error()
                if last_error:
                    raise RuntimeError(f"runtime error: {last_error}")
                raise RuntimeError("timed out waiting for frames")

            input_tensor = pyneat.Tensor.from_numpy(batch.tensor, copy=False)
            infer_started = time_ms()
            head_tensors = model.run([input_tensor], timeout_ms=cfg.timeout_ms)
            if len(head_tensors) != EXPECTED_OUTPUTS:
                raise RuntimeError(
                    f"expected {EXPECTED_OUTPUTS} head tensors, got {len(head_tensors)}"
                )
            head_arrays = [tensor.to_numpy() for tensor in head_tensors]
            infer_ms = time_ms() - infer_started

            decode_started = time_ms()
            per_stream = list(
                decode_pool.map(lambda s: decode_lane(s, head_arrays, batch), streams)
            )
            decoded = time_ms()

            found = 0
            publish: list[tuple[Stream, list]] = []
            for stream, detections in zip(streams, per_stream):
                found += len(detections)
                stream.processed += 1
                if stream.processed > cfg.warmup_frames:
                    publish.append((stream, detections))
            # Each stream has its own sender, so the encodes proceed in parallel.
            list(decode_pool.map(lambda item: publish_stream(cfg, batch, labels, *item), publish))

            profile.add(
                batch_ms, infer_ms, decoded - decode_started,
                time_ms() - decoded, found,
            )
    except KeyboardInterrupt:
        raise
    finally:
        profile.flush()
        decode_pool.shutdown(wait=False, cancel_futures=True)
        prefetcher.close()
        run.close()
        for stream in streams:
            print(f"[stream {stream.index}] processed={stream.processed}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-4 YOLO26 detector over four RTSP streams")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if not args.config.is_file():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 2
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            print(
                f"Config validated: {args.config} (streams={len(cfg.rtsp_urls)}, "
                f"score_threshold={cfg.score_threshold}, max_detections={cfg.max_detections})"
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
