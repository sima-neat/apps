"""Single-camera RTSP/MJPEG instance segmentation Insight example using pyneat.

Runs one YOLO26 or YOLOv8 segmentation package over the same graph, host decode,
overlay, and Insight metadata path. ``model.family`` selects the decode boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_LABELS = DEFAULT_CONFIG.parent / "coco_label.txt"
MASK_ALPHA = 0.55
MASK_THRESHOLD = 0.50
DEFAULT_INPUT_SIZE = 640

#: Model families this application decodes. YOLO26 decodes on the MLA through the packaged
#: BoxDecode route; YOLOv8 surfaces raw heads that are decoded here. Both produce the same
#: detection records, so everything after ``decode_segments`` is family independent.
YOLO26 = "yolo26"
YOLOV8 = "yolov8"

# MetadataSender rejects a payload above 65507 bytes, and pyneat raises on the rejection. Half of
# that leaves room for the envelope and keeps the datagram count low enough for Insight to
# reassemble within its 250 ms window.
METADATA_BYTE_BUDGET = 32768
#: Both families emit masks at one quarter of the model input per dimension, so a 160x160
#: mask grid corresponds to a 640x640 input.
MASK_STRIDE = 4
#: BoxDecode returns YOLO26 masks on a fixed grid, independent of the model input size.
YOLO26_MASK_GRID = 160
#: YOLOv8 head contract: three feature levels of box, class, and mask-coefficient tensors,
#: followed by the mask prototypes.
YOLOV8_HEAD_TENSORS = 10
YOLOV8_STRIDES = (8, 16, 32)
#: Distribution-focal-loss bins per box side, and prototype/coefficient depth.
DFL_BINS = 16
MASK_COEFFICIENTS = 32

cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class OutputConfig:
    save_dir: str
    save_every: int
    mask_alpha: float
    mask_threshold: float
    draw_boxes: bool


@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    source_url: str
    model_family: str = YOLO26
    input_size: int = DEFAULT_INPUT_SIZE
    source_type: str = "rtsp"
    source_codec: str = "h264"
    latency_ms: int = 200
    tcp: bool = True
    source_fps: int = 0
    ssl_strict: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    output: OutputConfig = OutputConfig("", 0, MASK_ALPHA, MASK_THRESHOLD, True)
    # Which key supplied source_url: "source.url" or the legacy "source.rtsp_url".
    source_key: str = "source.url"


@dataclass
class PipelineRuntime:
    model: object
    graph: object
    run: object
    metadata_sender: object
    labels: list[str]
    #: Run output the loop pulls: the segments alone, or the frame-joined bundle when saving.
    output_name: str
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int
    #: Separate decoded-frame output, used when frames are paired by this application instead
    #: of by a graph-side join. Empty when the graph joins them.
    frame_output_name: str = ""
    #: Host copies of recent decoded frames, oldest first, waiting to be paired with their
    #: segments.
    frames: list = field(default_factory=list)


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.instances = 0
        self.dropped_segments = 0
        self.start_ms = 0.0
        self.pull_ms = 0.0
        self.decode_ms = 0.0
        self.metadata_ms = 0.0

    def add(
        self,
        pull_ms: float,
        decode_ms: float,
        metadata_ms: float,
        instance_count: int,
        dropped: int,
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.instances += instance_count
        self.dropped_segments += dropped
        self.pull_ms += pull_ms
        self.decode_ms += decode_ms
        self.metadata_ms += metadata_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed_ms = max(time_ms() - self.start_ms, 1e-6)
        frames = float(self.frames)
        print(
            f"[profile] frames={self.frames} "
            f"output_fps={self.frames * 1000.0 / elapsed_ms} "
            f"avg_pull_ms={self.pull_ms / frames} "
            f"avg_decode_ms={self.decode_ms / frames} "
            f"avg_metadata_ms={self.metadata_ms / frames} "
            f"avg_instances={self.instances / frames} "
            f"dropped_segments={self.dropped_segments}",
            flush=True,
        )
        self.reset()


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
        description="Single-camera RTSP/MJPEG instance segmentation Insight example"
    )
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
    if isinstance(value, str) and value.strip():
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str) and value.strip():
        return float(value)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return bool(value)


def parse_model_family(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"yolo26", "yolo-26", "yolov26"}:
        return YOLO26
    if lowered in {"yolov8", "yolo-v8", "yolo_v8"}:
        return YOLOV8
    raise ValueError("model.family must be yolo26 or yolov8")


def parse_source_type(value: str) -> str:
    lowered = value.lower()
    if lowered in {"rtsp", "http", "https"}:
        return "http" if lowered == "https" else lowered
    raise ValueError("source.type must be rtsp or http")


def parse_source_codec(value: str) -> str:
    lowered = value.lower()
    if lowered in {"h264", "avc", "h.264"}:
        return "h264"
    if lowered in {"h265", "hevc", "h.265"}:
        return "h265"
    if lowered in {"mjpeg", "jpeg"}:
        return "mjpeg"
    raise ValueError("source.codec must be h264/avc, h265/hevc, or mjpeg")


def validate_config(cfg: AppConfig) -> None:
    if not cfg.source_url:
        raise ValueError("source.url or source.rtsp_url must be set")
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if cfg.input_size <= 0 or cfg.input_size % (MASK_STRIDE * max(YOLOV8_STRIDES)) != 0:
        raise ValueError(
            "model.input_size must be a positive multiple of "
            f"{MASK_STRIDE * max(YOLOV8_STRIDES)}"
        )
    # Path("") is ".", so an empty value in the file arrives here as ".".
    if str(cfg.labels_path) in ("", "."):
        raise ValueError("model.labels must be set")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.source_fps < 0:
        raise ValueError("source.fps must be >= 0")
    if cfg.source_type == "http" and cfg.source_codec != "mjpeg":
        raise ValueError("source.codec must be mjpeg for source.type=http")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")
    if cfg.output.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.output.mask_alpha <= 1.0:
        raise ValueError("output.mask_alpha must be between 0 and 1")
    if not 0.0 <= cfg.output.mask_threshold <= 1.0:
        raise ValueError("output.mask_threshold must be between 0 and 1")


def load_app_config(config_path: Path) -> AppConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model = section(raw, "model")
    source = section(raw, "source")
    inference = section(raw, "inference")
    runtime = section(raw, "runtime")
    output = section(raw, "output")
    insight = section(output, "insight")

    labels_path = string_or(model, "labels", str(DEFAULT_LABELS))
    # config.yaml documents source.rtsp_url as the fallback "when source.url
    # is empty", so an empty value must fall through, not just an absent key.
    # The key that supplied the URL is kept so --validate-config-only can report
    # the choice without echoing the URL, which can carry credentials.
    source_url = string_or(source, "url")
    source_key = "source.url" if source_url else "source.rtsp_url"
    source_url = source_url or string_or(source, "rtsp_url")

    cfg = AppConfig(
        model_path=string_or(model, "path"),
        labels_path=Path(labels_path),
        source_url=source_url,
        source_key=source_key,
        model_family=parse_model_family(string_or(model, "family", YOLO26)),
        input_size=int_or(model, "input_size", DEFAULT_INPUT_SIZE),
        source_type=parse_source_type(string_or(source, "type", "rtsp")),
        source_codec=parse_source_codec(string_or(source, "codec", "h264")),
        latency_ms=int_or(source, "latency_ms", 200),
        tcp=bool_or(source, "tcp", True),
        source_fps=int_or(source, "fps", 0),
        ssl_strict=bool_or(source, "ssl_strict", True),
        frames=int_or(inference, "frames", 0),
        min_score=float_or(inference, "min_score", 0.55),
        nms_iou=float_or(inference, "nms_iou", 0.60),
        max_detections=int_or(inference, "max_detections", 50),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=string_or(insight, "host"),
        video_port=int_or(insight, "video_port", 9000),
        metadata_port=int_or(insight, "metadata_port", 9100),
        output=OutputConfig(
            save_dir=string_or(output, "save_dir"),
            save_every=int_or(output, "save_every", 0),
            mask_alpha=float_or(output, "mask_alpha", MASK_ALPHA),
            mask_threshold=float_or(output, "mask_threshold", MASK_THRESHOLD),
            draw_boxes=bool_or(output, "draw_boxes", True),
        ),
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


def tensor_to_numpy(tensor) -> object:
    return np.asarray(tensor.to_numpy(copy=True))


def numpy_dtype(tensor) -> object:
    dtypes = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    dtype = dtypes.get(tensor.dtype)
    if dtype is None:
        raise RuntimeError(f"unsupported tensor dtype: {tensor.dtype}")
    return dtype


def tensor_to_hwc_f32(tensor) -> object:
    """One head tensor as a dense HWC float32 array, dropping a leading batch axis of 1."""
    shape = tuple(int(dim) for dim in tensor.shape)
    array = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=numpy_dtype(tensor))
    array = array.reshape(shape).astype(np.float32)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise RuntimeError(f"only batch size 1 is supported, got {array.shape[0]}")
        array = array[0]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected head tensor rank {array.ndim}")
    return array


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

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

    frame = tensor_to_numpy(tensor)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"unexpected decoded tensor shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def extract_tensors(sample) -> list:
    if isinstance(sample, (list, tuple)) and all(hasattr(item, "to_numpy") for item in sample):
        return list(sample)
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


def find_field(sample, label: str):
    if sample is None:
        return None
    if getattr(sample, "stream_label", "") == label:
        return sample
    for field in getattr(sample, "fields", []):
        found = find_field(field, label)
        if found is not None:
            return found
    return None


def joined_field(sample, label: str, bundle_index: int):
    field = find_field(sample, label)
    fields = list(getattr(sample, "fields", []))
    if field is not None:
        return field
    if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle and len(fields) > bundle_index:
        return fields[bundle_index]
    raise RuntimeError(f"joined output missing {label} field")


#: How many decoded frames may wait for their segments. The segments branch trails the frame
#: branch by the model and host-decode latency, so this only has to cover that lag. The ring
#: holds host copies, so a retained frame costs memory, not a pipeline buffer.
FRAME_RING_CAPACITY = 16


def host_frame_copy(sample):
    """Copies a decoded frame out of the pipeline so the pulled sample can be released at once.

    A pulled sample keeps a loan on the decoder buffer behind it even when output memory is
    owned, and the decoder stalls once its few in-flight frames are all on loan. The ring
    therefore keeps pixels, never samples. NV12 stays NV12 until a frame is actually saved.
    """
    tensor = frame_tensor_from_sample(sample)
    if not tensor.is_nv12():
        return tensor_bgr_from_decoded(tensor)
    width = tensor_dim(tensor, "width")
    height = tensor_dim(tensor, "height")
    payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
    expected = width * height * 3 // 2
    if payload.size < expected:
        raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
    return payload[:expected].reshape((height * 3 // 2, width))


def bgr_from_host_frame(frame):
    """The BGR picture for a host_frame_copy() result."""
    if frame.ndim == 2:
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12))
    return frame


def drain_frames(runtime) -> None:
    """Moves every frame the run has ready into the ring, dropping the oldest past capacity.

    Draining every iteration is what keeps the frame output queue from backing up.
    """
    while True:
        frame = runtime.run.pull(runtime.frame_output_name, 0)
        if frame is None:
            return
        runtime.frames.append((frame.frame_id, host_frame_copy(frame)))
        if len(runtime.frames) > FRAME_RING_CAPACITY:
            runtime.frames.pop(0)


def pull_segments(runtime, timeout_ms: int):
    """Waits for the next segments sample while keeping the frame branch drained.

    A single long blocking pull would leave the frame output queue unattended, and the frames
    discarded there under the keep-latest policy are exactly the partners a saved frame needs,
    so the wait is split into short slices with a drain between them.
    """
    if not runtime.frame_output_name:
        return runtime.run.pull(runtime.output_name, timeout_ms)
    slice_ms = 20
    deadline = time_ms() + timeout_ms
    while True:
        drain_frames(runtime)
        sample = runtime.run.pull(runtime.output_name, slice_ms)
        if sample is not None or time_ms() >= deadline:
            return sample


def frame_for(runtime, frame_id: int):
    """The retained frame a segments sample was computed from, or None when it has aged out."""
    if frame_id < 0:
        return None
    for retained_id, frame in reversed(runtime.frames):
        if retained_id == frame_id:
            return frame
    return None


def frame_tensor_from_sample(sample):
    # A graph-joined bundle carries the frame in a field; a separate frame output is the sample.
    field_sample = (
        joined_field(sample, "frame", 0)
        if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle
        else sample
    )
    tensors = extract_tensors(field_sample)
    if not tensors:
        raise RuntimeError("joined frame field has no tensor")
    return tensors[0]


def segment_tensors_from_sample(sample) -> list:
    # Without save_dir there is nothing to combine, so the pulled sample is the segments payload.
    field = (
        joined_field(sample, "segments", 1)
        if getattr(sample, "kind", None) == pyneat.SampleKind.Bundle
        else sample
    )
    tensors = extract_tensors(field)
    if not tensors:
        raise RuntimeError("segments field has no tensors")
    return tensors


def detection(x1: float, y1: float, x2: float, y2: float, score: float, class_id: int, mask):
    """One decoded instance, in the single representation both families produce.

    ``x1..y2`` are frame pixels, ``score`` is a probability, and ``mask`` is a uint8 mask grid
    covering the letterboxed model input at ``MASK_STRIDE`` cells per pixel. Everything
    downstream of the decoders - overlays, polygons, Insight metadata - reads only these keys.
    """
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "score": float(score),
        "class_id": int(class_id),
        "mask": mask,
    }


def decode_yolo26_segments(tensors: list, frame_w: int, frame_h: int, max_detections: int):
    """YOLO26 boundary: the MLA already ran BoxDecode, so this only unpacks its payload."""
    decoded = pyneat.decode_segmentation(
        tensors,
        clamp_to=(frame_w, frame_h),
        top_k=max_detections,
        strict=False,
    )
    detections = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32)
        masks = tensor_to_numpy(item.masks).astype(np.uint8)
        mask_shape = (-1, YOLO26_MASK_GRID, YOLO26_MASK_GRID)
        for row, mask in zip(boxes.reshape((-1, 6)), masks.reshape(mask_shape)):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(detection(x1, y1, x2, y2, score, int(class_id), mask))
            if len(detections) >= max_detections:
                return detections
    return detections


def letterbox_params(frame_w: int, frame_h: int, input_size: int) -> tuple[float, float, float]:
    """Scale and padding the preprocessor applies when it letterboxes a frame."""
    scale = min(input_size / frame_w, input_size / frame_h)
    return scale, (input_size - frame_w * scale) * 0.5, (input_size - frame_h * scale) * 0.5


def split_yolov8_heads(heads: list, input_size: int):
    """Group the packaged YOLOv8 head tensors and check them against ``model.input_size``."""
    if len(heads) < YOLOV8_HEAD_TENSORS:
        raise RuntimeError(
            f"YOLOv8 decode expects {YOLOV8_HEAD_TENSORS} head tensors, got {len(heads)}"
        )
    boxes, scores, coefficients, proto = heads[0:3], heads[3:6], heads[6:9], heads[9]
    if proto.ndim != 3 or proto.shape[2] != MASK_COEFFICIENTS:
        raise RuntimeError(f"unexpected prototype tensor shape {proto.shape}")
    if proto.shape[0] != proto.shape[1] or proto.shape[0] * MASK_STRIDE != input_size:
        raise RuntimeError(
            f"prototype grid {proto.shape[0]}x{proto.shape[1]} does not match "
            f"model.input_size {input_size}"
        )
    for stride, box, score, coefficient in zip(YOLOV8_STRIDES, boxes, scores, coefficients):
        grid = input_size // stride
        if box.shape[:2] != (grid, grid) or box.shape[2] != 4 * DFL_BINS:
            raise RuntimeError(
                f"unexpected box head shape {box.shape} for stride {stride} at "
                f"model.input_size {input_size}"
            )
        if score.shape[:2] != (grid, grid) or score.shape[2] <= 0:
            raise RuntimeError(f"unexpected class head shape {score.shape}")
        if coefficient.shape[:2] != (grid, grid) or coefficient.shape[2] != MASK_COEFFICIENTS:
            raise RuntimeError(f"unexpected mask-coefficient head shape {coefficient.shape}")
    return boxes, scores, coefficients, proto


def yolov8_candidates(boxes, scores, coefficients, input_size: int, min_score: float):
    """Boxes in letterboxed model pixels, class probabilities, and mask coefficients."""
    bins = np.arange(DFL_BINS, dtype=np.float32)
    threshold = np.float32(min_score)
    level_boxes, level_scores, level_classes, level_coefficients = [], [], [], []
    for box, score, coefficient in zip(boxes, scores, coefficients):
        stride = np.float32(input_size / box.shape[0])
        # The packaged class head already carries probabilities, so it is thresholded as is.
        class_ids = score.argmax(axis=2)
        best = np.take_along_axis(score, class_ids[..., None], axis=2)[..., 0]
        rows, columns = np.nonzero(best >= threshold)
        if rows.size == 0:
            continue
        logits = box[rows, columns].reshape((-1, 4, DFL_BINS)).astype(np.float32)
        weights = np.exp(logits - logits.max(axis=2, keepdims=True))
        distance = (weights @ bins) / weights.sum(axis=2) * stride
        center_x = (columns.astype(np.float32) + np.float32(0.5)) * stride
        center_y = (rows.astype(np.float32) + np.float32(0.5)) * stride
        level_box = np.clip(
            np.stack(
                [
                    center_x - distance[:, 0],
                    center_y - distance[:, 1],
                    center_x + distance[:, 2],
                    center_y + distance[:, 3],
                ],
                axis=1,
            ),
            np.float32(0.0),
            np.float32(input_size),
        )
        # Degenerate cells cannot become instances, and dropping them here keeps the
        # candidate list identical to the C++ decoder's.
        valid = (level_box[:, 2] > level_box[:, 0]) & (level_box[:, 3] > level_box[:, 1])
        level_boxes.append(level_box[valid])
        level_scores.append(best[rows, columns].astype(np.float32)[valid])
        level_classes.append(class_ids[rows, columns].astype(np.int32)[valid])
        level_coefficients.append(coefficient[rows, columns].astype(np.float32)[valid])

    if not level_boxes:
        empty_boxes = np.zeros((0, 4), dtype=np.float32)
        return (
            empty_boxes,
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0, MASK_COEFFICIENTS), dtype=np.float32),
        )
    return (
        np.concatenate(level_boxes),
        np.concatenate(level_scores),
        np.concatenate(level_classes),
        np.concatenate(level_coefficients),
    )


def nms_per_class(boxes, scores, classes, nms_iou: float, max_detections: int):
    """Greedy per-class NMS, highest score first, capped at ``max_detections``."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.intp)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    # Stable descending, so candidates tying on score are kept in the order both
    # implementations visit their cells in.
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        best = order[0]
        keep.append(best)
        rest = order[1:]
        if rest.size == 0:
            break
        overlap_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        intersection = overlap_w * overlap_h
        union = areas[best] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[(classes[rest] != classes[best]) | (iou <= nms_iou)]
    return np.asarray(keep, dtype=np.intp)


def yolov8_instance_mask(proto, coefficients, frame_rect, frame_shape):
    """Prototype mask for one instance, on the same grid BoxDecode returns for YOLO26."""
    mask = np.zeros(proto.shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = mask_rect_for_frame_rect(frame_rect, frame_shape, proto.shape[:2])
    activated = 255.0 / (1.0 + np.exp(-(proto[y0:y1, x0:x1] @ coefficients)))
    # Rounded, not truncated, so the C++ saturate_cast path produces identical mask bytes.
    mask[y0:y1, x0:x1] = np.clip(np.rint(activated), 0.0, 255.0).astype(np.uint8)
    return mask


def decode_yolov8_segments(
    heads: list,
    frame_w: int,
    frame_h: int,
    input_size: int,
    min_score: float,
    nms_iou: float,
    max_detections: int,
):
    """YOLOv8 boundary: decode raw heads here into the shared detection representation."""
    boxes, scores, coefficients, proto = split_yolov8_heads(heads, input_size)
    boxes, scores, classes, coefficients = yolov8_candidates(
        boxes, scores, coefficients, input_size, min_score
    )

    scale, pad_x, pad_y = letterbox_params(frame_w, frame_h, input_size)
    frame_shape = (frame_h, frame_w, 3)
    padding = np.array([pad_x, pad_y, pad_x, pad_y], dtype=np.float64)
    limit = np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float64)
    # Single precision throughout, so the C++ implementation decodes the same coordinates.
    frame_boxes = np.clip((boxes - padding) / scale, 0.0, limit).astype(np.float32)
    # A prediction lying entirely in the letterbox padding has no frame to occupy, so it is
    # dropped before the cap. Discarding it afterwards would let it consume one of the
    # max_detections slots and push out a valid lower-ranked detection.
    inside = (frame_boxes[:, 2] > frame_boxes[:, 0]) & (frame_boxes[:, 3] > frame_boxes[:, 1])
    frame_boxes, scores = frame_boxes[inside], scores[inside]
    classes, coefficients = classes[inside], coefficients[inside]

    keep = nms_per_class(frame_boxes, scores, classes, nms_iou, max_detections)
    detections = []
    for index in keep:
        x1, y1, x2, y2 = (float(value) for value in frame_boxes[index])
        candidate = detection(x1, y1, x2, y2, scores[index], int(classes[index]), None)
        frame_rect = frame_rect_for_detection(candidate, frame_shape)
        candidate["mask"] = yolov8_instance_mask(
            proto, coefficients[index], frame_rect, frame_shape
        )
        detections.append(candidate)
    return detections


def decode_segments(cfg: AppConfig, tensors: list, frame_w: int, frame_h: int):
    """The one place model family changes behavior."""
    if cfg.model_family == YOLO26:
        return decode_yolo26_segments(tensors, frame_w, frame_h, cfg.max_detections)
    return decode_yolov8_segments(
        [tensor_to_hwc_f32(tensor) for tensor in tensors[:YOLOV8_HEAD_TENSORS]],
        frame_w,
        frame_h,
        cfg.input_size,
        cfg.min_score,
        cfg.nms_iou,
        cfg.max_detections,
    )


def fps_from_rate(value: str) -> int:
    if not value or value in {"0/0", "0/1"}:
        return 0
    try:
        fps = float(Fraction(value)) if "/" in value else float(value)
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0


def int_from_probe(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def probe_ffprobe(cfg: AppConfig) -> tuple[int, int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rw_timeout",
        "5000000",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "default=nw=1",
    ]
    if cfg.source_type == "rtsp" and cfg.tcp:
        cmd.extend(["-rtsp_transport", "tcp"])
    if not cfg.ssl_strict:
        cmd.extend(["-tls_verify", "0"])
    cmd.append(cfg.source_url)
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0, 0, 0
    if result.returncode != 0:
        return 0, 0, 0
    values = {}
    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )
    return int_from_probe(values.get("width")), int_from_probe(values.get("height")), fps


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
    return width, height, fps


def set_output_caps(caps, fps: int, width: int, height: int) -> None:
    if width <= 0 or height <= 0 or fps <= 0:
        return
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any


def make_rtsp_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.source_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.codec = (
        pyneat.RtspCodec.H264
        if cfg.source_codec == "h264"
        else pyneat.RtspCodec.H265
        if cfg.source_codec == "h265"
        else pyneat.RtspCodec.MJPEG
    )
    opt.source_fps = fps
    if cfg.source_codec == "h264":
        opt.auto_caps_from_stream = True
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
    elif cfg.source_codec == "h265":
        opt.auto_caps_from_stream = True
        opt.dec_width = width
        opt.dec_height = height
    else:
        opt.dec_width = width
        opt.dec_height = height
    set_output_caps(opt.output_caps, fps, width, height)
    return opt


def make_http_mjpeg_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = cfg.source_url
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.source_fps = fps
    opt.ssl_strict = cfg.ssl_strict
    set_output_caps(opt.output_caps, fps, width, height)
    return opt


def make_source_graph(cfg: AppConfig, fps: int, width: int, height: int):
    if cfg.source_type == "rtsp":
        return pyneat.groups.rtsp_decoded_input(make_rtsp_source_options(cfg, fps, width, height))
    return pyneat.groups.http_mjpeg_decoded_input(
        make_http_mjpeg_source_options(cfg, fps, width, height)
    )


def require_mjpeg_fps(cfg: AppConfig, fps: int) -> None:
    if cfg.source_codec == "mjpeg" and fps <= 0:
        raise RuntimeError(
            "MJPEG source did not provide a valid frame rate; set source.fps or use a source "
            "with probeable FPS metadata"
        )


def probe_decoded_source(cfg: AppConfig, fps: int) -> tuple[int, int, int]:
    graph = pyneat.Graph("source_probe")
    graph.add(make_source_graph(cfg, fps, 0, 0))
    graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(1)))

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)
    try:
        sample = run.pull("frame", 20000)
    finally:
        run.close()
    if sample is None:
        raise RuntimeError("failed to probe decoded source frame")
    tensors = extract_tensors(sample)
    if not tensors:
        raise RuntimeError("decoded source probe did not produce a tensor")
    return tensor_dim(tensors[0], "width"), tensor_dim(tensors[0], "height"), fps


def resolve_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    probed_w, probed_h, probed_fps = probe_ffprobe(cfg)
    fps = cfg.source_fps if cfg.source_fps > 0 else probed_fps
    if cfg.source_type == "rtsp":
        width, height = probed_w, probed_h
        if width <= 0 or height <= 0 or fps <= 0:
            rtsp_w, rtsp_h, rtsp_fps = probe_rtsp(cfg.source_url)
            width = width if width > 0 else rtsp_w
            height = height if height > 0 else rtsp_h
            fps = fps if fps > 0 else rtsp_fps
        require_mjpeg_fps(cfg, fps)
        return width, height, fps

    require_mjpeg_fps(cfg, fps)
    if probed_w > 0 and probed_h > 0:
        return probed_w, probed_h, fps
    width, height, _ = probe_decoded_source(cfg, fps)
    return width, height, fps


def make_model(cfg: AppConfig, frame_w: int, frame_h: int):
    """Load the segmentation package. Only the postprocess contract differs per family."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    # The decoder emits NV12; the model preprocess letterboxes it onto the model input.
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    if frame_w > 0 and frame_h > 0:
        opt.preprocess.input_max_width = frame_w
        opt.preprocess.input_max_height = frame_h
    if cfg.model_family == YOLO26:
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        # BoxDecode runs on device and emits the segmentation payload this app unpacks.
        opt.decode_type = pyneat.BoxDecodeType.YoloV26Seg
        opt.score_threshold = cfg.min_score
        opt.nms_iou_threshold = cfg.nms_iou
        opt.top_k = cfg.max_detections
    else:
        # YOLOv8 leaves decode_type unset so the route surfaces the raw float heads that
        # decode_yolov8_segments() consumes. The decode inverts a letterbox, so the resize
        # policy is requested rather than assumed, and the YOLO normalization is requested
        # explicitly: without it the model sees unnormalized pixels and its class scores
        # collapse to a fraction of their proper range.
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.preprocess.resize.enable = pyneat.AutoFlag.On
        opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    return pyneat.Model(cfg.model_path, opt)


def build_metadata_sender(cfg: AppConfig):
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.channel = 0
    options.metadata_port_base = cfg.metadata_port
    return pyneat.MetadataSender(options)


def build_video_graph(cfg: AppConfig, width: int, height: int, fps: int):
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
    sender_opt.host = cfg.insight_host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.video_port
    sender_opt.encoder.bitrate_kbps = 1000

    graph = pyneat.Graph("video")
    graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(sender_opt))
    return graph, sender_opt.video_port


def build_pipeline(cfg: AppConfig) -> PipelineRuntime:
    frame_w, frame_h, fps = resolve_source_geometry(cfg)
    if fps <= 0:
        raise RuntimeError(
            "failed to resolve source frame rate; set source.fps or use a source with "
            "probeable FPS metadata"
        )
    model = make_model(cfg, frame_w, frame_h)
    labels = load_labels(cfg.labels_path)

    video_graph, video_port = build_video_graph(cfg, frame_w, frame_h, fps)

    # Insight correlates the RTP timestamp with the metadata timestamp, so the encoder and the
    # segments must stay in one Run and therefore on one GStreamer timeline.
    save_frames = bool(cfg.output.save_dir)
    source = make_source_graph(cfg, fps, frame_w, frame_h)
    branch = pyneat.graphs.branch(
        "source", ["video", "model", "frame"] if save_frames else ["video", "model"]
    )

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)

    # The YOLO26 decode is a cheap payload unpack, so its output can queue frames. The YOLOv8
    # decode runs on the host and cannot match the source rate; queueing there pins the whole
    # detess stage output pool and starves it, so that output keeps only the newest sample.
    segments_graph = pyneat.Graph("segments")
    segments_options = (
        pyneat.OutputOptions.every_frame(4)
        if cfg.model_family == YOLO26
        else pyneat.OutputOptions.every_frame(1)
    )
    segments_graph.add(pyneat.nodes.output("segments", segments_options))

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, video_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, segments_graph)
    frame_output_name = ""
    output_name = "segments"
    if save_frames:
        frame_graph = pyneat.Graph("frame")
        frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
        graph.connect(branch, frame_graph)
        if cfg.model_family == YOLO26:
            joined = pyneat.graphs.combine(
                ["frame", "segments"], "segmentation_output", pyneat.CombinePolicy.ByFrame
            )
            graph.connect(frame_graph, joined)
            graph.connect(segments_graph, joined)
            output_name = "segmentation_output"
        else:
            # The graph-side join retains more model-output buffers than a YOLOv8 package's
            # fixed pool can serve, so this route publishes both streams and pairs them on
            # frame_id in the run loop.
            frame_output_name = "frame"
    if cfg.profile:
        print(f"Backend:\n{graph.describe_backend()}")

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    # Measured on Modalix: the YOLOv8 head tensors must be copied out of the detess stage pool
    # at pull time, or the stage starves while the host decode runs. YOLO26 pulls one small
    # BoxDecode payload and keeps the cheaper zero-copy path.
    run_options.output_memory = (
        pyneat.OutputMemory.ZeroCopy if cfg.model_family == YOLO26 else pyneat.OutputMemory.Owned
    )
    run = graph.build(run_options)

    metadata_sender = build_metadata_sender(cfg)
    print(
        f"source={cfg.source_url} type={cfg.source_type} codec={cfg.source_codec} "
        f"model={cfg.model_family} stream={frame_w}x{frame_h}@{fps} "
        f"insight={cfg.insight_host} video={video_port} "
        f"metadata={metadata_sender.metadata_port()} channel=0"
    )
    return PipelineRuntime(
        model=model,
        graph=graph,
        run=run,
        metadata_sender=metadata_sender,
        labels=labels,
        output_name=output_name,
        frame_output_name=frame_output_name,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=fps,
        video_port=video_port,
    )


def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (255, 194, 0),
        (168, 153, 44),
    ]
    return palette[max(0, class_id) % len(palette)]


def frame_rect_for_detection(det: dict, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x0 = max(0, min(frame_w - 1, int(np.floor(float(det["x1"])))))
    y0 = max(0, min(frame_h - 1, int(np.floor(float(det["y1"])))))
    x1 = max(x0 + 1, min(frame_w, int(np.ceil(float(det["x2"])))))
    y1 = max(y0 + 1, min(frame_h, int(np.ceil(float(det["y2"])))))
    return x0, y0, x1, y1


def mask_rect_for_frame_rect(
    frame_rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    mask_h, mask_w = mask_shape[:2]
    model_w = mask_w * MASK_STRIDE
    model_h = mask_h * MASK_STRIDE
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) * 0.5
    pad_y = (model_h - frame_h * scale) * 0.5

    def to_mask_x(frame_x: float) -> float:
        return (frame_x * scale + pad_x) * mask_w / model_w

    def to_mask_y(frame_y: float) -> float:
        return (frame_y * scale + pad_y) * mask_h / model_h

    fx0, fy0, fx1, fy1 = frame_rect
    x0 = max(0, min(mask_w - 1, int(np.floor(to_mask_x(fx0)))))
    y0 = max(0, min(mask_h - 1, int(np.floor(to_mask_y(fy0)))))
    x1 = max(x0 + 1, min(mask_w, int(np.ceil(to_mask_x(fx1)))))
    y1 = max(y0 + 1, min(mask_h, int(np.ceil(to_mask_y(fy1)))))
    return x0, y0, x1, y1


def project_letterbox_mask_roi(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
):
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape, mask.shape)
    return cv2.resize(mask[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def draw_box(frame, det: dict, labels: list[str]) -> None:
    cls_id = int(det["class_id"])
    color = class_color(cls_id)
    x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        frame,
        f"{class_name(labels, cls_id)} {float(det['score']):.2f}",
        (x0, max(0, y0 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def overlay_segmentation(
    frame,
    detections: list[dict],
    min_score: float,
    cfg: OutputConfig,
    labels: list[str],
):
    annotated = frame.copy()
    for det in detections:
        if float(det["score"]) < min_score:
            continue
        mask = det.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        threshold = cfg.mask_threshold * 255.0
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
        roi_mask = project_letterbox_mask_roi(mask, frame.shape, (x0, y0, x1, y1))
        _, binary_mask = cv2.threshold(roi_mask, threshold, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(binary_mask) > 0:
            color = class_color(int(det["class_id"]))
            roi = annotated[y0:y1, x0:x1]
            mask_color = np.full(roi.shape, color, dtype=np.uint8)
            blended = cv2.addWeighted(roi, 1.0 - cfg.mask_alpha, mask_color, cfg.mask_alpha, 0.0)
            cv2.copyTo(blended, binary_mask, roi)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, color, 2, cv2.LINE_8)
        if cfg.draw_boxes:
            draw_box(annotated, det, labels)
    return annotated


def mask_polygon(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
    threshold: float,
) -> list[list[int]]:
    """Frame-absolute silhouette of ``mask`` inside ``frame_rect``.

    Empty when the thresholded mask holds nothing Insight can draw. ``threshold`` is a fraction of
    full scale, as ``output.mask_threshold`` is. Upscaling before thresholding is what makes the
    outline match the rendered overlay.
    """
    roi = project_letterbox_mask_roi(mask, frame_shape, frame_rect)
    _, binary = cv2.threshold(roi, threshold * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    # Contour points lie inside frame_rect, which is already clamped to the frame, so shifting them
    # into frame space cannot leave the image.
    x0, y0 = frame_rect[0], frame_rect[1]
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]


def metadata_segments(
    detections: list[dict],
    labels: list[str],
    frame_shape: tuple[int, ...],
    mask_threshold: float,
) -> list[dict]:
    segments = []
    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame_shape)
        polygon = mask_polygon(mask, frame_shape, (x0, y0, x1, y1), mask_threshold)
        if not polygon:
            continue
        segments.append(
            {
                "id": f"seg_{len(segments) + 1}",
                "label": class_name(labels, int(det["class_id"])),
                "confidence": float(det["score"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "mask_format": "polygon",
                "mask": polygon,
            }
        )
    return segments


def encode_segments(segments: list[dict]) -> tuple[str, int]:
    """``data`` object of a ``segmentation`` metadata message, and how many segments were dropped.

    Segments that do not fit the byte budget are dropped lowest-confidence first.
    """
    ordered = sorted(segments, key=lambda segment: segment["confidence"], reverse=True)
    kept = []
    total = len('{"segments":[]}')
    for segment in ordered:
        entry_bytes = len(json.dumps(segment, separators=(",", ":"))) + 1
        if total + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total += entry_bytes
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":")), len(ordered) - len(kept)


def send_metadata(
    runtime: PipelineRuntime, sample, detections: list[dict], mask_threshold: float
) -> int:
    """Sends one ``segmentation`` message and reports how many segments the byte budget dropped."""
    data_json, dropped = encode_segments(
        metadata_segments(
            detections, runtime.labels, (runtime.frame_h, runtime.frame_w), mask_threshold
        )
    )
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    if not runtime.metadata_sender.send_metadata(
        "segmentation", data_json, timestamp_ms, frame_id
    ):
        print("[warn] insight metadata send failed", file=sys.stderr)
    return dropped


def save_due(cfg: AppConfig, processed: int) -> bool:
    """Whether this result is due an annotated frame."""
    return bool(cfg.output.save_dir) and cfg.output.save_every > 0 and (
        processed % cfg.output.save_every == 0
    )


def save_frame(
    cfg: AppConfig, processed: int, frame, detections: list[dict], labels: list[str]
) -> bool:
    """Writes one annotated BGR frame. Returns False when the decoded frame it needs is gone.

    The YOLO26 route joins frames to results inside the graph and always has its partner. The
    YOLOv8 route pairs them in the run loop, and a source faster than the model makes the two
    branches retain different frames, so some results have no picture to annotate. Those are
    counted and reported rather than silently skipped.
    """
    if frame is None:
        return False
    annotated = overlay_segmentation(frame, detections, cfg.min_score, cfg.output, labels)
    out_path = Path(cfg.output.save_dir) / f"frame_{processed}.jpg"
    if not cv2.imwrite(str(out_path), annotated):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)
        return False
    return True


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


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    dropped_total = 0
    saved = 0
    unpaired = 0
    while cfg.frames <= 0 or processed < cfg.frames:
        pull_start = time_ms()
        sample = pull_segments(runtime, 20000)
        pull_end = time_ms()
        if not pull_result_has_sample(runtime.run, sample, runtime.output_name):
            print("[warn] timed out waiting for segmentation output", file=sys.stderr)
            continue

        decode_start = time_ms()
        detections = decode_segments(
            cfg, segment_tensors_from_sample(sample), runtime.frame_w, runtime.frame_h
        )
        decode_end = time_ms()

        metadata_start = time_ms()
        dropped = send_metadata(runtime, sample, detections, cfg.output.mask_threshold)
        metadata_end = time_ms()
        if dropped > 0 and dropped_total == 0:
            print(
                f"[warn] metadata byte budget exceeded, dropped {dropped} segments",
                file=sys.stderr,
            )
        dropped_total += dropped

        processed += 1
        if save_due(cfg, processed):
            if runtime.frame_output_name:
                drain_frames(runtime)
                retained = frame_for(runtime, sample.frame_id)
                frame = None if retained is None else bgr_from_host_frame(retained)
            else:
                frame = tensor_bgr_from_decoded(frame_tensor_from_sample(sample))
            if save_frame(cfg, processed, frame, detections, runtime.labels):
                saved += 1
            else:
                unpaired += 1
        profile.add(
            pull_end - pull_start,
            decode_end - decode_start,
            metadata_end - metadata_start,
            len(detections),
            dropped,
        )

    profile.flush()
    print(
        f"processed={processed} dropped_segments={dropped_total} "
        f"saved={saved} unpaired={unpaired} "
        f"video_sender={cfg.insight_host}:{runtime.video_port}"
    )
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            # Reports which key supplied the source, not its value: a URL can
            # carry credentials and this line ends up in terminal and CI logs.
            print(f"Config validated: {args.config} (source={cfg.source_key})")
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
            os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
        if cfg.output.save_dir:
            Path(cfg.output.save_dir).mkdir(parents=True, exist_ok=True)

        runtime = build_pipeline(cfg)
        try:
            return 0 if run_pipeline(runtime, cfg) > 0 else 3
        finally:
            runtime.run.close()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
