"""Single-camera RTSP YOLO26 segmentation Insight example using pyneat.

This mirrors the intent of the C++ reference sample in the same folder:

- pull one decoded RTSP stream
- run one YOLO26 instance segmentation model
- publish H.264 video with mask overlays plus metadata to Insight

The implementation keeps those responsibilities loosely separated so the main
runtime path is easy to reason about:

1. RTSP probe/build
2. YOLO26 segmentation inference and mask extraction
3. Insight video/metadata publishing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import sys
import time
from pathlib import Path
import yaml

DEFAULT_FPS = 30
SOURCE_RUN_QUEUE_DEPTH = 4
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_LABELS = DEFAULT_CONFIG.parent / "coco_label.txt"
MASK_ALPHA = 0.55
MASK_THRESHOLD = 0.5
cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class ModelConfig:
    path: str
    labels: str


@dataclass(frozen=True)
class SourceConfig:
    rtsp_url: str
    latency_ms: int
    tcp: bool


@dataclass(frozen=True)
class InferenceConfig:
    frames: int
    min_score: float
    nms_iou: float
    max_detections: int


@dataclass(frozen=True)
class RuntimeConfig:
    profile: bool
    profile_interval: int


@dataclass(frozen=True)
class InsightConfig:
    host: str
    video_port: int
    metadata_port: int


@dataclass(frozen=True)
class OutputConfig:
    save_dir: str
    save_every: int
    mask_alpha: float
    mask_threshold: float
    draw_boxes: bool


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    source: SourceConfig
    inference: InferenceConfig
    runtime: RuntimeConfig
    insight: InsightConfig
    output: OutputConfig


def load_runtime_dependencies() -> None:
    """Import runtime-only dependencies after argparse validation."""
    global cv2, np, pyneat
    if pyneat is not None:
        return

    # Prefer system OpenCV (built with GStreamer) when running inside a venv.
    for p in glob.glob("/usr/lib/python3*/dist-packages"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module


def tensor_to_numpy(tensor: pyneat.Tensor) -> np.ndarray:
    """Copy a pyneat tensor into a NumPy array owned by Python."""
    return np.asarray(tensor.to_numpy(copy=True))


def tensor_dim(tensor: pyneat.Tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def tensor_bgr_from_decoded(tensor: pyneat.Tensor) -> np.ndarray:
    """Normalize decoded output into a writable HWC uint8 BGR frame."""
    if tensor.is_nv12():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise ValueError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))

    if tensor.is_i420():
        width = tensor_dim(tensor, "width")
        height = tensor_dim(tensor, "height")
        payload = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise ValueError(f"I420 payload too small: {payload.size} < {expected}")
        i420 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420))

    arr = tensor_to_numpy(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected decoded tensor shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def tensor_from_bgr_frame(frame: np.ndarray) -> pyneat.Tensor:
    """Create an EV74 BGR image tensor for model-managed YOLO inference."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
        memory=pyneat.TensorMemory.EV74,
    )


def tensor_from_rgb_frame(frame: np.ndarray) -> pyneat.Tensor:
    """Create an EV74 RGB image tensor for VideoSender input."""
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def extract_tensors(result) -> list:
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return list(result)

    if not is_sample_like(result):
        return []

    tensors = []
    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            tensors.extend(current.tensors)
            continue
        if current.kind == pyneat.SampleKind.Tensor and current.tensor is not None:
            tensors.append(current.tensor)
    return tensors


def segmentation_results_from_output(result, width: int, height: int, max_detections: int) -> list[dict]:
    tensors = extract_tensors(result)
    if not tensors:
        raise RuntimeError("model returned no segmentation tensors")

    decoded = pyneat.decode_segmentation(
        tensors,
        clamp_to=(width, height),
        top_k=max_detections,
        strict=False,
    )
    detections = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32)
        masks = tensor_to_numpy(item.masks).astype(np.uint8)
        if boxes.size == 0:
            continue
        for row, mask in zip(boxes.reshape((-1, 6)), masks.reshape((-1, 160, 160))):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class_id": int(class_id),
                    "mask": mask,
                }
            )
    return detections[:max_detections]


def probe_rtsp(url: str) -> tuple[int, int, int]:
    """Probe the stream once so the rest of the pipeline uses real dimensions.

    The Python sample keeps this step explicit instead of hardcoding 640x480,
    which makes the output path behave correctly for streams such as 720p.
    """
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
        fps = DEFAULT_FPS
    return width, height, fps


def build_rtsp_run(
    url: str,
    width: int,
    height: int,
    fps: int,
    latency_ms: int,
    tcp: bool,
) -> tuple[pyneat.Graph, pyneat.Run]:
    """Build a decoded RTSP input graph that yields decoded frame tensors."""
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = latency_ms
    ro.tcp = tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.auto_caps_from_stream = True
    ro.fallback_h264_width = width
    ro.fallback_h264_height = height
    ro.fallback_h264_fps = fps
    ro.sima_allocator_type = 2
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.output_caps.enable = True
    ro.output_caps.width = width
    ro.output_caps.height = height
    ro.output_caps.fps = fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    graph = pyneat.Graph()
    graph.add(pyneat.groups.rtsp_decoded_input(ro))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = SOURCE_RUN_QUEUE_DEPTH
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = graph.build(run_opt)
    return graph, run


def build_model(model_path: str, inference: InferenceConfig) -> pyneat.Model:
    """Create the YOLO26 segmentation model with model-managed preprocess/decode."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26Seg
    opt.score_threshold = inference.min_score
    opt.nms_iou_threshold = inference.nms_iou
    opt.top_k = inference.max_detections
    return pyneat.Model(model_path, opt)


def build_detector_run(model_path: str, width: int, height: int, inference: InferenceConfig):
    """Build the explicit YOLO26 segmentation graph used by the RTSP loop."""
    model = build_model(model_path, inference)

    input_opt = model.input_appsrc_options(False)
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "BGR"
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(model.graph())
    graph.add(pyneat.nodes.output())

    seed = tensor_from_bgr_frame(np.zeros((height, width, 3), dtype=np.uint8))
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    return model, graph, graph.build([seed], run_opt)


def build_video_sender_run(cfg: AppConfig, width: int, height: int, fps: int):
    """Build the VideoSender used to publish annotated RGB frames to Insight."""
    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "RGB"
    input_opt.width = width
    input_opt.height = height
    input_opt.depth = 3
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.use_simaai_pool = False

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight.host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.insight.video_port
    sender_opt.encoder.bitrate_kbps = 4000

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(pyneat.groups.video_sender(sender_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    return graph, graph.build([seed])


def build_metadata_sender(cfg: AppConfig):
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight.host
    options.channel = 0
    options.metadata_port_base = cfg.insight.metadata_port
    return pyneat.MetadataSender(options)


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def make_instance_segmentation_data_json(detections: list[dict], labels: list[str]) -> str:
    """Build lightweight metadata; masks are rendered directly into the video."""
    objects = []
    for idx, det in enumerate(detections, start=1):
        cls_id = int(det["class_id"])
        objects.append(
            {
                "id": f"obj_{idx}",
                "label": class_name(labels, cls_id),
                "confidence": float(det["score"]),
                "bbox": [
                    float(det["x1"]),
                    float(det["y1"]),
                    float(det["x2"] - det["x1"]),
                    float(det["y2"] - det["y1"]),
                ],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


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


def draw_box(frame: np.ndarray, det: dict, labels: list[str]) -> None:
    cls_id = int(det["class_id"])
    color = class_color(cls_id)
    x1 = max(0, int(round(det["x1"])))
    y1 = max(0, int(round(det["y1"])))
    x2 = min(frame.shape[1] - 1, int(round(det["x2"])))
    y2 = min(frame.shape[0] - 1, int(round(det["y2"])))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{class_name(labels, cls_id)} {float(det['score']):.2f}",
        (x1, max(0, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


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
    model_w = mask_w * 4
    model_h = mask_h * 4
    scale = min(model_w / frame_w, model_h / frame_h)
    resized_w = frame_w * scale
    resized_h = frame_h * scale
    pad_x = (model_w - resized_w) * 0.5
    pad_y = (model_h - resized_h) * 0.5

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
    mask: np.ndarray,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape, mask.shape)
    return cv2.resize(mask[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def overlay_segmentation(
    frame: np.ndarray,
    detections: list[dict],
    min_score: float,
    cfg: OutputConfig,
    labels: list[str],
) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        score = float(det["score"])
        if score < min_score:
            continue
        mask = det.get("mask")
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=np.uint8)
        max_mask = int(mask.max()) if mask.size else 0
        threshold = cfg.mask_threshold * (255.0 if max_mask > 1 else 1.0)
        x0, y0, x1, y1 = frame_rect_for_detection(det, frame.shape)
        roi_mask = project_letterbox_mask_roi(mask, frame.shape, (x0, y0, x1, y1))
        region = roi_mask > threshold
        if np.any(region):
            color = np.array(class_color(int(det["class_id"])), dtype=np.float32)
            roi = annotated[y0:y1, x0:x1]
            roi[region] = (
                (1.0 - cfg.mask_alpha) * roi[region].astype(np.float32)
                + cfg.mask_alpha * color
            ).astype(np.uint8)
            contour_mask = region.astype(np.uint8)
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, contours, -1, class_color(int(det["class_id"])), 2, cv2.LINE_8)
        if cfg.draw_boxes:
            draw_box(annotated, det, labels)
    return annotated


def save_annotated_frame(frame: np.ndarray, out_path: Path) -> None:
    """Write one BGR frame for e2e/debug inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError(f"failed to write output frame: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose only the small set of controls needed for this reference flow."""
    parser = argparse.ArgumentParser(
        description="Single-camera RTSP YOLO26 segmentation Insight example"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate config and exit without opening the RTSP stream.",
    )
    return parser


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_string(mapping: dict, key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value


def _optional_int(mapping: dict, key: str, default: int) -> int:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _optional_float(mapping: dict, key: str, default: float) -> float:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_bool(mapping: dict, key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return bool(value)


def _optional_string(mapping: dict, key: str, default: str) -> str:
    value = mapping.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def load_app_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    model_cfg = _mapping(raw.get("model"), "model")
    source_cfg = _mapping(raw.get("source"), "source")
    inference_cfg = _mapping(raw.get("inference"), "inference")
    runtime_cfg = _mapping(raw.get("runtime"), "runtime")
    output_cfg = _mapping(raw.get("output"), "output")
    insight_cfg = _mapping(output_cfg.get("insight"), "output.insight")

    cfg = AppConfig(
        model=ModelConfig(
            path=_required_string(model_cfg, "path", "model"),
            labels=_optional_string(model_cfg, "labels", str(DEFAULT_LABELS)),
        ),
        source=SourceConfig(
            rtsp_url=_required_string(source_cfg, "rtsp_url", "source"),
            latency_ms=_optional_int(source_cfg, "latency_ms", 200),
            tcp=_optional_bool(source_cfg, "tcp", True),
        ),
        inference=InferenceConfig(
            frames=_optional_int(inference_cfg, "frames", 0),
            min_score=_optional_float(inference_cfg, "min_score", 0.55),
            nms_iou=_optional_float(inference_cfg, "nms_iou", 0.60),
            max_detections=_optional_int(inference_cfg, "max_detections", 50),
        ),
        runtime=RuntimeConfig(
            profile=_optional_bool(runtime_cfg, "profile", False),
            profile_interval=_optional_int(runtime_cfg, "profile_interval", 100),
        ),
        insight=InsightConfig(
            host=_required_string(insight_cfg, "host", "output.insight"),
            video_port=_optional_int(insight_cfg, "video_port", 9000),
            metadata_port=_optional_int(insight_cfg, "metadata_port", 9100),
        ),
        output=OutputConfig(
            save_dir=_optional_string(output_cfg, "save_dir", ""),
            save_every=_optional_int(output_cfg, "save_every", 0),
            mask_alpha=_optional_float(output_cfg, "mask_alpha", MASK_ALPHA),
            mask_threshold=_optional_float(output_cfg, "mask_threshold", MASK_THRESHOLD),
            draw_boxes=_optional_bool(output_cfg, "draw_boxes", True),
        ),
    )

    if cfg.source.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.inference.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.output.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.output.mask_alpha <= 1.0:
        raise ValueError("output.mask_alpha must be between 0 and 1")
    if not 0.0 <= cfg.output.mask_threshold <= 1.0:
        raise ValueError("output.mask_threshold must be between 0 and 1")
    if not 0.0 <= cfg.inference.min_score <= 1.0:
        raise ValueError("inference.min_score must be between 0 and 1")
    if not 0.0 <= cfg.inference.nms_iou <= 1.0:
        raise ValueError("inference.nms_iou must be between 0 and 1")
    if cfg.inference.max_detections <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if cfg.runtime.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.insight.video_port <= 0:
        raise ValueError("output.insight.video_port must be > 0")
    if cfg.insight.metadata_port <= 0:
        raise ValueError("output.insight.metadata_port must be > 0")

    return cfg


class ProfileWindow:
    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.boxes = 0
        self.started = 0.0
        self.rtsp_pull_ms = 0.0
        self.infer_ms = 0.0
        self.decode_ms = 0.0
        self.video_push_ms = 0.0
        self.e2e_ms = 0.0
        self.max_e2e_ms = 0.0

    def add(
        self,
        *,
        rtsp_pull_ms: float,
        infer_ms: float,
        decode_ms: float,
        video_push_ms: float,
        e2e_ms: float,
        boxes: int,
        published_total: int,
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.started = time.perf_counter()
        self.frames += 1
        self.boxes += boxes
        self.rtsp_pull_ms += rtsp_pull_ms
        self.infer_ms += infer_ms
        self.decode_ms += decode_ms
        self.video_push_ms += video_push_ms
        self.e2e_ms += e2e_ms
        self.max_e2e_ms = max(self.max_e2e_ms, e2e_ms)
        if self.frames >= self.interval:
            self.flush(published_total)

    def flush(self, published_total: int) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = max(time.perf_counter() - self.started, 1e-6)
        frames = float(self.frames)
        print(
            f"[profile] frames={self.frames} published={published_total} "
            f"fps={self.frames / elapsed:.2f} "
            f"avg_rtsp_pull_ms={self.rtsp_pull_ms / frames:.2f} "
            f"avg_infer_ms={self.infer_ms / frames:.2f} "
            f"avg_decode_ms={self.decode_ms / frames:.2f} "
            f"avg_video_push_ms={self.video_push_ms / frames:.2f} "
            f"avg_e2e_ms={self.e2e_ms / frames:.2f} "
            f"avg_boxes={self.boxes / frames:.2f} "
            f"max_e2e_ms={self.max_e2e_ms:.2f}",
            flush=True,
        )
        self.reset()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        cfg = load_app_config(args.config)
    except Exception as exc:
        print(f"Error: failed to load config {args.config}: {exc}", file=sys.stderr)
        return 2
    if args.validate_config_only:
        print(f"Config validated: {args.config}")
        return 0

    load_runtime_dependencies()
    model_path = cfg.model.path
    if not model_path or not Path(model_path).is_file():
        print("Failed to locate YOLO26 segmentation model package.", file=sys.stderr)
        print("Set model.path to a YOLO26 segmentation package.", file=sys.stderr)
        return 2
    try:
        labels = load_labels(Path(cfg.model.labels))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    rtsp_graph = None
    rtsp_run = None
    detector_graph = None
    detector_run = None
    video_graph = None
    video_run = None
    metadata_sender = None
    try:
        # Probe first so decode, inference, and VideoSender all agree on the
        # same live frame dimensions.
        frame_w, frame_h, fps = probe_rtsp(cfg.source.rtsp_url)
        print(f"[init] probed RTSP decode dims {frame_w}x{frame_h}")

        model, detector_graph, detector_run = build_detector_run(
            model_path,
            frame_w,
            frame_h,
            cfg.inference,
        )
        rtsp_graph, rtsp_run = build_rtsp_run(
            cfg.source.rtsp_url,
            frame_w,
            frame_h,
            fps,
            cfg.source.latency_ms,
            tcp=cfg.source.tcp,
        )
        video_graph, video_run = build_video_sender_run(cfg, frame_w, frame_h, fps)
        metadata_sender = build_metadata_sender(cfg)

        print(f"insight host={cfg.insight.host} video_port={cfg.insight.video_port} "
              f"metadata_port={metadata_sender.metadata_port()} channel=0")

        processed = 0
        started = time.perf_counter()
        profile_window = ProfileWindow(cfg.runtime.profile, cfg.runtime.profile_interval)
        save_dir = Path(cfg.output.save_dir) if cfg.output.save_dir else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        # Contract: single-threaded frame order is pull -> infer -> publish video -> publish metadata.
        while cfg.inference.frames <= 0 or processed < cfg.inference.frames:
            t_pull0 = time.perf_counter()
            tensors = rtsp_run.pull_tensors(timeout_ms=20000)
            t_pull1 = time.perf_counter()
            if not tensors:
                print("RTSP pull timed out / stream closed", file=sys.stderr)
                break

            decoded_frame = tensors[0]
            frame = tensor_bgr_from_decoded(decoded_frame)
            infer_input = tensor_from_bgr_frame(frame)

            t_inf0 = time.perf_counter()
            result = detector_run.run([infer_input], timeout_ms=20000)
            t_inf1 = time.perf_counter()

            t_decode0 = time.perf_counter()
            detections = segmentation_results_from_output(
                result,
                frame.shape[1],
                frame.shape[0],
                cfg.inference.max_detections,
            )
            t_decode1 = time.perf_counter()
            annotated = overlay_segmentation(
                frame,
                detections,
                cfg.inference.min_score,
                cfg.output,
                labels,
            )
            video_frame = tensor_from_rgb_frame(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            # Contract: publish video first, then publish matching metadata.
            t_video0 = time.perf_counter()
            video_ok = video_run.push([video_frame])
            if not video_ok:
                raise RuntimeError("Insight video push failed")
            t_video1 = time.perf_counter()
            fid = str(processed)
            data_json = make_instance_segmentation_data_json(detections, labels)
            metadata_sender.send_metadata(
                "instance-segmentation",
                data_json,
                int(time.time() * 1000),
                fid,
            )
            should_save = (
                save_dir is not None
                and cfg.output.save_every > 0
                and (processed + 1) % cfg.output.save_every == 0
            )
            if should_save:
                save_annotated_frame(
                    annotated,
                    save_dir / f"frame_{processed:06d}.jpg",
                )

            processed += 1
            profile_window.add(
                rtsp_pull_ms=(t_pull1 - t_pull0) * 1000.0,
                infer_ms=(t_inf1 - t_inf0) * 1000.0,
                decode_ms=(t_decode1 - t_decode0) * 1000.0,
                video_push_ms=(t_video1 - t_video0) * 1000.0,
                e2e_ms=(t_video1 - t_pull1) * 1000.0,
                boxes=len(detections),
                published_total=processed,
            )

        elapsed = max(time.perf_counter() - started, 1e-6)
        profile_window.flush(processed)
        print(f"processed={processed} fps={processed / elapsed:.2f} "
              f"video_sender={cfg.insight.host}:{cfg.insight.video_port}")
        return 0 if processed > 0 else 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        # Contract: close output run, then close RTSP run.
        try:
            if video_run is not None:
                video_run.close()
        except Exception:
            pass
        try:
            if detector_run is not None:
                detector_run.close()
        except Exception:
            pass
        try:
            if rtsp_run is not None:
                rtsp_run.close()
        except Exception:
            pass
        try:
            metadata_sender = None
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
