"""Single-camera RTSP YOLO26 Insight example using pyneat.

This mirrors the intent of the C++ reference sample in the same folder:

- pull one decoded RTSP stream
- run one YOLO26 detector
- publish H.264 video plus detection metadata to Insight

The implementation keeps those responsibilities loosely separated so the main
runtime path is easy to reason about:

1. RTSP probe/build
2. YOLO26 inference and box extraction
3. Insight video/metadata publishing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
import yaml

DEFAULT_FPS = 30
SOURCE_RUN_QUEUE_DEPTH = 4
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
cv2 = None
np = None
pyneat = None
DFL_BINS_16 = None

# Standard COCO class order for the bundled YOLO26 detector model used by this sample.
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

@dataclass(frozen=True)
class ModelConfig:
    path: str


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
class AppConfig:
    model: ModelConfig
    source: SourceConfig
    inference: InferenceConfig
    runtime: RuntimeConfig
    insight: InsightConfig
    save_dir: str
    save_every: int


def load_runtime_dependencies() -> None:
    """Import runtime-only dependencies after argparse validation."""
    global cv2, np, pyneat, DFL_BINS_16
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
    DFL_BINS_16 = np.arange(16, dtype=np.float32)


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


def blank_nv12_tensor(width: int, height: int) -> pyneat.Tensor:
    """Create a CPU-backed blank NV12 tensor with explicit Y and UV plane metadata."""
    if width <= 0 or height <= 0 or width % 2 != 0 or height % 2 != 0:
        raise ValueError("NV12 requires positive even width and height")
    payload = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    tensor = pyneat.Tensor.from_numpy(
        payload,
        copy=True,
        layout=pyneat.TensorLayout.HW,
        image_format=pyneat.PixelFormat.NV12,
        memory=pyneat.TensorMemory.CPU,
    )
    tensor.shape = [height, width]
    tensor.strides_bytes = [width, 1]

    y_plane = pyneat.Plane()
    y_plane.role = pyneat.PlaneRole.Y
    y_plane.shape = [height, width]
    y_plane.strides_bytes = [width, 1]
    y_plane.byte_offset = 0

    uv_plane = pyneat.Plane()
    uv_plane.role = pyneat.PlaneRole.UV
    uv_plane.shape = [height // 2, width]
    uv_plane.strides_bytes = [width, 1]
    uv_plane.byte_offset = width * height

    tensor.planes = [y_plane, uv_plane]
    return tensor


def is_tensor_like(value) -> bool:
    return hasattr(value, "copy_payload_bytes") and hasattr(value, "to_numpy")


def is_sample_like(value) -> bool:
    return hasattr(value, "kind") and hasattr(value, "fields")


def extract_bbox_payload_from_tensors(tensors) -> bytes | None:
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def extract_bbox_payload(result) -> bytes | None:
    """Prefer the runtime's pre-decoded BBOX payload when the model emits it.

    Some YOLO pipelines already attach a compact BBOX payload. When that exists
    it is more reliable and cheaper to parse than re-decoding the raw YOLO head
    tensors in Python.
    """
    if isinstance(result, (list, tuple)) and all(is_tensor_like(item) for item in result):
        return extract_bbox_payload_from_tensors(result)

    if not is_sample_like(result):
        return None

    stack = [result]
    while stack:
        current = stack.pop()
        stack.extend(reversed(list(current.fields)))
        if current.kind == pyneat.SampleKind.TensorSet:
            payload = extract_bbox_payload_from_tensors(current.tensors)
            if payload:
                return payload
            continue
        if current.kind != pyneat.SampleKind.Tensor or current.tensor is None:
            continue
        fmt = (current.payload_tag or current.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = current.tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, max_detections: int) -> list[dict]:
    """Decode the packed BBOX payload format used by NEAT samples."""
    if len(payload) < 4:
        return []
    # Payload layout:
    #   uint32 count
    #   repeated { int32 x, int32 y, int32 w, int32 h, float score, int32 class_id }
    #
    # Guard the parsed count against truncated payloads so we never read past
    # the actual buffer contents.
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24, max_detections)
    boxes = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        # Clamp every box back into the decoded frame. This keeps Insight metadata
        # consistent even if the payload contains slightly out-of-bounds values.
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        # Skip degenerate boxes after clamping so downstream consumers only see
        # valid xyxy coordinates.
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": float(score),
                "class_id": int(cls_id),
            }
        )
    return boxes


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
    """Create the YOLO model for model-managed YOLO26 preprocessing and decode."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = inference.min_score
    opt.nms_iou_threshold = inference.nms_iou
    opt.top_k = inference.max_detections
    return pyneat.Model(model_path, opt)


def build_detector_run(model_path: str, width: int, height: int, inference: InferenceConfig):
    """Build the explicit YOLO26 detector graph used by the RTSP loop."""
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
    """Build the generic NEAT video sender used by this Insight-targeted example."""
    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "NV12"
    input_opt.width = width
    input_opt.height = height
    input_opt.fps_n = max(1, fps)
    input_opt.fps_d = 1
    input_opt.caps_override = (
        f"video/x-raw,format=NV12,width={width},height={height},framerate={max(1, fps)}/1"
    )
    input_opt.use_simaai_pool = False

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, max(1, fps))
    sender_opt.host = cfg.insight.host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.insight.video_port
    sender_opt.encoder.bitrate_kbps = 4000

    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(pyneat.groups.video_sender(sender_opt))

    seed = blank_nv12_tensor(width, height)
    return graph, graph.build([seed])


def build_metadata_sender(cfg: AppConfig):
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight.host
    options.channel = 0
    options.metadata_port_base = cfg.insight.metadata_port
    return pyneat.MetadataSender(options)


def make_object_detection_data_json(boxes: list[dict]) -> str:
    """Build the object-detection metadata data object."""
    objects = []
    for idx, box in enumerate(boxes, start=1):
        cls_id = int(box["class_id"])
        label = COCO80_NAMES[cls_id] if 0 <= cls_id < len(COCO80_NAMES) else "Unknown"
        objects.append(
            {
                "id": f"obj_{idx}",
                "label": label,
                "confidence": float(box["score"]),
                "bbox": [
                    float(box["x1"]),
                    float(box["y1"]),
                    float(box["x2"] - box["x1"]),
                    float(box["y2"] - box["y1"]),
                ],
            }
        )
    return json.dumps({"objects": objects}, separators=(",", ":"))


def save_annotated_frame(frame: np.ndarray, boxes: list[dict], min_score: float, out_path: Path) -> None:
    """Write one BGR frame with detection boxes for e2e/debug inspection."""
    annotated = frame.copy()
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        cls_id = int(box["class_id"])
        label = COCO80_NAMES[cls_id] if 0 <= cls_id < len(COCO80_NAMES) else "unknown"
        x1 = int(round(box["x1"]))
        y1 = int(round(box["y1"]))
        x2 = int(round(box["x2"]))
        y2 = int(round(box["y2"]))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), annotated):
        raise RuntimeError(f"failed to write output frame: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose only the small set of controls needed for this reference flow."""
    parser = argparse.ArgumentParser(description="Single-camera RTSP YOLO26 Insight example")
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
        model=ModelConfig(path=_required_string(model_cfg, "path", "model")),
        source=SourceConfig(
            rtsp_url=_required_string(source_cfg, "rtsp_url", "source"),
            latency_ms=_optional_int(source_cfg, "latency_ms", 200),
            tcp=_optional_bool(source_cfg, "tcp", True),
        ),
        inference=InferenceConfig(
            frames=_optional_int(inference_cfg, "frames", 0),
            min_score=_optional_float(inference_cfg, "min_score", 0.55),
            nms_iou=_optional_float(inference_cfg, "nms_iou", 0.50),
            max_detections=_optional_int(inference_cfg, "max_detections", 100),
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
        save_dir=_optional_string(output_cfg, "save_dir", ""),
        save_every=_optional_int(output_cfg, "save_every", 0),
    )

    if cfg.source.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.inference.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
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
        self.bbox_ms = 0.0
        self.video_push_ms = 0.0
        self.e2e_ms = 0.0
        self.max_e2e_ms = 0.0

    def add(
        self,
        *,
        rtsp_pull_ms: float,
        infer_ms: float,
        bbox_ms: float,
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
        self.bbox_ms += bbox_ms
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
            f"avg_bbox_ms={self.bbox_ms / frames:.2f} "
            f"avg_video_push_ms={self.video_push_ms / frames:.2f} "
            f"avg_e2e_ms={self.e2e_ms / frames:.2f} "
            f"avg_boxes={self.boxes / frames:.2f} "
            f"max_e2e_ms={self.max_e2e_ms:.2f}",
            flush=True,
        )
        self.reset()


def resolve_yolo26_model(root: Path) -> str:
    """Mirror the C++ sample's local-first model lookup strategy.

    Resolution order:

    1. explicit environment override
    2. local/common modelzoo directories
    3. `sima-cli download <YOLO26 detector URL>`
    """
    env_path = os.environ.get("SIMA_YOLO_TAR", "")
    if env_path and Path(env_path).exists():
        return env_path

    tmp_dir = root / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_name = "yolo26m-det-bf16-mla_tess-b1.tar.gz"
    model_url = (
        "https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix/"
        f"yolo26-detection/{model_name}"
    )
    tmp_tar = tmp_dir / model_name
    direct_tar = root / model_name
    if direct_tar.exists():
        return str(direct_tar)
    if tmp_tar.exists():
        return str(tmp_tar)

    home = Path.home()
    search_dirs = [
        root / "models",
        root,
        Path.cwd(),
        root / "tmp",
        home / ".simaai",
        home / ".simaai" / "modelzoo",
        home / ".sima" / "modelzoo",
        Path("/data/simaai/modelzoo"),
    ]
    names = [
        model_name,
    ]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)

    try:
        subprocess.run(["sima-cli", "download", model_url], cwd=str(tmp_dir), check=True)
    except Exception:
        return ""

    if tmp_tar.exists():
        return str(tmp_tar)
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)
    return ""


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
    model_path = cfg.model.path or resolve_yolo26_model(Path.cwd())
    if not model_path or not Path(model_path).is_file():
        print("Failed to locate YOLO26 detector model package.", file=sys.stderr)
        print("Set model.path or download a YOLO26 detector package with sima-cli.", file=sys.stderr)
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

        # NEAT boundary: build YOLO detector graph from the resolved compiled model package.
        model, detector_graph, detector_run = build_detector_run(
            model_path,
            frame_w,
            frame_h,
            cfg.inference,
        )
        # NEAT boundary: build RTSP decode runtime used by pull_tensors().
        rtsp_graph, rtsp_run = build_rtsp_run(
            cfg.source.rtsp_url,
            frame_w,
            frame_h,
            fps,
            cfg.source.latency_ms,
            tcp=cfg.source.tcp,
        )
        # NEAT boundary: build VideoSender and MetadataSender.
        video_graph, video_run = build_video_sender_run(cfg, frame_w, frame_h, fps)
        metadata_sender = build_metadata_sender(cfg)

        print(f"insight host={cfg.insight.host} video_port={cfg.insight.video_port} "
              f"metadata_port={metadata_sender.metadata_port()} channel=0")

        processed = 0
        started = time.perf_counter()
        profile_window = ProfileWindow(cfg.runtime.profile, cfg.runtime.profile_interval)
        save_dir = Path(cfg.save_dir) if cfg.save_dir else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        # Contract: single-threaded frame order is pull -> infer -> publish video -> publish metadata.
        while cfg.inference.frames <= 0 or processed < cfg.inference.frames:
            # Push/pull integration point: pull one decoded frame from RTSP run.
            t_pull0 = time.perf_counter()
            tensors = rtsp_run.pull_tensors(timeout_ms=20000)
            t_pull1 = time.perf_counter()
            if not tensors:
                print("RTSP pull timed out / stream closed", file=sys.stderr)
                break

            decoded_frame = tensors[0]
            frame = tensor_bgr_from_decoded(decoded_frame)
            infer_input = tensor_from_bgr_frame(frame)

            # Push/pull integration point: run model and parse the BBOX payload.
            t_inf0 = time.perf_counter()
            result = detector_run.run([infer_input], timeout_ms=20000)
            t_inf1 = time.perf_counter()

            payload = extract_bbox_payload(result)
            if not payload:
                raise RuntimeError("model returned no BBOX payload")
            t_bbox0 = time.perf_counter()
            boxes = parse_bbox_payload(
                payload,
                frame.shape[1],
                frame.shape[0],
                cfg.inference.max_detections,
            )
            t_bbox1 = time.perf_counter()

            # Contract: publish video first, then publish matching metadata.
            t_video0 = time.perf_counter()
            video_ok = video_run.push([decoded_frame])
            if not video_ok:
                raise RuntimeError("Insight video push failed")
            t_video1 = time.perf_counter()
            fid = str(processed)
            data_json = make_object_detection_data_json(boxes)
            metadata_sender.send_metadata(
                "object-detection",
                data_json,
                int(time.time() * 1000),
                fid,
            )
            should_save = (
                save_dir is not None
                and cfg.save_every > 0
                and (processed + 1) % cfg.save_every == 0
            )
            if should_save:
                save_annotated_frame(
                    frame,
                    boxes,
                    cfg.inference.min_score,
                    save_dir / f"frame_{processed:06d}.jpg",
                )

            processed += 1
            profile_window.add(
                rtsp_pull_ms=(t_pull1 - t_pull0) * 1000.0,
                infer_ms=(t_inf1 - t_inf0) * 1000.0,
                bbox_ms=(t_bbox1 - t_bbox0) * 1000.0,
                video_push_ms=(t_video1 - t_video0) * 1000.0,
                e2e_ms=(t_video1 - t_pull1) * 1000.0,
                boxes=len(boxes),
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
