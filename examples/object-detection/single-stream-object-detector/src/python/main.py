"""Single-camera RTSP YOLO26 Insight example using pyneat.

This mirrors the intent of the C++ reference sample in the same folder:

- pull one decoded RTSP stream
- run one YOLO26 detector
- publish H.264 video plus detection metadata to Insight

The runtime path is one composed graph:

RtspDecodedInput -> Branch -> VideoSender
                         -> Model -> detections
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
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
cv2 = None
np = None
pyneat = None

@dataclass(frozen=True)
class ModelConfig:
    path: str
    labels: Path


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


def make_source_options(cfg: AppConfig, width: int, height: int, fps: int):
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = cfg.source.rtsp_url
    ro.latency_ms = cfg.source.latency_ms
    ro.tcp = cfg.source.tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.out_format = "NV12"
    ro.decoder_name = "decoder"
    ro.decoder_raw_output = True
    ro.auto_caps_from_stream = True
    ro.fallback_h264_width = width
    ro.fallback_h264_height = height
    ro.fallback_h264_fps = fps
    ro.sima_allocator_type = 2
    ro.use_videoconvert = False
    return ro


def build_model(model_path: str, inference: InferenceConfig) -> pyneat.Model:
    """Create the YOLO model for model-managed YOLO26 preprocessing and decode."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
    opt.decode_type = pyneat.BoxDecodeType.YoloV26
    opt.score_threshold = inference.min_score
    opt.nms_iou_threshold = inference.nms_iou
    opt.top_k = inference.max_detections
    return pyneat.Model(model_path, opt)


@dataclass
class PipelineRuntime:
    graph: object
    run: object
    metadata_sender: object
    labels: list[str]
    frame_w: int
    frame_h: int
    output_fps: int
    video_port: int


def build_pipeline(cfg: AppConfig, model_path: str) -> PipelineRuntime:
    frame_w, frame_h, fps = probe_rtsp(cfg.source.rtsp_url)
    model = build_model(model_path, cfg.inference)
    labels = load_labels(cfg.model.labels)

    source = pyneat.groups.rtsp_decoded_input(make_source_options(cfg, frame_w, frame_h, fps))
    source_outputs = ["video", "model"]
    if cfg.save_dir and cfg.save_every > 0:
        source_outputs.append("frames")
    branch = pyneat.graphs.branch("source", source_outputs)

    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(frame_w, frame_h, max(1, fps))
    sender_opt.host = cfg.insight.host
    sender_opt.channel = 0
    sender_opt.video_port_base = cfg.insight.video_port
    sender_opt.encoder.bitrate_kbps = 600

    video_graph = pyneat.Graph("video")
    video_graph.connect(
        pyneat.nodes.input("video"),
        pyneat.groups.video_sender(sender_opt),
    )

    model_graph = pyneat.Graph("model")
    model_graph.connect(pyneat.nodes.input("model"), model)
    detections = pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4))

    graph = pyneat.Graph()
    graph.connect(source, branch)
    graph.connect(branch, video_graph)
    graph.connect(branch, model_graph)
    graph.connect(model_graph, detections)
    if cfg.save_dir and cfg.save_every > 0:
        frames = pyneat.Graph("frames")
        frames.add(pyneat.nodes.output("frames", pyneat.OutputOptions.latest()))
        graph.connect(branch, frames)

    run_opt = pyneat.RunOptions()
    run_opt.preset = pyneat.RunPreset.Reliable
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = graph.build(run_opt)
    metadata_sender = build_metadata_sender(cfg)
    print(
        f"rtsp={cfg.source.rtsp_url} stream={frame_w}x{frame_h}@{fps} "
        f"insight={cfg.insight.host} video={sender_opt.video_port} "
        f"metadata={metadata_sender.metadata_port()} channel=0"
    )
    return PipelineRuntime(
        graph=graph,
        run=run,
        metadata_sender=metadata_sender,
        labels=labels,
        frame_w=frame_w,
        frame_h=frame_h,
        output_fps=fps,
        video_port=sender_opt.video_port,
    )


def build_metadata_sender(cfg: AppConfig):
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight.host
    options.channel = 0
    options.metadata_port_base = cfg.insight.metadata_port
    return pyneat.MetadataSender(options)


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def class_name(labels: list[str], class_id: int) -> str:
    return labels[class_id] if 0 <= class_id < len(labels) else "unknown"


def make_object_detection_data_json(boxes: list[dict], labels: list[str]) -> str:
    """Build the object-detection metadata data object."""
    objects = []
    for idx, box in enumerate(boxes, start=1):
        cls_id = int(box["class_id"])
        objects.append(
            {
                "id": f"obj_{idx}",
                "label": class_name(labels, cls_id),
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


def save_annotated_frame(
    frame: np.ndarray,
    boxes: list[dict],
    labels: list[str],
    min_score: float,
    out_path: Path,
) -> None:
    """Write one BGR frame with detection boxes for e2e/debug inspection."""
    annotated = frame.copy()
    for box in boxes:
        score = float(box["score"])
        if score < min_score:
            continue
        cls_id = int(box["class_id"])
        label = class_name(labels, cls_id)
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


def maybe_save_debug_frame(runtime: PipelineRuntime, cfg: AppConfig, processed: int, boxes: list[dict]) -> None:
    if not cfg.save_dir or cfg.save_every <= 0 or processed % cfg.save_every != 0:
        return
    frame_sample = runtime.run.pull("frames", 0)
    frame_tensor = first_tensor_from_sample(frame_sample)
    if frame_tensor is None:
        return
    frame = tensor_bgr_from_decoded(frame_tensor)
    save_annotated_frame(
        frame,
        boxes,
        runtime.labels,
        cfg.inference.min_score,
        Path(cfg.save_dir) / f"frame_{processed}.jpg",
    )


def send_metadata(runtime: PipelineRuntime, sample, boxes: list[dict]) -> None:
    data_json = make_object_detection_data_json(boxes, runtime.labels)
    frame_id = getattr(sample, "frame_id", -1)
    if frame_id is None or frame_id < 0:
        frame_id = 0
    runtime.metadata_sender.send_metadata(
        "object-detection",
        data_json,
        int(time.time() * 1000),
        str(frame_id),
    )


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
    default_labels = Path(__file__).resolve().parents[1] / "common" / "coco_label.txt"

    cfg = AppConfig(
        model=ModelConfig(
            path=_required_string(model_cfg, "path", "model"),
            labels=Path(_optional_string(model_cfg, "labels", str(default_labels))),
        ),
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
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(
        self,
        *,
        detection_pull_ms: float,
        metadata_send_ms: float,
        boxes: int,
    ) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.started = time.perf_counter()
        self.frames += 1
        self.boxes += boxes
        self.detection_pull_ms += detection_pull_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = max(time.perf_counter() - self.started, 1e-6)
        frames = float(self.frames)
        print(
            f"[profile] frames={self.frames} "
            f"output_fps={self.frames / elapsed:.2f} "
            f"avg_detection_pull_ms={self.detection_pull_ms / frames:.2f} "
            f"avg_metadata_send_ms={self.metadata_send_ms / frames:.2f} "
            f"avg_boxes={self.boxes / frames:.2f}",
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

    runtime = None
    try:
        runtime = build_pipeline(cfg, model_path)

        processed = 0
        started = time.perf_counter()
        profile_window = ProfileWindow(cfg.runtime.profile, cfg.runtime.profile_interval)
        save_dir = Path(cfg.save_dir) if cfg.save_dir else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
        while cfg.inference.frames <= 0 or processed < cfg.inference.frames:
            pull_start = time.perf_counter()
            detection_sample = runtime.run.pull("detections", 20000)
            pull_end = time.perf_counter()
            if detection_sample is None:
                print("[warn] timed out waiting for detections", file=sys.stderr)
                continue

            payload = extract_bbox_payload(detection_sample)
            if not payload:
                raise RuntimeError("model returned no BBOX payload")
            boxes = parse_bbox_payload(
                payload,
                runtime.frame_w,
                runtime.frame_h,
                cfg.inference.max_detections,
            )

            metadata_start = time.perf_counter()
            send_metadata(runtime, detection_sample, boxes)
            metadata_end = time.perf_counter()

            processed += 1
            maybe_save_debug_frame(runtime, cfg, processed, boxes)
            profile_window.add(
                detection_pull_ms=(pull_end - pull_start) * 1000.0,
                metadata_send_ms=(metadata_end - metadata_start) * 1000.0,
                boxes=len(boxes),
            )

        elapsed = max(time.perf_counter() - started, 1e-6)
        profile_window.flush()
        print(f"processed={processed} fps={processed / elapsed:.2f} "
              f"video_sender={cfg.insight.host}:{runtime.video_port}")
        return 0 if processed > 0 else 3
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            if runtime is not None:
                runtime.run.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
