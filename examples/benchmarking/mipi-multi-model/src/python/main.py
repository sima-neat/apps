#!/usr/bin/env python3
"""Run a selected model directly from a strict zero-copy MIPI camera."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from model_profiles import (
    PROFILES,
    ModelPackage,
    ModelPackageError,
    Profile,
    inspect_package,
    profile_named,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
DEFAULT_MODELS_DIR = Path("models")
OUTPUT_NAME = "results"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CAMERA_BUFFER = "camera0"
CAMERA_CAPTURE_BUFFERS = 32
MASK_STRIDE = 4
MASK_THRESHOLD = 0.50
METADATA_BYTE_BUDGET = 32_768
DEFAULT_LABELS = (
    Path(__file__).resolve().parents[5]
    / "examples"
    / "segmentation"
    / "single-stream-instance-segmenter"
    / "src"
    / "common"
    / "coco_label.txt"
)


@dataclass(frozen=True)
class AppConfig:
    profile: Profile
    model_path: Path
    frames: int
    timeout_ms: int


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_config(path: Path) -> AppConfig:
    try:
        with path.open("r", encoding="utf-8") as handle:
            root = _mapping(yaml.safe_load(handle), "config root")
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load {path}: {exc}") from exc

    model = _mapping(root.get("model"), "model")
    runtime = _mapping(root.get("runtime"), "runtime")
    profile_value = model.get("profile", "detect")
    path_value = model.get("path", "")
    if not isinstance(profile_value, str):
        raise TypeError("model.profile must be a string")
    if not isinstance(path_value, str):
        raise TypeError("model.path must be a string")

    profile = profile_named(profile_value)
    model_path = Path(path_value) if path_value.strip() else DEFAULT_MODELS_DIR / profile.archive
    return AppConfig(
        profile=profile,
        model_path=model_path,
        frames=_positive_integer(runtime.get("frames", 5), "runtime.frames"),
        timeout_ms=_positive_integer(runtime.get("timeout_ms", 30_000), "runtime.timeout_ms"),
    )


def model_options(pyneat, profile: Profile, package: ModelPackage):
    """Set only source information and model policy that cannot be inferred."""
    options = pyneat.ModelOptions()
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12

    if profile.decode_type in {"YoloV26", "YoloV26Pose", "YoloV26Seg"}:
        preprocess = options.preprocess
        preprocess.kind = pyneat.InputKind.Image
        preprocess.enable = pyneat.AutoFlag.On
        preprocess.input_max_width = CAMERA_WIDTH
        preprocess.input_max_height = CAMERA_HEIGHT
        preprocess.input_max_depth = 3
        preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
        preprocess.resize.enable = pyneat.AutoFlag.On
        preprocess.resize.width = 640
        preprocess.resize.height = 640
        preprocess.resize.mode = pyneat.ResizeMode.Letterbox
        preprocess.resize.pad_value = 114
        preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        options.advanced_execution.preprocess_target = "EV74"

    if profile.preprocessing == "torchvision_ssdlite":
        preprocess = options.preprocess
        preprocess.kind = pyneat.InputKind.Image
        preprocess.enable = pyneat.AutoFlag.On
        preprocess.resize.enable = pyneat.AutoFlag.On
        preprocess.resize.mode = pyneat.ResizeMode.Stretch
        preprocess.normalize.enable = pyneat.AutoFlag.On
        preprocess.normalize.mean = [0.485, 0.456, 0.406]
        preprocess.normalize.stddev = [0.229, 0.224, 0.225]
        preprocess.normalize.has_explicit_stats = True
        preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
        options.num_classes = 91

    if profile.decode_type:
        try:
            options.decode_type = getattr(pyneat.BoxDecodeType, profile.decode_type)
        except AttributeError as exc:
            raise RuntimeError(
                f"installed Neat does not support {profile.decode_type} decoding"
            ) from exc
        options.advanced_execution.postprocess_target = "EV74"

    # Decoding is a Neat runtime concern. The MPK only needs its manifest and MLA
    # executable; use the same explicit policy as the Insight examples.
    if profile.decode_type in {"YoloV26", "YoloV26Pose", "YoloV26Seg"}:
        options.score_threshold = (
            0.55 if profile.task in {"detection", "segmentation"} else 0.30
        )
        options.nms_iou_threshold = 0.45 if profile.task == "detection" else 0.60
        options.top_k = 50
    return options


def make_graph(
    pyneat,
    profile: Profile,
    package: ModelPackage,
    *,
    allow_camera_copy: bool = False,
    insight_host: str = "",
    insight_video_port: int = 9000,
    insight_channel: int = 0,
    insight_bitrate_kbps: int = 2500,
):
    model = pyneat.Model(str(package.path), model_options(pyneat, profile, package))
    camera = pyneat.CameraInputOptions()
    camera.width = CAMERA_WIDTH
    camera.height = CAMERA_HEIGHT
    camera.framerate_num = CAMERA_FPS
    camera.framerate_den = 1
    camera.format = "NV12"
    camera.buffer_name = CAMERA_BUFFER
    camera.allow_cpu_fallback = allow_camera_copy

    connected_insight = bool(insight_host)
    route = pyneat.ModelRouteOptions()
    route.include_input = False
    route.include_output = False
    route.upstream_name = "images" if connected_insight else CAMERA_BUFFER
    route.buffer_name = CAMERA_BUFFER
    route.name_suffix = "_camera0"
    route.advanced_execution.preprocess_target = "EV74"
    if profile.decode_type:
        route.advanced_execution.postprocess_target = "EV74"

    camera_graph = pyneat.Graph(f"mipi_{profile.name}_camera")
    camera_graph.add(
        pyneat.nodes.camera_input(
            camera,
            capture_buffer_count=CAMERA_CAPTURE_BUFFERS,
        )
    )
    model_graph = model.graph(route)
    model_graph.add(pyneat.nodes.output(OUTPUT_NAME))
    if not connected_insight:
        camera_graph.add(model_graph)
        return camera_graph

    if profile.decode_type not in {"YoloV26", "YoloV26Seg"}:
        raise RuntimeError("Insight streaming currently supports detect and segment profiles")

    branch = pyneat.graphs.branch(CAMERA_BUFFER, ["images", "insight_video"])
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.NV12
    input_options.width = CAMERA_WIDTH
    input_options.height = CAMERA_HEIGHT
    input_options.depth = 1
    input_options.fps_n = CAMERA_FPS
    input_options.fps_d = 1
    input_options.memory_policy = pyneat.InputMemoryPolicy.Ev74

    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        CAMERA_WIDTH,
        CAMERA_HEIGHT,
        CAMERA_FPS,
    )
    sender_options.host = insight_host
    sender_options.channel = insight_channel
    sender_options.video_port_base = insight_video_port
    sender_options.encoder.bitrate_kbps = insight_bitrate_kbps

    video_graph = pyneat.Graph("insight_video")
    video_graph.add(pyneat.nodes.input("insight_video", input_options))
    video_graph.add(pyneat.groups.video_sender(sender_options))

    video_link = pyneat.GraphLinkOptions()
    video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    model_link = pyneat.GraphLinkOptions()
    model_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
    graph = pyneat.Graph(f"mipi_{profile.name}_insight")
    graph.connect(camera_graph, branch)
    graph.connect(branch, model_graph, model_link)
    graph.connect(branch, video_graph, video_link)
    return graph


def sample_tensors(sample) -> list:
    """Flatten Tensor, TensorSet, and Bundle samples."""
    tensors = []
    tensor = getattr(sample, "tensor", None)
    if tensor is not None:
        tensors.append(tensor)
    tensors.extend(list(getattr(sample, "tensors", []) or []))
    for field in list(getattr(sample, "fields", []) or []):
        tensors.extend(sample_tensors(field))
    return tensors


def _rows(tensor) -> int:
    shape = list(tensor.shape)
    return int(shape[0]) if shape else 0


def summarize_output(pyneat, profile: Profile, sample) -> str:
    tensors = sample_tensors(sample)
    if not tensors:
        raise RuntimeError(f"{profile.name} produced no output tensors")

    if profile.task == "detection":
        return f"detections={sum(_rows(item) for item in pyneat.decode_bbox(tensors))}"
    if profile.task == "pose":
        poses = sum(_rows(item.boxes) for item in pyneat.decode_pose(tensors))
        return f"poses={poses}"
    if profile.task == "segmentation":
        instances = sum(_rows(item.boxes) for item in pyneat.decode_segmentation(tensors))
        return f"instances={instances}"

    import numpy as np

    first = tensors[0]
    values = np.asarray(first.to_numpy(copy=True))
    if values.size == 0:
        raise RuntimeError(f"{profile.task} output tensor is empty")
    shape = [int(value) for value in first.shape]
    if profile.task == "classification":
        flattened = values.reshape(-1)
        class_id = int(np.argmax(flattened))
        return f"class_id={class_id} value={float(flattened[class_id]):.6g} shape={shape}"
    return (
        f"depth_shape={shape} min={float(np.nanmin(values)):.6g} "
        f"max={float(np.nanmax(values)):.6g}"
    )


def load_labels(path: Path = DEFAULT_LABELS) -> list[str]:
    try:
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        raise RuntimeError(f"cannot load labels from {path}: {exc}") from exc
    if not labels:
        raise RuntimeError(f"labels file is empty: {path}")
    return labels


def insight_objects(pyneat, profile: Profile, sample, labels: list[str]) -> list[dict]:
    """Convert Neat's decoded detector output to Insight's object schema."""
    import numpy as np

    if profile.task != "detection":
        return []
    tensors = sample_tensors(sample)
    box_tensors = pyneat.decode_bbox(tensors)

    objects = []
    for box_tensor in box_tensors:
        rows = np.asarray(box_tensor.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        for row in rows:
            x1, y1, x2, y2, score, class_id_value = row.tolist()
            class_id = int(class_id_value)
            x = max(0.0, min(float(x1), float(CAMERA_WIDTH)))
            y = max(0.0, min(float(y1), float(CAMERA_HEIGHT)))
            x2 = max(x, min(float(x2), float(CAMERA_WIDTH)))
            y2 = max(y, min(float(y2), float(CAMERA_HEIGHT)))
            objects.append(
                {
                    "id": f"obj_{len(objects) + 1}",
                    "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                    "confidence": float(score),
                    "bbox": [x, y, x2 - x, y2 - y],
                }
            )
    return objects


def _frame_rect(row) -> tuple[int, int, int, int]:
    import numpy as np

    x1, y1, x2, y2 = (float(value) for value in row[:4])
    left = max(0, min(CAMERA_WIDTH - 1, int(np.floor(x1))))
    top = max(0, min(CAMERA_HEIGHT - 1, int(np.floor(y1))))
    right = max(left + 1, min(CAMERA_WIDTH, int(np.ceil(x2))))
    bottom = max(top + 1, min(CAMERA_HEIGHT, int(np.ceil(y2))))
    return left, top, right, bottom


def _mask_rect(
    frame_rect: tuple[int, int, int, int], mask_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    import numpy as np

    mask_height, mask_width = mask_shape[-2:]
    model_width = mask_width * MASK_STRIDE
    model_height = mask_height * MASK_STRIDE
    scale = min(model_width / CAMERA_WIDTH, model_height / CAMERA_HEIGHT)
    pad_x = (model_width - CAMERA_WIDTH * scale) * 0.5
    pad_y = (model_height - CAMERA_HEIGHT * scale) * 0.5
    left, top, right, bottom = frame_rect
    mask_left = max(
        0,
        min(mask_width - 1, int(np.floor((left * scale + pad_x) / MASK_STRIDE))),
    )
    mask_top = max(
        0,
        min(mask_height - 1, int(np.floor((top * scale + pad_y) / MASK_STRIDE))),
    )
    mask_right = max(
        mask_left + 1,
        min(mask_width, int(np.ceil((right * scale + pad_x) / MASK_STRIDE))),
    )
    mask_bottom = max(
        mask_top + 1,
        min(mask_height, int(np.ceil((bottom * scale + pad_y) / MASK_STRIDE))),
    )
    return mask_left, mask_top, mask_right, mask_bottom


def _mask_polygon(mask, frame_rect: tuple[int, int, int, int]) -> list[list[int]]:
    import cv2

    left, top, right, bottom = frame_rect
    mask_left, mask_top, mask_right, mask_bottom = _mask_rect(frame_rect, mask.shape)
    roi = cv2.resize(
        mask[mask_top:mask_bottom, mask_left:mask_right],
        (right - left, bottom - top),
        interpolation=cv2.INTER_LINEAR,
    )
    _, binary = cv2.threshold(roi, MASK_THRESHOLD * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    return [
        [int(point[0][0]) + left, int(point[0][1]) + top]
        for point in polygon
    ]


def insight_segments(pyneat, sample, labels: list[str]) -> list[dict]:
    """Convert decoded YOLO26 masks to Insight polygon segments."""
    import numpy as np

    segments = []
    decoded = pyneat.decode_segmentation(
        sample_tensors(sample),
        clamp_to=(CAMERA_WIDTH, CAMERA_HEIGHT),
        top_k=50,
        strict=False,
    )
    for item in decoded:
        boxes = np.asarray(item.boxes.to_numpy(copy=True), dtype=np.float32).reshape((-1, 6))
        masks_array = np.asarray(item.masks.to_numpy(copy=True), dtype=np.uint8)
        if masks_array.ndim < 3:
            continue
        masks = masks_array.reshape((-1, masks_array.shape[-2], masks_array.shape[-1]))
        for row, mask in zip(boxes, masks):
            frame_rect = _frame_rect(row)
            polygon = _mask_polygon(mask, frame_rect)
            if not polygon:
                continue
            left, top, right, bottom = frame_rect
            class_id = int(row[5])
            segments.append(
                {
                    "id": f"seg_{len(segments) + 1}",
                    "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                    "confidence": float(row[4]),
                    "bbox": [left, top, right - left, bottom - top],
                    "mask_format": "polygon",
                    "mask": polygon,
                }
            )
    return segments


def encode_segments(segments: list[dict]) -> tuple[str, int]:
    """Encode the strongest masks without exceeding Insight's UDP payload budget."""
    kept = []
    total_bytes = len('{"segments":[]}')
    for segment in sorted(segments, key=lambda item: item["confidence"], reverse=True):
        entry_bytes = len(json.dumps(segment, separators=(",", ":"))) + 1
        if total_bytes + entry_bytes > METADATA_BYTE_BUDGET:
            break
        total_bytes += entry_bytes
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":")), len(kept)


def make_metadata_sender(pyneat, host: str, port: int, channel: int):
    options = pyneat.MetadataSenderOptions()
    options.host = host
    options.channel = channel
    options.metadata_port_base = port
    return pyneat.MetadataSender(options)


def send_insight_metadata(sender, pyneat, profile: Profile, sample, labels: list[str]) -> int:
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    if profile.task == "segmentation":
        data_json, count = encode_segments(insight_segments(pyneat, sample, labels))
        metadata_type = "segmentation"
    else:
        objects = insight_objects(pyneat, profile, sample, labels)
        data_json = json.dumps({"objects": objects}, separators=(",", ":"))
        metadata_type = "object-detection"
        count = len(objects)
    sender.send_metadata(
        metadata_type,
        data_json,
        timestamp_ms,
        frame_id,
    )
    return count


def require_strict_zero_copy(backend: str) -> None:
    current_contract = "simaai-zero-copy-required=true" in backend
    legacy_040_contract = (
        "external-buffer-mode=required" in backend
        and "neatcamerabridge" in backend.lower()
        and "copy-allowed=false" in backend
    )
    if not (current_contract or legacy_040_contract):
        raise RuntimeError("camera backend does not require zero-copy")
    if "neatcamerabridge" in backend.lower() and "copy-allowed=false" not in backend:
        raise RuntimeError("camera bridge permits a CPU copy")


def _apply_overrides(config: AppConfig, args) -> AppConfig:
    profile = profile_named(args.profile) if args.profile else config.profile
    model_path = args.model or (
        config.model_path
        if profile == config.profile
        else DEFAULT_MODELS_DIR / profile.archive
    )
    return replace(
        config,
        profile=profile,
        model_path=model_path,
        frames=args.frames if args.frames is not None else config.frames,
        timeout_ms=args.timeout_ms if args.timeout_ms is not None else config.timeout_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--model", type=Path, help="override the selected profile's MPK")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--timeout-ms", type=int)
    parser.add_argument("--describe", action="store_true", help="print the negotiated backend")
    parser.add_argument(
        "--allow-camera-copy",
        action="store_true",
        help="allow Neat's camera bridge when strict DMA-BUF allocation is unavailable",
    )
    parser.add_argument("--insight-host", default="")
    parser.add_argument("--insight-video-port", type=int, default=9000)
    parser.add_argument("--insight-metadata-port", type=int, default=9100)
    parser.add_argument("--insight-channel", type=int, default=0)
    parser.add_argument("--insight-bitrate-kbps", type=int, default=2500)
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--validate-config-only", action="store_true")
    parser.add_argument("--validate-model-only", action="store_true")
    args = parser.parse_args(argv)

    if args.list_profiles:
        for profile in PROFILES.values():
            print(f"{profile.name:10} {profile.task:14} {profile.title} [{profile.source}]")
        return 0

    try:
        config = _apply_overrides(load_config(args.config), args)
        _positive_integer(config.frames, "frames")
        _positive_integer(config.timeout_ms, "timeout-ms")
        for name, value in (
            ("insight-video-port", args.insight_video_port),
            ("insight-metadata-port", args.insight_metadata_port),
            ("insight-bitrate-kbps", args.insight_bitrate_kbps),
        ):
            _positive_integer(value, name)
        if args.insight_channel < 0:
            raise ValueError("insight-channel must not be negative")
        if args.validate_config_only:
            print(f"config valid: {args.config}")
            return 0

        package = inspect_package(config.model_path, config.profile)
        print(f"profile={config.profile.name} package={package.path}")
        if args.validate_model_only:
            return 0

        try:
            import pyneat
        except ImportError as exc:
            raise RuntimeError(
                "pyneat is not importable; run: source ~/pyneat/bin/activate"
            ) from exc

        graph = make_graph(
            pyneat,
            config.profile,
            package,
            allow_camera_copy=args.allow_camera_copy,
            insight_host=args.insight_host,
            insight_video_port=args.insight_video_port,
            insight_channel=args.insight_channel,
            insight_bitrate_kbps=args.insight_bitrate_kbps,
        )
        backend = graph.describe_backend(False)
        if not args.allow_camera_copy and not args.insight_host:
            require_strict_zero_copy(backend)
        if args.describe:
            print(backend)

        run_options = pyneat.RunOptions()
        run_options.preset = pyneat.RunPreset.Realtime
        run_options.queue_depth = 4
        run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        run_options.output_memory = pyneat.OutputMemory.ZeroCopy
        run_options.advanced.copy_input = False
        metadata_sender = None
        labels = []
        if args.insight_host:
            labels = load_labels()
            metadata_sender = make_metadata_sender(
                pyneat,
                args.insight_host,
                args.insight_metadata_port,
                args.insight_channel,
            )
            print(
                f"insight={args.insight_host} channel={args.insight_channel} "
                f"video={args.insight_video_port + args.insight_channel} "
                f"metadata={args.insight_metadata_port + args.insight_channel}"
            )
        run = graph.build(run_options)
        try:
            frame = 0
            while args.continuous or frame < config.frames:
                sample = run.pull(OUTPUT_NAME, config.timeout_ms)
                if sample is None:
                    raise TimeoutError(
                        run.last_error() or f"timed out waiting for {config.profile.name}"
                    )
                summary = summarize_output(pyneat, config.profile, sample)
                metadata_count = None
                if metadata_sender is not None:
                    metadata_count = send_insight_metadata(
                        metadata_sender,
                        pyneat,
                        config.profile,
                        sample,
                        labels,
                    )
                if not args.continuous or frame % CAMERA_FPS == 0:
                    metadata_summary = (
                        f" metadata_objects={metadata_count}"
                        if metadata_count is not None
                        else ""
                    )
                    print(
                        f"frame={frame} profile={config.profile.name} "
                        f"{summary}{metadata_summary}",
                        flush=True,
                    )
                frame += 1
        finally:
            run.close()

        camera_mode = "camera-bridge" if args.allow_camera_copy else "strict-zero-copy"
        destination = " + Insight" if args.insight_host else ""
        print(
            f"PASS {camera_mode} MIPI -> {config.profile.title} -> "
            f"{OUTPUT_NAME}{destination}"
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except (ModelPackageError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
