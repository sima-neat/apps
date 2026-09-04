"""Single-stream RF-DETR detection and segmentation for Neat Insight."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import yaml

NUM_CLASSES = 91
MASK_SIZE = 108
METADATA_BYTE_BUDGET = 32_768


@dataclass(frozen=True)
class Config:
    task: str
    variant: str
    backbone: str
    transformer: str
    input_size: int
    feature_size: int
    top_k: int
    labels: Path
    rtsp_url: str
    codec: str
    tcp: bool
    latency_ms: int
    width: int
    height: int
    fps: int
    frames: int
    min_score: float
    max_results: int
    mask_threshold: float
    insight_host: str
    video_port: int
    metadata_port: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RF-DETR RTSP detection and segmentation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def _mapping(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def parse_source_codec(value: str) -> str:
    codec = value.lower()
    if codec in {"h264", "avc", "h.264"}:
        return "h264"
    if codec in {"h265", "hevc", "h.265"}:
        return "h265"
    if codec in {"mjpeg", "jpeg"}:
        return "mjpeg"
    raise ValueError("source.codec must be h264/avc, h265/hevc, or mjpeg")


def load_config(path: Path) -> Config:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = _mapping(raw, "model")
    source = _mapping(raw, "source")
    inference = _mapping(raw, "inference")
    insight = _mapping(_mapping(raw, "output"), "insight")
    task = str(model.get("task", "detection")).lower()
    if task not in {"detection", "segmentation"}:
        raise ValueError("model.task must be detection or segmentation")
    if task == "detection":
        detection = _mapping(model, "detection")
        variant = str(detection.get("variant", "small")).lower()
        if variant not in {"small", "medium"}:
            raise ValueError("model.detection.variant must be small or medium")
        selected = _mapping(detection, variant)
        inference_options = _mapping(inference, "detection")
        input_size, feature_size, top_k = {
            "small": (512, 32, 300),
            "medium": (576, 36, 300),
        }[variant]
        max_results_key = "max_detections"
        default_score = 0.5
        default_max_results = 100
    else:
        variant = "segmentation"
        selected = _mapping(model, "segmentation")
        inference_options = _mapping(inference, "segmentation")
        input_size, feature_size, top_k = 432, 36, 200
        max_results_key = "max_segments"
        default_score = 0.3
        default_max_results = 24
    labels_path = str(model.get("labels", ""))
    cfg = Config(
        task=task,
        variant=variant,
        backbone=str(selected.get("backbone", "")),
        transformer=str(selected.get("transformer", "")),
        input_size=input_size,
        feature_size=feature_size,
        top_k=top_k,
        labels=Path(labels_path),
        rtsp_url=str(source.get("rtsp_url", "")),
        codec=parse_source_codec(str(source.get("codec", "h264"))),
        tcp=bool(source.get("tcp", True)),
        latency_ms=int(source.get("latency_ms", 100)),
        width=int(source.get("width", 0)),
        height=int(source.get("height", 0)),
        fps=int(source.get("fps", 0)),
        frames=int(inference.get("frames", 0)),
        min_score=float(inference_options.get("min_score", default_score)),
        max_results=int(inference_options.get(max_results_key, default_max_results)),
        mask_threshold=float(inference_options.get("mask_threshold", 0.08)),
        insight_host=str(insight.get("host", "")),
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
    )
    if not cfg.backbone or not cfg.transformer:
        raise ValueError(f"model.{task} backbone and transformer must be set")
    if not labels_path:
        raise ValueError("model.labels must be set")
    if not cfg.rtsp_url.startswith("rtsp://"):
        raise ValueError("source.rtsp_url must be an RTSP URL")
    if cfg.latency_ms < 0 or cfg.frames < 0:
        raise ValueError("source.latency_ms and inference.frames must be >= 0")
    if cfg.width < 0 or cfg.height < 0 or cfg.fps < 0:
        raise ValueError("source.width, source.height, and source.fps must be >= 0")
    if not 0.0 <= cfg.min_score <= 1.0:
        raise ValueError(f"inference.{task}.min_score must be in [0, 1]")
    if cfg.max_results <= 0:
        raise ValueError(f"inference.{task}.{max_results_key} must be > 0")
    if not 0.0 <= cfg.mask_threshold <= 1.0:
        raise ValueError("inference.segmentation.mask_threshold must be in [0, 1]")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not all(0 < port <= 65535 for port in (cfg.video_port, cfg.metadata_port)):
        raise ValueError("Insight ports must be in [1, 65535]")
    return cfg


def _probe_fps(value: str) -> int:
    try:
        fps = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0
    return round(fps) if fps > 0 else 0


def probe_rtsp(cfg: Config) -> tuple[int, int, int]:
    command = [
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
    if cfg.tcp:
        command.extend(["-rtsp_transport", "tcp"])
    command.append(cfg.rtsp_url)
    width = height = fps = 0
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        values = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        fps = _probe_fps(values.get("avg_frame_rate", "")) or _probe_fps(
            values.get("r_frame_rate", "")
        )
        try:
            width = int(values.get("width", 0))
            height = int(values.get("height", 0))
        except ValueError:
            width = height = 0

    if width <= 0 or height <= 0 or fps <= 0:
        capture = cv2.VideoCapture(cfg.rtsp_url)
        if capture.isOpened():
            width = width or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = height or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = fps or round(capture.get(cv2.CAP_PROP_FPS) or 0)
        capture.release()
    return width, height, fps


def resolve_geometry(
    probed: tuple[int, int, int], configured: tuple[int, int, int]
) -> tuple[int, int, int]:
    return (
        probed[0] if probed[0] > 0 else configured[0],
        probed[1] if probed[1] > 0 else configured[1],
        configured[2] if configured[2] > 0 else probed[2],
    )


def probe_source_geometry(cfg: Config) -> tuple[int, int, int]:
    geometry = resolve_geometry(probe_rtsp(cfg), (cfg.width, cfg.height, cfg.fps))
    if cfg.codec == "mjpeg" and geometry[2] <= 0:
        raise RuntimeError("MJPEG source did not provide an FPS; set source.fps")
    if any(value <= 0 for value in geometry):
        raise RuntimeError(
            "failed to resolve RTSP width, height, and FPS; set source fallbacks if probing fails"
        )
    return geometry


def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(labels) != 91 or any(not label for label in labels):
        raise ValueError("model.labels must contain exactly 91 non-empty COCO labels")
    return labels


def stable_topk_gather(
    scores: np.ndarray, proposals: np.ndarray, top_k: int
) -> np.ndarray:
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    flat_proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
    if flat_scores.size != flat_proposals.shape[0] or flat_scores.size < top_k:
        raise ValueError("backbone score and proposal shapes do not match")
    if not np.isfinite(flat_scores).all() or not np.isfinite(flat_proposals).all():
        raise ValueError("backbone output contains non-finite values")
    indices = np.argsort(-flat_scores, kind="stable")[:top_k]
    return flat_proposals[indices].reshape(1, top_k, 4)


def postprocess(
    boxes: np.ndarray,
    logits: np.ndarray,
    width: int,
    height: int,
    labels: list[str],
    min_score: float,
    max_detections: int,
    top_k: int = 300,
) -> list[dict]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(top_k, 4)
    logits = np.asarray(logits, dtype=np.float32).reshape(top_k, NUM_CLASSES)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    ranked = np.argsort(-probabilities.reshape(-1), kind="stable")
    objects: list[dict] = []
    for flat_index in ranked:
        score = float(probabilities.reshape(-1)[flat_index])
        if score < min_score or len(objects) >= max_detections:
            break
        query, class_id = divmod(int(flat_index), NUM_CLASSES)
        if class_id == 0 or labels[class_id] == "unused":
            continue
        cx, cy, box_w, box_h = (float(value) for value in boxes[query])
        x = max(0.0, min((cx - box_w / 2.0) * width, float(width)))
        y = max(0.0, min((cy - box_h / 2.0) * height, float(height)))
        x2 = max(x, min((cx + box_w / 2.0) * width, float(width)))
        y2 = max(y, min((cy + box_h / 2.0) * height, float(height)))
        objects.append(
            {
                "id": f"obj_{len(objects) + 1}",
                "label": labels[class_id],
                "confidence": score,
                "bbox": [x, y, x2 - x, y2 - y],
            }
        )
    return objects


def segmentation_metadata(
    boxes: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    width: int,
    height: int,
    labels: list[str],
    min_score: float,
    max_segments: int,
    mask_threshold: float,
) -> str:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(200, 4)
    logits = np.asarray(logits, dtype=np.float32).reshape(200, NUM_CLASSES)
    masks = np.asarray(masks, dtype=np.float32).reshape(MASK_SIZE, MASK_SIZE, 200)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    classes = np.argmax(probabilities, axis=1)
    scores = probabilities[np.arange(200), classes]
    queries = np.argsort(-scores, kind="stable")

    segments: list[dict] = []
    for query in queries:
        class_id = int(classes[query])
        score = float(scores[query])
        if score < min_score or len(segments) >= max_segments:
            break
        if class_id == 0 or labels[class_id] == "unused":
            continue

        cx, cy, box_width, box_height = (float(value) for value in boxes[query])
        x0 = max(0, min(width - 1, round((cx - box_width / 2.0) * width)))
        y0 = max(0, min(height - 1, round((cy - box_height / 2.0) * height)))
        x1 = max(x0 + 1, min(width, round((cx + box_width / 2.0) * width)))
        y1 = max(y0 + 1, min(height, round((cy + box_height / 2.0) * height)))

        mx0 = max(0, min(MASK_SIZE - 1, int(np.floor(x0 * MASK_SIZE / width))))
        my0 = max(0, min(MASK_SIZE - 1, int(np.floor(y0 * MASK_SIZE / height))))
        mx1 = max(mx0 + 1, min(MASK_SIZE, int(np.ceil(x1 * MASK_SIZE / width))))
        my1 = max(my0 + 1, min(MASK_SIZE, int(np.ceil(y1 * MASK_SIZE / height))))
        mask = 1.0 / (1.0 + np.exp(-np.clip(masks[my0:my1, mx0:mx1, query], -80.0, 80.0)))
        mask = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        binary = (mask >= mask_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
        if len(polygon) < 3:
            continue
        points = [
            [int(point[0][0]) + x0, int(point[0][1]) + y0]
            for point in polygon
        ]
        segments.append(
            {
                "id": f"seg_{len(segments) + 1}",
                "label": labels[class_id],
                "confidence": score,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "mask_format": "polygon",
                "mask": points,
            }
        )

    kept: list[dict] = []
    size = len('{"segments":[]}')
    for segment in segments:
        entry_size = len(json.dumps(segment, separators=(",", ":"))) + 1
        if size + entry_size > METADATA_BYTE_BUDGET:
            break
        size += entry_size
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":"))


def collect_tensors(sample) -> list:
    if sample is None:
        return []
    tensors = []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        tensors.append(sample.tensor)
    elif sample.kind == pyneat.SampleKind.TensorSet:
        tensors.extend(sample.tensors)
    for field in sample.fields:
        tensors.extend(collect_tensors(field))
    return tensors


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def split_backbone(sample, proposal_count: int):
    feature = scores = proposals = None
    for tensor in collect_tensors(sample):
        shape = _shape(tensor)
        elements = int(np.prod(shape))
        if len(shape) >= 3 and shape[-1] == 256:
            feature = tensor
        elif shape[-1:] == (4,) and elements == proposal_count * 4:
            proposals = tensor
        elif elements == proposal_count:
            scores = tensor
    if feature is None or scores is None or proposals is None:
        raise RuntimeError("backbone did not produce feature, score, and proposal tensors")
    return feature, scores, proposals


def split_transformer(sample, cfg: Config):
    boxes = logits = masks = None
    for tensor in collect_tensors(sample):
        shape = _shape(tensor)
        elements = int(np.prod(shape))
        if elements == cfg.top_k * 4:
            boxes = tensor
        elif elements == cfg.top_k * NUM_CLASSES:
            logits = tensor
        elif cfg.task == "segmentation" and elements == cfg.top_k * MASK_SIZE * MASK_SIZE:
            masks = tensor
    if boxes is None or logits is None or (cfg.task == "segmentation" and masks is None):
        raise RuntimeError("transformer did not produce the expected task outputs")
    return boxes, logits, masks


def copy_identity(source, target) -> None:
    for name in (
        "frame_id", "stream_id", "stream_label", "input_seq", "orig_input_seq",
        "pts_ns", "dts_ns", "duration_ns",
    ):
        setattr(target, name, getattr(source, name))
    target.attributes = source.attributes


def identity_key(sample) -> int:
    return sample.frame_id if sample.frame_id >= 0 else sample.input_seq


def transformer_inputs(model, feature, gathered, top_k: int) -> list:
    ordered = []
    for spec in model.input_specs():
        expected = tuple(int(value) for value in spec.shape)
        elements = int(np.prod(expected))
        tensor = gathered if elements == top_k * 4 else feature
        current = _shape(tensor)
        if current == (1, *expected):
            tensor.shape = list(expected)
            if len(tensor.strides_bytes) == len(current):
                tensor.strides_bytes = list(tensor.strides_bytes)[1:]
            if len(tensor.axis_semantics) == len(current):
                tensor.axis_semantics = list(tensor.axis_semantics)[1:]
        elif current != expected:
            raise RuntimeError(
                f"transformer input shape mismatch: expected {expected}, got {current}"
            )
        ordered.append(tensor)
    if len(ordered) != 2 or not any(item is feature for item in ordered) or not any(
        item is gathered for item in ordered
    ):
        raise RuntimeError("unexpected transformer input contract")
    return ordered


def run(cfg: Config) -> int:
    global pyneat
    import pyneat

    labels = load_labels(cfg.labels)
    width, height, fps = probe_source_geometry(cfg)
    if cfg.codec == "h264":
        source_codec = pyneat.RtspCodec.H264
        decoder_type = pyneat.SimaDecodeType.H264
    elif cfg.codec == "h265":
        source_codec = pyneat.RtspCodec.H265
        decoder_type = pyneat.SimaDecodeType.H265
    else:
        source_codec = pyneat.RtspCodec.MJPEG
        decoder_type = pyneat.SimaDecodeType.MJPEG

    backbone_options = pyneat.ModelOptions()
    backbone_options.preprocess.kind = pyneat.InputKind.Image
    backbone_options.preprocess.enable = pyneat.AutoFlag.On
    backbone_options.preprocess.input_max_width = width
    backbone_options.preprocess.input_max_height = height
    backbone_options.preprocess.input_max_depth = 3
    backbone_options.preprocess.resize.enable = pyneat.AutoFlag.On
    backbone_options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    backbone_options.preprocess.color_convert.enable = pyneat.AutoFlag.On
    backbone_options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    backbone_options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    if cfg.task == "segmentation":
        backbone_options.preprocess.resize.width = cfg.input_size
        backbone_options.preprocess.resize.height = cfg.input_size
        backbone_options.preprocess.resize.scaling_type = "BILINEAR"
        backbone_options.preprocess.normalize.enable = pyneat.AutoFlag.On
        backbone_options.preprocess.normalize.mean = [0.485, 0.456, 0.406]
        backbone_options.preprocess.normalize.stddev = [0.229, 0.224, 0.225]
        backbone_options.preprocess.normalize.has_explicit_stats = True
    else:
        backbone_options.preprocess.preset = pyneat.NormalizePreset.ImageNet
    backbone_options.processcvu.pre_run_target = "EV74"
    backbone_options.processcvu.post_run_target = "A65"
    backbone = pyneat.Model(cfg.backbone, backbone_options)

    transformer_options = pyneat.ModelOptions()
    transformer_options.preprocess.kind = pyneat.InputKind.Tensor
    transformer_options.preprocess.enable = pyneat.AutoFlag.Off
    transformer_options.processcvu.pre_run_target = "A65"
    transformer_options.processcvu.post_run_target = "A65"
    transformer = pyneat.Model(cfg.transformer, transformer_options)

    side = cfg.feature_size
    transformer_outputs = [[1, cfg.top_k, 4], [1, cfg.top_k, NUM_CLASSES]]
    if cfg.task == "segmentation":
        transformer_outputs.append([MASK_SIZE, MASK_SIZE, cfg.top_k])
    expected_shapes = (
        [[1, side, side, 256], [1, side * side], [1, side * side, 4]],
        [[side, side, 256], [1, cfg.top_k, 4]],
        transformer_outputs,
    )
    actual_specs = (
        backbone.output_specs(),
        transformer.input_specs(),
        transformer.output_specs(),
    )
    backbone_inputs = backbone.input_specs()
    valid_contract = (
        len(backbone_inputs) == 1
        and list(backbone_inputs[0].shape) == [-1, -1, 3]
        and list(backbone_inputs[0].dtypes) == [pyneat.TensorDType.UInt8]
    )
    for specs, shapes in zip(actual_specs, expected_shapes, strict=True):
        if [list(spec.shape) for spec in specs] != shapes or any(
            list(spec.dtypes) != [pyneat.TensorDType.Float32] for spec in specs
        ):
            valid_contract = False
    if not valid_contract:
        raise RuntimeError("selected RF-DETR model pair has an unexpected I/O contract")

    encoded_options = pyneat.RtspEncodedInputOptions()
    encoded_options.url = cfg.rtsp_url
    encoded_options.codec = source_codec
    encoded_options.latency_ms = cfg.latency_ms
    encoded_options.tcp = cfg.tcp
    encoded_options.source_fps = fps
    if cfg.codec == "h264":
        encoded_options.fallback_h264_width = width
        encoded_options.fallback_h264_height = height
    source = pyneat.groups.rtsp_encoded_input(encoded_options)

    decode_options = pyneat.SimaDecodeOptions()
    decode_options.type = decoder_type
    decode_options.out_format = pyneat.Format.NV12
    decode_options.raw_output = True
    decode_options.dec_width = width
    decode_options.dec_height = height
    decode_options.dec_fps = fps
    decoder = pyneat.Graph("decoder")
    decoder.add(pyneat.nodes.sima_decode(decode_options))

    video_options = (
        pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
        if cfg.codec == "mjpeg"
        else pyneat.VideoSenderOptions.passthrough(source_codec)
    )
    video_options.host = cfg.insight_host
    video_options.video_port_base = cfg.video_port
    video_options.channel = 0
    video_options.async_ = True
    video = pyneat.groups.video_sender(video_options)

    queue_options = pyneat.QueueOptions()
    queue_options.max_buffers = 1
    queue_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    inference_graph = pyneat.Graph("inference")
    inference_graph.add(pyneat.nodes.queue(queue_options))
    inference_graph.add(backbone.graph())
    backbone_output = pyneat.Graph("backbone_output")
    backbone_output.add(pyneat.nodes.output("backbone", pyneat.OutputOptions.latest()))
    inference_graph.add(backbone_output)

    source_graph = pyneat.Graph("rfdetr_source")
    source_graph.connect(source, decoder)
    if cfg.codec == "mjpeg":
        source_graph.connect(decoder, video)
    else:
        source_graph.connect(source, video)
    source_graph.connect(decoder, inference_graph)

    transformer_run_options = pyneat.RunOptions()
    transformer_run_options.preset = pyneat.RunPreset.Realtime
    transformer_run_options.queue_depth = 1
    transformer_run_options.overflow_policy = pyneat.OverflowPolicy.Block
    transformer_run_options.output_memory = pyneat.OutputMemory.Owned
    dummy_inputs = [
        pyneat.Tensor.from_numpy(
            np.zeros(tuple(int(value) for value in spec.shape), dtype=np.float32),
            copy=True,
            memory=pyneat.TensorMemory.EV74,
        )
        for spec in transformer.input_specs()
    ]
    transformer_runner = transformer.build(
        dummy_inputs,
        route_options=pyneat.ModelRouteOptions(),
        run_options=transformer_run_options,
    )

    source_run_options = pyneat.RunOptions()
    source_run_options.preset = pyneat.RunPreset.Realtime
    source_run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    source_run_options.advanced.prepare_output_cpu_visible = True
    source_run = source_graph.build(source_run_options)
    video_port = video_options.video_port
    metadata_options = pyneat.MetadataSenderOptions()
    metadata_options.host = cfg.insight_host
    metadata_options.metadata_port_base = cfg.metadata_port
    metadata_options.channel = 0
    metadata_sender = pyneat.MetadataSender(metadata_options)
    print(
        f"RF-DETR {cfg.task} {cfg.variant} {cfg.codec}: {cfg.rtsp_url} "
        f"({width}x{height}@{fps}) -> "
        f"Insight video={video_port} metadata={metadata_sender.metadata_port()}",
        flush=True,
    )

    stop = threading.Event()
    bridge_error: list[BaseException] = []
    identity_lock = threading.Lock()
    source_pts: dict[int, int] = {}
    proposal_count = cfg.feature_size**2

    def transformer_bridge() -> None:
        try:
            while not stop.is_set():
                sample = source_run.pull("backbone", 500)
                if sample is None:
                    continue
                feature, scores, proposals = split_backbone(sample, proposal_count)
                gathered = stable_topk_gather(
                    scores.to_numpy(copy=False), proposals.to_numpy(copy=False), cfg.top_k
                )
                gathered_tensor = pyneat.Tensor.from_numpy(
                    gathered, copy=True, memory=pyneat.TensorMemory.EV74
                )
                transformer_sample = pyneat.Sample()
                transformer_sample.kind = pyneat.SampleKind.TensorSet
                transformer_sample.tensors = transformer_inputs(
                    transformer, feature, gathered_tensor, cfg.top_k
                )
                copy_identity(sample, transformer_sample)
                key = identity_key(sample)
                with identity_lock:
                    source_pts[key] = sample.pts_ns
                    if len(source_pts) > 8:
                        source_pts.pop(next(iter(source_pts)))
                if not transformer_runner.push_samples(transformer_sample):
                    if not stop.is_set():
                        raise RuntimeError("transformer input closed")
                    break
        except BaseException as exc:
            if not stop.is_set():
                bridge_error.append(exc)
            stop.set()

    previous_handlers = {
        signum: signal.signal(signum, lambda *_: stop.set())
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    transformer_worker = threading.Thread(
        target=transformer_bridge, name="rfdetr-transformer", daemon=True
    )
    transformer_worker.start()
    processed = 0
    first_completed_at = None
    last_completed_at = None
    try:
        while not stop.is_set() and (cfg.frames == 0 or processed < cfg.frames):
            sample = transformer_runner.pull(timeout_ms=500)
            tensors = collect_tensors(sample)
            if not tensors:
                continue
            box_tensor, logit_tensor, mask_tensor = split_transformer(sample, cfg)
            if cfg.task == "detection":
                data = json.dumps(
                    {
                        "objects": postprocess(
                            box_tensor.to_numpy(copy=False),
                            logit_tensor.to_numpy(copy=False),
                            width,
                            height,
                            labels,
                            cfg.min_score,
                            cfg.max_results,
                            cfg.top_k,
                        )
                    },
                    separators=(",", ":"),
                )
                metadata_type = "object-detection"
            else:
                data = segmentation_metadata(
                    box_tensor.to_numpy(copy=False),
                    logit_tensor.to_numpy(copy=False),
                    mask_tensor.to_numpy(copy=False),
                    width,
                    height,
                    labels,
                    cfg.min_score,
                    cfg.max_results,
                    cfg.mask_threshold,
                )
                metadata_type = "segmentation"
            source_frame_id = sample.frame_id
            with identity_lock:
                source_pts_ns = source_pts.pop(identity_key(sample), sample.pts_ns)
            timestamp_ms = source_pts_ns // 1_000_000 if source_pts_ns >= 0 else -1
            frame_id = str(source_frame_id) if source_frame_id >= 0 else ""
            if not metadata_sender.send_metadata(
                metadata_type,
                data,
                timestamp_ms,
                frame_id,
            ):
                print("[warn] Insight metadata send failed", file=sys.stderr)
            processed += 1
            last_completed_at = time.monotonic()
            if first_completed_at is None:
                first_completed_at = last_completed_at
        if bridge_error:
            raise bridge_error[0]
    finally:
        stop.set()
        source_run.stop()
        transformer_runner.close()
        transformer_worker.join(timeout=5)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        _ = (backbone, source_graph)
    elapsed = (last_completed_at - first_completed_at) if processed > 1 else 0.0
    output_fps = (processed - 1) / elapsed if elapsed > 0.0 else 0.0
    print(
        f"RF-DETR {cfg.task}: completed={processed} output_fps={output_fps:.1f}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_config(args.config)
        if args.validate_config_only:
            load_labels(cfg.labels)
            print(f"RF-DETR {cfg.task} {cfg.variant} configuration is valid")
            return 0
        return run(cfg)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
