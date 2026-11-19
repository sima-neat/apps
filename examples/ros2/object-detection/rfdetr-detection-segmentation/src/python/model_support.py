"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

import argparse

import json

import math

import signal

import subprocess

import sys

import threading

import time

from dataclasses import dataclass

from fractions import Fraction

from pathlib import Path
from ament_index_python.packages import get_package_share_directory

import cv2

import numpy as np

import yaml

NUM_CLASSES = 91

CLASSIFICATION_TOP_K = 300

METADATA_BYTE_BUDGET = 32_768

@dataclass(frozen=True)
class Config:
    task: str
    variant: str
    backbone: str
    transformer: str
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
    mask_grid_size: int
    insight_host: str
    video_port: int
    metadata_port: int

@dataclass(frozen=True)
class ModelContract:
    """Sizes read from the loaded model pair; they differ between RF-DETR sizes."""

    side: int
    top_k: int
    mask_size: int

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
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    model = _mapping(raw, "model")
    source = _mapping(raw, "source")
    inference = _mapping(raw, "inference")
    insight = _mapping(_mapping(raw, "output"), "insight")
    task = str(model.get("task", "detection")).lower()
    if task not in {"detection", "segmentation"}:
        raise ValueError("model.task must be detection or segmentation")
    task_models = _mapping(model, task)
    variant = str(task_models.get("variant", "small" if task == "detection" else "medium")).lower()
    selected = _mapping(task_models, variant)
    inference_options = _mapping(inference, task)
    if task == "detection":
        max_results_key = "max_detections"
        default_score = 0.5
        default_max_results = 100
    else:
        max_results_key = "max_segments"
        default_score = 0.3
        default_max_results = 24
    labels_path = str(model.get("labels", ""))
    cfg = Config(
        task=task,
        variant=variant,
        backbone=str(selected.get("backbone", "")),
        transformer=str(selected.get("transformer", "")),
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
        mask_grid_size=int(inference_options.get("mask_grid_size", 640)),
        insight_host=str(insight.get("host", "")),
        video_port=int(insight.get("video_port", 9000)),
        metadata_port=int(insight.get("metadata_port", 9100)),
    )
    if not cfg.backbone or not cfg.transformer:
        raise ValueError(f"model.{task}.{variant} backbone and transformer must be set")
    if not labels_path:
        raise ValueError("model.labels must be set")
    # Capture validation belongs to input_node.py; source=none needs no RTSP URL.
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
    if cfg.mask_grid_size <= 0:
        raise ValueError("inference.segmentation.mask_grid_size must be > 0")
    if not cfg.insight_host:
        raise ValueError("output.insight.host must be set")
    if not all(0 < port <= 65535 for port in (cfg.video_port, cfg.metadata_port)):
        raise ValueError("Insight ports must be in [1, 65535]")
    return cfg

def load_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if len(labels) != 91 or any(not label for label in labels):
        raise ValueError("model.labels must contain exactly 91 non-empty COCO labels")
    return labels

def stable_topk_indices(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = min(count, values.size)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.argpartition(-values, count - 1)[:count]
    cutoff = values[candidates].min()
    selected = np.concatenate(
        (np.flatnonzero(values > cutoff), np.flatnonzero(values == cutoff)[:count])
    )[:count]
    return selected[np.lexsort((selected, -values[selected]))]

def stable_topk_gather(
    scores: np.ndarray, proposals: np.ndarray, top_k: int
) -> np.ndarray:
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    flat_proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
    if flat_scores.size != flat_proposals.shape[0] or flat_scores.size < top_k:
        raise ValueError("backbone score and proposal shapes do not match")
    if not np.isfinite(flat_scores).all() or not np.isfinite(flat_proposals).all():
        raise ValueError("backbone output contains non-finite values")
    indices = stable_topk_indices(flat_scores, top_k)
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
    flat_probabilities = probabilities.reshape(-1)
    ranked = stable_topk_indices(flat_probabilities, CLASSIFICATION_TOP_K)
    objects: list[dict] = []
    for flat_index in ranked:
        score = float(flat_probabilities[flat_index])
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
    mask_grid_size: int = 640,
    top_k: int = 200,
    mask_size: int = 108,
) -> str:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(top_k, 4)
    logits = np.asarray(logits, dtype=np.float32).reshape(top_k, NUM_CLASSES)
    masks = np.asarray(masks, dtype=np.float32).reshape(mask_size, mask_size, top_k)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    flat_probabilities = probabilities.reshape(-1)
    ranked = stable_topk_indices(flat_probabilities, CLASSIFICATION_TOP_K)

    segments: list[dict] = []
    for flat_index in ranked:
        score = float(flat_probabilities[flat_index])
        if score < min_score or len(segments) >= max_segments:
            break
        query, class_id = divmod(int(flat_index), NUM_CLASSES)
        if class_id == 0 or labels[class_id] == "unused":
            continue

        cx, cy, box_width, box_height = (float(value) for value in boxes[query])
        x0 = max(0, min(width - 1, round((cx - box_width / 2.0) * width)))
        y0 = max(0, min(height - 1, round((cy - box_height / 2.0) * height)))
        x1 = max(x0 + 1, min(width, round((cx + box_width / 2.0) * width)))
        y1 = max(y0 + 1, min(height, round((cy + box_height / 2.0) * height)))

        mx0 = max(0, min(mask_size - 1, int(np.floor(x0 * mask_size / width))))
        my0 = max(0, min(mask_size - 1, int(np.floor(y0 * mask_size / height))))
        mx1 = max(mx0 + 1, min(mask_size, int(np.ceil(x1 * mask_size / width))))
        my1 = max(my0 + 1, min(mask_size, int(np.ceil(y1 * mask_size / height))))
        mask = 1.0 / (1.0 + np.exp(-np.clip(masks[my0:my1, mx0:mx1, query], -80.0, 80.0)))
        if mask_grid_size != mask_size:
            size = tuple((n * mask_grid_size + mask_size - 1) // mask_size for n in mask.shape[::-1])
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        binary = (mask >= mask_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        polygon = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
        if len(polygon) < 3:
            continue
        scale_x = (x1 - x0 - 1) / max(binary.shape[1] - 1, 1)
        scale_y = (y1 - y0 - 1) / max(binary.shape[0] - 1, 1)
        points = [
            [
                min(x1 - 1, x0 + round(int(point[0][0]) * scale_x)),
                min(y1 - 1, y0 + round(int(point[0][1]) * scale_y)),
            ]
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
        elements = math.prod(shape)
        if len(shape) >= 3 and shape[-1] == 256:
            feature = tensor
        elif shape[-1:] == (4,) and elements == proposal_count * 4:
            proposals = tensor
        elif elements == proposal_count:
            scores = tensor
    if feature is None or scores is None or proposals is None:
        raise RuntimeError("backbone did not produce feature, score, and proposal tensors")
    return feature, scores, proposals

def split_transformer(sample, cfg: Config, contract: ModelContract):
    boxes = logits = masks = None
    top_k, mask_size = contract.top_k, contract.mask_size
    for tensor in collect_tensors(sample):
        shape = _shape(tensor)
        elements = math.prod(shape)
        if elements == top_k * 4:
            boxes = tensor
        elif elements == top_k * NUM_CLASSES:
            logits = tensor
        elif cfg.task == "segmentation" and elements == top_k * mask_size * mask_size:
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

def transformer_inputs(input_shapes, feature, gathered, top_k: int) -> list:
    # Keep SiMa-backed features zero-copy; Core may return a CPU-owned output.
    if feature.storage.kind in (pyneat.StorageKind.CpuOwned, pyneat.StorageKind.CpuExternal):
        feature = feature.cvu()
    ordered = []
    for expected in input_shapes:
        elements = math.prod(expected)
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

def build_backbone(cfg: Config, width: int, height: int):
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.enable = pyneat.AutoFlag.On
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    opt.preprocess.color_convert.enable = pyneat.AutoFlag.On
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    opt.processcvu.pre_run_target = "EV74"
    opt.processcvu.post_run_target = "A65"
    return pyneat.Model(cfg.backbone, opt)

def build_transformer(cfg: Config):
    opt = pyneat.ModelOptions()
    opt.processcvu.pre_run_target = "A65"
    opt.processcvu.post_run_target = "A65"
    return pyneat.Model(cfg.transformer, opt)

def model_contract(cfg: Config, backbone, transformer) -> ModelContract:
    """Read the feature grid, query count, and mask size that differ between model sizes."""
    outputs = backbone.output_specs()
    if not outputs or len(outputs[0].shape) != 4:
        raise RuntimeError("RF-DETR backbone must output a feature grid")
    side = int(outputs[0].shape[1])
    inputs = transformer.input_specs()
    proposals = [list(spec.shape) for spec in inputs if list(spec.shape)[-1:] == [4]]
    top_k = int(proposals[0][-2]) if len(proposals) == 1 and len(proposals[0]) >= 2 else 0
    mask_size = 0
    if cfg.task == "segmentation":
        masks = [list(spec.shape) for spec in transformer.output_specs() if len(spec.shape) == 4]
        mask_size = int(masks[0][1]) if len(masks) == 1 else 0
    contract = ModelContract(side=side, top_k=top_k, mask_size=mask_size)
    require_model_contract(cfg, backbone, transformer, contract)
    if cfg.task == "segmentation" and cfg.mask_grid_size < mask_size:
        raise ValueError(
            f"inference.segmentation.mask_grid_size must be >= the model's mask size ({mask_size})"
        )
    return contract

def require_model_contract(cfg: Config, backbone, transformer, contract: ModelContract) -> None:
    side, top_k = contract.side, contract.top_k
    transformer_outputs = [[1, top_k, 4], [1, top_k, NUM_CLASSES]]
    if cfg.task == "segmentation":
        transformer_outputs.append([1, contract.mask_size, contract.mask_size, top_k])
    expected_shapes = (
        [[1, side, side, 256], [1, side * side], [1, side * side, 4]],
        [[1, side, side, 256], [1, top_k, 4]],
        transformer_outputs,
    )
    actual_specs = (
        backbone.output_specs(),
        transformer.input_specs(),
        transformer.output_specs(),
    )
    backbone_inputs = backbone.input_specs()
    valid_contract = (
        top_k > 0
        and (cfg.task == "detection" or contract.mask_size > 0)
        and len(backbone_inputs) == 1
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

def transformer_seed(transformer) -> list:
    return [
        pyneat.Tensor.from_numpy(
            np.zeros(tuple(int(value) for value in spec.shape), dtype=np.float32),
            copy=True,
            memory=pyneat.TensorMemory.EV74,
        )
        for spec in transformer.input_specs()
    ]

def transformer_run_options():
    """Every completed backbone result reaches the transformer; the bridge blocks when full."""
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Balanced
    run_options.queue_depth = 4
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.output_memory = pyneat.OutputMemory.Owned
    return run_options

def result_metadata(
    cfg: Config, contract: ModelContract, sample, width: int, height: int, labels: list[str]
):
    box_tensor, logit_tensor, mask_tensor = split_transformer(sample, cfg, contract)
    if cfg.task == "detection":
        objects = postprocess(
            box_tensor.to_numpy(copy=False),
            logit_tensor.to_numpy(copy=False),
            width,
            height,
            labels,
            cfg.min_score,
            cfg.max_results,
            contract.top_k,
        )
        return "object-detection", json.dumps({"objects": objects}, separators=(",", ":"))
    return "segmentation", segmentation_metadata(
        box_tensor.to_numpy(copy=False),
        logit_tensor.to_numpy(copy=False),
        mask_tensor.to_numpy(copy=False),
        width,
        height,
        labels,
        cfg.min_score,
        cfg.max_results,
        cfg.mask_threshold,
        cfg.mask_grid_size,
        contract.top_k,
        contract.mask_size,
    )
