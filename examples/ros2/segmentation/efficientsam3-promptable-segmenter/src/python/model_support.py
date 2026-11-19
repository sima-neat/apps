"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import json

import math

import os

import sys

import time

from collections import deque

from dataclasses import dataclass

from pathlib import Path

import cv2

import numpy as np

import pyneat

import yaml

from clip_tokenizer import ClipTokenizer

DEFAULT_CONFIG = Path(get_package_share_directory('neat_efficientsam3_promptable_segmenter')) / "common" / "config.yaml"

PULL_TIMEOUT_MS = 20000

MODEL_SIZE = 1008

TEXT_TOKENS = 16

TEXT_DIM = 256

VOCAB_SIZE = 49408

PIPELINE_DEPTH = 3

METADATA_BYTE_BUDGET = 32768

DUPLICATE_IOU = 0.6

CODECS = {"h264": "h264", "avc": "h264", "h265": "h265", "hevc": "h265"}

STREAM_CODECS = {"h264": "h264", "avc1": "h264", "hevc": "h265", "hev1": "h265", "hvc1": "h265", "h265": "h265"}

@dataclass(frozen=True)
class Config:
    model_path: str
    text_encoder: str
    prompt: str
    rtsp_url: str
    codec: str
    tcp: bool
    latency_ms: int
    frames: int
    min_score: float
    max_detections: int
    mask_threshold: float
    profile: bool
    profile_interval: int
    insight_host: str
    video_port: int
    metadata_port: int
    save_dir: str
    save_every: int

def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    source, inference, runtime, output = raw["source"], raw["inference"], raw["runtime"], raw["output"]
    insight = output["insight"]
    if not 0.0 <= inference["min_score"] <= 1.0:
        raise ValueError("inference.min_score must be in [0, 1]")
    if inference["max_detections"] <= 0:
        raise ValueError("inference.max_detections must be > 0")
    if not 0.0 <= inference["mask_threshold"] <= 1.0:
        raise ValueError("inference.mask_threshold must be in [0, 1]")
    if runtime["profile_interval"] <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    codec = CODECS.get(str(source.get("codec", "h264")).lower())
    if codec is None:
        raise ValueError("source.codec must be h264 or h265")
    return Config(
        model_path=raw["model"]["path"],
        text_encoder=raw["model"]["text_encoder"],
        prompt=raw["prompt"]["text"],
        rtsp_url=source["rtsp_url"],
        codec=codec,
        tcp=source["tcp"],
        latency_ms=source["latency_ms"],
        frames=inference["frames"],
        min_score=inference["min_score"],
        max_detections=inference["max_detections"],
        mask_threshold=inference["mask_threshold"],
        profile=runtime["profile"],
        profile_interval=runtime["profile_interval"],
        insight_host=insight["host"],
        video_port=insight["video_port"],
        metadata_port=insight["metadata_port"],
        save_dir=output["save_dir"],
        save_every=output["save_every"],
    )

def prompt_tokens(prompt) -> np.ndarray:
    """Tokenizes the prompt, rejecting one that the text encoder would not see whole."""
    tokenizer = ClipTokenizer()
    count = len(tokenizer.tokens(prompt)) if isinstance(prompt, str) else 0
    if count == 0:
        raise ValueError("prompt.text must name what to segment")
    if count > TEXT_TOKENS - 2:
        raise ValueError(f"prompt.text is {count} tokens; the text encoder reads at most {TEXT_TOKENS - 2}, "
                         "so shorten it")
    return tokenizer.encode(prompt, TEXT_TOKENS)

def encode_prompt(cfg: Config, tokens):
    # The MLA cannot look up token ids, so the text encoder takes them one-hot.
    one_hot = np.zeros((1, TEXT_TOKENS, VOCAB_SIZE), np.float32)
    one_hot[0, np.arange(TEXT_TOKENS), tokens] = 1.0
    encoder = pyneat.Model(cfg.text_encoder).build([ev74_tensor(one_hot)])
    encoder.push([ev74_tensor(one_hot)])
    features = encoder.pull().tensors[0].to_numpy(copy=True)
    encoder.close()
    text = np.zeros((1, TEXT_TOKENS + 1, TEXT_DIM), np.float32)
    text[0, :TEXT_TOKENS] = features.reshape(TEXT_TOKENS, TEXT_DIM)
    text[0, TEXT_TOKENS, :TEXT_TOKENS] = tokens != 0
    return text

def ev74_tensor(array):
    return pyneat.Tensor.from_numpy(array, copy=False, memory=pyneat.TensorMemory.EV74,
                                    layout=pyneat.TensorLayout.HWC)

def preprocess(frame, out) -> None:
    height, width = frame.shape[0] * 2 // 3, frame.shape[1]
    shrink = width > MODEL_SIZE or height > MODEL_SIZE
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    y = cv2.resize(frame[:height], (MODEL_SIZE, MODEL_SIZE), interpolation=interpolation)
    uv = frame[height:].reshape(height // 2, width // 2, 2)
    uv = cv2.resize(uv, (MODEL_SIZE // 2, MODEL_SIZE // 2), interpolation=interpolation)
    rgb = cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2RGB_NV12)
    np.multiply(rgb, np.float32(1.0 / 127.5), out=out, casting="unsafe")
    out -= np.float32(1.0)

def mask_outline(mask_logits, box, width: int, height: int, logit_threshold: float) -> list:
    mask_h, mask_w = mask_logits.shape
    cell_w, cell_h = width / mask_w, height / mask_h
    x0, y0, x1, y1 = box
    mx0, my0 = max(0, math.floor(x0 / cell_w) - 1), max(0, math.floor(y0 / cell_h) - 1)
    mx1, my1 = min(mask_w, math.ceil(x1 / cell_w) + 1), min(mask_h, math.ceil(y1 / cell_h) + 1)
    out_x0, out_y0 = round(mx0 * cell_w), round(my0 * cell_h)
    out_size = (round(mx1 * cell_w) - out_x0, round(my1 * cell_h) - out_y0)
    # Upscale the mask cells under the box before thresholding, as SAM 3 does, so the outline follows the
    # interpolated mask.
    cells = np.ascontiguousarray(mask_logits[my0:my1, mx0:mx1])
    inside = cv2.resize(cells, out_size, interpolation=cv2.INTER_LINEAR) > logit_threshold
    contours, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    return [[int(x) + out_x0, int(y) + out_y0] for [[x, y]] in polygon]

def box_iou(a, b) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union

def segments_of(detections, masks, cfg: Config, width: int, height: int) -> list[dict]:
    detections = detections.reshape(-1, detections.shape[-1])
    masks = masks.reshape(masks.shape[-3:])
    logit_threshold = np.log(np.float64(cfg.mask_threshold) / (1.0 - cfg.mask_threshold))
    best = np.argsort(-detections[:, 4], kind="stable")
    sx, sy = width / MODEL_SIZE, height / MODEL_SIZE
    segments, kept_boxes = [], []
    for n, query in enumerate((q for q in best if detections[q, 4] > cfg.min_score), start=1):
        # Duplicates and empty masks are skipped, so the limit counts the instances sent.
        if len(segments) == cfg.max_detections:
            break
        x1, y1, x2, y2, score = (float(v) for v in detections[query, :5])
        x0 = max(0, min(width - 1, math.floor(x1 * sx)))
        y0 = max(0, min(height - 1, math.floor(y1 * sy)))
        box = (x0, y0, max(x0 + 1, min(width, math.ceil(x2 * sx))), max(y0 + 1, min(height, math.ceil(y2 * sy))))
        if any(box_iou(box, kept) > DUPLICATE_IOU for kept in kept_boxes):
            continue
        outline = mask_outline(masks[:, :, query], box, width, height, logit_threshold)
        if outline:
            kept_boxes.append(box)
            segments.append({"id": f"seg_{n}", "label": cfg.prompt, "confidence": round(score, 4),
                             "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                             "mask_format": "polygon", "mask": outline})
    return segments

def segments_json(segments: list[dict]) -> str:
    """Serializes the best-scoring segments that fit the metadata byte budget."""
    kept, size = [], len('{"segments":[]}')
    for segment in segments:
        size += len(json.dumps(segment, separators=(",", ":"))) + 1
        if size > METADATA_BYTE_BUDGET:
            break
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":"))
