"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")

for _blas_env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import argparse

import json

import subprocess

import sys

import time

from dataclasses import dataclass, field

from fractions import Fraction

from pathlib import Path

import pyneat

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patchcore_scoring import (  # noqa: E402
    EMBED_DIM,
    MODEL_INPUT_SIZE,
    draw_overlay,
    draw_nv12_overlay,
    letterbox_rect,
    MemoryBank,
    build_bank_meta,
    extract_hwc,
    load_bank_meta,
    percentile_threshold,
    save_bank_meta,
    verify_bank_hash,
    verify_bank_matches_model,
)

from patchcore_scoring import cv2, np  # noqa: E402

DEFAULT_CONFIG = Path(get_package_share_directory('neat_patchcore_anomaly_detector')) / "common" / "config.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

BACKBONE = "wide_resnet50_2"

TORCHVISION_WEIGHTS = "IMAGENET1K_V1"

PATCH_GRID = (28, 28)  # 224x224 input -> 28x28 patch grid, see the model's compile config

JOINED_BUFFERS = 4  # RTSP joined-output queue; bounds lag to ~4 frames when scoring is slow

LETTERBOX_PAD = 114  # gray fill of Neat's letterbox resize, matched by the host letterbox

THRESHOLD_EVERY = 5  # --calibrate sets aside every 5th nominal image to measure the threshold

@dataclass(frozen=True)
class RtspConfig:
    url: str = ""
    codec: str = "h264"  # h264, h265, or mjpeg.
    tcp: bool = True
    latency_ms: int = 200
    width: int = 0  # optional hint; required for h265/mjpeg, unused for h264
    height: int = 0

@dataclass(frozen=True)
class CalibrationConfig:
    nominal_images_dir: str = "assets/datasets/patchcore/nominal"
    coreset_ratio: float = 0.0004
    seed: int = 0
    threshold_percentile: float = 99.0
    threshold_images_dir: str = ""  # empty = reuse nominal_images_dir
    threshold_margin: float = 1.0  # multiplier on both percentile thresholds (image and patch)

@dataclass(frozen=True)
class ScoringConfig:
    num_neighbors: int = 9
    gaussian_sigma: float = 4.0

@dataclass(frozen=True)
class OutputConfig:
    dir: str = "sandbox/patchcore-anomaly-detector"
    save_every: int = 0  # image_dir: always saves regardless. video_file/rtsp: 0 disables local saving.
    overlay_alpha: float = 0.45
    insight_host: str = ""  # video_file/rtsp only.
    insight_video_port: int = 9000
    insight_channel: int = 0
    insight_bitrate_kbps: int = 1000
    insight_metadata_port: int = 9100

@dataclass(frozen=True)
class AppConfig:
    model_path: str = ""
    source_type: str = "image_dir"  # image_dir | video_file | rtsp
    image_dir: str = "assets/datasets/patchcore/images"
    video_path: str = ""
    rtsp: RtspConfig = field(default_factory=RtspConfig)
    memory_bank_path: str = "sandbox/patchcore-anomaly-detector/memory_bank.npy"
    bank_meta_path: str = "sandbox/patchcore-anomaly-detector/bank_meta.json"
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    frames: int = 0  # video_file/rtsp frame limit; 0 = run until the source ends
    profile: bool = False

def section(raw: dict, key: str) -> dict:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value

def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = section(raw, "model")
    source = section(raw, "source")
    rtsp = section(source, "rtsp")
    memory_bank = section(raw, "memory_bank")
    calibration = section(raw, "calibration")
    scoring = section(raw, "scoring")
    output = section(raw, "output")
    insight = section(output, "insight")
    runtime = section(raw, "runtime")

    cfg = AppConfig(
        model_path=model.get("path", ""),
        source_type=source.get("type", "image_dir"),
        image_dir=source.get("image_dir", ""),
        video_path=source.get("video_path", ""),
        rtsp=RtspConfig(
            url=rtsp.get("url", ""),
            codec=rtsp.get("codec", "h264"),
            tcp=bool(rtsp.get("tcp", True)),
            latency_ms=int(rtsp.get("latency_ms", 200)),
            width=int(rtsp.get("width", 0)),
            height=int(rtsp.get("height", 0)),
        ),
        memory_bank_path=memory_bank.get("path", ""),
        bank_meta_path=memory_bank.get("meta_path", ""),
        calibration=CalibrationConfig(
            nominal_images_dir=calibration.get("nominal_images_dir", ""),
            coreset_ratio=float(calibration.get("coreset_ratio", 0.0004)),
            seed=int(calibration.get("seed", 0)),
            threshold_percentile=float(calibration.get("threshold_percentile", 99.0)),
            threshold_images_dir=calibration.get("threshold_images_dir", ""),
            threshold_margin=float(calibration.get("threshold_margin", 1.0)),
        ),
        scoring=ScoringConfig(
            num_neighbors=int(scoring.get("num_neighbors", 9)),
            gaussian_sigma=float(scoring.get("gaussian_sigma", 4.0)),
        ),
        output=OutputConfig(
            dir=output.get("dir", ""),
            save_every=int(output.get("save_every", 0)),
            overlay_alpha=float(output.get("overlay_alpha", 0.45)),
            insight_host=insight.get("host", ""),
            insight_video_port=int(insight.get("video_port", 9000)),
            insight_channel=int(insight.get("channel", 0)),
            insight_bitrate_kbps=int(insight.get("bitrate_kbps", 1000)),
            insight_metadata_port=int(insight.get("metadata_port", 9100)),
        ),
        frames=int(runtime.get("frames", 0)),
        profile=bool(runtime.get("profile", False)),
    )
    validate_config(cfg)
    return cfg

def validate_config(cfg: AppConfig) -> None:
    if not cfg.model_path:
        raise ValueError("model.path must be set")
    if cfg.source_type not in {"image_dir", "video_file", "rtsp"}:
        raise ValueError("source.type must be image_dir, video_file, or rtsp")
    if cfg.source_type == "image_dir" and not cfg.image_dir:
        raise ValueError("source.image_dir must be set when source.type=image_dir")
    if cfg.source_type == "video_file" and not cfg.video_path:
        raise ValueError("source.video_path must be set when source.type=video_file")
    if cfg.source_type == "rtsp" and not cfg.rtsp.url:
        raise ValueError("source.rtsp.url must be set when source.type=rtsp")
    if not cfg.memory_bank_path:
        raise ValueError("memory_bank.path must be set")
    if not cfg.bank_meta_path:
        raise ValueError("memory_bank.meta_path must be set")
    if not cfg.calibration.nominal_images_dir:
        raise ValueError("calibration.nominal_images_dir must be set")
    if not cfg.output.dir:
        raise ValueError("output.dir must be set")
    # Overlays written among the known-good references would be calibrated on next time.
    output_dir = Path(cfg.output.dir).resolve()
    calibration_dirs = [cfg.calibration.nominal_images_dir, cfg.calibration.threshold_images_dir]
    if any(d and output_dir == Path(d).resolve() for d in calibration_dirs):
        raise ValueError("output.dir must not be calibration.nominal_images_dir or "
                         "calibration.threshold_images_dir")
    if cfg.source_type == "rtsp" and cfg.rtsp.codec not in {"h264", "h265", "mjpeg"}:
        raise ValueError("source.rtsp.codec must be h264, h265, or mjpeg")
    if not 0.0 < cfg.calibration.coreset_ratio <= 1.0:
        raise ValueError("calibration.coreset_ratio must be in (0, 1]")
    if not 0.0 <= cfg.calibration.threshold_percentile <= 100.0:
        raise ValueError("calibration.threshold_percentile must be between 0 and 100")
    if cfg.calibration.threshold_margin < 1.0:
        raise ValueError("calibration.threshold_margin must be >= 1.0")
    if cfg.scoring.num_neighbors < 1:
        raise ValueError("scoring.num_neighbors must be >= 1")
    if cfg.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    # --calibrate writes both bank files; neither may overwrite the model or each other.
    model = Path(cfg.model_path).resolve()
    bank = Path(cfg.memory_bank_path).resolve()
    meta = Path(cfg.bank_meta_path).resolve()
    if model in (bank, meta):
        raise ValueError("memory_bank.path and memory_bank.meta_path must not point at model.path")
    if bank == meta:
        raise ValueError("memory_bank.path and memory_bank.meta_path must be different files")
    if cfg.source_type in {"video_file", "rtsp"}:
        if not cfg.output.insight_host:
            raise ValueError("output.insight.host must be set for source.type=video_file/rtsp")
        if not 0 < cfg.output.insight_video_port <= 65535:
            raise ValueError("output.insight.video_port must be in [1, 65535]")
        if cfg.output.insight_channel < 0:
            raise ValueError("output.insight.channel must be >= 0")
        if cfg.output.insight_bitrate_kbps <= 0:
            raise ValueError("output.insight.bitrate_kbps must be > 0")
        if not 0 < cfg.output.insight_metadata_port <= 65535:
            raise ValueError("output.insight.metadata_port must be in [1, 65535]")

def make_image_model(model_path: str) -> "pyneat.Model":
    """Model for calibration, image_dir, and video_file: each input is decoded
    to a host-side BGR frame before calling the model, so no graph-embedded
    decode source or hand-specified resize geometry is needed here. rtsp uses
    make_rtsp_graph_model instead."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(model_path, opt)

def model_input_tensor(bgr) -> "pyneat.Tensor":
    """Letterboxes a host-decoded frame into the model input the same way the RTSP
    graph's preprocess does (centered, gray LETTERBOX_PAD bars), so calibration
    images and every source match for any frame shape. A fixed input size also keeps
    the model from timing out when image sizes change."""
    if (bgr.shape[1], bgr.shape[0]) == MODEL_INPUT_SIZE:
        return rgb_tensor(bgr)
    x, y, w, h = letterbox_rect(bgr.shape[1], bgr.shape[0])
    canvas = np.full((MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0], 3), LETTERBOX_PAD, np.uint8)
    canvas[y:y + h, x:x + w] = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return rgb_tensor(canvas)

def rgb_tensor(bgr) -> "pyneat.Tensor":
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

def make_rtsp_graph_model(cfg: AppConfig, width: int, height: int) -> "pyneat.Model":
    """Model embedded in the RTSP graph: consumes the decoder's NV12 frames directly."""
    opt = pyneat.ModelOptions()
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
    opt.preprocess.input_max_width = width
    opt.preprocess.input_max_height = height
    # Letterbox exactly like model_input_tensor. Resize must be switched on explicitly:
    # on Auto the planner leaves it off for frames that aren't the model input size.
    opt.preprocess.resize.enable = pyneat.AutoFlag.On
    opt.preprocess.resize.mode = pyneat.ResizeMode.Letterbox
    opt.preprocess.resize.width, opt.preprocess.resize.height = MODEL_INPUT_SIZE
    opt.preprocess.resize.pad_value = LETTERBOX_PAD
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(cfg.model_path, opt)
