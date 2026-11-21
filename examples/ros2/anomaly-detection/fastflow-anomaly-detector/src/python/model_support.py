"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import os

import sys

import time

from collections import deque

from dataclasses import dataclass

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_fastflow_anomaly_detector')) / "common" / "config.yaml"

PULL_TIMEOUT_MS = 20000

DEFAULT_MEAN = [0.3893437, 0.35421189, 0.36142577]

DEFAULT_STDDEV = [1.0, 1.0, 1.0]

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class Config:
    model_path: str
    mean: list[float]        # input normalisation the package expects, RGB, on 0..1 pixels
    stddev: list[float]
    rtsp_url: str
    tcp: bool
    latency_ms: int
    frames: int              # 0 runs until stopped
    threshold: float         # map probability that counts as anomalous
    min_region_px: int       # smallest connected region, in map pixels, that counts as a defect
    profile: bool
    profile_interval: int
    insight_host: str
    video_port: int
    heat_max: float          # map probability drawn at full heatmap intensity
    alpha: float             # heatmap opacity over the frame
    save_dir: str
    save_every: int

def rgb_stats(value, key: str) -> list[float]:
    """One normalisation value per channel, in R, G, B order."""
    if isinstance(value, (list, tuple)) and len(value) == 3 and not any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
    ):
        return [float(v) for v in value]
    raise ValueError(f"{key} must be three numbers, one per RGB channel")

def int_or(raw: dict, key: str, default: int) -> int:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value

def float_or(raw: dict, key: str, default: float) -> float:
    value = raw.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric")
    return float(value)

def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be true or false")
    return value

def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    model = raw.get("model") or {}
    source = raw.get("source") or {}
    inference = raw.get("inference") or {}
    runtime = raw.get("runtime") or {}
    output = raw.get("output") or {}
    insight = output.get("insight") or {}
    normalize = model.get("normalize") or {}
    cfg = Config(
        model_path=str(model.get("path") or ""),
        mean=rgb_stats(normalize.get("mean", DEFAULT_MEAN), "model.normalize.mean"),
        stddev=rgb_stats(normalize.get("stddev", DEFAULT_STDDEV), "model.normalize.stddev"),
        rtsp_url=str(source.get("rtsp_url") or ""),
        tcp=bool_or(source, "tcp", True),
        latency_ms=int_or(source, "latency_ms", 100),
        frames=int_or(inference, "frames", 0),
        threshold=float_or(inference, "threshold", 0.5),
        min_region_px=int_or(inference, "min_region_px", 300),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int_or(runtime, "profile_interval", 100),
        insight_host=str(insight.get("host") or ""),
        video_port=int_or(insight, "video_port", 9000),
        heat_max=float_or(output, "heat_max", 0.7),
        alpha=float_or(output, "alpha", 0.55),
        save_dir=str(output.get("save_dir") or ""),
        save_every=int_or(output, "save_every", 0),
    )
    required = (
        ("model.path", cfg.model_path),
        ("source.rtsp_url", cfg.rtsp_url),
        ("output.insight.host", cfg.insight_host),
    )
    for key, value in required:
        if not value:
            raise ValueError(f"{key} must be set")
    if not all(s > 0 for s in cfg.stddev):
        raise ValueError("model.normalize.stddev must be > 0")
    if cfg.latency_ms < 0:
        raise ValueError("source.latency_ms must be >= 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.threshold <= 1.0:
        raise ValueError("inference.threshold must be between 0 and 1")
    if cfg.min_region_px <= 0:
        raise ValueError("inference.min_region_px must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if not 0 < cfg.video_port <= 65535:
        raise ValueError("output.insight.video_port must be in [1, 65535]")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.alpha <= 1.0:
        raise ValueError("output.alpha must be between 0 and 1")
    if cfg.heat_max <= cfg.threshold:
        raise ValueError("output.heat_max must be greater than inference.threshold")
    return cfg

def build_model(cfg: Config, width: int, height: int):
    """A BGR frame in, the anomaly map out. Also returns the side of the square map."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.normalize.mean = cfg.mean
    options.preprocess.normalize.stddev = cfg.stddev
    options.preprocess.normalize.has_explicit_stats = True
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.input_max_depth = 3
    model = pyneat.Model(cfg.model_path, options)
    return model, int(model.output_specs()[0].shape[1])

def regions_from_map(anomaly_map, threshold: float, min_region_px: int):
    """Regions above the threshold that cover at least min_region_px, and their mask."""
    above = (anomaly_map >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, 8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_region_px]
    regions = [{"bbox": tuple(int(v) for v in stats[i, :4]),
                "score": float(anomaly_map[labels == i].max())}
               for i in keep]
    return regions, np.isin(labels, keep).astype(np.uint8)

def render(frame, anomaly_map, regions, mask, cfg: Config, fps: float) -> None:
    """Blend the heatmap over the defect regions, box them and add the verdict banner."""
    height, width = frame.shape[:2]
    if regions:
        # Fixed display span, so the overlay brightness does not flicker between frames.
        span = cfg.heat_max - cfg.threshold
        intensity = np.clip((anomaly_map - cfg.threshold) / span, 0.0, 1.0)
        scaled = cv2.resize((intensity * 255).astype(np.uint8), (width, height))
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1.0 - cfg.alpha, heat, cfg.alpha, 0.0)
        full_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.copyTo(blended, full_mask, frame)
        sx = width / anomaly_map.shape[1]
        sy = height / anomaly_map.shape[0]
        for region in regions:
            x, y, w, h = region["bbox"]
            cv2.rectangle(frame, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)), (0, 0, 255), 2)
    verdict, color = ("ANOMALY", (0, 0, 255)) if regions else ("OK", (0, 200, 0))
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{verdict}  score={float(anomaly_map.max()):.3f}  regions={len(regions)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:5.1f} fps", (width - 130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
