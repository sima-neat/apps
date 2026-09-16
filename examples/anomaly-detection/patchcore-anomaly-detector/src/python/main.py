"""PatchCore anomaly detection on the SiMa MLA: a compiled `wide_resnet50_2`
patch-feature extractor plus host-side coreset memory-bank scoring (see
patchcore_scoring.py). Runs on the Modalix DevKit; from the Palette SDK host
use `dk main.py ...` instead of a plain local `python3 main.py ...`.

    python3 main.py --calibrate --config common/config.yaml  # build the bank
    python3 main.py --config common/config.yaml               # score input
"""
from __future__ import annotations

import os

# Must run before numpy (imported transitively below) picks a BLAS backend.
# OMP/MKL/NUMEXPR stay pinned to 1: unthrottled, their thread pool overhead
# dominates this app's tiny per-frame matmuls. OPENBLAS_NUM_THREADS=6 is the
# measured sweet spot on this (16-core) hardware for a real OpenBLAS build;
# it's a no-op if numpy only has reference BLAS (no such library to read it).
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
    MemoryBank,
    build_bank_meta,
    extract_hwc,
    load_bank_meta,
    percentile_threshold,
    save_bank_meta,
    upsample_and_smooth,
    verify_bank_hash,
    verify_bank_matches_model,
)
from patchcore_scoring import cv2, np  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

BACKBONE = "wide_resnet50_2"
TORCHVISION_WEIGHTS = "IMAGENET1K_V1"
PATCH_GRID = (28, 28)  # 224x224 input -> 28x28 patch grid, see the model's compile config


def time_ms() -> float:
    return time.perf_counter() * 1000.0


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


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
    coreset_ratio: float = 0.01
    seed: int = 0
    threshold_percentile: float = 99.0
    threshold_images_dir: str = ""  # empty = reuse nominal_images_dir
    threshold_margin: float = 1.0  # multiplier applied on top of the percentile threshold


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
    timeout_ms: int = 5000
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
            coreset_ratio=float(calibration.get("coreset_ratio", 0.01)),
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
        ),
        timeout_ms=int(runtime.get("timeout_ms", 5000)),
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
    if cfg.timeout_ms <= 0:
        raise ValueError("runtime.timeout_ms must be > 0")
    if cfg.frames < 0:
        raise ValueError("runtime.frames must be >= 0")
    if cfg.source_type in {"video_file", "rtsp"}:
        if not cfg.output.insight_host:
            raise ValueError("output.insight.host must be set for source.type=video_file/rtsp")
        if not 0 < cfg.output.insight_video_port <= 65535:
            raise ValueError("output.insight.video_port must be in [1, 65535]")
        if cfg.output.insight_channel < 0:
            raise ValueError("output.insight.channel must be >= 0")
        if cfg.output.insight_bitrate_kbps <= 0:
            raise ValueError("output.insight.bitrate_kbps must be > 0")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and is_image(p))


def make_image_model(model_path: str) -> "pyneat.Model":
    """One shared Options for image_dir/video_file. Each source decodes to a
    host-side BGR frame before calling the model, so no Graph-embedded decode
    source or hand-specified resize geometry is needed here."""
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
    return pyneat.Model(model_path, opt)


def rgb_tensor(bgr) -> "pyneat.Tensor":
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)


def extract_from_bgr(model: "pyneat.Model", bgr, timeout_ms: int = 5000) -> "np.ndarray":
    """Runs one host-decoded BGR image through the model and returns its (H, W, C)
    patch embedding. Resize is whatever the compiled model's preprocess negotiates;
    this only handles the BGR->RGB color swap `Model.run` needs."""
    outputs = model.run([rgb_tensor(bgr)], timeout_ms=timeout_ms)
    if not outputs:
        raise RuntimeError("model returned no output tensors")
    return extract_hwc(np.asarray(outputs[0].to_numpy(copy=True)))


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------


def draw_overlay(bgr, score_map, sigma: float, alpha: float, patch_scale_min: float,
                 patch_scale_max: float):
    """Heatmap overlay (blue = typical, red = anomalous) on a fixed scale
    calibrated from nominal patch scores (patch_scale_min/max from
    bank_meta.json), not this image's own min/max -- per-image scaling would
    make ordinary texture variation look as "hot" as a real defect."""
    h, w = bgr.shape[:2]
    heat = upsample_and_smooth(score_map, (w, h), sigma)
    scale = max(patch_scale_max - patch_scale_min, 1e-6)
    heat_norm = np.clip((heat - patch_scale_min) / scale, 0.0, 1.0)
    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat_color, alpha, 0)


# --------------------------------------------------------------------------
# Calibrate
# --------------------------------------------------------------------------


def cmd_calibrate(cfg: AppConfig) -> int:
    nominal_dir = Path(cfg.calibration.nominal_images_dir)
    paths = find_images(nominal_dir)
    if not paths:
        print(f"[FATAL] no images found in {nominal_dir}", file=sys.stderr)
        return 3

    model = make_image_model(cfg.model_path)
    print(f"Extracting patch embeddings from {len(paths)} nominal images ...")
    per_image = []
    for i, path in enumerate(paths, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] could not read image: {path}", file=sys.stderr)
            continue
        per_image.append(extract_from_bgr(model, bgr, cfg.timeout_ms))
        if i % 10 == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {path.name}")

    bank = MemoryBank.build(per_image, cfg.calibration.coreset_ratio, cfg.calibration.seed)
    print(
        f"Coreset: {bank.size} / {sum(e.shape[0] * e.shape[1] for e in per_image)} patches "
        f"(ratio={cfg.calibration.coreset_ratio})"
    )

    threshold_dir = Path(cfg.calibration.threshold_images_dir or cfg.calibration.nominal_images_dir)
    threshold_paths = find_images(threshold_dir)
    scores = []
    patch_scores: list[float] = []
    for path in threshold_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        embedding = extract_from_bgr(model, bgr, cfg.timeout_ms)
        scored = bank.score(embedding, cfg.scoring.num_neighbors)
        scores.append(scored.image_score)
        patch_scores.extend(scored.score_map.ravel().tolist())
    threshold = percentile_threshold(scores, cfg.calibration.threshold_percentile) * cfg.calibration.threshold_margin
    print(
        f"Threshold: {threshold:.4f} (p{cfg.calibration.threshold_percentile} over "
        f"{len(scores)} nominal images from {threshold_dir}, x{cfg.calibration.threshold_margin} margin)"
    )

    # Separate, patch-scoped threshold for the heatmap overlay (see
    # draw_overlay) -- not the image-level threshold above, which is
    # reweighted by neighborhood diversity and answers a different question.
    # Same threshold_margin applied here so "hot" in the overlay agrees with
    # the margin-padded image-level verdict.
    patch_threshold = (
        percentile_threshold(patch_scores, cfg.calibration.threshold_percentile) * cfg.calibration.threshold_margin
    )
    patch_scale_min = percentile_threshold(patch_scores, 50.0)
    patch_scale_max = patch_threshold
    print(
        f"Patch threshold: {patch_threshold:.4f} (p{cfg.calibration.threshold_percentile} over "
        f"{len(patch_scores)} nominal patches, x{cfg.calibration.threshold_margin} margin; "
        f"overlay scale {patch_scale_min:.4f}-{patch_scale_max:.4f})"
    )

    bank_path = Path(cfg.memory_bank_path)
    meta_path = Path(cfg.bank_meta_path)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    bank.save(bank_path)
    meta = build_bank_meta(
        model_path=cfg.model_path,
        bank_path=bank_path,
        backbone=BACKBONE,
        torchvision_weights=TORCHVISION_WEIGHTS,
        embed_dim=EMBED_DIM,
        patch_grid=PATCH_GRID,
        coreset_ratio=cfg.calibration.coreset_ratio,
        seed=cfg.calibration.seed,
        num_nominal_images=len(per_image),
        bank_size=bank.size,
        num_neighbors=cfg.scoring.num_neighbors,
        gaussian_sigma=cfg.scoring.gaussian_sigma,
        threshold=threshold,
        threshold_percentile=cfg.calibration.threshold_percentile,
        threshold_num_images=len(scores),
        patch_threshold=patch_threshold,
        patch_scale_min=patch_scale_min,
        patch_scale_max=patch_scale_max,
    )
    save_bank_meta(meta_path, meta)
    print(f"Saved {bank_path} ({bank.size} x {bank.embed_dim}) and {meta_path}")
    return 0


# --------------------------------------------------------------------------
# Score: image_dir -- writes annotated overlays to output.dir, no live view
# (matches every folder-based example in this repo: depth-estimator,
# classification/image-classifier, etc. -- none of them stream to Insight).
# --------------------------------------------------------------------------


def cmd_score_image_dir(cfg: AppConfig, bank: MemoryBank, threshold: float, num_neighbors: int,
                        patch_scale_min: float, patch_scale_max: float) -> int:
    paths = find_images(Path(cfg.image_dir))
    if not paths:
        print(f"[FATAL] no images found in {cfg.image_dir}", file=sys.stderr)
        return 3
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resolved (symlink-following) comparison: writing overlays into the input
    # directory would silently replace the original images with their overlays.
    if output_dir.resolve() == Path(cfg.image_dir).resolve():
        print(
            f"[FATAL] output.dir ({cfg.output.dir}) must not be the same as "
            f"source.image_dir ({cfg.image_dir}) -- this would overwrite the input images",
            file=sys.stderr,
        )
        return 2

    model = make_image_model(cfg.model_path)
    processed = 0
    write_failures = 0
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] could not read image: {path}", file=sys.stderr)
            continue

        mla_start = time_ms()
        embedding = extract_from_bgr(model, bgr, cfg.timeout_ms)
        mla_ms = time_ms() - mla_start

        host_start = time_ms()
        scored = bank.score(embedding, num_neighbors)
        overlay = draw_overlay(bgr, scored.score_map, cfg.scoring.gaussian_sigma, cfg.output.overlay_alpha,
                               patch_scale_min, patch_scale_max)
        host_ms = time_ms() - host_start

        verdict = "ANOMALOUS" if scored.image_score > threshold else "normal"
        print(
            f"{path}: score={scored.image_score:.4f} threshold={threshold:.4f} verdict={verdict} "
            f"(mla={mla_ms:.1f}ms host={host_ms:.1f}ms)"
        )
        out_path = output_dir / path.name
        if not cv2.imwrite(str(out_path), overlay):
            print(f"[ERROR] failed to write overlay: {out_path}", file=sys.stderr)
            write_failures += 1
            continue
        processed += 1

    print(f"Done: {processed} images processed -- overlays written to {output_dir}")
    # A partial run (some overlays written, some failed) is still a failure --
    # the caller asked for every input scored, not "at least one."
    if write_failures > 0:
        print(f"[FATAL] {write_failures} overlay(s) failed to write", file=sys.stderr)
        return 3
    return 0 if processed > 0 else 3


# --------------------------------------------------------------------------
# Score: video_file -- cv2.VideoCapture, streaming the annotated overlay live
# to Insight via a small host-pushed graph; `output.save_every > 0`
# additionally writes periodic local snapshots. Also used by rtsp.
# --------------------------------------------------------------------------


def build_video_sender(cfg: AppConfig, fps: float, width: int, height: int):
    """A small, separate Insight push-graph: the host manually pushes each
    annotated frame into it."""
    output_fps = max(1, round(fps))
    input_options = pyneat.InputOptions()
    input_options.payload_type = pyneat.PayloadType.Image
    input_options.format = pyneat.Format.RGB
    input_options.width = width
    input_options.height = height
    input_options.depth = 3
    input_options.fps_n = output_fps
    input_options.fps_d = 1
    input_options.memory_policy = pyneat.InputMemoryPolicy.Ev74

    sender_options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, output_fps)
    sender_options.host = cfg.output.insight_host
    sender_options.channel = cfg.output.insight_channel
    sender_options.video_port_base = cfg.output.insight_video_port
    sender_options.encoder.bitrate_kbps = cfg.output.insight_bitrate_kbps

    graph = pyneat.Graph("insight")
    graph.add(pyneat.nodes.input(input_options))
    graph.add(pyneat.groups.video_sender(sender_options))
    seed = pyneat.Tensor.from_numpy(
        np.zeros((height, width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
    )
    return graph, graph.build([seed]), sender_options.video_port


def stream_frame(run, frame_bgr) -> None:
    if not run.push([rgb_tensor(frame_bgr)]):
        raise RuntimeError("Insight video push failed")


def cmd_score_video_file(cfg: AppConfig, bank: MemoryBank, threshold: float, num_neighbors: int,
                         patch_scale_min: float, patch_scale_max: float) -> int:
    video = cv2.VideoCapture(cfg.video_path)
    ok, frame = video.read()
    if not video.isOpened() or not ok:
        print(f"[FATAL] failed to open video source: {cfg.video_path}", file=sys.stderr)
        return 2
    height, width = frame.shape[:2]

    fps = video.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0

    model = make_image_model(cfg.model_path)
    runner = model.build(
        [rgb_tensor(frame)], route_options=pyneat.ModelRouteOptions(), run_options=pyneat.RunOptions()
    )
    video_graph, video_run, video_port = build_video_sender(cfg, fps, width, height)
    print(f"streaming to Insight: {cfg.output.insight_host}:{video_port}")

    save_dir = Path(cfg.output.dir)
    if cfg.output.save_every > 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    write_failures = 0
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            mla_start = time_ms()
            outputs = runner.run([rgb_tensor(frame)], timeout_ms=cfg.timeout_ms)
            if not outputs:
                raise RuntimeError("model returned no output tensors")
            embedding = extract_hwc(np.asarray(outputs[0].to_numpy(copy=True)))
            mla_ms = time_ms() - mla_start

            host_start = time_ms()
            scored = bank.score(embedding, num_neighbors)
            overlay = draw_overlay(frame, scored.score_map, cfg.scoring.gaussian_sigma, cfg.output.overlay_alpha,
                               patch_scale_min, patch_scale_max)
            host_ms = time_ms() - host_start

            processed += 1
            verdict = "ANOMALOUS" if scored.image_score > threshold else "normal"
            print(
                f"frame={processed}: score={scored.image_score:.4f} threshold={threshold:.4f} "
                f"verdict={verdict} (mla={mla_ms:.1f}ms host={host_ms:.1f}ms)"
            )

            stream_frame(video_run, overlay)
            if cfg.output.save_every > 0 and processed % cfg.output.save_every == 0:
                snapshot_path = save_dir / f"frame_{processed}.jpg"
                if not cv2.imwrite(str(snapshot_path), overlay):
                    print(f"[ERROR] failed to write snapshot: {snapshot_path}", file=sys.stderr)
                    write_failures += 1

            if cfg.frames > 0 and processed >= cfg.frames:
                break
            ok, frame = video.read()
            if not ok:
                break
    finally:
        runner.close()
        video_run.close()
        video.release()

    print(
        f"Done: {processed} frames processed  video_sender={cfg.output.insight_host}:{video_port}"
    )
    if write_failures > 0:
        print(f"[FATAL] {write_failures} snapshot(s) failed to write", file=sys.stderr)
        return 3
    return 0 if processed > 0 else 3


# --------------------------------------------------------------------------
# Score: rtsp -- see build_rtsp_graph's docstring for why the model isn't
# embedded in the live graph.
# --------------------------------------------------------------------------


def probe_ffprobe(cfg: AppConfig) -> tuple[int, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-rw_timeout", "5000000",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
        "-of", "default=nw=1",
    ]
    if cfg.rtsp.tcp:
        cmd.extend(["-rtsp_transport", "tcp"])
    cmd.append(cfg.rtsp.url)
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

    def fps_from_rate(value: str) -> int:
        if not value or value in {"0/0", "0/1"}:
            return 0
        try:
            fps = float(Fraction(value)) if "/" in value else float(value)
        except (ValueError, ZeroDivisionError):
            return 0
        return int(round(fps)) if fps > 0 else 0

    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )

    def to_int(value):
        try:
            return int(value or 0)
        except ValueError:
            return 0

    return to_int(values.get("width")), to_int(values.get("height")), fps


def probe_rtsp_capture(url: str) -> tuple[int, int, int]:
    """Best-effort OpenCV probe. Returns zeros (never raises) on failure --
    resolve_rtsp_geometry applies the configured source.rtsp.width/height
    hints and reports one clear failure to the caller instead."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return 0, 0, 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    return width, height, fps


def resolve_rtsp_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    width, height, fps = probe_ffprobe(cfg)
    # Configured hints take priority over probed values -- required for
    # h265/mjpeg, whose caps aren't self-describing the way H.264 SPS is, so
    # probing frequently can't discover dimensions at all (matches main.cpp's
    # probe_rtsp_geometry).
    width = cfg.rtsp.width if cfg.rtsp.width > 0 else width
    height = cfg.rtsp.height if cfg.rtsp.height > 0 else height
    if width <= 0 or height <= 0 or fps <= 0:
        probed_w, probed_h, probed_fps = probe_rtsp_capture(cfg.rtsp.url)
        width = width if width > 0 else probed_w
        height = height if height > 0 else probed_h
        fps = fps if fps > 0 else probed_fps
    if cfg.rtsp.codec == "mjpeg" and fps <= 0:
        raise RuntimeError(
            "MJPEG source did not provide a valid frame rate; set source.rtsp.width/height "
            "or use a source with probeable FPS metadata"
        )
    return width, height, fps


def make_rtsp_source_fragment(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.rtsp.url
    opt.tcp = cfg.rtsp.tcp
    opt.latency_ms = cfg.rtsp.latency_ms
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.codec = {
        "h264": pyneat.RtspCodec.H264,
        "h265": pyneat.RtspCodec.H265,
        "mjpeg": pyneat.RtspCodec.MJPEG,
    }[cfg.rtsp.codec]
    opt.source_fps = fps
    if cfg.rtsp.codec == "h264":
        opt.auto_caps_from_stream = True
        opt.fallback_h264_width = cfg.rtsp.width or width
        opt.fallback_h264_height = cfg.rtsp.height or height
    else:
        # h265/mjpeg caps aren't self-describing the way h264 SPS is, so Neat
        # needs an explicit hint; h264 negotiates it from the stream itself.
        opt.dec_width = cfg.rtsp.width or width
        opt.dec_height = cfg.rtsp.height or height
    return pyneat.groups.rtsp_decoded_input(opt)


def extract_tensors(sample) -> list:
    if sample is None or not hasattr(sample, "kind"):
        return []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.kind == pyneat.SampleKind.TensorSet:
        return list(sample.tensors)
    tensors = []
    for f in getattr(sample, "fields", []):
        tensors.extend(extract_tensors(f))
    return tensors


def find_field(sample, label: str):
    if sample is None:
        return None
    if getattr(sample, "stream_label", "") == label:
        return sample
    for f in getattr(sample, "fields", []):
        found = find_field(f, label)
        if found is not None:
            return found
    return None


def tensor_dim(tensor, name: str) -> int:
    """`Tensor.width`/`.height` are plain attributes on some pyneat builds and
    bound methods on others; call through only when callable."""
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)


def frame_bgr_from_sample(sample):
    # Falls back to the sample itself for a plain (non-joined) single-output pull.
    field = find_field(sample, "frame") or sample
    tensors = extract_tensors(field)
    if not tensors:
        raise RuntimeError("joined output missing a frame field")
    tensor = tensors[0]
    if tensor.is_nv12():
        width, height = tensor_dim(tensor, "width"), tensor_dim(tensor, "height")
        # .contiguous() repacks a row-padded plane before the raw byte copy --
        # without it, a stride wider than width (common for hardware-aligned
        # buffers) reads padding as pixel data and shifts every row after the
        # first. Matches C++'s nv12_to_bgr and every multi-stream Python example.
        payload = np.frombuffer(tensor.contiguous().copy_payload_bytes(), dtype=np.uint8)
        expected = width * height * 3 // 2
        if payload.size < expected:
            raise RuntimeError(f"NV12 payload too small: {payload.size} < {expected}")
        nv12 = payload[:expected].reshape((height * 3 // 2, width))
        return np.ascontiguousarray(cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12))
    return np.asarray(tensor.to_numpy(copy=True))



def build_rtsp_graph(cfg: AppConfig, width: int, height: int, fps: int):
    """Decode-only graph: the model is deliberately NOT embedded here; this
    just wires the RTSP source to a "frame" output, and cmd_score_rtsp scores
    each pulled frame host-side instead. An embedded-model graph hits
    [resource.output_pool_exhausted] once real per-frame work competes with
    its hardcoded 4-buffer output pool -- see the PR description for the
    full reproduction."""
    source = make_rtsp_source_fragment(cfg, fps, width, height)

    graph = pyneat.Graph("patchcore")
    graph.connect(source, pyneat.nodes.output("frame"))

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph.build(run_options)


def cmd_score_rtsp(cfg: AppConfig, bank: MemoryBank, threshold: float, num_neighbors: int,
                   patch_scale_min: float, patch_scale_max: float) -> int:
    width, height, fps = resolve_rtsp_geometry(cfg)
    if width <= 0 or height <= 0 or fps <= 0:
        print(f"[FATAL] failed to resolve source geometry for {cfg.rtsp.url}", file=sys.stderr)
        return 2

    model = make_image_model(cfg.model_path)
    run = build_rtsp_graph(cfg, width, height, fps)
    _video_graph, video_run, video_port = build_video_sender(cfg, fps, width, height)
    print(f"streaming to Insight: {cfg.output.insight_host}:{video_port}")

    save_frames = cfg.output.save_every > 0
    save_dir = Path(cfg.output.dir)
    if save_frames:
        save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    write_failures = 0
    pull_timed_out = False
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            # Bounded pull, not pull("frame", -1) -- that deadlocks the
            # pyneat Python binding against the decoder thread on this SDK
            # version. Don't retry on a timeout either: once one pull() call
            # times out, every later one does too, so retrying just spins
            # instead of hanging once -- report and end the run instead.
            sample = run.pull("frame", 5000)
            if sample is None:
                print("[FATAL] timed out waiting for an RTSP frame", file=sys.stderr)
                pull_timed_out = True
                break
            bgr = frame_bgr_from_sample(sample)
            mla_start = time_ms()
            embedding = extract_from_bgr(model, bgr, cfg.timeout_ms)
            mla_ms = time_ms() - mla_start

            host_start = time_ms()
            scored = bank.score(embedding, num_neighbors)
            overlay = draw_overlay(bgr, scored.score_map, cfg.scoring.gaussian_sigma, cfg.output.overlay_alpha,
                               patch_scale_min, patch_scale_max)
            verdict = "ANOMALOUS" if scored.image_score > threshold else "normal"
            processed += 1
            print(
                f"frame={processed}: score={scored.image_score:.4f} threshold={threshold:.4f} "
                f"verdict={verdict} (mla={mla_ms:.1f}ms host={time_ms() - host_start:.1f}ms)"
            )

            stream_frame(video_run, overlay)
            if save_frames and processed % cfg.output.save_every == 0:
                snapshot_path = save_dir / f"frame_{processed}.jpg"
                if not cv2.imwrite(str(snapshot_path), overlay):
                    print(f"[ERROR] failed to write snapshot: {snapshot_path}", file=sys.stderr)
                    write_failures += 1
    finally:
        run.close()
        video_run.close()

    print(
        f"Done: {processed} frames processed  video_sender={cfg.output.insight_host}:{video_port}"
    )
    if write_failures > 0:
        print(f"[FATAL] {write_failures} snapshot(s) failed to write", file=sys.stderr)
        return 3
    if pull_timed_out:
        return 3
    return 0 if processed > 0 else 3


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PatchCore anomaly detection on the SiMa MLA")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Build memory_bank.npy and bank_meta.json from calibration.nominal_images_dir, then exit.",
    )
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config}")
            return 0

        if args.calibrate:
            return cmd_calibrate(cfg)

        bank_path = Path(cfg.memory_bank_path)
        meta_path = Path(cfg.bank_meta_path)
        if not bank_path.exists() or not meta_path.exists():
            print(
                f"[FATAL] memory bank not found: {bank_path} / {meta_path}\n"
                "        Build it first: python3 main.py --calibrate --config <path>",
                file=sys.stderr,
            )
            return 2
        meta = load_bank_meta(meta_path)
        print("Verifying model package...")
        verify_bank_matches_model(meta, cfg.model_path)
        verify_bank_hash(meta, bank_path)
        bank = MemoryBank.load(bank_path)
        threshold = float(meta["threshold"]["value"])
        if "patch_threshold" not in meta or "scale_min" not in meta["patch_threshold"]:
            print(
                "[FATAL] bank_meta.json is missing patch_threshold.scale_min (built before "
                "this field existed); recalibrate with --calibrate to regenerate it",
                file=sys.stderr,
            )
            return 2
        patch_scale_min = float(meta["patch_threshold"]["scale_min"])
        patch_scale_max = float(meta["patch_threshold"]["scale_max"])
        # Score with the num_neighbors the bank was calibrated with, not the
        # live config -- it changes the score distribution the threshold was
        # derived from.
        num_neighbors = int(meta["num_neighbors"])
        if num_neighbors != cfg.scoring.num_neighbors:
            print(
                f"[WARN] scoring.num_neighbors={cfg.scoring.num_neighbors} in config differs from "
                f"the bank's calibrated value ({num_neighbors}); using {num_neighbors} to stay "
                "consistent with the saved threshold. Recalibrate to adopt the new value.",
                file=sys.stderr,
            )

        if cfg.source_type == "image_dir":
            return cmd_score_image_dir(cfg, bank, threshold, num_neighbors, patch_scale_min, patch_scale_max)
        if cfg.source_type == "video_file":
            return cmd_score_video_file(cfg, bank, threshold, num_neighbors, patch_scale_min, patch_scale_max)
        return cmd_score_rtsp(cfg, bank, threshold, num_neighbors, patch_scale_min, patch_scale_max)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
