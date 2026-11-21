"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

from dataclasses import dataclass

from fractions import Fraction

import glob

import json

import os

from pathlib import Path

import struct

import subprocess

import sys

import time

import yaml

DEFAULT_CONFIG = Path(get_package_share_directory('neat_multi_stream_multi_model')) / "common" / "config.yaml"

cv2 = None

np = None

pyneat = None

@dataclass(frozen=True)
class AppConfig:
    model_path: str
    labels_path: Path
    source_url: str
    source_type: str = "rtsp"
    source_codec: str = "h264"
    latency_ms: int = 200
    tcp: bool = True
    source_fps: int = 0
    ssl_strict: bool = True
    frames: int = 0
    min_score: float = 0.55
    nms_iou: float = 0.60
    max_detections: int = 50
    profile: bool = False
    profile_interval: int = 100
    insight_host: str = "127.0.0.1"
    video_port: int = 9000
    metadata_port: int = 9100
    save_dir: str = ""
    save_every: int = 0
    # The key that names the source: source.<type>.url, or source.usb.camera.
    source_key: str = "source.rtsp.url"
    # source.type=usb: the camera's stable ID (usb_camera_ids()) and one of its MJPEG modes.
    camera: str = ""
    source_width: int = 1920
    source_height: int = 1080

def load_runtime_dependencies() -> None:
    global cv2, np, pyneat
    if pyneat is not None:
        return

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import cv2 as cv2_module
    import numpy as np_module
    import pyneat as pyneat_module

    cv2 = cv2_module
    np = np_module
    pyneat = pyneat_module

def fps_from_rate(value: str) -> int:
    if not value or value in {"0/0", "0/1"}:
        return 0
    try:
        fps = float(Fraction(value)) if "/" in value else float(value)
    except (ValueError, ZeroDivisionError):
        return 0
    return int(round(fps)) if fps > 0 else 0

def int_from_probe(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0

def probe_ffprobe(cfg: AppConfig) -> tuple[int, int, int]:
    cmd = [
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
    if cfg.source_type == "rtsp" and cfg.tcp:
        cmd.extend(["-rtsp_transport", "tcp"])
    if not cfg.ssl_strict:
        cmd.extend(["-tls_verify", "0"])
    cmd.append(cfg.source_url)
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
    fps = fps_from_rate(values.get("avg_frame_rate", "")) or fps_from_rate(
        values.get("r_frame_rate", "")
    )
    return int_from_probe(values.get("width")), int_from_probe(values.get("height")), fps

def probe_rtsp(url: str) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("failed to probe RTSP frame size")
    return width, height, fps

def make_rtsp_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.RtspDecodedInputOptions()
    opt.url = cfg.source_url
    opt.latency_ms = cfg.latency_ms
    opt.tcp = cfg.tcp
    opt.insert_queue = True
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.source_fps = fps
    opt.codec = (
        pyneat.RtspCodec.H264
        if cfg.source_codec == "h264"
        else pyneat.RtspCodec.H265
        if cfg.source_codec == "h265"
        else pyneat.RtspCodec.MJPEG
    )
    if cfg.source_codec == "h264":
        opt.payload_type = 96
        opt.auto_caps_from_stream = True
        opt.fallback_h264_width = width
        opt.fallback_h264_height = height
    elif cfg.source_codec == "h265":
        opt.payload_type = 96
        opt.auto_caps_from_stream = True
        opt.dec_width = width
        opt.dec_height = height
    else:
        opt.mjpeg_payload_type = 26
        opt.dec_width = width
        opt.dec_height = height
    set_output_caps(opt.output_caps, fps, width, height)
    return opt

def make_http_mjpeg_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = cfg.source_url
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.source_fps = fps
    opt.ssl_strict = cfg.ssl_strict
    set_output_caps(opt.output_caps, fps, width, height)
    return opt

def set_output_caps(caps, fps: int, width: int, height: int) -> None:
    if width <= 0 or height <= 0 or fps <= 0:
        return
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any

def make_usb_camera_options(cfg: AppConfig):
    opt = pyneat.UsbCameraInputOptions()
    opt.camera_name = cfg.camera
    opt.width = cfg.source_width
    opt.height = cfg.source_height
    opt.framerate_num = cfg.source_fps
    return opt

def make_usb_decode_options(cfg: AppConfig, decoded_format):
    opt = pyneat.SimaDecodeOptions()
    opt.type = pyneat.SimaDecodeType.MJPEG
    opt.raw_output = True
    opt.out_format = decoded_format
    opt.dec_width = cfg.source_width
    opt.dec_height = cfg.source_height
    opt.dec_fps = cfg.source_fps
    return opt

def make_source_graph(cfg: AppConfig, fps: int, width: int, height: int, usb_camera=None):
    if cfg.source_type == "usb":
        camera = pyneat.Graph("usb_camera")
        camera.add(pyneat.nodes.usb_camera_input(make_usb_camera_options(cfg), usb_camera))
        return camera
    if cfg.source_type == "rtsp":
        return pyneat.groups.rtsp_decoded_input(make_rtsp_source_options(cfg, fps, width, height))
    return pyneat.groups.http_mjpeg_decoded_input(
        make_http_mjpeg_source_options(cfg, fps, width, height)
    )

def require_mjpeg_fps(cfg: AppConfig, fps: int) -> None:
    if cfg.source_codec == "mjpeg" and fps <= 0:
        raise RuntimeError(
            f"MJPEG source did not provide a valid frame rate; set source.{cfg.source_type}.fps "
            "or use a source with probeable FPS metadata"
        )

def probe_decoded_source(cfg: AppConfig, fps: int) -> tuple[int, int, int]:
    graph = pyneat.Graph("source_probe")
    graph.add(make_source_graph(cfg, fps, 0, 0))
    graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(1)))

    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    run = graph.build(run_options)
    try:
        sample = run.pull("frame")
    finally:
        run.close()
    if sample is None:
        raise RuntimeError("failed to probe decoded source frame")
    tensor = first_tensor_from_sample(sample)
    if tensor is None:
        raise RuntimeError("decoded source probe did not produce a tensor")
    return tensor_dim(tensor, "width"), tensor_dim(tensor, "height"), fps

def resolve_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    if cfg.source_type == "usb":
        return cfg.source_width, cfg.source_height, cfg.source_fps
    probed_w, probed_h, probed_fps = probe_ffprobe(cfg)
    fps = cfg.source_fps if cfg.source_fps > 0 else probed_fps
    if cfg.source_type == "rtsp":
        width, height = probed_w, probed_h
        if width <= 0 or height <= 0 or fps <= 0:
            rtsp_w, rtsp_h, rtsp_fps = probe_rtsp(cfg.source_url)
            width = width if width > 0 else rtsp_w
            height = height if height > 0 else rtsp_h
            fps = fps if fps > 0 else rtsp_fps
        require_mjpeg_fps(cfg, fps)
        return width, height, fps

    require_mjpeg_fps(cfg, fps)
    if probed_w > 0 and probed_h > 0:
        return probed_w, probed_h, fps
    width, height, _ = probe_decoded_source(cfg, fps)
    return width, height, fps

def tensor_dim(tensor, name: str) -> int:
    value = getattr(tensor, name)
    return int(value() if callable(value) else value)

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
