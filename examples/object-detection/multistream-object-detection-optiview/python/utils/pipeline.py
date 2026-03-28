"""Pipeline builders and runtime helpers for the Python object-detection example."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import glob
import os
import sys
from typing import Any

from .config import AppConfig, VideoMode
from .model_family import ModelFamily


SOURCE_RUN_QUEUE_DEPTH = 4
SOURCE_OUTPUT_EVERY_N = 1

_DEFAULT_MODEL_NUM_BUFFERS = "3"
_DEFAULT_DECODER_NUM_BUFFERS = "7"
_OPTIVIEW_VIDEO_ENCODER_ENV = "SIMA_OPTIVIEW_VIDEO_ENCODER"


@dataclass(frozen=True)
class RtspProbe:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class RuntimeModules:
    cv2: Any
    np: Any
    pyneat: Any


@dataclass(frozen=True)
class SessionRun:
    session: Any
    run: Any
    model: Any | None = None


class OptiViewVideoEncoder(Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"


_RUNTIME_MODULES: RuntimeModules | None = None


def _set_env_if_unset(key: str, value: str) -> None:
    if os.getenv(key) is None:
        os.environ[key] = value


def apply_graphpipes_runtime_defaults() -> None:
    _set_env_if_unset("SIMA_FORCE_MODEL_NUM_BUFFERS", _DEFAULT_MODEL_NUM_BUFFERS)
    _set_env_if_unset("SIMA_FORCE_DECODER_NUM_BUFFERS", _DEFAULT_DECODER_NUM_BUFFERS)
    _set_env_if_unset("SIMA_FORCE_DECODER_POOL_BUFFERS", _DEFAULT_DECODER_NUM_BUFFERS)
    _set_env_if_unset("SIMA_PULL_TIMEOUT_DIAG", "0")


def graphpipes_decoder_num_buffers() -> int:
    return 7


def effective_writer_fps(cfg: AppConfig, probe: RtspProbe) -> int:
    return cfg.fps if cfg.fps > 0 else max(probe.fps, 10)


def optiview_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def optiview_json_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def producer_emit_period_s(cfg: AppConfig, probe: RtspProbe) -> float:
    _ = probe
    if cfg.fps <= 0:
        return 0.0
    return 1.0 / float(cfg.fps)


def optiview_video_encoder_from_env() -> OptiViewVideoEncoder:
    raw = (os.getenv(_OPTIVIEW_VIDEO_ENCODER_ENV) or "").strip().lower()
    if raw in ("sw", "software"):
        return OptiViewVideoEncoder.SOFTWARE
    return OptiViewVideoEncoder.HARDWARE


def load_runtime_modules() -> RuntimeModules:
    global _RUNTIME_MODULES
    if _RUNTIME_MODULES is not None:
        return _RUNTIME_MODULES

    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import cv2
        import numpy as np
        import pyneat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "runtime dependencies are missing; install numpy, OpenCV, and pyneat "
            "or run inside the NEAT Python environment"
        ) from exc

    _RUNTIME_MODULES = RuntimeModules(cv2=cv2, np=np, pyneat=pyneat)
    return _RUNTIME_MODULES


def probe_rtsp(cfg: AppConfig, url: str) -> RtspProbe:
    apply_graphpipes_runtime_defaults()
    runtime = load_runtime_modules()
    cap = runtime.cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open RTSP source for probing: {url}")

    width = int(cap.get(runtime.cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(runtime.cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = int(round(cap.get(runtime.cv2.CAP_PROP_FPS) or 0))
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"failed to probe RTSP frame size: {url}")

    if fps <= 0:
        fps = max(cfg.fps, 0)

    return RtspProbe(width=width, height=height, fps=fps)


def build_source_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    url: str,
    probe: RtspProbe,
) -> SessionRun:
    pyneat = runtime.pyneat

    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = cfg.latency_ms
    ro.tcp = cfg.tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.auto_caps_from_stream = True
    ro.fallback_h264_width = probe.width
    ro.fallback_h264_height = probe.height
    ro.fallback_h264_fps = probe.fps
    ro.sima_allocator_type = 2
    ro.out_format = "RGB"
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.output_caps.enable = True
    ro.output_caps.format = "RGB"
    ro.output_caps.width = probe.width
    ro.output_caps.height = probe.height
    if probe.fps > 0:
        ro.output_caps.fps = probe.fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    session = pyneat.Session()
    session.add(pyneat.groups.rtsp_decoded_input(ro))
    session.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(SOURCE_OUTPUT_EVERY_N)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = SOURCE_RUN_QUEUE_DEPTH
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = session.build(run_opt)
    return SessionRun(session=session, run=run)


def build_detection_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    family: ModelFamily,
    probe: RtspProbe,
) -> SessionRun:
    if family is not ModelFamily.YOLOV8:
        raise ValueError("unsupported model family for detector graph")

    apply_graphpipes_runtime_defaults()

    pyneat = runtime.pyneat
    np = runtime.np

    model_opt = pyneat.ModelOptions()
    model_opt.media_type = "video/x-raw"
    model_opt.format = "RGB"
    model_opt.input_max_width = probe.width
    model_opt.input_max_height = probe.height
    model_opt.input_max_depth = 3
    model = pyneat.Model(cfg.model.path, model_opt)

    input_opt = model.input_appsrc_options(False)
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.width = probe.width
    input_opt.height = probe.height
    input_opt.depth = 3
    for attr, value in (
        ("max_width", probe.width),
        ("max_height", probe.height),
        ("max_depth", 3),
    ):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, value)

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(model.preprocess())
    session.add(pyneat.groups.mla(model))
    session.add(
        pyneat.nodes.sima_box_decode(
            model,
            decode_type="yolov8",
            original_width=probe.width,
            original_height=probe.height,
            detection_threshold=cfg.min_score,
            nms_iou_threshold=cfg.nms_iou,
            top_k=cfg.max_detections,
        )
    )
    session.add(pyneat.nodes.output())

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return SessionRun(session=session, run=run, model=model)


def build_optiview_video_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
    stream_index: int,
    video_mode: VideoMode,
) -> SessionRun:
    _ = video_mode
    pyneat = runtime.pyneat
    np = runtime.np

    encoder = optiview_video_encoder_from_env()
    if encoder is OptiViewVideoEncoder.SOFTWARE:
        raise RuntimeError(
            "software video encoding is not available through public pyneat; "
            "unset SIMA_OPTIVIEW_VIDEO_ENCODER or set it to hw"
        )

    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.use_simaai_pool = False
    for attr, value in (
        ("max_width", probe.width),
        ("max_height", probe.height),
        ("max_depth", 3),
    ):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, value)

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(pyneat.nodes.video_convert())
    session.add(
        pyneat.nodes.h264_encode_sima(
            probe.width,
            probe.height,
            effective_writer_fps(cfg, probe),
            bitrate_kbps=2500,
            profile="baseline",
            level="4.1",
        )
    )

    udp_opt = pyneat.UdpH264OutputGroupOptions()
    udp_opt.payload_type = 96
    udp_opt.config_interval = 1
    udp_opt.udp_host = cfg.optiview_host
    udp_opt.udp_port = optiview_video_port_for_stream(cfg.optiview_video_port_base, stream_index)
    udp_opt.udp_sync = False
    udp_opt.udp_async = False
    session.add(pyneat.groups.udp_h264_output_group(udp_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return SessionRun(session=session, run=run)


def build_optiview_json_output(
    runtime: RuntimeModules,
    cfg: AppConfig,
    stream_index: int,
):
    pyneat = runtime.pyneat
    channel = pyneat.OptiViewChannelOptions()
    channel.host = cfg.optiview_host
    channel.channel = stream_index
    channel.video_port_base = cfg.optiview_video_port_base
    channel.json_port_base = cfg.optiview_json_port_base
    return pyneat.OptiViewJsonOutput(channel)
