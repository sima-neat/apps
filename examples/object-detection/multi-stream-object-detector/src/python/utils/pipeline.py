"""Pipeline builders and runtime helpers for the Python object-detection example."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
import sys
from typing import Any

from .config import AppConfig
from .model_family import ModelFamily


SOURCE_RUN_QUEUE_DEPTH = 4
SOURCE_OUTPUT_EVERY_N = 1

_DEFAULT_MODEL_NUM_BUFFERS = "3"
_DEFAULT_DECODER_NUM_BUFFERS = "7"


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
class GraphRun:
    graph: Any
    run: Any
    model: Any | None = None


_RUNTIME_MODULES: RuntimeModules | None = None


def _set_env_if_unset(key: str, value: str) -> None:
    if os.getenv(key) is None:
        os.environ[key] = value


def apply_runtime_env_defaults() -> None:
    _set_env_if_unset("SIMA_FORCE_MODEL_NUM_BUFFERS", _DEFAULT_MODEL_NUM_BUFFERS)
    _set_env_if_unset("SIMA_FORCE_DECODER_NUM_BUFFERS", _DEFAULT_DECODER_NUM_BUFFERS)
    _set_env_if_unset("SIMA_FORCE_DECODER_POOL_BUFFERS", _DEFAULT_DECODER_NUM_BUFFERS)
    _set_env_if_unset("SIMA_PULL_TIMEOUT_DIAG", "0")


def effective_writer_fps(cfg: AppConfig, probe: RtspProbe) -> int:
    return cfg.fps if cfg.fps > 0 else max(probe.fps, 10)


def insight_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def insight_metadata_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def producer_emit_period_s(cfg: AppConfig, probe: RtspProbe) -> float:
    _ = probe
    if cfg.fps <= 0:
        return 0.0
    return 1.0 / float(cfg.fps)


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
    apply_runtime_env_defaults()
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


def _set_optional_input_limits(input_opt: Any, width: int, height: int, depth: int) -> None:
    for attr, value in (
        ("max_width", width),
        ("max_height", height),
        ("max_depth", depth),
    ):
        if hasattr(input_opt, attr):
            setattr(input_opt, attr, value)


def build_source_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    url: str,
    probe: RtspProbe,
) -> GraphRun:
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
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.output_caps.enable = True
    ro.output_caps.width = probe.width
    ro.output_caps.height = probe.height
    if probe.fps > 0:
        ro.output_caps.fps = probe.fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    graph = pyneat.Graph("rtsp_source")
    graph.add(pyneat.groups.rtsp_decoded_input(ro))
    graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(SOURCE_OUTPUT_EVERY_N)))

    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = SOURCE_RUN_QUEUE_DEPTH
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = graph.build(run_opt)
    return GraphRun(graph=graph, run=run)


def build_detection_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    family: ModelFamily,
    probe: RtspProbe,
) -> GraphRun:
    if family is not ModelFamily.YOLOV8:
        raise ValueError("unsupported model family for detector graph")

    apply_runtime_env_defaults()

    pyneat = runtime.pyneat
    np = runtime.np

    model_opt = pyneat.ModelOptions()
    model_opt.preprocess.kind = pyneat.InputKind.Image
    model_opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    model_opt.preprocess.input_max_width = probe.width
    model_opt.preprocess.input_max_height = probe.height
    model_opt.preprocess.input_max_depth = 3
    model_opt.decode_type = pyneat.BoxDecodeType.YoloV8
    model_opt.score_threshold = cfg.min_score
    model_opt.nms_iou_threshold = cfg.nms_iou
    model_opt.top_k = cfg.max_detections
    model_opt.boxdecode_original_width = probe.width
    model_opt.boxdecode_original_height = probe.height
    model = pyneat.Model(cfg.model.path, model_opt)

    input_opt = model.input_appsrc_options(False)
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "RGB"
    input_opt.width = probe.width
    input_opt.height = probe.height
    input_opt.depth = 3
    _set_optional_input_limits(input_opt, probe.width, probe.height, 3)

    graph = pyneat.Graph("detector")
    graph.add(pyneat.nodes.input(input_opt))
    graph.add(model.preprocess())
    graph.add(pyneat.groups.mla(model))
    graph.add(
        pyneat.nodes.sima_box_decode(
            model,
            decode_type=pyneat.BoxDecodeType.YoloV8,
            original_width=probe.width,
            original_height=probe.height,
            detection_threshold=cfg.min_score,
            nms_iou_threshold=cfg.nms_iou,
            top_k=cfg.max_detections,
        )
    )
    graph.add(pyneat.nodes.output())

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = graph.build([seed], pyneat.RunMode.Async, run_opt)
    return GraphRun(graph=graph, run=run, model=model)


def build_insight_video_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
    stream_index: int,
) -> GraphRun:
    pyneat = runtime.pyneat
    np = runtime.np

    input_opt = pyneat.InputOptions()
    input_opt.payload_type = pyneat.PayloadType.Image
    input_opt.format = "RGB"
    input_opt.width = probe.width
    input_opt.height = probe.height
    input_opt.depth = 3
    input_opt.use_simaai_pool = False
    _set_optional_input_limits(input_opt, probe.width, probe.height, 3)

    graph = pyneat.Graph("insight_video")
    graph.add(pyneat.nodes.input(input_opt))
    sender_opt = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(
        probe.width,
        probe.height,
        effective_writer_fps(cfg, probe),
    )
    sender_opt.host = cfg.insight_host
    sender_opt.channel = stream_index
    sender_opt.video_port_base = cfg.insight_video_port_base
    sender_opt.sync = False
    sender_opt.async_ = False
    sender_opt.rtp.payload_type = 96
    sender_opt.rtp.config_interval = 1
    sender_opt.encoder.bitrate_kbps = 2500
    sender_opt.encoder.profile = "baseline"
    sender_opt.encoder.level = "4.1"
    graph.add(pyneat.groups.video_sender(sender_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = graph.build([seed], pyneat.RunMode.Async, run_opt)
    return GraphRun(graph=graph, run=run)


def build_insight_metadata_output(
    runtime: RuntimeModules,
    cfg: AppConfig,
    stream_index: int,
):
    pyneat = runtime.pyneat
    options = pyneat.MetadataSenderOptions()
    options.host = cfg.insight_host
    options.channel = stream_index
    options.metadata_port_base = cfg.insight_metadata_port_base
    return pyneat.MetadataSender(options)
