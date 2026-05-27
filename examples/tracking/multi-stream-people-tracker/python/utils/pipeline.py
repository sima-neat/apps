"""Pipeline builders and runtime helpers for the Python multi-camera example."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import sys
from typing import Any

from .config import AppConfig


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
class QuantTessCpuPreproc:
    width: int
    height: int
    aspect_ratio: bool
    padding_type: str


_RUNTIME_MODULES: RuntimeModules | None = None

_YOLOV8_BOXDECODE_DEFAULTS = {
    "detection_threshold": 0.6,
    "nms_iou_threshold": 0.5,
    "topk": 24,
}

_SOURCE_STARTUP_PULL_TIMEOUT_MS = 50000
_SOURCE_PULL_TIMEOUT_MS = 10000
_SOURCE_STARTUP_STAGGER_S = 0.5


def insight_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def insight_metadata_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def effective_writer_fps(cfg: AppConfig, probe: RtspProbe) -> int:
    return cfg.fps if cfg.fps > 0 else max(probe.fps, 10)


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


def probe_rtsp(url: str) -> RtspProbe:
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
    return RtspProbe(width=width, height=height, fps=max(0, fps))


def load_detector_model(runtime: RuntimeModules, cfg: AppConfig):
    # Load the model pack in tensor mode because CPU preproc produces the
    # FP32 image tensor consumed by QuantTess in the explicit detector graph.
    pyneat = runtime.pyneat
    opt = pyneat.ModelOptions()
    opt.media_type = "application/vnd.simaai.tensor"
    opt.format = ""
    return pyneat.Model(cfg.model, opt)


def read_preproc_contract(runtime: RuntimeModules, model: Any) -> QuantTessCpuPreproc:
    # Reuse the packaged preproc geometry even though preprocessing happens on CPU.
    pyneat = runtime.pyneat
    pre = pyneat.PreprocOptions(model)
    cfg_json = dict(getattr(pre, "config_json", None) or {})
    width = int(cfg_json.get("output_width") or cfg_json.get("input_width") or 640)
    height = int(cfg_json.get("output_height") or cfg_json.get("input_height") or 640)
    return QuantTessCpuPreproc(
        width=width,
        height=height,
        aspect_ratio=bool(cfg_json.get("aspect_ratio", False)),
        padding_type=str(cfg_json.get("padding_type", "CENTER")).upper(),
    )


def build_source_run(runtime: RuntimeModules, cfg: AppConfig, url: str, probe: RtspProbe):
    pyneat = runtime.pyneat
    ro = pyneat.RtspDecodedInputOptions()
    ro.url = url
    ro.latency_ms = cfg.latency_ms
    ro.tcp = cfg.tcp
    ro.payload_type = 96
    ro.insert_queue = True
    ro.auto_caps_from_stream = True
    ro.sima_allocator_type = 2
    ro.decoder_raw_output = False
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.fallback_h264_width = probe.width
    ro.fallback_h264_height = probe.height
    if probe.fps > 0:
        ro.fallback_h264_fps = probe.fps
    ro.output_caps.enable = True
    ro.output_caps.width = probe.width
    ro.output_caps.height = probe.height
    if probe.fps > 0:
        ro.output_caps.fps = probe.fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    session = pyneat.Session()
    session.add(pyneat.groups.rtsp_decoded_input(ro))
    session.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(1)))
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 4
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = session.build(run_opt)
    return session, run


def build_detection_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
):
    pyneat = runtime.pyneat
    np = runtime.np

    model_opt = pyneat.ModelOptions()
    model_opt.preprocess.kind = pyneat.InputKind.Image
    model_opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    model_opt.preprocess.input_max_width = probe.width
    model_opt.preprocess.input_max_height = probe.height
    model_opt.preprocess.input_max_depth = 3
    model_opt.decode_type = pyneat.BoxDecodeType.YoloV8
    model_opt.score_threshold = (
        cfg.detection_threshold
        if cfg.detection_threshold is not None
        else _YOLOV8_BOXDECODE_DEFAULTS["detection_threshold"]
    )
    model_opt.nms_iou_threshold = (
        cfg.nms_iou_threshold
        if cfg.nms_iou_threshold is not None
        else _YOLOV8_BOXDECODE_DEFAULTS["nms_iou_threshold"]
    )
    model_opt.top_k = (
        cfg.top_k if cfg.top_k is not None else _YOLOV8_BOXDECODE_DEFAULTS["topk"]
    )
    model_opt.boxdecode_original_width = probe.width
    model_opt.boxdecode_original_height = probe.height
    model = pyneat.Model(cfg.model, model_opt)

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
            decode_type=pyneat.BoxDecodeType.YoloV8,
            original_width=probe.width,
            original_height=probe.height,
            detection_threshold=(
                cfg.detection_threshold
                if cfg.detection_threshold is not None
                else _YOLOV8_BOXDECODE_DEFAULTS["detection_threshold"]
            ),
            nms_iou_threshold=(
                cfg.nms_iou_threshold
                if cfg.nms_iou_threshold is not None
                else _YOLOV8_BOXDECODE_DEFAULTS["nms_iou_threshold"]
            ),
            top_k=cfg.top_k if cfg.top_k is not None else _YOLOV8_BOXDECODE_DEFAULTS["topk"],
        )
    )
    session.add(pyneat.nodes.output())

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.preset = pyneat.RunPreset.Realtime
    run_opt.queue_depth = 1
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_opt.output_memory = pyneat.OutputMemory.Owned
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return session, run


def build_insight_video_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
    stream_index: int,
):
    pyneat = runtime.pyneat
    np = runtime.np

    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.use_simaai_pool = False
    input_opt.max_width = probe.width
    input_opt.max_height = probe.height
    input_opt.max_depth = 3

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
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
    sender_opt.encoder.bitrate_kbps = cfg.bitrate_kbps
    sender_opt.encoder.profile = "baseline"
    sender_opt.encoder.level = "4.1"
    session.add(pyneat.groups.video_sender(sender_opt))

    seed = pyneat.Tensor.from_numpy(
        np.zeros((probe.height, probe.width, 3), dtype=np.uint8),
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 2
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return session, run


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
