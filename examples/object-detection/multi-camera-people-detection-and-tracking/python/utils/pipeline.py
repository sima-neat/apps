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


def optiview_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def optiview_json_port_for_stream(port_base: int, stream_index: int) -> int:
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
    ro.out_format = "BGR"
    ro.decoder_raw_output = False
    ro.auto_caps_from_stream = True
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.fallback_h264_width = probe.width
    ro.fallback_h264_height = probe.height
    if probe.fps > 0:
        ro.fallback_h264_fps = probe.fps
    ro.output_caps.enable = True
    ro.output_caps.format = "BGR"
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
    model: Any,
    probe: RtspProbe,
    quant_preproc: QuantTessCpuPreproc,
):
    pyneat = runtime.pyneat
    np = runtime.np

    # Keep the detector graph explicit in the example:
    # tensor input -> QuantTess -> MLA -> BoxDecode.
    session = pyneat.Session()
    session.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    session.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    session.add(pyneat.groups.mla(model))
    session.add(
        pyneat.nodes.sima_box_decode(
            model,
            decode_type="yolov8",
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

    seed = np.zeros((quant_preproc.height, quant_preproc.width, 3), dtype=np.float32)
    run = session.build(seed)
    return session, run


def build_optiview_video_run(
    runtime: RuntimeModules,
    cfg: AppConfig,
    probe: RtspProbe,
    stream_index: int,
):
    pyneat = runtime.pyneat
    np = runtime.np

    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "BGR"
    input_opt.use_simaai_pool = False

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(pyneat.nodes.video_convert())
    session.add(
        pyneat.nodes.h264_encode_sima(
            probe.width,
            probe.height,
            effective_writer_fps(cfg, probe),
            bitrate_kbps=cfg.bitrate_kbps,
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
        image_format=pyneat.PixelFormat.BGR,
    )
    run_opt = pyneat.RunOptions()
    run_opt.queue_depth = 2
    run_opt.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run = session.build(seed, pyneat.RunMode.Async, run_opt)
    return session, run


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
