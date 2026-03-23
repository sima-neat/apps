"""Detector pipeline helpers for the multistream YOLOX/YOLOv8 example."""

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

_SOURCE_STARTUP_PULL_TIMEOUT_MS = 50000
_SOURCE_PULL_TIMEOUT_MS = 10000
_SOURCE_STARTUP_STAGGER_S = 0.5
_OPTIVIEW_BITRATE_KBPS = 2500


def optiview_video_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def optiview_json_port_for_stream(port_base: int, stream_index: int) -> int:
    return int(port_base) + int(stream_index)


def effective_writer_fps(cfg: AppConfig, probe: RtspProbe) -> int:
    return cfg.fps if cfg.fps > 0 else max(probe.fps, 10)


def source_output_every_n(cfg: AppConfig, probe: RtspProbe) -> int:
    if cfg.fps <= 0 or probe.fps <= 0 or cfg.fps >= probe.fps:
        return 1
    return max(1, probe.fps // cfg.fps)


def detector_stage_names(family: str) -> tuple[str, ...]:
    lowered = str(family).strip().lower()
    if lowered == "yolov8":
        return ("input", "quant_tess", "mla", "sima_box_decode", "output")
    if lowered == "yolox":
        return ("input", "quant_tess", "mla", "detess_dequant", "output")
    raise ValueError(f"unsupported model family for detector graph: {family}")


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
    pyneat = runtime.pyneat
    opt = pyneat.ModelOptions()
    opt.media_type = "application/vnd.simaai.tensor"
    opt.format = ""
    return pyneat.Model(cfg.model.path, opt)


def read_preproc_contract(runtime: RuntimeModules, model: Any) -> QuantTessCpuPreproc:
    pyneat = runtime.pyneat
    pre = pyneat.PreprocOptions(model)
    cfg_json = dict(getattr(pre, "config_json", None) or {})
    return QuantTessCpuPreproc(
        width=int(cfg_json.get("output_width") or cfg_json.get("input_width") or 640),
        height=int(cfg_json.get("output_height") or cfg_json.get("input_height") or 640),
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
    ro.out_format = "RGB"
    ro.decoder_raw_output = False
    ro.auto_caps_from_stream = True
    ro.use_videoconvert = False
    ro.use_videoscale = True
    ro.fallback_h264_width = probe.width
    ro.fallback_h264_height = probe.height
    if probe.fps > 0:
        ro.fallback_h264_fps = probe.fps
    ro.output_caps.enable = True
    ro.output_caps.format = "RGB"
    ro.output_caps.width = probe.width
    ro.output_caps.height = probe.height
    if probe.fps > 0:
        ro.output_caps.fps = probe.fps
    ro.output_caps.memory = pyneat.CapsMemory.SystemMemory

    session = pyneat.Session()
    session.add(pyneat.groups.rtsp_decoded_input(ro))
    session.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(source_output_every_n(cfg, probe))))
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
    family: str,
    probe: RtspProbe,
    quant_preproc: QuantTessCpuPreproc,
):
    detector_stage_names(family)

    pyneat = runtime.pyneat
    np = runtime.np

    session = pyneat.Session()
    session.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    session.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    session.add(pyneat.groups.mla(model))

    if family == "yolov8":
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
    elif family == "yolox":
        session.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    else:
        raise ValueError(f"unsupported model family for detector graph: {family}")

    session.add(pyneat.nodes.output())

    seed = np.zeros((quant_preproc.height, quant_preproc.width, 3), dtype=np.float32)
    run = session.build(seed)
    return session, run


def build_optiview_video_run(runtime: RuntimeModules, cfg: AppConfig, probe: RtspProbe, stream_index: int):
    pyneat = runtime.pyneat
    np = runtime.np

    input_opt = pyneat.InputOptions()
    input_opt.media_type = "video/x-raw"
    input_opt.format = "RGB"
    input_opt.use_simaai_pool = False

    session = pyneat.Session()
    session.add(pyneat.nodes.input(input_opt))
    session.add(pyneat.nodes.video_convert())
    session.add(
        pyneat.nodes.h264_encode_sima(
            probe.width,
            probe.height,
            effective_writer_fps(cfg, probe),
            bitrate_kbps=_OPTIVIEW_BITRATE_KBPS,
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
    return session, run


def build_optiview_json_output(runtime: RuntimeModules, cfg: AppConfig, stream_index: int):
    pyneat = runtime.pyneat
    channel = pyneat.OptiViewChannelOptions()
    channel.host = cfg.optiview_host
    channel.channel = stream_index
    channel.video_port_base = cfg.optiview_video_port_base
    channel.json_port_base = cfg.optiview_json_port_base
    return pyneat.OptiViewJsonOutput(channel)
