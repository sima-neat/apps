"""This example's source setup and saved frames.

The source options and probes follow source.type: RTSP and HTTP MJPEG streams are decoded by
the source group, and a USB camera's MJPEG is decoded by a separate node so its own JPEG can
go to Insight unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import pyneat

from .app_config import AppConfig
from .detections import draw_boxes
from .frames import first_tensor_from_sample, joined_field, tensor_bgr_from_decoded
from .sources import probe_decoded_size, probe_ffprobe, probe_rtsp


def _set_output_caps(caps, fps: int, width: int, height: int) -> None:
    if width <= 0 or height <= 0 or fps <= 0:
        return
    caps.enable = True
    caps.format = pyneat.Format.NV12
    caps.width = width
    caps.height = height
    caps.fps = fps
    caps.memory = pyneat.CapsMemory.Any


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
    _set_output_caps(opt.output_caps, fps, width, height)
    return opt


def make_http_mjpeg_source_options(cfg: AppConfig, fps: int, width: int, height: int):
    opt = pyneat.HttpMjpegDecodedInputOptions()
    opt.url = cfg.source_url
    opt.decoder_name = "decoder"
    opt.decoder_raw_output = True
    opt.source_fps = fps
    opt.ssl_strict = cfg.ssl_strict
    _set_output_caps(opt.output_caps, fps, width, height)
    return opt


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
    """The source as one graph: a decoded RTSP or HTTP MJPEG group, or the USB camera node."""
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


def resolve_source_geometry(cfg: AppConfig) -> tuple[int, int, int]:
    """Width, height and frame rate of the source; source.<type>.fps, when set, wins.

    A USB camera uses its configured mode. A stream is probed with ffprobe, then OpenCV for an
    RTSP stream, or one decoded frame for an HTTP stream.
    """
    if cfg.source_type == "usb":
        return cfg.source_width, cfg.source_height, cfg.source_fps
    probed_w, probed_h, probed_fps = probe_ffprobe(
        cfg.source_url,
        rtsp_tcp=cfg.source_type == "rtsp" and cfg.tcp,
        tls_verify=cfg.ssl_strict,
    )
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
    width, height = probe_decoded_size(make_source_graph(cfg, fps, 0, 0))
    return width, height, fps


def maybe_save_debug_frame(cfg: AppConfig, processed: int, sample, boxes: list[dict]) -> None:
    """Every output.save_every-th frame, write the frame with its boxes to output.save_dir."""
    if not cfg.save_dir or cfg.save_every <= 0 or processed % cfg.save_every != 0:
        return

    frame = first_tensor_from_sample(joined_field(sample, "frame", 0))
    if frame is None:
        raise RuntimeError("joined frame field has no tensor")
    annotated = tensor_bgr_from_decoded(frame)
    draw_boxes(annotated, boxes, cfg.min_score)
    out_path = Path(cfg.save_dir) / f"frame_{processed}.jpg"
    if not cv2.imwrite(str(out_path), annotated):
        print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)
