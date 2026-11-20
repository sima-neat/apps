"""Read an RTSP stream's frame size and frame rate before the graph is built.

Mirrors support/cpp/sima_examples/rtsp_probe.h.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class RtspStreamInfo:
    width: int
    height: int
    fps: int


def probe_rtsp_stream_info(
    url: str, tcp: bool, configured: RtspStreamInfo = RtspStreamInfo(0, 0, 0)
) -> RtspStreamInfo:
    """The stream's frame size and rate: each value `configured` sets above 0, and the probe's
    for the rest. The probe reads the stream with OpenCV's FFmpeg backend, over TCP when `tcp`
    is set, and runs only when a value is 0."""
    if min(configured.width, configured.height, configured.fps) > 0:
        return configured
    probed = _read_stream_info(url, tcp)
    info = RtspStreamInfo(
        width=configured.width or probed.width,
        height=configured.height or probed.height,
        fps=configured.fps or probed.fps,
    )
    # An unreachable source fails here too, as in C++.
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError(
            f"failed to probe RTSP frame size: {url}; set input.width and input.height"
        )
    if info.fps <= 0:
        raise RuntimeError(f"failed to probe RTSP frame rate: {url}; set input.fps")
    return info


def _read_stream_info(url: str, tcp: bool) -> RtspStreamInfo:
    # The FFmpeg backend reads its RTSP options from this variable when it opens the URL.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp" if tcp else "rtsp_transport;udp"
    )
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        return RtspStreamInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=round(capture.get(cv2.CAP_PROP_FPS) or 0),
        )
    finally:
        capture.release()
