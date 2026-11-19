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


def probe_rtsp_stream_info(url: str, tcp: bool) -> RtspStreamInfo:
    """Read the stream with OpenCV's FFmpeg backend, over TCP when `tcp` is set."""
    # The FFmpeg backend reads its RTSP options from this variable when it opens the URL.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp" if tcp else "rtsp_transport;udp"
    )
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        info = RtspStreamInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=round(capture.get(cv2.CAP_PROP_FPS) or 0),
        )
    finally:
        capture.release()
    # An unreachable source and one that reports no frame rate fail the same way, as in C++.
    if info.fps <= 0:
        raise RuntimeError(f"failed to probe RTSP frame rate: {url}")
    return info
