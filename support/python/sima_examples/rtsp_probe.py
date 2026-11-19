"""Read an RTSP stream's frame size and frame rate before the graph is built.

Mirrors support/cpp/rtsp_probe.h.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class RtspStreamInfo:
    width: int
    height: int
    fps: int


def probe_rtsp_stream_info(url: str) -> RtspStreamInfo:
    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"failed to open RTSP source for probing: {url}")
        info = RtspStreamInfo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            fps=round(capture.get(cv2.CAP_PROP_FPS) or 0),
        )
    finally:
        capture.release()
    if info.fps <= 0:
        raise RuntimeError(f"failed to probe RTSP frame rate: {url}")
    return info
