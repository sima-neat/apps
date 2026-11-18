"""Build the object-detection metadata message that Insight draws over the video.

Mirrors `sima_examples::detection_egress` in support/object_detection/detection_egress.h,
so the C++ and Python examples send the same JSON.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

UNKNOWN_LABEL = "unknown"
RTP_CLOCK_HZ = 90_000


class Box(Protocol):
    """A detection in frame pixels, such as `pyneat.detections.Box`."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class FrameMetadata:
    """The identity of the frame the boxes belong to."""

    stream_index: int
    stream_id: str
    frame_id: int = -1
    pts_ns: int = -1
    dts_ns: int = -1
    duration_ns: int = -1
    input_seq: int = -1
    orig_input_seq: int = -1
    rtp_timestamp: int | None = None


def rtp_timestamp_from_pts_ns(pts_ns: int) -> int:
    """The 90 kHz RTP timestamp Insight uses to pair metadata with its video frame."""
    return (pts_ns * RTP_CLOCK_HZ // 1_000_000_000) & 0xFFFFFFFF


def serialize(
    boxes: Iterable[Box],
    labels: list[str],
    frame_w: int,
    frame_h: int,
    frame: FrameMetadata,
) -> str:
    objects = []
    for index, box in enumerate(boxes, start=1):
        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        width = min(max(0, int(box.x2 - box.x1)), frame_w - x1)
        height = min(max(0, int(box.y2 - box.y1)), frame_h - y1)
        label = (
            labels[box.class_id] if 0 <= box.class_id < len(labels) else UNKNOWN_LABEL
        )
        objects.append(
            {
                "id": f"obj_{index}",
                "label": label,
                "confidence": box.score,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(max(0, width)),
                    float(max(0, height)),
                ],
            }
        )

    payload = {
        "type": "object-detection",
        "timestamp": frame.pts_ns // 1_000_000 if frame.pts_ns >= 0 else -1,
        "frame_id": str(frame.frame_id) if frame.frame_id >= 0 else "",
        "stream_id": frame.stream_id,
        "stream_index": frame.stream_index,
        "pts_ns": frame.pts_ns,
        "dts_ns": frame.dts_ns,
        "duration_ns": frame.duration_ns,
        "input_seq": frame.input_seq,
        "orig_input_seq": frame.orig_input_seq,
        "data": {"objects": objects},
    }
    if frame.rtp_timestamp is not None:
        payload["rtp_timestamp"] = frame.rtp_timestamp
    return json.dumps(payload, separators=(",", ":"))
