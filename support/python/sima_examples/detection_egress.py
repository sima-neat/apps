"""Build the object-detection and tracking metadata messages that Insight draws over the video.

Mirrors `sima_examples::detection_egress` in support/cpp/sima_examples/detection_egress.h,
so the C++ and Python examples send the same JSON values (key order differs).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

UNKNOWN_LABEL = "unknown"


class Box(Protocol):
    """A detection in frame pixels, such as `pyneat.detections.Box`."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


class Track(Box, Protocol):
    """A tracked detection: a box and the ID its tracker gave it."""

    track_id: int


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
    # The RTP timestamp an application read from the source packets, sent when set. Insight
    # pairs metadata with video by `timestamp`, the frame's presentation time in milliseconds.
    rtp_timestamp: int | None = None


def frame_metadata(sample, stream_index: int) -> FrameMetadata:
    """The identity of a pulled sample, such as a `pyneat.Sample`, on stream `stream_index`."""
    return FrameMetadata(
        stream_index=stream_index,
        stream_id=sample.stream_id,
        frame_id=sample.frame_id,
        pts_ns=sample.pts_ns,
        dts_ns=sample.dts_ns,
        duration_ns=sample.duration_ns,
        input_seq=sample.input_seq,
        orig_input_seq=sample.orig_input_seq,
    )


def serialize(
    boxes: Iterable[Box],
    labels: list[str],
    frame_w: int,
    frame_h: int,
    frame: FrameMetadata,
) -> str:
    """The object-detection message: each box with its label, numbered obj_1, obj_2, and so on."""
    objects = [
        _entry(f"obj_{index}", box, labels, frame_w, frame_h)
        for index, box in enumerate(boxes, start=1)
    ]
    return _message("object-detection", {"objects": objects}, frame)


def track_entries(
    tracks: Iterable[Track], labels: list[str], frame_w: int, frame_h: int
) -> list[dict]:
    """The `data.tracks` entries of a tracking message, for an application that builds its own
    message: each box with its label and its track ID as `id`."""
    return [_entry(str(track.track_id), track, labels, frame_w, frame_h) for track in tracks]


def serialize_tracks(
    tracks: Iterable[Track],
    labels: list[str],
    frame_w: int,
    frame_h: int,
    frame: FrameMetadata,
) -> str:
    """The tracking message: each box with its label and its track ID, which Insight colors and
    draws with a trail."""
    return _message("tracking", {"tracks": track_entries(tracks, labels, frame_w, frame_h)}, frame)


def _clip(value: float, limit: int) -> float:
    """0 to `limit`, as Core clamps: max(0, min(value, limit)), so NaN gives 0."""
    return max(0.0, min(float(value), float(limit)))


def _entry(entry_id: str, box: Box, labels: list[str], frame_w: int, frame_h: int) -> dict:
    """One box as Insight reads it: [x, y, width, height] in frame pixels. Both corners are
    clipped to the frame before the size is measured, so a box past an edge gets its visible
    width, and a box outside the frame gets zero size."""
    x1 = int(_clip(box.x1, frame_w))
    y1 = int(_clip(box.y1, frame_h))
    width = max(0, int(_clip(box.x2, frame_w)) - x1)
    height = max(0, int(_clip(box.y2, frame_h)) - y1)
    label = labels[box.class_id] if 0 <= box.class_id < len(labels) else UNKNOWN_LABEL
    return {
        "id": entry_id,
        "label": label,
        "confidence": box.score,
        "bbox": [float(x1), float(y1), float(width), float(height)],
    }


def _message(metadata_type: str, data: dict, frame: FrameMetadata) -> str:
    payload = {
        "type": metadata_type,
        "timestamp": frame.pts_ns // 1_000_000 if frame.pts_ns >= 0 else -1,
        "frame_id": str(frame.frame_id) if frame.frame_id >= 0 else "",
        "stream_id": frame.stream_id,
        "stream_index": frame.stream_index,
        "pts_ns": frame.pts_ns,
        "dts_ns": frame.dts_ns,
        "duration_ns": frame.duration_ns,
        "input_seq": frame.input_seq,
        "orig_input_seq": frame.orig_input_seq,
        "data": data,
    }
    if frame.rtp_timestamp is not None:
        payload["rtp_timestamp"] = frame.rtp_timestamp
    return json.dumps(payload, separators=(",", ":"))
