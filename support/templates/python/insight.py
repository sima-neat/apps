"""Detection metadata for Insight, in the format it draws over the video."""

from __future__ import annotations

import json


def build_metadata_boxes(boxes: list[dict], labels: list[str], frame_w: int, frame_h: int) -> list[dict]:
    """Insight objects for parsed boxes: an id, a class label, the score and a clipped x, y, w, h."""
    metadata_boxes = []
    for index, box in enumerate(boxes, start=1):
        x = max(0, int(box["x1"]))
        y = max(0, int(box["y1"]))
        w = max(0, int(box["x2"] - box["x1"]))
        h = max(0, int(box["y2"] - box["y1"]))
        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y
        class_id = int(box["class_id"])
        metadata_boxes.append(
            {
                "id": f"obj_{index}",
                "label": labels[class_id] if 0 <= class_id < len(labels) else "unknown",
                "confidence": float(box["score"]),
                "bbox": [float(x), float(y), float(max(0, w)), float(max(0, h))],
            }
        )
    return metadata_boxes


def send_metadata(sender, metadata_type: str, data: dict, sample) -> None:
    """Send data as JSON, stamped with the sample's timestamp and frame ID.

    Insight matches the timestamp to the RTP timestamp of the video frame it came from.
    """
    timestamp_ms = int(sample.pts_ns // 1_000_000) if sample.pts_ns >= 0 else -1
    frame_id = str(sample.frame_id) if sample.frame_id >= 0 else ""
    sender.send_metadata(
        metadata_type,
        json.dumps(data, separators=(",", ":")),
        timestamp_ms,
        frame_id,
    )
