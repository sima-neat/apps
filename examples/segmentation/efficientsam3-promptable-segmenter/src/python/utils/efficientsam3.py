"""EfficientSAM3's host work: the model input made from a decoded frame, the instances read from
the model's outputs, and the metadata Insight draws. Mirrors utils/efficientsam3.cpp in the C++
implementation, so both send the same metadata.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# The model reads a 1008x1008 RGB image.
MODEL_SIZE = 1008
# MetadataSender rejects a message above 65507 bytes; half of it leaves room for the envelope.
METADATA_BYTE_BUDGET = 32768
# The model sometimes returns one object twice, e.g. a whole person and their upper body. An
# instance whose box overlaps a higher-scoring one by more than this intersection-over-union is
# dropped as a duplicate.
DUPLICATE_IOU = 0.6


@dataclass(frozen=True)
class SegmentOptions:
    width: int  # Frame size the boxes and outlines are scaled to, in pixels.
    height: int
    label: str  # The prompt, which labels every instance.
    min_score: float
    max_detections: int
    mask_threshold: float


def preprocess(frame: np.ndarray, out: np.ndarray) -> None:
    """Write the model input for an NV12 frame (height * 3 / 2 rows) into `out`: RGB at
    MODEL_SIZE x MODEL_SIZE, scaled to -1..1."""
    height, width = frame.shape[0] * 2 // 3, frame.shape[1]
    shrink = width > MODEL_SIZE or height > MODEL_SIZE
    interpolation = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    y = cv2.resize(frame[:height], (MODEL_SIZE, MODEL_SIZE), interpolation=interpolation)
    uv = frame[height:].reshape(height // 2, width // 2, 2)
    uv = cv2.resize(uv, (MODEL_SIZE // 2, MODEL_SIZE // 2), interpolation=interpolation)
    rgb = cv2.cvtColorTwoPlane(y, uv, cv2.COLOR_YUV2RGB_NV12)
    np.multiply(rgb, np.float32(1.0 / 127.5), out=out, casting="unsafe")
    out -= np.float32(1.0)


def mask_outline(mask_logits, box, width: int, height: int, logit_threshold: float) -> list:
    """The outline of the largest region of one instance's mask inside its box, in frame pixels;
    empty when there is none."""
    mask_h, mask_w = mask_logits.shape
    cell_w, cell_h = width / mask_w, height / mask_h
    x0, y0, x1, y1 = box
    mx0, my0 = max(0, math.floor(x0 / cell_w) - 1), max(0, math.floor(y0 / cell_h) - 1)
    mx1 = min(mask_w, math.ceil(x1 / cell_w) + 1)
    my1 = min(mask_h, math.ceil(y1 / cell_h) + 1)
    out_x0, out_y0 = round(mx0 * cell_w), round(my0 * cell_h)
    out_size = (round(mx1 * cell_w) - out_x0, round(my1 * cell_h) - out_y0)
    # Upscale the mask cells under the box before thresholding, as SAM 3 does, so the outline
    # follows the interpolated mask.
    cells = np.ascontiguousarray(mask_logits[my0:my1, mx0:mx1])
    inside = cv2.resize(cells, out_size, interpolation=cv2.INTER_LINEAR) > logit_threshold
    contours, _ = cv2.findContours(
        inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    return [[int(x) + out_x0, int(y) + out_y0] for [[x, y]] in polygon]


def box_iou(a, b) -> float:
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def segments_of(detections, masks, options: SegmentOptions) -> list[dict]:
    """The instances of one result, best first. `detections` holds queries x 6 values (x1, y1, x2,
    y2, score, -) in model pixels, and `masks` mask_h x mask_w x queries logits; either may have a
    leading batch axis."""
    detections = detections.reshape(-1, detections.shape[-1])
    masks = masks.reshape(masks.shape[-3:])
    threshold = np.float64(options.mask_threshold)
    logit_threshold = np.log(threshold / (1.0 - threshold))
    best = np.argsort(-detections[:, 4], kind="stable")
    sx, sy = options.width / MODEL_SIZE, options.height / MODEL_SIZE
    segments, kept_boxes = [], []
    for n, query in enumerate((q for q in best if detections[q, 4] > options.min_score), start=1):
        # Duplicates and empty masks are skipped, so the limit counts the instances sent.
        if len(segments) == options.max_detections:
            break
        x1, y1, x2, y2, score = (float(v) for v in detections[query, :5])
        x0 = max(0, min(options.width - 1, math.floor(x1 * sx)))
        y0 = max(0, min(options.height - 1, math.floor(y1 * sy)))
        box = (
            x0,
            y0,
            max(x0 + 1, min(options.width, math.ceil(x2 * sx))),
            max(y0 + 1, min(options.height, math.ceil(y2 * sy))),
        )
        if any(box_iou(box, kept) > DUPLICATE_IOU for kept in kept_boxes):
            continue
        outline = mask_outline(
            masks[:, :, query], box, options.width, options.height, logit_threshold
        )
        if outline:
            kept_boxes.append(box)
            segments.append(
                {
                    "id": f"seg_{n}",
                    "label": options.label,
                    "confidence": round(score, 4),
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "mask_format": "polygon",
                    "mask": outline,
                }
            )
    return segments


def segments_json(segments: list[dict]) -> str:
    """The "segments" metadata: the best-scoring segments that fit the metadata byte budget."""
    kept, size = [], len('{"segments":[]}')
    for segment in segments:
        size += len(json.dumps(segment, separators=(",", ":"))) + 1
        if size > METADATA_BYTE_BUDGET:
            break
        kept.append(segment)
    return json.dumps({"segments": kept}, separators=(",", ":"))


class OverlayClock:
    """Pairs every video frame with the result of the segmented frame nearest to it in time.

    Insight draws metadata only on the frame with its timestamp, and the model segments only some
    of the frames, so every frame is sent the nearest finished result.
    """

    def __init__(self) -> None:
        self.waiting: deque[tuple[int, str]] = deque()
        self.previous: tuple[int, str] | None = None

    def add_frame(self, pts_ms: int, frame_id: str) -> None:
        self.waiting.append((pts_ms, frame_id))

    def add_result(self, pts_ms: int, data: str) -> list[tuple[str, int, str]]:
        """The (data, timestamp_ms, frame_id) messages of the frames up to this result."""
        messages = []
        while self.waiting and self.waiting[0][0] <= pts_ms:
            frame_pts, frame_id = self.waiting.popleft()
            closer_to_previous = (
                self.previous is not None and frame_pts - self.previous[0] < pts_ms - frame_pts
            )
            messages.append((self.previous[1] if closer_to_previous else data, frame_pts, frame_id))
        self.previous = (pts_ms, data)
        return messages
