"""The yolov5s-face decode: faces with five landmarks from the model's six raw split heads, and
the Insight pose message that carries them. Mirrors src/cpp/utils/yolov5_face.h.

The model emits, at three pyramid levels, a box head of 18 channels (3 anchors x box, object
score, class score) and a landmark head of 30 channels (3 anchors x 5 landmarks x 2). Neat's
BBOX format has no landmark slots, so the decode runs here instead of in a Neat box decode.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np

# The model was compiled for an 800x800 canvas; its levels are 100, 50, and 25 cells wide.
INFER_SIZE = 800
NUM_ANCHORS = 3
NUM_LANDMARKS = 5
BOX_CHANNELS = 18
LANDMARK_CHANNELS = 30
# Fixed by the model: the stride and the three anchor sizes of each level, largest grid first.
STRIDES = (8, 16, 32)
ANCHORS = np.array(
    [
        [[4, 5], [8, 10], [13, 16]],
        [[23, 29], [43, 55], [73, 105]],
        [[146, 217], [231, 300], [335, 433]],
    ],
    dtype=np.float32,
)
# Landmark names in model output order. Insight joins keypoints whose names are COCO body
# joints, such as nose or left_eye; these names are not, so no lines are drawn across the face.
LANDMARK_NAMES = ("eye_l", "eye_r", "nose_tip", "mouth_l", "mouth_r")


class Letterbox(NamedTuple):
    """The scale and padding that fit a frame into the model canvas."""

    scale: float
    pad_left: int
    pad_top: int


class Faces(NamedTuple):
    """N faces: boxes (N, 4) as x1, y1, x2, y2, scores (N,), and landmarks (N, 5, 2)."""

    boxes: np.ndarray
    scores: np.ndarray
    landmarks: np.ndarray


def letterbox(frame_w: int, frame_h: int) -> Letterbox:
    """The centered letterbox the model's preprocessing applies to a frame of this size."""
    scale = min(INFER_SIZE / frame_w, INFER_SIZE / frame_h)
    pad_left = (INFER_SIZE - round(frame_w * scale)) // 2
    pad_top = (INFER_SIZE - round(frame_h * scale)) // 2
    return Letterbox(scale, pad_left, pad_top)


def channels_last(head: np.ndarray) -> np.ndarray:
    """A split head as (H, W, C). The channel axis is found by its size, 18 or 30, which no grid
    size equals: [1, H, W, C], Neat's raw [1, H, C, W], or ONNX's [1, C, H, W]."""
    if head.ndim != 4 or head.shape[0] != 1:
        raise ValueError(f"expected a [1, ...] split head, got shape {head.shape}")
    channels = (BOX_CHANNELS, LANDMARK_CHANNELS)
    if head.shape[3] in channels:
        return head[0]
    if head.shape[2] in channels:
        return head[0].transpose(0, 2, 1)
    if head.shape[1] in channels:
        return head[0].transpose(1, 2, 0)
    raise ValueError(
        f"split head shape {head.shape} has no axis of {BOX_CHANNELS} or "
        f"{LANDMARK_CHANNELS} channels"
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def decode_level(
    box_head: np.ndarray, landmark_head: np.ndarray, level: int, min_score: float
) -> Faces:
    """The faces of one pyramid level whose score is above `min_score`, in canvas pixels."""
    rows, cols, _ = box_head.shape
    box = box_head.reshape(rows, cols, NUM_ANCHORS, 6).astype(np.float32)
    landmark = landmark_head.reshape(rows, cols, NUM_ANCHORS, 2 * NUM_LANDMARKS)
    scores = sigmoid(box[..., 4]) * sigmoid(box[..., 5])
    y, x, anchor = np.nonzero(scores > min_score)
    stride = STRIDES[level]
    anchor_w = ANCHORS[level, anchor, 0]
    anchor_h = ANCHORS[level, anchor, 1]
    xywh = sigmoid(box[y, x, anchor, :4])
    center_x = (xywh[:, 0] * 2.0 - 0.5 + x) * stride
    center_y = (xywh[:, 1] * 2.0 - 0.5 + y) * stride
    half_w = (xywh[:, 2] * 2.0) ** 2 * anchor_w / 2.0
    half_h = (xywh[:, 3] * 2.0) ** 2 * anchor_h / 2.0
    boxes = np.stack(
        [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h], axis=1
    )
    points = landmark[y, x, anchor].astype(np.float32)
    landmarks = np.empty((len(y), NUM_LANDMARKS, 2), dtype=np.float32)
    landmarks[:, :, 0] = points[:, 0::2] * anchor_w[:, None] + (x * stride)[:, None]
    landmarks[:, :, 1] = points[:, 1::2] * anchor_h[:, None] + (y * stride)[:, None]
    return Faces(boxes.astype(np.float32), scores[y, x, anchor].astype(np.float32), landmarks)


def keep_after_nms(faces: Faces, nms_iou: float, max_detections: int) -> Faces:
    """Greedy non-maximum suppression, highest score first, stopping at `max_detections`."""
    order = np.argsort(-faces.scores, kind="stable")
    x1, y1, x2, y2 = faces.boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size > 0 and len(keep) < max_detections:
        best, rest = order[0], order[1:]
        keep.append(best)
        overlap_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        overlap_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        overlap = overlap_w * overlap_h
        union = areas[best] + areas[rest] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = rest[iou <= nms_iou]
    return Faces(faces.boxes[keep], faces.scores[keep], faces.landmarks[keep])


def decode_faces(
    heads: list[np.ndarray], min_score: float, nms_iou: float, max_detections: int
) -> Faces:
    """The faces in the model's six split heads, in model canvas pixels, after NMS."""
    if len(heads) != 6:
        raise ValueError(f"expected 6 split heads, got {len(heads)}")
    by_grid: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for head in heads:
        hwc = channels_last(np.asarray(head))
        by_grid.setdefault(hwc.shape[:2], {})[hwc.shape[2]] = hwc
    grids = sorted(by_grid, reverse=True)
    if len(grids) != 3 or any(len(by_grid[grid]) != 2 for grid in grids):
        raise ValueError(f"expected a box and a landmark head on 3 grids, got {grids}")
    levels = [
        decode_level(
            by_grid[grid][BOX_CHANNELS], by_grid[grid][LANDMARK_CHANNELS], level, min_score
        )
        for level, grid in enumerate(grids)
    ]
    candidates = Faces(*(np.concatenate(field) for field in zip(*levels)))
    return keep_after_nms(candidates, nms_iou, max_detections)


def to_frame(faces: Faces, box: Letterbox, frame_w: int, frame_h: int) -> Faces:
    """Canvas pixels to frame pixels. Boxes are clipped to the frame, landmarks are not."""
    boxes = faces.boxes.copy()
    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - box.pad_left) / box.scale, 0, frame_w)
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - box.pad_top) / box.scale, 0, frame_h)
    landmarks = faces.landmarks.copy()
    landmarks[..., 0] = (landmarks[..., 0] - box.pad_left) / box.scale
    landmarks[..., 1] = (landmarks[..., 1] - box.pad_top) / box.scale
    return Faces(boxes, faces.scores, landmarks)


def poses_json(faces: Faces, label: str) -> str:
    """The `data` of an Insight pose-estimation message: one pose of five keypoints per face,
    each keypoint carrying the face's score."""
    poses = [
        {
            "id": f"face_{index}",
            "label": label,
            "keypoints": [
                {"name": name, "x": float(x), "y": float(y), "confidence": float(score)}
                for name, (x, y) in zip(LANDMARK_NAMES, points)
            ],
        }
        for index, (points, score) in enumerate(zip(faces.landmarks, faces.scores), start=1)
    ]
    return json.dumps({"poses": poses}, separators=(",", ":"))
