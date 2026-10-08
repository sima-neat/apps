"""Compact two-stage, motion-aware tracking for small object detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BBox = tuple[float, float, float, float]


def _width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def _height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def _center(box: BBox) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _box_sizes(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0]),
        np.maximum(0.0, boxes[:, 3] - boxes[:, 1]),
    )


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every row of `a` with every row of `b`."""
    width_a, height_a = _box_sizes(a)
    width_b, height_b = _box_sizes(b)
    overlap_w = np.maximum(
        0.0,
        np.minimum(a[:, None, 2], b[None, :, 2])
        - np.maximum(a[:, None, 0], b[None, :, 0]),
    )
    overlap_h = np.maximum(
        0.0,
        np.minimum(a[:, None, 3], b[None, :, 3])
        - np.maximum(a[:, None, 1], b[None, :, 1]),
    )
    intersection = overlap_w * overlap_h
    union_area = (
        (width_a * height_a)[:, None] + (width_b * height_b)[None, :] - intersection
    )
    return np.divide(
        intersection, union_area, out=np.zeros_like(intersection), where=union_area > 0
    )


def _center_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Center distance of every row of `a` to every row of `b`, scaled by box diagonals."""
    width_a, height_a = _box_sizes(a)
    width_b, height_b = _box_sizes(b)
    dx = 0.5 * (a[:, None, 0] + a[:, None, 2]) - 0.5 * (b[None, :, 0] + b[None, :, 2])
    dy = 0.5 * (a[:, None, 1] + a[:, None, 3]) - 0.5 * (b[None, :, 1] + b[None, :, 3])
    scale = np.maximum(
        1.0,
        0.5
        * (np.hypot(width_a, height_a)[:, None] + np.hypot(width_b, height_b)[None, :]),
    )
    return np.hypot(dx, dy) / scale


def _bbox(detection: dict) -> BBox:
    return (
        float(detection["x1"]),
        float(detection["y1"]),
        float(detection["x2"]),
        float(detection["y2"]),
    )


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class TrackerConfig:
    high_score_threshold: float = 0.30
    new_track_threshold: float = 0.30
    match_iou_threshold: float = 0.30
    max_center_distance: float = 2.5
    velocity_momentum: float = 0.80
    max_missing_frames: int = 15
    min_confirmed_hits: int = 1
    max_active_tracks: int = 256
    center_distance_enabled: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.high_score_threshold) or not (
            0.0 <= self.high_score_threshold <= 1.0
        ):
            raise ValueError("high_score_threshold must be in [0, 1]")
        if not math.isfinite(self.new_track_threshold) or not (
            self.high_score_threshold <= self.new_track_threshold <= 1.0
        ):
            raise ValueError("new_track_threshold must be in [high_score_threshold, 1]")
        if not math.isfinite(self.match_iou_threshold) or not (
            0.0 <= self.match_iou_threshold <= 1.0
        ):
            raise ValueError("match_iou_threshold must be in [0, 1]")
        if (
            not math.isfinite(self.max_center_distance)
            or self.max_center_distance < 0.0
        ):
            raise ValueError("max_center_distance must be >= 0")
        if not math.isfinite(self.velocity_momentum) or not (
            0.0 <= self.velocity_momentum < 1.0
        ):
            raise ValueError("velocity_momentum must be in [0, 1)")
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be >= 0")
        if self.min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be >= 1")
        if self.max_active_tracks < 1:
            raise ValueError("max_active_tracks must be >= 1")


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    last_frame_index: int
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    missing_frames: int = 0
    hits: int = 1

    def predict(self, frame_index: int) -> BBox:
        elapsed = max(0, frame_index - self.last_frame_index)
        center_x, center_y = _center(self.bbox)
        vx, vy, vw, vh = self.velocity
        center_x += vx * elapsed
        center_y += vy * elapsed
        width = max(1.0, _width(self.bbox) + vw * elapsed)
        height = max(1.0, _height(self.bbox) + vh * elapsed)
        return (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )


class ObjectTracker:
    """Associate tiny boxes by score, IoU, and optional constant-velocity motion."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.config.validate()
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}
        self._last_frame_index = -1

    def active_track_count(self) -> int:
        return len(self._tracks)

    def _associate(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        detection_indices: list[int],
        frame_index: int,
        matched_tracks: set[int],
        matched_detections: set[int],
        assignments: dict[int, int],
        *,
        confirmed_only: bool,
    ) -> None:
        track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in matched_tracks
            and not (confirmed_only and track.hits < self.config.min_confirmed_hits)
        ]
        detection_ids = [
            index for index in detection_indices if index not in matched_detections
        ]
        if not track_ids or not detection_ids:
            return

        # Score every track/detection pair at once; a per-pair Python loop costs
        # tens of milliseconds per frame with the detector's 100 boxes.
        tracks = [self._tracks[track_id] for track_id in track_ids]
        predicted = np.array(
            [track.predict(frame_index) for track in tracks], dtype=np.float64
        )
        candidates = boxes[detection_ids]
        iou = _iou_matrix(predicted, candidates)
        eligible = (
            np.array([track.class_id for track in tracks])[:, None]
            == class_ids[detection_ids][None, :]
        )
        if self.config.center_distance_enabled:
            center_distance = _center_distance_matrix(predicted, candidates)
            eligible &= (iou >= self.config.match_iou_threshold) | (
                center_distance <= self.config.max_center_distance
            )
            affinity = iou + 1.0 / (1.0 + center_distance)
        else:
            eligible &= iou >= self.config.match_iou_threshold
            affinity = iou

        rows, cols = np.nonzero(eligible)
        pair_tracks = np.asarray(track_ids)[rows]
        pair_detections = np.asarray(detection_ids)[cols]
        # Greedy assignment in (-affinity, track_id, detection_index) order.
        order = np.lexsort((pair_detections, pair_tracks, -affinity[rows, cols]))
        for track_id, detection_index in zip(
            pair_tracks[order].tolist(), pair_detections[order].tolist()
        ):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

    def update(
        self, detections: list[dict], frame_index: int
    ) -> list[TrackedDetection]:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if frame_index < self._last_frame_index:
            raise ValueError("frame_index must be monotonic")
        self._last_frame_index = frame_index

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if max(0, frame_index - track.last_frame_index - 1)
            <= self.config.max_missing_frames
        }

        high = [
            index
            for index, detection in enumerate(detections)
            if float(detection["score"]) >= self.config.high_score_threshold
        ]
        low = [index for index in range(len(detections)) if index not in high]
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, int] = {}
        boxes = np.array(
            [_bbox(detection) for detection in detections], dtype=np.float64
        )
        boxes = boxes.reshape(-1, 4)
        class_ids = np.array(
            [int(detection["class_id"]) for detection in detections], dtype=np.int64
        )

        self._associate(
            boxes,
            class_ids,
            high,
            frame_index,
            matched_tracks,
            matched_detections,
            assignments,
            confirmed_only=False,
        )
        # A low-score observation may recover a confirmed identity, but it can
        # neither confirm a tentative identity nor create a new one.
        self._associate(
            boxes,
            class_ids,
            low,
            frame_index,
            matched_tracks,
            matched_detections,
            assignments,
            confirmed_only=True,
        )

        for detection_index, track_id in assignments.items():
            detection = detections[detection_index]
            bbox = _bbox(detection)
            track = self._tracks[track_id]
            elapsed = max(1, frame_index - track.last_frame_index)
            old_x, old_y = _center(track.bbox)
            new_x, new_y = _center(bbox)
            measured = (
                (new_x - old_x) / elapsed,
                (new_y - old_y) / elapsed,
                (_width(bbox) - _width(track.bbox)) / elapsed,
                (_height(bbox) - _height(track.bbox)) / elapsed,
            )
            momentum = self.config.velocity_momentum
            track.velocity = tuple(
                momentum * previous + (1.0 - momentum) * current
                for previous, current in zip(track.velocity, measured)
            )
            track.bbox = bbox
            track.score = float(detection["score"])
            track.last_frame_index = frame_index
            track.missing_frames = 0
            track.hits += 1

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missing_frames = frame_index - track.last_frame_index
            if track.missing_frames > self.config.max_missing_frames:
                del self._tracks[track_id]

        for detection_index in high:
            if detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            if float(detection["score"]) < self.config.new_track_threshold:
                continue
            if len(self._tracks) >= self.config.max_active_tracks:
                break
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=_bbox(detection),
                score=float(detection["score"]),
                class_id=int(detection["class_id"]),
                last_frame_index=frame_index,
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        tracked: list[TrackedDetection] = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None:
                continue
            track = self._tracks[track_id]
            if track.hits < self.config.min_confirmed_hits:
                continue
            bbox = _bbox(detection)
            tracked.append(
                TrackedDetection(
                    track_id=track_id,
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[2],
                    y2=bbox[3],
                    score=float(detection["score"]),
                    class_id=int(detection["class_id"]),
                )
            )
        return tracked
