"""Two-stage, motion-aware tracking for small object detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]


def _width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def _height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def _center(box: BBox) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _iou_xyxy(a: BBox, b: BBox) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    union_area = _width(a) * _height(a) + _width(b) * _height(b) - intersection
    return intersection / union_area if union_area > 0 else 0.0


def _normalized_center_distance(a: BBox, b: BBox) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    scale = max(
        1.0,
        0.5
        * (
            math.hypot(_width(a), _height(a))
            + math.hypot(_width(b), _height(b))
        ),
    )
    return math.hypot(ax - bx, ay - by) / scale


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
    match_iou_threshold: float = 0.10
    max_center_distance: float = 2.5
    velocity_momentum: float = 0.80
    max_missing_frames: int = 15
    min_confirmed_hits: int = 1

    def validate(self) -> None:
        if not 0.0 <= self.high_score_threshold <= 1.0:
            raise ValueError("high_score_threshold must be in [0, 1]")
        if not self.high_score_threshold <= self.new_track_threshold <= 1.0:
            raise ValueError("new_track_threshold must be in [high_score_threshold, 1]")
        if not 0.0 <= self.match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must be in [0, 1]")
        if not math.isfinite(self.max_center_distance) or self.max_center_distance < 0.0:
            raise ValueError("max_center_distance must be >= 0")
        if not 0.0 <= self.velocity_momentum < 1.0:
            raise ValueError("velocity_momentum must be in [0, 1)")
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be >= 0")
        if self.min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be >= 1")


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
    """Track tiny boxes with two-score association and constant-velocity prediction."""

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
        detections: list[dict],
        detection_indices: list[int],
        frame_index: int,
        matched_tracks: set[int],
        matched_detections: set[int],
        assignments: dict[int, int],
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_tracks:
                continue
            predicted = track.predict(frame_index)
            for detection_index in detection_indices:
                if detection_index in matched_detections:
                    continue
                detection = detections[detection_index]
                if int(detection["class_id"]) != track.class_id:
                    continue
                bbox = _bbox(detection)
                iou = _iou_xyxy(predicted, bbox)
                center_distance = _normalized_center_distance(predicted, bbox)
                if (
                    iou < self.config.match_iou_threshold
                    and center_distance > self.config.max_center_distance
                ):
                    continue
                affinity = iou + 1.0 / (1.0 + center_distance)
                candidates.append((affinity, track_id, detection_index))

        candidates.sort(reverse=True)
        for _affinity, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

    def update(self, detections: list[dict], frame_index: int) -> list[TrackedDetection]:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if frame_index < self._last_frame_index:
            raise ValueError("frame_index must be monotonic")
        self._last_frame_index = frame_index

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index <= self.config.max_missing_frames
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

        self._associate(
            detections,
            high,
            frame_index,
            matched_tracks,
            matched_detections,
            assignments,
        )
        self._associate(
            detections,
            low,
            frame_index,
            matched_tracks,
            matched_detections,
            assignments,
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

        for detection_index in high:
            if detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            if float(detection["score"]) < self.config.new_track_threshold:
                continue
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

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missing_frames = frame_index - track.last_frame_index
            if track.missing_frames > self.config.max_missing_frames:
                del self._tracks[track_id]

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


def _bbox(detection: dict) -> BBox:
    return (
        float(detection["x1"]),
        float(detection["y1"]),
        float(detection["x2"]),
        float(detection["y2"]),
    )
