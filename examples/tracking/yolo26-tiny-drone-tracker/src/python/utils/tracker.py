"""Two-stage, motion-aware tracking of small detections, one tracker per stream.

Mirrors src/cpp/utils/tracker.h, so both languages give the same track IDs to the same boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Center distance gate, in mean box diagonals. Tiny boxes rarely overlap, so it does all the
# matching, and IoU only ranks the pairs that pass it.
MAX_CENTER_DISTANCE = 3.0
# Weight of the previous velocity when a match updates it; the rest is the measured motion.
VELOCITY_MOMENTUM = 0.75
# New tracks stop at this many active tracks per stream.
MAX_ACTIVE_TRACKS = 256

BBox = tuple[float, float, float, float]


class Box(Protocol):
    """A detection in frame pixels, such as `pyneat.detections.Box`."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class TrackerSettings:
    high_score_threshold: float  # Score that lets a detection continue any track.
    new_track_threshold: float  # Score an unmatched detection needs to start a track.
    max_missing_frames: int  # Frames a track survives without a match.
    min_confirmed_hits: int  # Detections, counting the first, before a track is reported.


@dataclass(frozen=True)
class TrackedDetection:
    """A detection of this frame and the ID of the track it belongs to."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass
class _Track:
    bbox: BBox
    last_frame_index: int
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    hits: int = 1

    def predict(self, frame_index: int) -> BBox:
        """The box moved by its velocity to `frame_index`, at least 1 pixel on each side."""
        elapsed = frame_index - self.last_frame_index
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
    """Give each detection of one stream the ID of the track it continues, frame by frame."""

    def __init__(self, settings: TrackerSettings, fps: int) -> None:
        self.settings = settings
        self.fps = fps
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}
        # The timestamp of the frame that numbers the ones after it, and that frame's number.
        self._origin_pts_ns: int | None = None
        self._origin_frame_index = 0
        self._frame_index = -1

    def tracks_started(self) -> int:
        return self._next_track_id - 1

    def update(self, detections: list[Box], pts_ns: int) -> list[TrackedDetection]:
        """Match the detections of the frame at `pts_ns` to the tracks, and return those of
        confirmed tracks in detection order."""
        frame_index = self._frame_index_at(pts_ns)
        settings = self.settings
        # Frames the shared model's link dropped still age the tracks, so a track unmatched for
        # more than max_missing_frames ends before it can match again.
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index - 1 <= settings.max_missing_frames
        }
        boxes = np.array([_bbox(detection) for detection in detections], dtype=np.float64)
        boxes = boxes.reshape(-1, 4)
        high = []
        low = []
        for index, detection in enumerate(detections):
            (high if detection.score >= settings.high_score_threshold else low).append(index)

        assignments: dict[int, int] = {}
        self._associate(boxes, high, frame_index, assignments, confirmed_only=False)
        self._associate(boxes, low, frame_index, assignments, confirmed_only=True)

        for detection_index, track_id in assignments.items():
            self._continue(self._tracks[track_id], boxes[detection_index], frame_index)

        matched_tracks = set(assignments.values())
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track_id in matched_tracks
            or frame_index - track.last_frame_index <= settings.max_missing_frames
        }

        for detection_index in high:
            if detection_index in assignments:
                continue
            if detections[detection_index].score < settings.new_track_threshold:
                continue
            if len(self._tracks) >= MAX_ACTIVE_TRACKS:
                break
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _Track(tuple(boxes[detection_index]), frame_index)
            assignments[detection_index] = track_id

        tracked = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None or self._tracks[track_id].hits < settings.min_confirmed_hits:
                continue
            tracked.append(
                TrackedDetection(
                    track_id,
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    detection.score,
                    detection.class_id,
                )
            )
        return tracked

    def _frame_index_at(self, pts_ns: int) -> int:
        """The stream's frame number at `pts_ns`, counted at `fps`, since the link drops frames.

        No timestamp, or one before the last frame's after a source restart, is the next frame.
        """
        if pts_ns < 0:
            self._frame_index += 1
            return self._frame_index
        if self._origin_pts_ns is not None:
            frames = self._origin_frame_index + math.floor(
                (pts_ns - self._origin_pts_ns) * self.fps / 1e9 + 0.5
            )
            if frames >= self._frame_index:
                self._frame_index = frames
                return self._frame_index
        self._frame_index += 1
        self._origin_pts_ns = pts_ns
        self._origin_frame_index = self._frame_index
        return self._frame_index

    def _associate(
        self,
        boxes: np.ndarray,
        detection_indices: list[int],
        frame_index: int,
        assignments: dict[int, int],
        *,
        confirmed_only: bool,
    ) -> None:
        """Assign free detections to free tracks, best affinity first."""
        matched_tracks = set(assignments.values())
        track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track_id not in matched_tracks
            and not (confirmed_only and track.hits < self.settings.min_confirmed_hits)
        ]
        candidates = [index for index in detection_indices if index not in assignments]
        if not track_ids or not candidates:
            return

        # Score every pair at once, since a Python loop per pair is too slow for 100 detections.
        predicted = np.array(
            [self._tracks[track_id].predict(frame_index) for track_id in track_ids],
            dtype=np.float64,
        )
        iou = _iou_matrix(predicted, boxes[candidates])
        center_distance = _center_distance_matrix(predicted, boxes[candidates])
        eligible = center_distance <= MAX_CENTER_DISTANCE
        affinity = iou + 1.0 / (1.0 + center_distance)

        rows, cols = np.nonzero(eligible)
        pair_tracks = np.asarray(track_ids)[rows]
        pair_detections = np.asarray(candidates)[cols]
        # Highest affinity first; ties go to the lower track ID, then the lower detection index.
        order = np.lexsort((pair_detections, pair_tracks, -affinity[rows, cols]))
        for track_id, detection_index in zip(
            pair_tracks[order].tolist(), pair_detections[order].tolist()
        ):
            if track_id in matched_tracks or detection_index in assignments:
                continue
            matched_tracks.add(track_id)
            assignments[detection_index] = track_id

    @staticmethod
    def _continue(track: _Track, bbox: np.ndarray, frame_index: int) -> None:
        """Move the track to its matched box and blend the measured motion into its velocity."""
        new_bbox: BBox = tuple(bbox.tolist())
        elapsed = max(1, frame_index - track.last_frame_index)
        old_x, old_y = _center(track.bbox)
        new_x, new_y = _center(new_bbox)
        measured = (
            (new_x - old_x) / elapsed,
            (new_y - old_y) / elapsed,
            (_width(new_bbox) - _width(track.bbox)) / elapsed,
            (_height(new_bbox) - _height(track.bbox)) / elapsed,
        )
        track.velocity = tuple(
            VELOCITY_MOMENTUM * previous + (1.0 - VELOCITY_MOMENTUM) * current
            for previous, current in zip(track.velocity, measured)
        )
        track.bbox = new_bbox
        track.last_frame_index = frame_index
        track.hits += 1


def _bbox(detection: Box) -> BBox:
    return (float(detection.x1), float(detection.y1), float(detection.x2), float(detection.y2))


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
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]),
    )
    overlap_h = np.maximum(
        0.0,
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]),
    )
    intersection = overlap_w * overlap_h
    union_area = (width_a * height_a)[:, None] + (width_b * height_b)[None, :] - intersection
    return np.divide(
        intersection, union_area, out=np.zeros_like(intersection), where=union_area > 0
    )


def _center_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Center distance of every row of `a` to every row of `b`, in mean box diagonals."""
    width_a, height_a = _box_sizes(a)
    width_b, height_b = _box_sizes(b)
    dx = 0.5 * (a[:, None, 0] + a[:, None, 2]) - 0.5 * (b[None, :, 0] + b[None, :, 2])
    dy = 0.5 * (a[:, None, 1] + a[:, None, 3]) - 0.5 * (b[None, :, 1] + b[None, :, 3])
    scale = np.maximum(
        1.0,
        0.5 * (np.hypot(width_a, height_a)[:, None] + np.hypot(width_b, height_b)[None, :]),
    )
    return np.hypot(dx, dy) / scale
