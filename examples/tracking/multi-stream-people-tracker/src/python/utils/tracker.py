"""Deterministic two-stage tracking for small object detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]
CameraTransform = tuple[float, float, float, float, float, float]
_BLOCKED_COST = 1.0e6
_UNMATCHED_COST = 1.0e3


def _width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def _height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def _center(box: BBox) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _translate(box: BBox, motion: tuple[float, float]) -> BBox:
    x, y = motion
    return box[0] + x, box[1] + y, box[2] + x, box[3] + y


def _transform_bbox(box: BBox, transform: CameraTransform) -> BBox:
    a, b, tx, c, d, ty = transform
    points = (
        (a * box[0] + b * box[1] + tx, c * box[0] + d * box[1] + ty),
        (a * box[2] + b * box[1] + tx, c * box[2] + d * box[1] + ty),
        (a * box[0] + b * box[3] + tx, c * box[0] + d * box[3] + ty),
        (a * box[2] + b * box[3] + tx, c * box[2] + d * box[3] + ty),
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _valid_camera_transform(transform: CameraTransform | None) -> bool:
    if transform is None or not all(math.isfinite(value) for value in transform):
        return False
    a, b, _, c, d, _ = transform
    return abs(a * d - b * c) > 0.01


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


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
        0.5 * (math.hypot(_width(a), _height(a)) + math.hypot(_width(b), _height(b))),
    )
    return math.hypot(ax - bx, ay - by) / scale


def _solve_assignment(costs: list[list[float]]) -> list[int]:
    """Return the minimum-cost column for each row of a rows <= columns matrix."""
    rows = len(costs)
    if not rows:
        return []
    columns = len(costs[0])
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minv = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = math.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while column0:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
    result = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            result[p[column] - 1] = column - 1
    return result


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    predicted: bool = False


@dataclass(frozen=True)
class TrackerConfig:
    high_score_threshold: float = 0.30
    new_track_threshold: float = 0.30
    match_iou_threshold: float = 0.10
    max_center_distance: float = 2.5
    velocity_momentum: float = 0.80
    box_smoothing_alpha: float = 1.0
    max_missing_frames: int = 15
    min_confirmed_hits: int = 1
    max_prediction_frames: int = 0
    center_distance_enabled: bool = True
    camera_motion_compensation: bool = False

    def validate(self) -> None:
        if not 0.0 <= self.high_score_threshold <= 1.0:
            raise ValueError("high_score_threshold must be in [0, 1]")
        if not self.high_score_threshold <= self.new_track_threshold <= 1.0:
            raise ValueError("new_track_threshold must be in [high_score_threshold, 1]")
        if not 0.0 <= self.match_iou_threshold <= 1.0:
            raise ValueError("match_iou_threshold must be in [0, 1]")
        if (
            not math.isfinite(self.max_center_distance)
            or self.max_center_distance < 0.0
        ):
            raise ValueError("max_center_distance must be >= 0")
        if not 0.0 <= self.velocity_momentum < 1.0:
            raise ValueError("velocity_momentum must be in [0, 1)")
        if not 0.0 < self.box_smoothing_alpha <= 1.0:
            raise ValueError("box_smoothing_alpha must be in (0, 1]")
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be >= 0")
        if self.min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be >= 1")
        if not 0 <= self.max_prediction_frames <= self.max_missing_frames:
            raise ValueError("max_prediction_frames must be in [0, max_missing_frames]")


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    last_frame_index: int
    filtered_bbox: BBox | None = None
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    missing_frames: int = 0
    hits: int = 1
    confirmed: bool = True

    def predict(self, frame_index: int) -> BBox:
        elapsed = max(0, frame_index - self.last_frame_index)
        reference = self.filtered_bbox or self.bbox
        center_x, center_y = _center(reference)
        vx, vy, vw, vh = self.velocity
        center_x += vx * elapsed
        center_y += vy * elapsed
        width = max(1.0, _width(reference) + vw * elapsed)
        height = max(1.0, _height(reference) + vh * elapsed)
        return (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )


class ObjectTracker:
    """Track tiny boxes with global two-score association and motion prediction."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.config.validate()
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}
        self._last_frame_index = -1

    def active_track_count(self) -> int:
        return len(self._tracks)

    def _estimate_camera_motion(
        self, detections: list[dict], high: list[int], frame_index: int
    ) -> tuple[tuple[float, float], bool]:
        recent = [
            track
            for track in self._tracks.values()
            if track.last_frame_index == frame_index - 1
        ]
        if len(recent) < 3 or len(high) < 3:
            return (0.0, 0.0), False

        predictions = [track.predict(frame_index) for track in recent]
        boxes = [_bbox(detections[index]) for index in high]
        diagonals = [
            math.hypot(_width(box), _height(box)) for box in predictions + boxes
        ]
        typical_diagonal = max(1.0, _median(diagonals))
        maximum_shift = max(16.0, 4.0 * typical_diagonal)
        bin_size = max(2.0, 0.25 * typical_diagonal)

        votes: list[tuple[float, float, int, int]] = []
        histogram: dict[tuple[int, int], int] = {}
        for track, prediction in zip(recent, predictions):
            predicted_center = _center(prediction)
            for detection_index in high:
                detection = detections[detection_index]
                if int(detection["class_id"]) != track.class_id:
                    continue
                observed_center = _center(_bbox(detection))
                x = observed_center[0] - predicted_center[0]
                y = observed_center[1] - predicted_center[1]
                if math.hypot(x, y) > maximum_shift:
                    continue
                bin_xy = math.floor(x / bin_size), math.floor(y / bin_size)
                votes.append((x, y, *bin_xy))
                histogram[bin_xy] = histogram.get(bin_xy, 0) + 1
        if len(votes) < 3:
            return (0.0, 0.0), False

        def neighborhood_support(bin_xy: tuple[int, int]) -> int:
            return sum(
                histogram.get((bin_xy[0] + dx, bin_xy[1] + dy), 0)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            )

        best_bin = min(
            histogram,
            key=lambda bin_xy: (
                -neighborhood_support(bin_xy),
                bin_xy[0] ** 2 + bin_xy[1] ** 2,
                bin_xy,
            ),
        )
        consensus = [
            (x, y)
            for x, y, bin_x, bin_y in votes
            if abs(bin_x - best_bin[0]) <= 1 and abs(bin_y - best_bin[1]) <= 1
        ]
        initial = (
            _median([motion[0] for motion in consensus]),
            _median([motion[1] for motion in consensus]),
        )

        candidate_gate = max(0.75, self.config.max_center_distance)
        pairs: list[tuple[float, int, int]] = []
        for track, prediction in zip(recent, predictions):
            reference = _translate(prediction, initial)
            for detection_index in high:
                detection = detections[detection_index]
                if int(detection["class_id"]) != track.class_id:
                    continue
                distance = _normalized_center_distance(
                    reference, _bbox(detection)
                )
                if distance <= candidate_gate:
                    pairs.append((distance, track.track_id, detection_index))

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        offsets: list[tuple[float, float]] = []
        for _, track_id, detection_index in sorted(pairs):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            prediction = self._tracks[track_id].predict(frame_index)
            observed = _bbox(detections[detection_index])
            predicted_center = _center(prediction)
            observed_center = _center(observed)
            offsets.append(
                (
                    observed_center[0] - predicted_center[0],
                    observed_center[1] - predicted_center[1],
                )
            )
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        possible = min(len(recent), len(high))
        required = max(3, (possible + 2) // 3)
        if len(offsets) < required:
            return (0.0, 0.0), False
        return (
            _median([offset[0] for offset in offsets]),
            _median([offset[1] for offset in offsets]),
        ), False

    def _associate(
        self,
        detections: list[dict],
        detection_indices: list[int],
        frame_index: int,
        matched_tracks: set[int],
        matched_detections: set[int],
        assignments: dict[int, int],
    ) -> None:
        candidates = sorted(
            (
                track
                for track in self._tracks.values()
                if track.track_id not in matched_tracks
            ),
            key=lambda track: (-track.last_frame_index, track.track_id),
        )
        begin = 0
        while begin < len(candidates):
            end = begin + 1
            while (
                end < len(candidates)
                and candidates[end].last_frame_index
                == candidates[begin].last_frame_index
            ):
                end += 1
            self._associate_recency_group(
                detections,
                detection_indices,
                frame_index,
                [track.track_id for track in candidates[begin:end]],
                matched_tracks,
                matched_detections,
                assignments,
            )
            begin = end

    def _associate_recency_group(
        self,
        detections: list[dict],
        detection_indices: list[int],
        frame_index: int,
        track_ids: list[int],
        matched_tracks: set[int],
        matched_detections: set[int],
        assignments: dict[int, int],
    ) -> None:
        available_detections = [
            index for index in detection_indices if index not in matched_detections
        ]
        if not track_ids or not available_detections:
            return
        costs: list[list[float]] = []
        for track_id in track_ids:
            track = self._tracks[track_id]
            reference = (
                track.predict(frame_index)
                if self.config.center_distance_enabled
                else track.bbox
            )
            row_costs: list[float] = []
            for detection_index in available_detections:
                detection = detections[detection_index]
                if int(detection["class_id"]) != track.class_id:
                    row_costs.append(_BLOCKED_COST)
                    continue
                bbox = _bbox(detection)
                iou = _iou_xyxy(reference, bbox)
                center_distance = _normalized_center_distance(reference, bbox)
                center_match = (
                    self.config.center_distance_enabled
                    and center_distance <= self.config.max_center_distance
                )
                if iou < self.config.match_iou_threshold and not center_match:
                    row_costs.append(_BLOCKED_COST)
                    continue
                affinity = (
                    iou + 1.0 / (1.0 + center_distance)
                    if self.config.center_distance_enabled
                    else iou
                )
                row_costs.append(
                    2.0 - affinity
                    if self.config.center_distance_enabled
                    else 1.0 - affinity
                )
            row_costs.extend([_UNMATCHED_COST] * len(track_ids))
            costs.append(row_costs)
        for row, column in enumerate(_solve_assignment(costs)):
            if (
                column < 0
                or column >= len(available_detections)
                or costs[row][column] >= _BLOCKED_COST
            ):
                continue
            track_id = track_ids[row]
            detection_index = available_detections[column]
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

    def update(
        self,
        detections: list[dict],
        frame_index: int,
        camera_transform: CameraTransform | None = None,
    ) -> list[TrackedDetection]:
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if frame_index < self._last_frame_index:
            raise ValueError("frame_index must be monotonic")
        self._last_frame_index = frame_index

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if not (
                (not track.confirmed and frame_index - track.last_frame_index - 1 > 0)
                or frame_index - track.last_frame_index - 1
                > self.config.max_missing_frames
            )
        }
        high = [
            index
            for index, detection in enumerate(detections)
            if float(detection["score"]) >= self.config.high_score_threshold
        ]
        low = [index for index in range(len(detections)) if index not in high]
        applied_camera_transform: CameraTransform | None = None
        if self.config.camera_motion_compensation:
            if _valid_camera_transform(camera_transform):
                applied_camera_transform = camera_transform
            else:
                camera_motion, _ = self._estimate_camera_motion(
                    detections, high, frame_index
                )
                if camera_motion != (0.0, 0.0):
                    applied_camera_transform = (
                        1.0,
                        0.0,
                        camera_motion[0],
                        0.0,
                        1.0,
                        camera_motion[1],
                    )
        if applied_camera_transform is not None:
            a, b, _, c, d, _ = applied_camera_transform
            for track in self._tracks.values():
                bbox = track.bbox
                filtered_bbox = track.filtered_bbox or bbox
                track.bbox = _transform_bbox(bbox, applied_camera_transform)
                track.filtered_bbox = _transform_bbox(
                    filtered_bbox, applied_camera_transform
                )
                vx, vy, vw, vh = track.velocity
                track.velocity = (
                    a * vx + b * vy,
                    c * vx + d * vy,
                    vw * math.hypot(a, c),
                    vh * math.hypot(b, d),
                )
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
            prediction = track.predict(frame_index)
            track.velocity = tuple(
                momentum * previous + (1.0 - momentum) * current
                for previous, current in zip(track.velocity, measured)
            )
            alpha = self.config.box_smoothing_alpha
            if applied_camera_transform is not None and alpha < 1.0:
                innovation = _normalized_center_distance(prediction, bbox)
                response = min(
                    1.0, innovation / max(0.01, self.config.max_center_distance)
                )
                alpha += (1.0 - alpha) * response
            predicted_center = _center(prediction)
            observed_center = _center(bbox)
            filtered_center = tuple(
                (1.0 - alpha) * previous + alpha * current
                for previous, current in zip(predicted_center, observed_center)
            )
            filtered_width = max(
                1.0, (1.0 - alpha) * _width(prediction) + alpha * _width(bbox)
            )
            filtered_height = max(
                1.0, (1.0 - alpha) * _height(prediction) + alpha * _height(bbox)
            )
            track.filtered_bbox = (
                filtered_center[0] - filtered_width * 0.5,
                filtered_center[1] - filtered_height * 0.5,
                filtered_center[0] + filtered_width * 0.5,
                filtered_center[1] + filtered_height * 0.5,
            )
            track.bbox = bbox
            track.score = float(detection["score"])
            track.last_frame_index = frame_index
            track.missing_frames = 0
            track.hits += 1
            track.confirmed = track.hits >= self.config.min_confirmed_hits

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
                filtered_bbox=_bbox(detection),
                confirmed=self.config.min_confirmed_hits <= 1,
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.missing_frames = frame_index - track.last_frame_index

        tracked: list[TrackedDetection] = []
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None or not self._tracks[track_id].confirmed:
                continue
            bbox = self._tracks[track_id].filtered_bbox or _bbox(detection)
            tracked.append(
                TrackedDetection(
                    track_id,
                    *bbox,
                    float(detection["score"]),
                    int(detection["class_id"]),
                )
            )

        if self.config.max_prediction_frames:
            for track in self._tracks.values():
                if (
                    not track.confirmed
                    or not 0 < track.missing_frames <= self.config.max_prediction_frames
                    or track.score < self.config.high_score_threshold
                ):
                    continue
                bbox = track.predict(frame_index)
                if any(
                    output.class_id == track.class_id
                    and _iou_xyxy((output.x1, output.y1, output.x2, output.y2), bbox)
                    > 0.5
                    for output in tracked
                ):
                    continue
                tracked.append(
                    TrackedDetection(
                        track.track_id,
                        *bbox,
                        track.score * 0.9**track.missing_frames,
                        track.class_id,
                        True,
                    )
                )

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if not (
                (not track.confirmed and track.missing_frames > 0)
                or track.missing_frames > self.config.max_missing_frames
            )
        }
        return tracked


def _bbox(detection: dict) -> BBox:
    return (
        float(detection["x1"]),
        float(detection["y1"]),
        float(detection["x2"]),
        float(detection["y2"]),
    )


class FrameCameraMotionEstimator:
    """Downscaled ORB/RANSAC partial-affine camera-motion estimator."""

    def __init__(self, downscale: int = 2, max_features: int = 500) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self._downscale = max(1, int(downscale))
        self._orb = cv2.ORB_create(
            nfeatures=max(64, int(max_features)),
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            patchSize=15,
            fastThreshold=20,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._previous_keypoints = None
        self._previous_descriptors = None

    def reset(self) -> None:
        self._previous_keypoints = None
        self._previous_descriptors = None

    def update(self, gray_frame, detections: list[dict] | None = None) -> CameraTransform | None:
        cv2 = self._cv2
        np = self._np
        if gray_frame is None or gray_frame.ndim != 2 or gray_frame.dtype != np.uint8:
            self.reset()
            return None
        if self._downscale > 1:
            gray = cv2.resize(
                gray_frame,
                (
                    gray_frame.shape[1] // self._downscale,
                    gray_frame.shape[0] // self._downscale,
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            gray = gray_frame

        feature_mask = np.zeros(gray.shape, dtype=np.uint8)
        border_x = max(1, gray.shape[1] // 50)
        border_y = max(1, gray.shape[0] // 50)
        feature_mask[
            border_y : gray.shape[0] - border_y,
            border_x : gray.shape[1] - border_x,
        ] = 255
        for detection in detections or ():
            scale = float(self._downscale)
            x1 = max(0, math.floor(float(detection["x1"]) / scale) - 2)
            y1 = max(0, math.floor(float(detection["y1"]) / scale) - 2)
            x2 = min(gray.shape[1], math.ceil(float(detection["x2"]) / scale) + 2)
            y2 = min(gray.shape[0], math.ceil(float(detection["y2"]) / scale) + 2)
            if x2 > x1 and y2 > y1:
                feature_mask[y1:y2, x1:x2] = 0

        keypoints, descriptors = self._orb.detectAndCompute(gray, feature_mask)
        result = None
        if (
            self._previous_descriptors is not None
            and descriptors is not None
            and self._previous_keypoints is not None
            and len(self._previous_keypoints) >= 8
            and len(keypoints) >= 8
        ):
            pairs = self._matcher.knnMatch(
                self._previous_descriptors, descriptors, k=2
            )
            accepted = [
                pair[0]
                for pair in pairs
                if len(pair) >= 2 and pair[0].distance < 0.80 * pair[1].distance
            ]
            if len(accepted) >= 8:
                previous = np.float32(
                    [self._previous_keypoints[match.queryIdx].pt for match in accepted]
                )
                current = np.float32(
                    [keypoints[match.trainIdx].pt for match in accepted]
                )
                affine, inlier_mask = cv2.estimateAffinePartial2D(
                    previous,
                    current,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,
                    maxIters=500,
                    confidence=0.99,
                    refineIters=10,
                )
                inliers = (
                    int(np.count_nonzero(inlier_mask))
                    if inlier_mask is not None
                    else 0
                )
                if affine is not None and self._plausible(
                    affine, gray.shape[1], gray.shape[0], inliers, len(accepted)
                ):
                    scale = float(self._downscale)
                    result = (
                        float(affine[0, 0]),
                        float(affine[0, 1]),
                        scale * float(affine[0, 2]),
                        float(affine[1, 0]),
                        float(affine[1, 1]),
                        scale * float(affine[1, 2]),
                    )

        self._previous_keypoints = keypoints
        self._previous_descriptors = (
            None if descriptors is None else descriptors.copy()
        )
        return result

    @staticmethod
    def _plausible(affine, width: int, height: int, inliers: int, matches: int) -> bool:
        if inliers < 8 or matches <= 0 or inliers / matches < 0.25:
            return False
        a, b, tx = (float(value) for value in affine[0])
        c, d, ty = (float(value) for value in affine[1])
        values = (a, b, tx, c, d, ty)
        if not all(math.isfinite(value) for value in values):
            return False
        scale_x = math.hypot(a, c)
        scale_y = math.hypot(b, d)
        rotation = abs(math.atan2(c, a))
        return (
            0.85 <= scale_x <= 1.15
            and 0.85 <= scale_y <= 1.15
            and rotation <= 0.20
            and abs(tx) <= 0.5 * width
            and abs(ty) <= 0.5 * height
        )
