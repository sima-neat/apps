"""Deterministic two-stage tracking for small object detections."""

from __future__ import annotations

import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]
BoxCorners = tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]
CameraTransform = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CameraMotionEstimate:
    """Affine camera motion together with the estimator's reliability signals."""

    transform: CameraTransform
    confidence: float = 1.0
    reprojection_error: float = 0.0
    inliers: int = 0

    def __iter__(self):
        return iter(self.transform)

    def __len__(self) -> int:
        return len(self.transform)

    def __getitem__(self, index: int) -> float:
        return self.transform[index]


CameraMotionInput = CameraTransform | CameraMotionEstimate
_BLOCKED_COST = 1.0e6
_UNMATCHED_COST = 1.0e3
_MAXIMUM_LOG_SIZE = math.log(16384.0)
_MAXIMUM_LOG_SIZE_VELOCITY = 0.35
_MAXIMUM_LOG_SIZE_INNOVATION_PER_FRAME = 0.70


def _width(box: BBox) -> float:
    return max(0.0, box[2] - box[0])


def _height(box: BBox) -> float:
    return max(0.0, box[3] - box[1])


def _center(box: BBox) -> tuple[float, float]:
    return 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])


def _log_size(value: float) -> float:
    return min(_MAXIMUM_LOG_SIZE, max(0.0, math.log(max(1.0, value))))


def _size_from_log(value: float) -> float:
    return math.exp(min(_MAXIMUM_LOG_SIZE, max(0.0, value)))


def _stabilize_size_axis(axis: "KalmanAxis") -> None:
    axis.position = min(_MAXIMUM_LOG_SIZE, max(0.0, axis.position))
    axis.velocity = min(
        _MAXIMUM_LOG_SIZE_VELOCITY,
        max(-_MAXIMUM_LOG_SIZE_VELOCITY, axis.velocity),
    )


def _translate(box: BBox, motion: tuple[float, float]) -> BBox:
    x, y = motion
    return box[0] + x, box[1] + y, box[2] + x, box[3] + y


def _box_corners(box: BBox) -> BoxCorners:
    return (
        (box[0], box[1]),
        (box[2], box[1]),
        (box[0], box[3]),
        (box[2], box[3]),
    )


def _transform_corners(corners: BoxCorners, transform: CameraTransform) -> BoxCorners:
    a, b, tx, c, d, ty = transform
    return tuple((a * x + b * y + tx, c * x + d * y + ty) for x, y in corners)


def _corners_bbox(corners: BoxCorners) -> BBox:
    return (
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )


def _recenter_corners(corners: BoxCorners, target: BBox) -> BoxCorners:
    source = _corners_bbox(corners)
    if (
        not all(math.isfinite(value) for value in source)
        or _width(source) <= 1.0e-6
        or _height(source) <= 1.0e-6
    ):
        return _box_corners(target)
    source_center_x, source_center_y = _center(source)
    target_center_x, target_center_y = _center(target)
    return tuple(
        (
            x + target_center_x - source_center_x,
            y + target_center_y - source_center_y,
        )
        for x, y in corners
    )


def _camera_transform_values(transform: CameraMotionInput) -> CameraTransform:
    return (
        transform.transform
        if isinstance(transform, CameraMotionEstimate)
        else transform
    )


def _valid_camera_transform(transform: CameraMotionInput | None) -> bool:
    if transform is None:
        return False
    values = _camera_transform_values(transform)
    if not all(math.isfinite(value) for value in values):
        return False
    a, b, _, c, d, _ = values
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


def _overlap_coefficient(a: BBox, b: BBox) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    smaller_area = min(_width(a) * _height(a), _width(b) * _height(b))
    return intersection / smaller_area if smaller_area > 0.0 else 0.0


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
    occluded: bool = False
    association_confidence: float = 1.0


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
    covariance_motion_enabled: bool = True
    overlap_threshold: float = 0.20
    max_occlusion_frames: int = 0
    max_active_tracks: int = 128

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
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if not 0 <= self.max_occlusion_frames <= self.max_missing_frames:
            raise ValueError("max_occlusion_frames must be in [0, max_missing_frames]")
        if self.max_active_tracks < 1:
            raise ValueError("max_active_tracks must be >= 1")


@dataclass
class KalmanAxis:
    position: float
    velocity: float = 0.0
    p00: float = 4.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 16.0

    def predict(
        self, elapsed: float, position_noise: float, velocity_noise: float
    ) -> None:
        if elapsed <= 0.0:
            return
        self.position += elapsed * self.velocity
        p00 = (
            self.p00
            + elapsed * (self.p01 + self.p10)
            + elapsed * elapsed * self.p11
            + position_noise * elapsed
        )
        p01 = self.p01 + elapsed * self.p11
        p10 = self.p10 + elapsed * self.p11
        p11 = self.p11 + velocity_noise * elapsed
        self.p00 = max(1.0e-4, p00)
        self.p01 = p01
        self.p10 = p10
        self.p11 = max(1.0e-4, p11)

    def update(self, measurement: float, measurement_variance: float) -> None:
        variance = max(1.0e-4, measurement_variance)
        innovation_variance = max(1.0e-4, self.p00 + variance)
        gain_position = self.p00 / innovation_variance
        gain_velocity = self.p10 / innovation_variance
        innovation = measurement - self.position
        self.position += gain_position * innovation
        self.velocity += gain_velocity * innovation
        one_minus_gain = 1.0 - gain_position
        p00 = one_minus_gain**2 * self.p00 + gain_position**2 * variance
        p01 = one_minus_gain * (self.p01 - gain_velocity * self.p00) + (
            gain_position * gain_velocity * variance
        )
        p10 = one_minus_gain * (self.p10 - gain_velocity * self.p00) + (
            gain_position * gain_velocity * variance
        )
        p11 = (
            self.p11
            - gain_velocity * self.p01
            - gain_velocity * self.p10
            + gain_velocity**2 * (self.p00 + variance)
        )
        self.p00 = min(1.0e6, max(1.0e-4, p00))
        self.p01 = min(1.0e6, max(-1.0e6, p01))
        self.p10 = min(1.0e6, max(-1.0e6, p10))
        self.p11 = min(1.0e6, max(1.0e-4, p11))


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    score: float
    class_id: int
    last_frame_index: int
    filtered_bbox: BBox | None = None
    display_bbox: BBox | None = None
    bbox_corners: BoxCorners | None = None
    filtered_bbox_corners: BoxCorners | None = None
    display_bbox_corners: BoxCorners | None = None
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    center_x_filter: KalmanAxis | None = None
    center_y_filter: KalmanAxis | None = None
    log_width_filter: KalmanAxis | None = None
    log_height_filter: KalmanAxis | None = None
    state_frame_index: int | None = None
    previous_observation: tuple[float, float] | None = None
    previous_observation_frame: int = -1
    missing_frames: int = 0
    hits: int = 1
    confirmed: bool = True
    occluded: bool = False
    association_confidence: float = 1.0
    covariance_motion_enabled: bool = True

    def __post_init__(self) -> None:
        diagonal = max(1.0, math.hypot(_width(self.bbox), _height(self.bbox)))
        position_variance = max(0.25, 0.01 * diagonal * diagonal)
        cx, cy = _center(self.filtered_bbox or self.bbox)
        self.center_x_filter = self.center_x_filter or KalmanAxis(
            cx, self.velocity[0], position_variance, 0.0, 0.0, 4.0 * position_variance
        )
        self.center_y_filter = self.center_y_filter or KalmanAxis(
            cy, self.velocity[1], position_variance, 0.0, 0.0, 4.0 * position_variance
        )
        self.log_width_filter = self.log_width_filter or KalmanAxis(
            _log_size(_width(self.filtered_bbox or self.bbox)),
            self.velocity[2],
            0.04,
            0.0,
            0.0,
            0.04,
        )
        self.log_height_filter = self.log_height_filter or KalmanAxis(
            _log_size(_height(self.filtered_bbox or self.bbox)),
            self.velocity[3],
            0.04,
            0.0,
            0.0,
            0.04,
        )
        self.state_frame_index = (
            self.last_frame_index
            if self.state_frame_index is None
            else self.state_frame_index
        )
        self.filtered_bbox = self.filtered_bbox or self.bbox
        self.display_bbox = self.display_bbox or self.filtered_bbox
        self.bbox_corners = self.bbox_corners or _box_corners(self.bbox)
        self.filtered_bbox_corners = self.filtered_bbox_corners or _box_corners(
            self.filtered_bbox or self.bbox
        )
        self.display_bbox_corners = self.display_bbox_corners or _box_corners(
            self.display_bbox or self.filtered_bbox or self.bbox
        )

    def predict(self, frame_index: int) -> BBox:
        if not self.covariance_motion_enabled:
            elapsed = max(0, frame_index - self.last_frame_index)
            center_x, center_y = _center(self.filtered_bbox or self.bbox)
            width = max(
                1.0,
                _width(self.filtered_bbox or self.bbox) + self.velocity[2] * elapsed,
            )
            height = max(
                1.0,
                _height(self.filtered_bbox or self.bbox) + self.velocity[3] * elapsed,
            )
            center_x += self.velocity[0] * elapsed
            center_y += self.velocity[1] * elapsed
            return (
                center_x - width * 0.5,
                center_y - height * 0.5,
                center_x + width * 0.5,
                center_y + height * 0.5,
            )
        elapsed = max(0, frame_index - int(self.state_frame_index or 0))
        assert self.center_x_filter is not None
        assert self.center_y_filter is not None
        assert self.log_width_filter is not None
        assert self.log_height_filter is not None
        center_x = (
            self.center_x_filter.position + self.center_x_filter.velocity * elapsed
        )
        center_y = (
            self.center_y_filter.position + self.center_y_filter.velocity * elapsed
        )
        width = _size_from_log(
            self.log_width_filter.position + self.log_width_filter.velocity * elapsed
        )
        height = _size_from_log(
            self.log_height_filter.position + self.log_height_filter.velocity * elapsed
        )
        return (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )

    def advance(self, frame_index: int, camera_uncertainty: float = 0.0) -> None:
        if not self.covariance_motion_enabled:
            return
        elapsed_frames = max(0, frame_index - int(self.state_frame_index or 0))
        if not elapsed_frames:
            return
        diagonal = max(
            1.0,
            math.hypot(
                _width(self.filtered_bbox or self.bbox),
                _height(self.filtered_bbox or self.bbox),
            ),
        )
        position_noise = 0.0025 * diagonal * diagonal + camera_uncertainty
        assert self.center_x_filter is not None
        assert self.center_y_filter is not None
        assert self.log_width_filter is not None
        assert self.log_height_filter is not None
        elapsed = float(elapsed_frames)
        self.center_x_filter.predict(elapsed, position_noise, 0.25 * position_noise)
        self.center_y_filter.predict(elapsed, position_noise, 0.25 * position_noise)
        self.log_width_filter.predict(elapsed, 0.0025, 0.001)
        self.log_height_filter.predict(elapsed, 0.0025, 0.001)
        _stabilize_size_axis(self.log_width_filter)
        _stabilize_size_axis(self.log_height_filter)
        previous = self.filtered_bbox or self.bbox
        self.state_frame_index = frame_index
        predicted = self.predict(frame_index)
        dx = _center(predicted)[0] - _center(previous)[0]
        dy = _center(predicted)[1] - _center(previous)[1]
        if (
            abs(_width(predicted) - _width(previous)) > 1.0e-3
            or abs(_height(predicted) - _height(previous)) > 1.0e-3
        ):
            self.filtered_bbox_corners = _box_corners(predicted)
        else:
            self.filtered_bbox_corners = tuple(
                (x + dx, y + dy)
                for x, y in self.filtered_bbox_corners or _box_corners(previous)
            )
        self.filtered_bbox = predicted
        self.velocity = (
            self.center_x_filter.velocity,
            self.center_y_filter.velocity,
            self.log_width_filter.velocity,
            self.log_height_filter.velocity,
        )

    def freeze_unobserved_size(self) -> None:
        """Coast position without extrapolating an unmeasured object scale."""
        assert self.log_width_filter is not None
        assert self.log_height_filter is not None
        reliable = self.display_bbox or self.filtered_bbox or self.bbox
        current = self.filtered_bbox or self.bbox
        center_x, center_y = _center(current)
        width = _width(reliable)
        height = _height(reliable)
        self.log_width_filter.position = _log_size(width)
        self.log_width_filter.velocity = 0.0
        self.log_width_filter.p01 = 0.0
        self.log_width_filter.p10 = 0.0
        self.log_height_filter.position = _log_size(height)
        self.log_height_filter.velocity = 0.0
        self.log_height_filter.p01 = 0.0
        self.log_height_filter.p10 = 0.0
        self.filtered_bbox = (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        )
        self.filtered_bbox_corners = _recenter_corners(
            self.display_bbox_corners
            or self.filtered_bbox_corners
            or _box_corners(current),
            self.filtered_bbox,
        )
        self.velocity = (
            self.center_x_filter.velocity if self.center_x_filter else 0.0,
            self.center_y_filter.velocity if self.center_y_filter else 0.0,
            0.0,
            0.0,
        )


def _position_measurement_variance(bbox: BBox, score: float, occluded: bool) -> float:
    diagonal = max(1.0, math.hypot(_width(bbox), _height(bbox)))
    confidence = min(1.0, max(0.0, score))
    standard_deviation = max(0.5, diagonal * (0.035 + 0.10 * (1.0 - confidence)))
    return standard_deviation**2 * (16.0 if occluded else 1.0)


def _size_measurement_variance(score: float, occluded: bool) -> float:
    confidence = min(1.0, max(0.0, score))
    standard_deviation = 0.04 + 0.16 * (1.0 - confidence)
    return standard_deviation**2 * (25.0 if occluded else 1.0)


def _center_mahalanobis_squared(
    track: TrackState, bbox: BBox, score: float, occluded: bool
) -> float:
    assert track.center_x_filter is not None
    assert track.center_y_filter is not None
    measurement_variance = _position_measurement_variance(bbox, score, occluded)
    x, y = _center(bbox)
    dx = x - track.center_x_filter.position
    dy = y - track.center_y_filter.position
    return dx * dx / max(1.0e-4, track.center_x_filter.p00 + measurement_variance) + (
        dy * dy / max(1.0e-4, track.center_y_filter.p00 + measurement_variance)
    )


def _direction_disagreement(track: TrackState, bbox: BBox, frame_index: int) -> float:
    if (
        track.previous_observation is None
        or track.last_frame_index <= track.previous_observation_frame
        or frame_index <= track.last_frame_index
    ):
        return 0.0
    current_x, current_y = _center(track.bbox)
    candidate_x, candidate_y = _center(bbox)
    previous_dx = current_x - track.previous_observation[0]
    previous_dy = current_y - track.previous_observation[1]
    candidate_dx = candidate_x - current_x
    candidate_dy = candidate_y - current_y
    previous_norm = math.hypot(previous_dx, previous_dy)
    candidate_norm = math.hypot(candidate_dx, candidate_dy)
    if previous_norm < 0.5 or candidate_norm < 0.5:
        return 0.0
    cosine = min(
        1.0,
        max(
            -1.0,
            (previous_dx * candidate_dx + previous_dy * candidate_dy)
            / (previous_norm * candidate_norm),
        ),
    )
    return 0.5 * (1.0 - cosine)


def _geometry_match(
    track: TrackState,
    detection: dict,
    config: TrackerConfig,
    frame_index: int,
) -> bool:
    if int(detection["class_id"]) != track.class_id:
        return False
    bbox = _bbox(detection)
    reference = (
        track.predict(frame_index) if config.center_distance_enabled else track.bbox
    )
    iou = _iou_xyxy(reference, bbox)
    if not config.center_distance_enabled:
        return iou >= config.match_iou_threshold
    distance = _normalized_center_distance(reference, bbox)
    if iou < config.match_iou_threshold and distance > config.max_center_distance:
        return False
    return (
        not config.covariance_motion_enabled
        or _center_mahalanobis_squared(
            track, bbox, float(detection["score"]), track.occluded
        )
        <= 16.0
    )


def _swept_boxes_may_overlap(
    first: TrackState,
    first_prediction: BBox,
    second: TrackState,
    second_prediction: BBox,
) -> bool:
    first_previous = _center(first.bbox)
    second_previous = _center(second.bbox)
    first_current = _center(first_prediction)
    second_current = _center(second_prediction)
    previous_dx = first_previous[0] - second_previous[0]
    previous_dy = first_previous[1] - second_previous[1]
    current_dx = first_current[0] - second_current[0]
    current_dy = first_current[1] - second_current[1]
    delta_x = current_dx - previous_dx
    delta_y = current_dy - previous_dy
    delta_squared = delta_x * delta_x + delta_y * delta_y
    closest_time = (
        min(
            1.0,
            max(
                0.0,
                -(previous_dx * delta_x + previous_dy * delta_y) / delta_squared,
            ),
        )
        if delta_squared > 1.0e-6
        else 0.0
    )
    closest_distance = math.hypot(
        previous_dx + closest_time * delta_x,
        previous_dy + closest_time * delta_y,
    )
    collision_radius = 0.35 * (
        math.hypot(_width(first_prediction), _height(first_prediction))
        + math.hypot(_width(second_prediction), _height(second_prediction))
    )
    return closest_distance <= max(1.0, collision_radius)


def _prediction_horizon(track: TrackState, config: TrackerConfig) -> int:
    if track.occluded:
        return max(
            config.max_prediction_frames,
            min(config.max_occlusion_frames, config.max_missing_frames),
        )
    return config.max_prediction_frames


def _prediction_is_publishable(track: TrackState, config: TrackerConfig) -> bool:
    return 0 < track.missing_frames <= _prediction_horizon(track, config)


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
    ) -> CameraMotionEstimate | None:
        recent = [
            track
            for track in self._tracks.values()
            if track.last_frame_index == frame_index - 1
        ]
        if len(recent) < 3 or len(high) < 3:
            return None

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
            return None

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
                distance = _normalized_center_distance(reference, _bbox(detection))
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
            return None
        motion = (
            _median([offset[0] for offset in offsets]),
            _median([offset[1] for offset in offsets]),
        )
        if motion == (0.0, 0.0):
            return None
        return CameraMotionEstimate(
            transform=(1.0, 0.0, motion[0], 0.0, 1.0, motion[1]),
            confidence=0.25,
            reprojection_error=typical_diagonal * 0.10,
            inliers=len(offsets),
        )

    def _associate(
        self,
        detections: list[dict],
        detection_indices: list[int],
        frame_index: int,
        matched_tracks: set[int],
        matched_detections: set[int],
        assignments: dict[int, int],
        confirmed_only: bool = False,
    ) -> None:
        candidates = sorted(
            (
                track
                for track in self._tracks.values()
                if track.track_id not in matched_tracks
                and (track.confirmed or not confirmed_only)
            ),
            key=(
                (lambda track: track.track_id)
                if self.config.center_distance_enabled
                else (lambda track: (-track.last_frame_index, track.track_id))
            ),
        )
        if not self.config.center_distance_enabled:
            begin = 0
            while begin < len(candidates):
                end = begin + 1
                while (
                    end < len(candidates)
                    and candidates[end].last_frame_index
                    == candidates[begin].last_frame_index
                ):
                    end += 1
                self._associate_global_pool(
                    detections,
                    detection_indices,
                    frame_index,
                    [track.track_id for track in candidates[begin:end]],
                    matched_tracks,
                    matched_detections,
                    assignments,
                )
                begin = end
            return
        self._associate_global_pool(
            detections,
            detection_indices,
            frame_index,
            [track.track_id for track in candidates],
            matched_tracks,
            matched_detections,
            assignments,
        )

    def _associate_global_pool(
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
                bbox = _bbox(detection)
                iou = _iou_xyxy(reference, bbox)
                center_distance = _normalized_center_distance(reference, bbox)
                if not _geometry_match(track, detection, self.config, frame_index):
                    row_costs.append(_BLOCKED_COST)
                    continue
                if not self.config.center_distance_enabled:
                    row_costs.append(1.0 - iou)
                    continue
                diagonal = max(1.0, math.hypot(_width(reference), _height(reference)))
                iou_weight = 0.45 if diagonal < 32.0 else 0.60
                center_weight = 0.25 if diagonal < 32.0 else 0.15
                mahalanobis = _center_mahalanobis_squared(
                    track, bbox, float(detection["score"]), track.occluded
                )
                direction = _direction_disagreement(track, bbox, frame_index)
                assert track.log_width_filter is not None
                assert track.log_height_filter is not None
                size_difference = 0.5 * (
                    abs(_log_size(_width(bbox)) - track.log_width_filter.position)
                    + abs(_log_size(_height(bbox)) - track.log_height_filter.position)
                )
                row_costs.append(
                    iou_weight * (1.0 - iou)
                    + center_weight
                    * min(
                        1.0,
                        center_distance / max(0.01, self.config.max_center_distance),
                    )
                    + 0.15 * min(1.0, mahalanobis / 16.0)
                    + 0.10 * direction
                    + 0.05 * min(1.0, size_difference)
                    + 0.01 * track.missing_frames
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
        camera_transform: CameraMotionInput | None = None,
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
        camera_motion: CameraMotionEstimate | None = None
        if self.config.camera_motion_compensation:
            if _valid_camera_transform(camera_transform):
                assert camera_transform is not None
                camera_motion = (
                    camera_transform
                    if isinstance(camera_transform, CameraMotionEstimate)
                    else CameraMotionEstimate(camera_transform)
                )
            else:
                camera_motion = self._estimate_camera_motion(
                    detections, high, frame_index
                )
        applied_camera_transform = camera_motion.transform if camera_motion else None
        if applied_camera_transform is not None:
            a, b, tx, c, d, ty = applied_camera_transform
            for track in self._tracks.values():
                if track.previous_observation is not None:
                    previous_x, previous_y = track.previous_observation
                    track.previous_observation = (
                        a * previous_x + b * previous_y + tx,
                        c * previous_x + d * previous_y + ty,
                    )
                track.bbox_corners = _transform_corners(
                    track.bbox_corners or _box_corners(track.bbox),
                    applied_camera_transform,
                )
                track.filtered_bbox_corners = _transform_corners(
                    track.filtered_bbox_corners
                    or _box_corners(track.filtered_bbox or track.bbox),
                    applied_camera_transform,
                )
                track.bbox = _corners_bbox(track.bbox_corners)
                track.filtered_bbox = _corners_bbox(track.filtered_bbox_corners)
                track.display_bbox_corners = _transform_corners(
                    track.display_bbox_corners
                    or _box_corners(track.display_bbox or track.filtered_bbox),
                    applied_camera_transform,
                )
                track.display_bbox = _corners_bbox(track.display_bbox_corners)
                vx, vy, vw, vh = track.velocity
                track.velocity = (
                    a * vx + b * vy,
                    c * vx + d * vy,
                    vw if track.covariance_motion_enabled else vw * math.hypot(a, c),
                    vh if track.covariance_motion_enabled else vh * math.hypot(b, d),
                )
                if not track.covariance_motion_enabled:
                    continue
                assert track.center_x_filter is not None
                assert track.center_y_filter is not None
                assert track.log_width_filter is not None
                assert track.log_height_filter is not None
                vx = track.center_x_filter.velocity
                vy = track.center_y_filter.velocity
                track.center_x_filter.position, track.center_y_filter.position = (
                    _center(track.filtered_bbox)
                )
                track.center_x_filter.velocity = a * vx + b * vy
                track.center_y_filter.velocity = c * vx + d * vy
                track.log_width_filter.position = _log_size(_width(track.filtered_bbox))
                track.log_height_filter.position = _log_size(
                    _height(track.filtered_bbox)
                )
        previous_occluded = {
            track_id: track.occluded for track_id, track in self._tracks.items()
        }
        camera_uncertainty = (
            max(0.0, camera_motion.reprojection_error**2)
            + 4.0 * (1.0 - min(1.0, max(0.0, camera_motion.confidence)))
            if camera_motion is not None
            else (4.0 if self.config.camera_motion_compensation else 0.0)
        )
        for track in self._tracks.values():
            track.advance(frame_index, camera_uncertainty)
            track.occluded = False
            track.association_confidence = 1.0

        track_list = list(self._tracks.values())
        occlusion_tracking_enabled = (
            self.config.center_distance_enabled
            and self.config.covariance_motion_enabled
            and self.config.max_occlusion_frames > 0
        )
        if occlusion_tracking_enabled:
            predictions = {
                track.track_id: track.predict(frame_index) for track in track_list
            }
            visited: set[int] = set()
            for root in track_list:
                if root.track_id in visited:
                    continue
                component: list[TrackState] = []
                stack = [root]
                visited.add(root.track_id)
                while stack:
                    first = stack.pop()
                    component.append(first)
                    for second in track_list:
                        if (
                            second.track_id in visited
                            or first.class_id != second.class_id
                        ):
                            continue
                        if _overlap_coefficient(
                            predictions[first.track_id], predictions[second.track_id]
                        ) < self.config.overlap_threshold and not _swept_boxes_may_overlap(
                            first,
                            predictions[first.track_id],
                            second,
                            predictions[second.track_id],
                        ):
                            continue
                        visited.add(second.track_id)
                        stack.append(second)
                if len(component) < 2:
                    continue

                detection_owner: dict[int, int] = {}

                def augment(track: TrackState, seen: set[int]) -> bool:
                    for detection_index, detection in enumerate(detections):
                        if detection_index in seen or not _geometry_match(
                            track, detection, self.config, frame_index
                        ):
                            continue
                        seen.add(detection_index)
                        owner = detection_owner.get(detection_index)
                        if owner is None or augment(self._tracks[owner], seen):
                            detection_owner[detection_index] = track.track_id
                            return True
                    return False

                maximum_matches = sum(augment(track, set()) for track in component)
                if maximum_matches < len(component):
                    for track in component:
                        track.occluded = True

        feasible_track_counts = [
            sum(
                _geometry_match(track, detection, self.config, frame_index)
                for track in self._tracks.values()
            )
            for detection in detections
        ]
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
            confirmed_only=True,
        )

        for detection_index, track_id in assignments.items():
            detection = detections[detection_index]
            bbox = _bbox(detection)
            track = self._tracks[track_id]
            if track.occluded and feasible_track_counts[detection_index] > 1:
                track.missing_frames = frame_index - track.last_frame_index
                track.association_confidence = 0.0
                track.freeze_unobserved_size()
                continue
            elapsed = max(1, frame_index - track.last_frame_index)
            old_x, old_y = _center(track.bbox)
            new_x, new_y = _center(bbox)
            if not self.config.covariance_motion_enabled:
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
                        1.0,
                        innovation / max(0.01, self.config.max_center_distance),
                    )
                    alpha += (1.0 - alpha) * response
                predicted_center = _center(prediction)
                observed_center = _center(bbox)
                filtered_center = tuple(
                    (1.0 - alpha) * previous + alpha * current
                    for previous, current in zip(predicted_center, observed_center)
                )
                filtered_width = max(
                    1.0,
                    (1.0 - alpha) * _width(prediction) + alpha * _width(bbox),
                )
                filtered_height = max(
                    1.0,
                    (1.0 - alpha) * _height(prediction) + alpha * _height(bbox),
                )
                track.filtered_bbox = (
                    filtered_center[0] - filtered_width * 0.5,
                    filtered_center[1] - filtered_height * 0.5,
                    filtered_center[0] + filtered_width * 0.5,
                    filtered_center[1] + filtered_height * 0.5,
                )
                track.display_bbox = track.filtered_bbox
                track.bbox = bbox
                track.filtered_bbox_corners = _box_corners(track.filtered_bbox)
                track.display_bbox_corners = track.filtered_bbox_corners
                track.bbox_corners = _box_corners(track.bbox)
                track.score = float(detection["score"])
                track.last_frame_index = frame_index
                track.missing_frames = 0
                track.occluded = False
                track.association_confidence = 1.0
                track.hits += 1
                track.confirmed = track.hits >= self.config.min_confirmed_hits
                continue
            was_missing = track.missing_frames > 0
            measured = (
                (new_x - old_x) / elapsed,
                (new_y - old_y) / elapsed,
                (_log_size(_width(bbox)) - _log_size(_width(track.bbox))) / elapsed,
                (_log_size(_height(bbox)) - _log_size(_height(track.bbox))) / elapsed,
            )
            momentum = (
                min(0.50, self.config.velocity_momentum)
                if self.config.covariance_motion_enabled
                else self.config.velocity_momentum
            )
            prediction = track.filtered_bbox or track.predict(frame_index)
            mahalanobis = _center_mahalanobis_squared(
                track, bbox, float(detection["score"]), False
            )
            assert track.center_x_filter is not None
            assert track.center_y_filter is not None
            assert track.log_width_filter is not None
            assert track.log_height_filter is not None
            position_variance = _position_measurement_variance(
                bbox, float(detection["score"]), False
            )
            size_variance = _size_measurement_variance(float(detection["score"]), False)
            maximum_size_innovation = _MAXIMUM_LOG_SIZE_INNOVATION_PER_FRAME * elapsed
            width_measurement = min(
                track.log_width_filter.position + maximum_size_innovation,
                max(
                    track.log_width_filter.position - maximum_size_innovation,
                    _log_size(_width(bbox)),
                ),
            )
            height_measurement = min(
                track.log_height_filter.position + maximum_size_innovation,
                max(
                    track.log_height_filter.position - maximum_size_innovation,
                    _log_size(_height(bbox)),
                ),
            )
            track.center_x_filter.update(new_x, position_variance)
            track.center_y_filter.update(new_y, position_variance)
            track.log_width_filter.update(width_measurement, size_variance)
            track.log_height_filter.update(height_measurement, size_variance)
            filters = (
                track.center_x_filter,
                track.center_y_filter,
                track.log_width_filter,
                track.log_height_filter,
            )
            for axis, observed_velocity in zip(filters, measured):
                axis.velocity = (
                    momentum * axis.velocity + (1.0 - momentum) * observed_velocity
                )
            _stabilize_size_axis(track.log_width_filter)
            _stabilize_size_axis(track.log_height_filter)
            posterior = track.predict(frame_index)
            if was_missing:
                track.display_bbox = posterior
            else:
                alpha = self.config.box_smoothing_alpha
                predicted_center = _center(prediction)
                observed_center = _center(posterior)
                filtered_center = tuple(
                    (1.0 - alpha) * previous + alpha * current
                    for previous, current in zip(predicted_center, observed_center)
                )
                filtered_width = max(
                    1.0,
                    (1.0 - alpha) * _width(prediction) + alpha * _width(posterior),
                )
                filtered_height = max(
                    1.0,
                    (1.0 - alpha) * _height(prediction) + alpha * _height(posterior),
                )
                track.display_bbox = (
                    filtered_center[0] - filtered_width * 0.5,
                    filtered_center[1] - filtered_height * 0.5,
                    filtered_center[0] + filtered_width * 0.5,
                    filtered_center[1] + filtered_height * 0.5,
                )
            track.display_bbox_corners = _box_corners(track.display_bbox)
            track.filtered_bbox = posterior
            track.previous_observation = _center(track.bbox)
            track.previous_observation_frame = track.last_frame_index
            track.bbox = bbox
            track.filtered_bbox_corners = _box_corners(track.filtered_bbox)
            track.bbox_corners = _box_corners(track.bbox)
            track.score = float(detection["score"])
            track.last_frame_index = frame_index
            track.missing_frames = 0
            track.occluded = False
            track.association_confidence = math.exp(-0.125 * min(16.0, mahalanobis))
            track.velocity = tuple(axis.velocity for axis in filters)
            track.hits += 1
            track.confirmed = track.hits >= self.config.min_confirmed_hits

        for detection_index in high:
            if detection_index in matched_detections:
                continue
            detection = detections[detection_index]
            if float(detection["score"]) < self.config.new_track_threshold:
                continue
            bbox = _bbox(detection)
            if self.config.center_distance_enabled and any(
                track.confirmed
                and track.class_id == int(detection["class_id"])
                and _overlap_coefficient(track.predict(frame_index), bbox)
                >= max(0.50, self.config.overlap_threshold)
                for track in self._tracks.values()
            ):
                matched_detections.add(detection_index)
                continue
            if len(self._tracks) >= self.config.max_active_tracks:
                replacements = [
                    track
                    for track in self._tracks.values()
                    if track.track_id not in matched_tracks
                ]
                if not replacements:
                    continue
                replacement = min(
                    replacements,
                    key=lambda track: (
                        track.occluded,
                        track.confirmed,
                        track.last_frame_index,
                        track.track_id,
                    ),
                )
                del self._tracks[replacement.track_id]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=bbox,
                score=float(detection["score"]),
                class_id=int(detection["class_id"]),
                last_frame_index=frame_index,
                filtered_bbox=bbox,
                display_bbox=bbox,
                confirmed=self.config.min_confirmed_hits <= 1,
                covariance_motion_enabled=self.config.covariance_motion_enabled,
            )
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments[detection_index] = track_id

        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.freeze_unobserved_size()
                track.missing_frames = frame_index - track.last_frame_index
                if previous_occluded.get(track_id, False):
                    track.occluded = True

        tracked: list[TrackedDetection] = []
        published_tracks: set[int] = set()
        for detection_index, detection in enumerate(detections):
            track_id = assignments.get(detection_index)
            if track_id is None or not self._tracks[track_id].confirmed:
                continue
            track = self._tracks[track_id]
            predicted = track.last_frame_index != frame_index
            if predicted and not _prediction_is_publishable(track, self.config):
                continue
            bbox = (
                track.filtered_bbox
                if predicted
                else track.display_bbox or track.filtered_bbox
            ) or _bbox(detection)
            tracked.append(
                TrackedDetection(
                    track_id,
                    *bbox,
                    float(detection["score"]),
                    int(detection["class_id"]),
                    predicted,
                    track.occluded,
                    track.association_confidence,
                )
            )
            published_tracks.add(track_id)

        if self.config.max_prediction_frames or self.config.max_occlusion_frames:
            for track in self._tracks.values():
                if (
                    not track.confirmed
                    or not _prediction_is_publishable(track, self.config)
                    or track.score < self.config.high_score_threshold
                    or track.track_id in published_tracks
                ):
                    continue
                bbox = track.predict(frame_index)
                if not track.occluded and any(
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
                        track.occluded,
                        track.association_confidence,
                    )
                )
                published_tracks.add(track.track_id)

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
    """Background-masked sparse-flow partial-affine camera-motion estimator."""

    def __init__(self, downscale: int = 4, max_features: int = 200) -> None:
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self._downscale = max(1, int(downscale))
        self._max_features = max(32, int(max_features))
        self._previous_gray = None
        self._previous_points = None

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_points = None

    def update(
        self, gray_frame, detections: list[dict] | None = None
    ) -> CameraMotionEstimate | None:
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

        result = None
        if (
            self._previous_gray is not None
            and self._previous_points is not None
            and len(self._previous_points) >= 8
        ):
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                self._previous_points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
            )
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                self._previous_gray,
                forward,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03),
            )
            forward_status = forward_status.reshape(-1).astype(bool)
            backward_status = backward_status.reshape(-1).astype(bool)
            forward_backward_error = np.linalg.norm(
                self._previous_points.reshape(-1, 2) - backward.reshape(-1, 2), axis=1
            )
            valid = forward_status & backward_status & (forward_backward_error <= 1.0)
            previous = self._previous_points.reshape(-1, 2)[valid].astype(np.float32)
            current = forward.reshape(-1, 2)[valid].astype(np.float32)
            if len(previous) >= 8:
                affine, inlier_mask = cv2.estimateAffinePartial2D(
                    previous,
                    current,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,
                    maxIters=300,
                    confidence=0.995,
                    refineIters=10,
                )
                inlier_flags = (
                    np.zeros(len(previous), dtype=bool)
                    if inlier_mask is None
                    else inlier_mask.reshape(-1).astype(bool)
                )
                inliers = int(np.count_nonzero(inlier_flags))
                residual = self._residual(affine, previous, current, inlier_flags)
                if affine is not None and self._plausible(
                    affine,
                    gray.shape[1],
                    gray.shape[0],
                    inliers,
                    len(previous),
                    residual,
                ):
                    scale = float(self._downscale)
                    coverage = self._spatial_coverage(
                        previous, inlier_flags, gray.shape[1], gray.shape[0]
                    )
                    confidence = min(
                        1.0,
                        max(
                            0.0,
                            (inliers / len(previous))
                            * min(1.0, coverage / 0.25)
                            * math.exp(-0.5 * residual),
                        ),
                    )
                    result = CameraMotionEstimate(
                        transform=(
                            float(affine[0, 0]),
                            float(affine[0, 1]),
                            scale * float(affine[0, 2]),
                            float(affine[1, 0]),
                            float(affine[1, 1]),
                            scale * float(affine[1, 2]),
                        ),
                        confidence=confidence,
                        reprojection_error=scale * residual,
                        inliers=inliers,
                    )

        self._previous_points = cv2.goodFeaturesToTrack(
            gray,
            mask=feature_mask,
            maxCorners=self._max_features,
            qualityLevel=0.01,
            minDistance=4.0,
            blockSize=3,
        )
        self._previous_gray = gray.copy()
        return result

    @staticmethod
    def _residual(affine, previous, current, inliers) -> float:
        import numpy as np

        if affine is None or not inliers.any():
            return math.inf
        homogeneous = np.column_stack([previous, np.ones(len(previous))])
        projected = homogeneous @ affine.T
        return float(
            np.linalg.norm(projected[inliers] - current[inliers], axis=1).mean()
        )

    @staticmethod
    def _spatial_coverage(points, inliers, width: int, height: int) -> float:
        if width <= 0 or height <= 0 or not inliers.any():
            return 0.0
        selected = points[inliers]
        extent = selected.max(axis=0) - selected.min(axis=0)
        return min(1.0, max(0.0, float(extent[0] * extent[1]) / (width * height)))

    @staticmethod
    def _plausible(
        affine, width: int, height: int, inliers: int, matches: int, residual: float
    ) -> bool:
        if (
            inliers < 8
            or matches <= 0
            or inliers / matches < 0.25
            or not math.isfinite(residual)
            or residual > 3.0
        ):
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
