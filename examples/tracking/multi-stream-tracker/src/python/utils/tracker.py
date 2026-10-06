"""Per-class ByteTrack-style tracker for the Python multi-stream tracker example.

Each configured class owns one ``ByteTracker`` with its own settings. Tracks are
matched only against detections of their own class, so a track never changes
class. Track IDs come from one counter per stream and stay unique across classes.

The Kalman filter follows ByteTrack's constant-velocity model over
``(center_x, center_y, aspect, height)``. Its matrices keep each coordinate
independent of the others, so the filter is stored as four 2x2 blocks
(position, velocity) instead of one 8x8 matrix. The math is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math

from utils.coco import COCO_LABELS

MAX_CLASSES = 5
_GATED_COST = 1.0e6

BBox = tuple[float, float, float, float]


def _iou_xyxy(a: BBox, b: BBox) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def linear_assignment(cost: list[list[float]], max_cost: float) -> list[tuple[int, int]]:
    """Minimum-cost assignment (Hungarian); pairs above ``max_cost`` are dropped."""
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []
    transposed = rows > cols
    matrix = [list(column) for column in zip(*cost)] if transposed else cost
    n, m = (cols, rows) if transposed else (rows, cols)

    # Shortest augmenting path with potentials; requires n <= m.
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    owner = [0] * (m + 1)
    way = [0] * (m + 1)
    for row in range(1, n + 1):
        owner[0] = row
        col0 = 0
        min_value = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[col0] = True
            row0 = owner[col0]
            delta = math.inf
            col1 = 0
            for col in range(1, m + 1):
                if used[col]:
                    continue
                reduced = matrix[row0 - 1][col - 1] - u[row0] - v[col]
                if reduced < min_value[col]:
                    min_value[col] = reduced
                    way[col] = col0
                if min_value[col] < delta:
                    delta = min_value[col]
                    col1 = col
            for col in range(m + 1):
                if used[col]:
                    u[owner[col]] += delta
                    v[col] -= delta
                else:
                    min_value[col] -= delta
            col0 = col1
            if owner[col0] == 0:
                break
        while col0:
            col1 = way[col0]
            owner[col0] = owner[col1]
            col0 = col1

    pairs = []
    for col in range(1, m + 1):
        row = owner[col]
        if row == 0:
            continue
        r, c = (col - 1, row - 1) if transposed else (row - 1, col - 1)
        if cost[r][c] <= max_cost:
            pairs.append((r, c))
    pairs.sort()
    return pairs


@dataclass(frozen=True)
class ClassTrackerConfig:
    """Tracker settings for one detection class."""

    class_id: int
    label: str
    high_score_threshold: float = 0.50
    low_score_threshold: float = 0.10
    new_track_threshold: float = 0.60
    match_iou_threshold: float = 0.20
    low_match_iou_threshold: float = 0.50
    max_missing_frames: int = 30
    min_confirmed_hits: int = 2
    position_noise: float = 0.05
    velocity_noise: float = 0.00625

    def validate(self) -> None:
        name = f"tracking.classes[{self.label}]"
        for key in (
            "high_score_threshold",
            "low_score_threshold",
            "new_track_threshold",
            "match_iou_threshold",
            "low_match_iou_threshold",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}.{key} must be between 0 and 1")
        if self.low_score_threshold > self.high_score_threshold:
            raise ValueError(f"{name}.low_score_threshold must be <= high_score_threshold")
        if self.new_track_threshold < self.high_score_threshold:
            raise ValueError(f"{name}.new_track_threshold must be >= high_score_threshold")
        if self.max_missing_frames < 0:
            raise ValueError(f"{name}.max_missing_frames must be >= 0")
        if self.min_confirmed_hits < 1:
            raise ValueError(f"{name}.min_confirmed_hits must be >= 1")
        for key in ("position_noise", "velocity_noise"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name}.{key} must be > 0")


_SETTING_KEYS = {f.name for f in fields(ClassTrackerConfig)} - {"class_id", "label"}


def resolve_class(value: object) -> tuple[int, str]:
    """Map a COCO class name or id to ``(class_id, label)``."""
    if isinstance(value, bool):
        raise ValueError(f"unsupported class: {value!r}")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int):
        if 0 <= value < len(COCO_LABELS):
            return value, COCO_LABELS[value]
        raise ValueError(f"unsupported class id {value}; expected 0..{len(COCO_LABELS) - 1}")
    if isinstance(value, str) and value.strip():
        name = value.strip().lower()
        if name in COCO_LABELS:
            return COCO_LABELS.index(name), name
    raise ValueError(f"unsupported class: {value!r}; use a COCO class name or id")


def parse_class_configs(raw: object) -> tuple[ClassTrackerConfig, ...]:
    """Parse and validate the ``tracking.classes`` list."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_CLASSES:
        raise ValueError(f"tracking.classes must list 1 to {MAX_CLASSES} classes")
    configs: list[ClassTrackerConfig] = []
    seen: set[int] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or "class" not in entry:
            raise ValueError(f"tracking.classes[{index}] must be a mapping with a 'class' key")
        class_id, label = resolve_class(entry["class"])
        if class_id in seen:
            raise ValueError(f"tracking.classes[{index}]: duplicate class '{label}'")
        seen.add(class_id)
        settings = {}
        for key, value in entry.items():
            if key == "class":
                continue
            if key not in _SETTING_KEYS:
                raise ValueError(f"tracking.classes[{index}]: unknown key '{key}'")
            expected = int if key in {"max_missing_frames", "min_confirmed_hits"} else float
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"tracking.classes[{index}].{key} must be numeric")
            if expected is int and not isinstance(value, int):
                raise ValueError(f"tracking.classes[{index}].{key} must be an integer")
            settings[key] = expected(value)
        config = ClassTrackerConfig(class_id=class_id, label=label, **settings)
        config.validate()
        configs.append(config)
    return tuple(configs)


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str


class _KalmanBox:
    """ByteTrack's constant-velocity Kalman filter over (cx, cy, aspect, height)."""

    def __init__(self, box: BBox, position_noise: float, velocity_noise: float) -> None:
        self._wp = position_noise
        self._wv = velocity_noise
        measurement = _to_xyah(box)
        h = measurement[3]
        pos_std = (2 * self._wp * h, 2 * self._wp * h, 1e-2, 2 * self._wp * h)
        vel_std = (10 * self._wv * h, 10 * self._wv * h, 1e-5, 10 * self._wv * h)
        self.pos = list(measurement)
        self.vel = [0.0, 0.0, 0.0, 0.0]
        # Per coordinate covariance [[a, b], [b, c]] of (position, velocity).
        self.cov = [[p * p, 0.0, v * v] for p, v in zip(pos_std, vel_std)]

    def predict(self) -> None:
        h = self.pos[3]
        pos_std = (self._wp * h, self._wp * h, 1e-2, self._wp * h)
        vel_std = (self._wv * h, self._wv * h, 1e-5, self._wv * h)
        for i in range(4):
            self.pos[i] += self.vel[i]
            a, b, c = self.cov[i]
            self.cov[i] = [a + 2 * b + c + pos_std[i] ** 2, b + c, c + vel_std[i] ** 2]

    def update(self, box: BBox) -> None:
        measurement = _to_xyah(box)
        h = self.pos[3]
        meas_std = (self._wp * h, self._wp * h, 1e-1, self._wp * h)
        for i in range(4):
            a, b, c = self.cov[i]
            s = a + meas_std[i] ** 2
            k_pos, k_vel = a / s, b / s
            innovation = measurement[i] - self.pos[i]
            self.pos[i] += k_pos * innovation
            self.vel[i] += k_vel * innovation
            self.cov[i] = [a - k_pos * a, b - k_pos * b, c - k_vel * b]

    def box(self) -> BBox:
        cx, cy, aspect, h = self.pos
        w = aspect * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _to_xyah(box: BBox) -> tuple[float, float, float, float]:
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    return (box[0] + w / 2, box[1] + h / 2, w / h, h)


@dataclass
class _Track:
    track_id: int
    kalman: _KalmanBox
    score: float
    last_frame_index: int
    hits: int = 1
    confirmed: bool = False
    lost: bool = False


@dataclass
class _IdCounter:
    next_id: int = 1

    def take(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value


class ByteTracker:
    """ByteTrack-style tracker for one class."""

    def __init__(self, config: ClassTrackerConfig, ids: _IdCounter | None = None) -> None:
        config.validate()
        self.config = config
        self._ids = ids or _IdCounter()
        self._tracks: list[_Track] = []

    def active_track_count(self) -> int:
        return len(self._tracks)

    def _match(
        self, tracks: list[_Track], boxes: list[BBox], iou_threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        cost = []
        for track in tracks:
            predicted = track.kalman.box()
            row = []
            for box in boxes:
                iou = _iou_xyxy(predicted, box)
                row.append(1.0 - iou if iou >= iou_threshold else _GATED_COST)
            cost.append(row)
        pairs = linear_assignment(cost, 1.0 - iou_threshold)
        matched_tracks = {t for t, _ in pairs}
        matched_boxes = {d for _, d in pairs}
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in matched_tracks],
            [i for i in range(len(boxes)) if i not in matched_boxes],
        )

    def update(self, detections: list[dict], frame_index: int) -> list[TrackedDetection]:
        cfg = self.config
        high: list[tuple[BBox, float]] = []
        low: list[tuple[BBox, float]] = []
        for det in detections:
            if int(det["class_id"]) != cfg.class_id:
                continue
            score = float(det["score"])
            box = (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if score >= cfg.high_score_threshold:
                high.append((box, score))
            elif score >= cfg.low_score_threshold:
                low.append((box, score))

        for track in self._tracks:
            if track.lost:
                track.kalman.vel[3] = 0.0
            track.kalman.predict()

        updated: list[_Track] = []

        def apply(track: _Track, box: BBox, score: float) -> None:
            track.kalman.update(box)
            track.score = score
            track.last_frame_index = frame_index
            track.hits += 1
            track.lost = False
            track.confirmed = track.confirmed or track.hits >= cfg.min_confirmed_hits
            updated.append(track)

        # Stage 1: confirmed tracks (active or lost) against high-score detections.
        pool = [t for t in self._tracks if t.confirmed]
        pairs, left_pool, left_high = self._match(
            pool, [b for b, _ in high], cfg.match_iou_threshold
        )
        for t, d in pairs:
            apply(pool[t], *high[d])

        # Stage 2: still-active tracks recover through low-score detections.
        active = [pool[i] for i in left_pool if not pool[i].lost]
        pairs, left_active, _ = self._match(
            active, [b for b, _ in low], cfg.low_match_iou_threshold
        )
        for t, d in pairs:
            apply(active[t], *low[d])
        for i in left_active:
            active[i].lost = True

        # Stage 3: tentative tracks against the remaining high-score detections.
        tentative = [t for t in self._tracks if not t.confirmed]
        remaining = [high[i] for i in left_high]
        pairs, _, left_remaining = self._match(
            tentative, [b for b, _ in remaining], cfg.match_iou_threshold
        )
        matched_tentative = {t for t, _ in pairs}
        for t, d in pairs:
            apply(tentative[t], *remaining[d])
        dropped = {id(tentative[i]) for i in range(len(tentative)) if i not in matched_tentative}

        survivors = [
            t
            for t in self._tracks
            if id(t) not in dropped
            and frame_index - t.last_frame_index <= cfg.max_missing_frames
        ]
        for i in left_remaining:
            box, score = remaining[i]
            if score < cfg.new_track_threshold:
                continue
            track = _Track(
                track_id=self._ids.take(),
                kalman=_KalmanBox(box, cfg.position_noise, cfg.velocity_noise),
                score=score,
                last_frame_index=frame_index,
                confirmed=cfg.min_confirmed_hits <= 1,
            )
            survivors.append(track)
            updated.append(track)
        self._tracks = survivors

        output = []
        for track in updated:
            if not track.confirmed:
                continue
            x1, y1, x2, y2 = track.kalman.box()
            output.append(
                TrackedDetection(
                    track.track_id, x1, y1, x2, y2, track.score, cfg.class_id, cfg.label
                )
            )
        output.sort(key=lambda item: item.track_id)
        return output


@dataclass
class MultiClassTracker:
    """One ``ByteTracker`` per configured class with stream-unique track IDs."""

    configs: tuple[ClassTrackerConfig, ...]
    _trackers: list[ByteTracker] = field(init=False)

    def __post_init__(self) -> None:
        ids = _IdCounter()
        self._trackers = [ByteTracker(config, ids) for config in self.configs]

    def active_track_count(self) -> int:
        return sum(tracker.active_track_count() for tracker in self._trackers)

    def update(self, detections: list[dict], frame_index: int) -> list[TrackedDetection]:
        tracked: list[TrackedDetection] = []
        for tracker in self._trackers:
            tracked.extend(tracker.update(detections, frame_index))
        return tracked
