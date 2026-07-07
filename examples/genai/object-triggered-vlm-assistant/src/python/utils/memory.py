"""Linked frame and detector metadata memory for recent trigger-class objects."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    timestamp_ms: int
    jpeg: bytes
    width: int
    height: int


@dataclass(frozen=True)
class ObjectObservation:
    track_id: int
    frame_id: int
    timestamp_ms: int
    class_name: str
    bbox: list[int]
    score: float


@dataclass
class TrackState:
    track_id: int
    class_name: str
    bbox: list[int]
    score: float
    first_seen_ms: int
    last_seen_ms: int


def iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def encode_jpeg(frame_rgb: np.ndarray, quality: int) -> bytes:
    quality = max(1, min(100, int(quality)))
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode frame memory JPEG")
    return encoded.tobytes()


def decode_jpeg_rgb(jpeg: bytes) -> np.ndarray:
    payload = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("failed to decode frame memory JPEG")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class FrameRingBuffer:
    def __init__(self, retention_seconds: float, jpeg_quality: int):
        self.retention_ms = max(1, int(retention_seconds * 1000))
        self.jpeg_quality = jpeg_quality
        self.frames: deque[FrameRecord] = deque()
        self.frames_by_id: dict[int, FrameRecord] = {}
        self.latest_frame_id: int | None = None
        self.latest_timestamp_ms: int | None = None
        self.latest_frame_rgb: np.ndarray | None = None
        self.lock = threading.RLock()

    def set_latest(self, frame_id: int, timestamp_ms: int, frame_rgb: np.ndarray) -> None:
        with self.lock:
            self.latest_frame_id = frame_id
            self.latest_timestamp_ms = timestamp_ms
            self.latest_frame_rgb = frame_rgb.copy()

    def latest(self) -> tuple[int, int, np.ndarray] | None:
        with self.lock:
            if (
                self.latest_frame_id is None
                or self.latest_timestamp_ms is None
                or self.latest_frame_rgb is None
            ):
                return None
            return self.latest_frame_id, self.latest_timestamp_ms, self.latest_frame_rgb.copy()

    def store(self, frame_id: int, timestamp_ms: int, frame_rgb: np.ndarray) -> FrameRecord:
        height, width = frame_rgb.shape[:2]
        record = FrameRecord(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            jpeg=encode_jpeg(frame_rgb, self.jpeg_quality),
            width=width,
            height=height,
        )
        with self.lock:
            old = self.frames_by_id.get(frame_id)
            if old is not None:
                self.frames.remove(old)
            self.frames.append(record)
            self.frames_by_id[frame_id] = record
            self.prune_locked(timestamp_ms)
        return record

    def get(self, frame_id: int) -> FrameRecord | None:
        with self.lock:
            return self.frames_by_id.get(frame_id)

    def nearest(self, timestamp_ms: int) -> FrameRecord | None:
        with self.lock:
            if not self.frames:
                return None
            return min(self.frames, key=lambda frame: abs(frame.timestamp_ms - timestamp_ms))

    def prune(self, now_ms: int) -> None:
        with self.lock:
            self.prune_locked(now_ms)

    def prune_locked(self, now_ms: int) -> None:
        min_timestamp = now_ms - self.retention_ms
        while self.frames and self.frames[0].timestamp_ms < min_timestamp:
            old = self.frames.popleft()
            self.frames_by_id.pop(old.frame_id, None)


class ObjectTrackMemory:
    def __init__(
        self,
        trigger_classes,
        sample_interval_seconds: float,
        retention_seconds: float,
        iou_threshold: float,
        max_missing_seconds: float,
    ):
        self.trigger_classes = self._normalize_classes(trigger_classes)
        self.sample_interval_ms = max(1, int(sample_interval_seconds * 1000))
        self.retention_ms = max(1, int(retention_seconds * 1000))
        self.iou_threshold = float(iou_threshold)
        self.max_missing_ms = max(1, int(max_missing_seconds * 1000))
        self.tracks: dict[int, TrackState] = {}
        self.observations: deque[ObjectObservation] = deque()
        self.next_track_id = 1
        self.last_sample_ms = 0
        self.lock = threading.RLock()

    def _normalize_classes(self, trigger_classes) -> set[str]:
        if isinstance(trigger_classes, str):
            values = [trigger_classes]
        else:
            values = trigger_classes
        return {str(value).strip().lower() for value in values if str(value).strip()}

    def set_trigger_classes(self, trigger_classes) -> None:
        with self.lock:
            self.trigger_classes = self._normalize_classes(trigger_classes)
            self.tracks = {
                track_id: track
                for track_id, track in self.tracks.items()
                if track.class_name in self.trigger_classes
            }

    def set_trigger_class(self, trigger_class: str) -> None:
        self.set_trigger_classes([trigger_class])

    def update_tracks(
        self,
        boxes: list[dict],
        labels: list[str],
        timestamp_ms: int,
    ) -> list[TrackState]:
        detections = []
        for box in boxes:
            class_id = int(box["class_id"])
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            if label.lower() not in self.trigger_classes:
                continue
            detections.append(
                {
                    "class_name": label.lower(),
                    "bbox": [int(value) for value in box["bbox"]],
                    "score": float(box["score"]),
                }
            )

        with self.lock:
            assigned_tracks: set[int] = set()
            for detection in detections:
                best_track = None
                best_iou = 0.0
                for track in self.tracks.values():
                    if track.track_id in assigned_tracks:
                        continue
                    overlap = iou(track.bbox, detection["bbox"])
                    if overlap > best_iou:
                        best_iou = overlap
                        best_track = track

                if best_track is not None and best_iou >= self.iou_threshold:
                    best_track.bbox = detection["bbox"]
                    best_track.score = detection["score"]
                    best_track.last_seen_ms = timestamp_ms
                    assigned_tracks.add(best_track.track_id)
                    continue

                track = TrackState(
                    track_id=self.next_track_id,
                    class_name=detection["class_name"],
                    bbox=detection["bbox"],
                    score=detection["score"],
                    first_seen_ms=timestamp_ms,
                    last_seen_ms=timestamp_ms,
                )
                self.tracks[track.track_id] = track
                assigned_tracks.add(track.track_id)
                self.next_track_id += 1

            self._expire_tracks_locked(timestamp_ms)
            return [
                self.tracks[track_id]
                for track_id in assigned_tracks
                if track_id in self.tracks
            ]

    def maybe_sample(
        self,
        frame_id: int,
        timestamp_ms: int,
        tracks: list[TrackState],
    ) -> list[ObjectObservation]:
        if not tracks:
            return []
        with self.lock:
            if timestamp_ms - self.last_sample_ms < self.sample_interval_ms:
                return []
            self.last_sample_ms = timestamp_ms
            observations = [
                ObjectObservation(
                    track_id=track.track_id,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    class_name=track.class_name,
                    bbox=list(track.bbox),
                    score=float(track.score),
                )
                for track in tracks
            ]
            self.observations.extend(observations)
            self.prune_locked(timestamp_ms)
            return observations

    def find_near(
        self,
        timestamp_ms: int,
        tolerance_seconds: float,
        track_id: int | None = None,
        class_name: str | None = None,
    ) -> ObjectObservation | None:
        tolerance_ms = max(1, int(tolerance_seconds * 1000))
        normalized_class = class_name.lower() if class_name else None
        with self.lock:
            candidates = [
                obs
                for obs in self.observations
                if abs(obs.timestamp_ms - timestamp_ms) <= tolerance_ms
                and (track_id is None or obs.track_id == track_id)
                and (normalized_class is None or obs.class_name == normalized_class)
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda obs: abs(obs.timestamp_ms - timestamp_ms))

    def nearest_retained(
        self,
        timestamp_ms: int,
        now_ms: int,
        track_id: int | None = None,
        class_name: str | None = None,
    ) -> ObjectObservation | None:
        min_timestamp = now_ms - self.retention_ms
        normalized_class = class_name.lower() if class_name else None
        with self.lock:
            candidates = [
                obs
                for obs in self.observations
                if obs.timestamp_ms >= min_timestamp
                and (track_id is None or obs.track_id == track_id)
                and (normalized_class is None or obs.class_name == normalized_class)
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda obs: abs(obs.timestamp_ms - timestamp_ms))

    def latest(self) -> ObjectObservation | None:
        with self.lock:
            return self.observations[-1] if self.observations else None

    def prune(self, now_ms: int) -> None:
        with self.lock:
            self.prune_locked(now_ms)
            self._expire_tracks_locked(now_ms)

    def prune_locked(self, now_ms: int) -> None:
        min_timestamp = now_ms - self.retention_ms
        while self.observations and self.observations[0].timestamp_ms < min_timestamp:
            self.observations.popleft()

    def _expire_tracks_locked(self, now_ms: int) -> None:
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now_ms - track.last_seen_ms > self.max_missing_ms
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)
