"""Lightweight per-stream people tracker for the Python multi-camera example."""

from __future__ import annotations

from dataclasses import dataclass


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    width = max(0.0, xx2 - xx1)
    height = max(0.0, yy2 - yy1)
    inter = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    class_id: int
    last_frame_index: int
    missing_frames: int = 0


class PeopleTracker:
    """Greedy IoU tracker intended for the example's person-only detections."""

    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 2) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missing_frames = int(max_missing_frames)
        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}

    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(self, detections: list[dict], frame_index: int) -> list[TrackedDetection]:
        det_boxes = [
            (
                float(det["x1"]),
                float(det["y1"]),
                float(det["x2"]),
                float(det["y2"]),
            )
            for det in detections
        ]

        candidates: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks.keys())
        for track_id in track_ids:
            track = self._tracks[track_id]
            for det_index, bbox in enumerate(det_boxes):
                if int(detections[det_index]["class_id"]) != track.class_id:
                    continue
                iou = _iou_xyxy(track.bbox, bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, det_index))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        assignments: dict[int, int] = {}
        for _, track_id, det_index in candidates:
            if track_id in matched_tracks or det_index in matched_dets:
                continue
            matched_tracks.add(track_id)
            matched_dets.add(det_index)
            assignments[det_index] = track_id

        for det_index, track_id in assignments.items():
            det = detections[det_index]
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=det_boxes[det_index],
                score=float(det["score"]),
                class_id=int(det["class_id"]),
                last_frame_index=frame_index,
                missing_frames=0,
            )

        for det_index, det in enumerate(detections):
            if det_index in matched_dets:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=det_boxes[det_index],
                score=float(det["score"]),
                class_id=int(det["class_id"]),
                last_frame_index=frame_index,
                missing_frames=0,
            )
            assignments[det_index] = track_id

        expired: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_tracks or track_id in assignments.values():
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                expired.append(track_id)
        for track_id in expired:
            self._tracks.pop(track_id, None)

        tracked: list[TrackedDetection] = []
        for det_index, det in enumerate(detections):
            track_id = assignments[det_index]
            tracked.append(
                TrackedDetection(
                    track_id=track_id,
                    x1=det_boxes[det_index][0],
                    y1=det_boxes[det_index][1],
                    x2=det_boxes[det_index][2],
                    y2=det_boxes[det_index][3],
                    score=float(det["score"]),
                    class_id=int(det["class_id"]),
                )
            )
        return tracked
