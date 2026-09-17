"""Temporal edge-clearance gate for stable person VLM crops."""

from __future__ import annotations

import math

BBox = tuple[float, float, float, float]


def overlapping_bboxes(boxes: list[BBox]) -> set[BBox]:
    """Return every box with positive-area intersection with another box."""
    overlapping: set[BBox] = set()
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            intersection_width = min(first[2], second[2]) - max(first[0], second[0])
            intersection_height = min(first[3], second[3]) - max(first[1], second[1])
            if intersection_width > 0.0 and intersection_height > 0.0:
                overlapping.add(first)
                overlapping.add(second)
    return overlapping


class FullPersonVisibilityGate:
    """Wait for a tracked box to clear entry-sensitive frame edges."""

    def __init__(
        self,
        *,
        edge_margin_ratio: float = 0.03,
        min_clear_frames: int = 5,
    ) -> None:
        if not math.isfinite(edge_margin_ratio) or not (
            0.0 <= edge_margin_ratio < 0.5
        ):
            raise ValueError("edge_margin_ratio must be in [0, 0.5)")
        if min_clear_frames < 1:
            raise ValueError("min_clear_frames must be >= 1")
        self.edge_margin_ratio = edge_margin_ratio
        self.min_clear_frames = min_clear_frames
        self._clear_frames: dict[int, int] = {}

    def retain_tracks(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._clear_frames):
            if track_id not in active_track_ids:
                del self._clear_frames[track_id]

    def reset(self, track_id: int) -> None:
        self._clear_frames[track_id] = 0

    def observe(
        self,
        track_id: int,
        bbox: BBox,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """Return true after the box clears the top and sides long enough."""
        x1, y1, x2, y2 = bbox
        margin_x = frame_width * self.edge_margin_ratio
        margin_y = frame_height * self.edge_margin_ratio
        clear = (
            frame_width > 0
            and frame_height > 0
            and x2 > x1
            and y2 > y1
            and x1 >= margin_x
            and y1 >= margin_y
            and x2 <= frame_width - margin_x
            and y2 <= frame_height
        )
        if not clear:
            self._clear_frames[track_id] = 0
            return False
        count = self._clear_frames.get(track_id, 0) + 1
        self._clear_frames[track_id] = count
        return count >= self.min_clear_frames
