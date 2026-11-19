"""Receiver-side progress and throughput accounting for high-density e2e tests."""

from __future__ import annotations


class ReceivedMetadataTracker:
    def __init__(
        self,
        stream_count: int,
        warmup_per_stream: int,
        target_frames: int,
        initial_progress_timeout_s: float,
        stream_progress_timeout_s: float,
        start: float,
    ) -> None:
        if stream_count <= 0 or target_frames <= 0:
            raise ValueError("metadata tracker requires streams and a frame target")
        if initial_progress_timeout_s <= 0 or stream_progress_timeout_s <= 0:
            raise ValueError("metadata tracker timeouts must be positive")
        self._warmup_per_stream = warmup_per_stream
        self._target_frames = target_frames
        self._initial_progress_timeout_s = initial_progress_timeout_s
        self._stream_progress_timeout_s = stream_progress_timeout_s
        self._seen = [set() for _ in range(stream_count)]
        self.warmup_frames = [0] * stream_count
        self.measured_frames = [0] * stream_count
        self.useful_detection = [False] * stream_count
        self._last_progress = [start] * stream_count
        self.measurement_started = warmup_per_stream == 0
        self.complete = False
        self.total_measured = 0
        self._measurement_start = start
        self._measurement_end = start

    def observe(
        self,
        stream_index: int,
        frame_id: str,
        has_objects: bool,
        now: float,
    ) -> bool:
        if stream_index < 0 or stream_index >= len(self._seen):
            raise IndexError("metadata stream index is out of range")
        if not frame_id or self.complete or frame_id in self._seen[stream_index]:
            return False
        self._seen[stream_index].add(frame_id)
        self._last_progress[stream_index] = now
        self.useful_detection[stream_index] |= has_objects

        if not self.measurement_started:
            self.warmup_frames[stream_index] += 1
            return False

        self.measured_frames[stream_index] += 1
        self.total_measured += 1
        if self.total_measured == self._target_frames:
            self.complete = True
            self._measurement_end = now
        return self.complete

    @property
    def warmup_complete(self) -> bool:
        return min(self.warmup_frames) >= self._warmup_per_stream

    def start_measurement(self, now: float) -> None:
        if self.measurement_started:
            return
        if not self.warmup_complete:
            raise RuntimeError("metadata measurement cannot start before warmup completes")
        self.measurement_started = True
        self._measurement_start = now
        self._last_progress = [now] * len(self._last_progress)

    def stalled_streams(self, now: float) -> list[int]:
        timeout = (
            self._stream_progress_timeout_s
            if self.measurement_started
            else self._initial_progress_timeout_s
        )
        return [
            index
            for index, last_progress in enumerate(self._last_progress)
            if now - last_progress >= timeout
        ]

    @property
    def elapsed_s(self) -> float:
        return self._measurement_end - self._measurement_start if self.complete else 0.0
