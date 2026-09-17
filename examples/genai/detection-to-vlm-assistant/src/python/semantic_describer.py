"""Threaded one-shot appearance descriptions keyed by session-local track ID."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
import re
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class SemanticState:
    status: str
    description: str | None
    attempts: int


@dataclass(frozen=True)
class _DescriptionJob:
    track_id: int
    crop: Any


def normalize_appearance(raw: str) -> str:
    """Validate the model's complete comma-separated visual attributes."""
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty appearance description")
    value = lines[0].strip(" `\"'")
    value = re.sub(r"^(description|appearance)\s*:\s*", "", value, flags=re.I)
    value = value.strip(" `\"'")
    value = re.sub(r"[.!?;:]+$", "", value).strip(" `\"'")
    fields = [field.strip(" `\"'") for field in value.split(",") if field.strip()]
    if len(fields) < 2:
        raise ValueError("appearance description is not comma-separated")
    return ", ".join(fields)


class AppearanceDescriber:
    """Run at most two serialized VLM attempts for each active track."""

    def __init__(
        self,
        request_fn: Callable[[Any], str],
        on_update: Callable[[int, SemanticState, float | None], None],
        *,
        queue_size: int = 8,
        max_attempts: int = 2,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._request_fn = request_fn
        self._on_update = on_update
        self._queue: Queue[_DescriptionJob] = Queue(maxsize=queue_size)
        self._max_attempts = max_attempts
        self._states: dict[int, SemanticState] = {}
        self._active_tracks: set[int] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run, name="track-appearance-vlm", daemon=True
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._worker.start()

    def close(self) -> None:
        self._stop_event.set()
        if self._started:
            self._worker.join(timeout=2.0)

    def retain_tracks(self, active_track_ids: set[int]) -> None:
        with self._lock:
            self._active_tracks = set(active_track_ids)
            for track_id in list(self._states):
                if track_id not in self._active_tracks:
                    del self._states[track_id]

    def schedule(self, track_id: int, crop: Any) -> bool:
        with self._lock:
            if track_id not in self._active_tracks or track_id in self._states:
                return False
            try:
                self._queue.put_nowait(_DescriptionJob(track_id, crop))
            except Full:
                return False
            self._states[track_id] = SemanticState("pending", None, 0)
            return True

    def state_for(self, track_id: int) -> SemanticState | None:
        with self._lock:
            return self._states.get(track_id)

    def needs_description(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._active_tracks and track_id not in self._states

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: _DescriptionJob) -> None:
        with self._lock:
            state = self._states.get(job.track_id)
            if job.track_id not in self._active_tracks or state is None:
                return
            attempt = state.attempts + 1

        started_at = time.monotonic()
        try:
            description = normalize_appearance(self._request_fn(job.crop))
            latency_ms = (time.monotonic() - started_at) * 1000.0
        except Exception:
            self._handle_failure(job, attempt)
            return

        ready = SemanticState("ready", description, attempt)
        with self._lock:
            if job.track_id not in self._active_tracks:
                return
            self._states[job.track_id] = ready
        self._on_update(job.track_id, ready, latency_ms)

    def _handle_failure(self, job: _DescriptionJob, attempt: int) -> None:
        unavailable: SemanticState | None = None
        with self._lock:
            if job.track_id not in self._active_tracks:
                return
            if attempt < self._max_attempts:
                retry = SemanticState("pending", None, attempt)
                self._states[job.track_id] = retry
                try:
                    self._queue.put_nowait(job)
                except Full:
                    unavailable = SemanticState("unavailable", None, attempt)
                    self._states[job.track_id] = unavailable
            else:
                unavailable = SemanticState("unavailable", None, attempt)
                self._states[job.track_id] = unavailable
        if unavailable is not None:
            self._on_update(job.track_id, unavailable, None)
