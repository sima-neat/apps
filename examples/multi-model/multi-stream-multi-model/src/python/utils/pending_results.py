"""Stop a stream whose model stopped returning results while its decoder still sends frames.

Mirrors src/cpp/utils/pending_results.h, so both implementations fail the same way.
"""

from __future__ import annotations


class PendingResults:
    """A frame newer than the last result starts a deadline: `first_result_s` for the first
    result, while Core builds the pipeline, then `deadline_s`. A paused camera starts none."""

    def __init__(
        self, stream_index: int, model_path: str, deadline_s: float, first_result_s: float
    ) -> None:
        self.stream_index = stream_index
        self.model_path = model_path
        self.deadline_s = deadline_s
        self.first_result_s = first_result_s
        self.has_result = False
        self.last_result_frame_id = -1
        self.newest_frame_id = -1
        self.pending_since: float | None = None

    def observe(self, frame_id: int | None, result_frame_id: int | None, now_s: float) -> None:
        """Record one pull round, given the frame ID of the frame and of the result it pulled
        (None for no pull), then raise when the pending result is late. An ID of -1 is unknown."""
        if frame_id is not None:
            self.newest_frame_id = max(self.newest_frame_id, frame_id)
        if result_frame_id is not None:
            self.has_result = True
            self.last_result_frame_id = result_frame_id
            # A frame newer than this result, pulled in this or an earlier round, still waits.
            waiting = result_frame_id >= 0 and self.newest_frame_id > result_frame_id
            self.pending_since = now_s if waiting else None
        if frame_id is not None and self.pending_since is None:
            if frame_id >= 0 and self.last_result_frame_id >= 0:
                newer = frame_id > self.last_result_frame_id
            else:
                # With an unknown ID, a frame pulled with a result may be that result's own.
                newer = result_frame_id is None
            if newer:
                self.pending_since = now_s
        limit_s = self.deadline_s if self.has_result else self.first_result_s
        if self.pending_since is not None and now_s - self.pending_since > limit_s:
            raise RuntimeError(
                f"stream {self.stream_index}: model {self.model_path} returned no result for "
                f"{limit_s:g} s while its decoder sent frames"
            )
