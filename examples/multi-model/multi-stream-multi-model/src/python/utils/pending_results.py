"""Stop a stream whose model stopped returning results while its decoder still sends frames.

Mirrors src/cpp/utils/pending_results.h, so both implementations fail the same way.
"""

from __future__ import annotations


class PendingResults:
    """A frame that arrives after the last result starts a deadline: `first_result_s` for the
    first result, while Core builds the pipeline, then `deadline_s`. A paused camera starts none."""

    def __init__(
        self, stream_index: int, model_path: str, deadline_s: float, first_result_s: float
    ) -> None:
        self.stream_index = stream_index
        self.model_path = model_path
        self.deadline_s = deadline_s
        self.first_result_s = first_result_s
        self.has_result = False
        self.pending_since: float | None = None

    def observe(self, frame_arrived: bool, result_arrived: bool, now_s: float) -> None:
        """Record one pull round, then raise when the pending result is late. A frame that
        arrives with a result in the same round starts no deadline: the result may be its own."""
        if result_arrived:
            self.has_result = True
            self.pending_since = None
            return
        if frame_arrived and self.pending_since is None:
            self.pending_since = now_s
        limit_s = self.deadline_s if self.has_result else self.first_result_s
        if self.pending_since is not None and now_s - self.pending_since > limit_s:
            raise RuntimeError(
                f"stream {self.stream_index}: model {self.model_path} returned no result for "
                f"{limit_s:g} s while its decoder sent frames"
            )
