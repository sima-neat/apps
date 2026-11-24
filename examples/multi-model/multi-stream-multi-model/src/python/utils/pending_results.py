"""Stop a stream whose model stopped returning results while its camera still sends frames.

Mirrors src/cpp/utils/pending_results.h, so both implementations fail the same way.
"""

from __future__ import annotations


class PendingResults:
    """The pending-result deadline of one stream.

    From the stream's first result on, a frame that arrives after the last result has a result
    pending. When no result comes within `deadline_s` of the first such frame, `check` raises
    with the stream and its model. Before the first result the decoder may still wait for a
    keyframe, so nothing is pending. A camera that pauses sends no frames, so nothing is
    pending and the stream waits without a limit.
    """

    def __init__(self, stream_index: int, model_path: str, deadline_s: float) -> None:
        self.stream_index = stream_index
        self.model_path = model_path
        self.deadline_s = deadline_s
        self.started = False
        self.pending_since: float | None = None

    def frame_arrived(self, now_s: float) -> None:
        if self.started and self.pending_since is None:
            self.pending_since = now_s

    def result_arrived(self) -> None:
        self.started = True
        self.pending_since = None

    def check(self, now_s: float) -> None:
        if self.pending_since is not None and now_s - self.pending_since > self.deadline_s:
            raise RuntimeError(
                f"stream {self.stream_index}: model {self.model_path} returned no result for "
                f"{self.deadline_s:g} s while its camera sent frames"
            )
