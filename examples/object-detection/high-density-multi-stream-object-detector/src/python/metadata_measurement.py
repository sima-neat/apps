"""Successful metadata sends over a common post-warm-up interval."""


class MetadataMeasurement:
    def __init__(self, streams: int, warmup: int, target: int):
        self.target = target
        self.warmup = warmup
        self.ready = [warmup == 0] * streams
        self.frames = [0] * streams
        self.failures = [0] * streams
        self.start = None
        self.elapsed = 0.0
        self.total = 0

    def observe(
        self, stream: int, processed: int, sent: bool, failed: bool, now: float
    ) -> bool:
        if not self.target:
            return False
        if self.start is None:
            self.ready[stream] = processed >= self.warmup
            if all(self.ready):
                self.start = now
            return False
        self.frames[stream] += sent
        self.failures[stream] += failed
        self.total += sent
        self.elapsed = now - self.start
        return self.total >= self.target

    def summary(self) -> dict:
        return {
            "frames": self.total,
            "elapsed_s": self.elapsed,
            "aggregate_fps": self.total / self.elapsed,
            "per_stream_frames": self.frames,
            "per_stream_send_failures": self.failures,
        }
