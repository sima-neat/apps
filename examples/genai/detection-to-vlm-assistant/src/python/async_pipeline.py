"""Bounded asynchronous model submission and result collection."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Callable


class AsyncModelPipeline:
    """Drive a ModelRunner with independent push and pull stages.

    Context is retained by frame ID until the matching model result arrives.
    The runtime may discard queued inputs under KeepLatest; bounded context
    storage mirrors that behavior without retaining stale camera frames.
    """

    def __init__(
        self,
        runner: Any,
        on_result: Callable[[Any, Any], None],
        *,
        pull_timeout_ms: int = 100,
        max_contexts: int = 16,
    ) -> None:
        if pull_timeout_ms <= 0:
            raise ValueError("pull_timeout_ms must be positive")
        if max_contexts <= 0:
            raise ValueError("max_contexts must be positive")
        self.runner = runner
        self.on_result = on_result
        self.pull_timeout_ms = int(pull_timeout_ms)
        self.max_contexts = int(max_contexts)
        self._contexts: OrderedDict[int, Any] = OrderedDict()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._started = False
        self._closed = False
        self._worker = threading.Thread(
            target=self._collect_results,
            name="async-model-results",
            daemon=True,
        )

    @property
    def submitted(self) -> int:
        with self._lock:
            return self._submitted

    @property
    def completed(self) -> int:
        with self._lock:
            return self._completed

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def start(self) -> None:
        if not self._started:
            self._worker.start()
            self._started = True

    def submit(self, frame_id: int, sample: Any, context: Any) -> bool:
        if self._closed:
            return False
        self.raise_if_failed()
        frame_id = int(frame_id)
        with self._lock:
            self._contexts[frame_id] = context
            while len(self._contexts) > self.max_contexts:
                self._contexts.popitem(last=False)
                self._dropped += 1
        accepted = bool(self.runner.push([sample]))
        with self._lock:
            if accepted:
                self._submitted += 1
            else:
                self._contexts.pop(frame_id, None)
                self._dropped += 1
        return accepted

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"asynchronous model pipeline failed: {error}") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        close_input = getattr(self.runner, "close_input", None)
        if callable(close_input):
            close_input()
        if self._started and self._worker.is_alive():
            self._worker.join(timeout=max(2.0, self.pull_timeout_ms / 1000 + 1.0))
        self.runner.close()

    def _collect_results(self) -> None:
        try:
            while not self._stop_event.is_set():
                result = self.runner.pull(timeout_ms=self.pull_timeout_ms)
                if result is None:
                    continue
                frame_id = int(getattr(result, "frame_id", -1))
                context = self._take_context(frame_id)
                if context is None:
                    continue
                self.on_result(context, result)
                with self._lock:
                    self._completed += 1
        except BaseException as exc:
            with self._lock:
                self._error = exc
            self._stop_event.set()

    def _take_context(self, frame_id: int) -> Any | None:
        with self._lock:
            if frame_id >= 0:
                context = self._contexts.pop(frame_id, None)
                stale = [key for key in self._contexts if key < frame_id]
                for key in stale:
                    self._contexts.pop(key, None)
                    self._dropped += 1
                return context
            if not self._contexts:
                return None
            _key, context = self._contexts.popitem(last=False)
            return context
