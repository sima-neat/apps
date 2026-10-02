"""A stand-in for a Neat run whose pulls follow a script, for pull-loop unit tests.

After a pull returns no sample, every streaming application asks the run the same two
questions: is it still running, and does it have an error. A live run answers them one
way for a timeout, another for a closed output and a third for a runtime error. This
fake gives those answers on cue, so an application's pull loop can be driven through
each outcome without pyneat, a model or a stream.
"""

from __future__ import annotations


class FakeRun:
    """Answers pulls from a script of outcomes, one per call.

    Each outcome is ``"timeout"``, ``("closed", reason)``, ``("error", message)`` or
    ``("sample", value)``. A pull consumes the next outcome and returns the sample, or
    ``None`` for the other three; ``running()`` and ``last_error()`` then describe that
    outcome the way a live run would. Pulling past the end of the script fails the test
    instead of letting a loop spin.
    """

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self._state = "timeout"
        self._detail = ""
        self.pulls: list[tuple[str, int]] = []
        self.closed_by_app = False

    def pull(self, output_name: str, timeout_ms: int = 0):
        self.pulls.append((output_name, timeout_ms))
        if not self._outcomes:
            raise AssertionError(
                f"pull of {output_name!r} beyond the scripted outcomes: the loop kept going"
            )
        outcome = self._outcomes.pop(0)
        if outcome == "timeout":
            self._state, self._detail = "timeout", ""
            return None
        self._state, self._detail = outcome
        return self._detail if self._state == "sample" else None

    def running(self) -> bool:
        return self._state != "closed"

    def last_error(self) -> str:
        return self._detail if self._state in ("closed", "error") else ""

    def close(self) -> None:
        self.closed_by_app = True

    def stop(self) -> None:
        self.closed_by_app = True
