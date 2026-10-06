"""A stand-in for a Neat run whose pulls follow a script, for pull-loop unit tests.

It answers the way pyneat's ``Run`` does. ``Run.pull(name, timeout_ms)`` returns ``None``
both when the wait times out and when the output has closed because its source ended,
and after a source ends the run still reports ``running()`` with no ``last_error()``.
A runtime error raises from ``pull`` itself. So a Python pull loop can tell a sample,
an empty pull and an error apart, but not a timeout from a closed output; that needs a
pull that reports its status, which pyneat does not have.
"""

from __future__ import annotations


class FakeRun:
    """Answers pulls from a script of outcomes, one per call.

    Each outcome is ``"timeout"``, ``"closed"``, ``("error", message)`` or
    ``("sample", value)``. A timeout and a closed output both return ``None`` and leave
    the run running with no error; an error raises ``RuntimeError(message)`` from the
    pull and leaves the run stopped with that error. Pulling past the end of the script
    fails the test instead of letting a loop spin.
    """

    def __init__(self, *outcomes) -> None:
        self._outcomes = list(outcomes)
        self._error = ""
        self.pulls: list[tuple[str, int]] = []
        self.closed_by_app = False

    def pull(self, output_name: str, timeout_ms: int = 0):
        self.pulls.append((output_name, timeout_ms))
        if not self._outcomes:
            raise AssertionError(
                f"pull of {output_name!r} beyond the scripted outcomes: the loop kept going"
            )
        outcome = self._outcomes.pop(0)
        if outcome in ("timeout", "closed"):
            return None
        kind, detail = outcome
        if kind == "error":
            self._error = detail
            raise RuntimeError(detail)
        return detail

    def running(self) -> bool:
        return not self._error

    def last_error(self) -> str:
        return self._error

    def close(self) -> None:
        self.closed_by_app = True

    def stop(self) -> None:
        self.closed_by_app = True
