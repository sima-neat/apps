"""Prepare the Python path for pyneat, time the loop, and read pull outcomes."""

from __future__ import annotations

import glob
import sys
import time


def use_system_packages() -> None:
    """Put the system dist-packages first on sys.path when this Python does not see them yet.

    That is the case in a virtual environment created without --system-site-packages. Call it
    before importing pyneat or OpenCV. On the DevKit's ~/pyneat environment the system packages
    are already on the path, so nothing changes.
    """
    for path in glob.glob("/usr/lib/python3*/dist-packages"):
        if path not in sys.path:
            sys.path.insert(0, path)


def time_ms() -> float:
    return time.perf_counter() * 1000.0


def pull_result_has_sample(run, sample, output_name: str) -> bool:
    """True when the pull delivered a sample, False when it timed out.

    A run that stopped or reported an error raises instead, so the loop ends with the reason
    rather than waiting on a dead source.
    """
    if sample is not None:
        return True
    last_error_fn = getattr(run, "last_error", None)
    last_error = last_error_fn() if callable(last_error_fn) else ""
    running_fn = getattr(run, "running", None)
    running = running_fn() if callable(running_fn) else True
    if not running:
        message = f"{output_name} output closed unexpectedly"
        if last_error:
            message += f": {last_error}"
        raise RuntimeError(message)
    if last_error:
        raise RuntimeError(f"runtime error: {last_error}")
    return False
