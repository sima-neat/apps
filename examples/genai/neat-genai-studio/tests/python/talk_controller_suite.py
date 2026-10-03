"""Supertonic load failures in TalkController: recorded, reported and retried.

Builds the controller without running its constructor (which loads real
engines) and stubs the loader, so no models or accelerator are needed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

UI_DIR = Path(__file__).resolve().parents[2] / "src" / "python" / "ui"
sys.path.insert(0, str(UI_DIR))

import talk_controller  # noqa: E402


def _controller(st=None, st_error=None, last_attempt=0.0):
    ctrl = talk_controller.TalkController.__new__(talk_controller.TalkController)
    ctrl.st = st
    ctrl.st_error = st_error
    ctrl._st_last_attempt = last_attempt
    ctrl.st_lock = talk_controller.threading.Lock()
    return ctrl


class SupertonicRetryTests(unittest.TestCase):
    def test_failure_is_reported_until_supertonic_loads(self):
        failed = _controller(st_error="accelerator runtime is not available")
        self.assertEqual(failed.engine_failures(), {"supertonic": "accelerator runtime is not available"})
        self.assertEqual(_controller(st=object(), st_error=None).engine_failures(), {})
        self.assertEqual(_controller().engine_failures(), {})

    def test_retries_a_failed_load_once_the_interval_has_passed(self):
        ctrl = _controller(st_error="busy", last_attempt=100.0)
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(ctrl, "_init_supertonic") as init:
            ctrl.retry_supertonic()
        init.assert_called_once()

    def test_does_not_retry_too_soon_or_without_a_failure(self):
        recent = _controller(st_error="busy", last_attempt=100.0)
        with mock.patch.object(talk_controller.time, "monotonic", return_value=110.0), \
                mock.patch.object(recent, "_init_supertonic") as init:
            recent.retry_supertonic()
        init.assert_not_called()
        for ctrl in (_controller(st=object()), _controller()):
            with mock.patch.object(ctrl, "_init_supertonic") as init:
                ctrl.retry_supertonic()
            init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
