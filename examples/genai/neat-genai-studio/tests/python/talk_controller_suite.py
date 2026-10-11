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


    def test_waiters_do_not_retry_after_another_request_retried(self):
        # Two requests passed the deadline check; the first retried (and moved
        # the attempt time) while the second waited for the lock.
        ctrl = _controller(st_error="busy", last_attempt=100.0)
        now = 100.0 + ctrl.ST_RETRY_INTERVAL_S

        def first_retry(prefer=True):
            ctrl._st_last_attempt = now

        with mock.patch.object(talk_controller.time, "monotonic", return_value=now), \
                mock.patch.object(ctrl, "_init_supertonic", side_effect=first_retry) as init:
            ctrl.retry_supertonic()
            ctrl.retry_supertonic()
        init.assert_called_once()

    def test_a_retry_keeps_the_voice_engine_the_user_chose(self):
        ctrl = _controller(st_error="busy", last_attempt=100.0)
        ctrl.prefer_supertonic = False   # the user picked piper-tts meanwhile
        ctrl.engine_chosen = True
        ctrl.utterance_speed = 1.0
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS", return_value=loaded):
            ctrl.retry_supertonic()
        self.assertIs(ctrl.st, loaded)
        self.assertFalse(ctrl.prefer_supertonic)


    def test_a_retry_restores_supertonic_when_no_engine_was_chosen(self):
        # Startup failed, so Piper spoke meanwhile only because Supertonic was
        # missing; once it loads it is the default again.
        ctrl = _controller(st_error="busy", last_attempt=100.0)
        ctrl.prefer_supertonic = False
        ctrl.engine_chosen = False
        ctrl.utterance_speed = 1.0
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS", return_value=loaded):
            ctrl.retry_supertonic()
        self.assertTrue(ctrl.prefer_supertonic)


if __name__ == "__main__":
    unittest.main()
