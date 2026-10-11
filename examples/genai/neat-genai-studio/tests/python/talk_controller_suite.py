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
    ctrl._engine_lock = talk_controller.threading.Lock()
    ctrl._engine_choice_id = 0
    ctrl._engine_applied_id = 0
    ctrl.engine_chosen = False
    return ctrl


class SupertonicRetryTests(unittest.TestCase):
    def test_failure_is_reported_until_supertonic_loads(self):
        failed = _controller(st_error="accelerator-unavailable")
        self.assertEqual(failed.engine_failures(), {"supertonic": "accelerator-unavailable"})
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

        def first_retry():
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


    def test_an_engine_chosen_while_a_retry_loads_is_kept(self):
        ctrl = _controller(st_error="busy", last_attempt=100.0)
        ctrl.prefer_supertonic = False
        ctrl.engine_chosen = False
        ctrl.utterance_speed = 1.0
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})

        def slow_load(**_kwargs):
            ctrl.engine_chosen = True   # the user picks piper-tts during the load
            return loaded

        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS", side_effect=slow_load):
            ctrl.retry_supertonic()
        self.assertIs(ctrl.st, loaded)
        self.assertFalse(ctrl.prefer_supertonic)


    def test_a_choice_made_during_the_retry_check_waits_for_it(self):
        # Hold _engine_lock as set_voice_engine would while the retry finishes:
        # the retry's check-and-set must wait, then see the choice.
        ctrl = _controller(st_error="load-failed", last_attempt=100.0)
        ctrl.prefer_supertonic = False
        ctrl.utterance_speed = 1.0
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        ctrl._engine_lock.acquire()
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS", return_value=loaded):
            retry = talk_controller.threading.Thread(target=ctrl.retry_supertonic)
            retry.start()
            retry.join(0.2)
            self.assertTrue(retry.is_alive())        # blocked on _engine_lock
            ctrl.engine_chosen = True                # the user's choice lands first
            ctrl._engine_lock.release()
            retry.join(5)
        self.assertFalse(retry.is_alive())
        self.assertIs(ctrl.st, loaded)
        self.assertFalse(ctrl.prefer_supertonic)

    def test_failures_are_reported_as_fixed_reasons(self):
        reason = talk_controller.TalkController._load_failure_reason
        self.assertEqual(reason(RuntimeError("[dispatcher_unavailable] /opt/x busy")), "accelerator-unavailable")
        self.assertEqual(reason(RuntimeError("accelerator runtime is not available")), "accelerator-unavailable")
        self.assertEqual(reason(RuntimeError("Traceback ... /home/sima/x.py line 3")), "load-failed")

    def test_a_failed_load_records_only_the_reason(self):
        ctrl = _controller()
        with mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS",
                                  side_effect=RuntimeError("worker died: /home/sima/secret/path")), \
                mock.patch.object(talk_controller.supertonic_tts, "shutdown_worker"):
            ctrl._init_supertonic()
        self.assertIsNone(ctrl.st)
        self.assertEqual(ctrl.engine_failures(), {"supertonic": "load-failed"})

    def test_a_retry_that_finds_no_runtime_stops_retrying(self):
        ctrl = _controller(st_error="load-failed", last_attempt=100.0)
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=False), \
                mock.patch.object(talk_controller.supertonic_tts, "_supertonic_python", return_value=""), \
                mock.patch.object(talk_controller.supertonic_tts, "models_root", return_value=""):
            ctrl.retry_supertonic()
        self.assertIsNone(ctrl.st_error)
        self.assertEqual(ctrl.engine_failures(), {})

    def test_speech_api_says_load_failed_not_not_installed(self):
        ctrl = _controller(st_error="load-failed")
        with self.assertRaises(ctrl.EngineUnavailable) as raised:
            ctrl.engine_for_request("supertonic", "en")
        self.assertEqual(raised.exception.reason, "load-failed")
        self.assertIn("load-failed", ctrl.ENGINE_REFUSALS)
        with self.assertRaises(ctrl.EngineUnavailable) as raised:
            _controller().engine_for_request("supertonic", "en")
        self.assertEqual(raised.exception.reason, "not-installed")

    def test_choosing_supertonic_retries_a_failed_load(self):
        ctrl = _controller(st_error="load-failed", last_attempt=100.0)
        ctrl.browser_tts = False
        ctrl._preferred_engine_key = lambda: "supertonic"
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        ctrl.utterance_speed = 1.0
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0 + ctrl.ST_RETRY_INTERVAL_S), \
                mock.patch.object(talk_controller.supertonic_tts, "available", return_value=True), \
                mock.patch.object(talk_controller.supertonic_tts, "SupertonicTTS", return_value=loaded):
            self.assertTrue(ctrl.set_voice_engine("supertonic"))
        self.assertIs(ctrl.st, loaded)
        self.assertTrue(ctrl.prefer_supertonic and ctrl.engine_chosen)


    def test_choosing_supertonic_waits_for_a_retry_already_loading_it(self):
        # Another request is mid-retry: the cooldown has just moved, so this
        # choice does not retry itself, but must wait for that load to finish.
        ctrl = _controller(st_error="load-failed", last_attempt=100.0)
        ctrl.browser_tts = False
        ctrl._preferred_engine_key = lambda: "supertonic"
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        ctrl.st_lock.acquire()

        def finish_load():
            talk_controller.time.sleep(0.2)
            ctrl.st, ctrl.st_error = loaded, None
            ctrl.st_lock.release()

        loader = talk_controller.threading.Thread(target=finish_load)
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0):
            loader.start()
            chosen = ctrl.set_voice_engine("supertonic")
        loader.join(5)
        self.assertTrue(chosen)
        self.assertTrue(ctrl.prefer_supertonic and ctrl.engine_chosen)


    def test_speech_with_nothing_speakable_is_a_400_not_a_500(self):
        ctrl = _controller()
        with mock.patch.object(ctrl, "engine_for_request") as pick:
            for text in ("\U0001F44D\U0001F3FD", "**\u2728**"):
                with self.subTest(text=text), self.assertRaises(talk_controller.AudioApiError) as raised:
                    ctrl.tts_on_demand(text)
                self.assertEqual((raised.exception.status, raised.exception.param), (400, "input"))
        pick.assert_not_called()   # rejected before any engine is chosen or loaded


    def test_a_slow_older_engine_choice_does_not_override_a_newer_one(self):
        ctrl = _controller()
        ctrl.browser_tts = False
        ctrl.prefer_supertonic = False
        ctrl.prefer_piper_plus = False
        ctrl._preferred_engine_key = lambda: "x"
        newer_done = talk_controller.threading.Event()

        def slow_piper_plus_load():
            newer_done.wait(5)           # the newer "browser" choice completes meanwhile
            return True

        ctrl._ensure_piper_plus_loaded = slow_piper_plus_load
        older = talk_controller.threading.Thread(target=lambda: results.append(ctrl.set_voice_engine("piper-plus")))
        results = []
        older.start()
        talk_controller.time.sleep(0.1)
        self.assertTrue(ctrl.set_voice_engine("browser"))
        newer_done.set()
        older.join(5)
        self.assertEqual(results, [True])           # answered, but the stale choice was not applied
        self.assertTrue(ctrl.browser_tts)
        self.assertFalse(ctrl.prefer_piper_plus)


    def test_a_speech_request_waits_for_a_load_already_in_progress(self):
        ctrl = _controller(st_error="load-failed", last_attempt=100.0)
        loaded = mock.Mock(voice="M1", steps=8, languages={"en"})
        ctrl.st_lock.acquire()           # another request is mid-retry

        def finish_load():
            talk_controller.time.sleep(0.2)
            ctrl.st, ctrl.st_error = loaded, None
            ctrl.st_lock.release()

        loader = talk_controller.threading.Thread(target=finish_load)
        with mock.patch.object(talk_controller.time, "monotonic", return_value=100.0):
            loader.start()
            ctrl.retry_supertonic()      # cooldown not over: must wait, not return early
        loader.join(5)
        self.assertIs(ctrl.st, loaded)


    def test_an_invalid_newer_choice_does_not_cancel_a_valid_one_loading(self):
        ctrl = _controller()
        ctrl.browser_tts = False
        ctrl.prefer_supertonic = False
        ctrl.prefer_piper_plus = False
        ctrl._preferred_engine_key = lambda: "x"
        invalid_done = talk_controller.threading.Event()

        def slow_piper_plus_load():
            invalid_done.wait(5)
            return True

        ctrl._ensure_piper_plus_loaded = slow_piper_plus_load
        results = []
        valid = talk_controller.threading.Thread(target=lambda: results.append(ctrl.set_voice_engine("piper-plus")))
        valid.start()
        talk_controller.time.sleep(0.1)
        self.assertFalse(ctrl.set_voice_engine("not-an-engine"))
        invalid_done.set()
        valid.join(5)
        self.assertEqual(results, [True])
        self.assertTrue(ctrl.prefer_piper_plus)      # the valid choice was applied


if __name__ == "__main__":
    unittest.main()
