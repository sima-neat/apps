"""Harness self-tests for the scripted run that drives the applications' pull loops."""

from __future__ import annotations

import pytest

from tests.utils.fake_run import FakeRun

pytestmark = pytest.mark.unit


def test_outcomes_are_answered_in_order():
    run = FakeRun("timeout", ("sample", "frame"), "closed", ("error", "queue torn down"))

    assert run.pull("detections", 50) is None
    assert (run.running(), run.last_error()) == (True, "")
    assert run.pull("detections", 50) == "frame"
    assert (run.running(), run.last_error()) == (True, "")
    assert run.pull("detections", 50) is None
    assert (run.running(), run.last_error()) == (True, "")
    with pytest.raises(RuntimeError, match="queue torn down"):
        run.pull("detections", 50)
    assert (run.running(), run.last_error()) == (False, "queue torn down")
    assert run.pulls == [("detections", 50)] * 4


def test_a_closed_output_looks_like_a_timeout():
    timed_out = FakeRun("timeout")
    closed = FakeRun("closed")

    assert timed_out.pull("detections", 50) is None
    assert closed.pull("detections", 50) is None
    assert (closed.running(), closed.last_error()) == (timed_out.running(), timed_out.last_error())


def test_pulling_past_the_script_fails_instead_of_spinning():
    run = FakeRun("timeout")
    run.pull("detections", 50)

    with pytest.raises(AssertionError, match="beyond the scripted outcomes"):
        run.pull("detections", 50)


def test_close_and_stop_are_recorded():
    closed = FakeRun()
    stopped = FakeRun()

    closed.close()
    stopped.stop()

    assert closed.closed_by_app and stopped.closed_by_app
