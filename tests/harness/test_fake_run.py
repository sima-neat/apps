"""Harness self-tests for the scripted run that drives the applications' pull loops."""

from __future__ import annotations

import pytest

from tests.utils.fake_run import FakeRun

pytestmark = pytest.mark.unit


def test_outcomes_are_answered_in_order():
    run = FakeRun(
        "timeout",
        ("sample", "frame"),
        ("error", "queue torn down"),
        ("closed", "source reached EOS"),
    )

    assert run.pull("detections", 50) is None
    assert (run.running(), run.last_error()) == (True, "")
    assert run.pull("detections", 50) == "frame"
    assert (run.running(), run.last_error()) == (True, "")
    assert run.pull("detections", 50) is None
    assert (run.running(), run.last_error()) == (True, "queue torn down")
    assert run.pull("detections", 50) is None
    assert (run.running(), run.last_error()) == (False, "source reached EOS")
    assert run.pulls == [("detections", 50)] * 4


def test_a_closed_output_without_a_reason_has_no_error():
    run = FakeRun(("closed", ""))
    run.pull("detections", 50)

    assert (run.running(), run.last_error()) == (False, "")


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
