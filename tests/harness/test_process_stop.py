"""Harness self-tests for how run_until_output_files stops an application.

The streaming e2e suites let the application run until the harness has the
frames it asked for, then stop it. Whether the application shuts down cleanly on
that stop is part of what a suite proves, so the fixture must stop it the way an
operator would, SIGINT first, escalate only if that is ignored, and report the
real exit status rather than a stand-in 0. These tests drive the fixture with
producers that handle the interrupt, ignore it, or finish on their own.
"""

from __future__ import annotations

import signal
import sys

import pytest

np = pytest.importorskip("numpy", reason="the harness self-tests build frames with NumPy")
cv2 = pytest.importorskip("cv2", reason="the harness self-tests decode frames with OpenCV")

from tests.utils.process_assertions import assert_exited_cleanly

pytestmark = pytest.mark.unit


# Writes decodable frames slowly enough that the harness stops it, and records
# that its shutdown path ran by writing a marker OUTSIDE the output directory,
# where the fixture's discard-unfinished-writes step cannot remove it.
_PRODUCER = """
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
marker = Path(sys.argv[2])
mode = sys.argv[3]

if mode == "ignore":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

try:
    for index in range(1000 if mode != "finish" else 2):
        image = np.full((48, 64, 3), (index * 37) % 255, dtype=np.uint8)
        cv2.imwrite(str(out / f"frame_{index}.jpg"), image)
        time.sleep(0.2)
except KeyboardInterrupt:
    marker.write_text("closed", encoding="utf-8")
    sys.exit(130)
sys.exit(0)
"""


def _run(run_until_output_files, tmp_path, mode, expected=4, timeout_s=60.0):
    producer = tmp_path / "producer.py"
    producer.write_text(_PRODUCER)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    marker = tmp_path / "closed.marker"
    result = run_until_output_files(
        [sys.executable, str(producer), str(output_dir), str(marker), mode],
        output_dir,
        expected,
        timeout_s,
        cwd=str(tmp_path),
    )
    return result, marker


class TestStoppingAnApplication:
    def test_an_application_that_handles_sigint_reports_its_real_exit_and_ran_its_shutdown(
        self, tmp_path, run_until_output_files
    ):
        result, marker = _run(run_until_output_files, tmp_path, "handle")

        assert result.stopped_by_harness is True
        assert result.stop_signal == signal.SIGINT
        assert result.returncode == 130
        assert marker.read_text(encoding="utf-8") == "closed"
        assert_exited_cleanly(result)

    def test_an_application_that_ignores_the_interrupt_is_escalated_and_reported_as_unclean(
        self, tmp_path, run_until_output_files
    ):
        result, marker = _run(run_until_output_files, tmp_path, "ignore")

        assert result.stopped_by_harness is True
        assert result.stop_signal == signal.SIGKILL
        assert result.returncode == -signal.SIGKILL
        assert not marker.exists()
        with pytest.raises(AssertionError, match="SIGKILL.*shutdown path never ran"):
            assert_exited_cleanly(result)

    def test_an_application_that_finishes_on_its_own_is_not_stopped(
        self, tmp_path, run_until_output_files
    ):
        result, marker = _run(run_until_output_files, tmp_path, "finish", expected=10)

        assert result.stopped_by_harness is False
        assert result.returncode == 0
        assert not marker.exists()
        assert_exited_cleanly(result)


class TestExitedCleanly:
    """The judgement, without a process: what the two flags and the code mean."""

    @staticmethod
    def _result(returncode, *, stopped, stop_signal=None):
        from tests.utils.pytest_fixtures import StoppedProcess

        return StoppedProcess(["app"], returncode, "", "", stopped_by_harness=stopped, stop_signal=stop_signal)

    @pytest.mark.parametrize("code", [0, 130])
    def test_zero_or_130_after_sigint_is_clean(self, code):
        assert_exited_cleanly(self._result(code, stopped=True, stop_signal=signal.SIGINT))

    def test_died_from_sigint_is_not_clean(self):
        with pytest.raises(AssertionError, match="died from SIGINT"):
            assert_exited_cleanly(self._result(-signal.SIGINT, stopped=True, stop_signal=signal.SIGINT))

    def test_a_different_exit_after_sigint_is_not_clean(self):
        with pytest.raises(AssertionError, match="exited 1 after SIGINT"):
            assert_exited_cleanly(self._result(1, stopped=True, stop_signal=signal.SIGINT))

    def test_escalation_to_sigterm_is_not_clean(self):
        with pytest.raises(AssertionError, match="SIGTERM"):
            assert_exited_cleanly(self._result(0, stopped=True, stop_signal=signal.SIGTERM))

    def test_an_unstopped_process_must_exit_zero(self):
        assert_exited_cleanly(self._result(0, stopped=False))
        with pytest.raises(AssertionError, match="exited with code 3"):
            assert_exited_cleanly(self._result(3, stopped=False))
