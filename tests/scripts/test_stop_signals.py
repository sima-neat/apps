"""Contract tests for support/python/sima_examples/stop_signals.py, the examples' clean stop."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.stop_signals import stop_on_signals

STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@pytest.fixture
def restore_handlers():
    saved = {number: signal.getsignal(number) for number in STOP_SIGNALS}
    yield
    for number, handler in saved.items():
        signal.signal(number, handler)


@pytest.mark.parametrize("number", STOP_SIGNALS, ids=lambda number: signal.Signals(number).name)
def test_each_stop_signal_sets_the_event(number, restore_handlers):
    stop = stop_on_signals()
    assert not stop.is_set()
    os.kill(os.getpid(), number)
    assert stop.wait(1.0), f"{signal.Signals(number).name} did not set the stop event"
