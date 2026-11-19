"""Stop an example on Ctrl-C, SIGTERM, or a dropped ssh session (SIGHUP).

Mirrors support/cpp/sima_examples/stop_signals.h.
"""

from __future__ import annotations

import signal
import threading


def stop_on_signals() -> threading.Event:
    """An event that SIGINT, SIGTERM, and SIGHUP set. Pull loops check it between pulls.

    The application may also set it, for example when a worker thread fails.
    """
    stop = threading.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signal_number, lambda *_: stop.set())
    return stop
