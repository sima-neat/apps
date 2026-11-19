"""Stop an example on Ctrl-C, SIGTERM, or a dropped ssh session (SIGHUP).

Mirrors support/cpp/sima_examples/stop_signals.h.
"""

from __future__ import annotations

import signal
import threading


def stop_on_signals() -> threading.Event:
    """An event that SIGINT, SIGTERM, and SIGHUP set. Pull loops check it between pulls.

    A signal the process started with ignored stays ignored, so a run under `nohup` keeps
    going when its terminal closes. The application may also set the event, for example
    when a worker thread fails.
    """
    stop = threading.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if signal.getsignal(signal_number) is not signal.SIG_IGN:
            signal.signal(signal_number, lambda *_: stop.set())
    return stop
