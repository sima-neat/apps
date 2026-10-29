"""Exit-status assertions for applications the e2e harness starts and stops.

The streaming applications run until the harness has the frames it asked for
and then stops them with SIGINT, the way an operator would. Exiting cleanly on
that interrupt is part of the contract: the Python applications close their run
in a finally and exit 130, the C++ ones end their loop and exit 0. An
application that had to be escalated to SIGTERM or SIGKILL, or that died from
the signal instead of handling it, did not shut down; its close() never ran.
"""

from __future__ import annotations

import signal
import subprocess

# What a clean stop looks like: the C++ applications return 0 from main after
# their loop ends; Python's KeyboardInterrupt convention is 130.
CLEAN_AFTER_INTERRUPT = frozenset({0, 130})


def _tail(result: subprocess.CompletedProcess[str]) -> str:
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def assert_exited_cleanly(result: subprocess.CompletedProcess[str], what: str = "the application") -> None:
    """Pass when the application exited 0 on its own, or handled the harness's
    SIGINT and exited 0 or 130. Fail, naming the signal, when it had to be
    escalated or died from the signal (a negative return code)."""
    stopped = bool(getattr(result, "stopped_by_harness", False))
    if not stopped:
        assert result.returncode == 0, f"{what} exited with code {result.returncode}\n{_tail(result)}"
        return

    stop_signal = getattr(result, "stop_signal", None)
    signal_name = signal.Signals(stop_signal).name if stop_signal else "no signal"
    if stop_signal != signal.SIGINT:
        raise AssertionError(
            f"{what} ignored SIGINT and had to be stopped with {signal_name} "
            f"(exit {result.returncode}); its shutdown path never ran\n{_tail(result)}"
        )
    if result.returncode not in CLEAN_AFTER_INTERRUPT:
        died = result.returncode < 0
        raise AssertionError(
            f"{what} {'died from' if died else 'exited ' + str(result.returncode) + ' after'} SIGINT "
            f"instead of shutting down cleanly (expected exit 0 or 130)\n{_tail(result)}"
        )
