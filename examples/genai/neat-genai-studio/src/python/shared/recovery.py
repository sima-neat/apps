"""Classify model-load failures and request recovery after returning an error."""

import os
from pathlib import Path
import signal
import threading


def is_out_of_memory(detail: str) -> bool:
    text = detail.lower()
    return any(marker in text for marker in
               ('cannot allocate memory', 'out of memory', 'std::bad_alloc'))


def restart_callback():
    pid = os.environ.get('NEAT_RUN_PID', '')
    if not pid.isdigit() or int(pid) <= 1:
        return None

    def restart():
        try:
            os.kill(int(pid), signal.SIGUSR1)
        except ProcessLookupError:
            pass  # The system service recovers a killed supervisor.

    return restart


def restart_after_response(restart):
    # Give the HTTP relay time to send the error before its process is stopped.
    timer = threading.Timer(0.6, restart)
    timer.daemon = True
    timer.start()


def oom_kills():
    """OOM kills in this Studio's cgroup, or None when accounting is unavailable."""
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            if line.startswith('0::'):
                events = Path('/sys/fs/cgroup') / line[3:].lstrip('/') / 'memory.events'
                return int(dict(row.split() for row in events.read_text().splitlines())['oom_kill'])
    except (OSError, ValueError, KeyError):
        pass
    return None


def interrupted_load(name, before):
    after = oom_kills()
    out_of_memory = before is not None and after is not None and after > before
    if out_of_memory:
        message = f"Out of memory while loading '{name}'. The model did not fit into the available memory."
    else:
        message = f"Connection to the model server was lost while loading '{name}'; it may have run out of memory."

    restart = restart_callback()
    if restart is not None:
        message += ' Restarting GenAI Studio.'
        restart_after_response(restart)
    else:
        message += ' Restart GenAI Studio before loading another model.'
    return {'error': message, 'code': 'out_of_memory' if out_of_memory else 'model_server_lost',
            'restarting': restart is not None}
