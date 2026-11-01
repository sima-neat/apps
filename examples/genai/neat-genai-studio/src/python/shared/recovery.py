"""Classify model-load failures and request recovery after returning an error."""

import os
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
            pass  # The launcher already exited; there is nobody left to signal.

    return restart


def restart_after_response(restart):
    # Give the HTTP relay time to send the error before its process is stopped.
    timer = threading.Timer(0.6, restart)
    timer.daemon = True
    timer.start()


def interrupted_load(name):
    message = f"Connection to the model server was lost while loading '{name}'; it may have run out of memory."

    restart = restart_callback()
    if restart is not None:
        message += ' Restarting GenAI Studio.'
        restart_after_response(restart)
    else:
        message += ' Restart GenAI Studio before loading another model.'
    return {'error': message, 'code': 'model_server_lost',
            'restarting': restart is not None}
