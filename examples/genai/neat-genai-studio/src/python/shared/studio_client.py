"""Request-shaping helpers shared by the two GenAI Studio front ends.

The web UI (``ui/flask_app.py``) and the terminal client (``cli/main.py``) are
deliberately different presentations over the same model server and control API,
but a few request-side rules must stay identical between them. Those rules live
here so there is one implementation to reason about, not two that can drift.

Pure standard library, no framework imports, so both front ends — and host unit
tests — can import it without pulling in Flask or the model runtime.
"""

from __future__ import annotations

import http.client
import os
from pathlib import Path
from typing import Any

NO_THINK_SUFFIX = "/no_think"

MLA_RESET_ENV = "MLA_RESET"
RESET_REQUEST_ENV = "NEAT_RESET_REQUEST_FILE"


def apply_no_think(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with ``/no_think`` appended to the last user
    turn — the soft switch that disables reasoning on models such as Qwen3.

    The caller's message list and dictionaries are never mutated: only the
    modified message (and, for multipart content, the modified part) is copied,
    so the shared conversation history keeps its original text.
    """
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if not (isinstance(m, dict) and m.get("role") == "user"):
            continue
        m = dict(m)
        content = m.get("content")
        if isinstance(content, str):
            m["content"] = (content + " " + NO_THINK_SUFFIX).strip()
        elif isinstance(content, list):
            new = list(content)
            for j in range(len(new) - 1, -1, -1):
                part = new[j]
                if isinstance(part, dict) and part.get("type") == "text":
                    part = dict(part)
                    part["text"] = (part.get("text", "") + " " + NO_THINK_SUFFIX).strip()
                    new[j] = part
                    break
            else:
                new.append({"type": "text", "text": NO_THINK_SUFFIX})
            m["content"] = new
        out[i] = m
        break
    return out


def mla_reset_disabled() -> bool:
    """True when accelerator reset is turned off with ``MLA_RESET=0``."""
    return os.environ.get(MLA_RESET_ENV, "1") != "1"


def hand_reset_to_supervisor() -> str:
    """Ask the run.sh supervisor to reset a model server that cannot answer its
    control API, by writing the ``reset`` sentinel to the request file it polls.

    Returns ``"handed"`` when the request was written, ``"no_supervisor"`` when
    no request file is configured (the Studio was not started through run.sh),
    or ``"error"`` when the file could not be written. The caller renders the
    outcome for its own front end.
    """
    request_file = os.environ.get(RESET_REQUEST_ENV, "")
    if not request_file:
        return "no_supervisor"
    try:
        Path(request_file).write_text("reset\n", encoding="utf-8")
    except OSError:
        return "error"
    return "handed"


def is_reset_disconnect(reason: BaseException) -> bool:
    """True for the transport errors a server that replies and then exits
    produces — the only ones that mean an accelerator reset was accepted."""
    return isinstance(reason, (http.client.RemoteDisconnected, ConnectionResetError,
                               http.client.IncompleteRead, BrokenPipeError))
