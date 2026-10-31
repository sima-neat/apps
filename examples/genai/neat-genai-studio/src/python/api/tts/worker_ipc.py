"""Length-prefixed stdin/stdout framing for the TTS worker clients.

Both TTS clients — piper-tts (``pipertts.py``) and Supertonic
(``supertonic_tts.py``) — drive a persistent worker subprocess over the same
wire protocol, so the framing lives here once:

- a request is one JSON object followed by a newline, written to the worker's
  stdin;
- each response frame is a 1-byte status, a 4-byte big-endian length, then that
  many payload bytes.

Status codes: 0 = complete, 1 = error, 2 = audio chunk, 3 = fatal (the worker is
exiting). The higher-level request/stream flow — how each client spawns its
worker, and how it treats a fatal frame — stays in the client, because those
differ. Pure standard library; imported only on the client side (the UI venv),
never by the workers, which run in their own venvs and keep their own framing.
"""

from __future__ import annotations

import json
import struct

STATUS_COMPLETE = 0
STATUS_ERROR = 1
STATUS_CHUNK = 2
STATUS_FATAL = 3


def read_exact(stream, n: int, label: str = "worker") -> bytes:
    """Read exactly ``n`` bytes from ``stream`` or raise if the pipe closes."""
    chunks = []
    while n > 0:
        b = stream.read(n)
        if not b:
            raise RuntimeError(f"{label} closed the pipe")
        chunks.append(b)
        n -= len(b)
    return b"".join(chunks)


def read_frame(proc, label: str = "worker"):
    """Read one ``(status, payload)`` response frame from ``proc``'s stdout."""
    status = read_exact(proc.stdout, 1, label)[0]
    length = struct.unpack(">I", read_exact(proc.stdout, 4, label))[0]
    return status, read_exact(proc.stdout, length, label)


def send_request(proc, req) -> None:
    """Write one JSON request line to ``proc``'s stdin and flush it."""
    proc.stdin.write((json.dumps(req) + "\n").encode("utf-8"))
    proc.stdin.flush()
