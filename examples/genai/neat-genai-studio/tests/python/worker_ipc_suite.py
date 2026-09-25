"""Shared TTS worker framing (ui/worker_ipc.py).

Pure stdlib; drives the framing against fake in-memory streams, no worker
subprocess and no hardware. Collected through test_unit.py.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import unittest
from pathlib import Path

UI_PYTHON = Path(__file__).resolve().parents[2] / "src" / "python" / "ui"
if str(UI_PYTHON) not in sys.path:
    sys.path.insert(0, str(UI_PYTHON))

import worker_ipc as wi  # noqa: E402


class _ChunkedStream:
    """Stream whose read() hands back at most `step` bytes at a time, to prove
    read_exact reassembles a value split across several reads."""

    def __init__(self, data: bytes, step: int = 1):
        self._data = data
        self._pos = 0
        self._step = step

    def read(self, n: int) -> bytes:
        end = min(self._pos + min(n, self._step), self._pos + n, len(self._data))
        out = self._data[self._pos:end]
        self._pos = end
        return out


class _FakeProc:
    def __init__(self, stdout_bytes: bytes):
        self.stdout = io.BytesIO(stdout_bytes)
        self.stdin = io.BytesIO()
        self.flushed = False

    # BytesIO has no flush side effect we can see, so wrap stdin writes.
    def _wrap_stdin(self):
        real_flush = self.stdin.flush
        def flush():
            self.flushed = True
            return real_flush()
        self.stdin.flush = flush  # type: ignore[method-assign]


def _frame(status: int, payload: bytes) -> bytes:
    return bytes([status]) + struct.pack(">I", len(payload)) + payload


class ReadExactTests(unittest.TestCase):
    def test_reassembles_across_partial_reads(self):
        s = _ChunkedStream(b"abcdefgh", step=3)
        self.assertEqual(wi.read_exact(s, 8), b"abcdefgh")

    def test_raises_with_label_when_pipe_closes_early(self):
        s = _ChunkedStream(b"ab", step=1)
        with self.assertRaises(RuntimeError) as ctx:
            wi.read_exact(s, 4, "piper-tts worker")
        self.assertIn("piper-tts worker closed the pipe", str(ctx.exception))


class ReadFrameTests(unittest.TestCase):
    def test_parses_status_length_and_payload(self):
        proc = _FakeProc(_frame(wi.STATUS_CHUNK, b"\x00\x01audio"))
        status, payload = wi.read_frame(proc)
        self.assertEqual(status, wi.STATUS_CHUNK)
        self.assertEqual(payload, b"\x00\x01audio")

    def test_zero_length_payload(self):
        proc = _FakeProc(_frame(wi.STATUS_COMPLETE, b""))
        status, payload = wi.read_frame(proc)
        self.assertEqual((status, payload), (wi.STATUS_COMPLETE, b""))


class SendRequestTests(unittest.TestCase):
    def test_writes_one_json_line_and_flushes(self):
        proc = _FakeProc(b"")
        proc._wrap_stdin()
        wi.send_request(proc, {"cmd": "synthesize", "text": "hi"})
        written = proc.stdin.getvalue()
        self.assertTrue(written.endswith(b"\n"))
        self.assertEqual(json.loads(written.decode("utf-8")),
                         {"cmd": "synthesize", "text": "hi"})
        self.assertTrue(proc.flushed)


if __name__ == "__main__":
    unittest.main()
