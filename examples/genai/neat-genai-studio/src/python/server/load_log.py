#!/usr/bin/env python3
"""Capture the model server's own stdout to surface granular model-load
progress and a live loading log in the UI.

pyneat/llima expose no load-progress API, but the accelerator runtime prints one
triple per ELF stage it loads (a model is dozens–hundreds of stages):

    Loading model <ABS/..._stage1_mla.elf>
    [mlatiming] mlashm_load_model <ABS/...> took <N> ms ok=1
    Done loading <ABS/..._stage1_mla.elf>

Counting the ``Done loading …_mla.elf`` lines against the number of ELFs on disk
is the only *real* progress signal available. Those lines are written by native
code straight to fd 1, so a Python-level ``redirect_stdout`` cannot see them — we
tee the file descriptor itself: a pipe replaces fd 1, and a reader thread writes
every byte back to the real stdout (so ``run.sh`` logging is unchanged) while
parsing the load lines.

Defensive by design:
  * a failed install leaves stdout untouched (full rollback);
  * the reader loop ALWAYS drains the pipe first, so a parse/tee error can never
    let the pipe fill and block the writer (which would hang the server);
  * disable entirely with ``STUDIO_LOAD_LOG_TAP=0``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

# The board prints absolute .elf paths; match the stage-load lifecycle lines.
_LOADING_RE = re.compile(r"^Loading model\s+(?P<path>.+\.elf)\s*$")
_DONE_RE = re.compile(r"^Done loading\s+(?P<path>.+\.elf)\s*$")
_TIMING_RE = re.compile(r"\[mlatiming\]\s+mlashm_load_model\s+(?P<path>.+?)\s+took\s+(?P<ms>\d+)\s+ms\s+ok=(?P<ok>\d)")


class LoadLogTap:
    """Tees fd 1, counts completed ELF-stage loads, and keeps a line ring buffer."""

    def __init__(self, max_lines: int = 600) -> None:
        self._lock = threading.Lock()
        self._loaded_count = 0                       # monotonic completed ELF loads
        self._lines: deque = deque(maxlen=max_lines)  # (seq, ts, text)
        self._seq = 0                                 # monotonic line cursor for the UI
        self._orig_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._installed = False

    # -- snapshot API (thread-safe) ---------------------------------------
    @property
    def loaded_count(self) -> int:
        return self._loaded_count

    def snapshot(self, after_seq: int = 0) -> dict:
        with self._lock:
            lines = [
                {"seq": s, "t": ts, "text": txt}
                for (s, ts, txt) in self._lines
                if s > after_seq
            ]
            return {"loadedCount": self._loaded_count, "seq": self._seq, "lines": lines}

    # -- install / uninstall ----------------------------------------------
    def install(self) -> bool:
        if self._installed:
            return True
        orig_fd = None
        try:
            orig_fd = os.dup(1)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 1)
            os.close(write_fd)
            self._orig_fd = orig_fd
            self._read_fd = read_fd
            self._thread = threading.Thread(target=self._run, name="load-log-tap", daemon=True)
            self._installed = True
            self._thread.start()
            return True
        except Exception:
            # Never leave stdout broken: restore fd 1 if we managed to save it.
            try:
                if orig_fd is not None:
                    os.dup2(orig_fd, 1)
                    os.close(orig_fd)
            except Exception:
                pass
            self._orig_fd = None
            self._read_fd = None
            self._installed = False
            return False

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        try:
            # Restoring fd 1 closes the pipe's last writer; the reader sees EOF.
            if self._orig_fd is not None:
                os.dup2(self._orig_fd, 1)
        except Exception:
            pass

    # -- reader thread -----------------------------------------------------
    def _run(self) -> None:
        buf = b""
        orig = self._orig_fd
        read_fd = self._read_fd
        while True:
            try:
                data = os.read(read_fd, 65536)
            except Exception:
                break
            if not data:
                break  # EOF: fd 1 was restored / all writers closed
            # Tee to the real stdout FIRST and unconditionally — draining is what
            # keeps the pipe from filling and blocking the native writer.
            try:
                if orig is not None:
                    os.write(orig, data)
            except Exception:
                pass
            try:
                buf += data
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    self._note_line(raw.decode("utf-8", "replace").rstrip("\r"))
                # Guard against an unterminated flood holding unbounded memory.
                if len(buf) > 1_000_000:
                    buf = buf[-4096:]
            except Exception:
                buf = b""  # a parse error must never stop the drain loop

    def _note_line(self, line: str) -> None:
        done = _DONE_RE.match(line)
        if not (done or _LOADING_RE.match(line) or _TIMING_RE.search(line)):
            return
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, round(time.monotonic(), 2), line))
            if done:
                self._loaded_count += 1
