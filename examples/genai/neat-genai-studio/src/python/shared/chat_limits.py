"""Chat completion errors and compiled context metadata."""
from __future__ import annotations

import json
import re
import time

_BUSY = re.compile(r"rc=-11|resource temporarily unavailable|no free bank|queued wait failed", re.I)


def chat_choice_has_output(choice: dict) -> bool:
    message = choice.get("delta") or choice.get("message") or {}
    return any(message.get(key) for key in ("content", "reasoning", "reasoning_content", "tool_calls"))


def sse_error(message: str) -> bytes:
    return ("data: " + json.dumps({"error": {"message": message}}) + "\n\ndata: [DONE]\n\n").encode()


def first_chat_event(response):
    """Hold initial events until output or an error, for the busy-device retry."""
    held = []
    lines = response.iter_lines(chunk_size=None)
    for line in lines:
        held.append(line)
        text = line.decode("utf-8", "replace").strip()
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if data == "[DONE]":
            break
        try:
            reply = json.loads(data)
        except ValueError:
            continue
        if reply.get("error"):
            error = reply["error"]
            return held, lines, str(error.get("message") if isinstance(error, dict) else error)
        if any(chat_choice_has_output(choice) for choice in reply.get("choices", [])):
            return held, lines, None
    return held, lines, None


def accelerator_busy(text) -> bool:
    """Whether an error is the 3.0 driver refusing a job that never started."""
    return bool(_BUSY.search(str(text or "")))


def whole_reply(first, reopen, pause: float = 0.4):
    """Read a non-streamed reply from the model server, retrying once when the
    accelerator was busy. ``first`` is the response already opened; ``reopen()``
    sends the request again. A busy accelerator refused the job before it
    started, so the retry is safe. Returns ``(status, content_type, body)``.
    ``reopen`` may raise; the caller reports the model server as unreachable."""
    resp = first
    for attempt in range(2):
        try:
            body = resp.content
            status = resp.status_code
            content_type = resp.headers.get("Content-Type", "application/json")
        finally:
            resp.close()
        if attempt or not accelerator_busy(body.decode("utf-8", "replace")):
            return status, content_type, body
        time.sleep(pause)
        resp = reopen()
    return status, content_type, body


def context_window_from_elfs(names) -> int | None:
    """The window a model was compiled for, from its stage file names: the
    language stages are built per KV-cache position (``..._cache_token2047_...``),
    so the last position plus one is the window."""
    positions = [int(m.group(1)) for n in names or [] for m in [re.search(r"cache_token(\d+)", str(n))] if m]
    return max(positions) + 1 if positions else None
