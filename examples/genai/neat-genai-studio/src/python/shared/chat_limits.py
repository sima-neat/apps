"""Keeping chat requests within what a model and the accelerator can take.

Two limits the runtime enforces without saying so:

* **The context window.** Each model is compiled for a fixed number of tokens
  (2048 for LFM2.5-230M and Gemma 3, 8192 for the LFM2.5 VL models). Past it,
  the runtime ends the reply at once with no text and no error. The Studio
  drops the oldest turns of a conversation until the request fits.
* **A busy accelerator.** On Platform 3.0 the MLA driver now and then refuses
  a job while several models run at once ("dispatch: no free bank", rc=-11).
  The job never started, so sending it again is safe.

Text is counted with the model's own tokenizer (``devkit/tokenizer.json``,
which the catalog reports as ``tokenizerPath``) when it can be read here. Without
it, counts are estimates: about 4 characters a token (the board's LFM2.5-230M
fits about 4.5 of English), one a character for Chinese, Japanese and Korean.
An estimate can run low on unusual text, and the reply then comes back empty.
Either way, each message adds a few tokens for the chat template and each
picture a fixed allowance.
"""
from __future__ import annotations

import functools
import logging
import re
import time
from typing import Callable

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 6
IMAGE_TOKENS = 300
# Room left for the start of the answer; the runtime itself stops a reply
# that reaches the end of the window.
REPLY_RESERVE_TOKENS = 64

_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_BUSY = re.compile(r"rc=-11|resource temporarily unavailable|no free bank|queued wait failed", re.I)


def text_tokens(text: str) -> int:
    value = str(text or "")
    cjk = len(_CJK.findall(value))
    return cjk + -(-(len(value) - cjk) // CHARS_PER_TOKEN)


@functools.lru_cache(maxsize=16)
def tokenizer_counter(path: str | None) -> Callable[[str], int] | None:
    """A function counting the tokens of a text with the model's tokenizer
    file, or None when there is no path or it can't be read here (for example
    when the model server runs on another machine)."""
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 - fall back to the estimate
        logging.info("Token counts for %s are estimates: %s", path, exc)
        return None
    return lambda text: len(tokenizer.encode(str(text or ""), add_special_tokens=False).ids)


def message_tokens(message: dict, count: Callable[[str], int] | None = None) -> int:
    """Tokens of one chat message, in any of the content shapes the Studio
    sends (a string, or a list of text / image parts). ``count`` counts text
    with the model's tokenizer; without it, text is estimated."""
    count = count or text_tokens
    content = message.get("content") if isinstance(message, dict) else None
    total = MESSAGE_OVERHEAD_TOKENS
    if isinstance(content, str):
        return total + count(content)
    if not isinstance(content, list):
        # Malformed content: counted as text; the model server rejects it.
        return total + count("" if content is None else str(content))
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            total += count(part.get("text", ""))
        elif kind in ("image", "image_url"):
            total += IMAGE_TOKENS
    return total


def request_tokens(messages: list, count: Callable[[str], int] | None = None) -> int:
    return sum(message_tokens(m, count) for m in messages or [])


def fit_to_window(messages: list, window: int | None,
                  count: Callable[[str], int] | None = None) -> tuple[list, int]:
    """Drop the oldest turns until ``messages`` fit ``window`` tokens.

    System messages and the last message (the question being asked) always
    stay. Returns the messages to send and how many were left out; with no
    known window, or when only the system prompt and the question are left,
    the messages go as they are (the caller reports a reply that comes back
    empty).
    """
    if not isinstance(messages, list):
        return messages, 0          # malformed: left for the model server to reject
    messages = list(messages)
    if not window or window <= 0 or len(messages) < 2:
        return messages, 0
    budget = window - REPLY_RESERVE_TOKENS
    if request_tokens(messages, count) <= budget:
        return messages, 0
    last = messages[-1]
    head = [m for m in messages[:-1] if isinstance(m, dict) and m.get("role") == "system"]
    middle = [m for m in messages[:-1] if not (isinstance(m, dict) and m.get("role") == "system")]
    dropped = 0
    while middle and request_tokens(head + middle + [last], count) > budget:
        middle.pop(0)
        dropped += 1
        # Don't leave an answer without its question at the start.
        if middle and isinstance(middle[0], dict) and middle[0].get("role") == "assistant":
            middle.pop(0)
            dropped += 1
    return head + middle + [last], dropped


def too_long_alone(messages: list, window: int | None,
                   count: Callable[[str], int] | None = None) -> bool:
    """Whether the request is past the window even with every earlier turn left
    out, so the model will not answer it."""
    return (bool(window) and isinstance(messages, list)
            and request_tokens(messages, count) > window - REPLY_RESERVE_TOKENS)


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
