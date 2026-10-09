"""Keeping chat requests within what a model and the accelerator can take.

Two limits the runtime enforces without saying so:

* **The context window.** Each model is compiled for a fixed number of tokens
  (2048 for LFM2.5-230M and Gemma 3, 8192 for the LFM2.5 VL models). Past it,
  the runtime ends the reply at once with no text and no error. The Studio
  drops the oldest turns of a conversation until the request fits.
* **A busy accelerator.** On Platform 3.0 the MLA driver now and then refuses
  a job while several models run at once ("dispatch: no free bank", rc=-11).
  The job never started, so sending it again is safe.

There is no tokenizer here, so token counts are estimates that run high: about
4 characters a token (the board's LFM2.5-230M fits about 4.5 of English), one
a character for Chinese, Japanese and Korean, a few tokens per message for the
chat template, and a fixed allowance per picture.
"""
from __future__ import annotations

import re

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


def message_tokens(message: dict) -> int:
    """Estimated tokens of one chat message, in any of the content shapes the
    Studio sends (a string, or a list of text / image parts)."""
    content = message.get("content") if isinstance(message, dict) else None
    total = MESSAGE_OVERHEAD_TOKENS
    if isinstance(content, str):
        return total + text_tokens(content)
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            total += text_tokens(part.get("text", ""))
        elif kind in ("image", "image_url"):
            total += IMAGE_TOKENS
    return total


def request_tokens(messages: list) -> int:
    return sum(message_tokens(m) for m in messages or [])


def fit_to_window(messages: list, window: int | None) -> tuple[list, int]:
    """Drop the oldest turns until ``messages`` fit ``window`` tokens.

    System messages and the last message (the question being asked) always
    stay. Returns the messages to send and how many were left out; with no
    known window, or when only the system prompt and the question are left,
    the messages go as they are (the caller reports a reply that comes back
    empty).
    """
    messages = list(messages or [])
    if not window or window <= 0 or len(messages) < 2:
        return messages, 0
    budget = window - REPLY_RESERVE_TOKENS
    if request_tokens(messages) <= budget:
        return messages, 0
    last = messages[-1]
    head = [m for m in messages[:-1] if isinstance(m, dict) and m.get("role") == "system"]
    middle = [m for m in messages[:-1] if not (isinstance(m, dict) and m.get("role") == "system")]
    dropped = 0
    while middle and request_tokens(head + middle + [last]) > budget:
        middle.pop(0)
        dropped += 1
        # Don't leave an answer without its question at the start.
        if middle and isinstance(middle[0], dict) and middle[0].get("role") == "assistant":
            middle.pop(0)
            dropped += 1
    return head + middle + [last], dropped


def too_long_alone(messages: list, window: int | None) -> bool:
    """Whether the request is past the window even with every earlier turn left
    out, so the model will not answer it."""
    return bool(window) and request_tokens(messages) > window - REPLY_RESERVE_TOKENS


def accelerator_busy(text) -> bool:
    """Whether an error is the 3.0 driver refusing a job that never started."""
    return bool(_BUSY.search(str(text or "")))


def context_window_from_elfs(names) -> int | None:
    """The window a model was compiled for, from its stage file names: the
    language stages are built per KV-cache position (``..._cache_token2047_...``),
    so the last position plus one is the window."""
    positions = [int(m.group(1)) for n in names or [] for m in [re.search(r"cache_token(\d+)", str(n))] if m]
    return max(positions) + 1 if positions else None
