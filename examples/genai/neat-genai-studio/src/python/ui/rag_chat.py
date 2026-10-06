"""Answer from the user's documents on ``POST /v1/chat/completions``.

A client (the Insight GenAI tab, curl, another app) opts in with a non-OpenAI
request field, ``"neat_rag": true`` or ``{"k": 3}``. The Studio removes the
field before the request reaches the model server, searches the RAG database
with the last user message, and adds the matching passages as a system
message, so every client gets the same retrieval as the Studio's own chat.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

REQUEST_FIELD = "neat_rag"
DEFAULT_K = 3
MAX_K = 10
# Passages are trimmed so a large chunk can't crowd the question out of the
# model's context.
MAX_PASSAGE_CHARS = 1500

INSTRUCTION = (
    "Answer using the passages below from the user's documents when they are "
    "relevant, and say so when they don't contain the answer."
)


def rag_options(payload: dict) -> dict | None:
    """Remove the opt-in field from ``payload``; return ``{"k": n}`` when the
    request asks for documents, else None. Malformed values turn RAG off."""
    value = payload.pop(REQUEST_FIELD, None)
    if value is True:
        return {"k": DEFAULT_K}
    if isinstance(value, Mapping) and value.get("enabled", True):
        try:
            k = int(value.get("k", DEFAULT_K))
        except (TypeError, ValueError):
            k = DEFAULT_K
        return {"k": max(1, min(MAX_K, k))}
    return None


def message_text(content: Any) -> str:
    """The text of an OpenAI message ``content`` (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) for part in content
                        if isinstance(part, Mapping) and part.get("type") == "text").strip()
    return ""


def last_user_text(messages: Iterable[Mapping]) -> str:
    """The question to search with: the last user message's text, without the
    ``/no_think`` switch some clients prepend."""
    for message in reversed(list(messages or [])):
        if message.get("role") == "user":
            text = message_text(message.get("content"))
            return text.replace("/no_think", "").strip()
    return ""


def passage_source(hit: Mapping) -> dict:
    """Where a passage came from, for the client: file and section heading."""
    meta = hit.get("metadata") or {}
    headings = [str(meta[k]) for k in sorted(meta) if k.lower().startswith("header") and meta[k]]
    return {
        "source": str(meta.get("source") or meta.get("file") or ""),
        "heading": " › ".join(headings),
        "score": round(float(hit.get("score") or 0.0), 4),
    }


def with_passages(messages: list, hits: list) -> list:
    """``messages`` with the passages added as a system message right before
    the last user message (after the client's own system prompt, if any)."""
    if not hits:
        return list(messages)
    blocks = []
    for i, hit in enumerate(hits, 1):
        text = str(hit.get("content") or "").strip()[:MAX_PASSAGE_CHARS]
        where = passage_source(hit)
        label = " — ".join(p for p in (where["source"], where["heading"]) if p)
        blocks.append(f"[{i}]{' ' + label if label else ''}\n{text}")
    note = {"role": "system", "content": INSTRUCTION + "\n\n" + "\n\n".join(blocks)}
    out = list(messages)
    last_user = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=len(out))
    out.insert(last_user, note)
    return out


def sources_header(hits: list) -> str:
    """``X-RAG-Sources``: the passages used, as compact JSON (ASCII only, so it
    is a valid header value)."""
    return json.dumps([passage_source(h) for h in hits], ensure_ascii=True, separators=(",", ":"))
