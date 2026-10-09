"""Answer from the user's documents on ``POST /v1/chat/completions``.

A client (the Insight GenAI tab, curl, another app) opts in with a non-OpenAI
request field, ``"neat_rag": true`` or ``{"k": 3}``. The Studio removes the
field before the request reaches the model server, searches the RAG database
with the last user message, and adds the matching passages to that message,
so every client gets the same retrieval as the Studio's own chat.
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

# Plain wording and bare passage text: on a DevKit, Qwen3 0.6B copied
# numbered labels, headings and longer instructions into its answers, and
# answered correctly with this form (sources are reported in X-RAG-Sources).
INSTRUCTION = "Use this information from my documents to answer the question."


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
    """``messages`` with the passages placed in the last user message, ahead of
    the question, as the Studio's own chat does. A separate system message is
    not used: chat templates such as Qwen's honour only a leading system
    message, so one added mid-conversation would be ignored."""
    if not hits:
        return list(messages)
    passages = [str(hit.get("content") or "").strip()[:MAX_PASSAGE_CHARS] for hit in hits]
    context = INSTRUCTION + "\n\nInformation:\n" + "\n\n".join(p for p in passages if p) + "\n\nQuestion: "
    out = [dict(m) for m in messages]
    last_user = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=None)
    if last_user is None:
        return out + [{"role": "user", "content": context.rstrip()}]
    content = out[last_user].get("content")
    if isinstance(content, list):
        parts = [dict(p) for p in content]
        first_text = next((i for i, p in enumerate(parts) if p.get("type") == "text"), None)
        if first_text is None:
            parts.insert(0, {"type": "text", "text": context.rstrip()})
        else:
            parts[first_text]["text"] = context + str(parts[first_text].get("text", ""))
        out[last_user]["content"] = parts
    else:
        out[last_user]["content"] = context + str(content or "")
    return out


def sources_header(hits: list) -> str:
    """``X-RAG-Sources``: the passages used, as compact JSON (ASCII only, so it
    is a valid header value)."""
    return json.dumps([passage_source(h) for h in hits], ensure_ascii=True, separators=(",", ":"))


def search_failed_reply(database_exists: bool) -> tuple[str, int]:
    """The error message and HTTP status for a chat with documents whose
    search failed. Without a database (the documents were cleared, or none
    were ever built) the user has to upload or reset, so 409; otherwise the
    service is still starting or has stopped, so 503."""
    if not database_exists:
        return ("No documents are loaded on the board: they were cleared, or none were built. "
                "Upload a Markdown file or reset to the default document."), 409
    return ("The document search service is not available yet. "
            "Try again in a moment, or rebuild the documents."), 503
