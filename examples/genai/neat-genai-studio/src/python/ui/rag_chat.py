"""Answer from the user's documents on ``POST /v1/chat/completions``.

A client (the Insight GenAI tab, curl, another app) opts in with a non-OpenAI
request field, ``"neat_rag": true`` or ``{"k": 3}``. The Studio removes the
field before the request reaches the model server, searches the RAG database
with the last user message, and adds the matching passages to that message,
so every client gets the same retrieval as the Studio's own chat.

``chat_request_error`` checks the shape of every chat request first, with or
without ``neat_rag``, so the code after it can rely on that shape.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Any, Iterable, Mapping

REQUEST_FIELD = "neat_rag"
DEFAULT_K = 3
MAX_K = 10
# Passages are trimmed so a large chunk can't crowd the question out of the
# model's context.
MAX_PASSAGE_CHARS = 1500
# All passages together, as the CLI's chat caps them: the board's models have
# small fixed context windows, and the question must still fit after them.
MAX_CONTEXT_CHARS = 3000
# X-RAG-Sources stays well inside common proxy and client header limits.
MAX_HEADING_CHARS = 120
MAX_SOURCES_HEADER_BYTES = 4096

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
        except (TypeError, ValueError, OverflowError):   # OverflowError: 1e999 is infinity
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


def _part_error(part: Any, where: str) -> str | None:
    if not isinstance(part, Mapping) or not isinstance(part.get("type"), str):
        return f"{where} must be an object with a string type"
    if part["type"] == "text" and not isinstance(part.get("text"), str):
        return f"{where} of type text needs a string text"
    if part["type"] == "image_url":
        image = part.get("image_url")
        if not isinstance(image, Mapping) or not isinstance(image.get("url"), str):
            return f"{where} of type image_url needs image_url.url as a string"
    return None


def chat_request_error(payload: Any) -> tuple[str, str | None] | None:
    """Why a ``/v1/chat/completions`` body can't be handled, as ``(message,
    param)``, or None when its shape is sound: a JSON object whose
    ``messages`` is a non-empty list of objects with a string ``role`` and a
    ``content`` that is a string or a list of parts (``null`` only for an
    assistant message). The route answers 400 with it before anything reads
    the request. Other fields are left to the model server."""
    if not isinstance(payload, Mapping):
        return "The request body must be a JSON object", None
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list", "messages"
    for i, message in enumerate(messages):
        where = f"messages[{i}]"
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            return f"{where} must be an object with a string role", "messages"
        content = message.get("content")
        if content is None and message["role"] == "assistant":
            continue
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            return f"{where}.content must be a string or a list of parts", "messages"
        for j, part in enumerate(content):
            error = _part_error(part, f"{where}.content[{j}]")
            if error:
                return error, "messages"
    return None


# The search query goes to the RAG service in a URL, which its server limits
# (about 64 KiB, less for non-ASCII text); the embedder reads at most 512
# tokens anyway. The end is kept: that is where the question usually is.
MAX_QUERY_CHARS = 2000


def last_user_text(messages: Iterable[Mapping]) -> str:
    """The question to search with: the last user message's text, without the
    ``/no_think`` switch some clients prepend, at most ``MAX_QUERY_CHARS``."""
    for message in reversed(list(messages or [])):
        if message.get("role") == "user":
            text = message_text(message.get("content"))
            return text.replace("/no_think", "").strip()[-MAX_QUERY_CHARS:]
    return ""


def document_name(meta: Mapping) -> str:
    """The file a database was built from, from its build-metadata sidecar
    (``rag.inspect_db.read_rag_meta``): the base name of ``input``."""
    return PurePath(str((meta or {}).get("input") or "")).name


def passage_source(hit: Mapping, default_source: str = "") -> dict:
    """Where a passage came from, for the client: file and section heading.
    ``default_source`` names the file when the passage itself does not (the
    Studio's databases are built from one document)."""
    meta = hit.get("metadata") or {}
    headings = [str(meta[k]) for k in sorted(meta) if k.lower().startswith("header") and meta[k]]
    return {
        "source": str(meta.get("source") or meta.get("file") or default_source or ""),
        "heading": " › ".join(headings)[:MAX_HEADING_CHARS],
        "score": round(float(hit.get("score") or 0.0), 4),
    }


def fit_passages(hits: list) -> list:
    """The hits that fit the context budget, in search order, each trimmed to
    ``MAX_PASSAGE_CHARS`` and together at most ``MAX_CONTEXT_CHARS``. Only
    these are given to the model and reported as sources."""
    kept, total = [], 0
    for hit in hits or []:
        text = str(hit.get("content") or "").strip()[:MAX_PASSAGE_CHARS]
        if not text:
            continue
        room = MAX_CONTEXT_CHARS - total
        if len(text) > room:
            if room > 200:            # only a worthwhile remainder
                kept.append({**hit, "content": text[:room].rstrip() + "…"})
            break
        kept.append({**hit, "content": text})
        total += len(text)
    return kept


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
        parts = [dict(p) if isinstance(p, Mapping) else p for p in content]
        first_text = next((i for i, p in enumerate(parts)
                           if isinstance(p, dict) and p.get("type") == "text"), None)
        if first_text is None:
            parts.insert(0, {"type": "text", "text": context.rstrip()})
        else:
            parts[first_text]["text"] = context + str(parts[first_text].get("text", ""))
        out[last_user]["content"] = parts
    else:
        out[last_user]["content"] = context + str(content or "")
    return out


def sources_header(hits: list, default_source: str = "") -> str:
    """``X-RAG-Sources``: the passages used, as compact JSON (ASCII only, so it
    is a valid header value). Trailing entries are dropped to keep it within
    ``MAX_SOURCES_HEADER_BYTES``."""
    sources = [passage_source(h, default_source) for h in hits]
    while True:
        header = json.dumps(sources, ensure_ascii=True, separators=(",", ":"))
        if len(header) <= MAX_SOURCES_HEADER_BYTES or not sources:
            return header
        sources.pop()


def search_failed_reply(database_exists: bool, service_failed: bool = False) -> tuple[str, int]:
    """The error message and HTTP status for a chat with documents whose
    search failed. Without a database (the documents were cleared, or none
    were ever built) the user has to upload or reset, so 409; a service that
    exited won't recover by waiting, so 503 that says so; otherwise the
    service is still starting, so 503 to try again."""
    if not database_exists:
        return ("No documents are loaded on the board: they were cleared, or none were built. "
                "Upload a Markdown file or reset to the default document."), 409
    if service_failed:
        return ("The document search service stopped (see the Studio's log). "
                "Restart GenAI Studio, or reset the documents."), 503
    return ("The document search service is not available yet. "
            "Try again in a moment, or rebuild the documents."), 503


def other_database(served: str | None, expected: str) -> str | None:
    """The database the document search service serves when it is not this
    Studio's (``expected``), else None. A service that reports no path (not
    up, or an older service) is not treated as another database."""
    if not served:
        return None
    return None if Path(served).resolve() == Path(expected).resolve() else served
