"""Backend-only (headless) mode for Neat GenAI Studio: which paths the web
process serves when started with ``--backend-only``, and the opt-in CORS policy
for browser front ends on other origins (for example Insight).

Pure Python on purpose (no Flask import) so the unit suite covers every
decision; ``flask_app`` applies these in ``before_request`` / ``after_request``.
"""

from __future__ import annotations

from typing import Iterable, Mapping
from urllib.parse import urlsplit

# Path prefixes (ending in "/") and exact paths served in backend-only mode.
# Everything else (the web UI pages, the Studio's own chat/RAG/camera routes)
# answers 404 there.
API_PREFIXES = (
    "/v1/",          # OpenAI-compatible chat, audio speech/transcriptions/translations/voices
    "/audio/",       # the same audio routes without the /v1 prefix
    "/models/",      # model catalog, load/unload, ASR switch, logs, hub
    "/benchmark/",   # model benchmark runs
    "/tts/",         # preferred TTS engine
    "/piperplus/",   # Piper Plus voice selection
    "/supertonic/",  # Supertonic voice selection
    "/voices/",      # piper-tts voice selection
)
API_EXACT = ("/", "/health", "/voices", "/shutdown", "/favicon.ico")

# Paths that may answer cross-origin browser requests when CORS is enabled:
# the API surface, never /shutdown.
CORS_PREFIXES = ("/v1/", "/audio/", "/models/", "/benchmark/", "/tts/",
                 "/piperplus/", "/supertonic/", "/voices/")
CORS_EXACT = ("/health", "/voices")

CORS_METHODS = "GET, POST, OPTIONS"
CORS_MAX_AGE = "600"
# Response headers a cross-origin page may read (timings, engine, model).
EXPOSE_HEADERS = ("X-ASR-Model", "X-Task", "X-Elapsed-Time", "X-Engine", "X-Voice",
                  "X-Language", "X-Speed", "X-RTF", "X-Audio-Duration")


def _matches(path: str, prefixes: Iterable[str], exact: Iterable[str]) -> bool:
    path = path or "/"
    return path in exact or any(path.startswith(p) for p in prefixes)


def path_allowed(path: str) -> bool:
    """True when ``path`` is part of the backend-only API surface."""
    return _matches(path, API_PREFIXES, API_EXACT)


def cors_path(path: str) -> bool:
    """True when ``path`` may be called cross-origin (with CORS enabled)."""
    return _matches(path, CORS_PREFIXES, CORS_EXACT)


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]`` lower-cased, without a trailing slash or path;
    empty for anything that is not an http(s) origin."""
    parts = urlsplit((origin or "").strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def parse_cors_origins(raw: str | None):
    """``BACKEND_CORS_ORIGINS``: a comma/space separated list of origins, or
    ``*`` for any. Returns ``"*"`` or a tuple of normalized origins (empty:
    CORS off). Invalid entries are dropped."""
    text = (raw or "").strip()
    if not text:
        return ()
    items = [t for t in text.replace(",", " ").split() if t]
    if "*" in items:
        return "*"
    seen = []
    for item in items:
        origin = normalize_origin(item)
        if origin and origin not in seen:
            seen.append(origin)
    return tuple(seen)


def origin_allowed(origin: str, allowed) -> bool:
    """True when the browser ``Origin`` is on the allowlist."""
    normalized = normalize_origin(origin)
    if not normalized or not allowed:
        return False
    return allowed == "*" or normalized in allowed


def cors_headers(origin: str, request_headers: str | None = None) -> dict:
    """Headers for an allowed cross-origin response or preflight. The request's
    origin is echoed (never a literal ``*``) so credentials-free and
    credentialed callers both work, with ``Vary: Origin`` for caches."""
    return {
        "Access-Control-Allow-Origin": normalize_origin(origin),
        "Vary": "Origin",
        "Access-Control-Allow-Methods": CORS_METHODS,
        "Access-Control-Allow-Headers": (request_headers or "Content-Type").strip() or "Content-Type",
        "Access-Control-Max-Age": CORS_MAX_AGE,
        "Access-Control-Expose-Headers": ", ".join(EXPOSE_HEADERS),
    }


def health_payload(*, mode: str, version: str, status: Mapping | None,
                   engines: Iterable[Mapping], error: str | None = None) -> dict:
    """``GET /health``: what an integrating front end needs to know before it
    calls the API. ``status`` is the control API's /control/status (None when
    the model server is unreachable)."""
    reachable = bool(status) and not error
    status = status or {}
    catalog = status.get("catalog") or []
    chat_loaded = [m.get("name") for m in catalog
                   if m.get("loaded") and (m.get("type") or "chat") != "asr"]
    return {
        "ok": reachable,
        "mode": mode,
        "version": version,
        "model_server": {"reachable": reachable, "error": error},
        "asr_model": status.get("asrModel") or None,
        "chat_models_loaded": chat_loaded,
        "tts": {"engines": [{"key": e.get("key"), "loaded": bool(e.get("loaded"))}
                            for e in engines if e.get("key") != "browser"]},
    }
