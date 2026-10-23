"""The Neat GenAI Studio backend's API surface: which paths the web process
serves, and the opt-in CORS policy for browser front ends on other origins
(for example Insight).

Pure Python on purpose (no Flask import) so the unit suite covers every
decision; ``studio_api`` applies these in ``before_request`` / ``after_request``.
"""

from __future__ import annotations

from typing import Iterable, Mapping
from urllib.parse import urlsplit

# Path prefixes (ending in "/") and exact paths the backend serves. Everything
# else answers 404.
API_PREFIXES = (
    "/v1/",          # OpenAI-compatible chat, audio speech/transcriptions/translations/voices
    "/audio/",       # the same audio routes without the /v1 prefix
    "/models/",      # model catalog, load/unload, ASR switch, logs, hub
    "/benchmark/",   # model benchmark runs
    "/tts/",         # preferred TTS engine
    "/piperplus/",   # Piper Plus voice selection
    "/supertonic/",  # Supertonic voice selection
    "/voices/",      # piper-tts voice selection
    "/rag/",         # document search (RAG): status, upload, reset, clear, search
    "/board-camera/",  # cameras attached to the board: list, snapshot
)
API_EXACT = ("/", "/health", "/voices", "/shutdown")

# Paths that may answer cross-origin browser requests when CORS is enabled:
# the API surface, never /shutdown.
CORS_PREFIXES = ("/v1/", "/audio/", "/models/", "/benchmark/", "/tts/",
                 "/piperplus/", "/supertonic/", "/voices/", "/rag/", "/board-camera/")
CORS_EXACT = ("/health", "/voices")

CORS_METHODS = "GET, POST, OPTIONS"
CORS_MAX_AGE = "600"
# Response headers a cross-origin page may read (timings, engine, model).
EXPOSE_HEADERS = ("X-ASR-Model", "X-Task", "X-Elapsed-Time", "X-Engine", "X-Voice",
                  "X-Language", "X-Speed", "X-RTF", "X-Audio-Duration",
                  "X-RAG-Hits", "X-RAG-Sources", "X-Context-Dropped", "X-Context-Window")


def _matches(path: str, prefixes: Iterable[str], exact: Iterable[str]) -> bool:
    path = path or "/"
    return path in exact or any(path.startswith(p) for p in prefixes)


def path_allowed(path: str) -> bool:
    """True when ``path`` is part of the backend API surface."""
    return _matches(path, API_PREFIXES, API_EXACT)


def cors_path(path: str) -> bool:
    """True when ``path`` may be called cross-origin (with CORS enabled)."""
    return _matches(path, CORS_PREFIXES, CORS_EXACT)


# Ports a browser omits when it serializes an Origin header.
_DEFAULT_PORTS = {"http": "80", "https": "443"}


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]`` lower-cased, without a trailing slash or path;
    empty for anything that is not an http(s) origin.

    The scheme's default port is dropped, because a browser never sends it: an
    allowlist entry written as ``https://host:443`` must still match the
    ``https://host`` the browser actually presents, or the request is refused
    despite being configured.
    """
    parts = urlsplit((origin or "").strip())
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return ""
    netloc = parts.netloc.lower()
    default = _DEFAULT_PORTS[scheme]
    if netloc.endswith(f":{default}"):
        netloc = netloc[: -(len(default) + 1)]
    return f"{scheme}://{netloc}"


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


def effective_cors_setting(environ: Mapping, configured: str | None) -> str:
    """The allowlist text in force: BACKEND_CORS_ORIGINS when it is set at all
    (an empty value turns CORS off for that run, even with a persisted
    allowlist), otherwise app.web.cors_origins."""
    if "BACKEND_CORS_ORIGINS" in environ:
        return str(environ.get("BACKEND_CORS_ORIGINS") or "")
    return str(configured or "")


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


# Routes the backend relays to the model server's control API as they are: the
# query (GET) or JSON object (POST) goes through, the reply comes back unchanged,
# and the control API applies the defaults. The control API is plain HTTP on
# loopback while Insight is an HTTPS origin, hence the relay.
# (method, backend path) -> (control API path, timeout in seconds)
CONTROL_ROUTES = {
    ("GET", "/models/status"): ("/control/status", 10),
    ("GET", "/models/catalog"): ("/control/catalog", 30),
    ("GET", "/models/logs"): ("/control/logs", 10),
    ("GET", "/models/card"): ("/control/card", 15),
    ("POST", "/models/load"): ("/control/load", 600),
    ("POST", "/models/unload"): ("/control/unload", 60),
    ("POST", "/models/delete"): ("/control/delete", 120),
    ("POST", "/benchmark/run"): ("/control/benchmark", 30),
    ("GET", "/benchmark/status"): ("/control/benchmark/status", 10),
    ("POST", "/benchmark/stop"): ("/control/benchmark/stop", 10),
}


NOT_A_JSON_OBJECT = "Request body must be a JSON object"


def control_payload(method: str, query: Iterable, body) -> tuple:
    """(params, json) for relaying a request to the control API: the query
    pairs for GET; for POST the JSON object, or {} without a body. Raises
    ValueError for a POST body that is not a JSON object."""
    if method == "GET":
        return list(query), None
    if body is None:
        return None, {}
    if not isinstance(body, dict):
        raise ValueError(NOT_A_JSON_OBJECT)
    return None, body


# Version of the backend API that integrating front ends (Insight) build on.
# Bump it on a breaking change to a route or payload they use.
API_VERSION = 1


def health_payload(*, mode: str, version: str, status: Mapping | None,
                   engines: Iterable[Mapping], error: str | None = None,
                   engine_failures: Mapping[str, str] | None = None,
                   rag_enabled: bool = False) -> dict:
    """``GET /health``: what an integrating front end needs to know before it
    calls the API. ``status`` is the control API's /control/status (None when
    the model server is unreachable). ``engine_failures`` maps an installed TTS
    engine that failed to load to its error; it is listed with ``loaded: false``
    and that error so a front end can say what is wrong."""
    reachable = bool(status) and not error
    status = status or {}
    catalog = status.get("catalog") or []
    chat_loaded = [m.get("name") for m in catalog
                   if m.get("loaded") and (m.get("type") or "chat") != "asr"]
    tts_engines = [{"key": e.get("key"), "loaded": bool(e.get("loaded"))}
                   for e in engines if e.get("key") != "browser"]
    listed = {e["key"] for e in tts_engines}
    tts_engines += [{"key": key, "loaded": False, "error": str(err)}
                    for key, err in (engine_failures or {}).items() if key not in listed]
    return {
        "ok": reachable,
        "api_version": API_VERSION,
        "mode": mode,
        "version": version,
        "model_server": {"reachable": reachable, "error": error},
        "asr_model": status.get("asrModel") or None,
        "chat_models_loaded": chat_loaded,
        "tts": {"engines": tts_engines},
        # Optional features, so a front end can hide what this board lacks.
        "features": {"rag": bool(rag_enabled), "benchmark": True, "board_camera": True},
    }
