#!/usr/bin/env python3
"""Pipeline chooser - the front door to the three detection pipelines.

Runs ON the DevKit next to the three control panels:

    Browser --http--> launcher.py (DevKit:8080)  "which pipeline?"
                          |
                          +--> pipeline-scale ui_server.py :8090
                          +--> pipeline-live  ui_server.py :8091
                          +--> pipeline-group ui_server.py :8092

Why a chooser exists at all: the three pipelines share ONE MLA and one set of
Insight channels, so only one may hold the hardware at a time. Starting the
scale detector while the group detector runs would fight over both. Selecting a
pipeline here stops the other two's detectors first, which is the difference
between "pick one" and "hope you remembered to stop the last one".

The page also carries the guidance that used to live only in conversation: what
each pipeline is, what it is good at, and what it costs you.
"""

from __future__ import annotations

import json
import os
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
WEB = HERE / "launcher.html"
PORT = 8080

# Each pipeline is identified by its CONFIG filename, never by the app path:
# scale and group run the SAME binary, so an argv match cannot tell them apart.
# The bracket keeps the pattern from matching the pgrep carrying it.
PIPELINES = {
    "scale": {"dir": "pipeline-scale", "port": 8090, "pattern": r"[s]cale-run\.yaml"},
    "live":  {"dir": "pipeline-live",  "port": 8091, "pattern": r"[l]ive-run\.yaml"},
    "group": {"dir": "pipeline-group", "port": 8092, "pattern": r"[g]roup[0-9]*-run\.yaml"},
}


def sh(cmd: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(["bash", "-lc", cmd], capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception:  # noqa: BLE001 - status probing is best effort
        return ""


def detector_running(name: str) -> bool:
    return bool(sh(f'pgrep -f "{PIPELINES[name]["pattern"]}" | head -1'))


def ui_running(name: str) -> bool:
    port = PIPELINES[name]["port"]
    return bool(sh(f'(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null && echo up'))


def stream_count(name: str) -> int:
    """Streams currently configured, read from that pipeline's saved state."""
    state = HERE / PIPELINES[name]["dir"] / "ui-state.json"
    if not state.exists():
        return 0
    try:
        data = json.loads(state.read_text())
    except json.JSONDecodeError:
        return 0
    if "groups" in data:                      # grouped pipeline
        return sum(len(g) for g in data.get("groups", []))
    return len(data.get("streams", []))


# --------------------------------------------------------------------------
# implementation language (Python or C++)
# --------------------------------------------------------------------------
#
# The toggle lives HERE, in the chooser, not in the three panels: it is one
# global choice and all three pipelines read the same file. Each pipeline fixes
# its own --mode (scale/group = fused, live = adaptive); this only selects which
# implementation of that mode runs.
LANGUAGES = ("python", "cpp")
LANGUAGE_FILE = HERE / "language"
EXAMPLE = HERE.parent
CPP_CANDIDATES = (
    EXAMPLE / "src" / "cpp" / "pre-built" / "adaptive-resolution-object-detector",
    EXAMPLE.parents[2] / "build" / "examples" / "object-detection"
    / "adaptive-resolution-object-detector" / "adaptive-resolution-object-detector",
)


def language() -> str:
    try:
        value = LANGUAGE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "python"
    return value if value in LANGUAGES else "python"


def cpp_binary():
    for candidate in CPP_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def set_language(value: str) -> dict:
    """Switch implementation language for every pipeline.

    Refuses to select C++ when no binary exists - otherwise the next Start would
    fail deep inside a pipeline with a confusing error instead of here, where the
    user can see why.
    """
    if value not in LANGUAGES:
        return {"error": f"language must be one of {LANGUAGES}"}
    if value == "cpp" and cpp_binary() is None:
        return {"error": "C++ binary not found - build it with ./build.sh --clean",
                "language": language()}
    # A detector already running was started in the OLD language; stop them all
    # so the next Start actually uses the new one.
    for cfg in PIPELINES.values():
        pids = sh(f'pgrep -f "{cfg["pattern"]}" | tr "\\n" " "')
        if pids:
            sh(f'kill -TERM {pids} 2>/dev/null || true; sleep 2; '
               f'kill -9 {pids} 2>/dev/null || true')
    LANGUAGE_FILE.write_text(value + "\n", encoding="utf-8")
    return {"ok": True, "language": value, "status": status()}


def status() -> dict:
    return {
        "language": language(),
        "cpp_available": cpp_binary() is not None,
        "pipelines": {
            name: {
                "detector_running": detector_running(name),
                "ui_running": ui_running(name),
                "streams": stream_count(name),
                "port": cfg["port"],
            }
            for name, cfg in PIPELINES.items()
        }
    }


def activate(name: str) -> dict:
    """Give one pipeline the hardware: stop the others' detectors, start its UI.

    Only DETECTORS are stopped - the other control panels stay reachable, so
    switching back is one click and does not need a shell.
    """
    if name not in PIPELINES:
        return {"error": f"unknown pipeline: {name}"}
    for other, cfg in PIPELINES.items():
        if other == name:
            continue
        pids = sh(f'pgrep -f "{cfg["pattern"]}" | tr "\\n" " "')
        if pids:
            sh(f'kill -TERM {pids} 2>/dev/null || true; sleep 2; '
               f'kill -9 {pids} 2>/dev/null || true')
    ui = HERE / PIPELINES[name]["dir"] / "ui.sh"
    if ui.exists():
        sh(f'bash "{ui}" start', timeout=90)
    return {"ok": True, "pipeline": name, "port": PIPELINES[name]["port"],
            "status": status()}


# Shared static assets (SiMa logo + the fonts Neat Insight itself uses). Served
# from here, not linked to Insight: a cross-origin link would break on every IP
# change and trip its self-signed certificate.
ASSETS = HERE / "web-assets"
_MIME = {".png": "image/png", ".woff2": "font/woff2", ".svg": "image/svg+xml",
         ".ico": "image/x-icon"}


def serve_asset(handler, name: str) -> bool:
    # Reduce to a bare filename, then prove the resolved path is still inside
    # ASSETS. The basename alone already stops ../ and absolute paths; resolving
    # and re-checking containment keeps that true if a symlink ever lands in
    # web-assets, and is the form the path-injection scanners recognise as
    # sanitised (the bare "/" and ".." rejects were sound but not provable).
    safe = os.path.basename(name)
    if not safe or safe in (".", ".."):
        return False
    base = os.path.realpath(ASSETS)
    resolved = os.path.realpath(os.path.join(base, safe))
    if resolved != base and not resolved.startswith(base + os.sep):
        return False
    path = Path(resolved)
    if not path.is_file():
        return False
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", _MIME.get(path.suffix, "application/octet-stream"))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.end_headers()
    handler.wfile.write(body)
    return True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not WEB.exists():
                self._json({"error": "launcher.html missing"}, 500)
                return
            body = WEB.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/assets/"):
            if not serve_asset(self, path[len("/assets/"):]):
                self._json({"error": "not found"}, 404)
        elif path == "/api/status":
            self._json(status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/activate", "/api/language"):
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n)) if n else {}
        except json.JSONDecodeError:
            body = {}
        if path == "/api/language":
            self._json(set_language(str(body.get("language", ""))))
            return
        self._json(activate(str(body.get("pipeline", ""))))


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"pipeline chooser on http://0.0.0.0:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
