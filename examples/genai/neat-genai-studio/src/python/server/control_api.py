#!/usr/bin/env python3
"""Local HTTP control API for runtime model management.

Runs inside the model-server process (pyneat environment) so it can call
``GenAIServer.add_model`` / ``remove_model`` on the live server. Implemented
with only the standard library so it needs no extra dependencies in that
environment. Bound to localhost; the UI process proxies to it.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from server.model_manager import ModelManager


class _ControlServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, manager: ModelManager):
        super().__init__(address, handler)
        self.manager = manager


class _ControlHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Silence default request logging (server already logs model events).
    def log_message(self, *args) -> None:  # noqa: D401
        return

    @property
    def manager(self) -> ModelManager:
        return self.server.manager  # type: ignore[attr-defined]

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The client (UI proxy / browser) stopped waiting and closed the
            # socket — e.g. a /models/status poll abandoned during a slow model
            # switch. There is nothing left to send; swallow it so a dead client
            # does not surface as a traceback (and so the error handler below
            # does not try to write a 500 to the same closed socket).
            self.close_connection = True

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/")
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path == "/control/status":
                self._send_json(self.manager.status())
            elif path == "/control/logs":
                after = int((query.get("after", ["0"])[0]) or 0)
                self._send_json(self.manager.load_logs(after))
            elif path == "/control/catalog":
                self._send_json({"catalog": self.manager.scan_catalog()})
            elif path == "/control/card":
                name = (query.get("name", [""])[0])
                self._send_json(self.manager.model_card(name))
            elif path == "/control/benchmark/status":
                self._send_json(self.manager.benchmark_status())
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            body = self._read_json()
            if path == "/control/load":
                self._send_json(self.manager.load(str(body.get("name", ""))))
            elif path == "/control/asr":
                self._send_json(self.manager.set_active_asr(str(body.get("name", ""))))
            elif path == "/control/unload":
                self._send_json(self.manager.unload(str(body.get("name", ""))))
            elif path == "/control/delete":
                self._send_json(self.manager.delete(str(body.get("name", ""))))
            elif path == "/control/rescan":
                self._send_json({"catalog": self.manager.scan_catalog()})
            elif path == "/control/benchmark":
                self._send_json(self.manager.benchmark_start(
                    num_samples=body.get("num_samples", 5),
                    max_new_tokens=body.get("max_new_tokens", 128),
                    prompt=str(body.get("prompt", "")),
                    model=str(body.get("model", "")),
                ))
            elif path == "/control/benchmark/stop":
                self._send_json(self.manager.benchmark_stop())
            else:
                self._send_json({"error": "not found"}, 404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:  # noqa: BLE001
            # Accelerator (MLA) load failures are transient/infra, not a bad
            # request — surface as 503 so the UI can message it distinctly.
            msg = str(exc)
            status = 503 if ("accelerator" in msg.lower() or "mla" in msg.lower()) else 500
            self._send_json({"error": msg}, status)


def serve_control_api(manager: ModelManager, host: str, port: int) -> ThreadingHTTPServer:
    """Start the control API on its own thread and return the server handle."""
    server = _ControlServer((host, port), _ControlHandler, manager)
    thread = threading.Thread(target=server.serve_forever, name="control-api", daemon=True)
    thread.start()
    return server
