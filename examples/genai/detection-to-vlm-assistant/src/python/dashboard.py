"""Browser dashboard state and HTTP server for the visual AI assistant."""

from __future__ import annotations

import base64
from collections import deque
import copy
from datetime import datetime, timezone
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import ssl
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

_WEBSOCKET_XOR_TABLES = tuple(
    bytes(value ^ key for value in range(256)) for key in range(256)
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DashboardState:
    """Thread-safe state shared by tracking, GenAI, and browser threads."""

    def __init__(
        self,
        *,
        title: str,
        model_name: str,
        target_fps: int,
        upload_max_width: int = 640,
    ) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._revision = 1
        self._detection_revision = 0
        self._closed = False
        self._phase = "starting"
        self._phase_detail = "Preparing the visual pipeline"
        self._started_at = time.time()
        self._frame_times: deque[float] = deque()
        self._upload_times: deque[float] = deque()
        self._last_metrics_publish = 0.0
        self._last_upload_publish = 0.0
        self._latest_frame: tuple[int, bytes] | None = None
        self._frame_sequence = 0
        self._latest_source_frame: tuple[int, int] | None = None
        self._seen_track_ids: set[int] = set()
        self._described_track_ids: set[int] = set()
        self._components = {
            "camera": {"state": "waiting", "detail": "Waiting for browser webcam"},
            "detector": {"state": "starting", "detail": "Loading YOLO"},
            "mla": {"state": "starting", "detail": "Initializing accelerator"},
            "vlm": {"state": "starting", "detail": "Waiting for model server"},
        }
        self._metrics: dict[str, Any] = {
            "frames": 0,
            "fps": 0.0,
            "camera_fps": 0.0,
            "current_tracks": 0,
            "tracks_created": 0,
            "tracks_described": 0,
            "vlm_latency_ms": None,
            "tracker_latency_ms": None,
            "last_frame_at": None,
        }
        self._detections: list[dict[str, Any]] = []
        self._source = {"width": 0, "height": 0}
        self._config = {
            "title": title,
            "model_name": model_name,
            "target_fps": int(target_fps),
            "upload_max_width": int(upload_max_width),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()

    def set_phase(self, phase: str, detail: str) -> None:
        with self._condition:
            self._phase = phase
            self._phase_detail = detail
            self._touch_locked()

    def set_component(self, name: str, state: str, detail: str) -> None:
        with self._condition:
            if name not in self._components:
                raise KeyError(f"unknown component: {name}")
            value = {"state": state, "detail": detail}
            if self._components[name] != value:
                self._components[name] = value
                self._touch_locked()

    def submit_frame(
        self,
        payload: bytes,
        *,
        source_epoch: int | None = None,
        source_sequence: int | None = None,
    ) -> tuple[int, bool]:
        now = time.monotonic()
        with self._condition:
            source_frame = None
            if source_epoch is not None and source_sequence is not None:
                source_frame = (int(source_epoch), int(source_sequence))
                if (
                    self._latest_source_frame is not None
                    and source_frame <= self._latest_source_frame
                ):
                    return self._frame_sequence, False
                self._latest_source_frame = source_frame
            self._frame_sequence += 1
            self._latest_frame = (self._frame_sequence, payload)
            self._upload_times.append(now)
            while self._upload_times and self._upload_times[0] < now - 5.0:
                self._upload_times.popleft()
            if len(self._upload_times) > 1:
                elapsed = self._upload_times[-1] - self._upload_times[0]
                self._metrics["camera_fps"] = round(
                    (len(self._upload_times) - 1) / elapsed, 1
                )
            self._components["camera"] = {
                "state": "online",
                "detail": "Live browser webcam",
            }
            if now - self._last_upload_publish >= 0.5:
                self._last_upload_publish = now
                self._touch_locked()
            else:
                self._condition.notify_all()
            return self._frame_sequence, True

    def wait_for_frame(
        self, after_sequence: int, timeout: float = 1.0
    ) -> tuple[int, bytes] | None:
        with self._condition:
            if (
                not self._closed
                and (
                    self._latest_frame is None
                    or self._latest_frame[0] <= after_sequence
                )
            ):
                self._condition.wait(timeout=timeout)
            if self._closed or self._latest_frame is None:
                return None
            if self._latest_frame[0] <= after_sequence:
                return None
            return self._latest_frame

    def record_frame(
        self,
        detections: list[dict[str, Any]],
        *,
        width: int,
        height: int,
        tracker_latency_ms: float | None = None,
    ) -> None:
        now = time.monotonic()
        with self._condition:
            self._metrics["frames"] += 1
            self._metrics["current_tracks"] = len(detections)
            for detection in detections:
                self._seen_track_ids.add(int(detection["track_id"]))
            self._metrics["tracks_created"] = len(self._seen_track_ids)
            self._metrics["tracks_described"] = len(self._described_track_ids)
            if tracker_latency_ms is not None:
                self._metrics["tracker_latency_ms"] = round(
                    float(tracker_latency_ms), 3
                )
            self._metrics["last_frame_at"] = _utc_now()
            self._source = {"width": int(width), "height": int(height)}
            self._detections = copy.deepcopy(detections)
            self._detection_revision += 1
            self._frame_times.append(now)
            while self._frame_times and self._frame_times[0] < now - 5.0:
                self._frame_times.popleft()
            if len(self._frame_times) > 1:
                elapsed = self._frame_times[-1] - self._frame_times[0]
                self._metrics["fps"] = round((len(self._frame_times) - 1) / elapsed, 1)
            if now - self._last_metrics_publish >= 0.5:
                self._last_metrics_publish = now
                self._touch_locked()
            else:
                self._condition.notify_all()

    def update_track_semantic(
        self,
        track_id: int,
        *,
        status: str,
        description: str | None,
        latency_ms: float | None,
    ) -> None:
        if status not in {"pending", "ready", "unavailable"}:
            raise ValueError(f"invalid semantic status: {status}")
        with self._condition:
            patched = False
            for detection in self._detections:
                if int(detection.get("track_id", -1)) != int(track_id):
                    continue
                detection["semantic_status"] = status
                detection["semantic_description"] = description
                patched = True
            if status == "ready":
                self._described_track_ids.add(int(track_id))
            self._metrics["tracks_described"] = len(self._described_track_ids)
            if latency_ms is not None:
                self._metrics["vlm_latency_ms"] = round(float(latency_ms))
            if patched:
                self._detection_revision += 1
            self._touch_locked()

    def wait_for_update(
        self, revision: int, timeout: float = 15.0
    ) -> tuple[int, dict[str, Any] | None, bool]:
        with self._condition:
            if not self._closed and revision == self._revision:
                self._condition.wait_for(
                    lambda: self._closed or revision != self._revision,
                    timeout=timeout,
                )
            if self._closed:
                return self._revision, None, True
            if revision == self._revision:
                return self._revision, None, False
            return self._revision, self._snapshot_locked(), False

    def wait_for_detection_update(
        self, revision: int, timeout: float = 15.0
    ) -> tuple[int, dict[str, Any] | None, bool]:
        """Wait for the next lightweight, per-inference detection update."""
        with self._condition:
            if not self._closed and revision == self._detection_revision:
                self._condition.wait_for(
                    lambda: self._closed
                    or revision != self._detection_revision,
                    timeout=timeout,
                )
            if self._closed:
                return self._detection_revision, None, True
            if revision == self._detection_revision:
                return self._detection_revision, None, False
            return (
                self._detection_revision,
                {
                    "revision": self._detection_revision,
                    "source": copy.deepcopy(self._source),
                    "detections": copy.deepcopy(self._detections),
                    "processed_at": self._metrics["last_frame_at"],
                },
                False,
            )

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "revision": self._revision,
            "phase": self._phase,
            "phase_detail": self._phase_detail,
            "started_at": datetime.fromtimestamp(
                self._started_at, timezone.utc
            ).isoformat(timespec="seconds"),
            "uptime_seconds": round(time.time() - self._started_at),
            "components": copy.deepcopy(self._components),
            "metrics": copy.deepcopy(self._metrics),
            "config": copy.deepcopy(self._config),
            "source": copy.deepcopy(self._source),
            "detections": copy.deepcopy(self._detections),
        }

    def _touch_locked(self) -> None:
        self._revision += 1
        self._condition.notify_all()


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tls_context: ssl.SSLContext | None = None
        super().__init__(*args, **kwargs)

    def get_request(self):
        client, address = super().get_request()
        if self.tls_context is None:
            return client, address
        try:
            return (
                self.tls_context.wrap_socket(
                    client,
                    server_side=True,
                    do_handshake_on_connect=False,
                ),
                address,
            )
        except Exception:
            client.close()
            raise

    def handle_error(self, request: Any, client_address: Any) -> None:
        _exc_type, exc, _traceback = sys.exc_info()
        if isinstance(exc, ssl.SSLError):
            return
        super().handle_error(request, client_address)


class DashboardServer:
    """Small dependency-free HTTP/SSE server for the dashboard."""

    def __init__(
        self,
        state: DashboardState,
        host: str,
        port: int,
        web_root: Path | None = None,
        max_frame_bytes: int = 4 * 1024 * 1024,
        tls_cert: Path | None = None,
        tls_key: Path | None = None,
        ca_cert: Path | None = None,
    ) -> None:
        self.state = state
        self.web_root = web_root or Path(__file__).resolve().parent / "web"
        self.max_frame_bytes = int(max_frame_bytes)
        self.ca_cert = ca_cert
        handler = self._handler_type()
        self.httpd = _DashboardHTTPServer((host, port), handler)
        if tls_cert is not None or tls_key is not None:
            if tls_cert is None or tls_key is None:
                raise ValueError("both tls_cert and tls_key are required")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(certfile=tls_cert, keyfile=tls_key)
            self.httpd.tls_context = context
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name="visual-ai-dashboard",
            daemon=True,
        )

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.state.close()
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _handler_type(self):
        state = self.state
        web_root = self.web_root
        max_frame_bytes = self.max_frame_bytes
        ca_cert = self.ca_cert
        static_files = {
            "/": (web_root / "index.html", "text/html; charset=utf-8"),
            "/assets/app.css": (web_root / "app.css", "text/css; charset=utf-8"),
            "/assets/app.js": (
                web_root / "app.js",
                "application/javascript; charset=utf-8",
            ),
        }
        if ca_cert is not None:
            static_files["/modalix-webcam-ca.crt"] = (
                ca_cert,
                "application/x-x509-ca-cert",
            )

        class Handler(BaseHTTPRequestHandler):
            server_version = "NeatVisualAI/1.0"
            protocol_version = "HTTP/1.1"

            def setup(self) -> None:
                if isinstance(self.request, ssl.SSLSocket):
                    self.request.settimeout(10.0)
                    self.request.do_handshake()
                    self.request.settimeout(None)
                super().setup()

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/api/frames":
                    self._frames_websocket(max_frame_bytes)
                    return
                if path in static_files:
                    file_path, content_type = static_files[path]
                    if not file_path.is_file():
                        self._json_error(HTTPStatus.NOT_FOUND, "dashboard asset missing")
                        return
                    self._send(file_path.read_bytes(), content_type)
                    return
                if path == "/api/state":
                    self._json(HTTPStatus.OK, state.snapshot())
                    return
                if path == "/api/health":
                    snapshot = state.snapshot()
                    self._json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "phase": snapshot["phase"],
                            "revision": snapshot["revision"],
                        },
                    )
                    return
                if path == "/api/events":
                    self._events()
                    return
                if path == "/api/detections":
                    self._detection_events()
                    return
                self._json_error(HTTPStatus.NOT_FOUND, "not found")

            def _frames_websocket(self, max_bytes: int) -> None:
                if self.headers.get("Upgrade", "").lower() != "websocket":
                    self._json_error(
                        HTTPStatus.UPGRADE_REQUIRED,
                        "frame ingest requires a WebSocket upgrade",
                    )
                    return
                key = self.headers.get("Sec-WebSocket-Key", "").strip()
                if not key:
                    self._json_error(HTTPStatus.BAD_REQUEST, "missing WebSocket key")
                    return
                origin = self.headers.get("Origin", "").rstrip("/")
                host = self.headers.get("Host", "")
                scheme = "https" if isinstance(self.request, ssl.SSLSocket) else "http"
                if origin and origin != f"{scheme}://{host}":
                    self._json_error(HTTPStatus.FORBIDDEN, "WebSocket origin rejected")
                    return
                accept = base64.b64encode(
                    hashlib.sha1(
                        (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                    ).digest()
                ).decode()
                self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                self.close_connection = True

                message = bytearray()
                message_opcode: int | None = None
                try:
                    while True:
                        frame = self._read_websocket_frame(max_bytes)
                        if frame is None:
                            return
                        final, opcode, payload = frame
                        if opcode == 0x8:
                            self._send_websocket_frame(0x8, payload[:125])
                            return
                        if opcode == 0x9:
                            self._send_websocket_frame(0xA, payload[:125])
                            continue
                        if opcode in (0x1, 0x2):
                            if message_opcode is not None:
                                return
                            message_opcode = opcode
                            message = bytearray(payload)
                        elif opcode == 0x0 and message_opcode is not None:
                            message.extend(payload)
                        else:
                            return
                        if len(message) > max_bytes:
                            self._send_websocket_frame(0x8, b"\x03\xf1frame too large")
                            return
                        if final:
                            if message_opcode != 0x2:
                                return
                            state.submit_frame(bytes(message))
                            message.clear()
                            message_opcode = None
                except (
                    BrokenPipeError,
                    ConnectionResetError,
                    EOFError,
                    OSError,
                    ValueError,
                ):
                    return

            def _read_websocket_frame(
                self, max_bytes: int
            ) -> tuple[bool, int, bytes] | None:
                header = self.rfile.read(2)
                if not header:
                    return None
                if len(header) != 2:
                    raise EOFError("incomplete WebSocket header")
                first, second = header
                final = bool(first & 0x80)
                opcode = first & 0x0F
                masked = bool(second & 0x80)
                length = second & 0x7F
                if length == 126:
                    length = int.from_bytes(self._read_exact(2), "big")
                elif length == 127:
                    length = int.from_bytes(self._read_exact(8), "big")
                if not masked or length > max_bytes:
                    raise ValueError("invalid WebSocket frame")
                mask = self._read_exact(4)
                encoded = self._read_exact(length)
                payload = bytearray(length)
                for offset, key in enumerate(mask):
                    payload[offset::4] = encoded[offset::4].translate(
                        _WEBSOCKET_XOR_TABLES[key]
                    )
                return final, opcode, bytes(payload)

            def _read_exact(self, length: int) -> bytes:
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise EOFError("incomplete WebSocket payload")
                return payload

            def _send_websocket_frame(self, opcode: int, payload: bytes) -> None:
                length = len(payload)
                header = bytearray([0x80 | opcode])
                if length < 126:
                    header.append(length)
                elif length <= 65535:
                    header.append(126)
                    header.extend(length.to_bytes(2, "big"))
                else:
                    header.append(127)
                    header.extend(length.to_bytes(8, "big"))
                self.wfile.write(bytes(header) + payload)
                self.wfile.flush()

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/api/frame":
                    self._frame(max_frame_bytes)
                    return
                self._json_error(HTTPStatus.NOT_FOUND, "not found")

            def _frame(self, max_bytes: int) -> None:
                content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type not in {"image/jpeg", "image/png"}:
                    self._json_error(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        "frame must be image/jpeg or image/png",
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > max_bytes:
                    self._json_error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"frame must contain 1 to {max_bytes} bytes",
                    )
                    return
                payload = self.rfile.read(length)
                if len(payload) != length:
                    self._json_error(HTTPStatus.BAD_REQUEST, "incomplete frame body")
                    return
                try:
                    source_epoch = int(self.headers["X-Frame-Epoch"])
                    source_sequence = int(self.headers["X-Frame-Sequence"])
                except (KeyError, TypeError, ValueError):
                    source_epoch = None
                    source_sequence = None
                sequence, accepted = state.submit_frame(
                    payload,
                    source_epoch=source_epoch,
                    source_sequence=source_sequence,
                )
                self._json(
                    HTTPStatus.ACCEPTED,
                    {"accepted": accepted, "sequence": sequence},
                )

            def log_message(self, fmt: str, *args: Any) -> None:
                if args and str(args[1]).startswith("5"):
                    super().log_message(fmt, *args)

            def _events(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    revision = int(self.headers.get("Last-Event-ID", "0") or 0)
                except ValueError:
                    revision = 0
                try:
                    while True:
                        revision, snapshot, closed = state.wait_for_update(revision)
                        if closed:
                            break
                        if snapshot is None:
                            self.wfile.write(b": keepalive\n\n")
                        else:
                            payload = json.dumps(snapshot, separators=(",", ":"))
                            message = f"id: {revision}\ndata: {payload}\n\n".encode()
                            self.wfile.write(message)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _detection_events(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    revision = int(self.headers.get("Last-Event-ID", "0") or 0)
                except ValueError:
                    revision = 0
                try:
                    while True:
                        revision, update, closed = state.wait_for_detection_update(
                            revision
                        )
                        if closed:
                            break
                        if update is None:
                            self.wfile.write(b": keepalive\n\n")
                        else:
                            payload = json.dumps(update, separators=(",", ":"))
                            message = f"id: {revision}\ndata: {payload}\n\n".encode()
                            self.wfile.write(message)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

            def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode()
                self._send(body, "application/json; charset=utf-8", status)

            def _json_error(self, status: HTTPStatus, message: str) -> None:
                self._json(status, {"error": message})

            def _send(
                self,
                body: bytes,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Permissions-Policy", "camera=(self)")
                self.end_headers()
                self.wfile.write(body)

        return Handler
