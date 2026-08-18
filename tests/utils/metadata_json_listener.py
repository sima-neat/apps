"""Test-only UDP listener for Insight metadata JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import select
import socket
import time


@dataclass(frozen=True)
class MetadataJsonMessage:
    port: int
    payload: str
    frame_id: str
    timestamp_ms: int
    object_count: int


@dataclass(frozen=True)
class MetadataJsonResult:
    success: bool
    ports_with_valid_json: set[int] = field(default_factory=set)
    messages: list[MetadataJsonMessage] = field(default_factory=list)
    error: str = ""


class MetadataJsonListener:
    """Receive metadata JSON emitted by e2e examples."""

    def __init__(
        self,
        host: str,
        base_port: int,
        num_ports: int,
        metadata_type: str = "object-detection",
        data_array_key: str = "objects",
        require_all_ports: bool = False,
        minimum_items: int = 0,
    ) -> None:
        if num_ports <= 0:
            raise ValueError("num_ports must be > 0")
        if base_port <= 0:
            raise ValueError("base_port must be > 0")
        if minimum_items < 0:
            raise ValueError("minimum_items must be >= 0")

        self._metadata_type = metadata_type
        self._data_array_key = data_array_key
        self._require_all_ports = require_all_ports
        self._minimum_items = minimum_items
        self._sockets: dict[socket.socket, int] = {}
        try:
            for offset in range(num_ports):
                port = base_port + offset
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.setblocking(False)
                self._sockets[sock] = port
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for sock in self._sockets:
            sock.close()
        self._sockets.clear()

    def __enter__(self) -> "MetadataJsonListener":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def wait_for_messages(self, timeout_s: float) -> MetadataJsonResult:
        ports_with_valid_json: set[int] = set()
        messages: list[MetadataJsonMessage] = []
        last_error = "metadata timeout"
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            readable, _, _ = select.select(list(self._sockets.keys()), [], [], min(0.2, remaining))
            for sock in readable:
                payload, _ = sock.recvfrom(65536)
                port = self._sockets[sock]
                message, error = self._parse_message(port, payload)
                if message is None:
                    last_error = error
                    continue
                messages.append(message)
                if message.object_count >= self._minimum_items:
                    ports_with_valid_json.add(port)
                if self._success_reached(ports_with_valid_json):
                    return MetadataJsonResult(True, ports_with_valid_json, messages)

        missing = sorted(set(self._sockets.values()) - ports_with_valid_json)
        if missing:
            last_error = f"missing metadata on ports {missing}; last_error={last_error}"
        return MetadataJsonResult(False, ports_with_valid_json, messages, last_error)

    def _success_reached(self, ports_with_valid_json: set[int]) -> bool:
        if self._require_all_ports:
            return len(ports_with_valid_json) == len(self._sockets)
        return bool(ports_with_valid_json)

    def _parse_message(
        self, port: int, payload: bytes
    ) -> tuple[MetadataJsonMessage | None, str]:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            return None, f"json parse failed: {exc}"

        if not isinstance(parsed, dict):
            return None, "json root is not an object"
        if parsed.get("type") != self._metadata_type:
            return None, "missing or invalid type"
        timestamp = parsed.get("timestamp")
        if not isinstance(timestamp, int):
            return None, "missing or invalid timestamp"
        frame_id = parsed.get("frame_id")
        if not isinstance(frame_id, str):
            return None, "missing or invalid frame_id"
        data = parsed.get("data")
        if not isinstance(data, dict):
            return None, "missing or invalid data"
        objects = data.get(self._data_array_key)
        if not isinstance(objects, list):
            return None, f"missing or invalid data.{self._data_array_key}"

        return (
            MetadataJsonMessage(
                port=port,
                payload=payload.decode("utf-8", errors="replace"),
                frame_id=frame_id,
                timestamp_ms=timestamp,
                object_count=len(objects),
            ),
            "",
        )
