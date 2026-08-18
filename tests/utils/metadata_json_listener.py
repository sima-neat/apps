"""Test-only UDP listener for Insight metadata JSON."""

from __future__ import annotations

import json
import select
import socket
import time
from dataclasses import dataclass, field
from typing import Self

_MAX_DATAGRAM_PAYLOAD = 1200
_CHUNK_HEADER_SIZE = 12
_MAX_CHUNK_PAYLOAD = _MAX_DATAGRAM_PAYLOAD - _CHUNK_HEADER_SIZE
_MAX_LOGICAL_PAYLOAD = 65507
_MAX_CHUNK_COUNT = 56


@dataclass
class _PartialMessage:
    chunk_count: int
    chunks: dict[int, bytes]
    size: int
    updated_at: float


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
        min_data_items_per_port: int = 0,
        chunk_expiry_s: float = 1.0,
        max_inflight_messages: int = 128,
    ) -> None:
        if num_ports <= 0:
            raise ValueError("num_ports must be > 0")
        if base_port <= 0:
            raise ValueError("base_port must be > 0")
        if (
            chunk_expiry_s <= 0
            or max_inflight_messages <= 0
            or min_data_items_per_port < 0
        ):
            raise ValueError("chunk reassembly limits must be > 0")

        self._metadata_type = metadata_type
        self._data_array_key = data_array_key
        self._require_all_ports = require_all_ports
        self._min_data_items_per_port = min_data_items_per_port
        self._chunk_expiry_s = chunk_expiry_s
        self._max_inflight_messages = max_inflight_messages
        self._partial: dict[tuple[int, tuple, int], _PartialMessage] = {}
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

    def __enter__(self) -> Self:
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
            readable, _, _ = select.select(
                list(self._sockets.keys()), [], [], min(0.2, remaining)
            )
            for sock in readable:
                payload, sender = sock.recvfrom(65536)
                port = self._sockets[sock]
                payload, error = self._accept_datagram(port, sender, payload)
                if payload is None:
                    if error:
                        last_error = error
                    continue
                message, error = self._parse_message(port, payload)
                if message is None:
                    last_error = error
                    continue
                messages.append(message)
                if message.object_count >= self._min_data_items_per_port:
                    ports_with_valid_json.add(port)
                if self._success_reached(ports_with_valid_json):
                    return MetadataJsonResult(True, ports_with_valid_json, messages)

        missing = sorted(set(self._sockets.values()) - ports_with_valid_json)
        if missing:
            last_error = f"missing metadata on ports {missing}; last_error={last_error}"
        return MetadataJsonResult(False, ports_with_valid_json, messages, last_error)

    def _expire_partial(self, now: float) -> None:
        expired = [
            key
            for key, state in self._partial.items()
            if now - state.updated_at >= self._chunk_expiry_s
        ]
        for key in expired:
            del self._partial[key]

    def _accept_datagram(
        self, port: int, sender: tuple, datagram: bytes
    ) -> tuple[bytes | None, str]:
        now = time.monotonic()
        self._expire_partial(now)
        if not datagram or datagram[0] != 0x4E:
            return datagram, ""
        if len(datagram) < _CHUNK_HEADER_SIZE:
            return None, "invalid chunk header: datagram is too short"
        if datagram[1] != 0x01:
            return None, "invalid chunk header: unsupported version"

        message_id = int.from_bytes(datagram[2:10], byteorder="big")
        chunk_index = datagram[10]
        chunk_count = datagram[11]
        chunk = datagram[_CHUNK_HEADER_SIZE:]
        if not 2 <= chunk_count <= _MAX_CHUNK_COUNT or chunk_index >= chunk_count:
            return None, "invalid chunk header: index or count is out of range"
        if chunk_index + 1 < chunk_count and len(chunk) != _MAX_CHUNK_PAYLOAD:
            return None, "invalid chunk: non-final payload has the wrong size"
        if chunk_index + 1 == chunk_count and not 0 < len(chunk) <= _MAX_CHUNK_PAYLOAD:
            return None, "invalid chunk: final payload has the wrong size"

        key = (port, sender, message_id)
        state = self._partial.get(key)
        if state is None:
            if len(self._partial) >= self._max_inflight_messages:
                return None, "chunk reassembly state limit reached"
            state = _PartialMessage(chunk_count, {}, 0, now)
            self._partial[key] = state
        elif state.chunk_count != chunk_count:
            del self._partial[key]
            return None, "invalid chunk: inconsistent chunk count"

        previous = state.chunks.get(chunk_index)
        if previous is not None:
            if previous != chunk:
                del self._partial[key]
                return None, "invalid chunk: conflicting duplicate"
            state.updated_at = now
            return None, ""
        if state.size + len(chunk) > _MAX_LOGICAL_PAYLOAD:
            del self._partial[key]
            return None, "chunked message exceeds the logical payload limit"
        state.chunks[chunk_index] = chunk
        state.size += len(chunk)
        state.updated_at = now
        if len(state.chunks) != state.chunk_count:
            return None, ""

        payload = b"".join(state.chunks[index] for index in range(state.chunk_count))
        del self._partial[key]
        if not _MAX_DATAGRAM_PAYLOAD < len(payload) <= _MAX_LOGICAL_PAYLOAD:
            return None, "chunked message has an invalid logical payload size"
        return payload, ""

    def _success_reached(self, ports_with_valid_json: set[int]) -> bool:
        if self._require_all_ports:
            return len(ports_with_valid_json) == len(self._sockets)
        return bool(ports_with_valid_json)

    def _parse_message(
        self, port: int, payload: bytes
    ) -> tuple[MetadataJsonMessage | None, str]:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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
