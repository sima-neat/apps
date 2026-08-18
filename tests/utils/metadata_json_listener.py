"""Test-only UDP listener for Insight metadata JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import select
import socket
import struct
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


_CHUNK_MAGIC = 0x4E
_CHUNK_VERSION = 0x01
_CHUNK_HEADER_SIZE = 12
_MAX_DATAGRAM_SIZE = 1200
_MAX_LOGICAL_MESSAGE_SIZE = 65507
_MAX_CHUNK_COUNT = 56
_REASSEMBLY_MAX_AGE_S = 0.25
_REASSEMBLY_CAPACITY = 4


@dataclass
class _Assembly:
    chunks: list[bytes | None]
    started: float
    received: int = 0
    size: int = 0


class _MetadataReassembler:
    def __init__(self) -> None:
        self._assemblies: dict[int, _Assembly] = {}

    def accept(self, datagram: bytes) -> tuple[bytes | None, str]:
        now = time.monotonic()
        self._drop_expired(now)
        if not datagram or datagram[0] != _CHUNK_MAGIC:
            return datagram, ""
        if (
            len(datagram) < _CHUNK_HEADER_SIZE
            or len(datagram) > _MAX_DATAGRAM_SIZE
            or datagram[1] != _CHUNK_VERSION
        ):
            return None, "invalid metadata chunk header"

        message_id, index, count = struct.unpack(">QBB", datagram[2:_CHUNK_HEADER_SIZE])
        if (
            count == 0
            or count > _MAX_CHUNK_COUNT
            or index >= count
            or len(datagram) == _CHUNK_HEADER_SIZE
        ):
            return None, "invalid metadata chunk fields"

        assembly = self._assemblies.get(message_id)
        if assembly is None:
            if len(self._assemblies) == _REASSEMBLY_CAPACITY:
                oldest = min(
                    self._assemblies, key=lambda key: self._assemblies[key].started
                )
                del self._assemblies[oldest]
            assembly = _Assembly([None] * count, now)
            self._assemblies[message_id] = assembly
        if len(assembly.chunks) != count:
            del self._assemblies[message_id]
            return None, "metadata chunk count changed"

        fragment = datagram[_CHUNK_HEADER_SIZE:]
        existing = assembly.chunks[index]
        if existing is None:
            if assembly.size + len(fragment) > _MAX_LOGICAL_MESSAGE_SIZE:
                del self._assemblies[message_id]
                return None, "metadata message exceeds maximum size"
            assembly.chunks[index] = fragment
            assembly.received += 1
            assembly.size += len(fragment)
        elif existing != fragment:
            del self._assemblies[message_id]
            return None, "metadata chunk contents changed"

        if assembly.received != len(assembly.chunks):
            return None, ""

        payload = b"".join(chunk for chunk in assembly.chunks if chunk is not None)
        del self._assemblies[message_id]
        return payload, ""

    def _drop_expired(self, now: float) -> None:
        expired = [
            message_id
            for message_id, assembly in self._assemblies.items()
            if now - assembly.started > _REASSEMBLY_MAX_AGE_S
        ]
        for message_id in expired:
            del self._assemblies[message_id]


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
        min_object_count: int = 0,
    ) -> None:
        if num_ports <= 0:
            raise ValueError("num_ports must be > 0")
        if base_port <= 0:
            raise ValueError("base_port must be > 0")
        if min_object_count < 0:
            raise ValueError("min_object_count must be >= 0")

        self._metadata_type = metadata_type
        self._data_array_key = data_array_key
        self._require_all_ports = require_all_ports
        self._min_object_count = min_object_count
        self._sockets: dict[socket.socket, int] = {}
        self._reassemblers: dict[socket.socket, _MetadataReassembler] = {}
        try:
            for offset in range(num_ports):
                port = base_port + offset
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                sock.setblocking(False)
                self._sockets[sock] = port
                self._reassemblers[sock] = _MetadataReassembler()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for sock in self._sockets:
            sock.close()
        self._sockets.clear()
        self._reassemblers.clear()

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
                datagram, _ = sock.recvfrom(65536)
                port = self._sockets[sock]
                payload, error = self._reassemblers[sock].accept(datagram)
                if payload is None:
                    if error:
                        last_error = error
                    continue
                message, error = self._parse_message(port, payload)
                if message is None:
                    last_error = error
                    continue
                messages.append(message)
                if message.object_count < self._min_object_count:
                    last_error = (
                        f"data.{self._data_array_key} contains {message.object_count} objects; "
                        f"expected at least {self._min_object_count}"
                    )
                    continue
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
