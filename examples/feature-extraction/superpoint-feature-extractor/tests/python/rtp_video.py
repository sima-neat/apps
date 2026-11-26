"""Receive an example's Insight video as RTP and check that its metadata pairs with the frames.

Mirrors tests/cpp/rtp_video.h, so the C++ and Python end-to-end tests
check video the same way.
"""

from __future__ import annotations

import socket
import threading


def drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
    sock.settimeout(0.2)
    while True:
        try:
            packet = sock.recv(65536)
        except TimeoutError:
            if stop.is_set():
                return
            continue
        if len(packet) >= 12:
            headers.append(packet[:12])


def rtp_timestamps(headers: list[bytes]) -> set[int]:
    """The RTP timestamp of each kept header."""
    return {int.from_bytes(header[4:8], "big") for header in headers}


def pairs(timestamp_ms: int, timestamps: set[int]) -> bool:
    """Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a
    video frame's RTP timestamp. Both wrap at 2^32."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(
        min((ticks - rtp) & 0xFFFFFFFF, (rtp - ticks) & 0xFFFFFFFF) <= 90 for rtp in timestamps
    )
