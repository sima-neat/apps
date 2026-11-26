"""End-to-end tests: RTSP streams reach Insight as video and tracking metadata, each on its own
channel. tests/cpp/test_e2e.cpp runs the same checks in C++."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 150
# Above the shipped 0.05, so the tracks show that the example applies inference.min_score.
MIN_SCORE = 0.1
# Each stream's results per second, about 80% of what CI measured. Four streams share the model.
MIN_RESULTS_PER_S = 20.0
MIN_FOUR_STREAM_RESULTS_PER_S = 10.0
# RTP payload types: H.264 96, H.265 98.
PAYLOAD_TYPES = {"h264": 96, "h265": 98}
# A keyframe arrives as a burst that a small buffer drops. net.core.rmem_max caps the size, so
# the report prints what the socket got.
VIDEO_RECEIVE_BUFFER = 8 << 20

# The last value: whether the config sets input.width, input.height, and input.fps to the
# stream's values instead of 0, which skips the app's probe.
CASES = [
    pytest.param("h264", 2, MIN_RESULTS_PER_S, False, id="h264"),
    pytest.param("h265", 2, MIN_RESULTS_PER_S, False, id="h265"),
    # With one source, Core names the samples after the decoder instead of the stream.
    pytest.param("h264", 1, MIN_RESULTS_PER_S, False, id="h264-one-stream"),
    # Four streams of the first URL, the most the README lists, share 4 frames in flight.
    pytest.param("h264", 4, MIN_FOUR_STREAM_RESULTS_PER_S, False, id="h264-four-streams"),
    pytest.param("h264", 1, MIN_RESULTS_PER_S, True, id="h264-configured-size"),
]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
    # The loop only reads, into one buffer, so it empties the socket as fast as it can.
    sock.settimeout(0.2)
    packet = bytearray(65536)
    while True:
        try:
            sock.recv_into(packet)
            headers.append(bytes(packet[:12]))
        except TimeoutError:
            if stop.is_set():
                return


def _udp_counters() -> dict[str, int]:
    """The host's UDP counters, from the Udp: lines of /proc/net/snmp."""
    try:
        lines = Path("/proc/net/snmp").read_text().splitlines()
    except OSError:
        return {}
    names, values = [line.split()[1:] for line in lines if line.startswith("Udp:")][:2]
    return dict(zip(names, map(int, values)))


def _socket_drops(port: int) -> int | None:
    """Datagrams the kernel dropped for the socket bound to 127.0.0.1:`port`, the last column of
    /proc/net/udp."""
    try:
        rows = Path("/proc/net/udp").read_text().splitlines()[1:]
    except OSError:
        return None
    for row in rows:
        fields = row.split()
        if fields[1] == f"0100007F:{port:04X}":
            return int(fields[-1])
    return None


def _sequence_gaps(headers: list[bytes]) -> list[tuple[int, int, int]]:
    """Each gap in the RTP sequence: the packets missing, and the RTP timestamps of the packets
    before and after it. A frame wholly in a gap has a timestamp between those two."""
    gaps = []
    for previous, current in zip(headers, headers[1:]):
        step = (int.from_bytes(current[2:4], "big") - int.from_bytes(previous[2:4], "big")) & 0xFFFF
        if 1 < step < 0x8000:
            gaps.append(
                (
                    step - 1,
                    int.from_bytes(previous[4:8], "big"),
                    int.from_bytes(current[4:8], "big"),
                )
            )
    return gaps


def _in_gap(timestamp_ms: int, gaps: list[tuple[int, int, int]]) -> bool:
    """Whether the frame at `timestamp_ms` may be one of the packets a gap is missing."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(
        (ticks - before + 90) & 0xFFFFFFFF <= (after - before + 180) & 0xFFFFFFFF
        for _, before, after in gaps
    )


def _video_report(headers: list[bytes], timestamps_ms: list[int], fps: int) -> str:
    """What one video socket got: its packets, the gaps in their sequence, and the frames received
    against the source's frames in the span the metadata covers."""
    missing = sum(lost for lost, _, _ in _sequence_gaps(headers))
    rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers}
    first, last = min(timestamps_ms), max(timestamps_ms)
    start = (first * 90 - 90) & 0xFFFFFFFF
    in_span = sum(
        1 for rtp in rtp_timestamps if (rtp - start) & 0xFFFFFFFF <= (last - first) * 90 + 180
    )
    expected = round((last - first) * fps / 1000) + 1
    return (
        f"{len(headers)} RTP packets, {missing} missing from the RTP sequence, "
        f"{len(rtp_timestamps)} frame timestamps, {in_span} of the source's {expected} frames "
        f"from {first} to {last} ms"
    )


def _pairs(timestamp_ms: int, rtp_timestamps: set[int]) -> bool:
    """Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a
    video frame's RTP timestamp."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(abs(ticks - rtp) <= 90 for rtp in rtp_timestamps)


@pytest.mark.e2e
@pytest.mark.parametrize(("codec", "streams", "min_results_per_s", "configured_size"), CASES)
def test_streams_reach_their_insight_channels_as_video_and_tracks(
    request,
    codec,
    streams,
    min_results_per_s,
    configured_size,
    e2e_model_path,
    tmp_output_dir,
    test_timeout_ms,
    e2e_config_writer,
    skip_unless_e2e_ready,
):
    urls = request.getfixturevalue(f"rtsp_{codec}_urls")
    if streams == 4:
        urls = urls[:1] * 4
    needed = 1 if streams == 4 else streams
    skip_unless_e2e_ready(
        len(urls) >= needed, f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS needs {needed} URL(s)"
    )
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_dir = tmp_output_dir.parent
    model_path = os.path.relpath(Path(e2e_model_path).resolve(), config_dir.resolve())
    # The app runs from the Apps root, so only a path joined to the config file finds the model.
    assert not (APPS_ROOT / model_path).exists(), model_path
    size = {}
    if configured_size:
        # Imported here, so collecting this file needs no OpenCV.
        sys.path.insert(0, str(APPS_ROOT / "support" / "python"))
        from sima_examples.rtsp_probe import probe_rtsp_stream_info

        # The packaged config receives RTSP over TCP.
        stream = probe_rtsp_stream_info(urls[0], True)
        size = {"width": stream.width, "height": stream.height, "fps": stream.fps}
    config_path = e2e_config_writer(
        {
            # Relative to the generated config, so the test runs the app's path join.
            "model": {"path": model_path},
            "input": {"source": urls[:streams], "codec": codec, **size},
            "inference": {"frames": FRAMES, "min_score": MIN_SCORE},
            # This test is the Insight receiver, so the example sends to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
        }
    )
    assert config_path.parent == config_dir

    messages = []
    arrivals = {index: [] for index in range(streams)}
    errors = []
    headers: list[list[bytes]] = [[] for _ in range(streams)]
    stop = threading.Event()
    video_sockets = []
    drains = []
    receive_buffers: list[int] = []
    socket_drops: list[int | None] = []
    udp_before = udp_after = {}
    process = None
    stdout = stderr = ""
    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=streams,
        metadata_type="tracking",
        data_array_key="tracks",
    ) as listener:
        try:
            for index in range(streams):
                video_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                video_sockets[-1].setsockopt(
                    socket.SOL_SOCKET, socket.SO_RCVBUF, VIDEO_RECEIVE_BUFFER
                )
                receive_buffers.append(
                    video_sockets[-1].getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                )
                video_sockets[-1].bind(("127.0.0.1", video_port + index))
                drains.append(
                    threading.Thread(
                        target=_drain_video,
                        args=(video_sockets[-1], stop, headers[index]),
                        daemon=True,
                    )
                )
                drains[-1].start()
            udp_before = _udp_counters()
            process = subprocess.Popen(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(APPS_ROOT),
            )
            deadline = time.monotonic() + test_timeout_ms / 1000
            # Read during the run and at exit, so no message waits long enough to overflow the
            # socket buffer.
            while process.poll() is None and time.monotonic() < deadline:
                result = listener.wait_for_messages(0.2)
                if result.error and not result.timed_out:
                    errors.append(result.error)
                for message in result.messages:
                    messages.append(message)
                    arrivals[message.port - metadata_port].append(time.monotonic())
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    stderr = f"killed after {test_timeout_ms} ms\n"
                out, err = process.communicate()
                stdout, stderr = out, stderr + err
                rest = listener.drain_pending()
                messages.extend(rest.messages)
                if rest.error:
                    errors.append(rest.error)
            stop.set()
            for drain in drains:
                drain.join()
            socket_drops = [
                _socket_drops(video_port + index) for index in range(len(video_sockets))
            ]
            udp_after = _udp_counters()
            for sock in video_sockets:
                sock.close()

    assert process.returncode == 0, (
        f"exit {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert not errors, errors
    # The host's UDP counters over the run: RcvbufErrors counts datagrams dropped because a
    # receiving socket's buffer was full.
    udp_change = ", ".join(
        f"{name} +{udp_after[name] - udp_before[name]}"
        for name in ("RcvbufErrors", "SndbufErrors")
        if name in udp_before and name in udp_after
    )
    stream_size = rf"{size['width']}x{size['height']}@({size['fps']})" if size else r"\d+x\d+@(\d+)"
    for index in range(streams):
        start = re.search(
            rf"^\[stream {index}\] rtsp={re.escape(urls[index])} stream={stream_size} "
            rf"insight=127\.0\.0\.1 video={video_port + index} metadata={metadata_port + index}$",
            stdout,
            re.MULTILINE,
        )
        assert start, stdout
        summary = re.search(
            rf"^\[stream {index}\] processed={FRAMES} tracks_started=(\d+)$", stdout, re.MULTILINE
        )
        assert summary, stdout
        tracks_started = int(summary.group(1))
        assert {header[1] & 0x7F for header in headers[index]} == {PAYLOAD_TYPES[codec]}, (
            f"stream {index} video payload types"
        )

        stream_messages = [m for m in messages if m.port == metadata_port + index]
        # UDP on loopback may drop a message under load, so allow a few to go missing.
        assert len(stream_messages) >= FRAMES * 95 // 100, (
            f"stream {index}: {len(stream_messages)} of {FRAMES} messages arrived"
        )
        assert all(m.frame_id for m in stream_messages), f"stream {index}: a message lacks frame_id"
        frame_ids = [int(message.frame_id) for message in stream_messages]
        assert frame_ids == sorted(set(frame_ids)), f"stream {index}: frame IDs out of order"
        # The per-track checks cover only the tracks a stream sent: CI's second URLs send none.
        per_message = [json.loads(m.payload)["data"]["tracks"] for m in stream_messages]
        tracks = [track for message_tracks in per_message for track in message_tracks]
        # CI's first H.264 and H.265 URLs show objects the model tracks.
        if index == 0:
            assert tracks, "stream 0 sent no tracks"
            # The tracker keeps an ID across frames: some ID is in two or more messages.
            messages_per_id = Counter(
                track_id
                for message_tracks in per_message
                for track_id in {track["id"] for track in message_tracks}
            )
            assert max(messages_per_id.values()) >= 2, (
                f"stream 0: no track ID is in two messages: {messages_per_id}"
            )
        for track in tracks:
            # The model has one class, so a class ID other than 0 would arrive as "unknown".
            assert track["label"] == "drone", track
            assert MIN_SCORE <= track["confidence"] <= 1.0, track
            assert 1 <= int(track["id"]) <= tracks_started, track
        # Insight draws each message's tracks on the video frame its timestamp names.
        rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers[index]}
        unpaired = [
            m.timestamp_ms for m in stream_messages if not _pairs(m.timestamp_ms, rtp_timestamps)
        ]
        # The report names the unpaired messages whose frame fell in a gap of the RTP sequence.
        gaps = _sequence_gaps(headers[index])
        lost = [timestamp for timestamp in unpaired if _in_gap(timestamp, gaps)]
        fps = int(start.group(1))
        report = _video_report(headers[index], [m.timestamp_ms for m in stream_messages], fps)
        video = (
            f"video port {video_port + index}: {report}, {socket_drops[index]} dropped by the "
            f"socket, receive buffer {receive_buffers[index]} bytes from getsockopt for "
            f"{VIDEO_RECEIVE_BUFFER} asked, {len(lost)} messages in a sequence gap. "
            f"Host UDP: {udp_change or 'not read'}"
        )
        # The video link sends every frame, so a message stays unpaired only when its frame was
        # lost after the sender. Allow 5%, as for the count.
        assert len(unpaired) <= len(stream_messages) * 5 // 100, (
            f"stream {index}: {len(unpaired)} timestamps with no video frame: {unpaired[:5]}. "
            f"{video}"
        )
        times = arrivals[index][len(arrivals[index]) // 10 :]
        assert len(times) > 10, f"stream {index}: {len(times)} messages arrived during the run"
        results_per_s = (len(times) - 1) / (times[-1] - times[0])
        print(
            f"{codec} stream {index}: {len(stream_messages)} messages, "
            f"{results_per_s:.1f} results/s, {len(tracks)} tracks in {tracks_started} track IDs, "
            f"{len(unpaired)} unpaired. {video}"
        )
        assert results_per_s >= min_results_per_s, f"stream {index}: {results_per_s:.1f} results/s"
