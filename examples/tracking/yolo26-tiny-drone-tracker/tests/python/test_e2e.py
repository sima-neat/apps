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
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 150
# Above the shipped 0.05, so the tracks show that the example applies inference.min_score.
MIN_SCORE = 0.1
# Each stream's results per second. Not verified on a DevKit yet.
MIN_STREAM_RESULTS_PER_S = 20.0
# RTP payload types: H.264 96, H.265 98.
PAYLOAD_TYPES = {"h264": 96, "h265": 98}

CASES = [
    pytest.param("h264", 2, id="h264"),
    pytest.param("h265", 2, id="h265"),
    # With one source, Core names the samples after the decoder instead of the stream.
    pytest.param("h264", 1, id="h264-one-stream"),
    # Four streams of the first URL: the README's most, and the first count past Core's default
    # in-flight total of 8.
    pytest.param("h264", 4, id="h264-four-streams"),
]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
    # A copy of single-stream's helper. #823 moves it into tests/utils/rtp_video.py.
    sock.settimeout(0.2)
    while True:
        try:
            headers.append(sock.recv(65536)[:12])
        except TimeoutError:
            if stop.is_set():
                return


def _pairs(timestamp_ms: int, rtp_timestamps: set[int]) -> bool:
    """Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a
    video frame's RTP timestamp."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(abs(ticks - rtp) <= 90 for rtp in rtp_timestamps)


@pytest.mark.e2e
@pytest.mark.parametrize(("codec", "streams"), CASES)
def test_streams_reach_their_insight_channels_as_video_and_tracks(
    request,
    codec,
    streams,
    e2e_model_path,
    tmp_output_dir,
    test_timeout_ms,
    e2e_config_writer,
    skip_unless_e2e_ready,
):
    urls = request.getfixturevalue(f"rtsp_{codec}_urls")
    if streams == 4:
        urls = urls[:1] * 4
    skip_unless_e2e_ready(
        len(urls) >= streams, f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS needs {streams} URLs"
    )
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_dir = tmp_output_dir.parent
    model_path = os.path.relpath(Path(e2e_model_path).resolve(), config_dir.resolve())
    # The app runs from the Apps root, so only a path joined to the config file finds the model.
    assert not (APPS_ROOT / model_path).exists(), model_path
    config_path = e2e_config_writer(
        {
            # Relative to the generated config, so the test runs the app's path join.
            "model": {"path": model_path},
            "input": {"source": urls[:streams], "codec": codec},
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
                video_sockets[-1].bind(("127.0.0.1", video_port + index))
                drains.append(
                    threading.Thread(
                        target=_drain_video,
                        args=(video_sockets[-1], stop, headers[index]),
                        daemon=True,
                    )
                )
                drains[-1].start()
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
            for sock in video_sockets:
                sock.close()

    assert process.returncode == 0, (
        f"exit {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert not errors, errors
    for index in range(streams):
        assert re.search(
            rf"^\[stream {index}\] rtsp={re.escape(urls[index])} stream=\d+x\d+@\d+ "
            rf"insight=127\.0\.0\.1 video={video_port + index} metadata={metadata_port + index}$",
            stdout,
            re.MULTILINE,
        ), stdout
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
        frame_ids = [int(message.frame_id) for message in stream_messages]
        assert frame_ids == sorted(set(frame_ids)), f"stream {index}: frame IDs out of order"
        # These checks cover the tracks the run sent. A stream without drones sends none, and the
        # count printed below shows how many there were.
        tracks = [t for m in stream_messages for t in json.loads(m.payload)["data"]["tracks"]]
        for track in tracks:
            assert track["label"] == "drone", track
            assert MIN_SCORE <= track["confidence"] <= 1.0, track
            assert 1 <= int(track["id"]) <= tracks_started, track
        # Insight draws each message's tracks on the video frame its timestamp names.
        rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers[index]}
        unpaired = [
            m.timestamp_ms for m in stream_messages if not _pairs(m.timestamp_ms, rtp_timestamps)
        ]
        # A frame the video link dropped leaves its message unpaired, so allow as few as for
        # the count.
        assert len(unpaired) <= len(stream_messages) * 5 // 100, (
            f"stream {index}: {len(unpaired)} timestamps with no video frame: {unpaired[:5]}"
        )
        times = arrivals[index][len(arrivals[index]) // 10:]
        assert len(times) > 10, f"stream {index}: {len(times)} messages arrived during the run"
        results_per_s = (len(times) - 1) / (times[-1] - times[0])
        print(
            f"{codec} stream {index}: {len(stream_messages)} messages, "
            f"{results_per_s:.1f} results/s, {len(tracks)} tracks in {tracks_started} track IDs"
        )
        assert results_per_s >= MIN_STREAM_RESULTS_PER_S, (
            f"stream {index}: {results_per_s:.1f} results/s"
        )
