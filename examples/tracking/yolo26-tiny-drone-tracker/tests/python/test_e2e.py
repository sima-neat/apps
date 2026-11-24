"""End-to-end tests: one RTSP stream reaches Insight as video and tracking metadata.
tests/cpp/test_e2e.cpp runs the same checks in C++."""

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
# The model must keep up with the 30 FPS test stream.
MIN_RESULTS_PER_S = 20.0
# RTP payload types: H.264 96, H.265 98.
PAYLOAD_TYPES = {"h264": 96, "h265": 98}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _drain_video(sock: socket.socket, stop: threading.Event, headers: list[bytes]) -> None:
    """Keep each video packet's RTP header until `stop` is set and the socket is empty."""
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
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_stream_reaches_insight_as_video_and_tracks(
    request, codec, e2e_model_path, tmp_output_dir, test_timeout_ms, e2e_config_writer
):
    url = request.getfixturevalue(f"rtsp_{codec}_url")
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
            "input": {"source": url, "codec": codec},
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
    arrivals = []
    errors = []
    headers: list[bytes] = []
    stop = threading.Event()
    process = None
    stdout = stderr = ""
    with (
        MetadataJsonListener(
            "127.0.0.1", metadata_port, num_ports=1, metadata_type="tracking",
            data_array_key="tracks",
        ) as listener,
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as video,
    ):
        video.bind(("127.0.0.1", video_port))
        drain = threading.Thread(target=_drain_video, args=(video, stop, headers), daemon=True)
        drain.start()
        try:
            process = subprocess.Popen(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(APPS_ROOT),
            )
            deadline = time.monotonic() + test_timeout_ms / 1000
            # Read during the run, so no message waits long enough to overflow the socket buffer.
            while process.poll() is None and time.monotonic() < deadline:
                result = listener.wait_for_messages(0.2)
                if result.error and not result.timed_out:
                    errors.append(result.error)
                messages.extend(result.messages)
                arrivals.extend(time.monotonic() for _ in result.messages)
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
            drain.join()

    assert process.returncode == 0, (
        f"exit {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert not errors, errors
    assert re.search(
        rf"^source={re.escape(url)} codec={codec} stream=\d+x\d+@\d+ insight=127\.0\.0\.1 "
        rf"video={video_port} metadata={metadata_port}$",
        stdout,
        re.MULTILINE,
    ), stdout
    summary = re.search(rf"^processed={FRAMES} tracks_started=(\d+)$", stdout, re.MULTILINE)
    assert summary, stdout
    tracks_started = int(summary.group(1))

    assert headers and {header[1] & 0x7F for header in headers} == {PAYLOAD_TYPES[codec]}
    # UDP on loopback may drop a message under load, so allow a few to go missing.
    assert len(messages) >= FRAMES * 95 // 100, f"{len(messages)} of {FRAMES} messages arrived"
    frame_ids = [int(message.frame_id) for message in messages]
    assert frame_ids == sorted(set(frame_ids)), "frame IDs do not strictly increase"
    tracks = [track for m in messages for track in json.loads(m.payload)["data"]["tracks"]]
    for track in tracks:
        assert track["label"] == "drone", track
        assert MIN_SCORE <= track["confidence"] <= 1.0, track
        assert 1 <= int(track["id"]) <= tracks_started, track
    # Insight draws each message's tracks on the video frame its timestamp names.
    rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers}
    unpaired = [m.timestamp_ms for m in messages if not _pairs(m.timestamp_ms, rtp_timestamps)]
    assert not unpaired, f"metadata timestamps with no video frame: {unpaired[:5]}"
    times = arrivals[len(arrivals) // 10:]
    assert len(times) > 10, f"{len(times)} messages arrived during the run"
    results_per_s = (len(times) - 1) / (times[-1] - times[0])
    # The test streams show no drones, so the track count is printed, not asserted.
    print(f"{codec}: {len(messages)} messages, {results_per_s:.1f} results/s, "
          f"{len(tracks)} tracks in {tracks_started} track IDs")
    assert results_per_s >= MIN_RESULTS_PER_S, f"{results_per_s:.1f} results/s"
