"""End-to-end tests: RTSP streams reach Insight as video and oriented-box metadata, each on its
own channel. tests/cpp/test_e2e.cpp runs the same checks in C++."""

from __future__ import annotations

import json
import math
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
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
LABELS = set((EXAMPLE_DIR / "src" / "common" / "dota_label.txt").read_text().splitlines())
FRAMES = 150
# Above the shipped 0.30, so the boxes show that the example applies inference.min_score.
MIN_SCORE = 0.4
# The box and rate checks run on the suite's model only. Larger variants run slower.
DEFAULT_MODEL = "yolo26n_obb_w8a8_mpk.tar.gz"
# Each stream's results per second, about 80% of Vulcan CI run 45418480248 on 5baed52 (Modalix
# devkit-1, firmware 3.0.1_master_B2798, Core develop/530580ae4d76, 25 fps sources). With one or
# two streams, each stream gave 24.6 to 25.2, the source's rate.
MIN_STREAM_FPS = 20.0
# RTP payload types: H.264 96, H.265 98, and RTP/JPEG 26.
PAYLOAD_TYPES = {"h264": 96, "h265": 98, "mjpeg": 26}

CASES = [
    pytest.param("h264", 2, id="h264"),
    pytest.param("h265", 2, id="h265"),
    pytest.param("mjpeg", 2, id="mjpeg"),
    # Core candidate: with one source, Core names the sample after the decoder instead
    # of the link's stream_id, so the index is 0 here.
    pytest.param("h264", 1, id="h264-one-stream"),
]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def _model_id(model_path: Path) -> str:
    """Registry package files start with their model ID, such as yolov8n_obb_w8a8_mpk.tar.gz."""
    return next((family for family in ("yolov8", "yolo11") if model_path.name.startswith(family)),
                "yolo26")


def _drain_video(sock: socket.socket, stop: threading.Event, payload_types: set[int]) -> None:
    """Keep each video packet's RTP payload type until `stop` is set and the socket is empty."""
    sock.settimeout(0.2)
    while True:
        try:
            payload_types.add(sock.recv(65536)[1] & 0x7F)
        except TimeoutError:
            if stop.is_set():
                return


def _object_problem(obj: dict) -> str:
    if set(obj) != {"id", "label", "confidence", "bbox", "oriented_bbox"}:
        return "unexpected keys"
    if obj["label"] not in LABELS:
        return "label is not a DOTA class"
    if not MIN_SCORE <= obj["confidence"] <= 1.0:
        return "confidence outside [min_score, 1]"
    cx, cy, width, height, angle = obj["oriented_bbox"]
    if width <= 0 or height <= 0 or not math.isfinite(angle):
        return "degenerate oriented box"
    x, y, extent_w, extent_h = obj["bbox"]
    # Each value is rounded to 0.1 px, so the centers can differ by up to 0.1 px.
    if abs(x + extent_w / 2 - cx) > 0.15 or abs(y + extent_h / 2 - cy) > 0.15:
        return "bbox is not centered on the oriented box"
    return ""


@pytest.mark.e2e
@pytest.mark.parametrize(("codec", "streams"), CASES)
def test_streams_reach_their_insight_channels(
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
    skip_unless_e2e_ready(
        len(urls) >= streams, f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS needs {streams} URLs"
    )
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    model_path = Path(e2e_model_path)
    config_dir = tmp_output_dir.parent
    config_path = e2e_config_writer(
        {
            # Relative to the generated config, so the test runs the app's path join.
            "model": {
                "id": _model_id(model_path),
                "path": os.path.relpath(model_path.resolve(), config_dir.resolve()),
            },
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
    errors = []
    arrivals = {index: [] for index in range(streams)}
    payload_types = [set() for _ in range(streams)]
    stop = threading.Event()
    video_sockets = []
    drains = []
    process = None
    stdout = stderr = ""
    with MetadataJsonListener("127.0.0.1", metadata_port, num_ports=streams) as listener:
        try:
            for index in range(streams):
                video_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                video_sockets[-1].bind(("127.0.0.1", video_port + index))
                drains.append(
                    threading.Thread(
                        target=_drain_video,
                        args=(video_sockets[-1], stop, payload_types[index]),
                        daemon=True,
                    )
                )
                drains[-1].start()
            process = subprocess.Popen(
                [sys.executable, str(MAIN_PY), "--config", str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(EXAMPLE_DIR.parents[2]),
            )
            deadline = time.monotonic() + test_timeout_ms / 1000
            # Read during the run, so no message waits long enough to overflow the socket buffer.
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

    # pytest shows this output only when the test fails.
    print(f"application exit code {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    assert process.returncode == 0, f"exit code {process.returncode}"
    assert not errors, errors
    default_model = model_path.name == DEFAULT_MODEL
    boxes = 0
    for index in range(streams):
        assert re.search(
            rf"^\[stream {index}\] rtsp={re.escape(urls[index])} stream=\d+x\d+@\d+ "
            rf"insight=127\.0\.0\.1 video={video_port + index} metadata={metadata_port + index}$",
            stdout,
            re.MULTILINE,
        ), f"stream {index} did not print its source and ports"
        assert f"[stream {index}] processed={FRAMES}" in stdout, (
            f"stream {index} did not report processed={FRAMES}"
        )
        assert payload_types[index] == {PAYLOAD_TYPES[codec]}, (
            f"stream {index} video: RTP payload types {payload_types[index]}"
        )
        stream_messages = [m for m in messages if m.port == metadata_port + index]
        # UDP on loopback may drop a message under load, so allow a few to go missing.
        assert len(stream_messages) >= FRAMES * 95 // 100, (
            f"stream {index}: {len(stream_messages)} of {FRAMES} messages arrived"
        )
        # Each result names its frame and carries its timestamp, so Insight can pair it with video.
        frame_ids = [int(m.frame_id) if m.frame_id.isdigit() else -1 for m in stream_messages]
        assert all(b > a for a, b in zip([-1, *frame_ids], frame_ids)), (
            f"stream {index}: frame IDs must increase: {[m.frame_id for m in stream_messages]}"
        )
        assert all(m.timestamp_ms >= 0 for m in stream_messages), (
            f"stream {index}: timestamps {[m.timestamp_ms for m in stream_messages]}"
        )
        for message in stream_messages:
            assert message.metadata_type == "object-detection"
            for obj in json.loads(message.payload)["data"]["objects"]:
                problem = _object_problem(obj)
                assert not problem, f"stream {index} frame {message.frame_id}: {problem} {obj}"
                boxes += 1
        times = arrivals[index][len(arrivals[index]) // 10:]
        assert len(times) > 10, f"stream {index}: {len(times)} messages arrived during the run"
        fps = (len(times) - 1) / (times[-1] - times[0])
        print(f"{codec} stream {index}: {len(stream_messages)} messages, {fps:.1f} results/s")
        if default_model:
            assert fps >= MIN_STREAM_FPS, f"stream {index}: {fps:.1f} results/s"
    # The model is trained on top-down aerial video, and the CI streams show streets. Its boxes
    # there are still enough to check: the C++ cases in CI run 45584903721 on d291e61 checked 1265
    # boxes on h264, 680 on h264-one-stream, 848 on h265, and 141 on mjpeg.
    print(f"{codec}: {boxes} oriented boxes checked")
    if default_model:
        assert boxes > 0, "no oriented boxes arrived"
