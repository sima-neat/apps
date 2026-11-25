"""End-to-end tests: four RTSP streams, each with its own model, reach Insight as video and as
that model's metadata, each on its own channel. tests/cpp/test_e2e.cpp runs the same checks in
C++."""

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
import yaml

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
LABELS = (EXAMPLE_DIR / "src" / "common" / "coco_label.txt").read_text().splitlines()
FRAMES = 150
# Above the shipped 0.30, so the results show that the example applies inference.min_score.
MIN_SCORE = 0.35
# Each stream's results per second, about 80% of the lowest tests/cpp/test_e2e.cpp gave in Vulcan
# CI run 45615743474 on 02572b9 (Modalix devkit-1, firmware 3.0.1_master_B2798, Core
# develop/0420557512d1): 24.6 to 26.4 on 25 fps sources, 32.3 to 33.2 on a 30 fps source. On
# e83d42f the same DevKit gave 19.9 to 20.0, and that log does not show the sources' rate. Whether
# the app keeps up with its sources is measured on a board, not here.
MIN_RESULTS_PER_S = 19.5
# RTP payload types: H.264 96, H.265 98.
PAYLOAD_TYPES = {"h264": 96, "h265": 98}
# The shipped config's streams, in order: the package test-scope.yaml downloads for each, the
# Insight metadata type its model sends, and the key of the entries in that type's data.
STREAMS = [
    ("yolo11s_w8a8_mpk.tar.gz", "object-detection", "objects"),
    ("yolo26s_seg_w8a16_mpk.tar.gz", "segmentation", "segments"),
    ("yolo26m_pose_w8a8_mpk.tar.gz", "pose-estimation", "poses"),
    ("yolo26m_w8a8_mpk.tar.gz", "object-detection", "objects"),
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


def _udp_rcvbuf_errors() -> int | None:
    """RcvbufErrors of the Udp: line of /proc/net/snmp: datagrams the kernel dropped because a
    UDP socket's receive buffer was full, for every UDP socket of the host. None without the
    file."""
    try:
        rows = [
            line.split()
            for line in Path("/proc/net/snmp").read_text().splitlines()
            if line.startswith("Udp:")
        ]
    except OSError:
        return None
    return int(rows[1][rows[0].index("RcvbufErrors")])


def _video_report(headers: list[list[bytes]], rcvbufs: list[int], drops_before: int | None) -> str:
    """What each video port received, to tell packets the test's sockets dropped from frames the
    app did not send. A gap in the RTP sequence numbers is a packet sent and not received."""
    lines = []
    for index, port_headers in enumerate(headers):
        sequence = [int.from_bytes(header[2:4], "big") for header in port_headers]
        missing = sum((b - a - 1) & 0xFFFF for a, b in zip(sequence, sequence[1:]))
        timestamps = {int.from_bytes(header[4:8], "big") for header in port_headers}
        lines.append(
            f"video port {index}: {len(port_headers)} RTP packets, {missing} sequence numbers "
            f"missing between them, {len(timestamps)} RTP timestamps for {FRAMES} frames, "
            f"SO_RCVBUF {rcvbufs[index]}"
        )
    drops_after = _udp_rcvbuf_errors()
    drops = "unknown" if None in (drops_before, drops_after) else drops_after - drops_before
    try:
        rmem_max = Path("/proc/sys/net/core/rmem_max").read_text().strip()
    except OSError:
        rmem_max = "unknown"
    lines.append(f"Udp RcvbufErrors during the run, all sockets: {drops}; rmem_max {rmem_max}")
    return "\n".join(lines)


def _pairs(timestamp_ms: int, rtp_timestamps: set[int]) -> bool:
    """Insight's pairing: the metadata's timestamp on the 90 kHz RTP clock is within 1 ms of a
    video frame's RTP timestamp."""
    ticks = (timestamp_ms * 90) & 0xFFFFFFFF
    return any(abs(ticks - rtp) <= 90 for rtp in rtp_timestamps)


def _entry_problem(metadata_type: str, entry: dict) -> str:
    """Why `entry` is not a result the stream's model can send, or "" when it is."""
    if not MIN_SCORE <= entry["confidence"] <= 1.0:
        return "confidence below inference.min_score"
    if metadata_type == "pose-estimation":
        names = [point["name"] for point in entry["keypoints"]]
        if entry["label"] != "person" or len(names) != 17 or names[0] != "nose":
            return "not a person with the 17 COCO keypoints"
        return ""
    if entry["label"] not in LABELS:
        return "label not in coco_label.txt"
    if metadata_type == "segmentation" and len(entry["mask"]) < 3:
        return "mask outline with fewer than three points"
    return ""


@pytest.mark.e2e
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_each_stream_reaches_its_insight_channel_as_video_and_its_models_results(
    request,
    codec,
    models_dir,
    tmp_output_dir,
    test_timeout_ms,
    e2e_config_writer,
    skip_unless_e2e_ready,
):
    urls = request.getfixturevalue(f"rtsp_{codec}_urls")
    skip_unless_e2e_ready(len(urls) >= 1, f"SIMANEAT_TEST_RTSP_{codec.upper()}_URLS needs 1 URL")
    model_files = [models_dir / file for file, _, _ in STREAMS]
    missing = [str(path) for path in model_files if not path.is_file()]
    skip_unless_e2e_ready(not missing, f"models not found: {', '.join(missing)}")
    metadata_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", 9100)
    video_port = _env_int("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", 9000)
    config_dir = tmp_output_dir.parent
    model_paths = [os.path.relpath(path.resolve(), config_dir.resolve()) for path in model_files]
    # The app runs from the Apps root, so only a path joined to the config file finds a model.
    assert not any((APPS_ROOT / path).exists() for path in model_paths), model_paths
    sources = [urls[index % len(urls)] for index in range(len(STREAMS))]
    # The shipped entries, each with its path relative to the generated config, so the test runs
    # the app's path join. The labels path starts at the working directory, as shipped.
    shipped = yaml.safe_load((EXAMPLE_DIR / "src" / "common" / "config.yaml").read_text())["model"]
    models = [dict(entry, path=path) for entry, path in zip(shipped, model_paths)]
    # The app runs from the Apps root, so the shipped labels paths resolve from there.
    assert all((APPS_ROOT / entry["labels"]).is_file() for entry in models if "labels" in entry)
    config_path = e2e_config_writer(
        {
            "input": {"source": sources, "codec": codec},
            "inference": {"frames": FRAMES, "min_score": MIN_SCORE},
            # This test is the Insight receiver, so the example sends to loopback.
            "output": {
                "insight": {
                    "host": "127.0.0.1",
                    "video_port_base": video_port,
                    "metadata_port_base": metadata_port,
                }
            },
            "model": models,
        }
    )
    assert config_path.parent == config_dir

    streams = len(STREAMS)
    messages = []
    arrivals = {index: [] for index in range(streams)}
    errors = []
    headers: list[list[bytes]] = [[] for _ in range(streams)]
    stop = threading.Event()
    video_sockets = []
    rcvbufs = []
    drains = []
    drops_before = None
    process = None
    stdout = stderr = ""
    with MetadataJsonListener(
        "127.0.0.1",
        metadata_port,
        num_ports=streams,
        metadata_contracts={metadata_type: key for _, metadata_type, key in STREAMS},
    ) as listener:
        try:
            for index in range(streams):
                video_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                video_sockets[-1].bind(("127.0.0.1", video_port + index))
                rcvbufs.append(video_sockets[-1].getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
                drains.append(
                    threading.Thread(
                        target=_drain_video,
                        args=(video_sockets[-1], stop, headers[index]),
                        daemon=True,
                    )
                )
                drains[-1].start()
            drops_before = _udp_rcvbuf_errors()
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
    video_report = _video_report(headers, rcvbufs, drops_before)
    entries_by_stream = {}
    for index, (_, metadata_type, key) in enumerate(STREAMS):
        start_line = re.search(
            rf"^\[stream {index}\] [a-z0-9-]+ model={re.escape(model_paths[index])} "
            rf"rtsp={re.escape(sources[index])} stream=\d+x\d+@(\d+) insight=127\.0\.0\.1 "
            rf"video={video_port + index} metadata={metadata_port + index}$",
            stdout,
            re.MULTILINE,
        )
        assert start_line, stdout
        stream_fps = int(start_line.group(1))
        assert re.search(
            rf"^\[stream {index}\] processed={FRAMES} left_out=0$", stdout, re.MULTILINE
        ), stdout
        assert {header[1] & 0x7F for header in headers[index]} == {PAYLOAD_TYPES[codec]}, (
            f"stream {index} video payload types"
        )

        stream_messages = [m for m in messages if m.port == metadata_port + index]
        assert {m.metadata_type for m in stream_messages} == {metadata_type}, (
            f"stream {index} metadata types"
        )
        # UDP on loopback may drop a message under load, so allow a few to go missing.
        assert len(stream_messages) >= FRAMES * 95 // 100, (
            f"stream {index}: {len(stream_messages)} of {FRAMES} messages arrived"
        )
        assert all(message.frame_id for message in stream_messages), (
            f"stream {index}: a message without a frame ID"
        )
        frame_ids = [int(message.frame_id) for message in stream_messages]
        assert frame_ids == sorted(set(frame_ids)), f"stream {index}: frame IDs out of order"
        entries = [e for m in stream_messages for e in json.loads(m.payload)["data"][key]]
        entries_by_stream[index] = entries
        problems = [(problem, e) for e in entries if (problem := _entry_problem(metadata_type, e))]
        assert not problems, f"stream {index}: {problems[:3]}"
        # The detection and segmentation models find objects in every test stream. The pose
        # model needs a person in view: when stream 0 found a person on the same URL, the pose
        # stream must send poses. The pose counts seen on a DevKit are not verified yet.
        if metadata_type != "pose-estimation":
            assert entries, f"stream {index} sent no {key}"
        elif sources[index] == sources[0] and any(
            entry["label"] == "person" for entry in entries_by_stream[0]
        ):
            assert entries, f"stream {index} sent no poses while stream 0 saw a person"
        # Insight draws each message's entries on the video frame its timestamp names.
        rtp_timestamps = {int.from_bytes(header[4:8], "big") for header in headers[index]}
        unpaired = [
            m.timestamp_ms for m in stream_messages if not _pairs(m.timestamp_ms, rtp_timestamps)
        ]
        # A frame the video link dropped leaves its message unpaired, so allow 5%, as for the
        # count.
        assert len(unpaired) <= len(stream_messages) * 5 // 100, (
            f"stream {index}: {len(unpaired)} timestamps with no video frame: {unpaired[:5]}\n"
            f"{video_report}"
        )
        times = arrivals[index][len(arrivals[index]) // 10:]
        assert len(times) > 10, f"stream {index}: {len(times)} messages arrived during the run"
        results_per_s = (len(times) - 1) / (times[-1] - times[0])
        print(
            f"{codec} stream {index}: {len(stream_messages)} messages, "
            f"{results_per_s:.1f} results/s of {stream_fps} fps, {len(entries)} {key}"
        )
        assert results_per_s >= MIN_RESULTS_PER_S, (
            f"stream {index}: {results_per_s:.1f} results/s, below {MIN_RESULTS_PER_S}"
        )
