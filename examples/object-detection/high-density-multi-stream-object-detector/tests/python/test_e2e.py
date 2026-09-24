"""Run 16 real RTSP pipelines and verify Insight metadata and send throughput."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml
import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from high_density_qualification import qualification_options

EXAMPLE_DIR = Path(__file__).resolve().parents[2]


def select_source(urls, codec, expected_fps):
    errors = []
    for url in dict.fromkeys(urls):
        cap = cv2.VideoCapture(
            url, cv2.CAP_ANY,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000],
        )
        try:
            if not cap.isOpened():
                errors.append(f"Cannot open source: {url}")
                continue
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if (
            width == 1280
            and height == 720
            and (abs(fps - expected_fps) < 0.001 or
                 (expected_fps == 30 and abs(fps - 30000 / 1001) < 0.001))
        ):
            print(
                f"source codec={codec} width=1280 height=720 fps={fps} unique_publishers=1"
            )
            return url
        errors.append(f"width={width} height={height} fps={fps}")
    pytest.fail(f"No 720p{expected_fps} {codec} source: {errors}")


@pytest.mark.e2e
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_metadata_throughput(
    codec, request, e2e_model_path, e2e_config_writer, test_timeout_ms
):
    streams, fps, frames, profile, decoder_overrides = qualification_options()
    url = select_source(request.getfixturevalue(f"rtsp_{codec}_urls"), codec, fps)
    profile_config = yaml.safe_load((EXAMPLE_DIR / "src/common" / profile).read_text())
    port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", "9100"))
    config = e2e_config_writer(
        {
            "streams": [url] * streams,
            "input": {**profile_config["input"], **decoder_overrides,
                      "codec": codec, "width": 1280, "height": 720},
            "inference": profile_config["inference"],
            "runtime": {**profile_config["runtime"], "warmup_frames": 100, "profile": False},
            "output": {
                "video_enabled": True,
                "insight": {
                    "host": "127.0.0.1",
                    "metadata_port_base": port,
                    "video_port_base": int(
                        os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000")
                    ),
                    "max_visible_streams": streams,
                },
            },
        }
    )
    env = dict(
        os.environ,
        HIGH_DENSITY_DETECTOR_MEASURE_FRAMES=str(frames),
        HIGH_DENSITY_DETECTOR_FRAMES_PER_STREAM="0",
    )
    messages = []
    with (
        MetadataJsonListener("127.0.0.1", port, num_ports=streams) as listener,
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        process = executor.submit(
            subprocess.run,
            [
                sys.executable,
                str(EXAMPLE_DIR / "src/python/main.py"),
                "--config",
                str(config),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000 if test_timeout_ms > 0 else None,
            check=False,
        )
        drain_until = float("inf")
        while True:
            metadata = listener.wait_for_messages(0.2)
            messages.extend(metadata.messages)
            if process.done():
                now = time.monotonic()
                if not metadata.messages or now >= drain_until:
                    break
                drain_until = min(drain_until, now + 1)
        result = process.result()
    assert result.returncode == 0, result.stdout + result.stderr
    received = [set() for _ in range(streams)]
    detected_streams = set()
    last_pts = [-1] * streams
    for message in messages:
        payload = json.loads(message.payload)
        index = message.port - port
        assert payload["stream_index"] == index
        assert payload["stream_id"] == f"stream{index}"
        assert payload["frame_id"] and payload["pts_ns"] >= 0
        assert "rtp_timestamp" in payload
        assert payload["frame_id"] not in received[index], "duplicate frame identity"
        assert payload["pts_ns"] > last_pts[index], "non-increasing per-stream timestamp"
        last_pts[index] = payload["pts_ns"]
        received[index].add(payload["frame_id"])
        if message.object_count > 0:
            detected_streams.add(index)
    assert detected_streams == set(range(streams)), "missing detections on one or more streams"
    summaries = [
        json.loads(line.removeprefix("[measurement] "))
        for line in result.stdout.splitlines()
        if line.startswith("[measurement] ")
    ]
    assert len(summaries) == 1, result.stdout
    summary = summaries[0]
    print(f"{codec}: {json.dumps(summary)}")
    assert summary["frames"] == frames
    assert len(summary["per_stream_frames"]) == streams
    assert sum(summary["per_stream_frames"]) == frames
    assert summary["elapsed_s"] > 0
    assert summary["aggregate_fps"] == pytest.approx(frames / summary["elapsed_s"])
    minimum_fps = 28.125 if fps == 30 else 0.95 * fps
    assert summary["aggregate_fps"] > streams * minimum_fps
    assert min(summary["per_stream_frames"]) / summary["elapsed_s"] > minimum_fps
    assert summary["per_stream_send_failures"] == [0] * streams
    assert [len(frames) for frames in received] == summary["per_stream_total_sent"]
