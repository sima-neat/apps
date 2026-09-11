"""Run 16 real RTSP pipelines and verify Insight metadata and send throughput."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

EXAMPLE_DIR = Path(__file__).resolve().parents[2]


def select_source(urls, codec):
    errors = []
    for url in dict.fromkeys(urls):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,width,height,avg_frame_rate,has_b_frames",
                    "-of",
                    "json",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except subprocess.TimeoutExpired:
            errors.append("RTSP source probe timed out")
            continue
        if result.returncode:
            errors.append(result.stderr)
            continue
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            continue
        stream = streams[0]
        rate = stream.get("avg_frame_rate", "0/1")
        numerator, denominator = map(int, rate.split("/"))
        fps = numerator / denominator if denominator else 0
        if (
            stream.get("codec_name") == {"h264": "h264", "h265": "hevc"}[codec]
            and stream.get("width") == 1280
            and stream.get("height") == 720
            and stream.get("has_b_frames") == 0
            and (abs(fps - 30) < 0.001 or abs(fps - 30000 / 1001) < 0.001)
        ):
            print(
                f"source codec={codec} width=1280 height=720 fps={fps} unique_publishers=1"
            )
            return url
        errors.append(str(stream))
    pytest.fail(f"No 720p30 {codec} source without B-frames: {errors}")


@pytest.mark.e2e
@pytest.mark.parametrize("codec", ["h264", "h265"])
def test_metadata_throughput(
    codec, request, e2e_model_path, e2e_config_writer, test_timeout_ms
):
    url = select_source(request.getfixturevalue(f"rtsp_{codec}_urls"), codec)
    port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_METADATA_PORT", "9100"))
    config = e2e_config_writer(
        {
            "streams": [url] * 16,
            "input": {"codec": codec, "width": 1280, "height": 720, "fps": 0},
            "runtime": {"warmup_frames": 100, "profile": False},
            "output": {
                "video_enabled": True,
                "insight": {
                    "host": "127.0.0.1",
                    "metadata_port_base": port,
                    "video_port_base": int(
                        os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000")
                    ),
                    "max_visible_streams": 16,
                },
            },
        }
    )
    env = dict(
        os.environ,
        HIGH_DENSITY_DETECTOR_MEASURE_FRAMES="5000",
        HIGH_DENSITY_DETECTOR_FRAMES_PER_STREAM="0",
    )
    messages = []
    with (
        MetadataJsonListener("127.0.0.1", port, num_ports=16) as listener,
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
            timeout=test_timeout_ms / 1000,
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
    received = [set() for _ in range(16)]
    detected_streams = set()
    for message in messages:
        payload = json.loads(message.payload)
        index = message.port - port
        assert payload["stream_index"] == index
        assert payload["stream_id"] == f"stream{index}"
        assert payload["frame_id"] and payload["pts_ns"] >= 0
        assert "rtp_timestamp" in payload
        received[index].add(payload["frame_id"])
        if message.object_count > 0:
            detected_streams.add(index)
    assert detected_streams == set(range(16)), "missing detections on one or more streams"
    summaries = [
        json.loads(line.removeprefix("[measurement] "))
        for line in result.stdout.splitlines()
        if line.startswith("[measurement] ")
    ]
    assert len(summaries) == 1, result.stdout
    summary = summaries[0]
    print(f"{codec}: {json.dumps(summary)}")
    assert summary["frames"] == 5000
    assert len(summary["per_stream_frames"]) == 16
    assert sum(summary["per_stream_frames"]) == 5000
    assert summary["elapsed_s"] > 0
    assert summary["aggregate_fps"] == pytest.approx(5000 / summary["elapsed_s"])
    assert summary["aggregate_fps"] > 450
    assert min(summary["per_stream_frames"]) / summary["elapsed_s"] > 450 / 16
    assert [len(frames) for frames in received] == summary["per_stream_total_sent"]
