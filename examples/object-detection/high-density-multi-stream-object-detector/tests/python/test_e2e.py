"""Run 16 real RTSP pipelines and verify Insight metadata and send throughput."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.utils.metadata_json_listener import MetadataJsonListener

sys.path.insert(0, str(Path(__file__).resolve().parent))
from received_metadata_tracker import ReceivedMetadataTracker

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


def validate_metadata_message(message, base_port):
    payload = json.loads(message.payload)
    index = message.port - base_port
    assert 0 <= index < 16, "metadata arrived on an unexpected port"
    assert payload["type"] == "object-detection"
    assert isinstance(payload["data"]["objects"], list)
    assert payload["stream_index"] == index
    assert payload["stream_id"] == f"stream{index}"
    assert payload["frame_id"] and payload["pts_ns"] >= 0
    assert "rtp_timestamp" in payload
    return index, payload


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
    start = time.monotonic()
    tracker = ReceivedMetadataTracker(
        16,
        100,
        5000,
        int(
            os.environ.get(
                "SIMANEAT_APPS_HIGH_DENSITY_INITIAL_PROGRESS_TIMEOUT_MS", "90000"
            )
        )
        / 1000,
        int(
            os.environ.get(
                "SIMANEAT_APPS_HIGH_DENSITY_STREAM_PROGRESS_TIMEOUT_MS", "30000"
            )
        )
        / 1000,
        start,
    )
    receiver_failure = ""
    forced_kill = False
    with MetadataJsonListener("127.0.0.1", port, num_ports=16) as listener:
        process = subprocess.Popen(
            [
                sys.executable,
                str(EXAMPLE_DIR / "src/python/main.py"),
                "--config",
                str(config),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = start + test_timeout_ms / 1000
        try:
            while not tracker.complete:
                metadata = listener.wait_for_messages(0.2)
                if metadata.error and not metadata.timed_out:
                    receiver_failure = f"metadata listener failed: {metadata.error}"
                    break
                for message in metadata.messages:
                    index, payload = validate_metadata_message(message, port)
                    tracker.observe(
                        index,
                        payload["frame_id"],
                        message.object_count > 0,
                        time.monotonic(),
                    )
                    if tracker.warmup_complete and not tracker.measurement_started:
                        # Timing includes the flush, so concurrent arrivals can only lower FPS.
                        tracker.start_measurement(time.monotonic())
                        drained = listener.drain_pending()
                        assert not drained.error, drained.error
                        for queued_message in drained.messages:
                            validate_metadata_message(queued_message, port)
                        break
                    if tracker.complete:
                        break

                now = time.monotonic()
                stalled = tracker.stalled_streams(now)
                if stalled:
                    phase = "ongoing" if tracker.measurement_started else "initial"
                    receiver_failure = (
                        f"{phase} metadata progress timeout; missing streams={stalled}; "
                        f"warmup={tracker.warmup_frames}; "
                        f"measured={tracker.measured_frames}"
                    )
                    break
                if process.poll() is not None:
                    receiver_failure = (
                        "application exited before the receiver reached its frame target"
                    )
                    break
                if now >= deadline:
                    receiver_failure = (
                        f"total test timeout; warmup={tracker.warmup_frames}; "
                        f"measured={tracker.measured_frames}"
                    )
                    break
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                forced_kill = True
                process.kill()
                stdout, stderr = process.communicate()

    assert not forced_kill, f"application required forced kill after SIGINT\n{stdout}{stderr}"
    assert process.returncode == 0, stdout + stderr
    assert not receiver_failure, f"{receiver_failure}\n{stdout}{stderr}"
    assert tracker.complete
    assert all(tracker.useful_detection), "missing useful detections on one or more streams"
    assert tracker.total_measured == 5000
    assert tracker.elapsed_s > 0
    aggregate_fps = tracker.total_measured / tracker.elapsed_s
    summary = {
        "frames": tracker.total_measured,
        "elapsed_s": tracker.elapsed_s,
        "aggregate_fps": aggregate_fps,
        "per_stream_frames": tracker.measured_frames,
    }
    print(f"{codec}: {json.dumps(summary)}")
    assert aggregate_fps > 450
    assert min(tracker.measured_frames) / tracker.elapsed_s > 28.125
