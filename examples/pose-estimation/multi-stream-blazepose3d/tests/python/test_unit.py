"""Unit tests for the multi-stream YOLO26-to-BlazePose application."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
PYTHON_DIR = MAIN_PY.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import main

main.np = np
pytestmark = pytest.mark.unit


def write_config(tmp_path: Path, streams: list[dict]) -> Path:
    config = {
        "models": {"detector_path": "detector.tar.gz", "pose_path": "pose.tar.gz"},
        "streams": streams,
        "output": {"insight": {"host": "127.0.0.1"}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def stream(
    index: int, *, stream_id: str | None = None, channel: int | None = None
) -> dict:
    return {
        "id": stream_id or f"camera{index}",
        "url": f"rtsp://127.0.0.1/src{index}",
        "codec": "hevc" if index == 1 else "h264",
        "insight_channel": index if channel is None else channel,
    }


def test_cli_help_and_missing_config():
    help_result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--config" in help_result.stdout
    missing = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", "does-not-exist.yaml"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "config file not found" in missing.stderr


def test_dynamic_stream_configuration_accepts_more_than_four(tmp_path: Path):
    cfg = main.load_app_config(
        write_config(tmp_path, [stream(index) for index in range(5)])
    )
    assert len(cfg.streams) == 5
    assert cfg.streams[1].codec == "h265"
    assert cfg.streams[4].insight_channel == 4


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda values: values.__setitem__(1, stream(1, stream_id="camera0")),
            "ids must be unique",
        ),
        (
            lambda values: values.__setitem__(1, stream(1, channel=0)),
            "channels must be unique",
        ),
    ],
)
def test_duplicate_stream_fields_are_rejected(tmp_path: Path, mutator, message: str):
    streams = [stream(0), stream(1)]
    mutator(streams)
    with pytest.raises(ValueError, match=message):
        main.load_app_config(write_config(tmp_path, streams))


def test_roi_landmark_and_metadata_contract():
    box = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0, "score": 0.9, "class_id": 0}
    assert main.square_roi(box, 1.5) == (-10, 10, 60, 60)
    raw = np.zeros((39, 5), dtype=np.float32)
    raw[0] = [4.0, 8.0, 0.0, 2.0, -2.0]
    pose = main.decode_pose(raw, (2.0, 0.0, 10.0, 0.0, 3.0, 20.0), box, 2)
    assert pose["keypoints"][0]["x"] == pytest.approx(18.0)
    assert pose["keypoints"][0]["y"] == pytest.approx(44.0)
    assert pose["keypoints"][0]["confidence"] == pytest.approx(main.sigmoid(-2.0))
    data = main.poses_data([pose])
    assert data["poses"][0]["id"] == "pose_3"
    assert len(data["poses"][0]["keypoints"]) == 33
    assert data["poses"][0]["keypoints"][0]["name"] == "nose"
    json.dumps(data)


def test_frame_identity_falls_back_through_source_sequence_fields():
    assert main.select_frame_id(9, 8, 7, 6) == 9
    assert main.select_frame_id(-1, 8, 7, 6) == 8
    assert main.select_frame_id(-1, -1, 7, 6) == 7
    assert main.select_frame_id(-1, -1, -1, 6) == 6
