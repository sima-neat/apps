"""End-to-end test for the Python SuperPoint example."""

import re
import subprocess
import sys
from pathlib import Path

import cv2
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.e2e
def test_video_pipeline(
    apps_root,
    e2e_model_path,
    e2e_config_writer,
    tmp_output_dir,
    test_timeout_ms,
):
    input_video = apps_root / "assets" / "datasets" / "tum-rgbd" / "freiburg1-desk.mp4"
    assert input_video.is_file()
    output_video = tmp_output_dir / "annotated.mp4"
    config = e2e_config_writer(
        {
            "model": {"path": str(e2e_model_path)},
            "io": {"input": str(input_video), "output": str(output_video)},
            "runtime": {"frames": 8},
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=apps_root,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    match = re.search(
        r"frames=8 average_points=([0-9]+(?:\.[0-9]+)?) descriptor_dim=256 ",
        result.stdout,
    )
    assert match, result.stdout
    assert 0 < float(match.group(1)) <= 600

    video = cv2.VideoCapture(str(output_video))
    ok, frame = video.read()
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_value = int(video.get(cv2.CAP_PROP_FOURCC))
    video.release()
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    assert ok
    assert frame.shape[:2] == (480, 640)
    assert frame_count == 8
    assert fourcc == "avc1"
