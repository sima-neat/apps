"""End-to-end test for the Python stereo depth estimator on a DevKit.

It needs a rectified stereo pair as two RTSP H.264 streams, named left then right in
SIMANEAT_TEST_STEREO_RTSP_H264_URLS."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
MODEL_FILE = "lightstereo_s_w16a16_mpk.tar.gz"
STREAMS_KEY = "SIMANEAT_TEST_STEREO_RTSP_H264_URLS"
FRAMES = 30
SAVE_EVERY = 10


@pytest.mark.e2e
def test_insight_pipeline(
    apps_root,
    models_dir,
    e2e_config_writer,
    tmp_output_dir,
    skip_unless_e2e_ready,
    test_timeout_ms,
):
    import cv2

    model = models_dir / MODEL_FILE
    skip_unless_e2e_ready(model.is_file(), f"model not found: {model}")
    urls = [url.strip() for url in os.environ.get(STREAMS_KEY, "").split(",") if url.strip()]
    skip_unless_e2e_ready(
        len(urls) == 2, f"set {STREAMS_KEY} to the left and the right stream of a stereo pair"
    )
    insight_host = os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_HOST", "127.0.0.1")
    video_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000"))
    config = e2e_config_writer(
        {
            "model": {"path": str(model)},
            "source": {"type": "rtsp", "rtsp": {"left_url": urls[0], "right_url": urls[1]}},
            "inference": {"frames": FRAMES},
            "output": {
                "save_dir": str(tmp_output_dir),
                "save_every": SAVE_EVERY,
                "insight": {"host": insight_host, "video_port": video_port},
            },
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=apps_root,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    assert re.search(r"source=rtsp cameras=\d+x\d+@\d+", result.stdout), result.stdout
    summary = re.search(
        rf"frames={FRAMES} depth_fps=([0-9.]+) insight={insight_host}:{video_port}-{video_port + 2}",
        result.stdout,
    )
    assert summary, result.stdout
    assert float(summary.group(1)) > 0

    saved = sorted(tmp_output_dir.glob("depth_*.jpg"))
    assert len(saved) == FRAMES // SAVE_EVERY
    for path in saved:
        heatmap = cv2.imread(str(path))
        assert heatmap is not None
        # A stereo pair has near and far points, so its disparity is not one flat color.
        assert heatmap.std() > 5, f"{path.name}: the disparity heatmap is flat"
