"""End-to-end tests for the Python SuperPoint feature matcher on a DevKit."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 30
SAVE_EVERY = 10
MODEL_FILES = {
    "superpoint": "superpoint_w8a8_mpk.tar.gz",
    "lightglue": "lightglue_superpoint_w8a8_mpk.tar.gz",
    "superglue": "superglue_outdoor_w16a16_mpk.tar.gz",
}


@pytest.mark.e2e
@pytest.mark.parametrize("matcher", ["lightglue", "superglue"])
def test_insight_pipeline(
    matcher,
    apps_root,
    models_dir,
    e2e_config_writer,
    tmp_output_dir,
    skip_unless_e2e_ready,
    test_timeout_ms,
    rtsp_h264_url,
):
    import cv2

    models = {name: models_dir / file for name, file in MODEL_FILES.items()}
    for name in ("superpoint", matcher):
        skip_unless_e2e_ready(models[name].is_file(), f"model not found: {models[name]}")
    insight_host = os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_HOST", "127.0.0.1")
    video_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000"))
    config = e2e_config_writer(
        {
            "superpoint": {"path": str(models["superpoint"])},
            "matcher": {"type": matcher, matcher: {"path": str(models[matcher])}},
            "source": {"type": "rtsp", "rtsp": {"url": rtsp_h264_url, "codec": "h264"}},
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

    stream = re.search(r"stream=(\d+)x(\d+)@\d+ matcher=(\w+) ", result.stdout)
    assert stream, result.stdout
    assert stream.group(3) == {"lightglue": "LightGlue", "superglue": "SuperGlue"}[matcher]
    summary = re.search(
        rf"frames={FRAMES} matched=(\d+) skipped=(\d+) average_matches=([0-9.]+) "
        r"inlier_ratio=([0-9.]+) output_fps=([0-9.]+)",
        result.stdout,
    )
    assert summary, result.stdout
    matched, skipped = int(summary.group(1)), int(summary.group(2))
    assert matched + skipped == FRAMES
    assert matched > 0, result.stdout
    assert 0 < float(summary.group(3)) <= 512
    assert float(summary.group(5)) > 0
    assert f"video_sender={insight_host}:{video_port}" in result.stdout

    saved = sorted(tmp_output_dir.glob("frame_*.jpg"))
    assert len(saved) == FRAMES // SAVE_EVERY
    width, height = int(stream.group(1)), int(stream.group(2))
    for path in saved:
        image = cv2.imread(str(path))
        assert image is not None and image.shape == (height, 2 * width, 3)
