"""End-to-end tests for the Python SuperPoint feature matcher on a DevKit.

Every source runs with LightGlue, and SuperGlue runs on RTSP H.264: the source graph does not
depend on the matcher, and the matcher does not depend on the source."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.pytest_fixtures import usb_camera_cases

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
FRAMES = 30
SAVE_EVERY = 10
MODEL_FILES = {
    "superpoint": "superpoint_w8a8_mpk.tar.gz",
    "lightglue": "lightglue_superpoint_w8a8_mpk.tar.gz",
    "superglue": "superglue_outdoor_w16a16_mpk.tar.gz",
}
MATCHER_NAMES = {"lightglue": "LightGlue", "superglue": "SuperGlue"}
RTSP_H264 = {"type": "rtsp", "codec": "h264", "url_fixture": "rtsp_h264_url"}
SOURCES = [
    pytest.param(RTSP_H264, id="rtsp-h264"),
    pytest.param(
        {"type": "rtsp", "codec": "h265", "url_fixture": "rtsp_h265_url"},
        id="rtsp-h265",
    ),
    pytest.param(
        {"type": "rtsp", "codec": "mjpeg", "url_fixture": "rtsp_mjpeg_url", "fps": 30},
        id="rtsp-mjpeg",
    ),
    pytest.param(
        {
            "type": "http",
            "url_fixture": "http_mjpeg_url",
            "fps": 30,
            "ssl_strict": False,
        },
        id="http-mjpeg",
    ),
] + usb_camera_cases({"type": "usb", "url_fixture": "usb_camera_id", "fps": 30})
# MIPI capture needs a configured camera on the DevKit, which CI runners do not have.
if os.environ.get("SIMA_NEAT_TEST_MIPI_CAMERA") == "1":
    SOURCES.append(pytest.param({"type": "mipi"}, id="mipi-camera"))
CASES = [
    pytest.param("lightglue", source.values[0], id=f"lightglue-{source.id}")
    for source in SOURCES
]
CASES.append(pytest.param("superglue", RTSP_H264, id="superglue-rtsp-h264"))


def source_section(request, source: dict) -> dict:
    """The config's source section for one case; a USB or MIPI case's camera stands in for a URL."""
    if source["type"] == "mipi":
        section = {"camera": os.environ.get("SIMA_NEAT_MIPI_CAMERA_NAME", "")}
    else:
        value = request.getfixturevalue(source["url_fixture"])
        section = {"camera" if source["type"] == "usb" else "url": value}
    if "codec" in source:
        section["codec"] = source["codec"]
    if "ssl_strict" in source:
        section["ssl_strict"] = source["ssl_strict"]
    if "fps" in source:
        section["fps"] = source["fps"]
    return {"type": source["type"], source["type"]: section}


@pytest.mark.e2e
@pytest.mark.parametrize(("matcher", "source"), CASES)
def test_insight_pipeline(
    request,
    matcher,
    source,
    apps_root,
    models_dir,
    e2e_config_writer,
    tmp_output_dir,
    skip_unless_e2e_ready,
    test_timeout_ms,
):
    import cv2

    models = {name: models_dir / file for name, file in MODEL_FILES.items()}
    for name in ("superpoint", matcher):
        skip_unless_e2e_ready(
            models[name].is_file(), f"model not found: {models[name]}"
        )
    insight_host = os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_HOST", "127.0.0.1")
    video_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000"))
    config = e2e_config_writer(
        {
            "superpoint": {"path": str(models["superpoint"])},
            "matcher": {"type": matcher, matcher: {"path": str(models[matcher])}},
            "source": source_section(request, source),
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

    stream = re.search(
        rf"source={source['type']} stream=(\d+)x(\d+)@\d+ matcher=(\w+) ", result.stdout
    )
    assert stream, result.stdout
    assert stream.group(3) == MATCHER_NAMES[matcher]
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
