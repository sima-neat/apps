"""End-to-end tests for the Python SuperPoint example."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def load_example():
    spec = importlib.util.spec_from_file_location("superpoint_example", MAIN_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.e2e
def test_insight_pipeline(
    apps_root,
    e2e_model_path,
    e2e_config_writer,
    test_timeout_ms,
):
    input_video = apps_root / "assets" / "datasets" / "tum-rgbd" / "freiburg1-desk.mp4"
    assert input_video.is_file()
    insight_host = os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_HOST", "127.0.0.1")
    video_port = int(os.environ.get("SIMANEAT_APPS_TEST_INSIGHT_VIDEO_PORT", "9000"))
    config = e2e_config_writer(
        {
            "model": {"path": str(e2e_model_path)},
            "io": {"input": str(input_video)},
            "output": {
                "insight": {
                    "host": insight_host,
                    "video_port": video_port,
                    "channel": 0,
                }
            },
            "runtime": {"frames": 8},
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

    match = re.search(
        r"frames=8 average_points=([0-9]+(?:\.[0-9]+)?) descriptor_dim=256 ",
        result.stdout,
    )
    assert match, result.stdout
    assert 0 < float(match.group(1)) <= 600
    assert f"video_sender={insight_host}:{video_port}" in result.stdout


@pytest.mark.e2e
def test_non_model_resolution_runs_through_core_preproc(
    apps_root, e2e_model_path, test_timeout_ms
):
    import cv2
    import numpy as np
    import pyneat

    input_video = apps_root / "assets" / "datasets" / "tum-rgbd" / "freiburg1-desk.mp4"
    video = cv2.VideoCapture(str(input_video))
    ok, source = video.read()
    video.release()
    assert ok

    frame = cv2.resize(source, (960, 540), interpolation=cv2.INTER_LINEAR)
    example = load_example()
    options = example.model_options(pyneat, frame.shape[1], frame.shape[0])
    assert options.boxdecode_original_width == 0
    assert options.boxdecode_original_height == 0
    assert options.boxdecode_resize_mode is None
    model = pyneat.Model(
        str(e2e_model_path),
        options,
    )
    model_input = example.input_tensor(frame, np, pyneat)
    runner = model.build(
        [model_input],
        route_options=pyneat.ModelRouteOptions(),
        run_options=pyneat.RunOptions(),
    )
    try:
        output = runner.run([model_input], timeout_ms=test_timeout_ms)
        points = example.feature_points(
            output,
            frame.shape[1],
            frame.shape[0],
            np,
            pyneat,
        )
    finally:
        runner.close()

    assert points.ndim == 2 and points.shape[1:] == (2,)
    assert len(points) > 0
    assert np.all(points[:, 0] >= 0) and np.all(points[:, 0] < frame.shape[1])
    assert np.all(points[:, 1] >= 0) and np.all(points[:, 1] < frame.shape[0])
