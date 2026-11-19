"""End-to-end test: the shipped desk video in, the blue book's mask and position out. The same assertions as
tests/cpp/test_e2e.cpp."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from tests.utils.e2e_config import APPS_ROOT

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
VIDEO = APPS_ROOT / "assets" / "datasets" / "tum-rgbd" / "freiburg1-desk.mp4"
PACKAGES = {
    name: f"edgetam_w16a16_w8a16_w16a16_w8a8_{name}_mpk.tar.gz"
    for name in ("image_encoder", "memory_attention", "mask_decoder", "memory_encoder")
}
FRAMES = 100
SAVE_EVERY = 10
# The shipped prompt point is on a blue book. The camera pans, so the book crosses the frame and leaves it at
# frame 48. Each entry is a frame and the book's centre in it, measured on a Modalix DevKit.
BOOK_CENTERS = {0: (0.885, 0.388), 20: (0.479, 0.362), 40: (0.830, 0.473)}
CENTER_TOLERANCE = 0.06
LAST_FRAME_IN_VIEW = 40
FIRST_FRAME_OUT_OF_VIEW = 60


def model_files(models_dir: Path, skip_unless_e2e_ready) -> dict[str, str]:
    paths = {name: models_dir / file_name for name, file_name in PACKAGES.items()}
    for path in paths.values():
        skip_unless_e2e_ready(path.is_file(), f"missing EdgeTAM package: {path}")
    return {name: str(path) for name, path in paths.items()}


@pytest.mark.e2e
def test_follows_the_book_and_reports_when_it_leaves_the_view(
    models_dir, tmp_output_dir, test_timeout_ms, skip_unless_e2e_ready, e2e_config_writer
):
    config_path = e2e_config_writer(
        {
            "model": model_files(models_dir, skip_unless_e2e_ready),
            "source": {"video_path": str(VIDEO)},
            "inference": {"frames": FRAMES},
            "output": {"dir": str(tmp_output_dir), "save_every": SAVE_EVERY},
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert f"tracked {FRAMES} frames" in result.stdout

    frames = json.loads((tmp_output_dir / "results.json").read_text(encoding="utf-8"))["frames"]
    assert [frame["frame"] for frame in frames] == list(range(FRAMES))
    for frame in frames[: LAST_FRAME_IN_VIEW + 1]:
        assert frame["visible"], frame
        # The book covers 1.6% to 3.5% of the frame while it is in view.
        assert 0.008 < frame["area"] < 0.07, frame
    for index, (center_x, center_y) in BOOK_CENTERS.items():
        frame = frames[index]
        assert abs(frame["center_x"] - center_x) < CENTER_TOLERANCE, frame
        assert abs(frame["center_y"] - center_y) < CENTER_TOLERANCE, frame
    for frame in frames[FIRST_FRAME_OUT_OF_VIEW:]:
        assert frame == {"frame": frame["frame"], "visible": False, "area": 0.0, "center_x": None,
                         "center_y": None}, frame

    overlays = [f"frame_{index:06d}.jpg" for index in range(0, FRAMES, SAVE_EVERY)]
    assert sorted(path.name for path in tmp_output_dir.iterdir()) == [*overlays, "results.json"]
    for overlay in overlays:
        image = cv2.imread(str(tmp_output_dir / overlay))
        assert image is not None and image.shape == (480, 640, 3), overlay


@pytest.mark.e2e
def test_another_prompt_point_follows_another_object(
    models_dir, tmp_output_dir, test_timeout_ms, skip_unless_e2e_ready, e2e_config_writer
):
    # The point is on the monitor, which is far larger than the book.
    config_path = e2e_config_writer(
        {
            "model": model_files(models_dir, skip_unless_e2e_ready),
            "source": {"video_path": str(VIDEO)},
            "prompt": {"x": 0.42, "y": 0.2},
            "inference": {"frames": 5},
            "output": {"dir": str(tmp_output_dir), "save_every": 0},
        }
    )

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config_path)],
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=str(EXAMPLE_DIR),
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    frames = json.loads((tmp_output_dir / "results.json").read_text(encoding="utf-8"))["frames"]
    assert len(frames) == 5
    for frame in frames:
        assert frame["visible"] and frame["area"] > 0.08, frame
        assert abs(frame["center_x"] - 0.42) < 0.15 and abs(frame["center_y"] - 0.2) < 0.15, frame
    # save_every 0 saves no image.
    assert [path.name for path in tmp_output_dir.iterdir()] == ["results.json"]
