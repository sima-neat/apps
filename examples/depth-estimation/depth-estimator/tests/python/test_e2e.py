"""E2E tests for depth-estimator (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.output_assertions import (
    assert_saved_frames_are_usable,
    supported_image_files,
)

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

# Every depth_anything_v2_small variant in tests/test-scope.yaml predicts a 518x686 (height x width)
# depth map, and each saved PNG puts the resized input beside the depth colormap.
MODEL_HEIGHT, MODEL_WIDTH = 518, 686
# A depth map changes gradually between neighbouring pixels. Reading the model output with the wrong
# layout or stride scrambles it, which raises the mean neighbour difference to about half the spread
# of the values or more; a real depth map stays near 0.02.
MAX_NEIGHBOUR_DIFFERENCE_RATIO = 0.1
MIN_DEPTH_SPREAD = 10.0


def assert_depth_map_is_coherent(path: Path) -> None:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"{path.name} does not decode"
    assert image.shape[:2] == (MODEL_HEIGHT, 2 * MODEL_WIDTH), (
        f"{path.name} is {image.shape[1]}x{image.shape[0]}, expected the input and a "
        f"{MODEL_WIDTH}x{MODEL_HEIGHT} depth map side by side"
    )
    depth = cv2.cvtColor(image[:, MODEL_WIDTH:], cv2.COLOR_BGR2GRAY).astype(np.float32)
    spread = float(depth.std())
    assert spread >= MIN_DEPTH_SPREAD, f"{path.name} has a flat depth map (std {spread:.1f})"
    across = float(np.abs(np.diff(depth, axis=1)).mean())
    down = float(np.abs(np.diff(depth, axis=0)).mean())
    ratio = max(across, down) / spread
    assert ratio <= MAX_NEIGHBOUR_DIFFERENCE_RATIO, (
        f"{path.name} depth map is not spatially coherent: neighbour difference "
        f"{max(across, down):.1f} is {ratio:.2f} of its spread {spread:.1f}"
    )


@pytest.mark.e2e
class TestE2E:
    def test_full_pipeline(
        self,
        e2e_model_path,
        tmp_output_dir,
        test_images_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
    ):
        skip_unless_e2e_ready(
            test_images_dir.exists() and bool(supported_image_files(test_images_dir)),
            "test_images_dir has no images to process",
        )

        config_path = e2e_config_writer(
            {
                "io": {"input_dir": str(test_images_dir), "output_dir": str(tmp_output_dir)},
            }
        )

        result = subprocess.run(
            [
                sys.executable, str(MAIN_PY),
                "--config", str(config_path),
            ],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        saved = assert_saved_frames_are_usable(
            tmp_output_dir, len(supported_image_files(test_images_dir))
        )
        for path in saved:
            assert_depth_map_is_coherent(path)
