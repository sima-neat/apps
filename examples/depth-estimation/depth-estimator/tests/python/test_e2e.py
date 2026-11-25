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

# reference_depth/<image>.png holds the FP32 depth of each assets/datasets-test/coco image, from the
# upstream ONNX the Model Registry compiles (onnx-community/depth-anything-v2-small at 4472b736,
# sha256 afb6a5c2...), fed the registry's preprocessing: RGB / 255, cv2 INTER_CUBIC stretch to 686x518,
# ImageNet mean and std. Each map is shrunk with INTER_AREA to 172x130 and stored min-max scaled
# as 16-bit PNG.
REFERENCE_DIR = EXAMPLE_DIR / "tests" / "fixtures" / "reference_depth"
REFERENCE_WIDTH, REFERENCE_HEIGHT = 172, 130
# The test images are letterboxed. The model calls the black bars nearest, which would dominate the
# correlation, so only the image content is compared: pixels brighter than this gray level.
CONTENT_THRESHOLD = 16
# On the host, the FP32 model through the app's 8-bit colormap correlates at 1.00 with the reference,
# and bilinear instead of bicubic resizing or a square 518x518 input still gives 0.99 or more. Feeding
# raw 0-255 pixels without normalization peaks at 0.46, and reading a row-major map column-major at
# 0.04, so these floors pass quantized builds and fail the wrong input or layout.
MIN_IMAGE_CORRELATION = 0.8
MIN_MEAN_CORRELATION = 0.9
# A depth map changes gradually between neighbouring pixels. Reading the model output with the wrong
# layout or stride scrambles it, which raises the mean neighbour difference to about half the spread
# of the values or more; a real depth map stays near 0.02.
MAX_NEIGHBOUR_DIFFERENCE_RATIO = 0.1
MIN_DEPTH_SPREAD = 10.0


def colormap_depth(path: Path):
    """The 8-bit depth the app colored, recovered exactly from the right half of its saved PNG."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"{path.name} does not decode"
    assert image.shape[:2] == (MODEL_HEIGHT, 2 * MODEL_WIDTH), (
        f"{path.name} is {image.shape[1]}x{image.shape[0]}, expected the input and a "
        f"{MODEL_WIDTH}x{MODEL_HEIGHT} depth map side by side"
    )
    colored = image[:, MODEL_WIDTH:].astype(np.int64)
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    lut = cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO).reshape(256, 3).astype(np.int64)
    lut_keys = (lut[:, 0] << 16) | (lut[:, 1] << 8) | lut[:, 2]
    keys = (colored[..., 0] << 16) | (colored[..., 1] << 8) | colored[..., 2]
    order = np.argsort(lut_keys)
    found = np.searchsorted(lut_keys[order], keys).clip(0, 255)
    levels = order[found]
    assert (lut_keys[levels] == keys).all(), f"{path.name} depth half is not an INFERNO colormap"
    return levels.astype(np.float32)


def assert_depth_map_is_coherent(path: Path, depth) -> None:
    import numpy as np

    spread = float(depth.std())
    assert spread >= MIN_DEPTH_SPREAD, f"{path.name} has a flat depth map (std {spread:.1f})"
    across = float(np.abs(np.diff(depth, axis=1)).mean())
    down = float(np.abs(np.diff(depth, axis=0)).mean())
    ratio = max(across, down) / spread
    assert ratio <= MAX_NEIGHBOUR_DIFFERENCE_RATIO, (
        f"{path.name} depth map is not spatially coherent: neighbour difference "
        f"{max(across, down):.1f} is {ratio:.2f} of its spread {spread:.1f}"
    )


def reference_depth_for(image_path: Path):
    """The FP32 reference depth for an image, or None when this image set has no reference.

    The fixtures cover the bundled assets/datasets-test/coco images. A custom
    SIMANEAT_APPS_TEST_INPUT_DIR holds other images, which have no reference, so only the
    coherence checks apply to them. A reference that is present but not the expected size is a
    broken fixture rather than a custom image set, and fails.
    """
    import cv2

    reference_path = REFERENCE_DIR / f"{image_path.stem}.png"
    if not reference_path.is_file():
        return None
    reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    assert reference is not None and reference.shape == (REFERENCE_HEIGHT, REFERENCE_WIDTH), (
        f"{reference_path} is not a {REFERENCE_WIDTH}x{REFERENCE_HEIGHT} reference depth map"
    )
    return reference


def reference_correlation(image_path: Path, depth, reference) -> float:
    """Pearson correlation of the app's depth with the FP32 reference over the image content."""
    import cv2
    import numpy as np

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero((gray > CONTENT_THRESHOLD).astype(np.uint8)))
    rows, cols = gray.shape
    # Shrink the content box onto the reference grid, rounding inward so no bar pixels remain.
    x0 = -(-x * REFERENCE_WIDTH // cols)
    x1 = (x + w) * REFERENCE_WIDTH // cols
    y0 = -(-y * REFERENCE_HEIGHT // rows)
    y1 = (y + h) * REFERENCE_HEIGHT // rows
    small = cv2.resize(depth, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
    a = small[y0:y1, x0:x1].astype(np.float64).ravel()
    b = reference[y0:y1, x0:x1].astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


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
        capsys,
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

        assert REFERENCE_DIR.is_dir() and any(REFERENCE_DIR.glob("*.png")), (
            f"no reference depth fixtures at {REFERENCE_DIR}"
        )

        images = supported_image_files(test_images_dir)
        assert_saved_frames_are_usable(tmp_output_dir, len(images))
        correlations = {}
        for image_path in images:
            output = tmp_output_dir / f"{image_path.stem}.png"
            assert output.is_file(), f"no depth map saved for {image_path.name}"
            depth = colormap_depth(output)
            assert_depth_map_is_coherent(output, depth)
            reference = reference_depth_for(image_path)
            if reference is not None:
                correlations[image_path.name] = reference_correlation(image_path, depth, reference)

        mean = sum(correlations.values()) / len(correlations) if correlations else 0.0
        with capsys.disabled():
            if correlations:
                print(
                    f"\n[DEPTH] {e2e_model_path.name}: correlation with the FP32 reference "
                    f"mean={mean:.4f} min={min(correlations.values()):.4f} "
                    + " ".join(f"{name}={value:.4f}" for name, value in correlations.items())
                )
            else:
                print(
                    f"\n[DEPTH] {e2e_model_path.name}: {len(images)} coherent depth maps. None of "
                    f"these images has a reference depth in {REFERENCE_DIR.name}, so the FP32 "
                    f"comparison was skipped"
                )

        # A custom SIMANEAT_APPS_TEST_INPUT_DIR can hold images with no reference depth. The
        # coherence checks above still ran on every one; only the FP32 floors need a reference.
        if not correlations:
            return

        low = {name: value for name, value in correlations.items() if value < MIN_IMAGE_CORRELATION}
        assert not low, f"depth maps disagree with the FP32 reference: {low}"
        assert mean >= MIN_MEAN_CORRELATION, (
            f"mean correlation with the FP32 reference {mean:.3f} < {MIN_MEAN_CORRELATION}"
        )
