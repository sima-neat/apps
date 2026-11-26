"""Contract tests for support/python/sima_examples/image_io.py, the image reader of the examples
that read images from disk. The DETR example's C++ unit test runs the same cases on
support/cpp/sima_examples/image_io.h.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The contract job installs only pytest and PyYAML, so these tests skip there and run wherever
# numpy and OpenCV are installed.
cv2 = pytest.importorskip("cv2", reason="OpenCV is not installed")
np = pytest.importorskip("numpy", reason="numpy is not installed")

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.image_io import read_image


def test_read_image_returns_a_bgr_array_of_the_files_size(tmp_path: Path):
    path = tmp_path / "wide.png"
    assert cv2.imwrite(str(path), np.full((30, 64, 3), (10, 20, 30), np.uint8))
    image = read_image(path)
    assert image.shape == (30, 64, 3)
    assert tuple(image[0, 0]) == (10, 20, 30)


def test_read_image_names_a_file_it_cannot_read(tmp_path: Path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot read image: .*broken.png"):
        read_image(broken)
    with pytest.raises(ValueError, match="cannot read image: .*absent.png"):
        read_image(tmp_path / "absent.png")
