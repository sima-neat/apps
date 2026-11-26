"""Contract tests for support/python/sima_examples/image_folder.py, the image helpers of the
examples that read a folder of image files: the image list, reading an image, the annotated PNG's
name, and drawing detections. The DETR example's C++ unit test runs the same cases on
support/cpp/sima_examples/image_folder.h.

The cases that read or draw an image need numpy and OpenCV and skip where they are missing, as in
the contract job, which installs only pytest and PyYAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    # Import OpenCV first when it is installed: test_rtsp_probe.py stubs cv2 with an empty module
    # when nobody has imported it, which would hide the real one from the cases below.
    import cv2  # noqa: F401
except ImportError:
    pass

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.image_folder import draw_detections, image_paths, output_name, read_image


def write_files(folder: Path, names: list[str]) -> None:
    for name in names:
        (folder / name).write_bytes(b"")


def test_image_paths_matches_suffixes_in_any_case(tmp_path: Path):
    write_files(tmp_path, ["a.jpg", "b.JPEG", "c.Png", "d.BMP", "e.gif", "f.txt", "jpg"])
    assert [path.name for path in image_paths(tmp_path)] == ["a.jpg", "b.JPEG", "c.Png", "d.BMP"]


def test_image_paths_lists_images_in_file_name_order(tmp_path: Path):
    # The order is by character code, so 10.jpg comes before 2.jpg and B.png before a.png.
    write_files(tmp_path, ["c.bmp", "b.jpg", "a.png", "B.png", "2.jpg", "10.jpg"])
    assert [path.name for path in image_paths(tmp_path)] == [
        "10.jpg",
        "2.jpg",
        "B.png",
        "a.png",
        "b.jpg",
        "c.bmp",
    ]


def test_image_paths_skips_subfolders_and_their_files(tmp_path: Path):
    write_files(tmp_path, ["a.jpg"])
    # A folder is not an image, whatever its name ends with, and images inside one are skipped.
    (tmp_path / "archive.jpg").mkdir()
    (tmp_path / "more").mkdir()
    write_files(tmp_path / "more", ["b.jpg"])
    assert image_paths(tmp_path) == [tmp_path / "a.jpg"]


def test_image_paths_names_the_folder_when_it_has_no_images(tmp_path: Path):
    write_files(tmp_path, ["notes.txt"])
    (tmp_path / "archive.jpg").mkdir()
    with pytest.raises(ValueError, match=r"no \.jpg, \.jpeg, \.png, or \.bmp files in "):
        image_paths(tmp_path)


def test_image_paths_names_a_missing_folder(tmp_path: Path):
    with pytest.raises(ValueError, match="image folder does not exist: .*absent"):
        image_paths(tmp_path / "absent")


def require_opencv():
    """The cv2 and numpy modules, or skip. Another contract test stubs cv2 with an empty module
    when OpenCV is missing, so a module without imread counts as missing."""
    cv2 = pytest.importorskip("cv2", reason="OpenCV is not installed")
    np = pytest.importorskip("numpy", reason="numpy is not installed")
    if not hasattr(cv2, "imread"):
        pytest.skip("OpenCV is not installed")
    return cv2, np


def box(x1, y1, x2, y2, score, class_id):
    return SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)


def test_output_name_puts_the_suffix_between_the_stem_and_png():
    assert output_name(Path("in/street.jpg"), "_detr") == "street_detr.png"
    assert output_name(Path("in/frame.PNG"), "_PNG") == "frame_PNG.png"
    assert output_name(Path("in/a.b.jpg"), "") == "a.b.png"


def test_read_image_returns_a_bgr_array_of_the_files_size(tmp_path: Path):
    cv2, np = require_opencv()
    path = tmp_path / "wide.png"
    assert cv2.imwrite(str(path), np.full((30, 64, 3), (10, 20, 30), np.uint8))
    image = read_image(path)
    assert image.shape == (30, 64, 3)
    assert tuple(image[0, 0]) == (10, 20, 30)


def test_read_image_names_a_file_it_cannot_read(tmp_path: Path):
    require_opencv()
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="cannot read image: .*broken.png"):
        read_image(broken)
    with pytest.raises(ValueError, match="cannot read image: .*absent.png"):
        read_image(tmp_path / "absent.png")


def test_draw_detections_draws_the_box_in_its_class_color_on_a_copy():
    _, np = require_opencv()
    image = np.zeros((60, 80, 3), np.uint8)
    annotated = draw_detections(image, [box(9.6, 20.4, 50.0, 55.0, 0.9, 1)], ["N/A", "person"])
    assert not image.any()
    # Class 1 is blue in BGR. x1 = 9.6 rounds to 10, and a 2-pixel line there covers columns 9
    # to 11; truncating to 9 would cover 8 to 10.
    assert tuple(annotated[40, 11]) == (255, 0, 0)
    assert tuple(annotated[40, 8]) == (0, 0, 0)
    assert tuple(annotated[40, 30]) == (0, 0, 0)


def test_draw_detections_repeats_the_colors_every_eight_classes():
    _, np = require_opencv()
    image = np.zeros((60, 80, 3), np.uint8)
    labels = [str(index) for index in range(10)]
    annotated = draw_detections(image, [box(10, 20, 50, 55, 0.9, 9)], labels)
    # Class 9 takes class 1's color.
    assert tuple(annotated[40, 10]) == (255, 0, 0)
