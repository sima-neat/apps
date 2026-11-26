"""Contract tests for support/python/sima_examples/image_folder.py, the image list of the examples
that read a folder of image files. The DETR example's C++ unit test runs the same cases on
support/cpp/sima_examples/image_folder.h.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.image_folder import image_paths


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
