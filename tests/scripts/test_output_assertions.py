"""Contract tests for the shared e2e output assertions.

These guard the helpers themselves: an assertion that cannot fail is worse than
no assertion, because it reads like coverage. Each test drives the helper with
output an application would produce when it is broken in a way the previous
`st_size > 0` check waved through.
"""

import numpy as np
import pytest

import cv2

from tests.utils.output_assertions import (
    assert_frames_are_not_all_identical,
    assert_frames_decode,
    assert_saved_frames_are_usable,
    saved_image_files,
)


def _write_frame(path, *, shade=0, size=(64, 48)):
    width, height = size
    image = np.full((height, width, 3), shade, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


class TestSavedImageFiles:
    def test_config_artifacts_are_not_counted_as_output(self, tmp_path):
        _write_frame(tmp_path / "frame_1.jpg")
        (tmp_path / "config.yaml").write_text("source: {}\n")

        files = saved_image_files(tmp_path)

        assert [path.name for path in files] == ["frame_1.jpg"]

    def test_nested_output_directories_are_included(self, tmp_path):
        (tmp_path / "stream0").mkdir()
        _write_frame(tmp_path / "stream0" / "frame_1.jpg")

        assert len(saved_image_files(tmp_path)) == 1


class TestAssertFramesDecode:
    def test_truncated_file_is_rejected_although_it_is_not_empty(self, tmp_path):
        path = tmp_path / "frame_1.jpg"
        _write_frame(path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 3])

        assert path.stat().st_size > 0, "precondition: the old check would pass this"
        with pytest.raises(AssertionError, match="does not decode"):
            assert_frames_decode([path])

    def test_non_image_content_is_rejected(self, tmp_path):
        path = tmp_path / "frame_1.jpg"
        path.write_text("this is not a JPEG")

        with pytest.raises(AssertionError, match="does not decode"):
            assert_frames_decode([path])

    def test_degenerate_dimensions_are_rejected(self, tmp_path):
        path = tmp_path / "frame_1.jpg"
        _write_frame(path, size=(8, 8))

        with pytest.raises(AssertionError, match="implausible"):
            assert_frames_decode([path], min_side=16)

    def test_no_output_at_all_is_rejected(self):
        with pytest.raises(AssertionError, match="no output frames"):
            assert_frames_decode([])

    def test_valid_frames_are_returned_for_further_assertions(self, tmp_path):
        paths = [tmp_path / f"frame_{i}.jpg" for i in range(3)]
        for index, path in enumerate(paths):
            _write_frame(path, shade=index * 40)

        frames = assert_frames_decode(paths)

        assert len(frames) == 3
        assert all(frame.shape == (48, 64, 3) for frame in frames)


class TestAssertFramesAdvance:
    def test_a_stalled_pipeline_writing_one_frame_repeatedly_is_rejected(self, tmp_path):
        paths = [tmp_path / f"frame_{i}.jpg" for i in range(4)]
        for path in paths:
            _write_frame(path, shade=120)

        frames = assert_frames_decode(paths)
        with pytest.raises(AssertionError, match="not advancing"):
            assert_frames_are_not_all_identical(frames, paths)

    def test_frames_that_differ_are_accepted(self, tmp_path):
        paths = [tmp_path / f"frame_{i}.jpg" for i in range(3)]
        for index, path in enumerate(paths):
            _write_frame(path, shade=index * 60)

        frames = assert_frames_decode(paths)
        assert_frames_are_not_all_identical(frames, paths)

    def test_a_single_frame_run_is_not_treated_as_stalled(self, tmp_path):
        path = tmp_path / "frame_1.jpg"
        _write_frame(path)

        assert_frames_are_not_all_identical(assert_frames_decode([path]), [path])


class TestAssertSavedFramesAreUsable:
    def test_too_few_frames_is_reported_with_both_counts(self, tmp_path):
        _write_frame(tmp_path / "frame_1.jpg")

        with pytest.raises(AssertionError, match="expected at least 3 saved frames, got 1"):
            assert_saved_frames_are_usable(tmp_path, 3)

    def test_a_healthy_run_passes_and_returns_its_frames(self, tmp_path):
        for index in range(3):
            _write_frame(tmp_path / f"frame_{index}.jpg", shade=index * 50)

        paths = assert_saved_frames_are_usable(tmp_path, 3)

        assert len(paths) == 3
