"""Contract tests for the shared e2e output assertions.

These guard the helpers themselves: an assertion that cannot fail is worse than
no assertion, because it reads like coverage. Each test drives the helper with
output an application would produce when it is broken in a way the previous
`st_size > 0` check waved through.

The harness side of the same contract is here too, because the assertions are
only about the application if the harness stopped it between writes.
"""

import sys

import numpy as np
import pytest

import cv2

from tests.utils.output_assertions import (
    assert_every_stream_advances,
    assert_frames_decode,
    assert_saved_frames_are_usable,
    group_frames_by_stream,
    saved_image_files,
)
from tests.utils.pytest_fixtures import (
    _confirm_finished_outputs,
    _discard_unfinished_writes,
    _output_sizes,
)


# A producer that takes a visible amount of time over each frame, the way a
# large JPEG does on a loaded device, and keeps running afterwards like the
# streaming applications do. Anything that stops it on first sight of a file
# catches one of these part-written.
_SLOW_PRODUCER = """
import sys
import time
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1])
for index in range(40):
    image = np.full((48, 64, 3), (index * 11) % 255, dtype=np.uint8)
    data = cv2.imencode(".jpg", image)[1].tobytes()
    with open(out / f"stream_{index % 2}_frame_{index}.jpg", "wb") as handle:
        handle.write(data[: len(data) // 2])
        handle.flush()
        time.sleep(0.3)
        handle.write(data[len(data) // 2 :])
    time.sleep(0.2)
time.sleep(60)
"""


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
            assert_every_stream_advances(frames, paths)

    def test_frames_that_differ_are_accepted(self, tmp_path):
        paths = [tmp_path / f"frame_{i}.jpg" for i in range(3)]
        for index, path in enumerate(paths):
            _write_frame(path, shade=index * 60)

        frames = assert_frames_decode(paths)
        assert_every_stream_advances(frames, paths)

    def test_a_single_frame_run_is_not_treated_as_stalled(self, tmp_path):
        path = tmp_path / "frame_1.jpg"
        _write_frame(path)

        assert_every_stream_advances(assert_frames_decode([path]), [path])


class TestAssertEveryStreamAdvances:
    """A multi-stream run writes every stream into one directory.

    Whole-directory comparison is satisfied by a single moving stream, so these
    drive the helper with the output of a run where only some streams work.
    """

    def _write_streams(self, tmp_path, shades_per_stream):
        for stream, shades in enumerate(shades_per_stream):
            for index, shade in enumerate(shades):
                _write_frame(tmp_path / f"stream_{stream}_frame_{index}.jpg", shade=shade)

    def test_a_frozen_stream_beside_an_advancing_one_is_rejected(self, tmp_path):
        self._write_streams(tmp_path, [[100, 100, 100], [0, 60, 120]])

        with pytest.raises(AssertionError, match=r"stream 0 .*not advancing"):
            assert_saved_frames_are_usable(tmp_path, 6)

    def test_a_stream_showing_its_own_static_image_is_rejected(self, tmp_path):
        self._write_streams(tmp_path, [[0, 60, 120], [200, 200, 200]])

        with pytest.raises(AssertionError, match=r"stream 1 .*not advancing"):
            assert_saved_frames_are_usable(tmp_path, 6)

    def test_every_stream_advancing_is_accepted(self, tmp_path):
        self._write_streams(tmp_path, [[0, 60, 120], [10, 70, 130]])

        assert len(assert_saved_frames_are_usable(tmp_path, 6)) == 6

    def test_frames_are_grouped_by_the_stream_named_in_their_filename(self, tmp_path):
        self._write_streams(tmp_path, [[0, 60], [10, 70, 130]])
        paths = saved_image_files(tmp_path)

        groups = group_frames_by_stream(paths, assert_frames_decode(paths))

        assert {label: len(group) for label, group in groups.items()} == {
            "stream 0": 2,
            "stream 1": 3,
        }

    def test_frames_without_a_stream_in_their_name_stay_one_group(self, tmp_path):
        for index in range(3):
            _write_frame(tmp_path / f"frame_{index}.jpg", shade=index * 60)
        paths = saved_image_files(tmp_path)

        groups = group_frames_by_stream(paths, assert_frames_decode(paths))

        assert [len(group) for group in groups.values()] == [3]


class TestOutputCompletion:
    """The harness has to stop the producer between writes, not during one.

    These assertions are only about the application if the frames on disk are
    the ones it finished writing. `run_until_output_files` polls while the
    application writes frames straight to their final paths, so that is where a
    truncated frame would come from, and the decode assertion would blame it on
    the application.
    """

    def _truncate(self, path):
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 3])

    def test_a_file_seen_only_once_is_not_confirmed(self, tmp_path):
        _write_frame(tmp_path / "frame_0.jpg")

        assert _confirm_finished_outputs(_output_sizes(tmp_path), None, {}) == {}

    def test_a_file_still_growing_is_not_confirmed(self, tmp_path):
        path = tmp_path / "frame_0.jpg"
        _write_frame(path)
        previous = _output_sizes(tmp_path)
        _write_frame(path, size=(128, 96))

        assert _confirm_finished_outputs(_output_sizes(tmp_path), previous, {}) == {}

    def test_a_file_that_holds_still_but_does_not_decode_is_not_confirmed(self, tmp_path):
        path = tmp_path / "frame_0.jpg"
        _write_frame(path)
        self._truncate(path)
        previous = _output_sizes(tmp_path)

        assert _confirm_finished_outputs(_output_sizes(tmp_path), previous, {}) == {}

    def test_a_file_that_holds_still_and_decodes_is_confirmed(self, tmp_path):
        path = tmp_path / "frame_0.jpg"
        _write_frame(path)
        previous = _output_sizes(tmp_path)

        finished = _confirm_finished_outputs(_output_sizes(tmp_path), previous, {})

        assert finished == {path: path.stat().st_size}

    def test_a_confirmed_file_that_changes_again_is_unconfirmed(self, tmp_path):
        path = tmp_path / "frame_0.jpg"
        _write_frame(path)
        previous = _output_sizes(tmp_path)
        finished = _confirm_finished_outputs(_output_sizes(tmp_path), previous, {})
        _write_frame(path, size=(128, 96))

        assert _confirm_finished_outputs(_output_sizes(tmp_path), None, finished) == {}

    def test_a_frame_written_during_shutdown_is_discarded(self, tmp_path):
        _write_frame(tmp_path / "frame_0.jpg")
        finished = _confirm_finished_outputs(
            _output_sizes(tmp_path), _output_sizes(tmp_path), {}
        )
        # The frame the producer had started when our SIGTERM reached it.
        _write_frame(tmp_path / "frame_1.jpg")
        self._truncate(tmp_path / "frame_1.jpg")

        _discard_unfinished_writes(tmp_path, finished)

        assert [path.name for path in saved_image_files(tmp_path)] == ["frame_0.jpg"]

    def test_confirmed_frames_survive_the_discard(self, tmp_path):
        for index in range(2):
            _write_frame(tmp_path / f"frame_{index}.jpg", shade=index * 60)
        finished = _confirm_finished_outputs(
            _output_sizes(tmp_path), _output_sizes(tmp_path), {}
        )

        _discard_unfinished_writes(tmp_path, finished)

        assert len(saved_image_files(tmp_path)) == 2

    def test_a_slow_producer_is_never_stopped_part_way_through_a_frame(
        self, tmp_path, run_until_output_files
    ):
        producer = tmp_path / "producer.py"
        producer.write_text(_SLOW_PRODUCER)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        result = run_until_output_files(
            [sys.executable, str(producer), str(output_dir)],
            output_dir,
            4,
            60.0,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0
        assert_saved_frames_are_usable(output_dir, 4)


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
