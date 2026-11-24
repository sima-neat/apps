"""Unit tests for the tiny-drone tracker: the checks the application leaves to its config, and the
tracker's matching rules. tests/cpp/test_unit.cpp runs the same tests in C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "yolo26_tiny_drone_tracker_main")
tracker = main_module.tracker

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class Box:
    """A detection as pyneat.detections.decode_bbox_tensor returns it."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.9
    class_id: int = 0


def settings(**changes) -> tracker.TrackerSettings:
    values = {
        "high_score_threshold": 0.5,
        "new_track_threshold": 0.5,
        "max_missing_frames": 30,
        "min_confirmed_hits": 1,
    }
    return tracker.TrackerSettings(**(values | changes))


FPS = 30
FRAME_NS = 33_333_333


def pts(frame: int) -> int:
    """The presentation time of a 30 FPS stream's frame, from an origin that is not 0."""
    return 5_000_000_000 + frame * FRAME_NS


def ids(tracks) -> list[int]:
    return [track.track_id for track in tracks]


def test_shipped_config_is_usable():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    # One to four streams, each on its own Insight channel.
    assert 1 <= len(config.string_list("input.source")) <= 4
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26n_p2_tiny_drone_int8_qat_b1_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.nms_iou") <= 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    # Read for its type only: true and false are both valid.
    config.boolean("input.rtsp.tcp")
    tracking = main_module.tracker_settings(config)
    # Detections below inference.min_score never reach the tracker, so a lower
    # high_score_threshold would make every detection a high-score one.
    assert (
        0
        <= config.number("inference.min_score")
        <= tracking.high_score_threshold
        <= tracking.new_track_threshold
        <= 1
    )
    assert tracking.max_missing_frames >= 0
    assert tracking.min_confirmed_hits >= 1


def test_nearby_detection_keeps_its_track_id():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], pts(0))) == [1]
    assert ids(object_tracker.update([Box(12, 11, 52, 81, 0.8)], pts(1))) == [1]


def test_higher_affinity_wins_then_lower_track_id():
    def two_tracks(first: Box, second: Box) -> tracker.ObjectTracker:
        object_tracker = tracker.ObjectTracker(settings(), FPS)
        object_tracker.update([first, second], pts(0))
        return object_tracker

    # Affinity is IoU plus 1 / (1 + center distance). Track 1 overlaps the box with IoU 0.64 at
    # 0.11 diagonals, track 2 with IoU 0.04 at 0.08 diagonals: track 1 scores 1.54, track 2 0.96.
    big_and_near = two_tracks(Box(0, 0, 25, 25), Box(9, 9, 13, 13))
    assert ids(big_and_near.update([Box(0, 0, 20, 20)], pts(1))) == [1]
    # No overlap with either track: the closer one wins, at 0.85 diagonals against 1.27.
    apart = two_tracks(Box(0, 0, 10, 10), Box(30, 0, 40, 10))
    assert ids(apart.update([Box(18, 0, 28, 10)], pts(1))) == [2]
    # The same IoU (0.54) and center distance with both tracks: the lower ID wins.
    side_by_side = two_tracks(Box(0, 0, 10, 10), Box(6, 0, 16, 10))
    assert ids(side_by_side.update([Box(3, 0, 13, 10)], pts(1))) == [1]


@pytest.mark.parametrize("max_missing_frames, track_id", [(2, 1), (1, 2)])
def test_track_ends_after_max_missing_frames(max_missing_frames, track_id):
    object_tracker = tracker.ObjectTracker(settings(max_missing_frames=max_missing_frames), FPS)
    object_tracker.update([Box(10, 10, 50, 80)], pts(0))
    object_tracker.update([], pts(1))
    object_tracker.update([], pts(2))
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], pts(3))) == [track_id]


@pytest.mark.parametrize("max_missing_frames, track_id", [(2, 1), (1, 2)])
def test_dropped_frames_age_a_track(max_missing_frames, track_id):
    """The shared model's link drops frames 1 and 2, so the tracker sees frames 0 and 3."""
    object_tracker = tracker.ObjectTracker(settings(max_missing_frames=max_missing_frames), FPS)
    object_tracker.update([Box(10, 10, 50, 80)], pts(0))
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], pts(3))) == [track_id]


def test_predicted_motion_keeps_a_fast_box_on_its_track():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    object_tracker.update([Box(0, 0, 10, 10)], pts(0))
    # 40 px is 2.8 box diagonals, within the 3-diagonal gate, and sets a velocity of 10 px.
    assert ids(object_tracker.update([Box(40, 0, 50, 10)], pts(1))) == [1]
    # 44 px from the last box is 3.1 diagonals, outside the gate. The predicted box is 34 px
    # away, 2.4 diagonals, so the track continues.
    assert ids(object_tracker.update([Box(84, 0, 94, 10)], pts(2))) == [1]


@pytest.mark.parametrize("third_box", [Box(90, 0, 100, 10), Box(10, 0, 20, 10)])
def test_velocity_takes_a_quarter_of_the_measured_motion(third_box):
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    object_tracker.update([Box(0, 0, 10, 10)], pts(0))
    object_tracker.update([Box(40, 0, 50, 10)], pts(1))
    # The 40 px move sets a velocity of 10 px, so the box is predicted at x 50..60. Both third
    # boxes are 40 px from there, within the 42.4 px gate. A velocity below 7.6 px loses the
    # first, and one above 12.4 px loses the second.
    assert ids(object_tracker.update([third_box], pts(2))) == [1]


def test_predicted_motion_spans_dropped_frames():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    object_tracker.update([Box(0, 0, 10, 10)], pts(0))
    object_tracker.update([Box(40, 0, 50, 10)], pts(1))
    # Frames 2 and 3 were dropped. Three frames at 10 px put the predicted box at x 70..80, 40 px
    # from this one. The box is 70 px from the last one, outside the 42.4 px gate.
    assert ids(object_tracker.update([Box(110, 0, 120, 10)], pts(4))) == [1]


def test_velocity_is_measured_per_frame():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    object_tracker.update([Box(0, 0, 10, 10)], pts(0))
    # Frame 1 was dropped, so the 40 px move is 20 px per frame, and the velocity 5 px.
    object_tracker.update([Box(40, 0, 50, 10)], pts(2))
    # The box is 40 px from the predicted x 45..55, within the gate. A velocity of 10 px would
    # put it 45 px away, outside.
    assert ids(object_tracker.update([Box(5, 0, 15, 10)], pts(3))) == [1]


def test_score_at_high_score_threshold_confirms_a_track():
    object_tracker = tracker.ObjectTracker(settings(min_confirmed_hits=2), FPS)
    assert object_tracker.update([Box(0, 0, 10, 10, 0.9)], pts(0)) == []
    assert ids(object_tracker.update([Box(0, 0, 10, 10, 0.5)], pts(1))) == [1]


@pytest.mark.parametrize("max_missing_frames, track_id", [(2, 1), (1, 2)])
def test_frame_without_timestamp_counts_as_the_next_frame(max_missing_frames, track_id):
    object_tracker = tracker.ObjectTracker(settings(max_missing_frames=max_missing_frames), FPS)
    object_tracker.update([Box(10, 10, 50, 80)], pts(0))
    object_tracker.update([], -1)
    object_tracker.update([], -1)
    # Frame 3 is 2 frames past the track's last match.
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], -1)) == [track_id]


def test_timestamp_before_the_last_counts_as_the_next_frame():
    object_tracker = tracker.ObjectTracker(settings(max_missing_frames=0), FPS)
    object_tracker.update([Box(10, 10, 50, 80)], pts(10))
    # The source restarted, and its timestamps start over. Frame 1 follows frame 0 directly.
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], pts(0))) == [1]


@pytest.mark.parametrize("max_missing_frames, track_id", [(2, 1), (1, 2)])
def test_timestamps_after_a_restart_count_from_it(max_missing_frames, track_id):
    object_tracker = tracker.ObjectTracker(settings(max_missing_frames=max_missing_frames), FPS)
    object_tracker.update([Box(10, 10, 50, 80)], pts(10))
    object_tracker.update([Box(10, 10, 50, 80)], pts(0))
    # The restart at pts(0) is frame 1, so pts(3) is frame 4, 2 frames past the last match.
    assert ids(object_tracker.update([Box(10, 10, 50, 80)], pts(3))) == [track_id]


def test_box_beyond_the_distance_gate_starts_a_new_track():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    object_tracker.update([Box(0, 0, 10, 10)], pts(0))
    # 44 px is 3.1 box diagonals from the predicted box, just outside the 3-diagonal gate.
    assert ids(object_tracker.update([Box(44, 0, 54, 10)], pts(1))) == [2]


def test_track_is_reported_from_its_min_confirmed_hits_th_detection():
    object_tracker = tracker.ObjectTracker(settings(min_confirmed_hits=3), FPS)
    assert object_tracker.update([Box(0, 0, 10, 10)], pts(0)) == []
    assert object_tracker.update([Box(0, 0, 10, 10)], pts(1)) == []
    assert ids(object_tracker.update([Box(0, 0, 10, 10)], pts(2))) == [1]


def test_low_score_detection_continues_only_a_confirmed_track():
    object_tracker = tracker.ObjectTracker(settings(min_confirmed_hits=2), FPS)
    high = Box(0, 0, 10, 10, 0.9)
    low = Box(0, 0, 10, 10, 0.2)
    assert object_tracker.update([high], pts(0)) == []
    # A low score cannot confirm the new track.
    assert object_tracker.update([low], pts(1)) == []
    assert ids(object_tracker.update([high], pts(2))) == [1]
    # Once confirmed, the track continues on a low score.
    assert ids(object_tracker.update([low], pts(3))) == [1]
    assert object_tracker.tracks_started() == 1


def test_new_track_needs_new_track_threshold():
    object_tracker = tracker.ObjectTracker(
        settings(high_score_threshold=0.2, new_track_threshold=0.3), FPS
    )
    assert object_tracker.update([Box(0, 0, 10, 10, 0.25)], pts(0)) == []
    assert object_tracker.tracks_started() == 0
    assert ids(object_tracker.update([Box(0, 0, 10, 10, 0.3)], pts(1))) == [1]


def test_tracks_stop_at_max_active_tracks():
    object_tracker = tracker.ObjectTracker(settings(), FPS)
    boxes = [Box(20 * index, 0, 20 * index + 10, 10) for index in range(300)]
    assert len(object_tracker.update(boxes, pts(0))) == tracker.MAX_ACTIVE_TRACKS == 256
    assert object_tracker.tracks_started() == 256
