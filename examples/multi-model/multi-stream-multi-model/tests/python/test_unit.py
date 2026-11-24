"""Unit tests for the multi-stream multi-model example: the checks the application leaves to its
config, the Insight metadata of each task, and the pending-result deadline.
tests/cpp/test_unit.cpp runs the same tests in C++."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main
from tests.utils.fake_run import FakeRun

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
APPS_ROOT = EXAMPLE_DIR.parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "multi_stream_multi_model_main")
insight_metadata = main_module.insight_metadata
PendingResults = main_module.PendingResults

pytestmark = pytest.mark.unit

LABELS = ["person", "bicycle", "car"]


def test_shipped_config_is_usable():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    sources = config.string_list("input.source")
    assert 1 <= len(sources) <= 4, "the shipped config lists one to four streams"
    model_paths = config.string_list("app.model_paths")
    model_ids = config.string_list("app.model_ids")
    assert len(model_paths) == len(model_ids) == len(sources), "one model per stream"
    assert model_ids == ["yolov8", "yolo26-seg", "yolo26-pose", "yolo26"]
    assert set(model_ids) <= set(main_module.MODEL_IDS)
    for path in model_paths:
        # The packaged path reaches prebuilt-apps/models from the config file.
        assert (COMMON_DIR / path).resolve().is_relative_to(APPS_ROOT / "models"), path
    assert [Path(path).name for path in model_paths] == [
        "yolo11s_w8a8_mpk.tar.gz",
        "yolo26s_seg_w8a16_mpk.tar.gz",
        "yolo26m_pose_w8a8_mpk.tar.gz",
        "yolo26m_w8a8_mpk.tar.gz",
    ]
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    labels = (APPS_ROOT / config.string("model.labels")).read_text().splitlines()
    assert len(labels) == 80, "the COCO labels the detection and segmentation models score"
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames", "output.insight.video_port_base",
                "output.insight.metadata_port_base"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def test_objects_clamp_boxes_to_the_frame():
    boxes = [
        SimpleNamespace(x1=10.7, y1=20.2, x2=110.9, y2=220.5, score=0.91234, class_id=2),
        SimpleNamespace(x1=1200.0, y1=700.0, x2=1400.0, y2=800.0, score=0.5, class_id=7),
    ]
    assert insight_metadata.objects(boxes, LABELS, 1280, 720) == [
        {"id": "obj_1", "label": "car", "confidence": 0.912, "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"id": "obj_2", "label": "unknown", "confidence": 0.5, "bbox": [1200.0, 700.0, 80.0, 20.0]},
    ]


def test_mask_rect_maps_the_frame_through_the_letterbox():
    # A 1920x1080 frame fills the 640x640 input at 1/3 scale, with 140 rows of padding above.
    assert insight_metadata.mask_rect((0, 0, 1920, 1080), 1920, 1080, 160, 160) == (0, 35, 160, 125)
    assert insight_metadata.mask_rect((960, 540, 1920, 1080), 1920, 1080, 160, 160) == (
        80, 80, 160, 125
    )


def test_segment_outline_follows_the_mask():
    # In a 640x640 frame one mask cell is 4x4 pixels. The object covers cells 20-59 by 40-79,
    # pixels 80-239 by 160-319.
    masks = np.zeros((2, 160, 160), dtype=np.uint8)
    masks[0, 40:80, 20:60] = 255
    boxes = np.array(
        [[64.0, 144.0, 256.0, 336.0, 0.80004, 1.0], [0.0, 0.0, 100.0, 100.0, 0.9, 0.0]],
        dtype=np.float32,
    )
    # The second detection's mask is empty, so it has no outline and is left out.
    [segment] = insight_metadata.segments(boxes, masks, LABELS, 640, 640)
    outline = segment.pop("mask")
    assert segment == {
        "id": "seg_1",
        "label": "bicycle",
        "confidence": 0.8,
        "bbox": [64, 144, 192, 192],
        "mask_format": "polygon",
    }
    # The resize rounds the square's corners, so the outline has four corners, each within a
    # pixel of the square's, and spans the square exactly.
    assert len(outline) == 4
    xs = [x for x, _ in outline]
    ys = [y for _, y in outline]
    assert (min(xs), max(xs), min(ys), max(ys)) == (80, 239, 160, 319)


def test_poses_round_to_whole_pixels_and_name_the_keypoints():
    boxes = np.array([[10.5, 20.4, 110.5, 220.6, 0.87654, 0.0]], dtype=np.float32)
    keypoints = np.array(
        [[[k * 10 + 0.5, k * 5 + 0.4, 0.5] for k in range(17)]], dtype=np.float32
    )
    [pose] = insight_metadata.poses(boxes, keypoints)
    assert pose["id"] == "pose_1"
    assert pose["label"] == "person"
    assert pose["confidence"] == 0.877
    assert pose["bbox"] == [11, 20, 100, 200]
    assert pose["keypoints"][0] == {"name": "nose", "x": 1, "y": 0, "confidence": 0.5}
    assert pose["keypoints"][16] == {"name": "right_ankle", "x": 161, "y": 80, "confidence": 0.5}
    assert [point["name"] for point in pose["keypoints"]] == list(
        insight_metadata.COCO_KEYPOINT_NAMES
    )


def test_decodes_with_fewer_rows_than_boxes_fail():
    boxes = np.zeros((2, 6), dtype=np.float32)
    with pytest.raises(RuntimeError) as error:
        insight_metadata.segments(boxes, np.zeros((1, 160, 160), np.uint8), LABELS, 640, 640)
    assert str(error.value) == "segmentation decode returned 2 boxes but 1 masks"
    with pytest.raises(RuntimeError) as error:
        insight_metadata.poses(boxes, np.zeros((1, 17, 3), np.float32))
    assert str(error.value) == "pose decode returned 2 boxes but 1 keypoint sets"


def test_data_json_leaves_out_the_lowest_confidence_entries():
    # Two entries fit the 32768-byte budget, three do not.
    entries = [
        {"id": "a", "confidence": 0.5, "pad": "x" * 15000},
        {"id": "b", "confidence": 0.9, "pad": "x" * 15000},
        {"id": "c", "confidence": 0.7, "pad": "x" * 15000},
    ]
    data, left_out = insight_metadata.data_json("objects", entries)
    assert left_out == 1
    assert [entry["id"] for entry in json.loads(data)["objects"]] == ["b", "c"]
    assert insight_metadata.data_json("poses", []) == ('{"poses":[]}', 0)


def started(stream_index: int, model_path: str) -> PendingResults:
    """A deadline of 10 s after the stream's first result."""
    pending = PendingResults(stream_index, model_path, 10.0)
    pending.result_arrived()
    return pending


def test_pending_result_past_the_deadline_names_the_stream_and_model():
    pending = started(2, "models/m.tar.gz")
    pending.frame_arrived(100.0)
    pending.check(110.0)
    with pytest.raises(RuntimeError) as error:
        pending.check(110.5)
    assert str(error.value) == (
        "stream 2: model models/m.tar.gz returned no result for 10 s while its camera sent frames"
    )


def test_later_frames_keep_the_first_pending_time():
    pending = started(0, "m.tar.gz")
    pending.frame_arrived(100.0)
    pending.frame_arrived(105.0)
    with pytest.raises(RuntimeError):
        pending.check(110.5)


def test_result_clears_the_pending_frame():
    pending = started(0, "m.tar.gz")
    pending.frame_arrived(100.0)
    pending.result_arrived()
    pending.check(200.0)
    pending.frame_arrived(205.0)
    pending.check(215.0)
    with pytest.raises(RuntimeError):
        pending.check(215.5)


def test_frames_before_the_first_result_are_not_pending():
    # The decoder may wait for a keyframe before the model sees a frame.
    pending = PendingResults(0, "m.tar.gz", 10.0)
    pending.frame_arrived(100.0)
    pending.check(1_000_000.0)
    pending.result_arrived()
    pending.frame_arrived(1_000_001.0)
    with pytest.raises(RuntimeError):
        pending.check(1_000_011.5)


def test_paused_camera_waits_without_a_limit():
    pending = started(0, "m.tar.gz")
    pending.check(1_000_000.0)
    pending.frame_arrived(100.0)
    pending.result_arrived()
    pending.check(1_000_000.0)


class StoppingRun(FakeRun):
    """A FakeRun that sets `stop` once its script ends, instead of failing the test."""

    def __init__(self, stop: threading.Event, *outcomes) -> None:
        super().__init__(*outcomes)
        self.stop = stop

    def pull(self, output_name: str, timeout_ms: int = 0):
        if not self._outcomes:
            self.stop.set()
            return None
        return super().pull(output_name, timeout_ms)


def fake_stream(deadline_s: float):
    """A stream that has sent its first result, so its deadline runs."""
    pending = PendingResults(0, "models/m.tar.gz", deadline_s)
    pending.result_arrived()
    return main_module.Stream(
        info=main_module.RtspStreamInfo(640, 480, 30),
        metadata_type=main_module.DETECTION,
        metadata_sender=None,
        pending=pending,
    )


def test_send_results_stops_a_model_that_returns_nothing_while_frames_arrive():
    # A healthy run whose model never returns the result of the frame that arrived.
    stop = threading.Event()
    run = StoppingRun(stop, ("sample", object()), "timeout", "timeout", "timeout")
    with pytest.raises(RuntimeError) as error:
        main_module.send_results(run, [fake_stream(0.0)], LABELS, 50, 0, stop)
    assert str(error.value) == (
        "stream 0: model models/m.tar.gz returned no result for 0 s while its camera sent frames"
    )
    assert run.pulls == [("frames_0", 0), ("results_0", 0)] * 2


def test_send_results_lets_a_paused_camera_wait():
    # No frames and no results: the loop keeps waiting until the stop signal.
    stop = threading.Event()
    run = StoppingRun(stop, *["timeout"] * 20)
    main_module.send_results(run, [fake_stream(0.0)], LABELS, 50, 0, stop)
    assert len(run.pulls) == 20
