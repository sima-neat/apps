"""Unit tests: the config checks, the Insight metadata of each task, and the pending-result
deadline. tests/cpp/test_unit.cpp runs those that need no Run or main.py's tables."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

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
    models = config.mapping_list("model")
    assert len(models) == len(sources), "one model per stream"
    model_ids = [model.string("id") for model in models]
    model_paths = [model.string("path") for model in models]
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
    for model, model_id in zip(models, model_ids):
        if main_module.MODEL_IDS[model_id][1] == main_module.POSE:
            continue
        # The labels path starts at the working directory, the Apps root here.
        labels = (APPS_ROOT / model.string("labels")).read_text().splitlines()
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


def error_of(
    pending: PendingResults, frame_id: int | None, result_frame_id: int | None, now_s: float
) -> str:
    """The error `observe` raises for one round, or "" when it raises none."""
    try:
        pending.observe(frame_id, result_frame_id, now_s)
    except RuntimeError as error:
        return str(error)
    return ""


def test_deadlines_are_the_ones_the_readme_names():
    assert main_module.RESULT_DEADLINE_S == 10.0
    assert main_module.FIRST_RESULT_DEADLINE_S == 30.0


def test_late_result_names_the_stream_and_model():
    pending = PendingResults(2, "models/m.tar.gz", 10.0, 30.0)
    pending.observe(7, 7, 100.0)
    assert error_of(pending, 8, None, 101.0) == ""
    assert error_of(pending, None, None, 111.0) == ""
    assert error_of(pending, None, None, 111.5) == (
        "stream 2: model models/m.tar.gz returned no result for 10 s while its decoder sent frames"
    )


def test_later_frames_keep_the_first_pending_time():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(None, 7, 100.0)
    pending.observe(8, None, 101.0)
    pending.observe(9, None, 106.0)
    assert error_of(pending, 10, None, 111.5) != ""


def test_result_clears_the_pending_frame():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(None, 7, 100.0)
    pending.observe(8, None, 101.0)
    pending.observe(None, 8, 105.0)
    assert error_of(pending, None, None, 200.0) == ""
    pending.observe(9, None, 205.0)
    assert error_of(pending, None, None, 215.0) == ""
    assert error_of(pending, None, None, 215.5) != ""


def test_frame_with_its_result_in_one_round_is_not_pending():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(7, 7, 100.0)
    assert error_of(pending, None, None, 1_000_000.0) == ""


def test_newer_frame_with_a_result_in_one_round_starts_the_deadline():
    # The model holds frame 8 and returns nothing more, so no later frame restarts the deadline.
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(8, 7, 100.0)
    assert error_of(pending, None, None, 110.0) == ""
    assert error_of(pending, None, None, 110.5) == (
        "stream 0: model m.tar.gz returned no result for 10 s while its decoder sent frames"
    )


def test_newer_frame_before_an_older_result_keeps_the_deadline():
    # Frame 9 arrives while frame 8 is pending, then the result for 8 comes in the next round.
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(None, 7, 90.0)
    pending.observe(8, None, 95.0)
    pending.observe(9, None, 99.0)
    pending.observe(None, 8, 100.0)
    assert error_of(pending, None, None, 110.0) == ""
    assert error_of(pending, None, None, 110.5) == (
        "stream 0: model m.tar.gz returned no result for 10 s while its decoder sent frames"
    )


def test_frame_no_newer_than_the_last_result_is_not_pending():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(None, 8, 100.0)
    pending.observe(7, None, 101.0)
    pending.observe(8, None, 102.0)
    assert error_of(pending, None, None, 1_000_000.0) == ""


def test_frame_with_an_unknown_id_is_pending_only_without_a_result():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    pending.observe(-1, -1, 100.0)
    assert error_of(pending, None, None, 1_000_000.0) == ""
    pending.observe(-1, None, 1_000_001.0)
    assert error_of(pending, None, None, 1_000_011.5) != ""


def test_first_result_has_its_own_deadline():
    pending = PendingResults(1, "m.tar.gz", 10.0, 30.0)
    pending.observe(0, None, 100.0)
    assert error_of(pending, None, None, 125.0) == ""
    assert error_of(pending, None, None, 130.5) == (
        "stream 1: model m.tar.gz returned no result for 30 s while its decoder sent frames"
    )


def test_paused_camera_waits_without_a_limit():
    pending = PendingResults(0, "m.tar.gz", 10.0, 30.0)
    assert error_of(pending, None, None, 1_000_000.0) == ""
    pending.observe(None, 7, 1_000_001.0)
    assert error_of(pending, None, None, 2_000_000.0) == ""


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


class FakeSender:
    def __init__(self) -> None:
        self.sent = []

    def send_metadata(self, metadata_type, data, timestamp_ms, frame_id) -> bool:
        self.sent.append((metadata_type, data, timestamp_ms, frame_id))
        return True


RESULT = SimpleNamespace(pts_ns=2_000_000, frame_id=7)
RESULT_FRAME = SimpleNamespace(frame_id=7)
NEXT_FRAME = SimpleNamespace(frame_id=8)


def fake_stream(deadline_s: float, first_result_s: float):
    """A detection stream that has not sent a result yet."""
    return main_module.Stream(
        info=main_module.RtspStreamInfo(640, 480, 30),
        metadata_type=main_module.DETECTION,
        labels=LABELS,
        metadata_sender=FakeSender(),
        pending=PendingResults(0, "models/m.tar.gz", deadline_s, first_result_s),
    )


@pytest.fixture
def no_decode(monkeypatch):
    """Results that hold nothing, so the loop runs without a real model output."""
    monkeypatch.setattr(main_module, "metadata_entries", lambda *args: [])


def test_send_results_stops_a_model_that_stops_returning_results(no_decode):
    # Round 1: a result and its frame. Round 2: a frame. Round 3: nothing, past the deadline.
    stop = threading.Event()
    run = StoppingRun(
        stop, ("sample", RESULT), ("sample", RESULT_FRAME), "timeout", ("sample", NEXT_FRAME),
        "timeout", "timeout",
    )
    stream = fake_stream(0.0, 60.0)
    with pytest.raises(RuntimeError) as error:
        main_module.send_results(run, [stream], 50, 0, stop)
    assert str(error.value) == (
        "stream 0: model models/m.tar.gz returned no result for 0 s while its decoder sent frames"
    )
    assert stream.metadata_sender.sent == [("object-detection", '{"objects":[]}', 2, "7")]
    assert run.pulls == [("results_0", 0), ("frames_0", 0)] * 3


def test_send_results_stops_a_model_that_never_returns_a_result(no_decode):
    stop = threading.Event()
    run = StoppingRun(stop, "timeout", ("sample", NEXT_FRAME), "timeout", "timeout")
    with pytest.raises(RuntimeError) as error:
        main_module.send_results(run, [fake_stream(60.0, 0.0)], 50, 0, stop)
    assert str(error.value) == (
        "stream 0: model models/m.tar.gz returned no result for 0 s while its decoder sent frames"
    )


def test_send_results_stops_a_model_that_holds_a_frame_pulled_with_a_result(no_decode):
    # Round 1: a result and a newer frame, which the model never returns. Then nothing.
    stop = threading.Event()
    run = StoppingRun(
        stop, ("sample", RESULT), ("sample", NEXT_FRAME), *["timeout"] * 4
    )
    with pytest.raises(RuntimeError) as error:
        main_module.send_results(run, [fake_stream(0.0, 60.0)], 50, 0, stop)
    assert str(error.value) == (
        "stream 0: model models/m.tar.gz returned no result for 0 s while its decoder sent frames"
    )


def test_send_results_lets_a_paused_camera_wait(no_decode):
    # No frames and no results: the loop keeps waiting until the stop signal.
    stop = threading.Event()
    run = StoppingRun(stop, *["timeout"] * 20)
    main_module.send_results(run, [fake_stream(0.0, 0.0)], 50, 0, stop)
    assert len(run.pulls) == 20


def test_send_results_below_zero_frames_runs_until_stopped(no_decode):
    stop = threading.Event()
    run = StoppingRun(stop, *[("sample", RESULT), "timeout"] * 3)
    stream = fake_stream(60.0, 60.0)
    main_module.send_results(run, [stream], 50, -1, stop)
    assert len(stream.metadata_sender.sent) == 3


def test_send_results_without_a_timestamp_or_frame_id_sends_none(no_decode):
    stop = threading.Event()
    run = StoppingRun(stop, ("sample", SimpleNamespace(pts_ns=-1, frame_id=-1)), "timeout")
    stream = fake_stream(60.0, 60.0)
    main_module.send_results(run, [stream], 50, 1, stop)
    assert stream.metadata_sender.sent == [("object-detection", '{"objects":[]}', -1, "")]


def test_send_results_labels_each_stream_with_its_own_labels(monkeypatch):
    box = SimpleNamespace(x1=0.0, y1=0.0, x2=10.0, y2=10.0, score=0.9, class_id=1)
    monkeypatch.setattr(main_module, "sample_tensors", lambda sample: [object()])
    monkeypatch.setattr(
        main_module.pyneat.detections,
        "decode_bbox_tensor",
        lambda *args, **kwargs: SimpleNamespace(boxes=[box]),
    )
    stop = threading.Event()
    run = StoppingRun(stop, ("sample", RESULT), "timeout", ("sample", RESULT), "timeout")
    first, second = fake_stream(60.0, 60.0), fake_stream(60.0, 60.0)
    second.labels = ["cat", "dog"]
    main_module.send_results(run, [first, second], 50, 1, stop)
    sent = [json.loads(stream.metadata_sender.sent[0][1]) for stream in (first, second)]
    assert [data["objects"][0]["label"] for data in sent] == ["bicycle", "dog"]


def run_main(tmp_path: Path, change, last_line: bool = True) -> str:
    """The last stderr line, or all of it, for the shipped config with `change` applied. The set
    frame size and rate skip the probe, so no source is needed."""
    config = yaml.safe_load((COMMON_DIR / "config.yaml").read_text())
    config["input"].update(width=1280, height=720, fps=30)
    change(config)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_DIR / "src" / "python" / "main.py"), "--config", str(path)],
        capture_output=True, text=True, cwd=APPS_ROOT, timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    return result.stderr.strip().splitlines()[-1] if last_line else result.stderr.strip()


def test_model_needs_one_entry_per_stream(tmp_path):
    def drop_one(config):
        config["model"] = config["model"][:3]

    assert run_main(tmp_path, drop_one) == (
        "[ERR] model has 3 entries and input.source has 4: give one per stream"
    )


def test_unknown_model_id_names_the_entry_and_the_choices(tmp_path):
    def unknown(config):
        config["model"][1]["id"] = "yolo26-obb"

    assert run_main(tmp_path, unknown) == (
        "[ERR] model[1].id: unknown value 'yolo26-obb' (expected yolo26, yolo26-pose, "
        "yolo26-seg, yolov8, yolov8-pose, yolov8-seg)"
    )


def test_detection_model_without_labels_names_the_entry(tmp_path):
    def no_labels(config):
        del config["model"][0]["labels"]

    assert run_main(tmp_path, no_labels) == (
        f"[ERR] {tmp_path / 'config.yaml'} is missing model[0].labels"
    )


def test_each_model_reads_its_own_labels(tmp_path):
    def missing_labels(config):
        config["model"][1]["labels"] = "missing_labels.txt"

    assert run_main(tmp_path, missing_labels) == (
        "[ERR] labels file does not exist: missing_labels.txt"
    )


def test_pose_model_needs_no_labels(tmp_path):
    # The shipped pose entry has no labels. Every entry is read before the first model loads, so
    # a run whose first model is missing gets past the config and fails on that model.
    assert "labels" not in yaml.safe_load((COMMON_DIR / "config.yaml").read_text())["model"][2]

    def missing_model(config):
        config["model"][0]["path"] = "missing_mpk.tar.gz"

    # Core's error for a missing package: a line that names model[0].path, then a hint.
    first, *_, last = run_main(tmp_path, missing_model, last_line=False).splitlines()
    assert first.startswith("[ERR] ") and first.endswith(
        "invalid_archive: archive path does not exist or is not a regular file: "
        f"{tmp_path / 'missing_mpk.tar.gz'}"
    ), first
    assert last == "Hint: Re-export the model archive and verify it contains a valid MPK contract."


def test_model_ids_select_the_decoder_and_metadata_type():
    decode = main_module.pyneat.BoxDecodeType
    assert main_module.MODEL_IDS == {
        "yolov8": (decode.YoloV8, "object-detection"),
        "yolov8-seg": (decode.YoloV8Seg, "segmentation"),
        "yolov8-pose": (decode.YoloV8Pose, "pose-estimation"),
        "yolo26": (decode.YoloV26, "object-detection"),
        "yolo26-seg": (decode.YoloV26Seg, "segmentation"),
        "yolo26-pose": (decode.YoloV26Pose, "pose-estimation"),
    }
