"""Insight metadata must preserve detection values and the source video clock."""
from copy import deepcopy
from pathlib import Path

import pytest

from tests.ros2.local_modules import load_node

ROOT = Path(__file__).resolve().parents[2]
detection_metadata = load_node(
    ROOT / "examples/ros2/object-detection/high-density-multi-stream-object-detector/src/python/insight_sink.py",
    "tested_insight_sink", resolve_share=False).detection_metadata


def result():
    return {"schema_version": 1, "sequence": 7, "stream_id": "stream2", "source_frame_id": 97,
            "header": {"frame_id": "camera_2_optical_frame", "stamp": {"sec": 100000, "nanosec": 345678901}},
            "data": {"objects": [{"label": "person", "confidence": 0.75, "bbox": [10, 20, 30, 40]}]}}


def test_preserves_boxes_identity_and_wrapped_rtp_timestamp():
    value = result()
    original = deepcopy(value)
    index, metadata = detection_metadata(value)
    assert index == 2
    assert metadata == {"type": "object-detection", "data": original["data"],
                        "timestamp": 100000345, "pts_ns": 100000345678901,
                        "rtp_timestamp": 410096519, "frame_id": "97",
                        "stream_id": "stream2", "stream_index": 2}
    assert value == original


def test_empty_detections_are_forwarded_to_clear_previous_boxes():
    value = result()
    value["data"]["objects"] = []
    assert detection_metadata(value)[1]["data"] == {"objects": []}


@pytest.mark.parametrize("field,bad", [("sec", -1), ("sec", 1.5), ("nanosec", -1),
                                       ("nanosec", 1_000_000_000)])
def test_rejects_invalid_source_timestamps(field, bad):
    value = result()
    value["header"]["stamp"][field] = bad
    with pytest.raises(ValueError, match="presentation timestamp"):
        detection_metadata(value)


@pytest.mark.parametrize("stream", ["camera2", "stream-1", "stream", 2])
def test_rejects_unmapped_stream_ids(stream):
    value = result()
    value["stream_id"] = stream
    with pytest.raises(ValueError, match="native stream ID"):
        detection_metadata(value)


def test_does_not_substitute_ros_sequence_for_missing_source_frame():
    value = result()
    del value["source_frame_id"]
    with pytest.raises(KeyError, match="source_frame_id"):
        detection_metadata(value)
