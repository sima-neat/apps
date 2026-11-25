"""The Python metadata message matches the C++ one in support/cpp/sima_examples/detection_egress.h.

The inputs and expected values are the ones the high-density example's C++ unit test uses.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples import detection_egress


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


BOXES = [
    Box(10.9, -2.0, 35.9, 20.8, 0.75, 0),
    Box(90.0, 80.0, 120.0, 130.0, 0.5, 1),
    Box(105.0, 10.0, 115.0, 18.0, 0.25, 99),
]
LABELS = ["person", 'bicycle\n"quoted"']
FRAME = detection_egress.FrameMetadata(
    stream_index=7,
    stream_id="stream7",
    frame_id=42,
    pts_ns=1_234_567_890,
    dts_ns=1_200_000_000,
    duration_ns=50_000_000,
    input_seq=88,
    orig_input_seq=77,
    rtp_timestamp=111_111,
)


def test_message_matches_the_cpp_envelope():
    message = json.loads(detection_egress.serialize(BOXES, LABELS, 100, 100, FRAME))

    assert message == {
        "type": "object-detection",
        "timestamp": 1234,
        "frame_id": "42",
        "stream_id": "stream7",
        "stream_index": 7,
        "pts_ns": 1_234_567_890,
        "dts_ns": 1_200_000_000,
        "duration_ns": 50_000_000,
        "input_seq": 88,
        "orig_input_seq": 77,
        "rtp_timestamp": 111_111,
        "data": {
            "objects": [
                {
                    "id": "obj_1",
                    "label": "person",
                    "confidence": 0.75,
                    "bbox": [10.0, 0.0, 25.0, 20.0],
                },
                {
                    "id": "obj_2",
                    "label": 'bicycle\n"quoted"',
                    "confidence": 0.5,
                    "bbox": [90.0, 80.0, 10.0, 20.0],
                },
                {
                    "id": "obj_3",
                    "label": "unknown",
                    "confidence": 0.25,
                    "bbox": [100.0, 10.0, 0.0, 8.0],
                },
            ]
        },
    }


def test_message_without_pts_has_no_rtp_timestamp():
    frame = detection_egress.FrameMetadata(stream_index=0, stream_id="stream0")
    message = json.loads(detection_egress.serialize([], LABELS, 100, 100, frame))

    assert message["timestamp"] == -1
    assert message["frame_id"] == ""
    assert "rtp_timestamp" not in message


@dataclass
class Sample:
    """The fields frame_metadata reads from a pulled pyneat.Sample."""

    stream_id: str = "decoder"
    frame_id: int = 42
    pts_ns: int = 1_000_000_000
    dts_ns: int = 900_000_000
    duration_ns: int = 33_333_333
    input_seq: int = 7
    orig_input_seq: int = 6


def test_frame_metadata_copies_the_sample_identity():
    frame = detection_egress.frame_metadata(Sample(), 3)
    assert frame == detection_egress.FrameMetadata(
        stream_index=3,
        stream_id="decoder",
        frame_id=42,
        pts_ns=1_000_000_000,
        dts_ns=900_000_000,
        duration_ns=33_333_333,
        input_seq=7,
        orig_input_seq=6,
    )


@dataclass
class Track(Box):
    track_id: int


def test_tracking_message_matches_the_cpp_envelope():
    """The inputs and expected values are the ones yolo26-tiny-drone-tracker's C++ unit test
    uses. decode_bbox_tensor already clamps every corner to the frame. The second box ends past
    the right edge, as a box from another source may, and is cut at the frame."""
    tracks = [
        Track(10.9, 2.0, 35.9, 20.8, 0.75, 0, track_id=7),
        Track(90.0, 80.0, 120.0, 95.0, 0.5, 3, track_id=12),
    ]
    frame = detection_egress.FrameMetadata(
        stream_index=0, stream_id="decoder", frame_id=42, pts_ns=1_234_567_890
    )
    message = json.loads(detection_egress.serialize_tracks(tracks, ["drone"], 100, 100, frame))

    assert message["type"] == "tracking"
    assert message["timestamp"] == 1234
    assert (message["frame_id"], message["stream_id"]) == ("42", "decoder")
    assert message["data"] == {
        "tracks": [
            {"id": "7", "label": "drone", "confidence": 0.75, "bbox": [10.0, 2.0, 25.0, 18.0]},
            {"id": "12", "label": "unknown", "confidence": 0.5, "bbox": [90.0, 80.0, 10.0, 15.0]},
        ]
    }


# Boxes on a 640x480 frame and the [x, y, width, height] Insight gets: both corners are clipped
# to the frame first, then the size is measured. multi-stream-tracker's tests use the first three.
# high-density's C++ unit test checks the same boxes.
CLIPPED = [
    pytest.param(Box(-44.59, 10.0, 136.98, 90.0, 0.8, 0), [0.0, 10.0, 136.0, 80.0], id="left-edge"),
    pytest.param(
        Box(600.0, 400.0, 700.0, 500.0, 0.8, 0), [600.0, 400.0, 40.0, 80.0], id="right-bottom-edge"
    ),
    pytest.param(
        Box(-200.0, -200.0, -10.0, -10.0, 0.8, 0), [0.0, 0.0, 0.0, 0.0], id="outside-top-left"
    ),
    pytest.param(
        Box(700.0, 10.0, 760.0, 50.0, 0.8, 0), [640.0, 10.0, 0.0, 40.0], id="outside-right"
    ),
    # int(x2) - int(x1), not int(x2 - x1), which would give 29 x 39.
    pytest.param(Box(10.6, 20.6, 40.4, 60.4, 0.8, 0), [10.0, 20.0, 30.0, 40.0], id="rounding"),
]


@pytest.mark.parametrize(("box", "bbox"), CLIPPED)
def test_box_is_clipped_to_the_frame_before_its_size_is_measured(box, bbox):
    message = json.loads(detection_egress.serialize([box], LABELS, 640, 480, FRAME))
    assert message["data"]["objects"][0]["bbox"] == bbox


@pytest.mark.parametrize(("box", "bbox"), CLIPPED)
def test_track_entries_clip_as_boxes_do(box, bbox):
    track = Track(box.x1, box.y1, box.x2, box.y2, 0.8, 1, track_id=5)
    assert detection_egress.track_entries([track], LABELS, 640, 480) == [
        {"id": "5", "label": LABELS[1], "confidence": 0.8, "bbox": bbox}
    ]


def test_tracking_message_holds_the_track_entries():
    tracks = [Track(10.0, 20.0, 40.0, 60.0, 0.75, 0, track_id=7)]
    message = json.loads(detection_egress.serialize_tracks(tracks, LABELS, 640, 480, FRAME))
    assert message["data"] == {"tracks": detection_egress.track_entries(tracks, LABELS, 640, 480)}
