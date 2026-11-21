"""The Python metadata message matches the C++ one in support/cpp/sima_examples/detection_egress.h.

The inputs and expected values are the ones the high-density example's C++ unit test uses.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
                    "bbox": [10.0, 0.0, 25.0, 22.0],
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
                    "bbox": [105.0, 10.0, 0.0, 8.0],
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
