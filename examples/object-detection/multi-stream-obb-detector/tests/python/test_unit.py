"""Unit tests for the multi-stream OBB detector. tests/cpp/test_unit.cpp runs the same checks in
C++.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pyneat
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
# The C++ unit test checks the same fixture, so both implementations send the same objects.
OBJECTS_FIXTURE = EXAMPLE_DIR / "tests" / "fixtures" / "insight_objects.json"

main_module = load_example_main(EXAMPLE_DIR, "multi_stream_obb_detector_main")
oriented_boxes = main_module.oriented_boxes

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_model_input_and_labels():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("model.id") == "yolo26"
    assert config.string("model.id") in main_module.BOX_DECODE_TYPES
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26n_obb_w8a8_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    # The model's class IDs index the 15 DOTA labels.
    labels = (EXAMPLE_DIR.parents[2] / config.string("model.labels")).read_text().splitlines()
    assert len(labels) == 15
    assert len(config.string_list("input.source")) == 2
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    assert (config.integer("input.width"), config.integer("input.height")) == (0, 0), (
        "RTSP streams report their own frame size"
    )
    for key in ("input.fps", "input.rtsp.latency_ms", "inference.frames"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def _decoded_boxes(boxes: list[dict]):
    """The boxes as decode_obb_tensor returns them from a BBOX_OBB_V1 tensor, the model's output:
    a 32-byte header, then one 32-byte record per box. Core needs a capacity of at least 1."""
    capacity = max(len(boxes), 1)
    raw = bytearray(
        struct.pack(
            "<IHH6I", 0x3156424F, 1, 32, 32 + 32 * capacity, len(boxes), capacity, 32, 32, 0
        )
    )
    for b in boxes:
        raw += struct.pack(
            "<6fiI",
            b["cx"],
            b["cy"],
            b["width"],
            b["height"],
            b["angle_radians"],
            b["score"],
            b["class_id"],
            0,
        )
    raw += bytes(32 * (capacity - len(boxes)))
    tensor = pyneat.Tensor.from_numpy(
        np.frombuffer(raw, dtype=np.uint8).copy(), copy=True, memory=pyneat.TensorMemory.CPU
    )
    detection = pyneat.DetectionSpec()
    detection.format = "BBOX_OBB_V1"
    semantic = tensor.semantic
    semantic.detection = detection
    tensor.semantic = semantic
    return pyneat.decode_obb_tensor(tensor)


def test_objects_match_the_shared_fixture():
    fixture = json.loads(OBJECTS_FIXTURE.read_text(encoding="utf-8"))

    objects = json.loads(
        oriented_boxes.detections_json(_decoded_boxes(fixture["boxes"]), fixture["labels"])
    )["objects"]

    assert objects == fixture["objects"]


def test_a_frame_without_detections_sends_an_empty_list():
    """Insight clears the previous frame's boxes only when it receives the empty list."""
    data = oriented_boxes.detections_json(_decoded_boxes([]), ["plane"])

    assert json.loads(data) == {"objects": []}
