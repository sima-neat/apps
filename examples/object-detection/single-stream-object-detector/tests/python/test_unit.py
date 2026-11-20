"""Unit tests for the single-stream object detector: the checks the application leaves to its
config. tests/cpp/test_unit.cpp runs the same checks in C++.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "single_stream_object_detector_main")

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_input_model_and_labels():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("input.type") in main_module.INPUT_CODECS
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "yolo26m_w16a16_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    # YOLO26's class IDs index the 80 COCO labels.
    labels = (EXAMPLE_DIR.parents[2] / config.string("model.labels")).read_text().splitlines()
    assert len(labels) == 80
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert 0 <= config.number("inference.min_score") <= 1
    assert 0 <= config.number("inference.nms_iou") <= 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")
