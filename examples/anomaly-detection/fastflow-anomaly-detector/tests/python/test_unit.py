"""Unit tests for the FastFlow anomaly detector: the checks the application leaves to its
config, and the defect regions it sends. tests/cpp/test_unit.cpp runs the same checks in C++.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "fastflow_anomaly_detector_main")
anomaly_regions = main_module.anomaly_regions

pytestmark = pytest.mark.unit


def test_shipped_config_names_a_supported_model_and_input():
    """The checks the application leaves to its config: the values it reads are usable."""
    config = main_module.Config.load(COMMON_DIR / "config.yaml")
    assert config.string("model.id") in main_module.NORMALIZE_MEANS
    assert config.string("input.codec") in main_module.RTSP_CODECS
    assert config.string("input.codec") in main_module.DECODE_TYPES
    model_path = COMMON_DIR / config.string("model.path")
    assert model_path.name == "fastflow_demo_mpk.tar.gz"
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")
    for key in ("input.width", "input.height", "input.fps", "input.rtsp.latency_ms",
                "inference.frames"):
        assert config.integer(key) >= 0, key
    assert config.integer("inference.max_detections") > 0
    assert config.integer("app.min_region_px") > 0
    assert 0 < config.number("app.anomaly_threshold") < 1
    for key in ("output.insight.video_port_base", "output.insight.metadata_port_base"):
        assert 1 <= config.integer(key) <= 65535, key
    # Core sends to 127.0.0.1 when the host is empty.
    assert config.string("output.insight.host")
    assert config.boolean("input.rtsp.tcp")


def _three_region_map() -> np.ndarray:
    """A 16x16 map with a 16-pixel region scoring 0.9, a 9-pixel one peaking at 0.7, and a
    4-pixel one at 0.8."""
    anomaly_map = np.zeros((16, 16), np.float32)
    anomaly_map[2:6, 2:6] = 0.9
    anomaly_map[1:4, 10:13] = 0.6
    anomaly_map[2, 11] = 0.7
    anomaly_map[12:14, 12:14] = 0.8
    return anomaly_map


def _options(max_regions: int = 10) -> anomaly_regions.RegionOptions:
    # A 32x48 frame scales map x by 2 and map y by 3.
    return anomaly_regions.RegionOptions(
        width=32, height=48, threshold=0.5, min_region_px=5, max_regions=max_regions
    )


def test_regions_below_min_region_px_are_speckle():
    """The 4-pixel region is dropped; the others come best first, scaled to the frame."""
    regions = anomaly_regions.defect_regions(_three_region_map(), _options())
    assert [region.bbox for region in regions] == [(4, 6, 8, 12), (20, 3, 6, 9)]
    assert [round(region.score, 6) for region in regions] == [0.9, 0.7]
    assert regions[0].polygon == [[4, 6], [4, 15], [10, 15], [10, 6]]


def test_max_regions_keeps_the_highest_scores():
    regions = anomaly_regions.defect_regions(_three_region_map(), _options(max_regions=1))
    assert [region.bbox for region in regions] == [(4, 6, 8, 12)]


def test_a_map_below_the_threshold_has_no_region():
    good_part = np.full((16, 16), 0.49, np.float32)
    assert anomaly_regions.defect_regions(good_part, _options()) == []
    assert json.loads(anomaly_regions.segments_json([])) == {"segments": []}


def test_a_dot_or_a_line_is_outlined_by_its_box():
    """Insight draws a polygon only from three points, so a region approxPolyDP reduces to
    fewer gets its box."""
    anomaly_map = np.zeros((16, 16), np.float32)
    anomaly_map[9, 7] = 0.9
    anomaly_map[14, 2:7] = 0.8
    options = anomaly_regions.RegionOptions(
        width=32, height=48, threshold=0.5, min_region_px=1, max_regions=10
    )
    regions = anomaly_regions.defect_regions(anomaly_map, options)
    assert [region.bbox for region in regions] == [(14, 27, 2, 3), (4, 42, 10, 3)]
    assert regions[0].polygon == [[14, 27], [14, 30], [16, 30], [16, 27]]
    assert regions[1].polygon == [[4, 42], [4, 45], [14, 45], [14, 42]]


def test_segments_carry_the_box_outline_and_score():
    regions = anomaly_regions.defect_regions(_three_region_map(), _options(max_regions=1))
    (segment,) = json.loads(anomaly_regions.segments_json(regions))["segments"]
    assert segment["label"] == "anomaly"
    assert segment["bbox"] == [4, 6, 8, 12]
    assert segment["mask_format"] == "polygon"
    assert segment["mask"] == [[4, 6], [4, 15], [10, 15], [10, 6]]
    assert segment["confidence"] == pytest.approx(0.9)
