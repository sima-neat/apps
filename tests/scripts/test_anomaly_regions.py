"""Contract tests for support/python/sima_examples/anomaly_regions.py: the defect regions of an
anomaly map and the segments metadata. The FastFlow example's tests/cpp/test_unit.cpp checks the
C++ module the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples import anomaly_regions


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


def test_a_box_on_the_map_edge_stays_inside_the_frame():
    """A dot in the map's bottom-right pixel scales its box to the frame size; every point
    stays inside the frame, because Insight drops a point outside it."""
    anomaly_map = np.zeros((16, 16), np.float32)
    anomaly_map[15, 15] = 0.9
    options = anomaly_regions.RegionOptions(
        width=32, height=48, threshold=0.5, min_region_px=1, max_regions=10
    )
    (region,) = anomaly_regions.defect_regions(anomaly_map, options)
    assert region.polygon == [[30, 45], [30, 47], [31, 47], [31, 45]]
    # At 1:1 scaling the box still keeps an area.
    same_size = anomaly_regions.RegionOptions(
        width=16, height=16, threshold=0.5, min_region_px=1, max_regions=10
    )
    (region,) = anomaly_regions.defect_regions(anomaly_map, same_size)
    assert region.polygon == [[14, 14], [14, 15], [15, 15], [15, 14]]


def test_segments_carry_the_box_outline_and_score():
    regions = anomaly_regions.defect_regions(_three_region_map(), _options(max_regions=1))
    (segment,) = json.loads(anomaly_regions.segments_json(regions))["segments"]
    assert segment["label"] == "anomaly"
    assert segment["bbox"] == [4, 6, 8, 12]
    assert segment["mask_format"] == "polygon"
    assert segment["mask"] == [[4, 6], [4, 15], [10, 15], [10, 6]]
    assert segment["confidence"] == pytest.approx(0.9)
