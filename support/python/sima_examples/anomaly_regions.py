"""The defect regions of an anomaly map, as the segments Insight draws.

Mirrors support/cpp/sima_examples/anomaly_regions.h, so both languages send the same metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

LABEL = "anomaly"


@dataclass(frozen=True)
class RegionOptions:
    width: int  # Frame size the regions are scaled to, in pixels.
    height: int
    threshold: float  # Map value at which a pixel counts as anomalous.
    min_region_px: int  # Smallest region, in map pixels, that counts as a defect.
    max_regions: int  # Regions kept, highest score first.


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]  # x, y, width, height, in frame pixels.
    score: float  # Highest map value inside the region.
    polygon: list[list[int]]  # Outline as [x, y] points, in frame pixels.


def defect_regions(anomaly_map: np.ndarray, options: RegionOptions) -> list[Region]:
    """The connected regions at or above the threshold that cover at least min_region_px map
    pixels, highest score first, scaled from the map to the frame."""
    map_height, map_width = anomaly_map.shape
    above = (anomaly_map >= options.threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    regions = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < options.min_region_px:
            continue
        member = labels == index
        contours, _ = cv2.findContours(
            member.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        x, y, width, height = (int(value) for value in stats[index, :4])
        # Integer scaling, the same in both languages: map pixel m covers frame pixels from
        # m * width // map_width.
        x0, y0 = x * options.width // map_width, y * options.height // map_height
        x1 = (x + width) * options.width // map_width
        y1 = (y + height) * options.height // map_height
        outline = cv2.approxPolyDP(contours[0], 1.0, True)
        if len(outline) >= 3:
            # Contour points are map pixels, so they scale to points inside the frame.
            polygon = [
                [
                    int(point[0][0]) * options.width // map_width,
                    int(point[0][1]) * options.height // map_height,
                ]
                for point in outline
            ]
        else:
            # A polygon needs three points and an area; a dot or a line is sent as its box,
            # kept at least one pixel wide and high and inside the frame.
            left, top = min(x0, options.width - 2), min(y0, options.height - 2)
            right = min(max(x1, left + 1), options.width - 1)
            bottom = min(max(y1, top + 1), options.height - 1)
            polygon = [[left, top], [left, bottom], [right, bottom], [right, top]]
        regions.append(
            Region(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                score=float(anomaly_map[member].max()),
                polygon=polygon,
            )
        )
    regions.sort(key=lambda region: region.score, reverse=True)
    return regions[: options.max_regions]


def segments_json(regions: list[Region]) -> str:
    """The "segmentation" metadata: each region's box and outline, labelled as an anomaly."""
    segments = [
        {
            "id": f"seg_{index}",
            "label": LABEL,
            "confidence": region.score,
            "bbox": list(region.bbox),
            "mask_format": "polygon",
            "mask": region.polygon,
        }
        for index, region in enumerate(regions, start=1)
    ]
    return json.dumps({"segments": segments}, separators=(",", ":"))
