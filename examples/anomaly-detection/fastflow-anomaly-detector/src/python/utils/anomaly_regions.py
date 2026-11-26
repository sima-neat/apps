"""FastFlow postprocessing: the defect regions of an anomaly map, as the segments Insight draws.

Mirrors utils/anomaly_regions.h in the C++ implementation, so both send the same metadata.
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
    threshold: float  # Map probability at which a pixel counts as anomalous.
    min_region_px: int  # Smallest region, in map pixels, that counts as a defect.
    max_regions: int  # Regions kept, highest score first.


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]  # x, y, width, height, in frame pixels.
    score: float  # Highest probability inside the region.
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
        outline = [
            (int(point[0][0]), int(point[0][1]))
            for point in cv2.approxPolyDP(contours[0], 1.0, True)
        ]
        # A polygon needs three points; a dot or a line is sent as its box instead.
        if len(outline) < 3:
            outline = [(x, y), (x, y + height), (x + width, y + height), (x + width, y)]
        # Integer scaling, the same in both languages: map pixel m covers frame pixels from
        # m * width // map_width.
        x0, y0 = x * options.width // map_width, y * options.height // map_height
        x1 = (x + width) * options.width // map_width
        y1 = (y + height) * options.height // map_height
        regions.append(
            Region(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                score=float(anomaly_map[member].max()),
                polygon=[
                    [px * options.width // map_width, py * options.height // map_height]
                    for px, py in outline
                ],
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
