"""SuperPoint feature points, from the decoder's output to Insight's metadata. Mirrors
utils/feature_points.h."""

from __future__ import annotations

import json

import numpy as np


def source_points(
    keypoints: np.ndarray, model_size: tuple[int, int], source_size: tuple[int, int]
) -> np.ndarray:
    """The keypoints `[N, 2]`, scaled from model-input pixels (sima-neat/core#1328) to whole
    pixels on the source frame; sizes are `(width, height)`. Points on or past a frame edge are
    dropped."""
    scale = np.asarray(
        [source_size[0] / model_size[0], source_size[1] / model_size[1]], dtype=np.float32
    )
    # float32 like the C++ scale, then float64 so adding 0.5 rounds half up, like std::lround.
    points = (np.asarray(keypoints, dtype=np.float32).reshape(-1, 2) * scale).astype(np.float64)
    points = points[np.all(np.isfinite(points) & (points >= 0), axis=1)]
    pixels = np.floor(points + 0.5).astype(np.int64)
    inside = (pixels[:, 0] < source_size[0]) & (pixels[:, 1] < source_size[1])
    return pixels[inside]


def points_json(points: np.ndarray) -> str:
    """Insight's pose-estimation data: one pose whose unnamed keypoints Insight draws as dots."""
    keypoints = [{"x": x, "y": y} for x, y in np.asarray(points).tolist()]
    return json.dumps({"poses": [{"keypoints": keypoints}]}, separators=(",", ":"))
