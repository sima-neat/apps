"""SuperPoint feature points: from the decoder's model-input pixels to the metadata Insight draws.

Mirrors utils/feature_points.h in the C++ implementation, so both send the same metadata.
"""

from __future__ import annotations

import json

import numpy as np


def source_points(
    keypoints: np.ndarray, model_size: tuple[int, int], source_size: tuple[int, int]
) -> np.ndarray:
    """The keypoints `[N, 2]` on the source frame, in whole pixels; both sizes are
    `(width, height)`.

    The decoder reports model-input pixels (sima-neat/core#1328). After a Stretch resize
    (`resize.mode = Stretch` with `resize.enable = On`, as main sets it), each axis maps back by
    its own ratio. A letterbox would need its padding removed first. Each point rounds half up,
    like C++ std::lround on a point that is not negative. A point that lands on or beyond a
    frame edge is dropped: the edges are exclusive coordinates.
    """
    scale = np.asarray(
        [source_size[0] / model_size[0], source_size[1] / model_size[1]], dtype=np.float32
    )
    # float32 like the C++ scale, then float64 so adding 0.5 rounds exactly.
    points = (np.asarray(keypoints, dtype=np.float32).reshape(-1, 2) * scale).astype(np.float64)
    points = points[np.all(np.isfinite(points) & (points >= 0), axis=1)]
    pixels = np.floor(points + 0.5).astype(np.int64)
    inside = (pixels[:, 0] < source_size[0]) & (pixels[:, 1] < source_size[1])
    return pixels[inside]


def points_json(points: np.ndarray) -> str:
    """The `data` of Insight's pose-estimation overlay: one pose whose keypoints are the points.

    Insight draws each keypoint as a dot. It draws a skeleton only between COCO joint names and
    a box only around keypoints with a confidence above 0.3, so unnamed points without a
    confidence get neither. Whole-pixel `{"x", "y"}` keypoints keep 600 points near 10 KB, which
    sends in about 3 ms on a DevKit; one named, scored pose per point took 37 KB and 9 ms.
    """
    keypoints = [{"x": x, "y": y} for x, y in np.asarray(points).tolist()]
    return json.dumps({"poses": [{"keypoints": keypoints}]}, separators=(",", ":"))
