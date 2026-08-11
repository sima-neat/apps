#!/usr/bin/env python3
"""Replay frame-level detector JSONL through the reference object tracker."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from utils.tracker import CameraMotionEstimate, ObjectTracker, TrackerConfig


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _detection(document: Any, location: str) -> dict[str, float | int]:
    if not isinstance(document, dict):
        raise ValueError(f"{location} must be an object")
    if "bbox" in document:
        bbox = document["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{location}.bbox must contain [x, y, width, height]")
        x1, y1, width, height = (
            _finite_number(value, f"{location}.bbox[{index}]")
            for index, value in enumerate(bbox)
        )
        x2, y2 = x1 + width, y1 + height
    else:
        x1 = _finite_number(document.get("x1"), f"{location}.x1")
        y1 = _finite_number(document.get("y1"), f"{location}.y1")
        x2 = _finite_number(document.get("x2"), f"{location}.x2")
        y2 = _finite_number(document.get("y2"), f"{location}.y2")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{location} must have positive width and height")
    score = _finite_number(document.get("score"), f"{location}.score")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{location}.score must be in [0, 1]")
    class_id = document.get("class_id", 0)
    if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0:
        raise ValueError(f"{location}.class_id must be a non-negative integer")
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": score,
        "class_id": class_id,
    }


def read_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    previous_frame = -1
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            location = f"{path}:{line_number}"
            if not isinstance(frame, dict):
                raise ValueError(f"{location}: frame must be an object")
            frame_index = frame.get("frame_index")
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise ValueError(
                    f"{location}: frame_index must be a non-negative integer"
                )
            if frame_index <= previous_frame:
                raise ValueError(f"{location}: frame_index must be strictly increasing")
            previous_frame = frame_index
            detections = frame.get("detections")
            if not isinstance(detections, list):
                raise ValueError(f"{location}: detections must be a list")
            camera_transform = frame.get("camera_transform")
            if camera_transform is not None:
                if not isinstance(camera_transform, list) or len(camera_transform) != 6:
                    raise ValueError(
                        f"{location}: camera_transform must contain six affine values"
                    )
                camera_transform = tuple(
                    _finite_number(value, f"{location}.camera_transform[{index}]")
                    for index, value in enumerate(camera_transform)
                )
                diagnostics = frame.get("camera_diagnostics")
                if diagnostics is not None:
                    if not isinstance(diagnostics, dict):
                        raise ValueError(
                            f"{location}: camera_diagnostics must be an object"
                        )
                    inliers = diagnostics.get("inliers")
                    if (
                        isinstance(inliers, bool)
                        or not isinstance(inliers, int)
                        or inliers < 0
                    ):
                        raise ValueError(
                            f"{location}.camera_diagnostics.inliers must be a non-negative integer"
                        )
                    camera_transform = CameraMotionEstimate(
                        transform=camera_transform,
                        confidence=_finite_number(
                            diagnostics.get("confidence"),
                            f"{location}.camera_diagnostics.confidence",
                        ),
                        reprojection_error=_finite_number(
                            diagnostics.get("reprojection_error"),
                            f"{location}.camera_diagnostics.reprojection_error",
                        ),
                        inliers=inliers,
                    )
            frames.append(
                {
                    "frame_index": frame_index,
                    "detections": [
                        _detection(detection, f"{location}.detections[{index}]")
                        for index, detection in enumerate(detections)
                    ],
                    "camera_transform": camera_transform,
                }
            )
    if not frames:
        raise ValueError(f"{path}: no frames found")
    return frames


def replay(frames: list[dict[str, Any]], config: TrackerConfig) -> list[str]:
    tracker = ObjectTracker(config)
    records: list[str] = []
    for frame in frames:
        tracks = tracker.update(
            frame["detections"],
            frame["frame_index"],
            frame["camera_transform"],
        )
        record = {
            "frame_index": frame["frame_index"],
            "tracks": [
                {
                    "id": str(track.track_id),
                    "bbox": [
                        track.x1,
                        track.y1,
                        track.x2 - track.x1,
                        track.y2 - track.y1,
                    ],
                    "confidence": track.score,
                    "class_id": track.class_id,
                    "predicted": track.predicted,
                    "occluded": track.occluded,
                    "association_confidence": track.association_confidence,
                }
                for track in tracks
            ],
        }
        records.append(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-score-threshold", type=float, default=0.30)
    parser.add_argument("--new-track-threshold", type=float, default=0.30)
    parser.add_argument("--match-iou-threshold", type=float, default=0.10)
    parser.add_argument("--max-center-distance", type=float, default=2.5)
    parser.add_argument("--velocity-momentum", type=float, default=0.80)
    parser.add_argument("--box-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--max-missing-frames", type=int, default=15)
    parser.add_argument("--min-confirmed-hits", type=int, default=1)
    parser.add_argument("--max-prediction-frames", type=int, default=0)
    parser.add_argument("--overlap-threshold", type=float, default=0.20)
    parser.add_argument("--max-occlusion-frames", type=int, default=10)
    parser.add_argument("--max-active-tracks", type=int, default=128)
    parser.add_argument("--camera-motion-compensation", action="store_true")
    parser.add_argument("--disable-center-distance", action="store_true")
    parser.add_argument("--disable-covariance-motion", action="store_true")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="Run a second independent replay and fail if any output byte differs.",
    )
    return parser.parse_args(argv)


def tracker_config(args: argparse.Namespace) -> TrackerConfig:
    config = TrackerConfig(
        high_score_threshold=args.high_score_threshold,
        new_track_threshold=args.new_track_threshold,
        match_iou_threshold=args.match_iou_threshold,
        max_center_distance=args.max_center_distance,
        velocity_momentum=args.velocity_momentum,
        box_smoothing_alpha=args.box_smoothing_alpha,
        max_missing_frames=args.max_missing_frames,
        min_confirmed_hits=args.min_confirmed_hits,
        max_prediction_frames=args.max_prediction_frames,
        camera_motion_compensation=args.camera_motion_compensation,
        overlap_threshold=args.overlap_threshold,
        max_occlusion_frames=args.max_occlusion_frames,
        max_active_tracks=args.max_active_tracks,
        center_distance_enabled=not args.disable_center_distance,
        covariance_motion_enabled=not args.disable_covariance_motion,
    )
    config.validate()
    return config


def main() -> int:
    args = parse_args()
    config = tracker_config(args)
    frames = read_frames(args.detections)
    records = replay(frames, config)
    if args.verify_determinism and replay(frames, config) != records:
        raise RuntimeError("tracker replay is not deterministic")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(records) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
