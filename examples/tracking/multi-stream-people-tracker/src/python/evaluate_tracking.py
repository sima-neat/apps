#!/usr/bin/env python3
"""Evaluate detector and tracker JSONL output against frame-level ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"bbox must contain four values, got {box!r}")
    x, y, width, height = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError(f"bbox values must be finite, got {box!r}")
    if width < 0 or height < 0:
        raise ValueError(f"bbox width and height must be non-negative, got {box!r}")
    return x, y, x + width, y + height


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union_area = area_a + area_b - intersection
    return intersection / union_area if union_area > 0 else 0.0


def read_jsonl(path: Path, array_key: str) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict):
                raise ValueError(f"{path}:{line_number}: frame must be an object")
            frame_index = document.get("frame_index")
            if frame_index is None:
                frame_id = document.get("frame_id")
                if isinstance(frame_id, str) and frame_id.isdigit():
                    frame_index = int(frame_id)
                elif isinstance(frame_id, int):
                    frame_index = frame_id
            data = document.get("data")
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: data is not valid JSON") from exc
            if array_key not in document and isinstance(data, dict) and array_key in data:
                document = {**document, array_key: data[array_key]}
            if not isinstance(frame_index, int) or frame_index < 0:
                raise ValueError(f"{path}:{line_number}: frame_index must be a non-negative integer")
            if frame_index in frames:
                raise ValueError(f"{path}:{line_number}: duplicate frame_index {frame_index}")
            if not isinstance(document.get(array_key), list):
                raise ValueError(f"{path}:{line_number}: {array_key} must be a list")
            document["frame_index"] = frame_index
            frames[frame_index] = document
    if not frames:
        raise ValueError(f"{path}: no frames found")
    return frames


def greedy_matches(
    truth: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> list[tuple[int, int, float]]:
    candidates = []
    for truth_index, truth_object in enumerate(truth):
        truth_box = xywh_to_xyxy(truth_object["bbox"])
        for prediction_index, prediction in enumerate(predictions):
            overlap = iou(truth_box, xywh_to_xyxy(prediction["bbox"]))
            if overlap >= threshold:
                candidates.append((overlap, truth_index, prediction_index))
    candidates.sort(reverse=True)
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    matches = []
    for overlap, truth_index, prediction_index in candidates:
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, overlap))
    return matches


def model_box_area(box: list[float], source_width: int, source_height: int, model_size: int) -> float:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("ground-truth frame width and height must be positive")
    scale = min(model_size / source_width, model_size / source_height)
    return max(0.0, float(box[2])) * max(0.0, float(box[3])) * scale * scale


def size_bucket(area: float) -> str:
    if area < 16.0**2:
        return "tiny"
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def valid_track_id(value: Any) -> bool:
    return (
        isinstance(value, (str, int))
        and not isinstance(value, bool)
        and str(value) != ""
    )


def evaluate(
    truth_frames: dict[int, dict[str, Any]],
    prediction_frames: dict[int, dict[str, Any]],
    *,
    iou_threshold: float,
    fps: float,
    model_size: int,
) -> dict[str, Any]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if fps <= 0.0 or model_size <= 0:
        raise ValueError("fps and model_size must be positive")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    overlap_sum = 0.0
    size_counts = {name: {"ground_truth": 0, "matched": 0} for name in ("tiny", "small", "medium", "large")}
    last_prediction_by_truth: dict[str, str] = {}
    truth_was_matched: dict[str, bool] = {}
    truth_seen_match: set[str] = set()
    id_switches = 0
    fragmentations = 0

    ground_truth_ids_available = all(
        valid_track_id(obj.get("track_id"))
        for frame in truth_frames.values()
        for obj in frame.get("objects", [])
    )
    prediction_ids_available = all(
        valid_track_id(obj.get("id"))
        for frame in prediction_frames.values()
        for obj in frame.get("tracks", [])
    )
    tracking_available = ground_truth_ids_available and prediction_ids_available
    unavailable_reasons = []
    if not ground_truth_ids_available:
        unavailable_reasons.append("ground-truth objects require non-empty track_id values")
    if not prediction_ids_available:
        unavailable_reasons.append("predicted tracks require non-empty id values")

    frame_indices = sorted(set(truth_frames) | set(prediction_frames))
    for frame_index in frame_indices:
        truth_frame = truth_frames.get(frame_index, {"objects": []})
        prediction_frame = prediction_frames.get(frame_index, {"tracks": []})
        truth = truth_frame.get("objects", [])
        predictions = prediction_frame.get("tracks", [])
        width = int(truth_frame.get("width", 0))
        height = int(truth_frame.get("height", 0))
        matches = greedy_matches(truth, predictions, iou_threshold)
        matched_truth_indices = {match[0] for match in matches}
        true_positives += len(matches)
        false_positives += len(predictions) - len(matches)
        false_negatives += len(truth) - len(matches)
        overlap_sum += sum(match[2] for match in matches)

        for truth_index, truth_object in enumerate(truth):
            bucket = size_bucket(model_box_area(truth_object["bbox"], width, height, model_size))
            size_counts[bucket]["ground_truth"] += 1
            if truth_index in matched_truth_indices:
                size_counts[bucket]["matched"] += 1

        matches_by_truth = {truth_index: prediction_index for truth_index, prediction_index, _ in matches}
        if not tracking_available:
            continue
        for truth_index, truth_object in enumerate(truth):
            truth_id = str(truth_object["track_id"])
            prediction_index = matches_by_truth.get(truth_index)
            if prediction_index is None:
                if truth_id in truth_seen_match:
                    truth_was_matched[truth_id] = False
                continue
            raw_prediction_id = predictions[prediction_index].get("id")
            prediction_id = str(raw_prediction_id)
            if truth_id in truth_seen_match and not truth_was_matched.get(truth_id, False):
                fragmentations += 1
            previous_prediction = last_prediction_by_truth.get(truth_id)
            if previous_prediction is not None and prediction_id != previous_prediction:
                id_switches += 1
            last_prediction_by_truth[truth_id] = prediction_id
            truth_was_matched[truth_id] = True
            truth_seen_match.add(truth_id)

    precision = safe_ratio(true_positives, true_positives + false_positives)
    recall = safe_ratio(true_positives, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    duration_minutes = len(frame_indices) / fps / 60.0
    return {
        "schema_version": 1,
        "configuration": {"fps": fps, "iou_threshold": iou_threshold, "model_size": model_size},
        "frames": len(frame_indices),
        "detection": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": safe_ratio(overlap_sum, true_positives),
            "false_positives_per_minute": false_positives / duration_minutes if duration_minutes else 0.0,
            "recall_by_model_input_size": {
                name: {
                    **counts,
                    "recall": safe_ratio(counts["matched"], counts["ground_truth"]),
                }
                for name, counts in size_counts.items()
            },
        },
        "tracking": {
            "available": tracking_available,
            "unavailable_reason": "; ".join(unavailable_reasons) or None,
            "ground_truth_track_count": (
                len(
                    {
                        str(obj["track_id"])
                        for frame in truth_frames.values()
                        for obj in frame.get("objects", [])
                    }
                )
                if tracking_available
                else None
            ),
            "id_switches": id_switches if tracking_available else None,
            "fragmentations": fragmentations if tracking_available else None,
        },
    }


def enforce_gates(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures = []
    detection = report["detection"]
    tracking = report["tracking"]
    if report["frames"] < args.minimum_frames:
        failures.append(f"frames {report['frames']} < required {args.minimum_frames}")
    if detection["recall"] < args.minimum_recall:
        failures.append(f"recall {detection['recall']:.6f} < required {args.minimum_recall:.6f}")
    tiny_recall = detection["recall_by_model_input_size"]["tiny"]["recall"]
    if tiny_recall < args.minimum_tiny_recall:
        failures.append(f"tiny recall {tiny_recall:.6f} < required {args.minimum_tiny_recall:.6f}")
    if detection["false_positives_per_minute"] > args.maximum_false_positives_per_minute:
        failures.append(
            "false positives/minute "
            f"{detection['false_positives_per_minute']:.6f} > allowed "
            f"{args.maximum_false_positives_per_minute:.6f}"
        )
    tracking_gate_requested = (
        args.maximum_id_switches is not None
        or args.maximum_fragmentations is not None
    )
    if tracking_gate_requested and not tracking["available"]:
        failures.append(f"tracking metrics unavailable: {tracking['unavailable_reason']}")
    elif tracking["available"]:
        if (
            args.maximum_id_switches is not None
            and tracking["id_switches"] > args.maximum_id_switches
        ):
            failures.append(
                f"id switches {tracking['id_switches']} > allowed {args.maximum_id_switches}"
            )
        if (
            args.maximum_fragmentations is not None
            and tracking["fragmentations"] > args.maximum_fragmentations
        ):
            failures.append(
                f"fragmentations {tracking['fragmentations']} > allowed {args.maximum_fragmentations}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--model-size", type=int, default=640)
    parser.add_argument("--minimum-frames", type=int, default=1)
    parser.add_argument("--minimum-recall", type=float, default=0.0)
    parser.add_argument("--minimum-tiny-recall", type=float, default=0.0)
    parser.add_argument("--maximum-false-positives-per-minute", type=float, default=math.inf)
    parser.add_argument("--maximum-id-switches", type=int)
    parser.add_argument("--maximum-fragmentations", type=int)
    args = parser.parse_args()
    if args.minimum_frames < 1:
        parser.error("minimum-frames must be positive")
    if not 0.0 <= args.minimum_recall <= 1.0 or not 0.0 <= args.minimum_tiny_recall <= 1.0:
        parser.error("minimum recall gates must be in [0, 1]")
    if args.maximum_false_positives_per_minute < 0.0:
        parser.error("maximum-false-positives-per-minute must be >= 0")
    if (
        args.maximum_id_switches is not None
        and args.maximum_id_switches < 0
    ) or (
        args.maximum_fragmentations is not None
        and args.maximum_fragmentations < 0
    ):
        parser.error("tracking count gates must be >= 0")

    report = evaluate(
        read_jsonl(args.ground_truth, "objects"),
        read_jsonl(args.predictions, "tracks"),
        iou_threshold=args.iou_threshold,
        fps=args.fps,
        model_size=args.model_size,
    )
    failures = enforce_gates(report, args)
    report["gates"] = {"passed": not failures, "failures": failures}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
