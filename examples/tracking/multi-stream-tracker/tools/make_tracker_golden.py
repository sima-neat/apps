#!/usr/bin/env python3
"""Regenerate tests/common/tracker_golden.json from the Python tracker.

The C++ and Python unit tests replay every case and must reproduce the
expected track IDs, classes, labels, and boxes. Inputs are quantized to values
that float and double represent exactly, so both languages see the same numbers.

Usage:
  python3 tools/make_tracker_golden.py [--race-log PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_DIR / "src" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_trackers import load_log, make_scenario  # noqa: E402
from utils.tracker import MultiClassTracker, parse_class_configs  # noqa: E402

OUTPUT = EXAMPLE_DIR / "tests" / "common" / "tracker_golden.json"

TWO_CLASSES = [
    {"class": "person", "max_missing_frames": 20},
    {"class": "car", "match_iou_threshold": 0.15, "max_missing_frames": 45, "velocity_noise": 0.0125},
]
FIVE_CLASSES = [
    {"class": "person"},
    {"class": "bicycle", "min_confirmed_hits": 1},
    {"class": 2, "match_iou_threshold": 0.1, "max_missing_frames": 5},
    {"class": "dog", "high_score_threshold": 0.4, "new_track_threshold": 0.45},
    {"class": "truck", "position_noise": 0.1, "low_match_iou_threshold": 0.3},
]
RACE_CLASSES = [
    {"class": "person"},
    {"class": "motorcycle", "match_iou_threshold": 0.15, "max_missing_frames": 45, "velocity_noise": 0.0125},
]


def _q(value: float, step: float) -> float:
    return round(value / step) * step


def quantize(frames: list[list[dict]]) -> list[list[list[float]]]:
    return [
        [
            [_q(d["x1"], 1 / 16), _q(d["y1"], 1 / 16), _q(d["x2"], 1 / 16), _q(d["y2"], 1 / 16),
             _q(d["score"], 1 / 1024), int(d["class_id"])]
            for d in dets
        ]
        for dets in frames
    ]


def five_class_frames() -> list[list[dict]]:
    frames = []
    for f in range(40):
        dets = []
        for i, cls in enumerate([0, 1, 2, 16, 7]):
            if cls == 2 and 10 <= f < 20:
                continue  # car gone longer than its max_missing_frames
            if cls == 0 and f in (15, 16):
                continue  # short person gap
            x = 50 + 300 * i + 4 * f
            score = 0.47 if cls == 16 else 0.3 if (cls == 7 and f % 5 == 0) else 0.8
            dets.append({"x1": x, "y1": 100, "x2": x + 80, "y2": 260, "score": score, "class_id": cls})
        # Same box as the person, but an unconfigured class: must never join a track.
        dets.append({"x1": 50 + 4 * f, "y1": 100, "x2": 130 + 4 * f, "y2": 260, "score": 0.9, "class_id": 9})
        frames.append(dets)
    return frames


def run_case(name: str, classes: list[dict], inputs: list[list[list[float]]]) -> dict:
    tracker = MultiClassTracker(parse_class_configs(classes))
    expected = []
    for f, dets in enumerate(inputs):
        det_dicts = [
            {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "score": d[4], "class_id": d[5]} for d in dets
        ]
        expected.append(
            [
                [t.track_id, t.class_id, t.label, round(t.x1, 3), round(t.y1, 3), round(t.x2, 3), round(t.y2, 3)]
                for t in tracker.update(det_dicts, f)
            ]
        )
    return {"name": name, "classes": classes, "frames": inputs, "expected": expected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-log", type=Path, help="detections_log JSONL from a real run")
    args = parser.parse_args()

    cases = []
    for scenario, seed in (("crossing", 0), ("short_occlusion", 1), ("low_confidence", 2), ("different_speeds", 3)):
        frames = [dets for _gt, dets in make_scenario(scenario, seed, frames=120)]
        cases.append(run_case(scenario, TWO_CLASSES, quantize(frames)))
    cases.append(run_case("five_classes", FIVE_CLASSES, quantize(five_class_frames())))
    if args.race_log:
        race_frames = quantize(load_log(args.race_log)[:100])
    else:  # Keep the recorded real-video input already in the fixture.
        previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {"cases": []}
        race_frames = next((c["frames"] for c in previous["cases"] if c["name"] == "race_real"), None)
    if race_frames:
        cases.append(run_case("race_real", RACE_CLASSES, race_frames))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"cases": cases}, separators=(",", ":")) + "\n", encoding="utf-8")
    for case in cases:
        ids = {t[0] for frame in case["expected"] for t in frame}
        print(f"{case['name']}: frames={len(case['frames'])} tracks={len(ids)}")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
