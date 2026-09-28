#!/usr/bin/env python3
"""Compare tracker candidates on identical detections.

Candidates:
  greedy_iou   The previous multi-stream-people-tracker matcher (no motion model).
  greedy_cv    Greedy IoU with constant-velocity prediction (yolo26-tiny-drone-tracker).
  bytetrack    This example's per-class ByteTrack-style tracker.

Inputs:
  * Synthetic scenarios with ground truth (crossing, short occlusion, missed
    detections, low-confidence frames, different speeds). Reports MOTA, IDF1,
    ID switches, and fragmentations.
  * Optional detection logs written by the app's ``output.detections_log``.
    Real video has no ground truth, so the report uses proxies: unique IDs,
    mean track length, short tracks, and re-births (a new track that starts
    where a same-class track ended a few frames earlier).

Usage:
  python3 tools/compare_trackers.py [--config CONFIG] [--log LOG.jsonl ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
TRACKING_DIR = EXAMPLE_DIR.parent
sys.path.insert(0, str(EXAMPLE_DIR / "src" / "python"))

import yaml  # noqa: E402

from utils.tracker import (  # noqa: E402
    MultiClassTracker,
    _iou_xyxy,
    linear_assignment,
    parse_class_configs,
)

BASELINE_MIN_SCORE = 0.30  # Previous app's detector threshold.


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Candidates. Each takes per-frame detections and returns (track_id, box, class).
# --------------------------------------------------------------------------- #


class GreedyIouCandidate:
    """Previous people-tracker matcher, one instance per class, score >= 0.30."""

    def __init__(self, class_ids: list[int]) -> None:
        module = _load_module("legacy_greedy_iou", Path(__file__).with_name("legacy_greedy_iou.py"))
        self.class_ids = class_ids
        self.tracker = module.PeopleTracker(iou_threshold=0.3, max_missing_frames=15)

    def update(self, dets: list[dict], frame: int):
        dets = [d for d in dets if d["class_id"] in self.class_ids and d["score"] >= BASELINE_MIN_SCORE]
        return [((t.x1, t.y1, t.x2, t.y2), t.track_id, t.class_id) for t in self.tracker.update(dets, frame)]


class GreedyCvCandidate:
    """Drone-tracker matcher with its defaults, one instance per class."""

    def __init__(self, class_ids: list[int]) -> None:
        module = _load_module(
            "drone_tracker",
            TRACKING_DIR / "yolo26-tiny-drone-tracker" / "src" / "python" / "utils" / "tracker.py",
        )
        config = module.TrackerConfig(
            high_score_threshold=0.30,
            new_track_threshold=0.30,
            match_iou_threshold=0.30,
            max_missing_frames=15,
        )
        self.trackers = {cid: module.ObjectTracker(config) for cid in class_ids}
        self._offset = {cid: i * 1_000_000 for i, cid in enumerate(class_ids)}

    def update(self, dets: list[dict], frame: int):
        out = []
        for cid, tracker in self.trackers.items():
            own = [d for d in dets if d["class_id"] == cid and d["score"] >= 0.10]
            for t in tracker.update(own, frame):
                out.append(((t.x1, t.y1, t.x2, t.y2), t.track_id + self._offset[cid], cid))
        return out


class ByteTrackCandidate:
    def __init__(self, classes_cfg: list[dict]) -> None:
        self.tracker = MultiClassTracker(parse_class_configs(classes_cfg))

    def update(self, dets: list[dict], frame: int):
        return [((t.x1, t.y1, t.x2, t.y2), t.track_id, t.class_id) for t in self.tracker.update(dets, frame)]


def build_candidates(classes_cfg: list[dict]):
    class_ids = [c.class_id for c in parse_class_configs(classes_cfg)]
    return {
        "greedy_iou": GreedyIouCandidate(class_ids),
        "greedy_cv": GreedyCvCandidate(class_ids),
        "bytetrack": ByteTrackCandidate(classes_cfg),
    }


# --------------------------------------------------------------------------- #
# Synthetic scenarios with ground truth.
# --------------------------------------------------------------------------- #


def _obj(gid, cls, x, y, vx, vy, w, h):
    return {"gid": gid, "cls": cls, "x": x, "y": y, "vx": vx, "vy": vy, "w": w, "h": h}


def make_scenario(name: str, seed: int, frames: int = 150):
    """Returns list of frames; each frame = (gt list[(gid, cls, box)], dets list[dict])."""
    rng = random.Random(seed)
    person, car = 0, 2
    objs = []
    miss_rate = 0.0
    low_conf_rate = 0.0
    occluded = {}  # gid -> set(frames)
    hide_when_overlapped = set()  # gids not detected while overlapping another object
    if name == "crossing":
        objs = [
            _obj(1, person, 100, 300, 8, 0, 60, 150),
            _obj(2, person, 1100, 315, -8, 0, 55, 140),
            _obj(3, car, 100, 600, 14, 0, 200, 110),
            _obj(4, car, 1900, 615, -14, 0, 190, 105),
        ]
        hide_when_overlapped = {2, 4}
    elif name == "short_occlusion":
        objs = [_obj(1, person, 100, 300, 5, 0, 60, 150), _obj(2, car, 100, 600, 14, 0, 200, 110)]
        occluded = {1: set(range(60, 70)), 2: set(range(50, 58))}
    elif name == "missed_detections":
        objs = [_obj(g, person, 100 + 150 * g, 200 + 40 * g, 3, 1, 60, 150) for g in range(1, 5)]
        objs += [_obj(10 + g, car, 100, 500 + 150 * g, 10 + 3 * g, 0, 200, 110) for g in range(2)]
        miss_rate = 0.25
    elif name == "low_confidence":
        objs = [_obj(g, person, 200 + 200 * g, 300, 4, 0, 60, 150) for g in range(1, 4)]
        objs += [_obj(10, car, 100, 650, 15, 0, 200, 110)]
        low_conf_rate = 0.35
    elif name == "different_speeds":
        objs = [
            _obj(1, person, 400, 300, 1, 0, 60, 150),
            _obj(2, person, 600, 320, 2, 1, 60, 150),
            _obj(3, car, 50, 600, 45, 0, 200, 110),
            _obj(4, car, 1800, 800, -70, 0, 220, 120),
            _obj(5, person, 0, 100, 25, 0, 50, 120),
        ]
        miss_rate = 0.15
    else:
        raise ValueError(name)

    out = []
    for f in range(frames):
        gt, dets = [], []
        for o in objs:
            x, y = o["x"] + o["vx"] * f, o["y"] + o["vy"] * f
            if not (-o["w"] < x < 1920 and -o["h"] < y < 1080):
                continue
            gt.append((o["gid"], o["cls"], (x, y, x + o["w"], y + o["h"])))
        boxes = {g[0]: g[2] for g in gt}
        for gid, cls, box in gt:
            o = next(item for item in objs if item["gid"] == gid)
            hidden = gid in hide_when_overlapped and any(
                other != gid and _iou_xyxy(box, other_box) > 0.0 for other, other_box in boxes.items()
            )
            if hidden or f in occluded.get(gid, ()) or rng.random() < miss_rate:
                continue
            jitter = [rng.gauss(0, 0.03 * o["h"] / 4) for _ in range(4)]
            score = rng.uniform(0.6, 0.95)
            if rng.random() < low_conf_rate:
                score = rng.uniform(0.15, 0.45)
            b = [box[i] + jitter[i] for i in range(4)]
            dets.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "score": score, "class_id": o["cls"]})
        # A few false positives.
        if rng.random() < 0.05:
            x, y = rng.uniform(0, 1700), rng.uniform(0, 900)
            dets.append({"x1": x, "y1": y, "x2": x + 60, "y2": y + 150, "score": rng.uniform(0.3, 0.55), "class_id": person})
        out.append((gt, dets))
    return out


def evaluate_gt(frames, candidate) -> dict:
    """CLEAR-MOT (IoU >= 0.5, class-aware) plus IDF1 from matched pairs."""
    fn = fp = idsw = frag = gt_total = trk_total = 0
    last_id: dict[int, int] = {}
    was_tracked: dict[int, bool] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    for f, (gt, dets) in enumerate(frames):
        tracks = candidate.update(dets, f)
        gt_total += len(gt)
        trk_total += len(tracks)
        cost = [
            [1 - _iou_xyxy(g[2], t[0]) if g[1] == t[2] and _iou_xyxy(g[2], t[0]) >= 0.5 else 1e6 for t in tracks]
            for g in gt
        ]
        pairs = linear_assignment(cost, 0.5) if gt and tracks else []
        matched_gt = set()
        for gi, ti in pairs:
            gid, tid = gt[gi][0], tracks[ti][1]
            matched_gt.add(gid)
            if gid in last_id and last_id[gid] != tid:
                idsw += 1
            if gid in was_tracked and not was_tracked[gid]:
                frag += 1
            last_id[gid] = tid
            pair_counts[(gid, tid)] = pair_counts.get((gid, tid), 0) + 1
        for g in gt:
            if g[0] not in matched_gt:
                if was_tracked.get(g[0]):
                    was_tracked[g[0]] = False
            else:
                was_tracked[g[0]] = True
        fn += len(gt) - len(pairs)
        fp += len(tracks) - len(pairs)
    gids = sorted({g for g, _ in pair_counts})
    tids = sorted({t for _, t in pair_counts})
    idtp = 0
    if gids and tids:
        cost = [[-pair_counts.get((g, t), 0) for t in tids] for g in gids]
        idtp = sum(-cost[gi][ti] for gi, ti in linear_assignment(cost, 0.0))
    idf1 = 2 * idtp / (gt_total + trk_total) if gt_total + trk_total else 0.0
    mota = 1 - (fn + fp + idsw) / gt_total if gt_total else 0.0
    return {"MOTA": mota, "IDF1": idf1, "IDSW": idsw, "Frag": frag, "FN": fn, "FP": fp}


# --------------------------------------------------------------------------- #
# Real detection logs (no ground truth).
# --------------------------------------------------------------------------- #


def load_log(path: Path):
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frames.append(
            [
                {"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "score": d[4], "class_id": d[5]}
                for d in row["dets"]
            ]
        )
    return frames


def evaluate_proxy(frames, candidate, rebirth_window: int = 30) -> dict:
    spans: dict[int, dict] = {}
    for f, dets in enumerate(frames):
        for box, tid, cls in candidate.update(dets, f):
            span = spans.setdefault(tid, {"cls": cls, "first": f, "first_box": box, "frames": 0})
            span["last"], span["last_box"] = f, box
            span["frames"] += 1
    ordered = sorted(spans.values(), key=lambda s: s["first"])
    rebirths = 0
    for s in ordered:
        for prev in ordered:
            if (
                prev is not s
                and prev["cls"] == s["cls"]
                and 0 < s["first"] - prev["last"] <= rebirth_window
                and _iou_xyxy(prev["last_box"], s["first_box"]) >= 0.3
            ):
                rebirths += 1
                break
    lengths = [s["frames"] for s in spans.values()]
    return {
        "unique_ids": len(spans),
        "mean_track_len": sum(lengths) / len(lengths) if lengths else 0.0,
        "short_tracks(<5)": sum(1 for n in lengths if n < 5),
        "rebirths": rebirths,
        "boxes_out": sum(lengths),
    }


def _table(title: str, rows: dict[str, dict]) -> str:
    keys = list(next(iter(rows.values())).keys())
    lines = [f"\n### {title}\n", "| tracker | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name, metrics in rows.items():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in metrics.values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


SCENARIOS = ["crossing", "short_occlusion", "missed_detections", "low_confidence", "different_speeds"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="config.yaml whose tracking.classes ByteTrack uses")
    parser.add_argument("--log", type=Path, action="append", default=[], help="detections_log JSONL file")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--json", type=Path, help="write all results as JSON")
    args = parser.parse_args(argv)

    synthetic_classes = [{"class": "person"}, {"class": "car"}]
    results: dict = {"synthetic": {}, "logs": {}}
    for scenario in SCENARIOS:
        totals: dict[str, dict] = {}
        for seed in range(args.seeds):
            frames = make_scenario(scenario, seed)
            for name, cand in build_candidates(synthetic_classes).items():
                m = evaluate_gt(frames, cand)
                agg = totals.setdefault(name, {k: 0.0 for k in m})
                for k, v in m.items():
                    agg[k] += v / args.seeds
        results["synthetic"][scenario] = totals
        print(_table(f"synthetic: {scenario} (mean of {args.seeds} seeds)", totals))

    if args.log:
        raw = yaml.safe_load(args.config.read_text()) if args.config else {}
        classes_cfg = ((raw or {}).get("tracking") or {}).get("classes") or synthetic_classes
        for log in args.log:
            frames = load_log(log)
            rows = {name: evaluate_proxy(frames, cand) for name, cand in build_candidates(classes_cfg).items()}
            results["logs"][log.name] = rows
            print(_table(f"log: {log.name} ({len(frames)} frames, no ground truth)", rows))

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
