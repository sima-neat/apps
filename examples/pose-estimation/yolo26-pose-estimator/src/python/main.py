"""YOLO26 pose-estimation pipeline using pyneat."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


MIN_SCORE = 0.55
NMS_IOU = 0.60
MAX_DET = 50
KEYPOINT_SCORE = 0.50
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]
KEYPOINT_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
    (255, 153, 255), (153, 204, 255), (255, 102, 255), (255, 51, 255),
    (102, 178, 255), (51, 153, 255), (255, 153, 153), (255, 102, 102),
    (255, 51, 51), (153, 255, 153), (102, 255, 102), (51, 255, 51),
    (0, 255, 0),
]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def clear_output_images(output_dir: Path, input_dir: Path) -> int:
    if output_dir.resolve() == input_dir.resolve():
        print(
            f"Skipping output cleanup because output_dir matches input_dir: {output_dir}",
            file=sys.stderr,
        )
        return 0

    removed = 0
    for path in output_dir.iterdir():
        if path.is_file() and is_image(path):
            path.unlink()
            removed += 1
    return removed


def tensor_to_numpy(tensor) -> np.ndarray:
    return np.asarray(tensor.to_numpy(copy=True))


def pose_results_from_output(result, width: int, height: int, max_detections: int) -> list[dict]:
    decoded = pyneat.decode_pose(
        list(result),
        clamp_to=(width, height),
        top_k=max_detections,
        strict=False,
    )
    poses = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32).reshape((-1, 6))
        keypoints = tensor_to_numpy(item.keypoints).astype(np.float32).reshape((-1, 17, 3))
        for box, points in zip(boxes, keypoints):
            x1, y1, x2, y2, score, class_id = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            poses.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(score),
                    "class_id": int(class_id),
                    "keypoints": points,
                }
            )
    return poses[:max_detections]


def valid_keypoint(point: np.ndarray, pose: dict, width: int, height: int) -> bool:
    x, y, score = point.tolist()
    box_w = pose["x2"] - pose["x1"]
    box_h = pose["y2"] - pose["y1"]
    margin = max(8.0, 0.10 * max(box_w, box_h))
    return (
        score >= KEYPOINT_SCORE
        and 0 <= x < width
        and 0 <= y < height
        and pose["x1"] - margin <= x <= pose["x2"] + margin
        and pose["y1"] - margin <= y <= pose["y2"] + margin
    )


def draw_pose(frame: np.ndarray, pose: dict) -> None:
    x1 = max(0, int(round(pose["x1"])))
    y1 = max(0, int(round(pose["y1"])))
    x2 = min(frame.shape[1] - 1, int(round(pose["x2"])))
    y2 = min(frame.shape[0] - 1, int(round(pose["y2"])))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {pose['score']:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    keypoints = pose["keypoints"]
    for start, end in SKELETON:
        if valid_keypoint(keypoints[start], pose, frame.shape[1], frame.shape[0]) and valid_keypoint(
            keypoints[end], pose, frame.shape[1], frame.shape[0]
        ):
            p0 = tuple(int(round(v)) for v in keypoints[start, :2])
            p1 = tuple(int(round(v)) for v in keypoints[end, :2])
            cv2.line(frame, p0, p1, (255, 0, 255), 2, cv2.LINE_AA)

    for idx, point in enumerate(keypoints):
        if valid_keypoint(point, pose, frame.shape[1], frame.shape[0]):
            center = tuple(int(round(v)) for v in point[:2])
            cv2.circle(frame, center, 3, KEYPOINT_COLORS[idx % len(KEYPOINT_COLORS)], -1, cv2.LINE_AA)


def draw_poses(frame: np.ndarray, poses: list[dict]) -> np.ndarray:
    for pose in poses:
        draw_pose(frame, pose)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO26 folder pose-estimation pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    args = parser.parse_args()

    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    raw = load_config(args.config)
    model_cfg = raw.get("model", {})
    io_cfg = raw.get("io", {})
    decode_cfg = raw.get("decode", {})
    runtime_cfg = raw.get("runtime", {})
    output_cfg = raw.get("output", {})

    model_path = model_cfg.get("path", "assets/models/yolo26m-pose-bf16-b1.tar.gz")
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/yolo26-pose-estimator"))
    min_score = float(decode_cfg.get("score_threshold", MIN_SCORE))
    nms_iou = float(decode_cfg.get("nms_iou", NMS_IOU))
    max_detections = int(decode_cfg.get("max_detections", MAX_DET))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 20000))
    num_runs = int(runtime_cfg.get("num_runs", 1))
    profile = bool(runtime_cfg.get("profile", False))
    overlay = bool(output_cfg.get("overlay", True))

    if not 0.0 <= min_score <= 1.0:
        print(f"Error: decode.score_threshold must be in [0.0, 1.0], got {min_score}", file=sys.stderr)
        return 2
    if not 0.0 <= nms_iou <= 1.0:
        print(f"Error: decode.nms_iou must be in [0.0, 1.0], got {nms_iou}", file=sys.stderr)
        return 2
    if max_detections < 1:
        print(f"Error: decode.max_detections must be >= 1, got {max_detections}", file=sys.stderr)
        return 2
    if timeout_ms <= 0:
        print(f"Error: runtime.timeout_ms must be > 0, got {timeout_ms}", file=sys.stderr)
        return 2
    if num_runs < 1:
        print(f"Error: runtime.num_runs must be >= 1, got {num_runs}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    removed_outputs = clear_output_images(output_dir, input_dir)
    if removed_outputs:
        print(f"Cleared {removed_outputs} stale output images")

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    print(f"Found {len(images)} images")

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.decode_type = pyneat.BoxDecodeType.YoloV26Pose
        opt.score_threshold = min_score
        opt.nms_iou_threshold = nms_iou
        opt.top_k = max_detections
        model = pyneat.Model(model_path, opt)

        seed = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if seed is None:
            print(f"Error: failed to read build seed image: {images[0].name}", file=sys.stderr)
            return 4
        t_dummy = pyneat.Tensor.from_numpy(
            np.ascontiguousarray(seed, dtype=np.uint8),
            copy=True,
            image_format=pyneat.PixelFormat.BGR,
            memory=pyneat.TensorMemory.EV74,
        )
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 8
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced
        runner = model.build(
            [t_dummy],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )
        runner.run([t_dummy], timeout_ms=timeout_ms)
        print("[WARMUP] done")

        all_images = images * num_runs
        if num_runs > 1:
            print(f"Looping {num_runs}x over {len(images)} images ({len(all_images)} total)")

        pipeline_start = time.perf_counter()
        processed = 0

        for img_path in all_images:
            img_start = time.perf_counter()
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            height, width = bgr.shape[:2]
            t_in = pyneat.Tensor.from_numpy(
                np.ascontiguousarray(bgr, dtype=np.uint8),
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )

            infer_start = time.perf_counter()
            out = runner.run([t_in], timeout_ms=timeout_ms)
            infer_end = time.perf_counter()

            poses = pose_results_from_output(out, width, height, max_detections)
            decode_end = time.perf_counter()

            if overlay:
                draw_poses(bgr, poses)
                out_path = output_dir / f"{img_path.stem}.png"
                cv2.imwrite(str(out_path), bgr)
            img_end = time.perf_counter()

            processed += 1
            pose_str = f"({len(poses)} poses)"
            if overlay:
                print(f"[{processed}/{len(all_images)}] {img_path.name} -> {img_path.stem}.png {pose_str}")
            else:
                print(f"[{processed}/{len(all_images)}] {img_path.name} {pose_str}")

            if profile:
                pre_ms = (infer_start - img_start) * 1000
                inf_ms = (infer_end - infer_start) * 1000
                dec_ms = (decode_end - infer_end) * 1000
                post_ms = (img_end - decode_end) * 1000
                tot_ms = (img_end - img_start) * 1000
                print(
                    f"[PROFILE] {img_path.name}: preprocess={pre_ms:.1f}ms "
                    f"inference={inf_ms:.1f}ms decode={dec_ms:.1f}ms "
                    f"overlay+save={post_ms:.1f}ms total={tot_ms:.1f}ms"
                )

        if profile and processed > 0:
            pipeline_end = time.perf_counter()
            total_s = pipeline_end - pipeline_start
            avg_ms = (total_s * 1000) / processed
            fps = processed / total_s
            print(
                f"[PROFILE] Total: {processed} images in {total_s:.1f}s "
                f"(avg {avg_ms:.1f}ms/image, {fps:.1f} FPS)"
            )

        runner.close()
        print(f"Done: {processed} images processed")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
