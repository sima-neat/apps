"""yolo26m simple folder detection pipeline using pyneat."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml


INFER_SIZE = 640
MIN_SCORE = 0.25
NMS_IOU = 0.45
MAX_DET = 100
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def tensor_to_hwc_f32(t: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def nms_numpy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Vectorized NMS. boxes_xyxy shape (N,4), scores shape (N,). Returns kept indices."""
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.intp)


def decode_yolo26m_boxes_from_tensors(
    tensors, infer_size: int, min_score: float, nms_iou: float, max_detections: int,
) -> list[dict]:
    if len(tensors) < 6:
        raise ValueError(f"expected at least 6 tensors, got {len(tensors)}")
    regs = [tensor_to_hwc_f32(tensors[i]) for i in range(3)]
    clss = [tensor_to_hwc_f32(tensors[i]) for i in range(3, 6)]

    boxes_by_level = []
    probs_by_level = []
    for level, (reg, cls) in enumerate(zip(regs, clss)):
        if reg.shape[2] != 4:
            raise ValueError(f"expected 4-channel regression tensor, got {reg.shape[2]}")
        if reg.shape[:2] != cls.shape[:2]:
            raise ValueError("regression/class spatial mismatch")
        boxes_by_level.append(reg.reshape(-1, reg.shape[2]))
        probs_by_level.append(
            np.clip(
                cls.reshape(-1, cls.shape[2]),
                0.0,
                1.0,
            )
        )

    all_boxes = np.concatenate(boxes_by_level, axis=0)
    all_probs = np.concatenate(probs_by_level, axis=0)

    # Vectorized confidence filter (scores are already post-sigmoid).
    max_scores = all_probs.max(axis=1)
    mask = max_scores > min_score
    all_boxes = all_boxes[mask]
    all_probs = all_probs[mask]
    max_scores = max_scores[mask]

    if len(all_boxes) == 0:
        return []

    class_ids = all_probs.argmax(axis=1)

    # (cx, cy, w, h) → (x1, y1, x2, y2)
    boxes_xyxy = np.empty_like(all_boxes)
    boxes_xyxy[:, 0] = all_boxes[:, 0] - all_boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = all_boxes[:, 1] - all_boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = all_boxes[:, 0] + all_boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = all_boxes[:, 1] + all_boxes[:, 3] / 2.0
    np.clip(boxes_xyxy[:, [0, 2]], 0, infer_size, out=boxes_xyxy[:, [0, 2]])
    np.clip(boxes_xyxy[:, [1, 3]], 0, infer_size, out=boxes_xyxy[:, [1, 3]])

    # Vectorized NMS.
    keep_idx = nms_numpy(boxes_xyxy, max_scores, nms_iou)
    if len(keep_idx) > max_detections:
        keep_idx = keep_idx[:max_detections]

    result = []
    for i in keep_idx:
        result.append(dict(
            x1=float(boxes_xyxy[i, 0]), y1=float(boxes_xyxy[i, 1]),
            x2=float(boxes_xyxy[i, 2]), y2=float(boxes_xyxy[i, 3]),
            score=float(max_scores[i]), class_id=int(class_ids[i]),
        ))
    return result


def draw_boxes(frame: np.ndarray, boxes: list[dict], labels: list[str]) -> np.ndarray:
    for b in boxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        cls_id = b["class_id"]
        score = b["score"]
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def scale_boxes(boxes: list[dict], from_size: int, to_w: int, to_h: int) -> list[dict]:
    sx = to_w / float(from_size)
    sy = to_h / float(from_size)
    scaled = []
    for b in boxes:
        scaled.append(dict(
            x1=b["x1"] * sx, y1=b["y1"] * sy,
            x2=b["x2"] * sx, y2=b["y2"] * sy,
            score=b["score"], class_id=b["class_id"],
        ))
    return scaled


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="yolo26m simple folder detection pipeline")
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

    model_path = model_cfg.get("path", "assets/models/yolo26m_mod_mpk.tar.gz")
    labels_path = Path(
        model_cfg.get(
            "labels",
            "examples/object-detection/yolo26-object-detection-overlay/common/coco_label.txt",
        )
    )
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/yolo26_object_detection_overlay"))
    min_score = float(decode_cfg.get("score_threshold", MIN_SCORE))
    nms_iou = float(decode_cfg.get("nms_iou", NMS_IOU))
    max_detections = int(decode_cfg.get("max_detections", MAX_DET))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 5000))
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
    try:
        labels = load_labels(labels_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3
    print(f"Found {len(images)} images")

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.input_max_width = INFER_SIZE
        opt.preprocess.input_max_height = INFER_SIZE
        opt.preprocess.input_max_depth = 3
        model = pyneat.Model(model_path, opt)

        # Warmup inference to stabilize timing before profiling.
        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        t_dummy = pyneat.Tensor.from_numpy(
            dummy,
            copy=True,
            image_format=pyneat.PixelFormat.BGR,
            memory=pyneat.TensorMemory.EV74,
        )
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 8
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced
        runner = model.build(t_dummy, run_options=run_opt)
        runner.run(t_dummy, timeout_ms=timeout_ms)
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

            orig_h, orig_w = bgr.shape[:2]
            resized = cv2.resize(bgr, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
            resized = np.ascontiguousarray(resized, dtype=np.uint8)
            t_in = pyneat.Tensor.from_numpy(
                resized,
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )

            infer_start = time.perf_counter()
            out = runner.run(t_in, timeout_ms=timeout_ms)
            infer_end = time.perf_counter()

            boxes = decode_yolo26m_boxes_from_tensors(
                out, INFER_SIZE, min_score, nms_iou, max_detections,
            )
            decode_end = time.perf_counter()

            boxes = scale_boxes(boxes, INFER_SIZE, orig_w, orig_h)

            if overlay:
                draw_boxes(bgr, boxes, labels)
                out_path = output_dir / f"{img_path.stem}.png"
                cv2.imwrite(str(out_path), bgr)
            img_end = time.perf_counter()

            processed += 1
            det_str = f"({len(boxes)} detections)"
            if overlay:
                print(f"[{processed}/{len(all_images)}] {img_path.name} -> {img_path.stem}.png {det_str}")
            else:
                print(f"[{processed}/{len(all_images)}] {img_path.name} {det_str}")

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
