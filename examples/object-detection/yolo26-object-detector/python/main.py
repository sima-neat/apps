"""yolo26m simple folder detection pipeline using pyneat."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import yaml


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


def extract_bbox_payload_from_tensors(tensors) -> bytes | None:
    if len(tensors) != 1:
        return None
    for tensor in tensors:
        try:
            payload = tensor.copy_payload_bytes()
        except Exception:
            continue
        if payload:
            return payload
    return None


def parse_bbox_payload(payload: bytes, img_w: int, img_h: int, min_score: float) -> list[dict]:
    if len(payload) < 4:
        return []
    count = min(struct.unpack_from("<I", payload, 0)[0], (len(payload) - 4) // 24)
    out = []
    off = 4
    for _ in range(count):
        x, y, w, h, score, cls_id = struct.unpack_from("<iiiifi", payload, off)
        off += 24
        x1 = max(0.0, min(float(img_w), float(x)))
        y1 = max(0.0, min(float(img_h), float(y)))
        x2 = max(0.0, min(float(img_w), float(x + w)))
        y2 = max(0.0, min(float(img_h), float(y + h)))
        if x2 <= x1 or y2 <= y1 or float(score) < min_score:
            continue
        out.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score), class_id=int(cls_id)))
    return out


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
            "examples/object-detection/yolo26-object-detector/common/coco_label.txt",
        )
    )
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/yolo26-object-detector"))
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
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = min_score
        opt.nms_iou_threshold = nms_iou
        opt.top_k = max_detections
        model = pyneat.Model(model_path, opt)

        # Warmup inference to stabilize timing before profiling.
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

            orig_h, orig_w = bgr.shape[:2]
            t_in = pyneat.Tensor.from_numpy(
                np.ascontiguousarray(bgr, dtype=np.uint8),
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )

            infer_start = time.perf_counter()
            out = runner.run([t_in], timeout_ms=timeout_ms)
            infer_end = time.perf_counter()

            payload = extract_bbox_payload_from_tensors(out)
            if not payload:
                print(f"Model returned no BBOX payload for {img_path.name}", file=sys.stderr)
                continue
            boxes = parse_bbox_payload(payload, orig_w, orig_h, min_score)
            decode_end = time.perf_counter()

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
