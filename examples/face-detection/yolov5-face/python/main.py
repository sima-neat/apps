"""yolov5s-face Python inference pipeline: faces + 5 keypoints per face."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyneat


INFER_SIZE = 800
# MPK 0_preproc.json was compiled for NV12 1280x720 input → BGR uint8 frames up to
# this size are letterboxed to INFER_SIZE on-device by the EV74 (CVU) preproc plugin.
MAX_INPUT_W = 1280
MAX_INPUT_H = 720
MIN_SCORE = 0.25
NMS_IOU = 0.45
NUM_LANDMARKS = 5

LM_COLORS = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (0, 128, 255),
)

# Mirrored from compilation.py:62-71 — keep in sync if anchors/strides change.
_STRIDES = (8, 16, 32)
_ANCHORS = (
    np.array([[  4.,   5.], [  8.,  10.], [ 13.,  16.]], dtype=np.float32),
    np.array([[ 23.,  29.], [ 43.,  55.], [ 73., 105.]], dtype=np.float32),
    np.array([[146., 217.], [231., 300.], [335., 433.]], dtype=np.float32),
)
NUM_ANCHORS = 3


def letterbox_params(orig_w: int, orig_h: int,
                     target_w: int, target_h: int) -> tuple[float, int, int]:
    """Compute (scale, pad_l, pad_t) for the on-device letterbox without applying it.

    The actual scale + center-pad is done by the MPK's CVU preproc stage on EV74
    (`0_preproc.json` configures padding=CENTER, interpolation=BILINEAR). We need
    the same parameters host-side only to inverse-map model-canvas coords back to
    original-image pixels.
    """
    scale = min(target_w / orig_w, target_h / orig_h)
    nw = int(round(orig_w * scale))
    nh = int(round(orig_h * scale))
    pad_l = (target_w - nw) // 2
    pad_t = (target_h - nh) // 2
    return scale, pad_l, pad_t


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray,
              iou_threshold: float) -> np.ndarray:
    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
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


def _decode_level_filtered(box_nhwc: np.ndarray, lm_nhwc: np.ndarray,
                           stride: int, anchors: np.ndarray, conf_threshold: float):
    _, ny, nx, _ = box_nhwc.shape
    box = box_nhwc.reshape(ny, nx, NUM_ANCHORS, 6)
    lm = lm_nhwc.reshape(ny, nx, NUM_ANCHORS, 10)

    obj = _sigmoid(box[..., 4])
    cls = _sigmoid(box[..., 5])
    scores_all = obj * cls
    mask = scores_all > conf_threshold
    if not mask.any():
        return None

    yy, xx, aa = np.nonzero(mask)
    n = yy.size

    sig_xywh = _sigmoid(box[yy, xx, aa, :4])
    aw = anchors[aa, 0]
    ah = anchors[aa, 1]
    xx_f = xx.astype(np.float32)
    yy_f = yy.astype(np.float32)
    cx = (sig_xywh[:, 0] * 2.0 - 0.5 + xx_f) * stride
    cy = (sig_xywh[:, 1] * 2.0 - 0.5 + yy_f) * stride
    bw = (sig_xywh[:, 2] * 2.0) ** 2 * aw
    bh = (sig_xywh[:, 3] * 2.0) ** 2 * ah

    boxes_xyxy = np.empty((n, 4), dtype=np.float32)
    boxes_xyxy[:, 0] = cx - bw * 0.5
    boxes_xyxy[:, 1] = cy - bh * 0.5
    boxes_xyxy[:, 2] = cx + bw * 0.5
    boxes_xyxy[:, 3] = cy + bh * 0.5

    lm_raw = lm[yy, xx, aa]
    lms = np.empty((n, NUM_LANDMARKS, 2), dtype=np.float32)
    lms[:, :, 0] = lm_raw[:, 0::2] * aw[:, None] + xx_f[:, None] * stride
    lms[:, :, 1] = lm_raw[:, 1::2] * ah[:, None] + yy_f[:, None] * stride

    return boxes_xyxy, scores_all[yy, xx, aa].astype(np.float32, copy=False), lms


def postprocess_yolov5face_split(outputs: list[np.ndarray], conf_threshold: float,
                                 iou_threshold: float):
    """Filter-first decode of yolov5s-face split heads.

    Equivalent to ModelProcessor.{decode_yolov5face_split, postprocess_yolov5face} in
    compilation.py:258-326, but reordered: compute obj*cls first, threshold, decode only
    surviving anchors. Skips ~99% of the sigmoid + grid + landmark arithmetic on real
    inputs (a 100x100 P3 box head has 30k anchors, but typical face crops yield <10
    survivors total across all three pyramid levels).
    """
    pairs: dict[int, dict[int, np.ndarray]] = {}
    for o in outputs:
        a = np.asarray(o)
        if a.ndim != 4 or a.shape[0] != 1:
            raise ValueError(f"Expected 4D [1,...] tensor, got {a.shape}")
        if a.shape[-1] in (18, 30):
            ch = a.shape[-1]
            ny, nx = a.shape[1], a.shape[2]
            arr = a.astype(np.float32, copy=False)
        elif a.shape[1] in (18, 30):
            ch = a.shape[1]
            ny, nx = a.shape[2], a.shape[3]
            arr = a.astype(np.float32, copy=False).transpose(0, 2, 3, 1)
        else:
            raise ValueError(
                f"Unrecognized split output shape {a.shape}; expected channel dim 18 or 30")
        pairs.setdefault(max(ny, nx), {})[ch] = arr

    sizes = sorted(pairs.keys(), reverse=True)
    if len(sizes) != 3:
        raise ValueError(f"expected 3 pyramid levels, got {len(sizes)}")

    out_boxes, out_scores, out_lms = [], [], []
    for lvl, size in enumerate(sizes):
        heads = pairs[size]
        if 18 not in heads or 30 not in heads:
            raise ValueError(f"Level (size={size}) missing box (18ch) or lm (30ch) head")
        result = _decode_level_filtered(
            heads[18], heads[30], _STRIDES[lvl], _ANCHORS[lvl], conf_threshold)
        if result is None:
            continue
        b, s, l = result
        out_boxes.append(b)
        out_scores.append(s)
        out_lms.append(l)

    if not out_boxes:
        return (np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0, NUM_LANDMARKS, 2), dtype=np.float32))

    boxes = np.concatenate(out_boxes, axis=0)
    scores = np.concatenate(out_scores, axis=0)
    lms = np.concatenate(out_lms, axis=0)

    keep = _nms_xyxy(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], lms[keep]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file does not exist: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def iter_tensors(sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def sample_to_outputs(sample) -> list[np.ndarray]:
    return [t.to_numpy(copy=False) for t in iter_tensors(sample)]


def unletterbox(boxes_xyxy: np.ndarray, landmarks: np.ndarray, scale: float,
                pad_l: int, pad_t: int, orig_w: int, orig_h: int):
    if len(boxes_xyxy) == 0:
        return boxes_xyxy, landmarks
    boxes = boxes_xyxy.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / scale
    np.clip(boxes[:, [0, 2]], 0, orig_w, out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, orig_h, out=boxes[:, [1, 3]])
    lms = landmarks.copy()
    lms[..., 0] = (lms[..., 0] - pad_l) / scale
    lms[..., 1] = (lms[..., 1] - pad_t) / scale
    return boxes, lms


def draw_overlay(image: np.ndarray, boxes_xyxy: np.ndarray, scores: np.ndarray,
                 landmarks: np.ndarray, label: str) -> None:
    for (x1, y1, x2, y2), score, lms in zip(boxes_xyxy, scores, landmarks):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), (0, 255, 0), -1)
        cv2.putText(image, text, (p1[0], p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for k, (lx, ly) in enumerate(lms):
            cv2.circle(image, (int(lx), int(ly)), 2, LM_COLORS[k], -1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="yolov5s-face Python inference (faces + 5 keypoints per face)")
    parser.add_argument("--model", required=True,
                        help="Path to yolov5s-face compiled model package")
    parser.add_argument("--labels", required=True,
                        help="Path to labels txt file (one label per line)")
    parser.add_argument("--input-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help=f"Detection confidence threshold (default: {MIN_SCORE})")
    parser.add_argument("--nms-iou", type=float, default=NMS_IOU,
                        help=f"NMS IoU threshold (default: {NMS_IOU})")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Repeat the image set N times for FPS measurement (default: 1)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip drawing/saving outputs (pure-throughput mode)")
    parser.add_argument("--profile", action="store_true",
                        help="Print per-stage timing summary")
    args = parser.parse_args()

    if not 0.0 <= args.min_score <= 1.0:
        print(f"Error: --min-score must be in [0.0, 1.0], got {args.min_score}", file=sys.stderr)
        return 2
    if not 0.0 <= args.nms_iou <= 1.0:
        print(f"Error: --nms-iou must be in [0.0, 1.0], got {args.nms_iou}", file=sys.stderr)
        return 2
    if args.num_runs < 1:
        print(f"Error: --num-runs must be >= 1, got {args.num_runs}", file=sys.stderr)
        return 2

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels)
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
        opt.media_type = "video/x-raw"
        opt.format = "BGR"
        opt.input_max_width = MAX_INPUT_W
        opt.input_max_height = MAX_INPUT_H
        opt.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        # On-device preproc caches kernel params per input shape; warming up at
        # MAX_INPUT_* leaves the first real frame to pay a ~2 s reconfig. Use the
        # first real image's shape so the cache is primed for the loop.
        first_bgr = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
        if first_bgr is None:
            print(f"Cannot read first image {images[0]}", file=sys.stderr)
            return 3
        if first_bgr.shape[1] > MAX_INPUT_W or first_bgr.shape[0] > MAX_INPUT_H:
            print(
                f"First image dims {first_bgr.shape[1]}x{first_bgr.shape[0]} exceed "
                f"device preproc capacity {MAX_INPUT_W}x{MAX_INPUT_H}",
                file=sys.stderr,
            )
            return 3
        t_warm = pyneat.Tensor.from_numpy(
            first_bgr, copy=False, image_format=pyneat.PixelFormat.BGR)
        model.run(t_warm, timeout_ms=10000)
        print("[WARMUP] done")

        total_images = len(images) * args.num_runs
        if args.num_runs > 1:
            print(f"Looping {args.num_runs}x over {len(images)} images "
                  f"({total_images} total)")

        pipeline_start = time.perf_counter()
        processed = 0

        for _ in range(args.num_runs):
            for img_path in images:
                img_start = time.perf_counter()

                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                    continue
                orig_h, orig_w = bgr.shape[:2]
                if orig_w > MAX_INPUT_W or orig_h > MAX_INPUT_H:
                    print(
                        f"Skipping {img_path.name}: dims {orig_w}x{orig_h} exceed device "
                        f"preproc capacity {MAX_INPUT_W}x{MAX_INPUT_H}",
                        file=sys.stderr,
                    )
                    continue
                scale, pad_l, pad_t = letterbox_params(orig_w, orig_h, INFER_SIZE, INFER_SIZE)
                t_in = pyneat.Tensor.from_numpy(
                    bgr, copy=False, image_format=pyneat.PixelFormat.BGR)

                infer_start = time.perf_counter()
                out = model.run(t_in, timeout_ms=5000)
                infer_end = time.perf_counter()

                outputs = sample_to_outputs(out)
                boxes_xyxy, scores, landmarks = postprocess_yolov5face_split(
                    outputs, args.min_score, args.nms_iou)
                boxes_xyxy, landmarks = unletterbox(
                    boxes_xyxy, landmarks, scale, pad_l, pad_t, orig_w, orig_h)
                decode_end = time.perf_counter()

                if not args.no_overlay:
                    draw_overlay(bgr, boxes_xyxy, scores, landmarks, labels[0])
                    cv2.imwrite(str(output_dir / f"{img_path.stem}.png"), bgr)
                img_end = time.perf_counter()

                processed += 1
                suffix = "" if args.no_overlay else f" -> {img_path.stem}.png"
                print(f"[{processed}/{total_images}] {img_path.name}{suffix} "
                      f"({len(boxes_xyxy)} faces)")

                if args.profile:
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

        if args.profile and processed > 0:
            pipeline_end = time.perf_counter()
            total_s = pipeline_end - pipeline_start
            avg_ms = (total_s * 1000) / processed
            fps = processed / total_s
            print(f"[PROFILE] Total: {processed} images in {total_s:.2f}s "
                  f"(avg {avg_ms:.1f}ms/image, {fps:.1f} FPS)")

        print(f"Done: {processed} images processed")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
