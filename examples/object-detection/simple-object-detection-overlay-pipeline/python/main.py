"""YOLOv8n simple folder detection pipeline using pyneat."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat


INFER_SIZE = 640
MIN_SCORE = 0.55
NMS_IOU = 0.50
MAX_DET = 100

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


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def extract_bbox_payload(sample: pyneat.Sample) -> bytes | None:
    stack = [sample]
    while stack:
        s = stack.pop()
        stack.extend(reversed(list(s.fields)))
        if s.kind != pyneat.SampleKind.Tensor or s.tensor is None:
            continue
        fmt = (s.payload_tag or s.format or "").upper()
        if fmt and fmt != "BBOX":
            continue
        try:
            payload = s.tensor.copy_payload_bytes()
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


def tensor_to_hwc_f32(t: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t).astype(np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def dfl_distance_16(logits: np.ndarray) -> float:
    e = np.exp(logits - float(np.max(logits)))
    denom = float(np.sum(e))
    if denom <= 0:
        return 0.0
    return float(np.dot(np.arange(16, dtype=np.float32), e) / denom)


def iou_xyxy(a, b) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def decode_yolov8_boxes_from_sample(sample: pyneat.Sample, infer_size: int, min_score: float) -> list[dict]:
    tensors = list(iter_tensors(sample))
    if len(tensors) < 6:
        raise ValueError(f"expected at least 6 tensors, got {len(tensors)}")
    regs = [tensor_to_hwc_f32(tensors[i]) for i in range(3)]
    clss = [tensor_to_hwc_f32(tensors[i]) for i in range(3, 6)]

    cand: list[dict] = []
    for reg, cls in zip(regs, clss):
        h, w, c = reg.shape
        if c < 64:
            continue
        stride = infer_size / float(h)
        for y in range(h):
            for x in range(w):
                cls_vec = cls[y, x, :]
                cls_sig = 1.0 / (1.0 + np.exp(-cls_vec))
                class_id = int(np.argmax(cls_sig))
                score = float(cls_sig[class_id])
                if score < min_score:
                    continue
                r = reg[y, x, :]
                l = dfl_distance_16(r[0:16]) * stride
                t = dfl_distance_16(r[16:32]) * stride
                rr = dfl_distance_16(r[32:48]) * stride
                b = dfl_distance_16(r[48:64]) * stride
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                x1 = max(0.0, cx - l)
                y1 = max(0.0, cy - t)
                x2 = min(float(infer_size), cx + rr)
                y2 = min(float(infer_size), cy + b)
                if x2 <= x1 or y2 <= y1:
                    continue
                cand.append(
                    dict(x1=x1, y1=y1, x2=x2, y2=y2, score=score, class_id=class_id)
                )

    cand.sort(key=lambda d: d["score"], reverse=True)
    keep: list[dict] = []
    for d in cand:
        suppressed = False
        for k in keep:
            if k["class_id"] == d["class_id"] and iou_xyxy(
                (k["x1"], k["y1"], k["x2"], k["y2"]),
                (d["x1"], d["y1"], d["x2"], d["y2"]),
            ) > NMS_IOU:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
            if len(keep) >= MAX_DET:
                break
    return keep


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


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n simple folder detection pipeline")
    parser.add_argument("model", type=str, help="Path to yolov8n compiled model package")
    parser.add_argument("labels_file", type=str, help="Path to labels txt file (one label per line)")
    parser.add_argument("input_dir", type=str, help="Input image directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help="Detection confidence threshold (default: 0.55)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    labels_path = Path(args.labels_file)
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
        opt.input_max_width = INFER_SIZE
        opt.input_max_height = INFER_SIZE
        opt.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        sess = pyneat.Session()
        sess.add(model.session())
        print(f"[BUILD] Pipeline:\n{sess.describe_backend()}")

        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        t_dummy = pyneat.Tensor.from_numpy(dummy, copy=True, image_format=pyneat.PixelFormat.BGR)
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 4
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run = sess.build(t_dummy, pyneat.RunMode.Async, run_opt)

        processed = 0
        for img_path in images:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            orig_h, orig_w = bgr.shape[:2]
            resized = cv2.resize(bgr, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
            resized = np.ascontiguousarray(resized, dtype=np.uint8)
            t_in = pyneat.Tensor.from_numpy(resized, copy=True, image_format=pyneat.PixelFormat.BGR)

            if not run.push(t_in):
                print(f"Push failed for {img_path.name}", file=sys.stderr)
                continue
            out_opt = run.pull(timeout_ms=5000)
            if out_opt is None:
                print(f"Pull failed for {img_path.name}", file=sys.stderr)
                continue

            payload = extract_bbox_payload(out_opt)
            if payload:
                boxes = parse_bbox_payload(payload, INFER_SIZE, INFER_SIZE, args.min_score)
            else:
                boxes = decode_yolov8_boxes_from_sample(out_opt, INFER_SIZE, args.min_score)

            boxes = scale_boxes(boxes, INFER_SIZE, orig_w, orig_h)
            draw_boxes(bgr, boxes, labels)

            out_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), bgr)
            processed += 1
            print(f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name} ({len(boxes)} detections)")

        run.close()
        print(f"Done: {processed} images processed")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
