"""YOLOv8n simple folder detection pipeline using pyneat."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat


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


def extract_bbox_payload_from_tensors(tensors) -> bytes | None:
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
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.enable = pyneat.AutoFlag.On
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        opt.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        opt.decode_type = pyneat.BoxDecodeType.YoloV8
        opt.score_threshold = args.min_score
        opt.nms_iou_threshold = NMS_IOU
        opt.top_k = MAX_DET
        model = pyneat.Model(args.model, opt)

        processed = 0
        for img_path in images:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            orig_h, orig_w = bgr.shape[:2]
            t_in = pyneat.Tensor.from_numpy(
                np.ascontiguousarray(bgr),
                copy=True,
                image_format=pyneat.PixelFormat.BGR,
                memory=pyneat.TensorMemory.EV74,
            )

            outputs = model.run(t_in, timeout_ms=5000)
            if not outputs:
                print(f"Model returned no output for {img_path.name}", file=sys.stderr)
                continue

            payload = extract_bbox_payload_from_tensors(outputs)
            if not payload:
                print(f"Model returned no BBOX payload for {img_path.name}", file=sys.stderr)
                continue
            boxes = parse_bbox_payload(payload, orig_w, orig_h, args.min_score)

            draw_boxes(bgr, boxes, labels)

            out_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), bgr)
            processed += 1
            print(f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name} ({len(boxes)} detections)")

        print(f"Done: {processed} images processed")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
