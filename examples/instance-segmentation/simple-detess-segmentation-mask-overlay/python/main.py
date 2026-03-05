"""Minimal YOLOv5 instance-segmentation overlay from DetessDequant outputs."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat

INPUT_W = 640
INPUT_H = 640

COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def class_color(cid: int):
    if cid == 0:
        return (0, 0, 255)  # person -> red (BGR)
    b = (37 * (cid + 1)) % 255
    g = (71 * (cid + 1)) % 255
    r = (103 * (cid + 1)) % 255
    return (int(b), int(g), int(r))


def class_name(cid: int) -> str:
    if 0 <= cid < 80:
        return COCO80_NAMES[cid]
    return f"class_{cid}"


def tensor_to_numpy(tensor: pyneat.Tensor) -> np.ndarray:
    dtype_map = {
        pyneat.TensorDType.UInt8: np.uint8,
        pyneat.TensorDType.Int8: np.int8,
        pyneat.TensorDType.UInt16: np.uint16,
        pyneat.TensorDType.Int16: np.int16,
        pyneat.TensorDType.Int32: np.int32,
        pyneat.TensorDType.Float32: np.float32,
        pyneat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")
    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def collect_tensors(sample: pyneat.Sample) -> list:
    out = []
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        out.append(sample.tensor)
        return out
    if sample.fields:
        for f in sample.fields:
            out.extend(collect_tensors(f))
    return out


def tensor_to_hwc(tensor: pyneat.Tensor):
    arr = tensor_to_numpy(tensor).astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError("only batch=1 is supported")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr  # HWC


def iou(a, b):
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = area_a + area_b - inter
    return inter / uni if uni > 0 else 0.0


def nms_per_class(dets, iou_thr=0.5, max_det=100):
    dets.sort(key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        suppressed = False
        for k in keep:
            if k["class_id"] == d["class_id"] and iou(
                (k["x1"], k["y1"], k["x2"], k["y2"]),
                (d["x1"], d["y1"], d["x2"], d["y2"]),
            ) > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(d)
            if len(keep) >= max_det:
                break
    return keep


def decode_yolov5_seg(tensors, infer_size):
    """Decode YOLOv5-seg DetessDequant outputs."""
    if len(tensors) < 13:
        raise ValueError("expected 13 output tensors")

    proto = tensor_to_hwc(tensors[0])  # 160x160x32
    if proto.shape != (160, 160, 32):
        raise ValueError(f"unexpected proto shape: {proto.shape}")

    conf_thr = 0.35

    dets = []
    for lvl in range(3):
        txy = tensor_to_hwc(tensors[1 + lvl * 4])
        twh = tensor_to_hwc(tensors[2 + lvl * 4])
        tco = tensor_to_hwc(tensors[3 + lvl * 4])
        tmk = tensor_to_hwc(tensors[4 + lvl * 4])

        gh, gw = txy.shape[0], txy.shape[1]
        for y in range(gh):
            for x in range(gw):
                for a in range(3):
                    tx = txy[y, x, a * 2]
                    ty = txy[y, x, a * 2 + 1]
                    tw = twh[y, x, a * 2]
                    th = twh[y, x, a * 2 + 1]

                    cls_base = a * 81
                    # DetessDequant already emits dequantized box values and scores.
                    obj = float(np.clip(tco[y, x, cls_base], 0.0, 1.0))
                    cls_scores = np.clip(tco[y, x, cls_base + 1: cls_base + 81], 0.0, 1.0)
                    best_cls = int(np.argmax(cls_scores))
                    best_cls_score = cls_scores[best_cls]

                    score = float(obj * best_cls_score)
                    if score < conf_thr:
                        continue

                    cx = float(tx)
                    cy = float(ty)
                    bw = max(0.0, float(tw))
                    bh = max(0.0, float(th))

                    x1 = max(0.0, cx - bw * 0.5)
                    y1 = max(0.0, cy - bh * 0.5)
                    x2 = min(float(infer_size), cx + bw * 0.5)
                    y2 = min(float(infer_size), cy + bh * 0.5)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    coeff = np.array([tmk[y, x, a * 32 + k] for k in range(32)])
                    dets.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score, "class_id": best_cls, "coeff": coeff,
                    })

    dets = nms_per_class(dets, 0.5, 100)
    return dets, proto


def apply_mask_overlay(bgr, dets, proto, infer_size, alpha=0.45):
    for d in dets:
        mask_small = np.zeros((proto.shape[0], proto.shape[1]), dtype=np.float32)
        for y in range(proto.shape[0]):
            for x in range(proto.shape[1]):
                v = np.dot(proto[y, x, :], d["coeff"])
                mask_small[y, x] = sigmoid(v)

        scale = proto.shape[1] / infer_size
        bx1 = max(0, int(math.floor(d["x1"] * scale)))
        by1 = max(0, int(math.floor(d["y1"] * scale)))
        bx2 = min(proto.shape[1] - 1, int(math.ceil(d["x2"] * scale)))
        by2 = min(proto.shape[0] - 1, int(math.ceil(d["y2"] * scale)))

        mask_crop = np.zeros_like(mask_small)
        mask_crop[by1:by2 + 1, bx1:bx2 + 1] = mask_small[by1:by2 + 1, bx1:bx2 + 1]

        mask = cv2.resize(mask_crop, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = np.array(class_color(d["class_id"]), dtype=np.float32)
        where = mask > 0.5
        bgr[where] = (
            (1.0 - alpha) * bgr[where].astype(np.float32) + alpha * color
        ).astype(np.uint8)


def draw_bboxes(bgr, dets, infer_size):
    sx = bgr.shape[1] / infer_size
    sy = bgr.shape[0] / infer_size
    for d in dets:
        x1 = max(0, int(round(d["x1"] * sx)))
        y1 = max(0, int(round(d["y1"] * sy)))
        x2 = min(bgr.shape[1] - 1, int(round(d["x2"] * sx)))
        y2 = min(bgr.shape[0] - 1, int(round(d["y2"] * sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        col = class_color(d["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv5 segmentation overlay")
    parser.add_argument("model", type=str, help="Path to model MPK tarball")
    parser.add_argument("input_dir", type=str, help="Input image directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3

    try:
        opt = pyneat.ModelOptions()
        opt.media_type = "video/x-raw"
        opt.format = "RGB"
        opt.input_max_width = INPUT_W
        opt.input_max_height = INPUT_H
        opt.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        print(f"Found {len(images)} images")

        ok = 0
        for image_path in images:
            src_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if src_bgr is None:
                print(f"Skipping unreadable image: {image_path.name}", file=sys.stderr)
                continue

            resized_bgr = cv2.resize(src_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            rgb_arr = np.ascontiguousarray(resized_rgb, dtype=np.uint8)

            input_tensor = pyneat.Tensor.from_numpy(
                rgb_arr, copy=True, image_format=pyneat.PixelFormat.RGB
            )
            out = model.run(input_tensor, timeout_ms=3000)
            tensors = collect_tensors(out)
            if not tensors:
                print(f"No tensor outputs for {image_path.name}", file=sys.stderr)
                continue

            try:
                dets, proto = decode_yolov5_seg(tensors, INPUT_W)
            except Exception as e:
                print(f"Decode failed for {image_path.name}: {e}", file=sys.stderr)
                continue

            mask_overlay = resized_bgr.copy()
            apply_mask_overlay(mask_overlay, dets, proto, INPUT_W)
            mask_path = output_dir / (image_path.stem + "_mask_overlay.png")
            cv2.imwrite(str(mask_path), mask_overlay)

            bbox_overlay = resized_bgr.copy()
            draw_bboxes(bbox_overlay, dets, INPUT_W)
            bbox_path = output_dir / (image_path.stem + "_bbox_overlay.png")
            cv2.imwrite(str(bbox_path), bbox_overlay)

            print(f"Wrote: {mask_path} and {bbox_path} detections={len(dets)}")
            ok += 1

        print(f"Processed {ok} / {len(images)} images")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
