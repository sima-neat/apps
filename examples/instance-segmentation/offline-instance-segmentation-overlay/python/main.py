"""Minimal YOLOv8-seg pipeline using DetessDequant postprocess (no boxdecode)."""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat

INFER_SIZE = 640
SCORE_THR = 0.6
NMS_IOU = 0.45
MAX_DET = 200
MASK_ALPHA = 0.65

MASK_COLOR_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (255, 194, 0),
    (168, 153, 44),
]

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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def class_color(class_id: int):
    if class_id < 0:
        class_id = 0
    return MASK_COLOR_PALETTE[class_id % len(MASK_COLOR_PALETTE)]


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
    uni = area_a + area_b - inter
    return inter / uni if uni > 0 else 0.0


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


def tensors_from_sample(sample: pyneat.Sample) -> list:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return [sample.tensor]
    if sample.fields:
        out = []
        for field in sample.fields:
            out.extend(tensors_from_sample(field))
        return out
    raise RuntimeError("unexpected sample kind")


def tensor_to_hwc_f32(tensor: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(tensor).astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError("only batch=1 supported")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def dfl_distance_16(logits: np.ndarray) -> float:
    maxv = logits.max()
    e = np.exp(logits - maxv)
    denom = e.sum()
    if denom <= 0:
        return 0.0
    numer = np.dot(np.arange(16, dtype=np.float32), e)
    return float(numer / denom)


def decode_yolov8_instances(tensors, infer_size, conf_thr, nms_iou, max_det):
    """Decode YOLOv8 boxes and mask coefficients from DetessDequant outputs."""
    if len(tensors) < 10:
        raise ValueError("expected at least 10 tensors for instance-seg decode")

    reg80 = tensor_to_hwc_f32(tensors[0])
    reg40 = tensor_to_hwc_f32(tensors[1])
    reg20 = tensor_to_hwc_f32(tensors[2])
    cls80 = tensor_to_hwc_f32(tensors[3])
    cls40 = tensor_to_hwc_f32(tensors[4])
    cls20 = tensor_to_hwc_f32(tensors[5])
    mk80 = tensor_to_hwc_f32(tensors[6])
    mk40 = tensor_to_hwc_f32(tensors[7])
    mk20 = tensor_to_hwc_f32(tensors[8])
    proto = tensor_to_hwc_f32(tensors[9])
    if proto.shape[2] != 32:
        raise ValueError(f"unexpected proto channels: {proto.shape}")

    levels = [(reg80, cls80, mk80), (reg40, cls40, mk40), (reg20, cls20, mk20)]
    candidates = []

    for reg, cls, mk in levels:
        h, w, _ = reg.shape
        if cls.shape[0] != h or cls.shape[1] != w or reg.shape[2] != 64:
            raise ValueError("unexpected reg/cls shape")
        if mk.shape[0] != h or mk.shape[1] != w or mk.shape[2] != 32:
            raise ValueError("unexpected mask coeff shape")
        stride = infer_size / h

        for y in range(h):
            for x in range(w):
                cls_vec = cls[y, x, :]
                cls_sigmoid = 1.0 / (1.0 + np.exp(-cls_vec))
                best_cls = int(np.argmax(cls_sigmoid))
                best_score = float(cls_sigmoid[best_cls])
                if best_score < conf_thr:
                    continue

                reg_vec = reg[y, x, :]
                l = dfl_distance_16(reg_vec[0:16]) * stride
                t = dfl_distance_16(reg_vec[16:32]) * stride
                r = dfl_distance_16(reg_vec[32:48]) * stride
                b = dfl_distance_16(reg_vec[48:64]) * stride

                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride

                x1 = max(0.0, cx - l)
                y1 = max(0.0, cy - t)
                x2 = min(float(infer_size), cx + r)
                y2 = min(float(infer_size), cy + b)
                if x2 > x1 and y2 > y1:
                    candidates.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": best_score, "class_id": best_cls,
                        "coeff": mk[y, x, :].copy(),
                    })

    candidates.sort(key=lambda b: b["score"], reverse=True)
    keep = []
    for b in candidates:
        suppressed = False
        for k in keep:
            if k["class_id"] == b["class_id"] and iou_xyxy(
                (k["x1"], k["y1"], k["x2"], k["y2"]),
                (b["x1"], b["y1"], b["x2"], b["y2"]),
            ) > nms_iou:
                suppressed = True
                break
        if not suppressed:
            keep.append(b)
            if len(keep) >= max_det:
                break
    return keep, proto


def apply_mask_overlay(bgr, dets, proto, infer_size, alpha=MASK_ALPHA):
    ph, pw, pc = proto.shape
    if pc != 32:
        raise ValueError(f"unexpected proto channels: {proto.shape}")
    for d in dets:
        coeff = d.get("coeff")
        if coeff is None or coeff.shape[0] != pc:
            continue
        mask_small = np.tensordot(proto, coeff, axes=([2], [0]))
        mask_small = 1.0 / (1.0 + np.exp(-mask_small))

        sx = pw / infer_size
        sy = ph / infer_size
        bx1 = max(0, int(math.floor(d["x1"] * sx)))
        by1 = max(0, int(math.floor(d["y1"] * sy)))
        bx2 = min(pw - 1, int(math.ceil(d["x2"] * sx)))
        by2 = min(ph - 1, int(math.ceil(d["y2"] * sy)))

        mask_crop = np.zeros_like(mask_small)
        mask_crop[by1:by2 + 1, bx1:bx2 + 1] = mask_small[by1:by2 + 1, bx1:bx2 + 1]
        mask = cv2.resize(mask_crop, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        where = mask > 0.5
        if not np.any(where):
            continue
        class_col = class_color(d["class_id"])
        color = np.array(class_col, dtype=np.float32)
        bgr[where] = (
            (1.0 - alpha) * bgr[where].astype(np.float32) + alpha * color
        ).astype(np.uint8)

        # Draw crisp contour with the same class color used by the mask fill and bbox.
        contour_mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(bgr, contours, -1, class_col, 2, cv2.LINE_8)


def draw_boxes(bgr, boxes, infer_size):
    sx = bgr.shape[1] / infer_size
    sy = bgr.shape[0] / infer_size
    for b in boxes:
        x1 = max(0, int(round(b["x1"] * sx)))
        y1 = max(0, int(round(b["y1"] * sy)))
        x2 = min(bgr.shape[1] - 1, int(round(b["x2"] * sx)))
        y2 = min(bgr.shape[0] - 1, int(round(b["y2"] * sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        col = class_color(b["class_id"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
        name = COCO80_NAMES[b["class_id"]] if b["class_id"] < 80 else f"class_{b['class_id']}"
        label = f"{name} s={b['score']:.2f}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n instance segmentation overlay")
    parser.add_argument("model", type=str, help="Path to compiled model package")
    parser.add_argument("input_dir", type=str, help="Input image directory")
    parser.add_argument("output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        raise RuntimeError(f"no images found in: {input_dir}")

    try:
        opt = pyneat.ModelOptions()
        opt.media_type = "video/x-raw"
        opt.format = "RGB"
        opt.input_max_width = INFER_SIZE
        opt.input_max_height = INFER_SIZE
        opt.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        print(f"Found {len(images)} images")

        processed = 0
        for image_path in images:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable image: {image_path.name}", file=sys.stderr)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb_arr = np.ascontiguousarray(rgb, dtype=np.uint8)

            input_tensor = pyneat.Tensor.from_numpy(
                rgb_arr, copy=True, image_format=pyneat.PixelFormat.RGB
            )
            out = model.run(input_tensor, timeout_ms=3000)
            tensors = tensors_from_sample(out)

            try:
                boxes, proto = decode_yolov8_instances(tensors, INFER_SIZE, SCORE_THR, NMS_IOU, MAX_DET)
            except Exception as e:
                print(f"Decode failed for {image_path.name}: {e}", file=sys.stderr)
                continue

            overlay = bgr.copy()
            apply_mask_overlay(overlay, boxes, proto, INFER_SIZE)
            draw_boxes(overlay, boxes, INFER_SIZE)
            out_file = output_dir / (image_path.stem + "_overlay.jpg")
            cv2.imwrite(str(out_file), overlay)

            print(f"Wrote: {out_file} boxes={len(boxes)}")
            processed += 1

        print(f"Processed {processed} / {len(images)} images")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
