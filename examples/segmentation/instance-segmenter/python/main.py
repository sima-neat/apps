"""Minimal YOLOv8-seg pipeline using DetessDequant postprocess (no boxdecode)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat
import yaml

MASK_ALPHA = 0.65
DQ_SCALES = [
    11.110991093672158,
    13.956323724394647,
    14.356056332484277,
    925.8771585551377,
    343.5148027214258,
    291.36864499455385,
    50.265630377702685,
    56.57840021254144,
    54.44306583495789,
    43.49836481362829,
]

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


def tensor_to_hwc_f32(tensor: pyneat.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(tensor).astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError("only batch=1 supported")
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected tensor rank {arr.ndim}")
    return arr


def apply_dq_scale_squared_probe(arr: np.ndarray, tensor_index: int) -> np.ndarray:
    scale = DQ_SCALES[tensor_index]
    return arr / (scale * scale)


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor:
        if sample.tensor is None:
            raise RuntimeError("tensor sample missing payload")
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    elif sample.kind == pyneat.SampleKind.Bundle:
        for field in sample.fields:
            yield from iter_tensors(field)
    else:
        raise RuntimeError(f"unexpected sample kind: {sample.kind}")


def dfl_distance_16(logits: np.ndarray) -> float:
    maxv = logits.max()
    e = np.exp(logits - maxv)
    denom = e.sum()
    if denom <= 0:
        return 0.0
    numer = np.dot(np.arange(16, dtype=np.float32), e)
    return float(numer / denom)


def decode_yolov8_instances(tensors, infer_size, conf_thr, nms_iou, max_det, dq_scale_squared_probe=False):
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
    if dq_scale_squared_probe:
        reg80 = apply_dq_scale_squared_probe(reg80, 0)
        reg40 = apply_dq_scale_squared_probe(reg40, 1)
        reg20 = apply_dq_scale_squared_probe(reg20, 2)
        cls80 = apply_dq_scale_squared_probe(cls80, 3)
        cls40 = apply_dq_scale_squared_probe(cls40, 4)
        cls20 = apply_dq_scale_squared_probe(cls20, 5)
        mk80 = apply_dq_scale_squared_probe(mk80, 6)
        mk40 = apply_dq_scale_squared_probe(mk40, 7)
        mk20 = apply_dq_scale_squared_probe(mk20, 8)
        proto = apply_dq_scale_squared_probe(proto, 9)
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
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_path = raw.get("model", {}).get("path", "assets/models/yolo_v8n_seg_mpk.tar.gz")
    io_cfg = raw.get("io", {})
    runtime = raw.get("runtime", {})
    decode = raw.get("decode", {})
    visualization = raw.get("visualization", {})
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/instance-segmenter"))
    infer_size = int(runtime.get("infer_size", 640))
    timeout_ms = int(runtime.get("timeout_ms", 20000))
    queue_depth = int(runtime.get("queue_depth", 8))
    score_thr = float(decode.get("score_threshold", 0.6))
    nms_iou = float(decode.get("nms_iou", 0.45))
    max_det = int(decode.get("max_detections", 200))
    mask_alpha = float(visualization.get("mask_alpha", MASK_ALPHA))
    dq_scale_squared_probe = os.getenv("SIMANEAT_APPS_DQ_SCALE_SQUARED_PROBE", "") not in {"", "0"}
    if dq_scale_squared_probe:
        print("[DEBUG] Dividing detess outputs by dq_scale^2 from 0_postproc.json")

    if not input_dir.is_dir():
        raise RuntimeError(f"input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not images:
        raise RuntimeError(f"no images found in: {input_dir}")

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
        opt.preprocess.input_max_width = infer_size
        opt.preprocess.input_max_height = infer_size
        opt.preprocess.input_max_depth = 3
        model = pyneat.Model(model_path, opt)

        dummy = np.zeros((infer_size, infer_size, 3), dtype=np.uint8)
        dummy_tensor = pyneat.Tensor.from_numpy(
            dummy,
            copy=True,
            image_format=pyneat.PixelFormat.RGB,
            memory=pyneat.TensorMemory.EV74,
        )
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = queue_depth
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        run_opt.preset = pyneat.RunPreset.Balanced
        graph = pyneat.Graph()
        graph.add(pyneat.nodes.input(model.input_appsrc_options(False)))
        graph.add(model.preprocess())
        graph.add(model.inference())
        graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
        graph.add(pyneat.nodes.output())

        runner = graph.build([dummy_tensor], pyneat.RunMode.Async, run_opt)

        print(f"Found {len(images)} images")

        processed = 0
        for image_path in images:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable image: {image_path.name}", file=sys.stderr)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
            rgb_arr = np.ascontiguousarray(rgb, dtype=np.uint8)

            input_tensor = pyneat.Tensor.from_numpy(
                rgb_arr,
                copy=True,
                image_format=pyneat.PixelFormat.RGB,
                memory=pyneat.TensorMemory.EV74,
            )
            output = runner.run([input_tensor], timeout_ms=timeout_ms)
            tensors = list(iter_tensors(output)) if hasattr(output, "kind") else list(output)

            try:
                boxes, proto = decode_yolov8_instances(
                    tensors, infer_size, score_thr, nms_iou, max_det, dq_scale_squared_probe
                )
            except Exception as e:
                print(f"Decode failed for {image_path.name}: {e}", file=sys.stderr)
                continue

            overlay = bgr.copy()
            apply_mask_overlay(overlay, boxes, proto, infer_size, alpha=mask_alpha)
            draw_boxes(overlay, boxes, infer_size)
            out_file = output_dir / (image_path.stem + "_overlay.jpg")
            cv2.imwrite(str(out_file), overlay)

            print(f"Wrote: {out_file} boxes={len(boxes)}")
            processed += 1

        runner.close()
        print(f"Processed {processed} / {len(images)} images")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
