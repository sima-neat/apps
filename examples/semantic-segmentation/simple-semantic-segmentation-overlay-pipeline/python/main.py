"""Minimal semantic-segmentation overlay example for FCN-HRNet models."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat

INPUT_W = 512
INPUT_H = 512

VOC21_PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [0, 0, 128],      # aeroplane
        [0, 128, 0],      # bicycle
        [0, 128, 128],    # bird
        [128, 0, 0],      # boat
        [128, 0, 128],    # bottle
        [0, 255, 0],      # bus
        [255, 0, 0],      # car
        [0, 0, 64],       # cat
        [0, 64, 0],       # chair
        [0, 64, 64],      # cow
        [64, 0, 0],       # diningtable
        [64, 0, 64],      # dog
        [64, 64, 0],      # horse
        [64, 64, 64],     # motorbike
        [0, 0, 192],      # person
        [0, 192, 0],      # pottedplant
        [0, 192, 192],    # sheep
        [192, 0, 0],      # sofa
        [192, 0, 192],    # train
        [192, 192, 0],    # tvmonitor
    ],
    dtype=np.uint8,
)

VOC21_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "table", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


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


def find_first_tensor(sample: pyneat.Sample):
    """Find the first tensor in a sample (handles bundles)."""
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    if sample.fields:
        for field in sample.fields:
            t = find_first_tensor(field)
            if t is not None:
                return t
    return None


def logits_to_label_map(arr: np.ndarray) -> np.ndarray:
    """Convert segmentation logits to a label map via argmax."""
    dims = list(arr.shape)
    # Strip leading batch/depth=1 dims
    while len(dims) > 3 and dims[0] == 1:
        dims = dims[1:]
        arr = arr.reshape(dims)
    if len(dims) != 3:
        raise ValueError(f"Unsupported output shape: {arr.shape}")

    d0, d1, d2 = dims
    # Determine class axis (look for dim==21 for VOC)
    if d0 == 21:
        class_axis = 0
    elif d1 == 21:
        class_axis = 1
    elif d2 == 21:
        class_axis = 2
    else:
        class_axis = 2  # fallback: last dim is channels

    labels = np.argmax(arr, axis=class_axis).astype(np.uint8)
    return labels


def color_for_class(class_id: int) -> np.ndarray:
    if 0 <= class_id < 21:
        return VOC21_PALETTE[class_id]
    return np.array(
        [(37 * class_id) % 255, (67 * class_id) % 255, (97 * class_id) % 255],
        dtype=np.uint8,
    )


def draw_overlay(bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Draw segmentation overlay on BGR image."""
    overlay = bgr.copy()
    for cid in range(1, int(labels.max()) + 1):
        mask = labels == cid
        if not np.any(mask):
            continue
        color = color_for_class(cid)
        overlay[mask] = (
            (1.0 - alpha) * bgr[mask].astype(np.float32)
            + alpha * color.astype(np.float32)
        ).astype(np.uint8)
    return overlay


def print_histogram(labels: np.ndarray, image_name: str) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    items = sorted(zip(counts, unique), reverse=True)[:5]
    parts = []
    for count, cid in items:
        name = VOC21_NAMES[cid] if cid < 21 else f"class_{cid}"
        parts.append(f"{name}={count}")
    print(f"[classes] {image_name}: {', '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FCN-HRNet semantic segmentation overlay")
    parser.add_argument("model", type=str, help="Path to compiled model package")
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
            out = model.run(input_tensor, timeout_ms=5000)
            out_tensor = find_first_tensor(out)
            if out_tensor is None:
                print(f"No tensor output for: {image_path.name}", file=sys.stderr)
                continue

            logits = tensor_to_numpy(out_tensor).astype(np.float32)
            label_small = logits_to_label_map(logits)

            label_resized = cv2.resize(
                label_small, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST
            )
            print_histogram(label_resized, image_path.name)

            overlay = draw_overlay(resized_bgr, label_resized)
            out_path = output_dir / (image_path.stem + "_overlay.png")
            if not cv2.imwrite(str(out_path), overlay):
                print(f"Failed to write: {out_path}", file=sys.stderr)
                continue

            print(f"Wrote: {out_path}")
            ok += 1

        print(f"Processed {ok} / {len(images)} images")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
