"""Depth Anything V2 folder inference example using pyneat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat


INFER_SIZE = 518


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def _read_elem(raw: np.ndarray, idx: int, dtype) -> float:
    return float(raw[idx])


def depth_tensor_to_colormap(t: pyneat.Tensor) -> np.ndarray:
    if not t.is_dense():
        raise ValueError("depth output tensor is not dense")

    arr = tensor_to_numpy(t).reshape(-1)
    shape = [int(x) for x in t.shape]
    spatial = [d for d in shape if d > 1]
    if len(spatial) >= 2:
        h, w = spatial[0], spatial[1]
    elif len(spatial) == 1:
        h = w = spatial[0]
    else:
        raise ValueError(f"cannot infer spatial dims from shape={shape}")

    total = h * w
    if arr.size < total:
        raise ValueError(f"tensor payload too small: {arr.size} < {total}")

    # Match the C++ example's indexing (column-major unpack from model output).
    depth = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx = x * h + y
            depth[y, x] = _read_elem(arr, idx, t.dtype)

    minv = float(np.min(depth))
    maxv = float(np.max(depth))
    if np.isfinite(minv) and np.isfinite(maxv) and maxv > minv:
        depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_u8 = np.zeros((h, w), dtype=np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Depth Anything V2 folder inference")
    parser.add_argument("model", type=str, help="Path to depth_anything_v2_vits compiled model package")
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
    print(f"Found {len(images)} images")

    try:
        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
        opt.preprocess.input_max_width = INFER_SIZE
        opt.preprocess.input_max_height = INFER_SIZE
        opt.preprocess.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        dummy = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
        t_dummy = pyneat.Tensor.from_numpy(dummy, copy=True, image_format=pyneat.PixelFormat.RGB)
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = 4
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        runner = model.build(t_dummy, run_options=run_opt)

        processed = 0
        for img_path in images:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INFER_SIZE, INFER_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            t_in = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

            outputs = runner.run(t_in, timeout_ms=5000)
            if not outputs:
                print(f"No output tensors for {img_path.name}", file=sys.stderr)
                continue
            out_t = outputs[0]

            colormap = depth_tensor_to_colormap(out_t)
            input_resized = cv2.resize(bgr, (colormap.shape[1], colormap.shape[0]))
            subplot = np.concatenate([input_resized, colormap], axis=1)
            out_path = output_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(out_path), subplot)
            processed += 1
            print(f"[{processed}/{len(images)}] {img_path.name} -> {out_path.name}")

        runner.close()
        print(f"Done: {processed} images processed")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
