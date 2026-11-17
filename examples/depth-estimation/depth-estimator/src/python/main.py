"""Depth Anything V2 folder inference example using pyneat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def depth_tensor_to_colormap(t: pyneat.Tensor) -> np.ndarray:
    """Color one row-major height x width depth map; larger (closer) values come out brighter."""
    if not t.is_dense():
        raise ValueError("depth output tensor is not dense")

    shape = [int(x) for x in t.shape]
    spatial = [d for d in shape if d > 1]
    if len(spatial) != 2:
        raise ValueError(f"expected one height x width depth map, got shape={shape}")
    h, w = spatial
    depth = np.asarray(t.to_numpy(copy=True), dtype=np.float32).reshape(h, w)

    minv = float(np.min(depth))
    maxv = float(np.max(depth))
    if np.isfinite(minv) and np.isfinite(maxv) and maxv > minv:
        depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_u8 = np.zeros((h, w), dtype=np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)


def main() -> int:
    parser = argparse.ArgumentParser(description="Depth Anything V2 folder inference")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    args = parser.parse_args()

    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_path = raw.get("model", {}).get(
        "path", "models/depth_anything_v2_small/w8a8_mixed/depth_anything_v2_small_w8a8_mixed_mpk.tar.gz"
    )
    io_cfg = raw.get("io", {})
    runtime = raw.get("runtime", {})
    input_dir = Path(io_cfg.get("input_dir", "assets/datasets/coco"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/depth-estimator"))
    max_input_size = int(runtime.get("max_input_size", 1920))
    queue_depth = int(runtime.get("queue_depth", 4))
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
        opt.preprocess.input_max_width = max_input_size
        opt.preprocess.input_max_height = max_input_size
        opt.preprocess.input_max_depth = 3
        # Depth Anything V2 expects a bicubic stretch to the model input, whose size Neat reads from
        # the package, then ImageNet mean and std. Neat normalizes only when asked, so set the preset.
        opt.preprocess.resize.enable = pyneat.AutoFlag.On
        opt.preprocess.resize.mode = pyneat.ResizeMode.Stretch
        opt.preprocess.resize.scaling_type = "BICUBIC"
        opt.preprocess.preset = pyneat.NormalizePreset.ImageNet
        model = pyneat.Model(model_path, opt)
        contract = model.resolved_preprocess_plan().mla_contract
        model_w, model_h = int(contract.width), int(contract.height)
        if model_w <= 0 or model_h <= 0:
            raise RuntimeError("model input size is unresolved")

        dummy = np.zeros((model_h, model_w, 3), dtype=np.uint8)
        t_dummy = pyneat.Tensor.from_numpy(dummy, copy=True, image_format=pyneat.PixelFormat.RGB)
        run_opt = pyneat.RunOptions()
        run_opt.queue_depth = queue_depth
        run_opt.overflow_policy = pyneat.OverflowPolicy.Block
        runner = model.build(
            [t_dummy],
            route_options=pyneat.ModelRouteOptions(),
            run_options=run_opt,
        )

        processed = 0
        for img_path in images:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable: {img_path.name}", file=sys.stderr)
                continue

            scale = max_input_size / max(bgr.shape[:2])
            if scale < 1:
                size = (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale)))
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)

            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            t_in = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)

            outputs = runner.run([t_in])
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
