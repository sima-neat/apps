#!/usr/bin/env python3
"""MAE-based medical anomaly detection pipeline using two NEAT models.

This example:
- loads 2D medical slices from .npy files,
- normalizes and resizes them, duplicating to 3 channels (H, W, 3),
- runs a MAE reconstruction model with a deterministic grid mask,
- pastes reconstructed patches back into the original image,
- computes the absolute difference image, and
- runs an anomaly classifier on the diff to output normal/anomalous probabilities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import pyneat as neat


PATCH_SIZE = 16
INFER_HEIGHT, INFER_WIDTH = 224, 224


def get_deterministic_mask(num_patches: int, seed: int = 42) -> np.ndarray:
    """1-in-4 grid mask identical to the one baked into the model export."""
    state = np.random.get_state()
    np.random.seed(seed)

    arr = np.random.random(num_patches)
    mask = []
    step = 4
    for i in range(0, len(arr), step):
        max_val = np.max(arr[i : i + step])
        for j in range(i, min(i + step, len(arr))):
            mask.append(arr[j] != max_val)  # 0 = keep, 1 = remove

    np.random.set_state(state)
    return np.array(mask).astype(np.float32)


def tensor_to_numpy_dense(tensor: neat.Tensor) -> np.ndarray:
    """Convert a dense pyneat tensor to a NumPy array via copy_dense_bytes_tight."""
    dtype_map = {
        neat.TensorDType.UInt8: np.uint8,
        neat.TensorDType.Int8: np.int8,
        neat.TensorDType.UInt16: np.uint16,
        neat.TensorDType.Int16: np.int16,
        neat.TensorDType.Int32: np.int32,
        neat.TensorDType.Float32: np.float32,
        neat.TensorDType.Float64: np.float64,
    }
    np_dtype = dtype_map.get(tensor.dtype)
    if np_dtype is None:
        raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")
    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def load_npy_as_rgb(path: Path) -> np.ndarray:
    """Load a single-channel .npy, resize in float space, duplicate to 3 channels.

    This preserves the original intensity scale (no per-slice min–max). The only
    change is interpolation from the original size to (INFER_HEIGHT, INFER_WIDTH).
    """
    npy_image = np.load(path).astype(np.float32)  # (H, W, C) or (H, W)
    # Debug: inspect original NPY slice statistics before any processing.
    print(
        f"raw npy '{path}' stats:",
        "shape", npy_image.shape,
        "dtype", npy_image.dtype,
        "min", float(npy_image.min()),
        "max", float(npy_image.max()),
    )
    if npy_image.ndim == 3:
        npy_image = npy_image[:, :, 0]

    # (H, W) -> (H, W, 1)
    npy_image = np.expand_dims(npy_image, axis=2)

    # Resize to model resolution in float space, preserving the numeric range.
    npy_resized = resize(
        npy_image,
        (INFER_HEIGHT, INFER_WIDTH, 1),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    npy_resized = npy_resized[:, :, 0]  # back to (H, W)

    # Duplicate to 3 channels: (H, W, 3) float32 in the original intensity scale.
    return np.stack([npy_resized, npy_resized, npy_resized], axis=-1)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1D logits array."""
    logits = logits.astype(np.float32)
    exp_x = np.exp(logits - np.max(logits))
    return exp_x / exp_x.sum()


def run_pipeline(
    mae_model_path: Path,
    cls_model_path: Path,
    inputs: list[Path],
    output_dir: Path,
) -> int:
    """Run the MAE + classifier anomaly detection pipeline on a list of .npy inputs.

    For each input slice, this also saves a visualization (original, reconstruction, diff)
    to the output directory, similar to the ONNX reference script.
    """
    if not inputs:
        print("No input .npy files provided", file=sys.stderr)
        return 1

    H, W = INFER_HEIGHT, INFER_WIDTH

    mae_opt = neat.ModelOptions()
    mae_opt.media_type = "video/x-raw"
    # mae_opt.format = "RGB"
    mae_opt.input_max_width = W
    mae_opt.input_max_height = H
    mae_opt.input_max_depth = 3
    mae_opt.media_type ="application/vnd.simaai.tensor"
    mae_opt.format = ""

    cls_opt = neat.ModelOptions()
    cls_opt.media_type = "video/x-raw"
    # cls_opt.format = "RGB"
    cls_opt.input_max_width = W
    cls_opt.input_max_height = H
    cls_opt.input_max_depth = 3
    cls_opt.media_type ="application/vnd.simaai.tensor"
    cls_opt.format = ""

    print(f"[LOAD] MAE model: {mae_model_path}")
    print(f"[LOAD] Classifier model: {cls_model_path}")
    mae_model = neat.Model(str(mae_model_path), mae_opt)
    cls_model = neat.Model(str(cls_model_path), cls_opt)

    output_dir.mkdir(parents=True, exist_ok=True)

    num_patches = (H // PATCH_SIZE) * (W // PATCH_SIZE)
    mask = get_deterministic_mask(num_patches, seed=42)
    grid_h, grid_w = H // PATCH_SIZE, W // PATCH_SIZE
    mask_px = mask.reshape(grid_h, grid_w)
    mask_px = np.repeat(np.repeat(mask_px, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)  # (H, W)
    mask_px = mask_px[:, :, np.newaxis]  # (H, W, 1)

    for npy_path in inputs:
        print(f"\n--- Processing: {npy_path} ---")

        try:
            rgb = load_npy_as_rgb(npy_path)
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}", file=sys.stderr)
            continue

        # # NEAT video models require UInt8 input; quantize at the model boundary.
        # rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
        # # Debug: inspect the quantized input range and a few sample pixels.
        # print(
        #     "rgb_uint8 stats:",
        #     "shape", rgb_uint8.shape,
        #     "dtype", rgb_uint8.dtype,
        #     "min", int(rgb_uint8.min()),
        #     "max", int(rgb_uint8.max()),
        # )
        # print("rgb_uint8[0,0,:]:", rgb_uint8[0, 0, :])
        # Create MAE input tensor directly from the resized float image.
        img_tensor = neat.Tensor.from_numpy(rgb, copy=True)
        # Debug: inspect the numeric range and tensor metadata going into the MAE model.
        print(
            "img_np_for_mae stats:",
            "shape", rgb.shape,
            "dtype", rgb.dtype,
            "min", float(rgb.min()),
            "max", float(rgb.max()),
        )
        print(
            "img_tensor meta:",
            "shape", tuple(int(x) for x in img_tensor.shape),
            "dtype", img_tensor.dtype,
        )

        mae_out = mae_model.run(img_tensor, timeout_ms=5000)
        recon_raw = tensor_to_numpy_dense(mae_out.tensor).astype(np.float32)
        print(
            f"MAE recon raw shape: {recon_raw.shape}  dtype: {mae_out.tensor.dtype}"
        )

        # Work in float space without forcing values into [0, 1].
        img_np = rgb.astype(np.float32)

        recon_np = recon_raw.squeeze()
        if recon_np.ndim == 3:
            if recon_np.shape[0] < recon_np.shape[1]:
                recon_np = recon_np.transpose(1, 2, 0)
        elif recon_np.ndim == 2:
            recon_np = recon_np[:, :, np.newaxis]

        im_paste = img_np * (1 - mask_px) + recon_np * mask_px

        diff = np.abs(img_np - im_paste)
        diff_tensor = neat.Tensor.from_numpy(diff, copy=True)
        # Debug: inspect the numeric range and tensor metadata going into the classifier model.
        print(
            "diff_for_cls stats:",
            "shape", diff.shape,
            "dtype", diff.dtype,
            "min", float(diff.min()),
            "max", float(diff.max()),
        )
        print(
            "diff_tensor meta:",
            "shape", tuple(int(x) for x in diff_tensor.shape),
            "dtype", diff_tensor.dtype,
        )

        cls_out = cls_model.run(diff_tensor, timeout_ms=5000)
        logits = tensor_to_numpy_dense(cls_out.tensor).flatten().astype(np.float32)

        probs = stable_softmax(logits)
        if probs.size < 2:
            print(
                f"Unexpected classifier output size {probs.size} (expected at least 2)",
                file=sys.stderr,
            )
            continue

        normal_prob = float(probs[0])
        anomalous_prob = float(probs[1])
        label = "normal" if normal_prob > anomalous_prob else "anomalous"

        # Compute SSIM on a single channel to mirror ONNX reference script behavior.
        gray_orig = img_np[:, :, 0]
        gray_recon = im_paste[:, :, 0]
        data_range = float(gray_orig.max() - gray_orig.min()) or 1.0
        ssim_val = ssim(gray_orig, gray_recon, data_range=data_range)

        print("===== Inference Result =====")
        print(f"  Input       : {npy_path}")
        print(f"  SSIM        : {ssim_val:.4f}")
        print(f"  Normal prob : {normal_prob:.4f}")
        print(f"  Anomaly prob: {anomalous_prob:.4f}")
        print(f"  Prediction  : {label}")
        print("============================")

        # Visualization: original, reconstruction, and diff side by side.
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(gray_orig, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(gray_recon, cmap="gray")
        axes[1].set_title("MAE Reconstruction")
        axes[2].imshow(np.abs(gray_orig - gray_recon), cmap="gray")
        axes[2].set_title(f"Diff  |  {label} ({anomalous_prob:.2f})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()

        out_path = output_dir / f"{npy_path.stem}_viz.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[SAVE] Visualization written to: {out_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAE-based medical anomaly detection with NEAT"
    )
    parser.add_argument(
        "mae_model",
        type=str,
        help="Path to MAE reconstruction compiled model package",
    )
    parser.add_argument(
        "cls_model",
        type=str,
        help="Path to anomaly classifier compiled model package",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more .npy files containing 2D slices",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_mae_ad",
        help="Directory to save visualization images (default: outputs_mae_ad)",
    )
    args = parser.parse_args()

    mae_model_path = Path(args.mae_model)
    cls_model_path = Path(args.cls_model)
    input_paths = [Path(p) for p in args.inputs]
    output_dir = Path(args.output_dir)

    return run_pipeline(mae_model_path, cls_model_path, input_paths, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
