#!/usr/bin/env python3
"""MAE-based medical anomaly detection pipeline using two NEAT models.

This example:
- loads 2D medical slices from .npy files,
- normalizes and resizes them as a single-channel tensor (H, W, 1),
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


def load_npy_normalized(path: Path, dataset: str = "brats") -> np.ndarray:
    """Load a (potentially) multi-channel .npy slice, resize and normalize.

    Matches the preprocessing in infer_single_onnx.py:
    - Takes first channel if multi-channel
    - Resizes to (224, 224) for brats
    - Normalizes: (image - mean) / std
    - Returns a single-channel tensor shaped (H, W, 1)
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
    
    # Take first channel if multi-channel (matches ONNX: image[:, :, 0])
    if npy_image.ndim == 3:
        npy_image = npy_image[:, :, 0]

    # (H, W) -> (H, W, 1)
    npy_image = np.expand_dims(npy_image, axis=2)

    # Resize to model resolution (matches ONNX resize)
    if dataset == "brats":
        target_size = (224, 224)
    else:  # luna16_unnorm
        target_size = (64, 64)
    
    npy_resized = resize(
        npy_image,
        target_size,
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    # Normalize (matches ONNX: image = (image - mean) / std)
    if dataset == "brats":
        mean, std = 0.0, 1.0
    else:  # luna16_unnorm
        mean, std = 0.0, 100.0
    
    npy_normalized = (npy_resized - mean) / std

    return npy_normalized  # (H, W, 1) float32


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
    dataset: str = "brats",
) -> int:
    """Run the MAE + classifier anomaly detection pipeline on a list of .npy inputs.

    For each input slice, this also saves a visualization (original, reconstruction, diff)
    to the output directory, similar to the ONNX reference script.
    """
    if not inputs:
        print("No input .npy files provided", file=sys.stderr)
        return 1

    H, W = INFER_HEIGHT, INFER_WIDTH

    # Model options for single-channel input (matching ONNX preprocessing)
    mae_opt = neat.ModelOptions()
    mae_opt.media_type = "application/vnd.simaai.tensor"
    mae_opt.format = ""
    mae_opt.input_max_width = W
    mae_opt.input_max_height = H
    mae_opt.input_max_depth = 1  # single-channel MAE input

    cls_opt = neat.ModelOptions()
    cls_opt.media_type = "application/vnd.simaai.tensor"
    cls_opt.format = ""
    cls_opt.input_max_width = W
    cls_opt.input_max_height = H
    cls_opt.input_max_depth = 3  # 3-channel classifier input (duplicated from 1-channel diff)

    print(f"[LOAD] MAE model: {mae_model_path}")
    print(f"[LOAD] Classifier model: {cls_model_path}")
    mae_model = neat.Model(str(mae_model_path), mae_opt)
    cls_model = neat.Model(str(cls_model_path), cls_opt)

    # Build MAE session: Input -> QuantTess -> MLA -> DetessDequant -> Output
    mae_sess = neat.Session()
    mae_sess.add(neat.nodes.input(mae_model.input_appsrc_options(True)))
    mae_sess.add(neat.nodes.quant_tess(neat.QuantTessOptions(mae_model)))
    mae_sess.add(neat.groups.mla(mae_model))
    mae_sess.add(neat.nodes.detess_dequant(neat.DetessDequantOptions(mae_model)))
    mae_sess.add(neat.nodes.output())
    print(f"[BUILD] MAE Pipeline:\n{mae_sess.describe_backend()}")

    # Build Classifier session: Input -> QuantTess -> MLA -> DetessDequant -> Output
    cls_sess = neat.Session()
    cls_sess.add(neat.nodes.input(cls_model.input_appsrc_options(True)))
    cls_sess.add(neat.nodes.quant_tess(neat.QuantTessOptions(cls_model)))
    cls_sess.add(neat.groups.mla(cls_model))
    cls_sess.add(neat.nodes.detess_dequant(neat.DetessDequantOptions(cls_model)))
    cls_sess.add(neat.nodes.output())
    print(f"[BUILD] Classifier Pipeline:\n{cls_sess.describe_backend()}")

    # Build runs with dummy tensors (single channel for MAE, 3 channels for classifier)
    dummy_mae = np.zeros((H, W, 1), dtype=np.float32)
    mae_run = mae_sess.build(dummy_mae)

    dummy_cls = np.zeros((H, W, 3), dtype=np.float32)
    cls_run = cls_sess.build(dummy_cls)

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
            img_1ch = load_npy_normalized(npy_path, dataset)
        except Exception as e:
            print(f"Failed to load {npy_path}: {e}", file=sys.stderr)
            continue

        # Debug: Compare preprocessing with ONNX
        print("\n[DEBUG] === Preprocessing Comparison ===")
        print(f"[NEAT] img_1ch shape: {img_1ch.shape}, dtype: {img_1ch.dtype}")
        print(f"[NEAT] img_1ch stats: min={img_1ch.min():.6f}, max={img_1ch.max():.6f}, mean={img_1ch.mean():.6f}, std={img_1ch.std():.6f}")
        print(f"[NEAT] img_1ch sample[0:3, 0:3, 0]:\n{img_1ch[0:3, 0:3, 0]}")
        print(f"[ONNX] Expected: (1, 224, 224, 1) normalized with mean=0.0, std=1.0")
        print(f"[ONNX] Expected sample values should match img_1ch above")

        # Run MAE through session pipeline (single-channel input)
        if not mae_run.push(img_1ch):
            print(f"MAE push failed for {npy_path}", file=sys.stderr)
            continue
        mae_sample = mae_run.pull(timeout_ms=5000)
        if mae_sample is None:
            print(f"MAE pull failed for {npy_path}", file=sys.stderr)
            continue
        if mae_sample.tensor is None:
            print(f"MAE output tensor is None for {npy_path}", file=sys.stderr)
            continue
        recon_raw = tensor_to_numpy_dense(mae_sample.tensor).astype(np.float32)
        print("\n[DEBUG] === MAE Reconstruction Comparison ===")
        print(f"[NEAT] recon_raw shape: {recon_raw.shape}, dtype: {recon_raw.dtype}")
        print(f"[NEAT] recon_raw stats: min={recon_raw.min():.6f}, max={recon_raw.max():.6f}, mean={recon_raw.mean():.6f}, std={recon_raw.std():.6f}")
        if recon_raw.size > 0:
            print(f"[NEAT] recon_raw sample (first few values): {recon_raw.flatten()[:9]}")

        # Process reconstruction to match ONNX pipeline
        # NEAT model outputs may be (B, C, H, W) or (H, W, C), convert to (H, W, 1)
        recon_np = recon_raw.squeeze()
        if recon_np.ndim == 4:  # (B, C, H, W) or (B, 1, H, W)
            recon_np = recon_np[0]  # Remove batch dim
            if recon_np.ndim == 3:  # (C, H, W) or (1, H, W)
                recon_np = recon_np.transpose(1, 2, 0)  # (H, W, C)
        elif recon_np.ndim == 3:
            if recon_np.shape[0] < recon_np.shape[1]:  # (C, H, W)
                recon_np = recon_np.transpose(1, 2, 0)  # (H, W, C)
        elif recon_np.ndim == 2:
            recon_np = recon_np[:, :, np.newaxis]  # (H, W, 1)
        
        # Ensure single channel (take first channel if multi-channel)
        if recon_np.ndim == 3 and recon_np.shape[2] > 1:
            recon_np = recon_np[:, :, 0:1]  # Take first channel, keep dims
        
        print(f"[NEAT] recon_np (after processing) shape: {recon_np.shape}")
        print(f"[NEAT] recon_np stats: min={recon_np.min():.6f}, max={recon_np.max():.6f}, mean={recon_np.mean():.6f}, std={recon_np.std():.6f}")
        print(f"[NEAT] recon_np sample[0:3, 0:3, 0]:\n{recon_np[0:3, 0:3, 0]}")
        print(f"[ONNX] Expected: (1, 224, 224, 1) reconstruction output")
        print(f"[ONNX] Expected sample values should match recon_np above")

        # Match ONNX mask application: im_paste = x * (1 - mask_px) + reconstruction * mask_px
        # Use single-channel normalized image for mask application (matching ONNX)
        img_np_1ch = img_1ch  # (H, W, 1) normalized single channel
        
        # Debug mask
        print(f"\n[DEBUG] === Mask Comparison ===")
        print(f"[NEAT] mask_px shape: {mask_px.shape}, dtype: {mask_px.dtype}")
        print(f"[NEAT] mask_px stats: min={mask_px.min():.6f}, max={mask_px.max():.6f}, mean={mask_px.mean():.6f}")
        print(f"[NEAT] mask_px sample[0:3, 0:3, 0]:\n{mask_px[0:3, 0:3, 0]}")
        print(f"[ONNX] Expected: (1, 1, 224, 224) mask in NCHW format")
        
        im_paste_1ch = img_np_1ch * (1 - mask_px) + recon_np * mask_px
        
        print(f"\n[DEBUG] === Mask Application Comparison ===")
        print(f"[NEAT] im_paste_1ch shape: {im_paste_1ch.shape}")
        print(f"[NEAT] im_paste_1ch stats: min={im_paste_1ch.min():.6f}, max={im_paste_1ch.max():.6f}, mean={im_paste_1ch.mean():.6f}, std={im_paste_1ch.std():.6f}")
        print(f"[NEAT] im_paste_1ch sample[0:3, 0:3, 0]:\n{im_paste_1ch[0:3, 0:3, 0]}")
        print(f"[ONNX] Expected: (1, 224, 224, 1) im_paste in NHWC format")
        print(f"[ONNX] Expected sample values should match im_paste_1ch above")

        # Compute diff matching ONNX: diff = abs(img_batch - recon)
        diff_1ch = np.abs(img_np_1ch - im_paste_1ch)
        
        print(f"\n[DEBUG] === Diff Computation Comparison ===")
        print(f"[NEAT] diff_1ch shape: {diff_1ch.shape}")
        print(f"[NEAT] diff_1ch stats: min={diff_1ch.min():.6f}, max={diff_1ch.max():.6f}, mean={diff_1ch.mean():.6f}, std={diff_1ch.std():.6f}")
        print(f"[NEAT] diff_1ch sample[0:3, 0:3, 0]:\n{diff_1ch[0:3, 0:3, 0]}")
        print(f"[ONNX] Expected: (1, 224, 224, 1) diff = abs(img_batch - recon)")
        print(f"[ONNX] Expected sample values should match diff_1ch above")
        
        # Duplicate diff to 3 channels for NEAT classifier model
        diff_3ch = np.stack([diff_1ch[:, :, 0], diff_1ch[:, :, 0], diff_1ch[:, :, 0]], axis=-1)
        print(f"[NEAT] diff_3ch (for classifier) shape: {diff_3ch.shape}, stats: min={diff_3ch.min():.6f}, max={diff_3ch.max():.6f}")

        # Run Classifier through session pipeline (3-channel input for NEAT model)
        if not cls_run.push(diff_3ch):
            print(f"Classifier push failed for {npy_path}", file=sys.stderr)
            continue
        cls_sample = cls_run.pull(timeout_ms=5000)
        if cls_sample is None:
            print(f"Classifier pull failed for {npy_path}", file=sys.stderr)
            continue
        if cls_sample.tensor is None:
            print(f"Classifier output tensor is None for {npy_path}", file=sys.stderr)
            continue
        logits = tensor_to_numpy_dense(cls_sample.tensor).flatten().astype(np.float32)
        
        print(f"\n[DEBUG] === Classifier Output Comparison ===")
        print(f"[NEAT] logits shape: {logits.shape}, dtype: {logits.dtype}")
        print(f"[NEAT] logits values: {logits}")
        print(f"[ONNX] Expected: (2,) logits array")
        print(f"[ONNX] Expected logits values should match above")

        probs = stable_softmax(logits)
        
        print(f"[NEAT] probs values: {probs}")
        print(f"[ONNX] Expected: (2,) probs array from softmax(logits)")
        print(f"[ONNX] Expected probs values should match above")
        if probs.size < 2:
            print(
                f"Unexpected classifier output size {probs.size} (expected at least 2)",
                file=sys.stderr,
            )
            continue

        normal_prob = float(probs[0])
        anomalous_prob = float(probs[1])
        label = "normal" if normal_prob > anomalous_prob else "anomalous"

        # Compute SSIM matching ONNX: ssim(img_batch[0, :, :, 0], recon[0, :, :, 0], data_range=1.0)
        gray_orig = img_np_1ch[:, :, 0]
        gray_recon = im_paste_1ch[:, :, 0]
        data_range = 1.0  # Match ONNX data_range=1.0 for normalized brats data
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

    # Clean up runs
    try:
        mae_run.close()
        cls_run.close()
    except Exception:
        pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MAE-based medical anomaly detection with NEAT"
    )
    default_mae_model = (
        Path(__file__).resolve().parents[4]
        / "assets"
        / "models"
        / "mae_brats_deterministic_grid_masking_simplified_mpk.tar.gz"
    )
    parser.add_argument(
        "mae_model",
        type=str,
        nargs="?",
        default=str(default_mae_model),
        help="Path to MAE reconstruction compiled model package "
        "(default: apps/assets/models/mae_brats_deterministic_grid_masking_simplified_mpk.tar.gz)",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="brats",
        choices=["brats", "luna16_unnorm"],
        help="Dataset type (affects resize & normalization, default: brats)",
    )
    args = parser.parse_args()

    mae_model_path = Path(args.mae_model)
    cls_model_path = Path(args.cls_model)
    input_paths = [Path(p) for p in args.inputs]
    output_dir = Path(args.output_dir)

    return run_pipeline(mae_model_path, cls_model_path, input_paths, output_dir, args.dataset)


if __name__ == "__main__":
    raise SystemExit(main())
