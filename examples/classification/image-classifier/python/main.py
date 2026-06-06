"""Minimal Model usage with a ResNet50 compiled model package."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import yaml


def download_image(url: str, dest: Path) -> Path:
    """Download an image if it does not already exist."""
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_rgb_resized(path: str, width: int, height: int) -> np.ndarray:
    """Load an image with OpenCV and return an RGB uint8 HWC array."""
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.array(rgb, dtype=np.uint8, copy=True)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def tensor_to_numpy_dense(tensor: pyneat.Tensor) -> np.ndarray:
    """Convert a dense pyneat tensor to a NumPy array without DLPack."""
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
        raise TypeError(f"Unsupported tensor dtype for NumPy conversion: {tensor.dtype}")

    shape = tuple(int(x) for x in tensor.shape)
    arr = np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def print_topk(scores: np.ndarray, k: int = 5) -> int:
    """Print top-k predictions with softmax probabilities. Return top-1 index."""
    probs = softmax(scores)
    top_indices = np.argsort(scores)[::-1][:k]

    print(f"top1 index={top_indices[0]} score={scores[top_indices[0]]:.4f} "
          f"prob={probs[top_indices[0]]:.4f}")
    top_str = " ".join(f"{i}:{probs[i]:.4f}" for i in top_indices)
    print(f"top5: {top_str}")

    return int(top_indices[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="ResNet50 classification example")
    default_config = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML configuration")
    args = parser.parse_args()

    global np, pyneat
    import numpy as np
    import pyneat

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_path = raw.get("model", {}).get("path", "assets/models/resnet_50_mpk.tar.gz")
    io_cfg = raw.get("io", {})
    runtime = raw.get("runtime", {})
    validation = raw.get("validation", {})
    image_path = io_cfg.get("image") or ""
    fallback_image_url = io_cfg.get(
        "fallback_image_url",
        "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
        "n01443537_goldfish.JPEG",
    )
    input_width = int(runtime.get("input_width", 224))
    input_height = int(runtime.get("input_height", 224))
    timeout_ms = int(runtime.get("timeout_ms", 2000))
    expected_class_id = int(validation.get("expected_class_id", 1))
    min_probability = float(validation.get("min_probability", 0.2))

    # Resolve image path
    if not image_path:
        dest = Path("/tmp/goldfish.jpeg")
        try:
            download_image(fallback_image_url, dest)
            image_path = str(dest)
        except Exception as e:
            print(f"Failed to download goldfish image: {e}", file=sys.stderr)
            return 3

    print(f"Using model: {model_path}")
    print(f"Using image: {image_path}")

    # Load and resize image
    try:
        rgb = load_rgb_resized(image_path, input_width, input_height)
    except Exception as e:
        print(f"Failed to load image: {e}", file=sys.stderr)
        return 4

    # Configure model with ImageNet normalization
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Image
    opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = input_width
    opt.preprocess.input_max_height = input_height
    opt.preprocess.input_max_depth = 3
    opt.preprocess.preset = pyneat.NormalizePreset.ImageNet

    model = pyneat.Model(model_path, opt)

    # Run inference
    t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
    try:
        outputs = model.run([t], timeout_ms=timeout_ms)
        if not outputs:
            print("Model run returned empty output", file=sys.stderr)
            return 6
        scores = tensor_to_numpy_dense(outputs[0]).flatten().astype(np.float32)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 6

    if scores.size < 1000:
        print(f"Expected at least 1000 scores, got {scores.size}", file=sys.stderr)
        return 6

    scores = scores[:1000]

    # Print results and validate
    top1 = print_topk(scores, k=5)

    probs = softmax(scores)
    if top1 != expected_class_id:
        print(f"FAIL: expected top1={expected_class_id} (goldfish), got {top1}", file=sys.stderr)
        return 1
    if probs[top1] < min_probability:
        print(f"FAIL: top1 prob {probs[top1]:.4f} < {min_probability}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
