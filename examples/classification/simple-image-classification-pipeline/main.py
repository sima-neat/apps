"""Minimal Model usage with a ResNet50 MPK."""

import argparse
import sys
import urllib.request
from array import array
from pathlib import Path

import numpy as np
import pyneat

GOLDFISH_URL = (
    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
    "n01443537_goldfish.JPEG"
)
GOLDFISH_ID = 1  # ILSVRC2012 0-based index for "goldfish"
INFER_WIDTH = 224
INFER_HEIGHT = 224


def download_image(url: str, dest: Path) -> Path:
    """Download an image if it does not already exist."""
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_rgb_resized(path: str, width: int, height: int) -> np.ndarray:
    """Load an image, resize to (height, width), return RGB uint8 HWC array."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((width, height))
    # NumPy may expose PIL-backed arrays as read-only; pyneat's DLPack path requires writable input.
    return np.array(img, dtype=np.uint8, copy=True)


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
    parser.add_argument("--model", type=str, required=True, help="Path to ResNet50 MPK tarball")
    parser.add_argument("--image", type=str, default="", help="Path to input image")
    parser.add_argument("--min-prob", type=float, default=0.2, help="Minimum probability threshold")
    args = parser.parse_args()

    model_path = args.model

    # Resolve image path
    image_path = args.image
    if not image_path:
        dest = Path("/tmp/goldfish.jpeg")
        try:
            download_image(GOLDFISH_URL, dest)
            image_path = str(dest)
        except Exception as e:
            print(f"Failed to download goldfish image: {e}", file=sys.stderr)
            return 3

    print(f"Using model: {model_path}")
    print(f"Using image: {image_path}")

    # Load and resize image
    try:
        rgb = load_rgb_resized(image_path, INFER_WIDTH, INFER_HEIGHT)
    except Exception as e:
        print(f"Failed to load image: {e}", file=sys.stderr)
        return 4

    # Configure model with ImageNet normalization
    opt = pyneat.ModelOptions()
    opt.media_type = "video/x-raw"
    opt.format = "RGB"
    opt.input_max_width = INFER_WIDTH
    opt.input_max_height = INFER_HEIGHT
    opt.input_max_depth = 3
    opt.preproc.normalize = True
    # nanobind in this build accepts typed float arrays here (std::optional<std::array<float, 3>>).
    opt.preproc.channel_mean = array("f", (0.485, 0.456, 0.406))
    opt.preproc.channel_stddev = array("f", (0.229, 0.224, 0.225))

    model = pyneat.Model(model_path, opt)

    # Run inference
    t = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
    try:
        out = model.run(t, timeout_ms=2000)
        if out.tensor is None:
            print("Model run returned empty output", file=sys.stderr)
            return 6
        scores = tensor_to_numpy_dense(out.tensor).flatten().astype(np.float32)
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
    if top1 != GOLDFISH_ID:
        print(f"FAIL: expected top1={GOLDFISH_ID} (goldfish), got {top1}", file=sys.stderr)
        return 1
    if probs[top1] < args.min_prob:
        print(f"FAIL: top1 prob {probs[top1]:.4f} < {args.min_prob}", file=sys.stderr)
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
