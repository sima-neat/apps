"""RetinaFace face detection example using a compiled NEAT model.

This script is intentionally minimal. It:
  - Loads the compiled RetinaFace model from a fixed path
  - Runs inference on a single input image
  - Prints basic information about the output tensors

You can later extend this with proper RetinaFace post-processing
to decode bounding boxes and landmarks and visualize them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat


def _log(msg: str) -> None:
    """Lightweight debug logger."""
    print(f"[retinaface-debug] {msg}", flush=True)


DEFAULT_MODEL_PATH = "/mnt/Bitbucket/apps/assets/models/mobilenet0.25_QAT_Final_mod_4_mpk.tar.gz"
# RetinaFace compiled model expects 1280x720 BGR input with specific mean subtraction
INFER_WIDTH = 1280
INFER_HEIGHT = 720


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    for field in sample.fields:
        yield from iter_tensors(field)


def pad_image_bgr(
    image_bgr: np.ndarray,
    orig_h: int,
    orig_w: int,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Pad image to target aspect ratio using black borders, preserving content."""
    aspect_ratio = orig_w / float(orig_h)
    target_ratio = target_w / float(target_h)

    if aspect_ratio > target_ratio:
        # Image is wider than target, pad height
        new_h = int(orig_w / target_ratio)
        pad_top = (new_h - orig_h) // 2
        pad_bottom = new_h - orig_h - pad_top
        padded = cv2.copyMakeBorder(
            image_bgr,
            pad_top,
            pad_bottom,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    else:
        # Image is taller than target, pad width
        new_w = int(orig_h * target_ratio)
        pad_left = (new_w - orig_w) // 2
        pad_right = new_w - orig_w - pad_left
        padded = cv2.copyMakeBorder(
            image_bgr,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
    return padded


def run_retinaface_inference(
    model_path: Path,
    image_path: Path,
) -> pyneat.Sample:
    _log(f"Starting inference. model_path={model_path}, image_path={image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    _log("Reading input image with OpenCV")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    orig_h, orig_w = bgr.shape[:2]

    # Custom preprocessing to match RetinaFace pipeline:
    # 1) BGR mean subtraction [104, 117, 123]
    # 2) Pad to maintain aspect ratio for 1280x720
    # 3) Resize to model input dimensions (1280x720)
    _log("Applying BGR mean subtraction")
    img = bgr.astype(np.float32) - np.array([104.0, 117.0, 123.0], dtype=np.float32)

    _log("Padding image to target aspect ratio before resize")
    img = pad_image_bgr(img, orig_h, orig_w, INFER_WIDTH, INFER_HEIGHT)

    _log("Resizing padded image to model input size (1280x720)")
    img = cv2.resize(img, (INFER_WIDTH, INFER_HEIGHT), interpolation=cv2.INTER_LINEAR)

    # Convert back to uint8 if the compiled model expects uint8 inputs;
    # if your model is truly float-input, you can adjust this later together
    # with ModelOptions.preproc instead.
    resized = np.ascontiguousarray(img, dtype=np.uint8)

    _log("Configuring pyneat.ModelOptions")
    opt = pyneat.ModelOptions()
    opt.media_type = "video/x-raw"
    opt.format = "BGR"
    opt.input_max_width = INFER_WIDTH
    opt.input_max_height = INFER_HEIGHT
    opt.input_max_depth = 3

    _log("Creating pyneat.Model")
    model = pyneat.Model(str(model_path), opt)

    _log("Creating input tensor and running model directly (no Session pipeline)")
    t_in = pyneat.Tensor.from_numpy(
        resized,
        copy=True,
        image_format=pyneat.PixelFormat.BGR,
    )

    # Use the simple Model.run API, like the classification example,
    # to avoid building a GStreamer pipeline and any box-decode elements.
    out = model.run(t_in, timeout_ms=5000)
    if out is None:
        raise RuntimeError("Model.run() returned None")

    _log(f"Inference complete. Original image size: {orig_w}x{orig_h}")
    return out


def main() -> int:
    _log("Parsing command-line arguments")
    parser = argparse.ArgumentParser(description="RetinaFace face detection example")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to RetinaFace compiled model package (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    _log(f"Using model_path={model_path}")
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2

    try:
        _log("Invoking run_retinaface_inference()")
        sample = run_retinaface_inference(model_path, Path(args.image))
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        return 3

    _log("Collecting tensors from sample")
    tensors = list(iter_tensors(sample))
    if not tensors:
        print("No tensors found in model output", file=sys.stderr)
        return 4

    print(f"Model produced {len(tensors)} tensor(s):")
    for i, t in enumerate(tensors):
        arr = tensor_to_numpy(t)
        print(f"  [{i}] shape={arr.shape}, dtype={arr.dtype}")

    print("Post-processing and visualization of RetinaFace outputs can be added here later.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


