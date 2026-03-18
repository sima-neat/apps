"""Simple person re-identification pipeline using pyneat."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyneat


INFER_W = 128   # model input width
INFER_H = 256   # model input height


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


def find_first_tensor(sample: pyneat.Sample) -> pyneat.Tensor | None:
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor
    for field in sample.fields:
        t = find_first_tensor(field)
        if t is not None:
            return t
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D feature vectors."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as a contiguous uint8 RGB array resized to INFER_H x INFER_W."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    bgr = cv2.resize(bgr, (INFER_W, INFER_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def extract_embedding(model: pyneat.Model, rgb: np.ndarray) -> np.ndarray:
    """Run model inference and return a 1-D float32 feature embedding."""
    t_in = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
    out = model.run(t_in, timeout_ms=5000)
    t_out = find_first_tensor(out)
    if t_out is None:
        raise RuntimeError("model returned no output tensor")
    embedding = tensor_to_numpy(t_out).flatten().astype(np.float32)
    return embedding


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple re-identification pipeline: extract embeddings and compute similarity"
    )
    parser.add_argument("model", type=str, help="Path to compiled re-identification model package")
    parser.add_argument("input_dir", type=str, help="Input directory containing query and gallery images")
    parser.add_argument("output_dir", type=str, help="Output directory for similarity results")
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
        opt.media_type = "video/x-raw"
        opt.format = "RGB"
        opt.input_max_width = INFER_W
        opt.input_max_height = INFER_H
        opt.input_max_depth = 3
        model = pyneat.Model(args.model, opt)

        embeddings: list[tuple[str, np.ndarray]] = []
        for img_path in images:
            try:
                rgb = load_rgb(img_path)
            except Exception as e:
                print(f"Skipping unreadable image {img_path.name}: {e}", file=sys.stderr)
                continue
            emb = extract_embedding(model, rgb)
            embeddings.append((img_path.name, emb))
            print(f"Extracted embedding for {img_path.name}: shape={emb.shape}")

        if len(embeddings) < 2:
            print("Need at least 2 images to compute similarity", file=sys.stderr)
            return 3

        results: list[str] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                name_a, emb_a = embeddings[i]
                name_b, emb_b = embeddings[j]
                sim = cosine_similarity(emb_a, emb_b)
                line = f"{name_a} vs {name_b}: similarity={sim:.4f}"
                results.append(line)
                print(line)

        out_path = output_dir / "similarity.txt"
        out_path.write_text("\n".join(results) + "\n", encoding="utf-8")
        print(f"Results written to {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
