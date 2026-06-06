from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
import time
import yaml

INPUT_W = 128
INPUT_H = 256
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"


def find_first_tensor(value):
    """Find the first tensor in beta TensorList or Sample-style outputs."""
    if value is None:
        return None
    if isinstance(value, pyneat.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            t = find_first_tensor(item)
            if t is not None:
                return t
        return None
    if getattr(value, "kind", None) == pyneat.SampleKind.Tensor and value.tensor is not None:
        return value.tensor
    for tensor in getattr(value, "tensors", []) or []:
        return tensor
    for field in getattr(value, "fields", []) or []:
        t = find_first_tensor(field)
        if t is not None:
            return t
    return None

def make_rgb_tensor(image: np.ndarray) -> pyneat.Tensor:
    return pyneat.Tensor.from_numpy(
        image,
        copy=True,
        image_format=pyneat.PixelFormat.RGB,
        memory=pyneat.TensorMemory.EV74,
    )


def warmup_model(runner: pyneat.ModelRunner, timeout_ms: int) -> None:
    """Run one dummy inference to initialize the hardware pipeline before timing."""
    dummy = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
    runner.run([make_rgb_tensor(dummy)], timeout_ms=timeout_ms)
    print("Model warmed up.")


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

def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}

def preprocess_image(bgr_image: np.ndarray) -> np.ndarray:
    """Convert BGR image to normalized RGB, resize and prepare for inference."""
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std  = np.array([0.229, 0.224, 0.225]) * 255.0
    rgb_image = (rgb_image - mean) / std
    image = cv2.resize(rgb_image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(image, dtype=np.uint8)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a.flatten(), b.flatten()))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(euclidean(a.flatten(), b.flatten()))

def run_inference(
    runner: pyneat.ModelRunner, image_path: Path, timeout_ms: int
) -> tuple[np.ndarray, float]:
    """Run inference on a single image. Returns (embedding, inference_time_s)."""
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = preprocess_image(bgr_image)

    t0 = time.perf_counter()
    out = runner.run([make_rgb_tensor(image)], timeout_ms=timeout_ms)
    infer_time_s = time.perf_counter() - t0

    out_tensor = find_first_tensor(out)
    if out_tensor is None:
        raise ValueError(f"No tensor output for: {image_path.name}")

    embedding = tensor_to_numpy(out_tensor).astype(np.float32)
    return embedding, infer_time_s

def save_comparison_image(
    path1: Path,
    path2: Path,
    sim: float,
    decision: str,
    threshold: float,
    metric: str,
    output_path: Path,
):
    """Concatenate two input images side by side and overlay decision + similarity."""
    img1 = cv2.imread(str(path1))
    img2 = cv2.imread(str(path2))

    target_h = 400
    def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h))

    img1 = resize_to_height(img1, target_h)
    img2 = resize_to_height(img2, target_h)

    divider = np.ones((target_h, 4, 3), dtype=np.uint8) * 200
    canvas = np.concatenate([img1, divider, img2], axis=1)

    canvas_w = canvas.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def fit_font_scale(text: str, max_width: int, thickness: int, start_scale: float = 3.0) -> float:
        scale = start_scale
        while scale > 0.1:
            (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
            if w <= max_width:
                return scale
            scale -= 0.05
        return scale

    max_text_width = int(canvas_w * 0.85)

    decision_thickness = 3
    decision_scale = fit_font_scale(decision, max_text_width, decision_thickness)
    (_, decision_h), decision_baseline = cv2.getTextSize(decision, font, decision_scale, decision_thickness)

    metric_label = "Cosine similarity" if metric == "cosine" else "Euclidean distance"
    label = f"{metric_label}: {sim:.4f}   |   Threshold: {threshold:.2f}"
    details_thickness = 1
    details_scale = fit_font_scale(label, max_text_width, details_thickness)
    (_, details_h), details_baseline = cv2.getTextSize(label, font, details_scale, details_thickness)

    padding = 12
    bar_h = decision_h + details_h + decision_baseline + details_baseline + padding * 3
    bar = np.zeros((bar_h, canvas_w, 3), dtype=np.uint8)
    canvas = np.concatenate([canvas, bar], axis=0)

    color = (0, 200, 0) if decision == "SAME" else (0, 0, 220)

    decision_size = cv2.getTextSize(decision, font, decision_scale, decision_thickness)[0]
    decision_x = (canvas_w - decision_size[0]) // 2
    decision_y = target_h + padding + decision_h
    cv2.putText(canvas, decision, (decision_x, decision_y), font, decision_scale, color, decision_thickness, cv2.LINE_AA)

    details_size = cv2.getTextSize(label, font, details_scale, details_thickness)[0]
    details_x = (canvas_w - details_size[0]) // 2
    details_y = decision_y + decision_baseline + padding + details_h
    cv2.putText(canvas, label, (details_x, details_y), font, details_scale, (200, 200, 200), details_thickness, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)
    print(f"Comparison image saved to: {output_path}")


def save_result_json(
    output_path: Path,
    image_a: Path,
    image_b: Path,
    metric: str,
    score: float,
    threshold: float,
    decision: str,
):
    payload = {
        "image_a": str(image_a),
        "image_b": str(image_b),
        "metric": metric,
        "score": float(score),
        "threshold": float(threshold),
        "decision": decision,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Result json saved to: {output_path}")


def print_profile(
    infer_time_1_s: float,
    infer_time_2_s: float,
    total_time_s: float,
):
    print("\n[PROFILE] Timing report")
    print("[PROFILE]   note: one warmup run was performed before timing")

    print(f"[PROFILE]   inference image_a : {infer_time_1_s * 1000:.1f} ms")
    print(f"[PROFILE]   inference image_b : {infer_time_2_s * 1000:.1f} ms")
    print(f"[PROFILE]   total inference   : {(infer_time_1_s + infer_time_2_s) * 1000:.1f} ms")
    print(f"[PROFILE]   end-to-end        : {total_time_s * 1000:.1f} ms")




def main() -> int:
    parser = argparse.ArgumentParser(description="ReID inference with embedding comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    args = parser.parse_args()

    global np, cv2, pyneat, cosine, euclidean
    import numpy as np
    import cv2
    import pyneat
    from scipy.spatial.distance import cosine, euclidean

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_cfg = raw.get("model", {})
    io_cfg = raw.get("io", {})
    output_cfg = raw.get("output", {})
    comparison_cfg = raw.get("comparison", {})
    runtime_cfg = raw.get("runtime", {})

    model_path = Path(model_cfg.get("path", "assets/models/reid_mpk.tar.gz"))
    image1 = Path(io_cfg.get("image1", "assets/test_images/person_a.jpg"))
    image2 = Path(io_cfg.get("image2", "assets/test_images/person_b.jpg"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/person-re-identification"))
    output_type = output_cfg.get("type", "both")
    metric = comparison_cfg.get("metric", "cosine")
    threshold = float(comparison_cfg.get("threshold", 0.65 if metric == "cosine" else 25.0))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 5000))
    profile = bool(runtime_cfg.get("profile", False))

    if metric not in {"cosine", "euclidean"}:
        print(f"Invalid metric: {metric}", file=sys.stderr)
        return 2
    if output_type not in {"image", "json", "both"}:
        print(f"Invalid output type: {output_type}", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2
    for p in (image1, image2):
        if not p.is_file() or not is_image(p):
            print(f"Not a valid image file: {p}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    runner = None
    try:

        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Image
        opt.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
        opt.preprocess.input_max_width = INPUT_W
        opt.preprocess.input_max_height = INPUT_H
        opt.preprocess.input_max_depth = 3
        model = pyneat.Model(str(model_path), opt)
        runner = model.build(
            [make_rgb_tensor(np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8))],
            route_options=pyneat.ModelRouteOptions(),
        )
        warmup_model(runner, timeout_ms)

        t_start = time.perf_counter()
        print(f"Processing: {image1.name}")
        emb1, infer_time_1 = run_inference(runner, image1, timeout_ms)

        print(f"Processing: {image2.name}")
        emb2, infer_time_2 = run_inference(runner, image2, timeout_ms)

        if metric == "cosine":
            score = cosine_similarity(emb1, emb2)
            decision = "SAME" if score >= threshold else "DIFFERENT"
            print(f"\nCosine similarity : {score:.6f}")
            print(f"Threshold         : {threshold:.2f}")
            print(f"Decision          : {decision}")
        else:
            score = euclidean_distance(emb1, emb2)
            decision = "SAME" if score <= threshold else "DIFFERENT"
            print(f"\nEuclidean distance: {score:.6f}")
            print(f"Threshold         : {threshold:.2f}")
            print(f"Decision          : {decision}")

        total_time = time.perf_counter() - t_start

        if output_type in {"image", "both"}:
            save_comparison_image(
                image1, image2, score, decision, threshold, metric,
                output_dir / "comparison.jpg",
            )

        if output_type in {"json", "both"}:
            save_result_json(
                output_dir / "result.json",
                image1,
                image2,
                metric,
                score,
                threshold,
                decision,
            )

        if profile:
            print_profile(infer_time_1, infer_time_2, total_time)

        runner.close()
        return 0

    except Exception as e:
        if runner is not None:
            try:
                runner.close()
            except Exception:
                pass
        print(f"Error: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
