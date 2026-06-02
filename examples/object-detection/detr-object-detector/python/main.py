"""DETR single-image object detection example using manual tensor preprocessing."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VERBOSE = False
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "assets/models/detr_resnet50_modified_class_embed_bbox_embed_mpk.tar.gz"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
FRAME_WIDTH = 1333
FRAME_HEIGHT = 800
PERSON_CLASS_ID = 1
DETR_COCO_LABELS = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
BOX_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[detr-debug] {msg}", flush=True)


@dataclass(frozen=True)
class PreprocMeta:
    orig_h: int
    orig_w: int
    resized_h: int
    resized_w: int
    pad_top: int
    pad_left: int
    scale_x: float
    scale_y: float


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: pyneat.Sample):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    for field in sample.fields:
        yield from iter_tensors(field)


def collect_tensors(sample: pyneat.Sample) -> list[pyneat.Tensor]:
    return list(iter_tensors(sample))


def tensor_from_hwc_f32(array: np.ndarray) -> pyneat.Tensor:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC float32 image tensor, got {array.shape}")
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return pyneat.Tensor.from_numpy(
        arr,
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def tensor_numpy_outputs(tensors: list[pyneat.Tensor]) -> list[np.ndarray]:
    return [tensor_to_numpy(t) for t in tensors]


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(DETR_COCO_LABELS):
        label = DETR_COCO_LABELS[class_id]
        if label != "N/A":
            return label
    return f"class_{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_id) % len(BOX_COLORS)]


def preprocess_image_bgr(image_bgr: np.ndarray) -> tuple[np.ndarray, PreprocMeta]:
    """Resize with aspect ratio preservation, center-pad, convert to RGB, normalize."""
    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(FRAME_WIDTH / float(orig_w), FRAME_HEIGHT / float(orig_h))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    pad_left = (FRAME_WIDTH - resized_w) // 2
    pad_top = (FRAME_HEIGHT - resized_h) // 2

    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = np.ascontiguousarray((rgb - mean) / std, dtype=np.float32)

    meta = PreprocMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        resized_h=resized_h,
        resized_w=resized_w,
        pad_top=pad_top,
        pad_left=pad_left,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return normalized, meta


def extract_logits_and_boxes(tensors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = None
    boxes = None

    for arr in tensors:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim < 2:
            continue
        cols = int(arr.shape[-1])
        flat = arr.reshape(-1, cols)
        if cols == 4:
            boxes = flat
        elif cols > 4:
            logits = flat

    if logits is None or boxes is None:
        raise ValueError("Expected DETR logits and box tensors in model output")
    if logits.shape[0] != boxes.shape[0]:
        raise ValueError("DETR logits and boxes row counts do not match")
    if logits.shape[1] < 2:
        raise ValueError("DETR logits tensor must include foreground classes and background")
    return logits, boxes


def process_detr_output(
    boxes: np.ndarray,
    logits: np.ndarray,
    meta: PreprocMeta,
    *,
    confidence_threshold: float,
    detect_only_person: bool,
) -> list[dict[str, Any]]:
    """Decode DETR logits and normalized boxes into original-image detections."""
    boxes = sigmoid(boxes)
    prob = softmax(logits, axis=-1)
    scores = prob[..., :-1].max(axis=-1)
    class_ids = prob[..., :-1].argmax(axis=-1)

    keep = scores > confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if detect_only_person:
        person_mask = class_ids == PERSON_CLASS_ID
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]

    if len(boxes) == 0:
        return []

    x_c, y_c, w, h = boxes.T
    xyxy = np.stack(
        [
            (x_c - 0.5 * w) * FRAME_WIDTH,
            (y_c - 0.5 * h) * FRAME_HEIGHT,
            (x_c + 0.5 * w) * FRAME_WIDTH,
            (y_c + 0.5 * h) * FRAME_HEIGHT,
        ],
        axis=1,
    )

    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - meta.pad_left) / meta.scale_x
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - meta.pad_top) / meta.scale_y
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(meta.orig_w))
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(meta.orig_h))

    detections: list[dict[str, Any]] = []
    for box, score, class_id in zip(xyxy, scores, class_ids):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(
            {
                "box": box.astype(np.float32, copy=False),
                "score": float(score),
                "class_id": int(class_id),
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


def visualize_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    max_draw: int,
) -> np.ndarray:
    """Visualize DETR detections on the original image."""
    image_copy = image_bgr.copy()
    limit = len(detections) if max_draw <= 0 else min(len(detections), max_draw)

    for det in detections[:limit]:
        box = det["box"]
        score = det["score"]
        class_id = det["class_id"]
        x1, y1, x2, y2 = box.astype(np.int32)
        color = class_color(class_id)
        label = class_name(class_id)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_copy,
            f"{label} {score:.2f}",
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return image_copy


def prepare_input(image_path: Path) -> tuple[np.ndarray, np.ndarray, PreprocMeta]:
    _log(f"Reading image: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    preprocessed, meta = preprocess_image_bgr(bgr)
    return bgr, preprocessed, meta


def load_tensor_model(model_path: Path) -> pyneat.Model:
    _log(f"Building tensor-input model for {FRAME_WIDTH}x{FRAME_HEIGHT}")

    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = FRAME_WIDTH
    opt.preprocess.input_max_height = FRAME_HEIGHT
    opt.preprocess.input_max_depth = 3

    return pyneat.Model(str(model_path), opt)


def build_graph_runner(model: pyneat.Model) -> pyneat.Run:
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())

    dummy = tensor_from_hwc_f32(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32))
    return graph.build([dummy], pyneat.RunMode.Async)


def run_model_inference(model: pyneat.Model, preprocessed: np.ndarray) -> list[np.ndarray]:
    runner = build_graph_runner(model)
    tensor = tensor_from_hwc_f32(preprocessed)
    try:
        if not runner.push([tensor]):
            raise RuntimeError("Run.push() failed")
        sample = runner.pull(timeout_ms=5000)
        if sample is None:
            raise RuntimeError("Run.pull() returned no sample")
        return tensor_numpy_outputs(collect_tensors(sample))
    finally:
        runner.close()


def build_model_runner(model: pyneat.Model, preprocessed: np.ndarray) -> tuple[pyneat.Run, pyneat.Tensor]:
    tensor = tensor_from_hwc_f32(preprocessed)
    runner = build_graph_runner(model)
    return runner, tensor


def format_profile_stats(name: str, values: list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    runs = float(len(arr))
    fps = runs / arr.sum()
    return (
        f"  {name}: "
        f"mean={arr.mean():.8f}s, "
        f"min={arr.min():.8f}s, "
        f"max={arr.max():.8f}s, "
        f"FPS={fps:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DETR single-image object detection example")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    args = parser.parse_args()

    global cv2, np, pyneat
    import cv2
    import numpy as np
    import pyneat

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_cfg = raw.get("model", {})
    io_cfg = raw.get("io", {})
    decode_cfg = raw.get("decode", {})
    runtime_cfg = raw.get("runtime", {})

    image_path = Path(io_cfg.get("image", "assets/test_images/input.jpg"))
    model_path = Path(model_cfg.get("path", DEFAULT_MODEL_PATH))
    output_path = Path(io_cfg.get("output", "sandbox/detr-object-detector/output.png"))
    confidence_threshold = float(decode_cfg.get("confidence_threshold", 0.5))
    max_draw = int(decode_cfg.get("max_draw", 50))
    person_only = bool(decode_cfg.get("person_only", False))
    profile = bool(runtime_cfg.get("profile", False))
    num_runs = int(runtime_cfg.get("num_runs", 100))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 5000))

    global VERBOSE
    VERBOSE = bool(runtime_cfg.get("verbose", False))

    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2

    try:
        orig_bgr, preprocessed, meta = prepare_input(image_path)
        model = load_tensor_model(model_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        logger.debug("Inference failure", exc_info=exc)
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 3

    if profile:
        try:
            runner, tensor = build_model_runner(model, preprocessed)
            graph_times: list[float] = []
            post_times: list[float] = []
            last_detections: list[dict[str, Any]] = []

            for _ in range(max(1, num_runs)):
                t0 = time.perf_counter()
                if not runner.push([tensor]):
                    print("Profiling failed: runner.push returned False", file=sys.stderr)
                    break
                sample = runner.pull(timeout_ms=timeout_ms)
                t1 = time.perf_counter()
                if sample is None:
                    print("Profiling failed: runner.pull returned no sample", file=sys.stderr)
                    break

                arrays = tensor_numpy_outputs(collect_tensors(sample))
                logits, boxes = extract_logits_and_boxes(arrays)
                t2 = time.perf_counter()
                detections = process_detr_output(
                    boxes,
                    logits,
                    meta,
                    confidence_threshold=confidence_threshold,
                    detect_only_person=person_only,
                )
                t3 = time.perf_counter()

                graph_times.append(t1 - t0)
                post_times.append(t3 - t2)
                last_detections = detections

            runner.close()

            if not graph_times:
                print("Profiling aborted: no successful runs", file=sys.stderr)
                return 4

            total_times = [a + b for a, b in zip(graph_times, post_times)]
            print(f"Profiling over {len(graph_times)} runs (image='{image_path}'):")
            print(format_profile_stats("Graph run (push+pull)", graph_times))
            print(format_profile_stats("Postprocessing (decode+NMS)", post_times))
            print(format_profile_stats("Overall (graph + post)", total_times))
            print(f"Last run detections: {len(last_detections)}")
            for i, det in enumerate(last_detections[:20]):
                box = det["box"]
                print(
                    f"  [{i}] class={class_name(det['class_id'])}({det['class_id']}) score={det['score']:.4f} "
                    f"box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
                )
            return 0
        except Exception as exc:
            logger.debug("Profiling failure", exc_info=exc)
            print(f"Error during profiling: {exc}", file=sys.stderr)
            return 4

    try:
        arrays = run_model_inference(model, preprocessed)
    except Exception as exc:
        logger.debug("Inference failure", exc_info=exc)
        print(f"Error during inference: {exc}", file=sys.stderr)
        return 3

    if not arrays:
        print("No tensors found in model output", file=sys.stderr)
        return 4

    print(f"Model produced {len(arrays)} tensor(s):")
    for i, arr in enumerate(arrays):
        print(f"  [{i}] shape={arr.shape}, dtype={arr.dtype}")

    try:
        logits, boxes = extract_logits_and_boxes(arrays)
        detections = process_detr_output(
            boxes,
            logits,
            meta,
            confidence_threshold=confidence_threshold,
            detect_only_person=person_only,
        )
    except Exception as exc:
        logger.debug("Decode failure", exc_info=exc)
        print(f"Error during decode: {exc}", file=sys.stderr)
        return 4

    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections[:20]):
        box = det["box"]
        print(
            f"  [{i}] class={class_name(det['class_id'])}({det['class_id']}) score={det['score']:.4f} "
            f"box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
        )

    if str(output_path):
        out_img = visualize_detections(orig_bgr, detections, max_draw=max_draw)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), out_img)
        print(f"Wrote annotated image: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
