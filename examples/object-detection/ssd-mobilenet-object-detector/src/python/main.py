"""SSD-MobileNetV2 (TF COCO) folder object detection with host-side box decode.

The compiled pack emits the 12 grouped convolution heads of the TF Object Detection
API model ``ssd_mobilenet_v2_coco_2018_03_29`` (6 localization + 6 classification,
feature maps 19/10/5/3/2/1 -> 1917 priors). This example reassembles those heads in
anchor-major layout, applies the ``FasterRcnnBoxCoder`` anchor decode with the
recovered ``ssd_anchor_generator`` priors, SIGMOID class scoring, per-class NMS, and
draws the detections onto the original image.

Model input is normalized to ``[-1, 1]`` via ``(pixel / 127.5) - 1`` and the resize is
a direct STRETCH to 300x300 (the model was trained with ``fixed_shape_resizer``), so
boxes back-project onto the source with a simple per-axis scale.
"""

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

DEFAULT_MODEL_PATH = "assets/models/ssd_mobilenet_v2_heads_mpk.tar.gz"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
COMMON_DIR = Path(__file__).resolve().parents[1] / "common"

MODEL_SIZE = 300
NUM_CLASSES = 91  # index 0 = background, 1..90 = COCO ids.
# FasterRcnnBoxCoder scales: (y, x, h, w).
SCALE_Y, SCALE_X, SCALE_H, SCALE_W = 10.0, 10.0, 5.0, 5.0

# Per-level SSD priors, largest -> smallest feature map. Each (w, h) is a prior box
# size in normalized image coordinates; anchor centers are (cell + 0.5) / feature_size.
# This reproduces the ssd_anchor_generator anchors of ssd_mobilenet_v2_coco_2018_03_29 --
# the same per-level prior table baked into the on-device SSD decode kernel -- so no
# anchor asset needs to be shipped. Anchors are (ycenter, xcenter, h, w).
PRIOR_LEVELS: list[tuple[int, list[tuple[float, float]]]] = [
    (19, [(0.1, 0.099999994), (0.2828427, 0.14142135), (0.14142135, 0.28284273)]),
    (10, [(0.34999999, 0.35000002), (0.49497473, 0.2474874), (0.24748738, 0.49497464),
          (0.60621786, 0.20207258), (0.20206252, 0.60624808), (0.41833004, 0.41833001)]),
    (5, [(0.5, 0.49999997), (0.70710671, 0.35355341), (0.35355338, 0.70710677),
         (0.86602533, 0.28867519), (0.28866071, 0.86606878), (0.57008761, 0.57008779)]),
    (3, [(0.64999992, 0.65000004), (0.91923869, 0.45961937), (0.45961937, 0.91923869),
         (1.1258328, 0.37527764), (0.37525886, 1.1258892), (0.72111022, 0.72111011)]),
    (2, [(0.79999995, 0.79999995), (1.131371, 0.56568551), (0.56568539, 1.131371),
         (1.3856405, 0.46188015), (0.46185714, 1.3857099), (0.87177998, 0.87177974)]),
    (1, [(0.95000011, 0.94999999), (1.343503, 0.67175144), (0.67175162, 1.343503),
         (1.6454482, 0.54848289), (0.54845542, 1.6455302), (0.97467947, 0.97467953)]),
]

# Feature levels as (feature_map_size, anchors_per_cell), derived from PRIOR_LEVELS.
LEVELS = [(feat, len(priors)) for feat, priors in PRIOR_LEVELS]

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


def _feature_offsets() -> dict[int, tuple[int, int]]:
    """Map feature-map size -> (anchors_per_cell, flat prior offset)."""
    offsets: dict[int, tuple[int, int]] = {}
    off = 0
    for feat, anchors_per_cell in LEVELS:
        offsets[feat] = (anchors_per_cell, off)
        off += feat * feat * anchors_per_cell
    return offsets


FEATURE_OFFSETS = _feature_offsets()
NUM_PRIORS = sum(feat * feat * a for feat, a in LEVELS)  # 1917


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[ssd-debug] {msg}", flush=True)


def _resolve_asset(configured: str, default_name: str) -> Path:
    """Resolve a labels/anchors asset path, falling back to the packaged copy.

    ``config.yaml`` stores repo-relative paths (resolved from the apps root when run
    per the README). When the working directory differs (e.g. under the test harness),
    fall back to the copy that ships next to this example under ``src/common``.
    """
    candidate = Path(configured)
    if candidate.is_file():
        return candidate
    fallback = COMMON_DIR / default_name
    if fallback.is_file():
        return fallback
    return candidate


@dataclass(frozen=True)
class PreprocMeta:
    orig_h: int
    orig_w: int


def sigmoid(x: "np.ndarray") -> "np.ndarray":
    return 1.0 / (1.0 + np.exp(-x))


def generate_anchors() -> "np.ndarray":
    """Build the 1917x4 SSD priors (ycenter, xcenter, h, w) from PRIOR_LEVELS."""
    anchors = np.empty((NUM_PRIORS, 4), dtype=np.float32)
    idx = 0
    for feat, priors in PRIOR_LEVELS:
        for y in range(feat):
            cy = (y + 0.5) / feat
            for x in range(feat):
                cx = (x + 0.5) / feat
                for w, h in priors:
                    anchors[idx] = (cy, cx, h, w)
                    idx += 1
    return anchors


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"labels file does not exist: {path}")
    labels = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels


def class_name(labels: list[str], class_idx: int) -> str:
    if 0 <= class_idx < len(labels):
        name = labels[class_idx]
        if name and name != "N/A":
            return name
    return f"class_{class_idx}"


def class_color(class_idx: int) -> tuple[int, int, int]:
    return BOX_COLORS[abs(class_idx) % len(BOX_COLORS)]


def tensor_to_numpy(t: "pyneat.Tensor") -> "np.ndarray":
    return np.asarray(t.to_numpy(copy=True))


def iter_tensors(sample: "pyneat.Sample"):
    if sample.kind == pyneat.SampleKind.Tensor and sample.tensor is not None:
        yield sample.tensor
    elif sample.kind == pyneat.SampleKind.TensorSet:
        yield from sample.tensors
    for field in sample.fields:
        yield from iter_tensors(field)


def collect_tensors(sample: "pyneat.Sample") -> list["pyneat.Tensor"]:
    return list(iter_tensors(sample))


def tensor_from_hwc_f32(array: "np.ndarray") -> "pyneat.Tensor":
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HWC float32 image tensor, got {array.shape}")
    arr = np.ascontiguousarray(array, dtype=np.float32)
    return pyneat.Tensor.from_numpy(
        arr,
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def preprocess_image_bgr(image_bgr: "np.ndarray") -> tuple["np.ndarray", PreprocMeta]:
    """STRETCH resize to 300x300, BGR->RGB, normalize to [-1, 1]."""
    orig_h, orig_w = image_bgr.shape[:2]
    resized = cv2.resize(
        image_bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = np.ascontiguousarray(rgb / 127.5 - 1.0, dtype=np.float32)
    return normalized, PreprocMeta(orig_h=orig_h, orig_w=orig_w)


def head_to_hwc(array: "np.ndarray") -> "np.ndarray":
    """Return a single head as HWC float32 regardless of the reported layout.

    Heads have square feature maps (H == W), so the channel axis is the one that
    differs from the two equal spatial axes.
    """
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected a rank-3 head after squeezing batch, got {arr.shape}")
    d0, d1, d2 = arr.shape
    if d0 == d1 and d0 != d2:
        return arr  # already HWC
    if d1 == d2 and d1 != d0:
        return np.ascontiguousarray(np.transpose(arr, (1, 2, 0)))  # CHW -> HWC
    raise ValueError(f"cannot infer head layout from shape {arr.shape}")


def reassemble_heads(
    arrays: list["np.ndarray"],
) -> tuple["np.ndarray", "np.ndarray"]:
    """Reassemble the 12 conv heads into [N,4] encodings and [N,91] class logits.

    Localization channel layout is anchor-major ``a*4 + coord`` (coord = ty,tx,th,tw);
    classification is ``a*91 + class``. Priors are ordered largest->smallest feature
    map, then row-major cell, then anchor -- matching the anchor file.
    """
    encodings = np.zeros((NUM_PRIORS, 4), dtype=np.float32)
    logits = np.zeros((NUM_PRIORS, NUM_CLASSES), dtype=np.float32)
    seen_loc: set[int] = set()
    seen_conf: set[int] = set()

    for raw in arrays:
        hwc = head_to_hwc(raw)
        feat, feat_w, channels = hwc.shape
        if feat != feat_w or feat not in FEATURE_OFFSETS:
            raise ValueError(f"unexpected head feature map {hwc.shape}")
        anchors_per_cell, off = FEATURE_OFFSETS[feat]
        count = feat * feat * anchors_per_cell

        if channels % NUM_CLASSES == 0:  # classification head
            if channels // NUM_CLASSES != anchors_per_cell:
                raise ValueError(f"conf head channel/anchor mismatch: {hwc.shape}")
            block = hwc.reshape(feat, feat, anchors_per_cell, NUM_CLASSES).reshape(-1, NUM_CLASSES)
            logits[off : off + count] = block
            seen_conf.add(feat)
        elif channels % 4 == 0:  # localization head
            if channels // 4 != anchors_per_cell:
                raise ValueError(f"loc head channel/anchor mismatch: {hwc.shape}")
            block = hwc.reshape(feat, feat, anchors_per_cell, 4).reshape(-1, 4)
            encodings[off : off + count] = block
            seen_loc.add(feat)
        else:
            raise ValueError(f"unrecognized head channel depth {channels}")

    expected = {feat for feat, _ in LEVELS}
    if seen_loc != expected or seen_conf != expected:
        raise ValueError(
            f"missing heads: loc={sorted(seen_loc)} conf={sorted(seen_conf)} "
            f"expected {sorted(expected)}"
        )
    return encodings, logits


def decode_boxes(encodings: "np.ndarray", anchors: "np.ndarray") -> "np.ndarray":
    """encodings [N,4] (ty,tx,th,tw) -> normalized corners [N,4] (ymin,xmin,ymax,xmax)."""
    ty, tx, th, tw = encodings.T
    ay, ax, ah, aw = anchors.T
    yc = ty / SCALE_Y * ah + ay
    xc = tx / SCALE_X * aw + ax
    h = np.exp(th / SCALE_H) * ah
    w = np.exp(tw / SCALE_W) * aw
    return np.stack([yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2], axis=1)


def nms(boxes: "np.ndarray", scores: "np.ndarray", iou_thr: float) -> list[int]:
    """Single-class NMS on corner boxes [M,4] (ymin,xmin,ymax,xmax)."""
    if len(boxes) == 0:
        return []
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0.0, y2 - y1) * np.maximum(0.0, x2 - x1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, yy2 - yy1) * np.maximum(0.0, xx2 - xx1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def process_ssd_output(
    encodings: "np.ndarray",
    logits: "np.ndarray",
    anchors: "np.ndarray",
    meta: PreprocMeta,
    *,
    score_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    """Full SSD decode -> detections in original-image pixel coordinates."""
    boxes = decode_boxes(encodings, anchors)  # normalized corners
    scores = sigmoid(logits)[:, 1:]  # drop background -> [N,90]

    detections: list[dict[str, Any]] = []
    for cls in range(scores.shape[1]):
        cls_scores = scores[:, cls]
        mask = cls_scores >= score_threshold
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        keep = nms(boxes[idx], cls_scores[idx], nms_iou)
        for k in keep:
            j = idx[k]
            ymin, xmin, ymax, xmax = boxes[j]
            x1 = float(np.clip(xmin * meta.orig_w, 0.0, meta.orig_w))
            y1 = float(np.clip(ymin * meta.orig_h, 0.0, meta.orig_h))
            x2 = float(np.clip(xmax * meta.orig_w, 0.0, meta.orig_w))
            y2 = float(np.clip(ymax * meta.orig_h, 0.0, meta.orig_h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                {
                    "box": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "score": float(cls_scores[j]),
                    "class_id": cls + 1,  # foreground index -> COCO id (1..90)
                }
            )

    detections.sort(key=lambda d: d["score"], reverse=True)
    if max_detections > 0:
        detections = detections[:max_detections]
    return detections


def visualize_detections(
    image_bgr: "np.ndarray",
    detections: list[dict[str, Any]],
    labels: list[str],
) -> "np.ndarray":
    image_copy = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"].astype(np.int32)
        color = class_color(det["class_id"])
        text = f"{class_name(labels, det['class_id'])} {det['score']:.2f}"
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image_copy,
            text,
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image_copy


def build_graph_runner(model: "pyneat.Model") -> "pyneat.Run":
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())

    dummy = tensor_from_hwc_f32(np.zeros((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.float32))
    return graph.build([dummy])


def load_tensor_model(model_path: Path) -> "pyneat.Model":
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = MODEL_SIZE
    opt.preprocess.input_max_height = MODEL_SIZE
    opt.preprocess.input_max_depth = 3
    return pyneat.Model(str(model_path), opt)


def run_model_inference(
    runner: "pyneat.Run", preprocessed: "np.ndarray", timeout_ms: int
) -> list["np.ndarray"]:
    tensor = tensor_from_hwc_f32(preprocessed)
    if not runner.push([tensor]):
        raise RuntimeError("Run.push() failed")
    sample = runner.pull(timeout_ms=timeout_ms)
    if sample is None:
        raise RuntimeError("Run.pull() returned no sample")
    return [tensor_to_numpy(t) for t in collect_tensors(sample)]


def prepare_input(image_path: Path) -> tuple["np.ndarray", "np.ndarray", PreprocMeta]:
    _log(f"Reading image: {image_path}")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    preprocessed, meta = preprocess_image_bgr(bgr)
    return bgr, preprocessed, meta


def format_profile_stats(name: str, values: list[float]) -> str:
    arr = np.array(values, dtype=np.float64)
    fps = float(len(arr)) / arr.sum() if arr.sum() > 0 else 0.0
    return (
        f"  {name}: mean={arr.mean():.8f}s, min={arr.min():.8f}s, "
        f"max={arr.max():.8f}s, FPS={fps:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SSD-MobileNetV2 folder object detection example")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML configuration")
    args = parser.parse_args()

    global cv2, np, pyneat
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
    import pyneat  # noqa: F401

    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    model_cfg = raw.get("model", {})
    io_cfg = raw.get("io", {})
    decode_cfg = raw.get("decode", {})
    runtime_cfg = raw.get("runtime", {})

    model_path = Path(model_cfg.get("path", DEFAULT_MODEL_PATH))
    labels_path = _resolve_asset(model_cfg.get("labels", ""), "coco_labels.txt")
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/ssd-mobilenet-object-detector"))
    score_threshold = float(decode_cfg.get("score_threshold", 0.30))
    nms_iou = float(decode_cfg.get("nms_iou", 0.60))
    max_detections = int(decode_cfg.get("max_detections", 100))
    profile = bool(runtime_cfg.get("profile", False))
    num_runs = int(runtime_cfg.get("num_runs", 1))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 20000))

    global VERBOSE
    VERBOSE = bool(runtime_cfg.get("verbose", False))

    if not (0.0 <= score_threshold <= 1.0):
        print("decode.score_threshold must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    try:
        labels = load_labels(labels_path)
    except Exception as exc:
        print(f"Error loading assets: {exc}", file=sys.stderr)
        return 2
    anchors = generate_anchors()

    image_paths = sorted(p for p in input_dir.iterdir() if p.is_file() and is_image(p))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3

    try:
        model = load_tensor_model(model_path)
        runner = build_graph_runner(model)
    except Exception as exc:
        logger.debug("Model/graph build failure", exc_info=exc)
        print(f"Error loading model: {exc}", file=sys.stderr)
        return 3

    if profile:
        try:
            image_path = image_paths[0]
            _, preprocessed, meta = prepare_input(image_path)
            tensor = tensor_from_hwc_f32(preprocessed)
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
                arrays = [tensor_to_numpy(t) for t in collect_tensors(sample)]
                encodings, logits = reassemble_heads(arrays)
                detections = process_ssd_output(
                    encodings, logits, anchors, meta,
                    score_threshold=score_threshold, nms_iou=nms_iou,
                    max_detections=max_detections,
                )
                t2 = time.perf_counter()
                graph_times.append(t1 - t0)
                post_times.append(t2 - t1)
                last_detections = detections

            if not graph_times:
                print("Profiling aborted: no successful runs", file=sys.stderr)
                return 4

            print(f"Profiling over {len(graph_times)} runs (image='{image_path}'):")
            print(format_profile_stats("Graph run (push+pull)", graph_times))
            print(format_profile_stats("Postprocessing (decode+NMS)", post_times))
            print(f"Last run detections: {len(last_detections)}")
            for i, det in enumerate(last_detections[:20]):
                box = det["box"]
                print(
                    f"  [{i}] class={class_name(labels, det['class_id'])}({det['class_id']}) "
                    f"score={det['score']:.4f} box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
                )
            return 0
        except Exception as exc:
            logger.debug("Profiling failure", exc_info=exc)
            print(f"Error during profiling: {exc}", file=sys.stderr)
            return 4
        finally:
            runner.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        for image_path in image_paths:
            try:
                orig_bgr, preprocessed, meta = prepare_input(image_path)
                arrays = run_model_inference(runner, preprocessed, timeout_ms)
            except Exception as exc:
                logger.debug("Inference failure", exc_info=exc)
                print(f"Error during inference for {image_path}: {exc}", file=sys.stderr)
                return 3

            if not arrays:
                print(f"No tensors found in model output for {image_path}", file=sys.stderr)
                return 4

            try:
                encodings, logits = reassemble_heads(arrays)
                detections = process_ssd_output(
                    encodings, logits, anchors, meta,
                    score_threshold=score_threshold, nms_iou=nms_iou,
                    max_detections=max_detections,
                )
            except Exception as exc:
                logger.debug("Decode failure", exc_info=exc)
                print(f"Error during decode for {image_path}: {exc}", file=sys.stderr)
                return 4

            output_path = output_dir / f"{image_path.stem}_ssd.png"
            out_img = visualize_detections(orig_bgr, detections, labels)
            cv2.imwrite(str(output_path), out_img)
            processed += 1
            print(
                f"[{processed}/{len(image_paths)}] {image_path.name}: "
                f"{len(detections)} detections -> {output_path.name}"
            )
    finally:
        runner.close()

    print(f"Done: {processed} images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
