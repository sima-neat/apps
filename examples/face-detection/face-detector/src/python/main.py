"""RetinaFace face detection example using a compiled NEAT model.

This script performs an end-to-end folder-based RetinaFace pipeline:
  - Preprocesses the input image for the model (mean subtraction + pad + resize)
  - Runs inference through a NEAT Graph
  - Decodes RetinaFace outputs into face boxes, confidence scores, and landmarks
  - Applies confidence filtering and NMS
  - Optionally writes an annotated output image
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import yaml


VERBOSE = False
logger = logging.getLogger(__name__)


def _log(msg: str) -> None:
    if VERBOSE:
        print(f"[retinaface-debug] {msg}", flush=True)


DEFAULT_MODEL_PATH = "assets/models/retinaface_mobilenet25_mod_0_mpk.tar.gz"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
# RetinaFaceSpy postprocessing assumes 640x640 input space (80/40/20 feature maps).
INFER_WIDTH = 640
INFER_HEIGHT = 640

# Imported from `apps/backbone_cfg.py` (cfg_mnet), trimmed to needed fields.
CFG_MNET = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


class PreprocMeta(NamedTuple):
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    pad_top: int
    pad_left: int


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
    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(array, dtype=np.float32),
        copy=True,
        layout=pyneat.TensorLayout.HWC,
        memory=pyneat.TensorMemory.EV74,
    )


def tensor_numpy_outputs(tensors: list[pyneat.Tensor]) -> list[np.ndarray]:
    return [tensor_to_numpy(t) for t in tensors]


def pad_image_bgr(
    image_bgr: np.ndarray,
    orig_h: int,
    orig_w: int,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, PreprocMeta]:
    """Pad image to target aspect ratio using black borders, preserving content."""
    aspect_ratio = orig_w / float(orig_h)
    target_ratio = target_w / float(target_h)

    if aspect_ratio > target_ratio:
        # Image is wider than target, pad height
        new_h = int(orig_w / target_ratio)
        pad_top = (new_h - orig_h) // 2
        pad_bottom = new_h - orig_h - pad_top
        pad_left = 0
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
        pad_top = 0
        padded = cv2.copyMakeBorder(
            image_bgr,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    meta = PreprocMeta(
        orig_h=orig_h,
        orig_w=orig_w,
        padded_h=padded.shape[0],
        padded_w=padded.shape[1],
        pad_top=pad_top,
        pad_left=pad_left,
    )
    return padded, meta


# -----------------------------------------------------------------------------
# RetinaFace postprocessing (adapted from `apps/RetinaFaceSpy.py`, numpy-only)
# -----------------------------------------------------------------------------
def prior_boxes(image_height: int, image_width: int) -> np.ndarray:
    anchors: list[float] = []
    feature_maps = [
        [np.ceil(image_height / step), np.ceil(image_width / step)] for step in CFG_MNET["steps"]
    ]

    for k, f in enumerate(feature_maps):
        min_sizes = CFG_MNET["min_sizes"][k]
        for i in range(int(f[0])):
            for j in range(int(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_width
                    s_ky = min_size / image_height
                    dense_cx = [x * CFG_MNET["steps"][k] / image_width for x in [j + 0.5]]
                    dense_cy = [y * CFG_MNET["steps"][k] / image_height for y in [i + 0.5]]
                    for cy in dense_cy:
                        for cx in dense_cx:
                            anchors += [cx, cy, s_kx, s_ky]

    out = np.array(anchors, dtype=np.float32).reshape(-1, 4)
    if CFG_MNET["clip"]:
        out = np.clip(out, 0, 1)
    return out


def decode(loc: np.ndarray, priors: np.ndarray, variances: list[float]) -> np.ndarray:
    var0 = variances[0]
    var1 = variances[1]
    boxes = np.empty_like(priors)
    boxes[:, :2] = priors[:, :2] + loc[:, :2] * var0 * priors[:, 2:]
    boxes[:, 2:] = priors[:, 2:] * np.exp(loc[:, 2:] * var1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre: np.ndarray, priors: np.ndarray, variances: list[float]) -> np.ndarray:
    var0 = variances[0]
    priors_xy = priors[:, :2]
    priors_wh = priors[:, 2:]
    pre_reshaped = pre.reshape(-1, 5, 2)
    landms = priors_xy[:, None, :] + pre_reshaped * var0 * priors_wh[:, None, :]
    return landms.reshape(-1, 10)


def py_cpu_nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def process_landmark_outputs(landmark_0: np.ndarray, landmark_1: np.ndarray, landmark_2: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            landmark_0.reshape(1, -1, 10),
            landmark_1.reshape(1, -1, 10),
            landmark_2.reshape(1, -1, 10),
        ],
        axis=1,
    )


def process_bbox_outputs(bbox_0: np.ndarray, bbox_1: np.ndarray, bbox_2: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            bbox_0.reshape(1, -1, 4),
            bbox_1.reshape(1, -1, 4),
            bbox_2.reshape(1, -1, 4),
        ],
        axis=1,
    )


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def process_class_outputs(class_0: np.ndarray, class_1: np.ndarray, class_2: np.ndarray) -> np.ndarray:
    concatenated = np.concatenate(
        [
            class_0.reshape(1, -1, 2),
            class_1.reshape(1, -1, 2),
            class_2.reshape(1, -1, 2),
        ],
        axis=1,
    )
    return softmax(concatenated, axis=2)


def parse_retinaface_outputs(tensors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expect 9 tensors in this order (matches observed output + RetinaFaceSpy.model_outs):
      [landmark_2, landmark_1, landmark_0, bbox_2, bbox_1, bbox_0, class_2, class_1, class_0]
    Each arrives as NHWC (1, H, C, W) from the current pipeline; transpose to NCHW first.
    """
    if len(tensors) != 9:
        raise ValueError(f"Expected 9 output tensors, got {len(tensors)}")

    land2, land1, land0, box2, box1, box0, cls2, cls1, cls0 = tensors

    land0 = land0.transpose(0, 3, 1, 2)
    land1 = land1.transpose(0, 3, 1, 2)
    land2 = land2.transpose(0, 3, 1, 2)
    box0 = box0.transpose(0, 3, 1, 2)
    box1 = box1.transpose(0, 3, 1, 2)
    box2 = box2.transpose(0, 3, 1, 2)
    cls0 = cls0.transpose(0, 3, 1, 2)
    cls1 = cls1.transpose(0, 3, 1, 2)
    cls2 = cls2.transpose(0, 3, 1, 2)

    landmarks = process_landmark_outputs(land0, land1, land2)
    bboxes = process_bbox_outputs(box0, box1, box2)
    scores = process_class_outputs(cls0, cls1, cls2)
    return bboxes, scores, landmarks


def postprocess_retinaface(
    bboxes: np.ndarray,
    scores: np.ndarray,
    landmarks: np.ndarray,
    meta: PreprocMeta,
    *,
    confidence_threshold: float,
    nms_threshold: float,
    top_k: int,
    keep_top_k: int,
    with_landmarks: bool,
) -> list[dict[str, Any]]:
    priors = prior_boxes(INFER_HEIGHT, INFER_WIDTH)
    decoded_boxes = decode(bboxes.squeeze(0), priors, CFG_MNET["variance"])

    # scores: (1, N, 2) -> face prob
    face_scores = scores.squeeze(0)[:, 1]
    if with_landmarks:
        decoded_landms = decode_landm(landmarks.squeeze(0), priors, CFG_MNET["variance"])
    else:
        decoded_landms = None

    # Filter by confidence
    inds = np.where(face_scores > confidence_threshold)[0]
    if len(inds) == 0:
        return []

    decoded_boxes = decoded_boxes[inds]
    face_scores = face_scores[inds]
    if decoded_landms is not None:
        decoded_landms = decoded_landms[inds]

    # Sort by score
    order = face_scores.argsort()[::-1]
    if top_k > 0 and top_k < len(order):
        order = order[:top_k]
    decoded_boxes = decoded_boxes[order]
    face_scores = face_scores[order]
    if decoded_landms is not None:
        decoded_landms = decoded_landms[order]

    # NMS in input (normalized) coordinates
    dets = np.hstack((decoded_boxes, face_scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    if decoded_landms is not None:
        decoded_landms = decoded_landms[keep]

    if keep_top_k > 0 and keep_top_k < len(dets):
        dets = dets[:keep_top_k, :]
        if decoded_landms is not None:
            decoded_landms = decoded_landms[:keep_top_k, :]

    # Scale from model space -> padded image space -> original image space
    boxes = dets[:, :4].copy()
    boxes[:, 0] *= meta.padded_w
    boxes[:, 2] *= meta.padded_w
    boxes[:, 1] *= meta.padded_h
    boxes[:, 3] *= meta.padded_h
    boxes[:, 0] -= meta.pad_left
    boxes[:, 2] -= meta.pad_left
    boxes[:, 1] -= meta.pad_top
    boxes[:, 3] -= meta.pad_top

    output: list[dict[str, Any]] = []
    for i in range(boxes.shape[0]):
        item: dict[str, Any] = {
            "box": boxes[i],
            "score": float(dets[i, 4]),
        }
        if decoded_landms is not None:
            landm = decoded_landms[i].copy()
            for j in range(0, 10, 2):
                landm[j] = landm[j] * meta.padded_w - meta.pad_left
                landm[j + 1] = landm[j + 1] * meta.padded_h - meta.pad_top
            item["landmarks"] = landm
        output.append(item)
    return output


def draw_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for det in detections:
        box = det["box"]
        score = det["score"]
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{score:.3f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        landm = det.get("landmarks")
        if landm is not None:
            pts = landm.reshape(5, 2)
            for (x, y) in pts:
                cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
    return out


def build_retinaface_runner(model_path: Path) -> Any:
    _log("Configuring pyneat.ModelOptions for tensor input (FP32)")
    opt = pyneat.ModelOptions()
    opt.preprocess.kind = pyneat.InputKind.Tensor
    opt.preprocess.input_max_width = INFER_WIDTH
    opt.preprocess.input_max_height = INFER_HEIGHT
    opt.preprocess.input_max_depth = 3

    _log("Creating pyneat.Model")
    model = pyneat.Model(str(model_path), opt)

    _log("Building Graph pipeline: input -> quant_tess -> infer(MLA) -> detess_dequant -> output")
    graph = pyneat.Graph()
    graph.add(pyneat.nodes.input(model.input_appsrc_options(True)))
    graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(model)))
    graph.add(pyneat.groups.mla(model))
    graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(model)))
    graph.add(pyneat.nodes.output())
    _log(f"Graph backend description:\n{graph.describe_backend()}")

    _log("Building Graph run with dummy frame")
    dummy = tensor_from_hwc_f32(np.zeros((INFER_HEIGHT, INFER_WIDTH, 3), dtype=np.float32))
    return graph.build([dummy], pyneat.RunMode.Async)


def prepare_retinaface_frame(image_path: Path) -> tuple[pyneat.Tensor, np.ndarray, PreprocMeta]:
    _log(f"Preparing frame. image_path={image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    _log("Reading input image with OpenCV")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    orig_h, orig_w = bgr.shape[:2]

    _log("Applying BGR mean subtraction")
    img = bgr.astype(np.float32) - np.array([104.0, 117.0, 123.0], dtype=np.float32)

    _log("Padding image to target aspect ratio before resize")
    padded, meta = pad_image_bgr(img, orig_h, orig_w, INFER_WIDTH, INFER_HEIGHT)

    _log("Resizing padded image to model input size (640x640)")
    img = cv2.resize(padded, (INFER_WIDTH, INFER_HEIGHT), interpolation=cv2.INTER_LINEAR)
    resized = np.ascontiguousarray(img, dtype=np.float32)

    return tensor_from_hwc_f32(resized), bgr, meta


def run_retinaface_inference(
    run: Any,
    image_path: Path,
    timeout_ms: int,
) -> tuple[list[pyneat.Tensor], np.ndarray, PreprocMeta]:
    _log(f"Starting inference. image_path={image_path}")
    input_tensor, bgr, meta = prepare_retinaface_frame(image_path)

    _log("Pushing preprocessed frame into Graph")
    if not run.push([input_tensor]):
        raise RuntimeError("Failed to push frame into Graph pipeline")

    _log("Pulling output sample from Graph")
    sample = run.pull(timeout_ms=timeout_ms)
    if sample is None:
        raise RuntimeError("Graph.pull() returned None")

    _log(f"Inference complete. Original image size: {meta.orig_w}x{meta.orig_h}")

    return collect_tensors(sample), bgr, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="RetinaFace face detection folder example")
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

    model_path = Path(model_cfg.get("path", DEFAULT_MODEL_PATH))
    input_dir = Path(io_cfg.get("input_dir", "assets/test_images"))
    output_dir = Path(io_cfg.get("output_dir", "sandbox/face-detector"))
    confidence_threshold = float(decode_cfg.get("confidence_threshold", 0.4))
    nms_threshold = float(decode_cfg.get("nms_iou", 0.9))
    top_k = int(decode_cfg.get("top_k", 5000))
    keep_top_k = int(decode_cfg.get("keep_top_k", 750))
    max_draw = int(decode_cfg.get("max_draw", 50))
    landmarks = bool(decode_cfg.get("landmarks", True))
    profile = bool(runtime_cfg.get("profile", False))
    num_runs = int(runtime_cfg.get("num_runs", 100))
    timeout_ms = int(runtime_cfg.get("timeout_ms", 20000))

    global VERBOSE
    VERBOSE = bool(runtime_cfg.get("verbose", False))
    _log("Parsing command-line arguments")

    _log(f"Using model_path={model_path}")
    if not model_path.is_file():
        print(f"Model file does not exist: {model_path}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and is_image(path))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 3

    # Profiling mode: reuse a single graph run and frame, and profile graph vs postprocessing.
    if profile:
        image_path = image_paths[0]
        try:
            run = build_retinaface_runner(model_path)
            input_tensor, orig_bgr, meta = prepare_retinaface_frame(image_path)
        except Exception as e:
            print(f"Error during graph preparation: {e}", file=sys.stderr)
            return 3

        graph_times: list[float] = []
        post_times: list[float] = []
        total_runs = num_runs
        last_detections: list[dict[str, Any]] = []

        for i in range(total_runs):
            t0 = time.perf_counter()
            if not run.push([input_tensor]):
                print(f"Run {i}: failed to push frame into Graph pipeline", file=sys.stderr)
                break
            sample = run.pull(timeout_ms=timeout_ms)
            t1 = time.perf_counter()
            if sample is None:
                print(f"Run {i}: Graph.pull() returned None", file=sys.stderr)
                break

            tensors = collect_tensors(sample)
            np_outs = tensor_numpy_outputs(tensors)
            bboxes, scores, landmarks = parse_retinaface_outputs(np_outs)
            t2 = time.perf_counter()

            detections = postprocess_retinaface(
                bboxes,
                scores,
                landmarks,
                meta,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                keep_top_k=keep_top_k,
                with_landmarks=landmarks is not None,
            )
            t3 = time.perf_counter()

            graph_times.append(t1 - t0)
            post_times.append(t3 - t2)
            last_detections = detections

        try:
            run.close()
        except Exception as e:
            print(f"Error while closing graph run: {e}", file=sys.stderr)

        if not graph_times:
            print("Profiling aborted: no successful runs", file=sys.stderr)
            return 4

        graph_arr = np.array(graph_times, dtype=np.float64)
        post_arr = np.array(post_times, dtype=np.float64)
        total_arr = graph_arr + post_arr

        runs = float(len(graph_times))
        graph_fps = runs / graph_arr.sum()
        post_fps = runs / post_arr.sum()
        overall_fps = runs / total_arr.sum()

        print(f"Profiling over {len(graph_times)} runs (image='{image_path}'):")
        print(
            f"  Graph run (push+pull): "
            f"mean={graph_arr.mean():.6f}s, "
            f"min={graph_arr.min():.6f}s, "
            f"max={graph_arr.max():.6f}s, "
            f"FPS={graph_fps:.2f}"
        )
        print(
            f"  Postprocessing (parse+decode+NMS): "
            f"mean={post_arr.mean():.6f}s, "
            f"min={post_arr.min():.6f}s, "
            f"max={post_arr.max():.6f}s, "
            f"FPS={post_fps:.2f}"
        )
        print(
            f"  Overall (graph + post): "
            f"mean={total_arr.mean():.6f}s, "
            f"min={total_arr.min():.6f}s, "
            f"max={total_arr.max():.6f}s, "
            f"FPS={overall_fps:.2f}"
        )

        print(f"Last run detections: {len(last_detections)}")
        for i, det in enumerate(last_detections[:20]):
            box = det["box"]
            print(
                f"  [{i}] score={det['score']:.4f} "
                f"box=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
            )
        # Intentionally do NOT write an output image in profiling mode.
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    try:
        run = build_retinaface_runner(model_path)
    except Exception as e:
        print(f"Error during graph preparation: {e}", file=sys.stderr)
        return 3

    try:
        for image_path in image_paths:
            _log("Invoking run_retinaface_inference()")
            try:
                tensors, orig_bgr, meta = run_retinaface_inference(run, image_path, timeout_ms)
            except Exception as e:
                print(f"Error during inference for {image_path}: {e}", file=sys.stderr)
                return 3

            _log("Collecting tensors from output samples")
            if not tensors:
                print(f"No tensors found in model output for {image_path}", file=sys.stderr)
                return 4

            np_outs = tensor_numpy_outputs(tensors)
            _log("Running RetinaFace postprocessing")
            bboxes, scores, landmarks = parse_retinaface_outputs(np_outs)
            detections = postprocess_retinaface(
                bboxes,
                scores,
                landmarks,
                meta,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
                keep_top_k=keep_top_k,
                with_landmarks=landmarks is not None,
            )

            out_img = draw_detections(orig_bgr, detections)
            output_path = output_dir / f"{image_path.stem}_retinaface.png"
            cv2.imwrite(str(output_path), out_img)
            processed += 1
            print(f"[{processed}/{len(image_paths)}] {image_path.name}: {len(detections)} detections -> {output_path.name}")
    finally:
        try:
            run.close()
        except Exception as exc:
            logger.debug("Failed to close RetinaFace graph run cleanly", exc_info=exc)

    print(f"Done: {processed} images processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
