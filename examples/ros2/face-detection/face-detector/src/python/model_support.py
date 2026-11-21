"""Local model setup, decoding and numerical helpers for this example."""
from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
import argparse

import logging

import sys

import time

from pathlib import Path

from typing import Any, NamedTuple

import yaml

VERBOSE = False

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/retinaface_mobilenet25_mod_0_mpk.tar.gz"

DEFAULT_CONFIG = Path(get_package_share_directory('neat_face_detector')) / "common" / "config.yaml"

INFER_WIDTH = 640

INFER_HEIGHT = 640

CFG_MNET = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}

class PreprocMeta(NamedTuple):
    orig_h: int
    orig_w: int
    padded_h: int
    padded_w: int
    pad_top: int
    pad_left: int

def tensor_to_numpy(t: pyneat.Tensor) -> np.ndarray:
    return np.asarray(t.to_numpy(copy=True))

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
