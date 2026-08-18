import math
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
import pyneat as neat

MASK_STRIDE = 4.0
PROTO_MASK_SIDE = 160
MIN_CROP_AREA = 100
MAX_CROP_FRAC = 0.5


def lround(value):
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def _floats(tensor):
    if tensor.dtype != neat.TensorDType.Float32:
        raise RuntimeError("expected Float32 tensor")
    return np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np.float32)


def _u8(tensor):
    if tensor.dtype != neat.TensorDType.UInt8:
        raise RuntimeError("expected UInt8 tensor")
    return np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np.uint8)


@dataclass
class Geometry:
    scale: float = 1.0
    pad_x: int = 0
    pad_y: int = 0


@dataclass
class Segmentation:
    boxes: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)


@dataclass
class Crop:
    window: np.ndarray
    submask: np.ndarray


def _model_options(cfg, frame_w, frame_h):
    opt = neat.ModelOptions()
    opt.preprocess.kind = neat.InputKind.Image
    opt.preprocess.color_convert.input_format = neat.PreprocessColorFormat.RGB
    opt.preprocess.input_max_width = frame_w
    opt.preprocess.input_max_height = frame_h
    opt.preprocess.input_max_depth = 3
    opt.preprocess.resize.enable = neat.AutoFlag.On
    opt.preprocess.resize.width = cfg.infer_size
    opt.preprocess.resize.height = cfg.infer_size
    opt.preprocess.resize.mode = neat.ResizeMode.Letterbox
    opt.preprocess.resize.pad_value = 114
    opt.decode_type = neat.BoxDecodeType.YoloV26Seg
    opt.score_threshold = cfg.score_threshold
    opt.nms_iou_threshold = cfg.nms_iou
    opt.top_k = cfg.max_detections
    return opt


def _input_tensor(rgb):
    return neat.Tensor.from_numpy(rgb, image_format=neat.PixelFormat.RGB,
                                  memory=neat.TensorMemory.EV74)


class Fastsam:
    def __init__(self, cfg, run_opt, frame_w, frame_h):
        self._model = neat.Model(cfg.model_path, _model_options(cfg, frame_w, frame_h))
        dummy = np.zeros((frame_h, frame_w, 3), np.uint8)
        self._runner = self._model.build([_input_tensor(dummy)], neat.ModelRouteOptions(), run_opt)

    def run(self, rgb, timeout_ms):
        return self._runner.run([_input_tensor(rgb)], timeout_ms)

    def close(self):
        self._runner.close()


def letterbox_geometry(orig_w, orig_h, infer_size):
    scale = infer_size / max(orig_w, orig_h)
    return Geometry(scale,
                    (infer_size - lround(orig_w * scale)) // 2,
                    (infer_size - lround(orig_h * scale)) // 2)


def decode(model_out, top_k):
    decoded = neat.decode_segmentation(model_out, clamp_to=None, top_k=top_k, strict=False)
    seg = Segmentation()
    mask_pixels = PROTO_MASK_SIDE * PROTO_MASK_SIDE
    for item in decoded:
        flat = _floats(item.boxes)
        masks = _u8(item.masks)
        count = flat.size // 6
        boxes = flat[:count * 6].reshape(count, 6)
        for i in range(count):
            seg.boxes.append(boxes[i])
            if masks.size >= (i + 1) * mask_pixels:
                seg.masks.append(masks[i * mask_pixels:(i + 1) * mask_pixels]
                                 .reshape(PROTO_MASK_SIDE, PROTO_MASK_SIDE).copy())
            else:
                seg.masks.append(np.zeros((PROTO_MASK_SIDE, PROTO_MASK_SIDE), np.uint8))
    return seg


def mask_polygon(mask, geom, max_points=80, eps_frac=0.004):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) <= 0.0:
        return []

    approx = cv2.approxPolyDP(best, eps_frac * cv2.arcLength(best, True), True).reshape(-1, 2)
    if len(approx) > max_points:
        pts = [approx[int(k * (len(approx) - 1) / (max_points - 1))] for k in range(max_points)]
    else:
        pts = list(approx)
    if len(pts) < 3:
        return []

    return [(lround((p[0] * MASK_STRIDE - geom.pad_x) / geom.scale),
             lround((p[1] * MASK_STRIDE - geom.pad_y) / geom.scale)) for p in pts]


def object_crop(proto_mask, frame_rgb, geom, max_box_frac=1.0,
                min_area=MIN_CROP_AREA, max_frac=MAX_CROP_FRAC, margin=0.1):
    nz = cv2.findNonZero(proto_mask)
    if nz is None:
        return None
    orig_h, orig_w = frame_rgb.shape[:2]
    mh, mw = proto_mask.shape[:2]
    cnt = float(len(nz))
    if cnt * orig_w * orig_h < min_area * mw * mh:
        return None
    if cnt > max_frac * mw * mh:
        return None

    pts = nz.reshape(-1, 2)
    px1 = int(pts[:, 0].min())
    py1 = int(pts[:, 1].min())
    px2 = int(pts[:, 0].max()) + 1
    py2 = int(pts[:, 1].max()) + 1

    fx1 = (px1 * MASK_STRIDE - geom.pad_x) / geom.scale
    fy1 = (py1 * MASK_STRIDE - geom.pad_y) / geom.scale
    fx2 = (px2 * MASK_STRIDE - geom.pad_x) / geom.scale
    fy2 = (py2 * MASK_STRIDE - geom.pad_y) / geom.scale
    box_w = min(fx2, float(orig_w)) - max(fx1, 0.0)
    box_h = min(fy2, float(orig_h)) - max(fy1, 0.0)
    if box_w > max_box_frac * orig_w or box_h > max_box_frac * orig_h:
        return None
    dw = (fx2 - fx1) * margin
    dh = (fy2 - fy1) * margin
    x1 = max(0, int(fx1 - dw))
    y1 = max(0, int(fy1 - dh))
    x2 = min(orig_w, lround(fx2 + dw))
    y2 = min(orig_h, lround(fy2 + dh))
    if x2 <= x1 or y2 <= y1:
        return None

    # Map the proto mask into the exact frame ROI. Keeping window and submask
    # pixel-aligned is important because CLIP resizes and crops both together.
    proto_to_frame = MASK_STRIDE / geom.scale
    center_offset = (proto_to_frame - 1.0) / 2.0
    transform = np.array([
        [proto_to_frame, 0.0, -geom.pad_x / geom.scale - x1 + center_offset],
        [0.0, proto_to_frame, -geom.pad_y / geom.scale - y1 + center_offset],
    ], dtype=np.float32)
    submask = cv2.warpAffine(
        proto_mask,
        transform,
        (x2 - x1, y2 - y1),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return Crop(window=frame_rgb[y1:y2, x1:x2], submask=submask)
