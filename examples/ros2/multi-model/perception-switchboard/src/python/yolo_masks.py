"""YOLO26 payload and mask geometry, kept identical to the native segmenter."""
import cv2
import numpy as np
import pyneat

MASK_STRIDE = 4
YOLO26_MASK_GRID = 160

def tensor_to_numpy(tensor) -> object:
    return np.asarray(tensor.to_numpy(copy=True))

def detection(x1: float, y1: float, x2: float, y2: float, score: float, class_id: int, mask):
    """One decoded instance, in the single representation both families produce.

    ``x1..y2`` are frame pixels, ``score`` is a probability, and ``mask`` is a uint8 mask grid
    covering the letterboxed model input at ``MASK_STRIDE`` cells per pixel. Everything
    downstream of the decoders - overlays, polygons, Insight metadata - reads only these keys.
    """
    return {
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "score": float(score),
        "class_id": int(class_id),
        "mask": mask,
    }

def decode_yolo26_segments(tensors: list, frame_w: int, frame_h: int, max_detections: int):
    """YOLO26 boundary: the MLA already ran BoxDecode, so this only unpacks its payload."""
    decoded = pyneat.decode_segmentation(
        tensors,
        clamp_to=(frame_w, frame_h),
        top_k=max_detections,
        strict=False,
    )
    detections = []
    for item in decoded:
        boxes = tensor_to_numpy(item.boxes).astype(np.float32)
        masks = tensor_to_numpy(item.masks).astype(np.uint8)
        mask_shape = (-1, YOLO26_MASK_GRID, YOLO26_MASK_GRID)
        for row, mask in zip(boxes.reshape((-1, 6)), masks.reshape(mask_shape)):
            x1, y1, x2, y2, score, class_id = row.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(detection(x1, y1, x2, y2, score, int(class_id), mask))
            if len(detections) >= max_detections:
                return detections
    return detections

def frame_rect_for_detection(det: dict, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    x0 = max(0, min(frame_w - 1, int(np.floor(float(det["x1"])))))
    y0 = max(0, min(frame_h - 1, int(np.floor(float(det["y1"])))))
    x1 = max(x0 + 1, min(frame_w, int(np.ceil(float(det["x2"])))))
    y1 = max(y0 + 1, min(frame_h, int(np.ceil(float(det["y2"])))))
    return x0, y0, x1, y1

def mask_rect_for_frame_rect(
    frame_rect: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    mask_h, mask_w = mask_shape[:2]
    model_w = mask_w * MASK_STRIDE
    model_h = mask_h * MASK_STRIDE
    scale = min(model_w / frame_w, model_h / frame_h)
    pad_x = (model_w - frame_w * scale) * 0.5
    pad_y = (model_h - frame_h * scale) * 0.5

    def to_mask_x(frame_x: float) -> float:
        return (frame_x * scale + pad_x) * mask_w / model_w

    def to_mask_y(frame_y: float) -> float:
        return (frame_y * scale + pad_y) * mask_h / model_h

    fx0, fy0, fx1, fy1 = frame_rect
    x0 = max(0, min(mask_w - 1, int(np.floor(to_mask_x(fx0)))))
    y0 = max(0, min(mask_h - 1, int(np.floor(to_mask_y(fy0)))))
    x1 = max(x0 + 1, min(mask_w, int(np.ceil(to_mask_x(fx1)))))
    y1 = max(y0 + 1, min(mask_h, int(np.ceil(to_mask_y(fy1)))))
    return x0, y0, x1, y1

def project_letterbox_mask_roi(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
):
    x0, y0, x1, y1 = frame_rect
    mx0, my0, mx1, my1 = mask_rect_for_frame_rect(frame_rect, frame_shape, mask.shape)
    return cv2.resize(mask[my0:my1, mx0:mx1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)

def mask_polygon(
    mask,
    frame_shape: tuple[int, ...],
    frame_rect: tuple[int, int, int, int],
    threshold: float,
) -> list[list[int]]:
    """Frame-absolute silhouette of ``mask`` inside ``frame_rect``.

    Empty when the thresholded mask holds nothing Insight can draw. ``threshold`` is a fraction of
    full scale, as ``output.mask_threshold`` is. Upscaling before thresholding is what makes the
    outline match the rendered overlay.
    """
    roi = project_letterbox_mask_roi(mask, frame_shape, frame_rect)
    _, binary = cv2.threshold(roi, threshold * 255.0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.004 * cv2.arcLength(largest, True), True)
    if len(polygon) < 3:
        return []
    # Contour points lie inside frame_rect, which is already clamped to the frame, so shifting them
    # into frame space cannot leave the image.
    x0, y0 = frame_rect[0], frame_rect[1]
    return [[int(point[0][0]) + x0, int(point[0][1]) + y0] for point in polygon]
