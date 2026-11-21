"""Drawing only: changing the view never changes numerical inference results."""
import cv2
import numpy as np


def render_yolo(frame, objects, mode):
    result = frame.copy()
    if mode in ('segmentation', 'both'):
        colors = result.copy()
        for item in objects:
            if item['mask']:
                cv2.fillPoly(colors, [np.asarray(item['mask'], np.int32)], (0, 180, 0))
        result = cv2.addWeighted(result, 0.55, colors, 0.45, 0)
    if mode in ('detection', 'both'):
        for item in objects:
            x, y, w, h = item['bbox']
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 220, 0), 2)
            cv2.putText(result, f"{item['label']} {item['confidence']:.2f}",
                        (x, max(12, y-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return result


def render_depth(frame, depth, mode):
    """Restore the source aspect ratio for display; leave numerical depth unchanged."""
    colors = cv2.applyColorMap(
        cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_INFERNO)
    height, width = frame.shape[:2]
    if mode == 'depth':
        return cv2.resize(colors, (width, height), interpolation=cv2.INTER_LINEAR)
    if mode == 'side_by_side':
        panel = (max(1, width // 2), max(1, height // 2))
        return np.concatenate((cv2.resize(frame, panel), cv2.resize(colors, panel)), axis=1)
    raise ValueError('preview_mode must be depth or side_by_side')
