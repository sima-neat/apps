"""Image overlay and debug-save helpers for the Python multistream object detection example."""

from __future__ import annotations

from pathlib import Path

from .sample_utils import Detection


def sample_output_path(output_dir: Path, stream_index: int, frame_index: int) -> Path:
    return output_dir / f"stream_{stream_index}" / f"frame_{frame_index:06d}.jpg"


def class_color(class_id: int) -> tuple[int, int, int]:
    return (
        int((37 * class_id + 17) % 256),
        int((97 * class_id + 73) % 256),
        int((53 * class_id + 191) % 256),
    )


def _class_label(class_labels: list[str], class_id: int) -> str:
    if 0 <= int(class_id) < len(class_labels):
        return class_labels[int(class_id)]
    return str(class_id)


def draw_detection_boxes(runtime, frame, detections: list[Detection], class_labels: list[str]):
    cv2 = runtime.cv2
    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        color = class_color(det.class_id)
        text = f"{_class_label(class_labels, det.class_id)} {det.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, max(0, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def save_debug_frame(
    output_dir: str | Path | None,
    stream_index: int,
    frame_index: int,
    frame,
    save_every: int,
    runtime=None,
) -> bool:
    if output_dir is None:
        return False
    if save_every <= 0 or frame_index % save_every != 0:
        return False
    out_path = sample_output_path(Path(output_dir), stream_index, frame_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        import cv2  # local import keeps module import light

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return bool(cv2.imwrite(str(out_path), frame_bgr))
    frame_bgr = runtime.cv2.cvtColor(frame, runtime.cv2.COLOR_RGB2BGR)
    return bool(runtime.cv2.imwrite(str(out_path), frame_bgr))
