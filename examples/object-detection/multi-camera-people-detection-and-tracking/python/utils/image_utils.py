"""Image preprocessing and overlay helpers for the Python multi-camera example."""

from __future__ import annotations

from pathlib import Path

from .pipeline import QuantTessCpuPreproc


def sample_output_path(output_dir: Path, stream_index: int, frame_index: int) -> Path:
    return output_dir / f"stream_{stream_index}" / f"frame_{frame_index:06d}.jpg"


def cpu_quanttess_input(runtime, frame_bgr, contract: QuantTessCpuPreproc):
    cv2 = runtime.cv2
    np = runtime.np
    src_h, src_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if contract.aspect_ratio:
        scale = min(contract.width / src_w, contract.height / src_h)
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
    else:
        scaled_w = contract.width
        scaled_h = contract.height

    pad_x = 0
    pad_y = 0
    if contract.padding_type == "CENTER":
        pad_x = (contract.width - scaled_w) // 2
        pad_y = (contract.height - scaled_h) // 2

    resized = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    quant_input = np.zeros((contract.height, contract.width, 3), dtype=np.float32)
    quant_input[pad_y : pad_y + scaled_h, pad_x : pad_x + scaled_w] = (
        resized.astype(np.float32) / 255.0
    )
    return np.ascontiguousarray(quant_input)


def class_color(track_id: int) -> tuple[int, int, int]:
    return (
        int((37 * track_id + 17) % 256),
        int((97 * track_id + 73) % 256),
        int((53 * track_id + 191) % 256),
    )


def draw_tracked_people(runtime, frame, tracked):
    cv2 = runtime.cv2
    for det in tracked:
        color = class_color(det.track_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        text = f"person #{det.track_id} {det.score:.2f}"
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
        )
    return frame


def save_overlay_frame(
    runtime,
    output_dir: Path | None,
    stream_index: int,
    frame_index: int,
    frame,
    save_every: int,
) -> bool:
    if output_dir is None:
        return False
    if save_every <= 0 or frame_index % save_every != 0:
        return False
    out_path = sample_output_path(output_dir, stream_index, frame_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(runtime.cv2.imwrite(str(out_path), frame))
