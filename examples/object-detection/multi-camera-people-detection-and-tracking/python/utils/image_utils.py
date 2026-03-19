"""Image preprocessing and overlay helpers for the Python multi-camera example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pipeline import QuantTessCpuPreproc


@dataclass
class QuantTessCpuPreprocState:
    src_width: int
    src_height: int
    scaled_w: int
    scaled_h: int
    pad_x: int
    pad_y: int
    quant_input: Any


def sample_output_path(output_dir: Path, stream_index: int, frame_index: int) -> Path:
    return output_dir / f"stream_{stream_index}" / f"frame_{frame_index:06d}.jpg"


def build_cpu_quanttess_preproc_state(
    runtime,
    contract: QuantTessCpuPreproc,
    src_width: int,
    src_height: int,
) -> QuantTessCpuPreprocState:
    np = runtime.np
    if contract.aspect_ratio:
        scale = min(contract.width / src_width, contract.height / src_height)
        scaled_w = max(1, int(round(src_width * scale)))
        scaled_h = max(1, int(round(src_height * scale)))
    else:
        scaled_w = contract.width
        scaled_h = contract.height

    pad_x = 0
    pad_y = 0
    if contract.padding_type == "CENTER":
        pad_x = (contract.width - scaled_w) // 2
        pad_y = (contract.height - scaled_h) // 2

    quant_input = np.zeros((contract.height, contract.width, 3), dtype=np.float32)
    return QuantTessCpuPreprocState(
        src_width=src_width,
        src_height=src_height,
        scaled_w=scaled_w,
        scaled_h=scaled_h,
        pad_x=pad_x,
        pad_y=pad_y,
        quant_input=quant_input,
    )


def cpu_quanttess_input(
    runtime,
    frame_rgb,
    state: QuantTessCpuPreprocState,
):
    cv2 = runtime.cv2
    np = runtime.np
    src_h, src_w = frame_rgb.shape[:2]
    if src_w != state.src_width or src_h != state.src_height:
        raise RuntimeError(
            "unexpected frame size for cached QuantTess preproc state: "
            f"got {src_w}x{src_h}, expected {state.src_width}x{state.src_height}"
        )

    resized = cv2.resize(
        frame_rgb,
        (state.scaled_w, state.scaled_h),
        interpolation=cv2.INTER_LINEAR,
    )
    state.quant_input.fill(0.0)
    roi = state.quant_input[
        state.pad_y : state.pad_y + state.scaled_h,
        state.pad_x : state.pad_x + state.scaled_w,
    ]
    np.multiply(resized, 1.0 / 255.0, out=roi, casting="unsafe")
    return state.quant_input


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
