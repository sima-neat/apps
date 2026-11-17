"""Unit tests for detection-to-vlm-assistant (Python)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML = EXAMPLE_DIR / "src" / "common" / "config.yaml"
DETECTOR_APP_PY = EXAMPLE_DIR / "src" / "python" / "detector_app.py"
GENAI_SERVER_PY = EXAMPLE_DIR / "src" / "python" / "genai_server.py"
DETECTOR_MODULE = "detection_to_vlm_assistant_detector_app"


def load_detector_app():
    """Import detector_app.py once, under a name no other example uses."""
    loaded = sys.modules.get(DETECTOR_MODULE)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(DETECTOR_MODULE, DETECTOR_APP_PY)
    module = importlib.util.module_from_spec(spec)
    # Config is a dataclass, which resolves its annotations through sys.modules.
    sys.modules[DETECTOR_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_input_options_format_uses_pyneat_enum() -> None:
    tree = ast.parse(DETECTOR_APP_PY.read_text(encoding="utf-8"))
    bad_lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "format":
                bad_lines.append(node.lineno)

    assert bad_lines == []


@pytest.mark.unit
def test_genai_config_and_scripts_use_genai_names() -> None:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))

    assert raw["genai_server"]["host"]
    assert raw["genai_server"]["port"] > 0
    assert raw["genai_server"]["model"]["name"]
    assert raw["genai_server"]["model"]["path"]
    assert raw["genai"]["host"]
    assert raw["genai"]["port"] > 0
    assert raw["genai"]["system_prompt"]
    assert raw["genai"]["user_prompt"]

    assert "factual" in raw["genai"]["system_prompt"].lower()
    assert "what is the person doing" in raw["genai"]["user_prompt"].lower()


def padded_nv12_tensor(packed, width: int, height: int, stride: int, uv_row: int):
    """An NV12 tensor laid out the way the decoder hands frames to the app.

    Each row is ``stride`` bytes and the UV plane starts at row ``uv_row`` of the buffer.
    The padding bytes are 255 so that reading them as pixels changes the picture.
    """
    import pyneat

    buffer = np.full((uv_row + height // 2, stride), 255, dtype=np.uint8)
    buffer[:height, :width] = packed[:height]
    buffer[uv_row:uv_row + height // 2, :width] = packed[height:]
    tensor = pyneat.Tensor.from_numpy(
        buffer, copy=True, image_format=pyneat.PixelFormat.NV12, memory=pyneat.TensorMemory.CPU
    )
    planes = []
    for role, rows, offset in (
        (pyneat.PlaneRole.Y, height, 0),
        (pyneat.PlaneRole.UV, height // 2, uv_row * stride),
    ):
        plane = pyneat.Plane()
        plane.role, plane.shape = role, [rows, width]
        plane.strides_bytes, plane.byte_offset = [stride, 1], offset
        planes.append(plane)
    tensor.shape, tensor.strides_bytes = [height, width], [stride, 1]
    tensor.axis_semantics = [pyneat.TensorAxisSemantic.H, pyneat.TensorAxisSemantic.W]
    tensor.planes = planes
    return tensor


# Decoder layouts measured on a DevKit for 1080p and 720p H.264 RTSP streams.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("width", "height", "stride", "uv_row"),
    [(1920, 1080, 2048, 1088), (1280, 720, 1280, 768)],
    ids=["1080p", "720p"],
)
def test_padded_decoded_frame_converts_like_a_packed_one(
    width: int, height: int, stride: int, uv_row: int
) -> None:
    app = load_detector_app()

    rows = np.arange(height, dtype=np.uint16)[:, None]
    cols = np.arange(width, dtype=np.uint16)[None, :]
    luma = ((rows * 3 + cols * 5) % 220 + 16).astype(np.uint8)
    chroma = np.empty((height // 2, width), dtype=np.uint8)
    chroma[:, 0::2] = ((rows[: height // 2] * 7) % 200 + 28).astype(np.uint8)
    chroma[:, 1::2] = ((cols[:, : width // 2] * 3) % 200 + 28).astype(np.uint8)
    packed = np.vstack([luma, chroma])
    expected = cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_NV12)

    frame = app.decoded_tensor_to_rgb(padded_nv12_tensor(packed, width, height, stride, uv_row))

    assert frame.shape == (height, width, 3)
    assert np.array_equal(frame, expected)
