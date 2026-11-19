"""Host tests for image layout and result recording; no board or ROS mocks."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.ros2.local_modules import load_node

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "examples/ros2/face-detection/face-detector/src/python"
transport = load_node(SOURCE / "image_transport.py", "tested_transport", resolve_share=False)
image_array, planar_array = transport.image_array, transport.planar_array
json_value = load_node(SOURCE / "app_config.py", "tested_json", resolve_share=False).json_value


def test_padded_bgr_rows_preserve_visible_pixels():
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.full((2, 16), 255, np.uint8)
    padded[:, :9] = pixels.reshape(2, 9)
    message = SimpleNamespace(width=3, height=2, encoding="bgr8", step=16,
                              data=bytearray(padded.tobytes()))
    view = image_array(message)
    np.testing.assert_array_equal(view, pixels)
    message.data[0] = 42
    assert view[0, 0, 0] == 42  # The caller must retain the owning message.


@pytest.mark.parametrize("step,length", [(8, 16), (9, 17), (9, 19)])
def test_rejects_truncated_or_inconsistent_images(step, length):
    message = SimpleNamespace(width=3, height=2, encoding="bgr8", step=step,
                              data=bytearray(length))
    with pytest.raises(ValueError):
        image_array(message)


def test_planar_geometry_uses_visible_height_and_preserves_both_planes():
    payload = bytearray(range(12))
    message = SimpleNamespace(width=4, height=2, encoding="nv12", step=4, data=payload)
    view = planar_array(message)
    assert view.shape == (3, 4)
    assert view[2].tolist() == [8, 9, 10, 11]
    message.data = payload[:-1]
    with pytest.raises(ValueError):
        planar_array(message)
    message.data = payload
    message.step = 8
    with pytest.raises(ValueError):
        planar_array(message)



def test_result_json_preserves_full_numerical_precision_and_empty_results():
    score = np.float32(0.123456789)
    payload = json.loads(json.dumps({"score": score, "boxes": np.empty((0, 4), np.float32)},
                                    default=json_value, allow_nan=False))
    assert payload == {"score": float(score), "boxes": []}


def test_nonfinite_predictions_fail_instead_of_publishing_invalid_json():
    with pytest.raises(ValueError):
        json.dumps({"score": float("nan")}, default=json_value, allow_nan=False)
