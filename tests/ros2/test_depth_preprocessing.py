"""Depth preprocessing contracts; real ROS messages and Neat tensors, no model load."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pyneat
import pytest
from builtin_interfaces.msg import Time

from tests.ros2.local_modules import load_node

ROOT = Path(__file__).resolve().parents[2] / 'examples/ros2'
standalone = load_node(ROOT / 'depth-estimation/depth-estimator/src/python/depth_node.py',
                       'tested_depth_preprocessing')
switchboard = load_node(ROOT / 'multi-model/perception-switchboard/src/python/depth_node.py',
                        'tested_switchboard_preprocessing')
pytestmark = pytest.mark.ros2


@pytest.mark.parametrize('module', [standalone, switchboard])
@pytest.mark.parametrize('height,width', [(39, 73), (96, 51), (131, 211), (211, 131)])
def test_source_pixels_and_native_area_limit(module, height, width):
    # A small limit exercises the same boundary without allocating huge fixtures.
    limit = 96
    bgr = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    message = standalone.to_message(bgr, Time(), 'camera', 'bgr8')
    node = SimpleNamespace(max_input_size=limit)
    tensor = module.DepthNode.prepare_input(node, message)[0]
    expected = bgr
    if max(height, width) > limit:
        scale = limit / max(height, width)
        expected = cv2.resize(bgr, (max(1, int(width * scale)), max(1, int(height * scale))),
                              interpolation=cv2.INTER_AREA)
    expected = np.ascontiguousarray(expected[:, :, ::-1])
    actual = np.asarray(tensor.to_numpy(copy=True))
    assert tuple(tensor.shape) == expected.shape
    np.testing.assert_array_equal(actual.reshape(expected.shape), expected)


def test_standalone_uses_native_bicubic_model_route():
    node = SimpleNamespace()
    with patch.object(standalone.pyneat, 'Model') as model:
        standalone.DepthNode.initialize_model(node, {
            'model': {'path': 'depth.tar.gz'},
            'runtime': {'infer_size': 518, 'max_input_size': 2048},
        })
    options = model.call_args.args[1].preprocess
    assert options.kind == pyneat.InputKind.Image
    assert options.color_convert.input_format == pyneat.PreprocessColorFormat.RGB
    assert (options.input_max_width, options.input_max_height, options.input_max_depth) == (2048, 2048, 3)
    assert options.resize.enable == pyneat.AutoFlag.On
    assert options.resize.mode == pyneat.ResizeMode.Stretch
    assert (options.resize.width, options.resize.height) == (518, 518)
    assert options.resize.scaling_type == 'BICUBIC'


@pytest.mark.parametrize('setting', ['infer_size', 'max_input_size'])
def test_invalid_size_rejected_before_model_load(setting):
    with patch.object(standalone.pyneat, 'Model') as model:
        with pytest.raises(ValueError, match='must be positive'):
            standalone.DepthNode.initialize_model(SimpleNamespace(), {
                'model': {'path': 'depth.tar.gz'}, 'runtime': {setting: 0},
            })
        model.assert_not_called()
