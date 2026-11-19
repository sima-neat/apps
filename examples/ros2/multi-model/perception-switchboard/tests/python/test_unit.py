"""Host checks for view semantics and the declared runtime controls."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import yaml

APP = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('switchboard_overlays', APP/'src/python/overlays.py')
overlays = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overlays)


@pytest.mark.parametrize('mode', ['detection', 'segmentation', 'both'])
def test_overlay_modes_change_only_requested_pixels(mode):
    image = np.zeros((100, 100, 3), np.uint8)
    objects = [dict(bbox=[20, 30, 60, 60], mask=[[40, 50], [60, 50], [60, 70], [40, 70]],
                    label='person', confidence=.9)]
    before = deepcopy(objects)
    result = overlays.render_yolo(image, objects, mode)
    assert bool(result[30, 20].any()) == (mode in ('detection', 'both'))
    assert bool(result[60, 50].any()) == (mode in ('segmentation', 'both'))
    assert objects == before and not image.any()


def test_config_defaults_keep_depth_on_standby_and_vlm_on_demand():
    config = yaml.safe_load((APP/'src/common/config.yaml').read_text())
    assert config['/**/yolo']['ros__parameters']['enabled'] is True
    assert config['/**/depth']['ros__parameters']['enabled'] is False
    assert 'interval_seconds' not in config['/**/vlm']['ros__parameters']


@pytest.mark.parametrize('mode,shape', [('depth', (90, 160, 3)), ('side_by_side', (45, 160, 3))])
def test_depth_preview_preserves_camera_aspect_and_numerical_depth(mode, shape):
    frame = np.zeros((90, 160, 3), np.uint8)
    depth = np.arange(64, dtype=np.float32).reshape(8, 8)
    before = depth.copy()
    result = overlays.render_depth(frame, depth, mode)
    assert result.shape == shape
    assert result.any() and not frame.any()
    np.testing.assert_array_equal(depth, before)
