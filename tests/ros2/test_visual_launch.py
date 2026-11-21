"""Validate output-mode selection without starting camera/model processes."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node
import pytest
import yaml

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]


def actions(tmp_path, visual, *, source='none', app='face-detection/face-detector'):
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({'ros': {'visualization': visual}, 'output': {'video_enabled': True}}))
    spec=importlib.util.spec_from_file_location('visual_launch_test',ROOT/'examples/ros2'/app/'launch/app.launch.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    context=LaunchContext()
    context.launch_configurations.update(app=app, apps_root=str(ROOT), config=str(path),
                                        source=source, namespace='neat', stream_index='all')
    return module.nodes(context)


def test_image_topic_visualization_needs_no_extra_ros_process(tmp_path):
    result=actions(tmp_path, {'insight': {'enabled': True, 'host': 'example.test'}})
    assert sum(isinstance(action,Node) for action in result)==1


def test_live_graph_yaml_switch_starts_existing_metadata_sink(tmp_path):
    app='object-detection/high-density-multi-stream-object-detector'
    result=actions(tmp_path, {'insight': {'enabled': True}}, source='live_graph', app=app)
    assert sum(isinstance(action,Node) for action in result)==2
    result=actions(tmp_path, {'insight': {'enabled': False}}, source='live_graph', app=app)
    assert sum(isinstance(action,Node) for action in result)==1


def test_live_graph_rejects_unavailable_decoded_image_saving(tmp_path):
    with pytest.raises(ValueError,match='Saving images requires'):
        actions(tmp_path, {'save': {'enabled': True}}, source='live_graph',
                app='object-detection/high-density-multi-stream-object-detector')


@pytest.mark.parametrize('app',['genai/neat-genai-studio','benchmarking/model-benchmark'])
def test_nonvisual_app_does_not_silently_accept_video_config(tmp_path, app):
    with pytest.raises(ValueError,match='no video output'):
        actions(tmp_path, {'insight': {'enabled': True}}, app=app)


def test_multi_model_waits_for_all_finite_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'streams': ['first', 'second']}))
    spec = importlib.util.spec_from_file_location('multi_launch_test', ROOT /
        'examples/ros2/multi-model/multi-stream-multi-model/launch/app.launch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OnProcessExit', lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, 'RegisterEventHandler', lambda handler: handler)
    context = LaunchContext()
    context.launch_configurations.update(config=str(path), source='live',
                                        namespace='neat', stream_index='all')
    result = module.nodes(context)
    handlers = [item for item in result if not isinstance(item, Node)]
    first, second = handlers[1], handlers[3]  # inference/source for each stream
    assert first.on_exit(SimpleNamespace(action=first.target_action, returncode=0), context) == []
    assert len(second.on_exit(SimpleNamespace(action=second.target_action, returncode=0), context)) == 1
    with pytest.raises(RuntimeError, match='exit code 1'):
        first.on_exit(SimpleNamespace(action=first.target_action, returncode=1), context)


def test_multi_model_assigns_distinct_insight_channels(tmp_path, monkeypatch):
    from tests.ros2.local_modules import load_node
    module = load_node(ROOT / 'examples/ros2/multi-model/multi-stream-multi-model/src/python/inference_node.py',
                       'multi_model_channel_test')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'ros': {'visualization': {'insight': {'channel_base': 5}}}}))
    channels = []
    class ConfigObserved(Exception):
        pass
    def inspect_config(config, config_path):
        channels.append(config['ros']['visualization']['insight']['channel_base'])
        raise ConfigObserved()
    monkeypatch.setattr(module.rclpy, 'init', lambda **kwargs: None)
    monkeypatch.setattr(module, 'InferenceNode', inspect_config)
    for index in (0, 1, 2, 3):
        with pytest.raises(ConfigObserved):
            module.main(['--config', str(path), '--stream-index', str(index)])
    assert channels == [5, 6, 7, 8]


def test_folder_preview_preserves_aspect_ratio_with_fixed_even_canvas():
    import numpy as np
    from tests.ros2.local_modules import load_node
    module = load_node(ROOT / 'examples/ros2/face-detection/face-detector/src/python/visual_output.py',
                       'folder_preview_test')
    landscape = np.full((3, 6, 3), 200, np.uint8)
    preview = module.fit_video_frame(landscape, (8, 8, 3))
    assert preview.shape == (8, 8, 3)
    assert np.all(preview[2:6] == 200)
    assert not preview[:2].any() and not preview[6:].any()
    portrait = np.full((6, 3, 3), 100, np.uint8)
    preview = module.fit_video_frame(portrait, (8, 8, 3))
    assert np.all(preview[:, 2:6] == 100)
    assert not preview[:, :2].any() and not preview[:, 6:].any()
    odd = np.full((5, 7, 3), 75, np.uint8)
    preview = module.fit_video_frame(odd, (6, 8, 3))
    assert preview.shape == (6, 8, 3)
    assert np.all(odd == 75)  # Original images are retained for inference/saving.
