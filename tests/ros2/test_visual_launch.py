"""Validate output-mode selection without starting camera/model processes."""
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch_ros.actions import Node
import pytest
import yaml

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]


def actions(tmp_path, visual, *, source='none', insight='auto', app='face-detection/face-detector'):
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump({'ros': {'visualization': visual}, 'output': {'video_enabled': True}}))
    spec=importlib.util.spec_from_file_location('visual_launch_test',ROOT/'support/ros2/launch/app.launch.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    context=LaunchContext()
    context.launch_configurations.update(app=app, apps_root=str(ROOT), config=str(path),
                                        source=source, insight=insight, namespace='neat', stream_index='all')
    return module.nodes(context)


def test_image_topic_visualization_needs_no_extra_ros_process(tmp_path):
    result=actions(tmp_path, {'insight': {'enabled': True, 'host': 'example.test'}})
    assert sum(isinstance(action,Node) for action in result)==1


def test_live_graph_yaml_switch_starts_existing_metadata_sink(tmp_path):
    app='object-detection/high-density-multi-stream-object-detector'
    result=actions(tmp_path, {'insight': {'enabled': True}}, source='live_graph', app=app)
    assert sum(isinstance(action,Node) for action in result)==2
    result=actions(tmp_path, {'insight': {'enabled': True}}, source='live_graph', app=app, insight='false')
    assert sum(isinstance(action,Node) for action in result)==1


def test_live_graph_rejects_unavailable_decoded_image_saving(tmp_path):
    with pytest.raises(ValueError,match='Saving images needs'):
        actions(tmp_path, {'save': {'enabled': True}}, source='live_graph',
                app='object-detection/high-density-multi-stream-object-detector')


@pytest.mark.parametrize('app',['genai/neat-genai-studio','benchmarking/model-benchmark'])
def test_nonvisual_app_does_not_silently_accept_video_config(tmp_path, app):
    with pytest.raises(ValueError,match='no video output'):
        actions(tmp_path, {'insight': {'enabled': True}}, app=app)
