"""Launch this package's input, detection, VLM and optional GenAI service."""
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def nodes(context):
    config_path = str(Path(LaunchConfiguration('config').perform(context)).resolve(strict=True))
    config = yaml.safe_load(Path(config_path).read_text())
    common = dict(package='neat_detection_vlm', output='screen',
                  namespace=LaunchConfiguration('namespace').perform(context),
                  arguments=['--config', config_path])
    detector = Node(executable='detection_node.py', **common)
    launched = [detector]
    if LaunchConfiguration('source').perform(context) == 'live':
        launched.append(Node(executable='input_node.py', **common))
    if config['genai']['enabled']:
        timeout = str(float(config['genai']['timeout_seconds']) + 5)
        if LaunchConfiguration('server').perform(context) == 'true':
            server = Path(get_package_prefix('neat_detection_vlm')) / 'lib/neat_detection_vlm/genai_server.py'
            launched.append(ExecuteProcess(cmd=[str(server), '--config', config_path],
                                           output='screen', sigterm_timeout=timeout))
        launched.append(Node(executable='vlm_node.py', sigterm_timeout=timeout, **common))

    def finished(event, launch_context):
        if launch_context.is_shutdown:
            return []
        if event.returncode:
            raise RuntimeError(f'App component failed with exit code {event.returncode}')
        return [EmitEvent(event=Shutdown(reason='App component exited'))]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=finished)) for node in launched]
    return [*handlers, *launched]


def generate_launch_description():
    share = Path(get_package_share_directory('neat_detection_vlm'))
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(share / 'common/config.yaml')),
        DeclareLaunchArgument('namespace', default_value='neat'),
        DeclareLaunchArgument('source', default_value='live', choices=['live', 'none'],
                             description='Start RTSP input, or use an external image topic'),
        DeclareLaunchArgument('server', default_value='true', choices=['true', 'false'],
                             description='Start the local VLM service, or use an existing service'),
        OpaqueFunction(function=nodes),
    ])
