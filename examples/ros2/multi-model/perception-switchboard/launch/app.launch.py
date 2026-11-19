"""Launch the five local nodes with one explicit ROS parameter section per node."""
from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def nodes(context):
    path = Path(LaunchConfiguration('config').perform(context)).resolve(strict=True)
    config = yaml.safe_load(path.read_text())
    namespace = LaunchConfiguration('namespace').perform(context)
    names = ['yolo', 'depth', 'vlm', 'output']
    if LaunchConfiguration('source').perform(context) == 'local':
        names.insert(0, 'input')
    processes = []
    for name in names:
        parameters = dict(config['/**/' + name]['ros__parameters'])
        for key in ('path', 'model_path', 'labels_path', 'save.directory'):
            if parameters.get(key):
                parameters[key] = str((path.parent / parameters[key]).resolve())
        processes.append(Node(package='neat_perception_switchboard', executable=name+'_node.py',
                              name=name, namespace=namespace, parameters=[parameters], output='screen'))

    def stopped(event, context):
        if context.is_shutdown:
            return []
        if event.returncode:
            raise RuntimeError(f'Switchboard node failed with exit code {event.returncode}')
        return [EmitEvent(event=Shutdown(reason='Switchboard node exited'))]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=stopped)) for node in processes]
    return [*handlers, *processes]


def generate_launch_description():
    share = Path(get_package_share_directory('neat_perception_switchboard'))
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(share/'common/config.yaml')),
        DeclareLaunchArgument('namespace', default_value='neat'),
        DeclareLaunchArgument('source', default_value='local', choices=['local', 'none']),
        OpaqueFunction(function=nodes),
    ])
