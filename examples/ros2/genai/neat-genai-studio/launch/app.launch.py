"""Wire this example's local nodes; source=none accepts an existing image topic."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def nodes(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    config_path = str(Path(value("config")).resolve(strict=True))
    config = yaml.safe_load(Path(config_path).read_text())
    visual = config.get("ros", {}).get("visualization", {})
    if visual.get("insight", {}).get("enabled", False) or visual.get("save", {}).get("enabled", False):
        raise ValueError("This request/report application has no video output")
    common = dict(package='neat_genai_studio', namespace=value("namespace"), output="screen")
    arguments = ["--config", config_path]
    mode = value("source")
    launched = []
    launched.append(Node(executable='studio_node.py', arguments=arguments, **common))

    def finished(event, launch_context):
        if launch_context.is_shutdown:
            return []
        if event.returncode:
            raise RuntimeError(f"App component failed with exit code {event.returncode}")
        return [EmitEvent(event=Shutdown(reason="App component exited"))]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=finished)) for node in launched]
    return [*handlers, *launched]


def generate_launch_description():
    common = Path(get_package_share_directory('neat_genai_studio')) / "common"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(common / "config.yaml")),
        DeclareLaunchArgument("namespace", default_value="neat"),
        DeclareLaunchArgument("source", default_value='none', choices=['none']),
        OpaqueFunction(function=nodes),
    ])
