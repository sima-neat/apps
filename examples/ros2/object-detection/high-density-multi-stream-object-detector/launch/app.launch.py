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
    common = dict(package='neat_high_density_multi_stream_object_detector', namespace=value("namespace"), output="screen")
    arguments = ["--config", config_path]
    mode = value("source")
    launched = []
    if mode == "live_graph":
        if value("stream_index") != "all":
            raise ValueError("live_graph uses all configured streams; edit streams in the config")
        if config.get("ros", {}).get("visualization", {}).get("save", {}).get("enabled", False):
            raise ValueError("Saving images requires source=live or none")
        arguments.append("--live-graph")
        if config.get("ros", {}).get("visualization", {}).get("insight", {}).get("enabled", False):
            if not config.get("output", {}).get("video_enabled", True):
                raise ValueError("Set output.video_enabled=true for Insight")
            launched.append(Node(executable="insight_sink.py", arguments=["--config", config_path], **common))
    launched.append(Node(executable='detection_node.py', arguments=arguments, **common))
    if mode == "live":
        launched.append(Node(executable="input_node.py", arguments=["--config", config_path, "--stream-index", value("stream_index")], **common))

    def finished(event, launch_context):
        if launch_context.is_shutdown:
            return []
        if event.returncode:
            raise RuntimeError(f"App component failed with exit code {event.returncode}")
        return [EmitEvent(event=Shutdown(reason="App component exited"))]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=finished)) for node in launched]
    return [*handlers, *launched]


def generate_launch_description():
    common = Path(get_package_share_directory('neat_high_density_multi_stream_object_detector')) / "common"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(common / "config.yaml")),
        DeclareLaunchArgument("namespace", default_value="neat"),
        DeclareLaunchArgument("source", default_value='live_graph', choices=['none', 'live', 'live_graph']),
        DeclareLaunchArgument("stream_index", default_value="all"),
        OpaqueFunction(function=nodes),
    ])
