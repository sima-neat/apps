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
    common = dict(package='neat_multi_stream_multi_model', namespace=value("namespace"), output="screen")
    arguments = ["--config", config_path]
    mode = value("source")
    launched = []
    sources = []
    indices = range(len(config["streams"])) if value("stream_index") == "all" else [int(value("stream_index"))]
    for index in indices:
        scoped = dict(common, namespace=common["namespace"].rstrip("/") + f"/stream_{index}")
        launched.append(Node(executable="inference_node.py", arguments=arguments + ["--stream-index", str(index)], **scoped))
        if mode == "live":
            source = Node(executable="input_node.py", arguments=["--config", config_path, "--stream-index", str(index)], **scoped)
            sources.append(source)
            launched.append(source)

    def finished(event, launch_context):
        if launch_context.is_shutdown:
            return []
        if event.returncode:
            raise RuntimeError(f"App component failed with exit code {event.returncode}")
        # A finite source finishes only after its stream has matching results.
        # Let the other streams drain before stopping their inference nodes.
        if event.action in sources:
            sources.remove(event.action)
            if sources:
                return []
        return [EmitEvent(event=Shutdown(reason="App component exited"))]

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=finished)) for node in launched]
    return [*handlers, *launched]


def generate_launch_description():
    common = Path(get_package_share_directory('neat_multi_stream_multi_model')) / "common"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(common / "config.yaml")),
        DeclareLaunchArgument("namespace", default_value="neat"),
        DeclareLaunchArgument("source", default_value='live', choices=['none', 'live']),
        DeclareLaunchArgument("stream_index", default_value="all"),
        OpaqueFunction(function=nodes),
    ])
