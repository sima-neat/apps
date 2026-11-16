"""Launch image source -> inference -> results from an Apps bundle or checkout."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path
import yaml


def nodes(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    common = {"package": "neat_apps", "output": "screen", "namespace": value("namespace"),
              "cwd": value("apps_root")}
    app_args = ["--app", value("app"), "--apps-root", value("apps_root"), "--config", value("config")]
    source_kind = value("source")
    config = yaml.safe_load(Path(value("config")).read_text())
    visual = config.get("ros", {}).get("visualization", {})
    configured_insight = visual.get("insight", {}).get("enabled", False)
    if type(configured_insight) is not bool:
        raise ValueError("ros.visualization.insight.enabled must be true or false")
    insight = configured_insight if value("insight") == "auto" else value("insight") == "true"
    nonvisual = value("app") in ("genai/neat-genai-studio", "benchmarking/model-benchmark")
    if nonvisual and (insight or visual.get("save", {}).get("enabled", False)):
        raise ValueError("This request/report application has no video output")
    if source_kind != "live_graph" and value("insight") != "auto":
        if not nonvisual:
            app_args += ["--visual-insight", value("insight")]
    inference = Node(executable="app_node", arguments=app_args, **common)
    if source_kind == "live_graph":
        if value("app") != "object-detection/high-density-multi-stream-object-detector":
            raise ValueError("live_graph is supported by the high-density detector only")
        if value("stream_index") != "all":
            raise ValueError("live_graph uses all configured streams; edit streams in the config")
        if visual.get("save", {}).get("enabled", False):
            raise ValueError("Saving images needs source:=live or none; live_graph retains encoded-video passthrough")
        inference = Node(executable="app_node", arguments=app_args + ["--live-graph"], **common)
        launched = [inference]
        if insight:
            if not config.get("output", {}).get("video_enabled", True):
                raise ValueError("Set output.video_enabled: true to view Insight video and overlays")
            launched.insert(0, Node(executable="insight_sink", arguments=["--config", value("config")], **common))
        def finished(event, launch_context):
            if event.returncode and not launch_context.is_shutdown:
                raise RuntimeError(f"Live graph or Insight sink failed with exit code {event.returncode}")
            if not launch_context.is_shutdown:
                return [EmitEvent(event=Shutdown(reason="Live graph or Insight sink exited"))]
            return []
        handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=finished)) for node in launched]
        return [*handlers, *launched]
    if source_kind == "none":
        return [inference]
    if source_kind not in ("folder", "live"):
        raise ValueError("source must be folder, live, live_graph, or none")
    index = value("stream_index")
    indices = list(range(len(config.get("streams", [None])))) if index == "all" else [int(index)]
    separate_models = value("app") == "multi-model/multi-stream-multi-model"
    if source_kind == "live" and index == "all" and not separate_models:
        indices = ["all"]  # One source node owns all camera windows and acknowledgments.
    inferences = [] if separate_models else [inference]
    sources = []
    for index in indices if source_kind == "live" else [0]:
        stream_common = dict(common)
        if separate_models:
            stream_common["namespace"] = common["namespace"].rstrip("/") + f"/stream_{index}"
            inferences.append(Node(executable="app_node", arguments=app_args + ["--stream-index", str(index)],
                                   **stream_common))
        arguments = ["--config", value("config")]
        if source_kind == "folder":
            arguments += ["--app", value("app")]
        else:
            arguments += ["--apps-root", value("apps_root"), "--app", value("app"), "--stream-index", str(index)]
        sources.append(Node(executable="image_source" if source_kind == "folder" else "live_source",
                            arguments=arguments, **stream_common))

    remaining = set(sources)
    def source_finished(event, launch_context):
        remaining.discard(event.action)
        if event.returncode and not launch_context.is_shutdown:
            raise RuntimeError(f"Image source failed with exit code {event.returncode}")
        if not launch_context.is_shutdown and (event.returncode or not remaining):
            return [EmitEvent(event=Shutdown(reason=f"Source exited with code {event.returncode}"))]
        return []

    def inference_finished(event, launch_context):
        if event.returncode and not launch_context.is_shutdown:
            raise RuntimeError(f"Inference failed with exit code {event.returncode}")
        if not launch_context.is_shutdown:
            return [EmitEvent(event=Shutdown(reason=f"Inference exited with code {event.returncode}"))]
        return []

    handlers = [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=source_finished)) for node in sources]
    handlers += [RegisterEventHandler(OnProcessExit(target_action=node, on_exit=inference_finished)) for node in inferences]
    return [*handlers, *inferences, *sources]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("app", description="category/example"),
        DeclareLaunchArgument("apps_root", description="Absolute Apps bundle/repository path"),
        DeclareLaunchArgument("config", description="Absolute native config.yaml path"),
        DeclareLaunchArgument("namespace", default_value="neat"),
        DeclareLaunchArgument("source", default_value="folder", description="folder, live, live_graph (high-density), or none"),
        DeclareLaunchArgument("insight", default_value="auto", choices=["auto", "true", "false"],
                             description="auto reads ros.visualization.insight.enabled; true/false overrides it"),
        DeclareLaunchArgument("stream_index", default_value="all", description="all or one configured stream index"),
        OpaqueFunction(function=nodes),
    ])
