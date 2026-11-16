"""Capture timestamped NV12 snapshots from the common ROS camera source."""
import argparse
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.images import planar_array, header_dict
from neat_apps.visual_output import VisualOutput


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visual-insight", choices=("true", "false"))
    args, ros_args = parser.parse_known_args(argv)
    config = native.load_config(args.config)
    settings = native.yaml.safe_load(args.config.read_text()).get("ros", {})
    if args.visual_insight is not None:
        settings.setdefault("visualization", {}).setdefault("insight", {})["enabled"] = args.visual_insight == "true"
    visual_output = VisualOutput(settings.get("visualization", {}))
    rclpy.init(args=ros_args)
    node = Node("mipi_camera_capture")
    results = node.create_publisher(String, settings.get("results_topic", "results"), 16)
    pending, pts = [], []
    started = None
    last_frame = time.monotonic()
    frames = target_index = 0

    def receive(message):
        nonlocal started, last_frame, frames, target_index
        if (message.encoding, message.width, message.height) != ("nv12", config.camera.width, config.camera.height):
            raise ValueError("Camera image geometry/encoding differs from the capture configuration")
        pixels = planar_array(message)
        now = time.monotonic()
        started = now if started is None else started
        last_frame = now
        elapsed = now - started
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if pts and stamp <= pts[-1]:
            raise ValueError("Camera timestamps must increase")
        pts.append(stamp)
        targets = native.reached_target_indices(config.capture.sample_times_seconds, target_index, elapsed)
        if targets:
            payload = pixels.tobytes()
            for index in targets:
                pending.append(native.PendingCapture(config.capture.sample_times_seconds[index], elapsed,
                    frames + 1, f"frame_{index:02d}_{elapsed:07.3f}s.nv12", payload))
            target_index = targets.stop
        visual_output.write(message, {}, frames)
        results.publish(String(data=json.dumps({"schema_version": 1, "sequence": frames,
            "header": header_dict(message.header), "data": {"width": message.width, "height": message.height}})))
        frames += 1

    node.create_subscription(Image, settings.get("image_topic", "image"), receive, 16)
    code, error = 0, None
    try:
        while rclpy.ok() and (started is None or time.monotonic() - started < config.capture.duration_seconds):
            rclpy.spin_once(node, timeout_sec=0.05)
            timeout = (float(settings.get("source_timeout_seconds", 30)) if started is None
                       else config.capture.pull_timeout_ms / 1000)
            if time.monotonic() - last_frame > timeout:
                raise TimeoutError("Camera image topic stopped delivering frames")
        if target_index != len(config.capture.sample_times_seconds):
            raise RuntimeError("Capture ended before all requested snapshots arrived")
    except (Exception, KeyboardInterrupt) as exc:
        code, error = 1, str(exc) or "Capture interrupted"
        node.get_logger().error(error)
    finally:
        visual_output.close()
        elapsed = time.monotonic() - started if started is not None else 0.0
        config.output_directory.mkdir(parents=True, exist_ok=True)
        captures = native.write_captures(pending, config.output_directory, config.camera.width, config.camera.height)
        mean_ms, max_ms = native.pts_interarrival_stats_ms(pts)
        summary = {"frames_received": frames, "elapsed_s": elapsed,
                   "received_fps": (frames - 1) / (last_frame - started) if frames > 1 else None,
                   "source_stamp_fps": (frames - 1) * 1e9 / (pts[-1] - pts[0]) if frames > 1 else None,
                   "interarrival_mean_ms": mean_ms, "interarrival_max_ms": max_ms,
                   "captures": captures, "error": error}
        native.write_json(config.output_directory / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
