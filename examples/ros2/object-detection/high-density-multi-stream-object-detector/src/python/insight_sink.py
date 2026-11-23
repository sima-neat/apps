#!/usr/bin/env python3
"""Forward source-timestamped ROS detections to Neat Insight metadata channels."""
import argparse
import json
from pathlib import Path

import yaml


def detection_metadata(result):
    """Use source PTS, never ROS arrival time, to match the encoded video."""
    if result.get("schema_version") != 1:
        raise ValueError("Expected result schema_version 1")
    stream = result["stream_id"]
    if not isinstance(stream, str) or not stream.startswith("stream") or not stream[6:].isdigit():
        raise ValueError("Expected a native stream ID such as stream0")
    index = int(stream[6:])
    stamp = result["header"]["stamp"]
    sec, nsec = stamp["sec"], stamp["nanosec"]
    frame = result["source_frame_id"]
    if (type(sec) is not int or type(nsec) is not int or sec < 0 or
            not 0 <= nsec < 1_000_000_000 or type(frame) is not int or frame < 0):
        raise ValueError("Expected a nonnegative source frame ID and presentation timestamp")
    data = result["data"]
    if not isinstance(data.get("objects"), list):
        raise ValueError("Expected detection data.objects")
    pts = sec * 1_000_000_000 + nsec
    return index, {"type": "object-detection", "data": data,
                   "timestamp": pts // 1_000_000, "pts_ns": pts,
                   "rtp_timestamp": ((pts * 90_000) // 1_000_000_000) & 0xFFFFFFFF,
                   "frame_id": str(frame), "stream_id": stream, "stream_index": index}


def main(argv=None):
    import pyneat
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from std_msgs.msg import String

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    output = config["output"]["insight"]
    stream_count = len(config["streams"])
    visible = int(output.get("max_visible_streams", -1))
    visible = stream_count if visible < 0 else min(visible, stream_count)
    if visible < 1:
        raise ValueError("Insight requires at least one visible stream")
    senders = []
    for index in range(visible):
        options = pyneat.MetadataSenderOptions()
        options.host = output["host"]
        options.channel = index
        options.metadata_port_base = int(output.get("metadata_port_base", 9100))
        send_options = pyneat.MetadataSenderSendOptions()
        send_options.nonblocking = True
        senders.append(pyneat.MetadataSender(options, send_options))

    rclpy.init(args=ros_args)
    node = Node("neat_insight_sink")
    forwarded = failures = 0

    def receive(message):
        nonlocal forwarded, failures
        index, metadata = detection_metadata(json.loads(message.data))
        if index >= stream_count:
            raise ValueError(f"Result stream {index} is outside the configured streams")
        if index >= visible:
            return
        sent = senders[index].send_raw_json(json.dumps(metadata, separators=(",", ":"), allow_nan=False))
        if sent:
            forwarded += 1
        else:
            failures += 1
            if failures & (failures - 1) == 0:
                node.get_logger().warning(f"Insight metadata send failed ({failures} failures)")

    node.create_subscription(String, config.get("ros", {}).get("results_topic", "results"), receive, 64)
    node.get_logger().info(f"Forwarding ROS detections to Insight at {output['host']}, channels 0–{visible - 1}")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(json.dumps({"insight_forwarded": forwarded, "insight_send_failures": failures}), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
