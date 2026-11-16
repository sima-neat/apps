"""Record actual ROS messages for parity and delivery checks."""
import argparse
import json
from pathlib import Path
import time
import hashlib

from .images import header_dict, stamp_key


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--image-topic", action="append", default=[],
                        help="Also require count Images on this topic; repeat for multiple dense outputs")
    args, ros_args = parser.parse_known_args(argv)
    if args.count < 1 or args.timeout <= 0:
        parser.error("count and timeout must be positive")
    rclpy.init(args=ros_args)
    node = Node("neat_result_sink")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    received = 0
    expected_sequence = 0
    image_headers = {topic: set() for topic in args.image_topic}
    result_headers = set()
    deadline = time.monotonic() + args.timeout
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            if args.image_topic:
                import numpy as np
                from sensor_msgs.msg import Image
                image_dir = args.output.with_suffix(".images")
                image_dir.mkdir(exist_ok=False)

                def receive_image(message, topic):
                    key = stamp_key(message.header)
                    if key in image_headers[topic]:
                        raise RuntimeError(f"Duplicate image header on {topic}: {key}")
                    if message.encoding not in ("32FC1", "mono8"):
                        raise ValueError("Dense recorder expects 32FC1 or mono8")
                    dtype = np.dtype(">f4" if message.is_bigendian else "<f4") if message.encoding == "32FC1" else np.dtype("u1")
                    if message.step < message.width * dtype.itemsize or len(message.data) != message.step * message.height:
                        raise ValueError("Invalid dense image stride/payload")
                    values = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                                        strides=(message.step, dtype.itemsize))
                    name = hashlib.sha256(json.dumps([topic, key]).encode()).hexdigest()
                    np.save(image_dir / (name + ".npy"), values, allow_pickle=False)
                    (image_dir / (name + ".json")).write_text(json.dumps({
                        "topic": topic, "header": header_dict(message.header), "encoding": message.encoding,
                        "file": name + ".npy"}) + "\n")
                    image_headers[topic].add(key)

                for topic in args.image_topic:
                    node.create_subscription(Image, topic,
                        lambda message, topic=topic: receive_image(message, topic), 100)

            def receive(message):
                nonlocal received, expected_sequence
                result = json.loads(message.data)
                if result["sequence"] != expected_sequence:
                    raise RuntimeError(f"Result sequence gap: expected {expected_sequence}, "
                                       f"received {result['sequence']}")
                handle.write(message.data + "\n")
                if args.image_topic:
                    header = result["header"]
                    result_headers.add((header["frame_id"], header["stamp"]["sec"], header["stamp"]["nanosec"]))
                expected_sequence += 1
                received += 1

            node.create_subscription(String, args.topic, receive, 100)
            while rclpy.ok() and (received < args.count or any(len(headers) < args.count for headers in image_headers.values())):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Received {received}/{args.count} results")
                rclpy.spin_once(node, timeout_sec=0.1)
            if any(headers != result_headers for headers in image_headers.values()):
                raise RuntimeError("Dense image headers do not match result headers")
        print(f"Received {received} consecutive results", flush=True)
        return 0 if received == args.count else 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
