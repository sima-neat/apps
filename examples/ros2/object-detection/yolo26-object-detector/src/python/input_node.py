#!/usr/bin/env python3
"""Folder source with bounded, acknowledged image delivery.

Images retain their original size and BGR pixels. Complete mode keeps at most
``window`` images outstanding and advances when matching results arrive.
It overlaps inference instead of waiting after every single image.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import time
import tempfile
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
from image_transport import to_message, stamp_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    settings = config.get("ros", {})
    io = config.get("io", {})
    location = io.get("input_dir", io.get("input", config.get("source", {}).get("image_dir")))
    if not location:
        raise ValueError("Folder source requires io.input_dir, io.input or source.image_dir")
    else:
        directory = Path(location)
        paths = [directory] if directory.is_file() else sorted(p for p in directory.iterdir() if p.is_file() and
                   p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError(f"No images in {directory}")
    # In DETR and RetinaFace num_runs controls their separate profiling mode,
    # whereas YOLO26 and PCB repeat the normal folder loop with this setting.
    native_repeats = config.get("runtime", {}).get("num_runs", 1)
    repeats = int(settings.get("source_repeats", native_repeats))
    if repeats < 1:
        raise ValueError("Source repeat count must be positive")
    paths *= repeats
    window = int(settings.get("source_window", 4))
    if window < 1 or window > int(settings.get("pending_frames", 8)):
        raise ValueError("ros.source_window must be positive and <= ros.pending_frames")
    rclpy.init(args=ros_args)
    node = Node("neat_folder_source")
    history = max(16, settings.get("pending_frames", 8) + settings.get("inflight_frames", 4))
    publisher = node.create_publisher(Image, settings.get("image_topic", "image"), history)
    pending = {}
    received = []
    next_index = 0
    started = time.monotonic()
    last_progress = started
    manifest = []
    # Decode only the next image while the current image is packed/published.
    # One worker preserves order and bounds prefetch storage to one image.
    reader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-read")
    next_image = None

    def receive(message):
        nonlocal last_progress
        result = json.loads(message.data)
        header = result["header"]
        if header["frame_id"] != settings.get("frame_id", "camera_optical_frame"):
            return
        key = (header["frame_id"], header["stamp"]["sec"], header["stamp"]["nanosec"])
        if key not in pending:
            raise RuntimeError(f"Unexpected/duplicate result header: {key}")
        index = pending.pop(key)
        received.append(index)
        last_progress = time.monotonic()

    subscription = node.create_subscription(String, settings.get("results_topic", "results"), receive, history)
    try:
        while rclpy.ok() and len(received) < len(paths):
            if time.monotonic() - last_progress > args.timeout:
                raise TimeoutError(f"Source received {len(received)}/{len(paths)} outputs")
            connected = publisher.get_subscription_count() > 0 and subscription.get_publisher_count() > 0
            if connected and next_index < len(paths) and len(pending) < window:
                path = paths[next_index]
                if next_image is None:
                    next_image = reader.submit(cv2.imread, str(path), cv2.IMREAD_COLOR)
                pixels = next_image.result()
                next_image = (reader.submit(cv2.imread, str(paths[next_index + 1]), cv2.IMREAD_COLOR)
                              if next_index + 1 < len(paths) else None)
                if pixels is None:
                    raise ValueError(f"Cannot decode {path}")
                stamp = node.get_clock().now().to_msg()
                message = to_message(pixels, stamp, settings.get("frame_id", "camera_optical_frame"))
                key = stamp_key(message.header)
                if key in pending:
                    raise RuntimeError("Source clock produced duplicate outstanding timestamps")
                pending[key] = next_index
                manifest.append({"index": next_index, "path": str(path),
                                 "header": {"frame_id": key[0], "stamp": {"sec": key[1], "nanosec": key[2]}}})
                publisher.publish(message)
                next_index += 1
            else:
                rclpy.spin_once(node, timeout_sec=0.01)
        if len(received) != len(paths):
            raise RuntimeError("Source stopped before all images completed")
        print(json.dumps({"published": next_index, "received": len(received),
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        return 0
    finally:
        reader.shutdown(wait=True, cancel_futures=True)
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
