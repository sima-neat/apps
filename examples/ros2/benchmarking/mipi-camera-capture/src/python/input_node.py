#!/usr/bin/env python3
"""Shared RTSP, HTTP MJPEG, USB and MIPI source -> ROS Image.

The camera/decode graph uses the same public Neat source options as the native
examples. Decoder-owned planes are packed once before crossing the ROS boundary.
"""
from array import array
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pyneat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
from image_transport import sample_tensors, to_message, stamp_key
import camera_support as helpers


def source_settings(config, index):
    if "camera" in config:
        return "mipi", config["camera"]
    if "profile" in config.get("model", {}) and "source" not in config:
        return "mipi", {"width": 1920, "height": 1080, "fps_num": 30}
    if "streams" in config:
        streams = config["streams"]
        if not 0 <= index < len(streams):
            raise ValueError("Source index is outside the configured streams")
        stream = streams[index]
        stream = {"url": stream} if isinstance(stream, str) else stream
        return "rtsp", {**config.get("input", {}), **stream}
    source = config.get("source", {})
    kind = source.get("type", "rtsp")
    if "rtsp_url" in source:
        return "rtsp", {**source, "url": source["rtsp_url"]}
    if kind in ("rtsp", "http", "usb"):
        return kind, source.get(kind, {})
    if kind == "video_file":
        return kind, {"path": source.get("video_path") or source.get(kind, {}).get("path")}
    raise ValueError(f"Unsupported live source type: {kind}")


def build_source(kind, settings):
    graph = pyneat.Graph("ros_source")
    if kind == "video_file":
        options = pyneat.VideoInputGroupOptions()
        options.path = str(settings["path"])
        graph.add(pyneat.groups.video_input(options))
    elif kind == "mipi":
        options = pyneat.CameraInputOptions()
        options.camera_name = settings.get("name", "")
        options.width = int(settings.get("width", 1920))
        options.height = int(settings.get("height", 1080))
        options.framerate_num = int(settings.get("fps_num", 30))
        options.framerate_den = int(settings.get("fps_den", 1))
        options.format = settings.get("format", "NV12")
        options.buffer_name = "camera0"
        options.queue_depth = int(settings.get("queue_depth", 2))
        options.allow_cpu_fallback = not settings.get("strict_zero_copy", True)
        graph.add(pyneat.nodes.camera_input(options,
                    capture_buffer_count=int(settings.get("capture_buffers", 0))))
    else:
        # These source helpers are shared with the shipped single-stream example.
        helpers.load_runtime_dependencies()
        cfg = SimpleNamespace(
            source_type=kind, source_url=settings.get("url", ""),
            source_codec=settings.get("codec", "mjpeg" if kind in ("usb", "http") else "h264"),
            latency_ms=int(settings.get("latency_ms", 100)), tcp=bool(settings.get("tcp", True)),
            ssl_strict=bool(settings.get("ssl_strict", True)), source_fps=int(settings.get("fps", 0)),
            camera=settings.get("camera", ""), source_width=int(settings.get("width", 1280)),
            source_height=int(settings.get("height", 720)),
        )
        # Multi-camera examples can declare their geometry explicitly. Honor
        # those values instead of opening an extra decoder to probe every URL.
        if kind == "rtsp" and all(int(settings.get(key, 0)) > 0 for key in ("width", "height", "fps")):
            width, height, fps = cfg.source_width, cfg.source_height, cfg.source_fps
        else:
            width, height, fps = helpers.resolve_source_geometry(cfg)
        camera = pyneat.usb_camera_info(helpers.make_usb_camera_options(cfg)) if kind == "usb" else None
        if kind == "rtsp":
            options = helpers.make_rtsp_source_options(cfg, fps, width, height)
            # High-density configurations tune decoder buffers explicitly.
            # Preserve those settings when decoding moves into the source node.
            for key, attribute in (("decoder_buffers", "num_buffers"),
                                   ("decoder_input_buffers", "decoder_input_buffers"),
                                   ("decoder_tuning", "decoder_tuning")):
                if key in settings:
                    setattr(options, attribute, settings[key])
            if "decoder_tuning" in settings:
                options.decoder_memory_opt = settings["decoder_tuning"] in (
                    "low-memory", "throughput-low-latency")
            graph.add(pyneat.groups.rtsp_decoded_input(options))
        else:
            graph.add(helpers.make_source_graph(cfg, fps, width, height, camera))
        if camera is not None:
            graph.add(pyneat.nodes.sima_decode(helpers.make_usb_decode_options(cfg, camera.decoded_format)))
    graph.add(pyneat.nodes.output("frames", pyneat.OutputOptions.every_frame(4)
                                  if kind == "video_file" else pyneat.OutputOptions.latest()))
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 2
    run_options.overflow_policy = (pyneat.OverflowPolicy.Block if kind == "video_file"
                                   else pyneat.OverflowPolicy.KeepLatest)
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph, graph.build(run_options)


def pack_image(tensor, stamp, frame_id, output_encoding):
    width, height = int(tensor.width()), int(tensor.height())
    if tensor.is_nv12() or tensor.image_format() == pyneat.PixelFormat.NV16:
        nv12 = tensor.is_nv12()
        uv_rows = height // 2 if nv12 else height
        packed = np.empty((height + uv_rows, width), np.uint8)
        payload = tensor.copy_payload_bytes()
        planes = {plane.role: plane for plane in tensor.planes}
        for role, rows, row0 in ((pyneat.PlaneRole.Y, height, 0),
                                 (pyneat.PlaneRole.UV, uv_rows, height)):
            plane = planes[role]
            stride = int(plane.strides_bytes[0]) if plane.strides_bytes else width
            offset = int(plane.byte_offset)
            if stride < width or offset < 0 or offset + (rows - 1) * stride + width > len(payload):
                raise ValueError("Decoder plane extends beyond its backing payload")
            packed[row0:row0 + rows] = np.ndarray((rows, width), np.uint8, buffer=payload,
                                                 offset=offset, strides=(stride, 1))
        if output_encoding == "bgr8":
            if nv12:
                bgr = cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12)
            else:
                bgr = cv2.cvtColor(np.dstack([packed[:height], packed[height:]]),
                                   cv2.COLOR_YUV2BGR_YUY2)
            return to_message(bgr, stamp, frame_id)
        if output_encoding not in ("native", "nv12" if nv12 else "nv16"):
            raise ValueError("Requested planar encoding differs from decoder output")
        message = Image()
        message.header.stamp, message.header.frame_id = stamp, frame_id
        message.width, message.height, message.step = width, height, width
        message.encoding = "nv12" if nv12 else "nv16"
        data = array("B")
        data.frombytes(memoryview(packed).cast("B"))
        message.data = data
        return message
    raise ValueError("Live source expected NV12 or NV16 decoder output")


class Source:
    """One camera/file and its bounded set of outstanding image headers."""
    def __init__(self, config, index, settings):
        self.index = index
        self.kind, self.options = source_settings(config, index)
        self.frame_id = self.options.get("id", settings.get("frame_id", f"camera_{index}_optical_frame"))
        self.graph = self.runner = self.video = None
        self.pending = set()
        self.published = 0
        self.finished = False
        self.last_frame = time.monotonic()

    def open(self):
        if self.kind == "video_file":
            self.video = cv2.VideoCapture(str(self.options["path"]))
            if not self.video.isOpened():
                raise ValueError("Cannot open configured video file")
        else:
            self.graph, self.runner = build_source(self.kind, self.options)

    def read(self, node, settings):
        if self.video is not None:
            ok, pixels = self.video.read()
            if not ok:
                self.finished = True
                return None
            return to_message(pixels, node.get_clock().now().to_msg(), self.frame_id)
        # Finite video uses the native app's blocking read: None then means EOF.
        # Live cameras are polled so one slow stream cannot hold up the others.
        sample = self.runner.pull("frames", -1 if self.kind == "video_file" else 0)
        if sample is None:
            if self.kind == "video_file" and not self.runner.last_error():
                self.finished = True
                return None
            if not self.runner.running():
                if self.kind == "video_file" and not self.runner.last_error():
                    self.finished = True
                    return None
                raise RuntimeError(f"Source {self.index} stopped: {self.runner.last_error()}")
            if time.monotonic() - self.last_frame > settings.get("source_timeout_seconds", 30):
                raise TimeoutError(f"Source {self.index} stopped delivering frames")
            return None
        return pack_image(sample_tensors(sample)[0], node.get_clock().now().to_msg(),
                          self.frame_id, settings.get("source_encoding", "native"))

    def close(self):
        if self.runner is not None:
            self.runner.close()
        if self.video is not None:
            self.video.release()


def main(argv=None):
    from rclpy.executors import SingleThreadedExecutor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stream-index", default="0", help="A configured index or all")
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    settings = config.get("ros", {})
    indices = (range(len(config.get("streams", [None]))) if args.stream_index == "all"
               else [int(args.stream_index)])
    sources = [Source(config, index, settings) for index in indices]
    by_frame = {source.frame_id: source for source in sources}
    if not sources or len(by_frame) != len(sources) or any(not frame_id for frame_id in by_frame):
        raise ValueError("Sources require distinct, nonempty camera frame IDs")
    complete = settings.get("input_policy", "complete") == "complete"
    window = int(settings.get("source_window", 4))
    if window < 1 or (complete and window * len(sources) > int(settings.get("pending_frames", 8))):
        raise ValueError("ros.pending_frames must cover source_window times the number of sources")
    limit = int(settings.get("source_frames", config.get("inference", {}).get("frames",
                         config.get("runtime", {}).get("frames", 0))))
    history = max(16, settings.get("pending_frames", 8) + settings.get("inflight_frames", 4))
    rclpy.init(args=ros_args)
    node = Node("neat_sources" if len(sources) > 1 else f"neat_source_{sources[0].index}")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    publisher = node.create_publisher(Image, settings.get("image_topic", "image"), history)
    statistics = node.create_publisher(String, "~/statistics", 1)

    def receive(result):
        header = json.loads(result.data)["header"]
        source = by_frame.get(header["frame_id"])
        if source is None:
            return
        key = (header["frame_id"], header["stamp"]["sec"], header["stamp"]["nanosec"])
        if key not in source.pending:
            raise RuntimeError(f"Unexpected/duplicate result: {key}")
        source.pending.remove(key)

    if complete:
        node.create_subscription(String, settings.get("results_topic", "results"), receive, history)
    start = time.monotonic()
    published = 0
    try:
        while rclpy.ok() and publisher.get_subscription_count() == 0:
            if time.monotonic() - start > settings.get("source_timeout_seconds", 30):
                raise TimeoutError("No image subscriber became ready")
            executor.spin_once(timeout_sec=0.05)
        for source in sources:
            source.open()
        for source in sources:
            source.last_frame = time.monotonic()
        while rclpy.ok():
            progress = False
            for source in sources:
                if source.pending and time.monotonic() - source.last_frame > settings.get("result_timeout_seconds", 60):
                    raise TimeoutError(f"Source {source.index} is waiting for inference acknowledgments")
                if source.finished or (complete and len(source.pending) >= window):
                    continue
                message = source.read(node, settings)
                if message is None:
                    continue
                if complete:
                    key = stamp_key(message.header)
                    if key in source.pending:
                        raise RuntimeError("Source produced duplicate outstanding timestamps")
                    source.pending.add(key)
                publisher.publish(message)
                published += 1
                source.published += 1
                source.last_frame = time.monotonic()
                source.finished = limit > 0 and source.published >= limit
                progress = True
                if published % 100 == 0:
                    statistics.publish(String(data=json.dumps({"published": published,
                        "published_fps": published / (time.monotonic() - start)})))
            if all(source.finished and not source.pending for source in sources):
                break
            executor.spin_once(timeout_sec=0.0 if progress else 0.005)
        print(json.dumps({"published": published,
            "unacknowledged": sum(len(source.pending) for source in sources),
            "streams": [{"frame_id": source.frame_id, "published": source.published} for source in sources]}), flush=True)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for source in sources:
            source.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
