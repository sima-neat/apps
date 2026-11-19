#!/usr/bin/env python3
"""Decode one RTSP feed and publish images with bounded result acknowledgments."""
import json
import time

import cv2
import pyneat
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from image_transport import sample_tensors, stamp_key
from image_transport import pack_image
from app_config import read_arguments, positive_int


class InputNode(Node):
    def __init__(self, config):
        super().__init__('input')
        self.config = config
        ros = config['ros']
        history = max(16, ros['pending_frames'] + ros['inflight_frames'])
        self.images = self.create_publisher(Image, ros['image_topic'], history)
        self.acknowledgments = self.create_subscription(String, ros['results_topic'], self.on_result, history)
        self.pending = set()
        self.window = positive_int(ros['source_window'], 'ros.source_window')
        if self.window > ros['pending_frames']:
            raise ValueError('ros.source_window must not exceed ros.pending_frames')
        self.complete = ros['input_policy'] == 'complete'
        self.published = 0
        self.limit = int(config['inference'].get('frames', 0))
        self.runner = self.graph = None
        self.last_image = time.monotonic()

    def open_camera(self):
        url = self.config['source']['rtsp_url']
        camera = cv2.VideoCapture(url)
        try:
            if not camera.isOpened():
                raise ValueError('Cannot open source.rtsp_url')
            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(round(camera.get(cv2.CAP_PROP_FPS))) or 30
        finally:
            camera.release()
        options = pyneat.RtspDecodedInputOptions()
        options.url, options.payload_type = url, 96
        options.insert_queue = True
        options.auto_caps_from_stream = True
        options.fallback_h264_width, options.fallback_h264_height = width, height
        options.fallback_h264_fps = fps
        options.sima_allocator_type = 2
        options.decoder_raw_output = True
        options.output_caps.enable = True
        options.output_caps.format = pyneat.Format.NV12
        options.output_caps.width, options.output_caps.height, options.output_caps.fps = width, height, fps
        options.output_caps.memory = pyneat.CapsMemory.Any
        self.graph = pyneat.Graph('camera')
        self.graph.add(pyneat.groups.rtsp_decoded_input(options))
        self.graph.add(pyneat.nodes.output('frames', pyneat.OutputOptions.latest()))
        settings = pyneat.RunOptions()
        settings.preset = pyneat.RunPreset.Realtime
        settings.queue_depth = 2
        settings.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        settings.output_memory = pyneat.OutputMemory.ZeroCopy
        self.runner = self.graph.build(settings)
        self.last_image = time.monotonic()

    def on_result(self, message):
        result = json.loads(message.data)
        header = result['header']
        key = (header['frame_id'], header['stamp']['sec'], header['stamp']['nanosec'])
        self.pending.discard(key)

    def publish_next(self):
        if not self.images.get_subscription_count() or not self.acknowledgments.get_publisher_count():
            return
        if self.complete and len(self.pending) >= self.window:
            return
        sample = self.runner.pull('frames', 100)
        if sample is None:
            if time.monotonic() - self.last_image > self.config['ros'].get('source_timeout_seconds', 30):
                raise TimeoutError('RTSP source stopped delivering frames')
            return
        image = pack_image(sample_tensors(sample)[0], self.get_clock().now().to_msg(),
                           self.config['ros'].get('frame_id', 'camera_optical_frame'), 'nv12')
        if self.complete:
            key = stamp_key(image.header)
            if key in self.pending:
                raise RuntimeError('Source clock repeated an outstanding timestamp')
            self.pending.add(key)
        self.images.publish(image)
        self.published += 1
        self.last_image = time.monotonic()

    def close(self):
        if self.runner is not None:
            self.runner.close()


def main(argv=None):
    config, ros_args = read_arguments(argv)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = InputNode(config)
        node.open_camera()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.001)
            if node.limit and node.published >= node.limit:
                if not node.pending:
                    break
            else:
                node.publish_next()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
