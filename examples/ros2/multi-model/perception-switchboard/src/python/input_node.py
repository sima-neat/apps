#!/usr/bin/env python3
"""One image publisher for all models. Folder replay or Neat RTSP decoding."""
from pathlib import Path
import threading
import time

import cv2
import pyneat
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from image_transport import to_message, sample_tensors
from camera_image import pack_image
from node_tools import read_only, run


class InputNode(Node):
    def __init__(self, **kwargs):
        super().__init__('input', **kwargs)
        self.declare_parameter('kind', 'folder', read_only())
        self.declare_parameter('path', '', read_only())
        self.declare_parameter('loop', True, read_only())
        self.declare_parameter('fps', 15.0, read_only())
        self.declare_parameter('rtsp_url', '', read_only())
        self.declare_parameter('width', 1280, read_only())
        self.declare_parameter('height', 720, read_only())
        self.declare_parameter('frame_id', 'camera', read_only())
        self.declare_parameter('image_topic', 'image', read_only())
        self.kind = self.get_parameter('kind').value
        self.fps = self.get_parameter('fps').value
        if self.kind not in ('folder', 'rtsp') or self.fps <= 0:
            raise ValueError('kind must be folder or rtsp; fps must be positive')
        self.images = self.create_publisher(Image, self.get_parameter('image_topic').value,
                                            qos_profile_sensor_data)
        self.stopping = threading.Event()
        self.error = None
        self.runner = self.graph = None
        self.worker = threading.Thread(target=self.produce, name='input-reader')
        self.worker.start()

    def open_camera(self):
        options = pyneat.RtspDecodedInputOptions()
        options.url = self.get_parameter('rtsp_url').value
        if not options.url:
            raise ValueError('Set rtsp_url for the rtsp input')
        width, height = self.get_parameter('width').value, self.get_parameter('height').value
        options.payload_type = 96
        options.insert_queue = True
        options.auto_caps_from_stream = True
        options.fallback_h264_width, options.fallback_h264_height = width, height
        # Leave decoder FPS unspecified: the H.264 stream supplies it.
        options.sima_allocator_type = 2
        options.decoder_raw_output = True
        options.output_caps.enable = True
        options.output_caps.format = pyneat.Format.NV12
        # Keep the stream's actual rate. A caps filter cannot resize
        # or turn a 25 FPS source into 30 FPS; fps only caps ROS publication.
        options.output_caps.memory = pyneat.CapsMemory.Any
        self.graph = pyneat.Graph('switchboard-camera')
        self.graph.add(pyneat.groups.rtsp_decoded_input(options))
        self.graph.add(pyneat.nodes.output('frames', pyneat.OutputOptions.latest()))
        settings = pyneat.RunOptions()
        settings.preset = pyneat.RunPreset.Realtime
        settings.queue_depth = 2
        settings.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        settings.output_memory = pyneat.OutputMemory.ZeroCopy
        self.runner = self.graph.build(settings)

    def produce(self):
        try:
            paths = []
            if self.kind == 'folder':
                path = Path(self.get_parameter('path').value)
                paths = [path] if path.is_file() else sorted(p for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
                if not paths:
                    raise ValueError('No images found in path')
            else:
                self.open_camera()
            index = 0
            last_frame = time.monotonic()
            while not self.stopping.is_set():
                if not self.images.get_subscription_count():
                    self.stopping.wait(0.1)
                    continue
                started = time.monotonic()
                stamp = self.get_clock().now().to_msg()
                frame_id = self.get_parameter('frame_id').value
                if self.kind == 'folder':
                    frame = cv2.imread(str(paths[index % len(paths)]), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f'Cannot read {paths[index % len(paths)]}')
                    image = to_message(frame, stamp, frame_id)
                else:
                    sample = self.runner.pull('frames', 100)
                    if not sample:
                        if time.monotonic() - last_frame > 30:
                            raise TimeoutError('RTSP source stopped delivering frames')
                        continue
                    image = pack_image(sample_tensors(sample)[0], stamp, frame_id, 'bgr8')
                self.images.publish(image)
                last_frame = time.monotonic()
                index += 1
                if paths and index == len(paths) and not self.get_parameter('loop').value:
                    self.get_logger().info('Folder finished; nodes remain available for inspection')
                    return
                self.stopping.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        except Exception as error:
            self.error = str(error)
            self.get_logger().error(self.error)
            self.stopping.set()

    def close(self):
        self.stopping.set()
        if self.runner is not None:
            self.runner.close()
        self.worker.join()
        if self.runner is not None:
            self.runner.close()


if __name__ == '__main__':
    run(InputNode)
