#!/usr/bin/env python3
"""Save or send already-matched previews to separate Insight channels; no inference."""
import threading
from pathlib import Path

from rclpy.node import Node
from sensor_msgs.msg import Image
from visual_output import VisualOutput
from node_tools import read_only, run


class OutputNode(Node):
    def __init__(self, **kwargs):
        super().__init__('output', **kwargs)
        self.declare_parameter('insight.enabled', False, read_only())
        self.declare_parameter('insight.host', '', read_only())
        self.declare_parameter('insight.video_port_base', 9000, read_only())
        self.declare_parameter('insight.channel_base', 0, read_only())
        self.declare_parameter('insight.fps', 30, read_only())
        self.declare_parameter('save.directory', '', read_only())
        self.stopping = threading.Event()
        self.error = None
        self.condition = threading.Condition()
        self.pending = {}  # At most one preview per stream, regardless of camera rate.
        self.outputs = {}
        self.sequence = dict(yolo=0, depth=0, vlm=0)
        directory = self.get_parameter('save.directory').value
        for index, name in enumerate(self.sequence):
            settings = {'max_streams': 1,
                'save': {'enabled': bool(directory), 'directory': str(Path(directory) / name)},
                'insight': {'enabled': self.get_parameter('insight.enabled').value,
                    'host': self.get_parameter('insight.host').value,
                    'video_port_base': self.get_parameter('insight.video_port_base').value,
                    'channel_base': self.get_parameter('insight.channel_base').value + index,
                    'fps': self.get_parameter('insight.fps').value}}
            self.outputs[name] = VisualOutput(settings)
            if settings['insight']['enabled']:
                channel = settings['insight']['channel_base']
                self.get_logger().info(f'Insight {name}: channel {channel}')
            self.create_subscription(Image, name + '/preview',
                lambda image, stream=name: self.on_image(stream, image), 2)
        self.worker = threading.Thread(target=self.write_previews, name='output-writer')
        self.worker.start()

    def on_image(self, name, image):
        if self.outputs[name].enabled:
            with self.condition:
                self.pending[name] = image
                self.condition.notify()

    def write_previews(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.stopping.is_set())
                    if self.stopping.is_set():
                        return
                    batch, self.pending = self.pending, {}
                for name, image in batch.items():
                    self.outputs[name].write(image, {}, self.sequence[name])
                    self.sequence[name] += 1
        except Exception as error:
            self.error = str(error)
            self.get_logger().error(self.error)
            self.stopping.set()

    def close(self):
        with self.condition:
            self.stopping.set()
            self.condition.notify_all()
        self.worker.join()
        for output in self.outputs.values():
            output.close()


if __name__ == '__main__':
    run(OutputNode)
