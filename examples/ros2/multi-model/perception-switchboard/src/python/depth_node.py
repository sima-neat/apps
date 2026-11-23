#!/usr/bin/env python3
"""Depth Anything V2 with a live enable parameter. Callbacks admit frames; two workers overlap Neat push/pull."""
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pyneat
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from image_transport import header_dict, image_array, sample_tensors, to_message
from visual_output import image_bgr
from overlays import render_depth
from node_tools import read_only, run



class DepthNode(Node):
    def __init__(self, **kwargs):
        super().__init__('depth', **kwargs)
        # Only enabled may change while running. Model options are startup-only.
        self.declare_parameter('enabled', False)

        self.declare_parameter('model_path', '', read_only())
        self.declare_parameter('image_topic', 'image', read_only())
        self.declare_parameter('inflight_frames', 2, read_only())
        self.declare_parameter('timeout_seconds', 30.0, read_only())
        self.declare_parameter('publish_preview', True, read_only())
        self.declare_parameter('preview_mode', 'depth', read_only())
        self.declare_parameter('infer_size', 518, read_only())
        self.declare_parameter('max_input_size', 1920, read_only())
        self.limit = self.get_parameter('inflight_frames').value
        self.timeout = self.get_parameter('timeout_seconds').value
        if not 1 <= self.limit <= 8 or self.timeout <= 0:
            raise ValueError('inflight_frames must be 1..8 and timeout_seconds positive')
        self.preview_enabled = self.get_parameter('publish_preview').value
        self.preview_mode = self.get_parameter('preview_mode').value
        if self.preview_mode not in ('depth', 'side_by_side'):
            raise ValueError('preview_mode must be depth or side_by_side')
        self.enabled = self.get_parameter('enabled').value
        self.size = self.get_parameter('infer_size').value
        self.max_input_size = self.get_parameter('max_input_size').value
        if min(self.size, self.max_input_size) < 1:
            raise ValueError('infer_size and max_input_size must be positive')
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.error = None
        self.pending = None  # One latest unadmitted image; never an unbounded queue.
        self.inflight = {}
        self.runner = self.model = None
        self.generation = 0
        self.next_id = 1_000_000
        self.received = self.replaced = self.completed = self.discarded = 0
        self.published = 0
        # All ROS interfaces are declared here. Each result keeps its source header.
        self.results = self.create_publisher(Image, 'depth/image', 4)
        self.previews = self.create_publisher(Image, 'depth/preview', 2)
        self.status = self.create_publisher(String, 'depth/status', 1)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        self.add_on_set_parameters_callback(self.validate_parameters)
        self.add_post_set_parameters_callback(self.apply_parameters)
        self.create_timer(1.0, self.report)
        self.feeder = threading.Thread(target=self.submit_frames, name='depth-submit')
        self.consumer = threading.Thread(target=self.receive_results, name='depth-results')
        self.feeder.start()
        self.consumer.start()

    def validate_parameters(self, parameters):
        # Validation has no side effects: a rejected atomic update changes nothing.
        for parameter in parameters:
            if parameter.name == 'enabled' and type(parameter.value) is not bool:
                return SetParametersResult(successful=False, reason='enabled must be boolean')

        return SetParametersResult(successful=True)

    def apply_parameters(self, parameters):
        # Runs only after ROS commits the update. Generation invalidates old results.
        with self.condition:
            for parameter in parameters:
                if parameter.name == 'enabled' and parameter.value != self.enabled:
                    self.enabled = parameter.value
                    self.generation += 1
                    if self.pending is not None:
                        self.replaced += 1
                    self.pending = None

            self.condition.notify_all()

    def on_image(self, message):
        with self.condition:
            if not self.enabled or self.stopping.is_set():
                return
            self.received += 1
            if self.pending is not None:
                self.replaced += 1
            self.pending = message
            self.condition.notify_all()

    def prepare_input(self, image):
        # Read BGR directly from the ROS buffer; the Neat tensor still owns its copy.
        bgr = image_array(image) if image.encoding == "bgr8" else image_bgr(image)
        scale = self.max_input_size / max(bgr.shape[:2])
        if scale < 1:
            size = (max(1, int(bgr.shape[1] * scale)), max(1, int(bgr.shape[0] * scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return [pyneat.Tensor.from_numpy(np.ascontiguousarray(rgb), copy=True,
                                        image_format=pyneat.PixelFormat.RGB)]

    def initialize_model(self, tensors, image):
        # EV74 bicubic resize and ImageNet normalization required by this model.
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.preset = pyneat.NormalizePreset.ImageNet
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
        options.preprocess.input_max_width = options.preprocess.input_max_height = self.max_input_size
        options.preprocess.input_max_depth = 3
        options.preprocess.resize.enable = pyneat.AutoFlag.On
        options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
        options.preprocess.resize.width = options.preprocess.resize.height = self.size
        options.preprocess.resize.scaling_type = 'BICUBIC'
        self.model = pyneat.Model(self.get_parameter('model_path').value, options)
        settings = pyneat.RunOptions()
        settings.queue_depth = max(4, self.limit)
        settings.overflow_policy = pyneat.OverflowPolicy.Block
        dummy = pyneat.Tensor.from_numpy(np.zeros((self.size, self.size, 3), np.uint8),
                                        copy=True, image_format=pyneat.PixelFormat.RGB)
        return self.model.build([dummy], route_options=pyneat.ModelRouteOptions(), run_options=settings)

    def decode_result(self, sample, context):
        image = context['image']
        tensor = sample_tensors(sample)[0]
        spatial = [int(d) for d in tensor.shape if d > 1]
        if not tensor.is_dense() or len(spatial) < 2:
            raise ValueError('Expected a dense spatial depth tensor')
        height, width = spatial[:2]
        values = np.asarray(tensor.to_numpy(copy=True)).reshape(-1)
        depth = np.ascontiguousarray(values[:height*width].reshape(width, height).T, dtype=np.float32)
        preview = None
        if self.preview_enabled:
            preview = render_depth(image_bgr(image), depth, self.preview_mode)
        return to_message(depth, image.header.stamp, image.header.frame_id, '32FC1'), preview


    def submit_frames(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping.is_set() or
                        (self.enabled and self.pending is not None and len(self.inflight) < self.limit))
                    if self.stopping.is_set():
                        return
                    image, self.pending = self.pending, None
                    request_id = self.next_id
                    self.next_id += 1
                    context = dict(image=image, generation=self.generation, accepted=False,
                                   submitted=time.monotonic(), )
                    self.inflight[request_id] = context
                # Model setup/CPU preparation happen outside callbacks and locks.
                tensors = self.prepare_input(image)
                if self.runner is None:
                    self.runner = self.initialize_model(tensors, image)
                context['tensors'] = tensors  # Own every backing tensor until completion.
                with self.condition:
                    if self.stopping.is_set() or context['generation'] != self.generation:
                        del self.inflight[request_id]
                        self.discarded += 1
                        self.condition.notify_all()
                        continue
                samples = []
                for tensor in tensors:
                    sample = pyneat.Sample()
                    sample.kind, sample.tensor = pyneat.SampleKind.Tensor, tensor
                    sample.frame_id = request_id
                    sample.stream_id = image.header.frame_id
                    sample.pts_ns = image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec
                    samples.append(sample)
                # Admission was reserved before a possible disable; that work drains.
                if not self.runner.push(samples):
                    raise RuntimeError('Neat rejected an admitted frame')
                with self.condition:
                    context['accepted'] = True
                    context['submitted'] = time.monotonic()
                    self.condition.notify_all()
        except Exception as error:
            self.fail(error)

    def receive_results(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping.is_set() or
                        any(c['accepted'] for c in self.inflight.values()))
                    if self.stopping.is_set():
                        return
                sample = self.runner.pull(timeout_ms=100)
                if not sample:
                    with self.condition:
                        oldest = min((c['submitted'] for c in self.inflight.values() if c['accepted']),
                                     default=time.monotonic())
                    if time.monotonic() - oldest > self.timeout:
                        raise TimeoutError('Inference result timed out')
                    continue
                with self.condition:
                    context = self.inflight.get(int(sample.frame_id))
                    if context is None:
                        raise RuntimeError('Unknown or duplicate inference result')
                image = context['image']
                pts = image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec
                if sample.pts_ns != pts:
                    raise RuntimeError('Inference/source timestamp mismatch')
                result, preview = self.decode_result(sample, context)
                # Serialize the last check AND publication with mode changes.
                with self.condition:
                    self.completed += 1
                    if not self.stopping.is_set() and self.enabled and context['generation'] == self.generation:
                        self.results.publish(result)
                        if preview is not None:
                            self.previews.publish(to_message(preview, image.header.stamp, image.header.frame_id))
                        self.published += 1
                    else:
                        self.discarded += 1
                    del self.inflight[int(sample.frame_id)]
                    self.condition.notify_all()
        except Exception as error:
            self.fail(error)

    def report(self):
        with self.condition:
            state = 'active' if self.enabled else ('draining' if self.inflight else 'standby')
            value = dict(state=state, enabled=self.enabled, generation=self.generation,
                         model_loaded=self.runner is not None, received=self.received,
                         replaced_before_admission=self.replaced, completed=self.completed,
                         published=self.published, discarded=self.discarded,
                         pending=int(self.pending is not None), inflight=len(self.inflight), error=self.error)
        self.status.publish(String(data=json.dumps(value)))

    def fail(self, error):
        with self.condition:
            if not self.stopping.is_set():
                self.error = str(error)
                self.get_logger().error(self.error)
            self.stopping.set()
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping.set()
            self.pending = None
            self.condition.notify_all()
        # Close releases an admitted native push. Join before destroying ROS objects.
        closed_runner = self.runner
        if closed_runner is not None:
            closed_runner.close()
        self.feeder.join()
        self.consumer.join()
        # A model may have finished building while shutdown was requested.
        if self.runner is not None and self.runner is not closed_runner:
            self.runner.close()


if __name__ == '__main__':
    run(DepthNode)
