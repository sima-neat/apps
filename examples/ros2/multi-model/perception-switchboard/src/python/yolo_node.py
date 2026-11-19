#!/usr/bin/env python3
"""YOLO26 boxes and masks with live enable/overlay parameters. Callbacks admit frames; two workers overlap Neat push/pull."""
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

from image_transport import header_dict, sample_tensors, to_message
from visual_output import image_bgr
from node_tools import read_only, run
from ament_index_python.packages import get_package_share_directory
from yolo_masks import decode_yolo26_segments, frame_rect_for_detection, mask_polygon
from overlays import render_yolo


class YoloNode(Node):
    def __init__(self, **kwargs):
        super().__init__('yolo', **kwargs)
        # Only enabled and overlay_mode may change while running. Model options are startup-only.
        self.declare_parameter('enabled', True)
        self.declare_parameter('overlay_mode', 'both')
        self.declare_parameter('model_path', '', read_only())
        self.declare_parameter('image_topic', 'image', read_only())
        self.declare_parameter('inflight_frames', 2, read_only())
        self.declare_parameter('timeout_seconds', 30.0, read_only())
        self.declare_parameter('publish_preview', True, read_only())
        self.declare_parameter('labels_path', '', read_only())
        self.declare_parameter('max_width', 1920, read_only())
        self.declare_parameter('max_height', 1080, read_only())
        self.declare_parameter('min_score', 0.55, read_only())
        self.declare_parameter('nms_iou', 0.60, read_only())
        self.declare_parameter('max_detections', 50, read_only())
        if not 0 < self.get_parameter('min_score').value <= 1 or not 0 < self.get_parameter('nms_iou').value <= 1:
            raise ValueError('min_score and nms_iou must be in (0, 1]')
        if min(self.get_parameter(n).value for n in ('max_width', 'max_height', 'max_detections')) < 1:
            raise ValueError('Maximum geometry and detection count must be positive')
        self.limit = self.get_parameter('inflight_frames').value
        self.timeout = self.get_parameter('timeout_seconds').value
        if not 1 <= self.limit <= 8 or self.timeout <= 0:
            raise ValueError('inflight_frames must be 1..8 and timeout_seconds positive')
        self.preview_enabled = self.get_parameter('publish_preview').value
        self.enabled = self.get_parameter('enabled').value
        self.overlay_mode = self.get_parameter('overlay_mode').value
        if self.overlay_mode not in ('detection', 'segmentation', 'both'):
            raise ValueError('overlay_mode must be detection, segmentation or both')
        labels = self.get_parameter('labels_path').value
        if not labels:
            labels = str(Path(get_package_share_directory('neat_perception_switchboard')) / 'common/coco_label.txt')
        self.labels = Path(labels).read_text().splitlines()
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
        self.results = self.create_publisher(String, 'yolo/results', 4)
        self.previews = self.create_publisher(Image, 'yolo/preview', 2)
        self.status = self.create_publisher(String, 'yolo/status', 1)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        self.add_on_set_parameters_callback(self.validate_parameters)
        self.add_post_set_parameters_callback(self.apply_parameters)
        self.create_timer(1.0, self.report)
        self.feeder = threading.Thread(target=self.submit_frames, name='yolo-submit')
        self.consumer = threading.Thread(target=self.receive_results, name='yolo-results')
        self.feeder.start()
        self.consumer.start()

    def validate_parameters(self, parameters):
        # Validation has no side effects: a rejected atomic update changes nothing.
        for parameter in parameters:
            if parameter.name == 'enabled' and type(parameter.value) is not bool:
                return SetParametersResult(successful=False, reason='enabled must be boolean')
            if parameter.name == 'overlay_mode' and parameter.value not in ('detection', 'segmentation', 'both'):
                return SetParametersResult(successful=False, reason='overlay_mode must be detection, segmentation or both')
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
                elif parameter.name == 'overlay_mode':
                    self.overlay_mode = parameter.value
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
        bgr = image_bgr(image)
        return [pyneat.Tensor.from_numpy(np.ascontiguousarray(bgr), copy=True,
                    image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74)]

    def initialize_model(self, tensors, image):
        # One segmentation model produces boxes AND masks. Overlay mode never loads another model.
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.input_max_width = self.get_parameter('max_width').value
        options.preprocess.input_max_height = self.get_parameter('max_height').value
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        options.decode_type = pyneat.BoxDecodeType.YoloV26Seg
        options.score_threshold = self.get_parameter('min_score').value
        options.nms_iou_threshold = self.get_parameter('nms_iou').value
        options.top_k = self.get_parameter('max_detections').value
        self.model = pyneat.Model(self.get_parameter('model_path').value, options)
        settings = pyneat.RunOptions()
        settings.queue_depth = max(4, self.limit)
        settings.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(tensors, run_options=settings)

    def decode_result(self, sample, context):
        image = context['image']
        instances = decode_yolo26_segments(sample_tensors(sample), image.width, image.height,
                                           self.get_parameter('max_detections').value)
        objects = []
        for instance in instances:
            rect = frame_rect_for_detection(instance, (image.height, image.width))
            polygon = mask_polygon(instance['mask'], (image.height, image.width), rect, 0.5)
            x, y, right, bottom = rect
            class_id = instance['class_id']
            objects.append(dict(class_id=class_id, label=self.labels[class_id] if 0 <= class_id < len(self.labels) else str(class_id),
                                confidence=instance['score'], bbox=[x, y, right-x, bottom-y], mask=polygon))
        payload = dict(header=header_dict(image.header), generation=context['generation'],
                       overlay_mode=context['overlay_mode'], objects=objects)
        preview = render_yolo(image_bgr(image), objects, context['overlay_mode']) if self.preview_enabled else None
        return String(data=json.dumps(payload, allow_nan=False)), preview


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
                                   submitted=time.monotonic(), overlay_mode=self.overlay_mode)
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
    run(YoloNode)
