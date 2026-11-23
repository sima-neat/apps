#!/usr/bin/env python3
"""Image -> YOLO26 inference -> detections and the highest-score person crop."""
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time

import pyneat
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from image_transport import image_tensor, header_dict, stamp_key, to_message
from visual_output import VisualOutput
from app_config import read_arguments, image_rgb, positive, positive_int
from box_decode import parse_boxes


class DetectionNode(Node):
    def __init__(self, config):
        super().__init__('detection')
        self.config = config
        ros = config['ros']
        self.labels = Path(config['model']['labels']).read_text().splitlines()
        self.pending_limit = positive_int(ros['pending_frames'], 'ros.pending_frames')
        self.inflight_limit = positive_int(ros['inflight_frames'], 'ros.inflight_frames')
        self.timeout = positive(config['runtime']['timeout_ms'], 'runtime.timeout_ms') / 1000
        self.policy = ros['input_policy']
        if self.policy not in ('complete', 'latest'):
            raise ValueError('ros.input_policy must be complete or latest')
        self.crop_interval = positive(ros['crop_interval_seconds'], 'ros.crop_interval_seconds')
        history = max(16, self.pending_limit + self.inflight_limit)

        # ROS interfaces: all inputs and outputs are declared here.
        self.results = self.create_publisher(String, ros['results_topic'], history)
        self.crops = self.create_publisher(Image, ros['crop_topic'], 1)
        self.statistics = self.create_publisher(String, '~/statistics', 1)
        self.create_subscription(Image, ros['image_topic'], self.on_image, history)
        self.create_timer(self.crop_interval, self.publish_crop)
        self.create_timer(1.0, self.publish_statistics)
        self.visuals = VisualOutput(ros.get('visualization', {}))

        # Two workers overlap submission/completion. Keep each input alive until
        # its matching result is published; do not block the ROS image callback.
        self.pending = Queue(maxsize=self.pending_limit)
        self.slots = threading.Semaphore(self.inflight_limit)
        self.lock = threading.Lock()
        self.inflight = {}
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.model = self.runner = None
        self.error = None
        self.next_id = 1_000_000
        self.received = self.completed = self.dropped = 0
        self.first = self.last = None
        self.latest_person = None
        self.last_crop = None
        self.submitter = threading.Thread(target=self.submit_images, name='detector-submit')
        self.receiver = threading.Thread(target=self.receive_results, name='detector-results')
        self.submitter.start()
        self.receiver.start()

    def initialize_model(self, image):
        # Geometry is known on the first image. This is the native YOLO26 route.
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.NV12
        options.preprocess.input_max_width = image.width
        options.preprocess.input_max_height = image.height
        options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        options.decode_type = pyneat.BoxDecodeType.YoloV26
        options.score_threshold = self.config['inference']['min_score']
        options.nms_iou_threshold = self.config['inference']['nms_iou']
        options.top_k = self.config['inference']['max_detections']
        self.model = pyneat.Model(self.config['model']['path'], options)
        run_options = pyneat.RunOptions()
        run_options.queue_depth = 8
        run_options.overflow_policy = pyneat.OverflowPolicy.Block
        seed = image_tensor(image)
        self.runner = self.model.build([seed], run_options=run_options)
        self.runner.run([seed])  # One native warmup, fully drained before submission.
        self.get_logger().info('Detector initialized; accepting images')

    def on_image(self, image):
        if self.stopping.is_set():
            return
        self.received += 1
        try:
            self.pending.put_nowait(image)
        except Full:
            if self.policy == 'complete':
                self.fail(RuntimeError('Input queue full; reduce ros.source_window'))
            else:
                try:
                    self.pending.get_nowait()
                    self.dropped += 1
                except Empty:
                    pass
                self.pending.put_nowait(image)

    def submit_images(self):
        try:
            while not self.stopping.is_set():
                try:
                    image = self.pending.get(timeout=0.1)
                except Empty:
                    continue
                while not self.slots.acquire(timeout=0.1):
                    if self.stopping.is_set():
                        return
                if self.stopping.is_set():
                    return
                if image.encoding != 'nv12':
                    raise ValueError('Detector input must use packed nv12')
                if self.runner is None:
                    self.initialize_model(image)
                if self.stopping.is_set():
                    return
                tensor = image_tensor(image)
                request_id = self.next_id
                self.next_id += 1
                sample = pyneat.Sample()
                sample.kind, sample.tensor = pyneat.SampleKind.Tensor, tensor
                sample.frame_id = request_id
                sample.pts_ns = image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec
                sample.stream_id = image.header.frame_id
                with self.lock:
                    self.inflight[request_id] = (image, tensor, time.monotonic())
                # Submit inference asynchronously; receive_results() pulls outputs.
                if not self.runner.push([sample]):
                    raise RuntimeError('Detector rejected an admitted image')
                self.ready.set()
        except Exception as error:
            self.fail(error)

    def receive_results(self):
        try:
            while not self.stopping.is_set():
                if not self.ready.wait(timeout=0.1):
                    continue
                # This is where inference completes; timeout is not a result.
                sample = self.runner.pull(timeout_ms=100)
                if not sample:
                    with self.lock:
                        oldest = min((item[2] for item in self.inflight.values()), default=time.monotonic())
                    if time.monotonic() - oldest > self.timeout:
                        raise TimeoutError('Detector inference timed out')
                    continue
                with self.lock:
                    context = self.inflight.get(int(sample.frame_id))
                if context is None:
                    raise RuntimeError(f'Unknown/duplicate inference result: {sample.frame_id}')
                image, tensor, _ = context
                expected_pts = image.header.stamp.sec * 1_000_000_000 + image.header.stamp.nanosec
                if sample.pts_ns != expected_pts:
                    raise RuntimeError('Detector output does not match its source image')
                self.publish_result(sample, image)
                with self.lock:
                    del self.inflight[int(sample.frame_id)]
                self.slots.release()
        except Exception as error:
            self.fail(error)

    def publish_result(self, sample, image):
        boxes = parse_boxes(sample)
        people = [box for box in boxes if self.label(box['class_id']).lower() == 'person']
        person = max(people, key=lambda box: box['score'], default=None)
        with self.lock:
            self.latest_person = (image, person, time.monotonic()) if person else None
        allowed = self.config['inference'].get('classes') or []
        if isinstance(allowed, str):
            allowed = allowed.split(',')
        allowed = {label.strip().lower() for label in allowed}
        objects = []
        for box in boxes:
            label = self.label(box['class_id'])
            if not allowed or label.lower() in allowed:
                objects.append(dict(id=f'obj_{len(objects) + 1}', label=label,
                                    confidence=box['score'], bbox=box['bbox']))
        data = {'objects': objects}
        self.visuals.write(image, data, self.completed)
        result = dict(schema_version=1, sequence=self.completed, header=header_dict(image.header), data=data)
        self.results.publish(String(data=json.dumps(result)))
        self.completed += 1
        self.last = time.monotonic()
        if self.first is None:
            self.first = self.last

    def label(self, class_id):
        return self.labels[class_id] if 0 <= class_id < len(self.labels) else f'class_{class_id}'

    def publish_crop(self):
        if not self.crops.get_subscription_count():
            return
        with self.lock:
            selected = self.latest_person
        if selected is None:
            return
        image, box, received_at = selected
        if time.monotonic() - received_at > self.crop_interval * 2 or stamp_key(image.header) == self.last_crop:
            return
        rgb = image_rgb(image)
        x, y, w, h = box['bbox']
        crop = rgb[max(0, y):min(image.height, y+h), max(0, x):min(image.width, x+w)]
        if not crop.size:
            raise ValueError('Detector returned an empty person crop')
        self.crops.publish(to_message(crop, image.header.stamp, image.header.frame_id, 'rgb8'))
        self.last_crop = stamp_key(image.header)

    def publish_statistics(self):
        elapsed = self.last - self.first if self.completed > 1 else 0
        self.statistics.publish(String(data=json.dumps(dict(
            received=self.received, completed=self.completed, dropped_before_admission=self.dropped,
            completed_output_fps=(self.completed - 1) / elapsed if elapsed else None))))

    def fail(self, error):
        if not self.stopping.is_set():
            self.error = error
            self.get_logger().error(str(error))
            self.stopping.set()

    def close(self):
        self.stopping.set()
        runner = self.runner
        if runner is not None:
            runner.close()
        self.submitter.join()
        self.receiver.join()
        if self.runner is not None and self.runner is not runner:
            self.runner.close()
        self.visuals.close()


def main(argv=None):
    config, ros_args = read_arguments(argv)
    rclpy.init(args=ros_args)
    node = None
    code = 0
    try:
        node = DetectionNode(config)
        while rclpy.ok() and node.error is None:
            rclpy.spin_once(node, timeout_sec=0.1)
        code = 1 if node.error else 0
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
