#!/usr/bin/env python3
from __future__ import annotations
"""RF-DETR: ROS image -> backbone -> native top-K proposals -> transformer."""

import json
import threading
import pyneat
import model_support as helpers
from image_transport import image_tensor

class TwoStageRun:
    """Overlap the two models; retain backbone buffers until their result returns."""
    def __init__(self, backbone, transformer, input_shapes, proposal_count, top_k):
        self.backbone, self.transformer = backbone, transformer
        self.input_shapes = input_shapes
        self.proposal_count, self.top_k = proposal_count, top_k
        self.stop = threading.Event()
        self.error = None
        self.retained = {}
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.bridge, name="rfdetr-proposals")
        self.worker.start()

    def push(self, samples):
        if self.error:
            raise self.error
        return self.backbone.push(samples)

    def bridge(self):
        try:
            while not self.stop.is_set():
                sample = self.backbone.pull(timeout_ms=100)
                if not sample:
                    continue
                feature, scores, proposals = helpers.split_backbone(sample, self.proposal_count)
                gathered = helpers.stable_topk_gather(scores.to_numpy(copy=False),
                                                     proposals.to_numpy(copy=False), self.top_k)
                tensor = pyneat.Tensor.from_numpy(gathered, copy=True, memory=pyneat.TensorMemory.EV74)
                request = pyneat.Sample()
                request.kind = pyneat.SampleKind.TensorSet
                request.tensors = helpers.transformer_inputs(self.input_shapes, feature, tensor, self.top_k)
                helpers.copy_identity(sample, request)
                with self.lock:
                    self.retained[int(sample.frame_id)] = (sample, request)
                if not self.transformer.push_samples(request):
                    raise RuntimeError("RF-DETR transformer rejected input")
        except BaseException as error:
            if not self.stop.is_set():
                self.error = error

    def pull(self, timeout_ms):
        if self.error:
            raise self.error
        output = self.transformer.pull(timeout_ms=timeout_ms)
        if output:
            with self.lock:
                if self.retained.pop(int(output.frame_id), None) is None:
                    raise RuntimeError("RF-DETR transformer returned an unknown frame")
        return output if output else None

    def close(self):
        self.stop.set()
        self.backbone.close()
        self.transformer.close()
        self.worker.join()
        self.retained.clear()
import argparse
from collections import deque
from pathlib import Path
import time
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
import math
from image_transport import header_dict, to_message
from visual_output import VisualOutput
from app_config import json_value


class DetectionNode(Node):
    warmup_runs = 0

    def __init__(self, config, config_path):
        super().__init__('rfdetr_detection_segmentation')
        settings = config.setdefault("ros", {})
        self.initialize_model(config, config_path)
        self.ros_settings = settings
        self.visual_output = VisualOutput(settings.get("visualization", {}),
                                          getattr(self, "render", None))
        self.pending = deque()
        self.inflight = {}
        self.condition = threading.Condition()
        self.stopping = False
        self.error = None
        self.runner = None
        self.next_id = 1_000_000  # Distinct from runtime-generated warmup IDs.
        self.received = self.dropped = self.admitted = self.completed = 0
        self.first = self.last = None
        self.image_publishers = {}
        # DDS must hold the complete configured admission window before the
        # executor takes callbacks, including a simultaneous multi-camera burst.
        history = max(16, settings.get("pending_frames", 8) + settings.get("inflight_frames", 4))
        self.results = self.create_publisher(String, settings.get("results_topic", "results"), history)
        self.statistics = self.create_publisher(String, "~/statistics", 1)
        self.create_subscription(Image, settings.get("image_topic", "image"), self.on_image, history)
        self.create_timer(1.0, self.report)
        self.feeder = threading.Thread(target=self.submit_images, name="neat-submit")
        self.consumer = threading.Thread(target=self.receive_results, name="neat-results")
        self.feeder.start()
        self.consumer.start()

    def initialize_model(self, config, config_path=None):
        helpers.pyneat = pyneat
        self.config = helpers.load_config(config_path)
        self.labels = helpers.load_labels(self.config.labels)
        self.geometry = None

    def prepare_input(self, message):
        if message.encoding != "nv12":
            raise ValueError("RF-DETR expects native nv12 source pixels")
        geometry = (message.width, message.height)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("RF-DETR source geometry changed")
        self.geometry = geometry
        return [image_tensor(message)]

    def build_runner(self, seed):
        self.backbone = helpers.build_backbone(self.config, *self.geometry)
        self.transformer = helpers.build_transformer(self.config)
        self.contract = helpers.model_contract(self.config, self.backbone, self.transformer)
        options = helpers.transformer_run_options()
        # Retain the device-backed feature buffer through transformer completion,
        # as the native backbone graph does. Owned output can wrap a CPU copy in
        # a GstSample, which is unsuitable for the transformer's EV74 input.
        options.output_memory = pyneat.OutputMemory.ZeroCopy
        backbone_run = self.backbone.build(seed, run_options=options)
        try:
            transformer_run = self.transformer.build(helpers.transformer_seed(self.transformer),
                                                     run_options=helpers.transformer_run_options())
        except BaseException:
            backbone_run.close()
            raise
        return TwoStageRun(backbone_run, transformer_run,
                           [tuple(spec.shape) for spec in self.transformer.input_specs()],
                           self.contract.side ** 2, self.contract.top_k)

    def decode_result(self, sample, message):
        _, payload = helpers.result_metadata(self.config, self.contract, sample, message.width, message.height, self.labels)
        return json.loads(payload)

    def on_image(self, message):
        with self.condition:
            if self.stopping:
                return
            self.received += 1
            capacity = int(self.ros_settings.get("pending_frames", 8))
            if len(self.pending) >= capacity:
                if self.ros_settings.get("input_policy", "complete") == "latest":
                    self.pending.popleft()
                    self.dropped += 1
                else:
                    self.fail(RuntimeError("Input queue full in complete mode; reduce source window"))
                    return
            self.pending.append(message)
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            if self.stopping:
                return
            if self.error is None:
                self.error = error
                self.get_logger().error(str(error))
            self.stopping = True
            self.condition.notify_all()

    def submit_images(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or (
                        self.pending and len(self.inflight) < self.ros_settings.get("inflight_frames", 4)))
                    if self.stopping:
                        return
                    message = self.pending.popleft()
                tensors = self.prepare_input(message)
                # Validation/recording can require additional result subscribers.
                # DDS discovery is asynchronous; process startup is not readiness.
                if self.runner is None:
                    deadline = time.monotonic() + self.ros_settings.get("result_timeout_seconds", 30)
                    while self.results.get_subscription_count() < self.ros_settings.get("result_subscribers", 0):
                        with self.condition:
                            if self.stopping:
                                return
                            self.condition.wait(timeout=0.05)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Required result subscribers did not connect")
                    runner = self.build_runner(tensors)
                    with self.condition:
                        self.runner = runner
                    # Match the native app's warmup policy, and drain it fully.
                    # Some explicit tensor graphs have different first-run output.
                    for _ in range(getattr(self, "warmup_runs", 1)):
                        runner.run(tensors)
                    # The build seed belongs to graph initialization. Native
                    # examples prepare their first measured tensor afterwards;
                    # do the same instead of reusing seed storage for a result.
                    tensors = self.prepare_input(message)
                    with self.condition:
                        self.condition.notify_all()
                with self.condition:
                    if self.stopping:
                        return
                    request_id = self.next_id
                    self.next_id += 1
                    self.inflight[request_id] = (message, tensors, time.monotonic())
                    self.condition.notify_all()
                samples = []
                for tensor in tensors:
                    sample = pyneat.Sample()
                    sample.kind = pyneat.SampleKind.Tensor
                    sample.tensor = tensor
                    sample.frame_id = request_id
                    sample.pts_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                    sample.stream_id = message.header.frame_id
                    sample.attributes = {"ros-request-id": str(request_id)}
                    samples.append(sample)
                if not self.runner.push(samples):
                    raise RuntimeError("Neat rejected an admitted frame")
                with self.condition:
                    self.admitted += 1
                    self.condition.notify_all()
        except BaseException as error:
            self.fail(error)

    def receive_results(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or (
                        self.runner is not None and self.inflight))
                    if self.stopping:
                        return
                    # Multi-input routes can initialize internal stages during
                    # their first push. Pull only after an input was accepted,
                    # while still allowing subsequent pushes to overlap pulls.
                    accepted = self.condition.wait_for(
                        lambda: self.stopping or self.admitted > self.completed,
                        timeout=self.ros_settings.get("result_timeout_seconds", 30))
                    if self.stopping:
                        return
                    if not accepted:
                        raise TimeoutError("Timed out submitting an inference input")
                sample = self.runner.pull(timeout_ms=100)
                # Graph Run returns None on timeout; ModelRunner can return an
                # empty Sample. Neither is a completed inference result.
                if not sample:
                    with self.condition:
                        oldest = min((v[2] for v in self.inflight.values()), default=time.monotonic())
                    if time.monotonic() - oldest > self.ros_settings.get("result_timeout_seconds", 30):
                        raise TimeoutError("Timed out waiting for an admitted inference result")
                    continue
                # The installed runner preserves explicit frame_id and pts_ns.
                # Its arbitrary attributes do not survive every model route.
                # Verify both fields; never guess FIFO headers.
                request_id = int(sample.frame_id)
                with self.condition:
                    if request_id not in self.inflight:
                        raise RuntimeError(f"Unknown/duplicate inference result: {request_id}")
                    message, tensors, submitted = self.inflight[request_id]
                expected_pts = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                if sample.pts_ns != expected_pts:
                    raise RuntimeError(f"Result timestamp mismatch for request {request_id}")
                data = self.decode_result(sample, message)
                if self.visual_output.enabled:
                    self.visual_output.write(message, data, self.completed)
                # Dense results use typed Image topics, not JSON float lists.
                for topic, pixels, encoding in data.pop("images", []):
                    publisher = self.image_publishers.get(topic)
                    if publisher is None:
                        publisher = self.create_publisher(Image, topic, 8)
                        self.image_publishers[topic] = publisher
                        deadline = time.monotonic() + self.ros_settings.get("result_timeout_seconds", 30)
                        while publisher.get_subscription_count() < self.ros_settings.get("image_subscribers", 0):
                            with self.condition:
                                if self.stopping:
                                    return
                                self.condition.wait(timeout=0.05)
                            if time.monotonic() > deadline:
                                raise TimeoutError(f"Required image subscribers did not connect to {topic}")
                    publisher.publish(to_message(pixels, message.header.stamp,
                                                 message.header.frame_id, encoding))
                payload = {"schema_version": 1, "sequence": self.completed,
                           "header": header_dict(message.header), "data": data}
                self.results.publish(String(data=json.dumps(payload, default=json_value,
                                                           separators=(",", ":"), allow_nan=False)))
                now = time.perf_counter()
                with self.condition:
                    del self.inflight[request_id]
                    self.completed += 1
                    if self.completed > self.ros_settings.get("warmup_results", 0):
                        if self.first is None:
                            self.first = now
                        self.last = now
                    self.condition.notify_all()
        except BaseException as error:
            self.fail(error)

    def summary(self):
        with self.condition:
            measured = max(0, self.completed - self.ros_settings.get("warmup_results", 0))
            elapsed = self.last - self.first if measured > 1 else 0.0
            return {"received": self.received, "dropped_before_admission": self.dropped,
                    "admitted": self.admitted, "completed": self.completed,
                    "pending": len(self.pending), "inflight": len(self.inflight),
                    "completion_interval_seconds": elapsed,
                    "completed_output_fps": (measured - 1) / elapsed if elapsed else None,
                    "error": str(self.error) if self.error else None}

    def report(self):
        self.statistics.publish(String(data=json.dumps(self.summary(), allow_nan=False)))

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        # A blocked native push is released by closing the runner. Pull is bounded.
        closed_runner = self.runner
        if closed_runner is not None:
            closed_runner.close()
        self.feeder.join()
        self.consumer.join()
        self.visual_output.close()
        if self.runner is not None and self.runner is not closed_runner:
            self.runner.close()
        if hasattr(self, "close_model"):
            self.close_model()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROS rfdetr_detection_segmentation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--visual-insight", choices=("true", "false"))
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    settings = config.setdefault("ros", {})
    if args.stream_index is not None:
        settings["stream_index"] = args.stream_index
        settings.setdefault("visualization", {}).setdefault("insight", {}).setdefault("channel_base", args.stream_index)
    if args.visual_insight is not None:
        settings.setdefault("visualization", {}).setdefault("insight", {})["enabled"] = args.visual_insight == "true"
    for key, default, minimum in (("pending_frames", 8, 1), ("inflight_frames", 4, 1),
                                  ("warmup_results", 0, 0), ("result_subscribers", 0, 0),
                                  ("image_subscribers", 0, 0)):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"ros.{key} must be an integer >= {minimum}")
    timeout = settings.get("result_timeout_seconds", 30)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("ros.result_timeout_seconds must be a finite positive number")
    if settings.get("input_policy", "complete") not in ("complete", "latest"):
        raise ValueError("ros.input_policy must be complete or latest")
    rclpy.init(args=ros_args)
    node = DetectionNode(config, args.config)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    code = 0
    try:
        while rclpy.ok() and not node.error:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        executor.shutdown()
        report = node.summary()
        code = 1 if node.error else 0
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    return code

if __name__ == "__main__":
    raise SystemExit(main())
