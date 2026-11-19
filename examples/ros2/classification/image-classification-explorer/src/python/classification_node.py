#!/usr/bin/env python3
from __future__ import annotations
"""Classify each ROS image with every configured native model profile."""

import cv2
import numpy as np
import pyneat
import model_support as helpers
from image_transport import image_array, sample_tensors

class ModelRunners:
    """One runner per profile; join only outputs carrying the same frame ID."""
    def __init__(self, runners):
        self.runners = runners
        self.outputs = [None] * len(runners)

    def run(self, tensors):
        for runner, tensor in zip(self.runners, tensors):
            runner.run([tensor])

    def push(self, samples):
        return all(runner.push([sample]) for runner, sample in zip(self.runners, samples))

    def pull(self, timeout_ms):
        for index, runner in enumerate(self.runners):
            if not self.outputs[index]:
                self.outputs[index] = runner.pull(timeout_ms=timeout_ms)
            if not self.outputs[index]:
                return None
        first = self.outputs[0]
        if any((output.frame_id, output.pts_ns) != (first.frame_id, first.pts_ns)
               for output in self.outputs):
            raise RuntimeError("Classification model outputs disagree on frame identity")
        result = pyneat.Sample()
        result.kind = pyneat.SampleKind.Bundle
        result.frame_id, result.pts_ns = first.frame_id, first.pts_ns
        result.fields = self.outputs
        self.outputs = [None] * len(self.runners)
        return result

    def close(self):
        for runner in self.runners:
            runner.close()
import argparse
from collections import deque
from pathlib import Path
import threading
import time
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
import json
import math
from image_transport import header_dict, to_message
from visual_output import VisualOutput
from app_config import json_value


class ClassificationNode(Node):
    warmup_runs = 0  # build() warms every profile with the native blank image.

    def __init__(self, config, config_path):
        super().__init__('image_classification_explorer')
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
        helpers.CONFIG_DIRECTORY = config_path.resolve().parent
        self.profiles = helpers.load_profiles(config)
        if not self.profiles:
            raise ValueError("At least one classification profile is required")
        for profile in self.profiles:
            profile.labels = helpers.load_label_map(profile.label_map, profile.num_classes)
        self.models = [helpers.build_model(profile) for profile in self.profiles]

    def prepare_input(self, message):
        if message.encoding != "bgr8":
            raise ValueError("Classification input must be bgr8")
        bgr = image_array(message)
        tensors = []
        for profile in self.profiles:
            resized = cv2.resize(bgr, (profile.input_width, profile.input_height),
                                  interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(pyneat.Tensor.from_numpy(rgb, copy=True,
                                                    image_format=pyneat.PixelFormat.RGB))
        return tensors

    def build_runner(self, seed):
        runners = []
        try:
            for model, profile, tensor in zip(self.models, self.profiles, seed):
                options = pyneat.RunOptions()
                options.queue_depth = 8
                options.overflow_policy = pyneat.OverflowPolicy.Block
                runner = model.build([tensor], run_options=options)
                runners.append(runner)
                blank = np.zeros((profile.input_height, profile.input_width, 3), np.uint8)
                runner.run([pyneat.Tensor.from_numpy(blank, copy=True,
                                                     image_format=pyneat.PixelFormat.RGB)])
            return ModelRunners(runners)
        except BaseException:
            for runner in runners:
                runner.close()
            raise

    def decode_result(self, sample, message):
        predictions = {}
        for profile, output in zip(self.profiles, sample.fields):
            scores = helpers.tensor_to_numpy_dense(sample_tensors(output)[0]).flatten().astype(np.float32)
            if len(scores) < profile.num_classes or not np.isfinite(scores).all():
                raise ValueError("Invalid classification scores")
            scores = scores[:profile.num_classes]
            probabilities = helpers.softmax(scores)
            indices = np.lexsort((np.arange(scores.size), -scores))[:profile.top_k]
            predictions[profile.name] = [(int(i), profile.labels[int(i)], float(probabilities[i]))
                                         for i in indices]
        return {"predictions": predictions}

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
    parser = argparse.ArgumentParser(description="ROS image_classification_explorer")
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
    node = ClassificationNode(config, args.config)
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
