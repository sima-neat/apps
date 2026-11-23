#!/usr/bin/env python3
from __future__ import annotations
"""Native live-camera graph, or a shared detector fed by ROS image topics."""

import argparse
import json
from pathlib import Path
import time
import pyneat
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import yaml
import model_support as helpers


class DetectorRun:
    """Name the graph ports explicitly; the shared node owns admission bounds."""
    def __init__(self, runner):
        self.runner = runner

    def push(self, samples):
        for sample in samples:
            sample.port_name = "detector_frame"
        return self.runner.push(samples)

    def pull(self, timeout_ms):
        return self.runner.pull("detections", timeout_ms)

    def close(self):
        self.runner.close()
from collections import deque
import threading
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from sensor_msgs.msg import Image
import math
from image_transport import header_dict, to_message
from visual_output import VisualOutput
from app_config import json_value


class DetectionNode(Node):
    warmup_runs = 0

    def __init__(self, config, config_path):
        super().__init__('high_density_multi_stream_object_detector')
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
        helpers.load_runtime_dependencies()
        self.config = helpers.load_app_config(config_path)
        self.labels = helpers.load_labels(self.config.labels_path)
        self.geometry = None

    def prepare_input(self, message):
        from image_transport import image_tensor
        if message.encoding != "nv12":
            raise ValueError("High-density inference expects nv12")
        geometry = (message.width, message.height)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("All high-density streams must use the same geometry")
        self.geometry = geometry
        return [image_tensor(message)]

    def build_runner(self, seed):
        self.model, detector = helpers.build_detector_graph(self.config)
        self.graph = pyneat.Graph(options=helpers.graph_options(self.config.internal_queue_depth))
        self.graph.add(detector)
        options = helpers.realtime_options(self.config.queue_depth, pyneat.OverflowPolicy.Block)
        return DetectorRun(self.graph.build(options))

    def decode_result(self, sample, message):
        boxes = helpers.parse_boxes_strict(helpers.extract_bbox_payload(sample), message.width,
                                         message.height, self.config.max_detections)
        return {"objects": helpers.build_metadata_boxes(boxes, self.labels, message.width, message.height)}

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


def run_images(argv=None):
    parser = argparse.ArgumentParser(description="ROS high_density_multi_stream_object_detector")
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

def live_result(sample, source, config, sequence):
    """Keep the decoder's stream/frame identity and presentation timestamp."""
    if sample.pts_ns < 0 or sample.frame_id < 0:
        raise ValueError("Live detection is missing its source timestamp or frame ID")
    boxes = helpers.parse_boxes_strict(helpers.extract_bbox_payload(sample), source.frame_w,
                                     source.frame_h, config.max_detections)
    seconds, nanoseconds = divmod(int(sample.pts_ns), 1_000_000_000)
    return {"schema_version": 1, "sequence": sequence,
            "header": {"frame_id": f"camera_{source.index}_optical_frame",
                       "stamp": {"sec": seconds, "nanosec": nanoseconds}},
            "source_frame_id": int(sample.frame_id), "stream_id": str(sample.stream_id),
            "data": {"objects": helpers.build_metadata_boxes(
                boxes, source.labels, source.frame_w, source.frame_h)}}

def run_live(config_path, ros_args, summary_path=None):
    """Decode and infer in one native graph; only results cross the ROS boundary."""
    helpers.load_runtime_dependencies()
    config = helpers.load_app_config(config_path)
    settings = (yaml.safe_load(config_path.read_text()) or {}).get("ros", {})
    labels = helpers.load_labels(config.labels_path)
    rclpy.init(args=ros_args)
    node = Node("high_density_multi_stream_object_detector")
    results = node.create_publisher(String, settings.get("results_topic", "results"), 64)
    app = None
    counts = [0] * len(config.rtsp_urls)
    first = last = None
    completed = 0
    try:
        deadline = time.monotonic() + settings.get("result_timeout_seconds", 30)
        while results.get_subscription_count() < settings.get("result_subscribers", 0):
            if not rclpy.ok():
                return 0
            if time.monotonic() > deadline:
                raise TimeoutError("Required result subscribers did not connect")
            rclpy.spin_once(node, timeout_sec=0.05)

        # Reuse the native graph topology, decoder tuning and realtime admission.
        model, detector = helpers.build_detector_graph(config)
        app = helpers.AppRuntime(model, pyneat.Graph(options=helpers.graph_options(
            config.internal_queue_depth)), None, [])
        for index, url in enumerate(config.rtsp_urls):
            fps, width, height = helpers.resolve_source_geometry(config, url)
            source = helpers.SourceRuntime(index, url, None, labels, width, height, fps)
            app.sources.append(source)
            helpers.connect_source_graph(app, config, source, detector)
        helpers.validate_worker_pool_geometry(app.sources)
        app.run = app.graph.build(helpers.realtime_options(config.queue_depth))

        last_seen = [time.monotonic()] * len(app.sources)
        next_check = time.monotonic() + 1
        while rclpy.ok():
            did_work = False
            for _ in range(helpers.MAX_DETECTION_PULLS_PER_ROUND):
                sample = app.run.pull("detections", 0)
                if sample is None:
                    error = app.run.last_error()
                    if error or not app.run.running():
                        raise RuntimeError(f"Live detector stopped: {error}")
                    break
                did_work = True
                index = helpers.stream_index_from_detection(sample, len(app.sources))
                payload = live_result(sample, app.sources[index], config, completed)
                results.publish(String(data=json.dumps(payload, separators=(",", ":"), allow_nan=False)))
                now = time.monotonic()
                last_seen[index] = now
                counts[index] += 1
                completed += 1
                first = now if first is None else first
                last = now
            now = time.monotonic()
            if now >= next_check:
                for index, seen in enumerate(last_seen):
                    if now - seen > settings.get("source_timeout_seconds", 30):
                        raise TimeoutError(f"No detections received from stream {index}")
                next_check = now + 1
            # Match native polling: only sleep after a round with no results.
            if not did_work:
                time.sleep(0.001)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if app is not None and app.run is not None:
            app.run.close()
        summary = {"completed": completed, "streams": counts,
                   "completion_interval_seconds": last - first if first is not None else 0}
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-graph", action="store_true", help="Keep RTSP decoding and inference in one graph")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args, ros_args = parser.parse_known_args(argv)
    if args.live_graph:
        return run_live(args.config, ros_args, args.summary)
    return run_images(argv)

if __name__ == "__main__":
    raise SystemExit(main())
