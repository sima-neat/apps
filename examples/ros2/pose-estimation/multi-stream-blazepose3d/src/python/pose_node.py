#!/usr/bin/env python3
"""ROS images and results around the unchanged native BlazePose scheduler.

The native workers own latest-only detector/pose/publication mailboxes, model
admission, ROI tracking and smoothing. Only camera input and metadata output
are adapted here; no second inference scheduler is maintained.
"""
import argparse
import json
import math
from pathlib import Path
import threading
import time
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import String
import yaml
import model_support as helpers
from image_transport import image_tensor
from visual_output import VisualOutput, positive_int


class ImageInput:
    """One latest ROS image per camera, pulled by the native source worker."""
    def __init__(self, stream, timeout):
        self.stream, self.timeout = stream, timeout
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.last_stamp = -1
        self.received = self.replaced = 0
        self.visual_capacity = 0
        self.visual_frames = {}

    def receive(self, message):
        if message.encoding != "nv12":
            raise ValueError("BlazePose expects nv12 images")
        if (message.width, message.height) != (self.stream.width, self.stream.height):
            raise ValueError("BlazePose image dimensions must match its configured stream")
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if message.header.stamp.sec < 0 or not 0 <= message.header.stamp.nanosec < 1_000_000_000:
            raise ValueError("BlazePose requires nonnegative normalized ROS timestamps")
        with self.condition:
            if self.closed:
                return
            if stamp <= self.last_stamp:
                raise ValueError("BlazePose requires increasing timestamps per camera")
            self.last_stamp = stamp
            self.received += 1
            self.replaced += self.pending is not None
            self.pending = message
            self.condition.notify_all()

    def pull(self, _output, _timeout_ms):
        with self.condition:
            if not self.condition.wait_for(lambda: self.closed or self.pending is not None,
                                           timeout=self.timeout):
                raise TimeoutError(f"No image from {self.stream.id} for {self.timeout:g} seconds")
            if self.pending is None:
                return None
            message, self.pending = self.pending, None
            if self.visual_capacity:
                stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                self.visual_frames[stamp] = message
                while len(self.visual_frames) > self.visual_capacity:
                    del self.visual_frames[next(iter(self.visual_frames))]
        tensor = image_tensor(message)
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        # Native metadata keeps frame_id as an integer string. Using the unique
        # per-camera ROS timestamp preserves all nanoseconds without a header
        # lookup table that would otherwise retain dropped frames indefinitely.
        identity = helpers.FrameIdentity(self.stream.id, stamp, stamp, 0)
        return helpers.image_input_sample("frames", tensor, identity)

    def take_visual_frame(self, stamp):
        with self.condition:
            message = self.visual_frames.pop(stamp, None)
            # Native publication is monotonic; older retained frames cannot
            # become future publications after this result.
            for key in list(self.visual_frames):
                if key < stamp:
                    del self.visual_frames[key]
            return message

    def close(self):
        # Allow a graceful shutdown to consume the last waiting image first.
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class ResultOutput:
    """Native metadata sender interface -> one ROS result with its exact header."""
    def __init__(self, stream_id, publish):
        self.stream_id, self.publish = stream_id, publish

    def send_raw_json(self, payload):
        # helpers.metadata_message() writes compact, sorted-key JSON: data first,
        # then frame_id/timestamp/type. Keep its landmark JSON intact instead of
        # decoding and encoding hundreds of coordinates again for every frame.
        if not payload.startswith('{"data":'):
            raise ValueError("Unexpected native BlazePose metadata envelope")
        data_json, separator, metadata = payload[8:].rpartition(',"frame_id":')
        if not separator:
            raise ValueError("Native BlazePose metadata lacks a frame identity")
        message = json.loads('{"frame_id":' + metadata)
        if message["type"] == "pose-estimation":
            stamp = int(message["frame_id"])
            header = {"frame_id": self.stream_id,
                      "stamp": {"sec": stamp // 1_000_000_000,
                                "nanosec": stamp % 1_000_000_000}}
            self.publish(header, data_json)
        # The native auxiliary envelope repeats the same world landmarks already
        # included in poses_data. ROS publishes those once in the result data.
        return True


def build_runtime(cfg, timeout, publish):
    """Keep model construction visible and reuse the native shared state."""
    width = max(stream.width for stream in cfg.streams)
    height = max(stream.height for stream in cfg.streams)
    detector_model = helpers.make_detector_model(cfg, width, height)
    detector_graph, detector_run = helpers.build_detector_run(detector_model, width, height)
    try:
        pose_model = helpers.make_pose_model(cfg)
        pose_graph, pose_run = helpers.build_pose_run(pose_model)
    except BaseException:
        detector_run.close()
        raise
    streams = [helpers.StreamRuntime(
        index, stream, None, ResultOutput(stream.id, publish),
        stream.width, stream.height, stream.fps,
        pose_temporal_filter_enabled=cfg.pose_temporal_filter_enabled,
        source_run=ImageInput(stream, timeout)) for index, stream in enumerate(cfg.streams)]
    return helpers.AppRuntime(detector_graph, detector_run, pose_graph, pose_run,
                             detector_model, pose_model, streams, helpers.SharedState(len(streams)))


class BlazePoseNode(Node):
    def __init__(self, cfg, settings):
        super().__init__("multi_stream_blazepose3d")
        self.cfg, self.settings = cfg, settings
        self.visual_output = VisualOutput(settings.get("visualization", {}),
                                          auxiliary=helpers.world_pose_auxiliary_from_overlay)
        self.visual_misses = 0
        self.output_lock = threading.Lock()
        self.completed = 0
        self.first = self.last = None
        self.draining = False
        self.workers = []
        self.runtime = None
        self.results = self.create_publisher(String, settings.get("results_topic", "results"), 16)
        self.statistics = self.create_publisher(String, "~/statistics", 1)
        self.runtime = build_runtime(cfg, settings.get("source_timeout_seconds", 30), self.publish_result)
        self.inputs = {stream.config.id: stream.source_run for stream in self.runtime.streams}
        if self.visual_output.enabled:
            capacity = positive_int(settings.get("visualization", {}), "frame_cache", 8)
            for source in self.inputs.values():
                source.visual_capacity = capacity
        # Match native startup: both models exist before any camera is accepted.
        self.create_subscription(Image, settings.get("image_topic", "image"), self.receive,
                                 max(16, len(cfg.streams)))
        self.create_timer(1.0, self.report)

    def receive(self, message):
        try:
            if self.draining:
                return
            if message.header.frame_id not in self.inputs:
                raise ValueError("Image frame_id must match a configured streams[].id")
            self.inputs[message.header.frame_id].receive(message)
        except Exception as error:
            helpers.set_error(self.runtime, error)

    def read_source(self, stream):
        try:
            # Native source failures close that camera without stopping others.
            helpers.pull_source_frames(self.runtime, self.cfg, stream)
        finally:
            stream.source_run.close()

    def start(self):
        deadline = time.monotonic() + self.settings.get("result_timeout_seconds", 30)
        while self.results.get_subscription_count() < self.settings.get("result_subscribers", 0):
            if time.monotonic() >= deadline:
                raise TimeoutError("Required result subscribers did not connect")
            time.sleep(0.05)
        self.workers = [threading.Thread(target=self.read_source, args=(stream,),
                                        name=f"blazepose-{stream.config.id}")
                        for stream in self.runtime.streams]
        self.workers += [threading.Thread(target=worker, args=(self.runtime, self.cfg),
                                         name=worker.__name__)
                         for worker in (helpers.dispatch_detector_jobs, helpers.pull_detector_outputs,
                                        helpers.dispatch_pose_jobs, helpers.pull_pose_outputs,
                                        helpers.publish_results)]
        for worker in self.workers:
            worker.start()

    def publish_result(self, header, data_json):
        with self.output_lock:
            if self.visual_output.enabled:
                stamp = header["stamp"]["sec"] * 1_000_000_000 + header["stamp"]["nanosec"]
                message = self.inputs[header["frame_id"]].take_visual_frame(stamp)
                if message is not None:
                    try:
                        self.visual_output.write(message, json.loads(data_json), self.completed)
                    except Exception as error:
                        # Native metadata transport tolerates RuntimeError send
                        # failures. A requested visual save must fail visibly.
                        helpers.set_error(self.runtime, error)
                        raise
                else:
                    self.visual_misses += 1
            envelope = json.dumps({"schema_version": 1, "sequence": self.completed, "header": header},
                                  separators=(",", ":"), allow_nan=False)
            payload = envelope[:-1] + ',"data":' + data_json + '}'
            self.results.publish(String(data=payload))
            self.completed += 1
            if self.completed > self.settings.get("warmup_results", 0):
                self.last = time.perf_counter()
                if self.first is None:
                    self.first = self.last

    def summary(self):
        with self.output_lock:
            measured = max(0, self.completed - self.settings.get("warmup_results", 0))
            elapsed = self.last - self.first if measured > 1 else 0.0
            completed = self.completed
        with self.runtime.state.condition:
            streams = [{"frame_id": s.config.id, "received": s.source_run.received,
                        "replaced_before_admission": s.source_run.replaced,
                        "admitted": s.frames_in, "completed": s.frames_out,
                        "outstanding": s.outstanding_frames, "closed": s.closed,
                        # Native publication increments frames_out just before
                        # finish_frame releases outstanding work. Clamp that
                        # brief overlap; counts are exact once drained.
                        "dropped_after_admission": max(0, s.frames_in - s.frames_out - s.outstanding_frames)}
                       for s in self.runtime.streams]
            error = self.runtime.state.error
        return {"completed": completed, "streams": streams,
                "visualization_missing_frames": self.visual_misses,
                "completion_interval_seconds": elapsed,
                "completed_output_fps": (measured - 1) / elapsed if elapsed else None,
                "error": str(error) if error else None}

    def report(self):
        self.statistics.publish(String(data=json.dumps(self.summary(), allow_nan=False)))

    def close(self):
        if self.runtime is None:
            return
        self.draining = True
        for stream in self.runtime.streams:
            stream.source_run.close()
        # Keep ROS alive until queued model work has finished. In latest mode a
        # finite source need not receive one output for every image it published.
        if self.workers:
            state = self.runtime.state
            with state.condition:
                drained = state.condition.wait_for(
                    lambda: state.error is not None or helpers.all_streams_done(self.runtime),
                    timeout=self.settings.get("result_timeout_seconds", 30))
            if not drained:
                helpers.set_error(self.runtime, TimeoutError("BlazePose did not drain on shutdown"))
        helpers.stop_runtime(self.runtime)
        for worker in self.workers:
            worker.join()
        self.visual_output.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--visual-insight", choices=("true", "false"))
    args, ros_args = parser.parse_known_args(argv)
    config = yaml.safe_load(args.config.read_text())
    settings = config.get("ros", {})
    if args.visual_insight is not None:
        settings.setdefault("visualization", {}).setdefault("insight", {})["enabled"] = args.visual_insight == "true"
    if settings.get("input_policy", "latest") != "latest":
        raise ValueError("BlazePose mirrors native latest-frame scheduling; set ros.input_policy: latest")
    for key in ("source_timeout_seconds", "result_timeout_seconds"):
        value = settings.get(key, 30)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"ros.{key} must be a positive finite number")
    for key in ("warmup_results", "result_subscribers"):
        value = settings.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"ros.{key} must be a nonnegative integer")
    helpers.load_runtime_dependencies()
    cfg = helpers.load_app_config(args.config)
    # Python handles SIGINT so queued work drains before the ROS context closes.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = BlazePoseNode(cfg, settings)
        executor.add_node(node)
        node.start()
        while rclpy.ok() and node.runtime.state.error is None:
            executor.spin_once(timeout_sec=0.05)
            with node.runtime.state.condition:
                if helpers.all_streams_done(node.runtime):
                    helpers.require_successful_completion(node.runtime, cfg.frame_limit)
                    break
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is None:
            raise
        helpers.set_error(node.runtime, error)
    finally:
        if node is not None:
            node.close()
            report = node.summary()
            if args.summary:
                args.summary.parent.mkdir(parents=True, exist_ok=True)
                args.summary.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            executor.shutdown()
            node.destroy_node()
            helpers.release_runtime_objects(node.runtime)
        rclpy.try_shutdown()
    return 1 if node is not None and report["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
