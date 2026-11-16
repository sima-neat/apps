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
from neat_apps.runtime import load_native
native = load_native(__file__)


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


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        self.labels = native.load_labels(self.config.labels_path)
        self.geometry = None

    def prepare(self, message):
        from neat_apps.images import image_tensor
        if message.encoding != "nv12":
            raise ValueError("High-density inference expects nv12")
        geometry = (message.width, message.height)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("All high-density streams must use the same geometry")
        self.geometry = geometry
        return [image_tensor(message)]

    def build(self, seed):
        self.model, detector = native.build_detector_graph(self.config)
        self.graph = pyneat.Graph(options=native.graph_options(self.config.internal_queue_depth))
        self.graph.add(detector)
        options = native.realtime_options(self.config.queue_depth, pyneat.OverflowPolicy.Block)
        return DetectorRun(self.graph.build(options))

    def decode(self, sample, message):
        boxes = native.parse_boxes_strict(native.extract_bbox_payload(sample), message.width,
                                         message.height, self.config.max_detections)
        return {"objects": native.build_metadata_boxes(boxes, self.labels, message.width, message.height)}


def live_result(sample, source, config, sequence):
    """Keep the decoder's stream/frame identity and presentation timestamp."""
    if sample.pts_ns < 0 or sample.frame_id < 0:
        raise ValueError("Live detection is missing its source timestamp or frame ID")
    boxes = native.parse_boxes_strict(native.extract_bbox_payload(sample), source.frame_w,
                                     source.frame_h, config.max_detections)
    seconds, nanoseconds = divmod(int(sample.pts_ns), 1_000_000_000)
    return {"schema_version": 1, "sequence": sequence,
            "header": {"frame_id": f"camera_{source.index}_optical_frame",
                       "stamp": {"sec": seconds, "nanosec": nanoseconds}},
            "source_frame_id": int(sample.frame_id), "stream_id": str(sample.stream_id),
            "data": {"objects": native.build_metadata_boxes(
                boxes, source.labels, source.frame_w, source.frame_h)}}


def run_live(config_path, ros_args, summary_path=None):
    """Decode and infer in one native graph; only results cross the ROS boundary."""
    native.load_runtime_dependencies()
    config = native.load_app_config(config_path)
    settings = (yaml.safe_load(config_path.read_text()) or {}).get("ros", {})
    labels = native.load_labels(config.labels_path)
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
        model, detector = native.build_detector_graph(config)
        app = native.AppRuntime(model, pyneat.Graph(options=native.graph_options(
            config.internal_queue_depth)), None, [])
        for index, url in enumerate(config.rtsp_urls):
            fps, width, height = native.resolve_source_geometry(config, url)
            source = native.SourceRuntime(index, url, None, labels, width, height, fps)
            app.sources.append(source)
            native.connect_source_graph(app, config, source, detector)
        native.validate_worker_pool_geometry(app.sources)
        app.run = app.graph.build(native.realtime_options(config.queue_depth))

        last_seen = [time.monotonic()] * len(app.sources)
        next_check = time.monotonic() + 1
        while rclpy.ok():
            did_work = False
            for _ in range(native.MAX_DETECTION_PULLS_PER_ROUND):
                sample = app.run.pull("detections", 0)
                if sample is None:
                    error = app.run.last_error()
                    if error or not app.run.running():
                        raise RuntimeError(f"Live detector stopped: {error}")
                    break
                did_work = True
                index = native.stream_index_from_detection(sample, len(app.sources))
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
    from neat_apps.image_node import run
    return run("high_density_multi_stream_object_detector", Pipeline, argv)


if __name__ == "__main__":
    raise SystemExit(main())
