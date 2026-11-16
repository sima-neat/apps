"""Publish the native synthetic model benchmark report through ROS."""
import argparse
from pathlib import Path
import threading
import pyneat
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from neat_apps.runtime import load_native
native = load_native(__file__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args, ros_args = parser.parse_known_args(argv)
    config = native.load_config(args.config)
    model_path = Path(config["model"]["path"])
    benchmark = config.get("benchmark", {})
    frames = int(benchmark.get("frames", 1000))
    duration = int(benchmark.get("min_duration_ms", native.DEFAULT_MIN_DURATION_MS))
    if frames <= 0 or duration < 0:
        raise ValueError("Benchmark frames must be positive and duration nonnegative")
    output = Path(config.get("output", {}).get("report_json", "sandbox/model-benchmark/report.json"))
    options = pyneat.ModelOptions()
    decode = config.get("ros", {}).get("decode_type")
    if decode is not None:
        options.decode_type = getattr(pyneat.BoxDecodeType, native.DECODE_TYPES[decode])
        options.top_k = native.BOXDECODE_TOP_K
    model = pyneat.Model(str(model_path), options)
    rclpy.init(args=ros_args)
    node = Node("model_benchmark")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(String, "report", qos)
    worker = None
    finished = threading.Event()
    error = []

    def benchmark_model():
        try:
            options = pyneat.BenchmarkOptions()
            options.num_samples, options.min_duration_ms = frames, duration
            report = model.benchmark(options)
            native.write_report(output, model_path, frames, duration, decode, model, report)
        except BaseException as exc:
            error.append(exc)
        finally:
            finished.set()

    def start(_request, response):
        nonlocal worker
        if worker is not None and worker.is_alive():
            response.success, response.message = False, "Benchmark already running"
        else:
            finished.clear()
            error.clear()
            worker = threading.Thread(target=benchmark_model, name="neat-benchmark")
            worker.start()
            response.success, response.message = True, "Benchmark started; report will be published"
        return response

    def publish_finished():
        if finished.is_set():
            finished.clear()
            if error:
                node.get_logger().error(str(error[0]))
            else:
                publisher.publish(String(data=output.read_text()))

    node.create_service(Trigger, "~/run", start)
    node.create_timer(0.1, publish_finished)
    start(None, Trigger.Response())
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if worker is not None:
            worker.join()
        node.destroy_node()
        rclpy.try_shutdown()
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
