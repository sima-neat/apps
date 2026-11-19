"""ROS startup/teardown only; model construction and inference stay in each node."""
import signal

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions


def read_only():
    """A fresh descriptor for each parameter; rclpy fills its name and type."""
    return ParameterDescriptor(read_only=True)


def run(node_class):
    # Signals request an orderly stop; do not interrupt a native message conversion
    # or invalidate publishers before the inference workers finish.
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    node = None
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = node_class()
        while rclpy.ok() and not requested and not node.stopping.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.error:
            raise RuntimeError(node.error)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.close()
                node.destroy_node()
            rclpy.try_shutdown()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
