"""Exercise admission, frame identity and stop/restart with real ROS publishing.

The controlled runner replaces inference only; these tests are transport and
lifecycle evidence, not model accuracy or accelerator performance evidence.
"""
from array import array
import json
from pathlib import Path
from queue import Queue, Empty
import sys
import threading
import time

import numpy as np
import pyneat
import pytest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "support/ros2"))
from neat_apps.image_node import ImageNode
from neat_apps.images import image_array

pytestmark = pytest.mark.ros2


@pytest.mark.parametrize("encoding,channels,dtype", [
    ("bgr8", 3, np.uint8), ("rgb8", 3, np.uint8),
    ("mono8", 1, np.uint8), ("32FC1", 1, np.float32),
])
def test_packed_message_owns_visible_pixels(encoding, channels, dtype):
    from builtin_interfaces.msg import Time
    from neat_apps.images import to_message

    shape = (4, 6, channels) if channels > 1 else (4, 6)
    pixels = np.arange(np.prod(shape), dtype=dtype).reshape(shape)[::2, ::2]
    expected = np.ascontiguousarray(pixels).tobytes()
    message = to_message(pixels, Time(sec=123, nanosec=456), "camera", encoding)
    assert (message.height, message.width, message.step) == (2, 3, 3 * channels * pixels.itemsize)
    assert message.encoding == encoding
    assert message.header.frame_id == "camera"
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (123, 456)
    pixels.fill(0)
    assert message.data.tobytes() == expected


class ControlledRun:
    def __init__(self, *, corrupt_timestamp=False, block_push=False, empty_timeout=False):
        self.outputs = Queue()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.push_started = threading.Event()
        self.pull_started = threading.Event()
        self.corrupt_timestamp = corrupt_timestamp
        self.block_push = block_push
        self.empty_timeout = empty_timeout

    def push(self, samples):
        self.push_started.set()
        if self.block_push:
            self.closed.wait()
            raise RuntimeError("Runner closed during push")
        output = samples[0]
        if self.corrupt_timestamp:
            output.pts_ns += 1
        self.outputs.put(output)
        return True

    def pull(self, timeout_ms):
        self.pull_started.set()
        if not self.release.wait(timeout_ms / 1000):
            return pyneat.Sample() if self.empty_timeout else None
        try:
            return self.outputs.get(timeout=timeout_ms / 1000)
        except Empty:
            return None

    def close(self):
        self.closed.set()
        self.release.set()


class Pipeline:
    warmup_runs = 0

    def __init__(self, runner):
        self.runner = runner

    def prepare(self, message):
        return [pyneat.Tensor.from_numpy(np.ascontiguousarray(image_array(message)),
                                         copy=True, image_format=pyneat.PixelFormat.BGR)]

    def build(self, tensors):
        return self.runner

    def decode(self, sample, message):
        # A header attached to the wrong frame would expose the wrong pixel.
        return {"pixel": int(sample.tensor.to_numpy(copy=True).reshape(-1)[0])}


def frame(number):
    image = Image()
    image.header.frame_id = "camera_optical_frame"
    image.header.stamp.sec, image.header.stamp.nanosec = number, 42
    image.width = image.height = 1
    image.step, image.encoding = 3, "bgr8"
    image.data = array("B", [number, number, number])
    return image


def until(predicate, listener=None, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the tested lifecycle transition")
        if listener:
            rclpy.spin_once(listener, timeout_sec=0.01)
        else:
            time.sleep(0.005)


@pytest.fixture
def nodes():
    rclpy.init()
    created = []

    def make(runner, policy="complete", **overrides):
        settings = {
            "pending_frames": 2, "inflight_frames": 1, "input_policy": policy,
            "result_timeout_seconds": 1,
            **overrides,
        }
        node = ImageNode("transport_test", Pipeline(runner), settings)
        listener = Node("transport_observer")
        messages = []
        listener.create_subscription(String, "results", lambda msg: messages.append(json.loads(msg.data)),
                                     max(16, settings["pending_frames"] + settings["inflight_frames"]))
        created.extend([node, listener])
        until(lambda: node.results.get_subscription_count() == 1, listener)
        return node, listener, messages

    yield make
    for node in created:
        if isinstance(node, ImageNode):
            node.close()
        node.destroy_node()
    rclpy.try_shutdown()


def test_complete_mode_rejects_overflow_instead_of_losing_frames(nodes):
    runner = ControlledRun()
    node, _, _ = nodes(runner)
    node.receive(frame(1))
    until(lambda: node.admitted == 1)
    for number in (2, 3, 4):
        node.receive(frame(number))
    assert "Input queue full" in str(node.error)
    assert node.dropped == 0


def test_latest_drops_only_unadmitted_frames_and_preserves_headers(nodes):
    runner = ControlledRun()
    node, listener, received = nodes(runner, "latest")
    node.receive(frame(1))
    until(lambda: node.admitted == 1)
    for number in (2, 3, 4):
        node.receive(frame(number))
    runner.release.set()
    until(lambda: len(received) == 3, listener)
    assert node.error is None
    assert node.dropped == 1
    assert [r["sequence"] for r in received] == [0, 1, 2]
    assert [r["header"]["stamp"]["sec"] for r in received] == [1, 3, 4]
    assert [r["data"]["pixel"] for r in received] == [1, 3, 4]


def test_mismatched_runtime_identity_fails_without_publishing(nodes):
    runner = ControlledRun(corrupt_timestamp=True)
    runner.release.set()
    node, listener, received = nodes(runner)
    node.receive(frame(1))
    until(lambda: node.error is not None, listener)
    assert "timestamp mismatch" in str(node.error)
    assert not received


def test_close_releases_blocked_push_and_is_not_an_inference_failure(nodes):
    runner = ControlledRun(block_push=True)
    node, _, _ = nodes(runner)
    node.receive(frame(1))
    assert runner.push_started.wait(timeout=5)
    assert not runner.pull_started.wait(timeout=0.1)
    started = time.monotonic()
    node.close()
    assert time.monotonic() - started < 2
    assert not node.feeder.is_alive() and not node.consumer.is_alive()
    assert node.error is None


def test_unaccepted_first_push_times_out_without_pulling_initialization_output(nodes):
    runner = ControlledRun(block_push=True)
    node, listener, received = nodes(runner)
    node.receive(frame(1))
    until(lambda: node.error is not None, listener)
    assert isinstance(node.error, TimeoutError)
    assert not runner.pull_started.is_set()
    assert not received


def test_missing_result_times_out_with_bounded_admission(nodes):
    runner = ControlledRun()
    node, listener, _ = nodes(runner)
    node.receive(frame(1))
    until(lambda: node.error is not None, listener)
    assert isinstance(node.error, TimeoutError)
    assert node.admitted == 1 and node.completed == 0


def test_empty_modelrunner_timeout_is_not_a_completed_result(nodes):
    runner = ControlledRun(empty_timeout=True)
    node, listener, received = nodes(runner)
    node.receive(frame(1))
    until(lambda: runner.pull_started.is_set())
    time.sleep(0.2)  # Exercise at least one empty Sample timeout.
    assert node.error is None and node.completed == 0
    runner.release.set()
    until(lambda: len(received) == 1, listener)
    assert node.error is None and received[0]["data"]["pixel"] == 1


def test_dds_retains_the_configured_burst_before_callbacks_run(nodes):
    runner = ControlledRun()
    node, listener, received = nodes(runner, pending_frames=40, inflight_frames=4,
                                     result_timeout_seconds=5)
    publisher = listener.create_publisher(Image, "image", 40)
    until(lambda: publisher.get_subscription_count() == 1, listener)
    for number in range(1, 41):
        publisher.publish(frame(number))
    time.sleep(0.15)  # The DDS reader must retain this burst before taking it.
    until(lambda: node.received == 40, node)
    runner.release.set()
    until(lambda: len(received) == 40, listener)
    assert node.error is None and node.dropped == 0
    assert [r["data"]["pixel"] for r in received] == list(range(1, 41))


def test_finished_source_can_disconnect_after_initial_readiness(nodes):
    runner = ControlledRun()
    runner.release.set()
    node, listener, received = nodes(runner, result_subscribers=2)
    source_ack = listener.create_subscription(String, "results", lambda msg: None, 16)
    until(lambda: node.results.get_subscription_count() == 2, listener)
    node.receive(frame(1))
    until(lambda: len(received) == 1, listener)
    listener.destroy_subscription(source_ack)
    until(lambda: node.results.get_subscription_count() == 1, listener)
    node.receive(frame(2))
    until(lambda: len(received) == 2, listener)
    assert node.error is None and node.completed == 2


@pytest.mark.parametrize("encoding", ["nv12", "nv16"])
def test_decoder_plane_padding_is_removed_before_ros_transport(encoding):
    from builtin_interfaces.msg import Time
    from neat_apps.images import image_tensor, planar_array
    from neat_apps.live_source import pack_image

    width, height = 4, 4
    uv_rows = height // 2 if encoding == "nv12" else height
    packed = np.arange((height + uv_rows) * width, dtype=np.uint8).reshape(-1, width)
    backing = np.full((12, 8), 255, dtype=np.uint8)
    backing[1:5, :width] = packed[:height]
    backing[7:7 + uv_rows, :width] = packed[height:]
    pixel_format = pyneat.PixelFormat.NV12 if encoding == "nv12" else pyneat.PixelFormat.NV16
    tensor = pyneat.Tensor.from_numpy(backing, copy=True, image_format=pixel_format,
                                     memory=pyneat.TensorMemory.CPU)
    tensor.shape, tensor.strides_bytes = [height, width], []
    planes = []
    for role, rows, offset in ((pyneat.PlaneRole.Y, height, 8),
                               (pyneat.PlaneRole.UV, uv_rows, 56)):
        plane = pyneat.Plane()
        plane.role, plane.shape = role, [rows, width]
        plane.strides_bytes, plane.byte_offset = [8, 1], offset
        planes.append(plane)
    tensor.planes = planes
    message = pack_image(tensor, Time(sec=123, nanosec=456), "camera", "native")
    assert (message.encoding, message.width, message.height, message.step) == (encoding, width, height, width)
    assert message.header.frame_id == "camera" and message.header.stamp.sec == 123
    np.testing.assert_array_equal(planar_array(message), packed)
    restored = image_tensor(message, memory=pyneat.TensorMemory.CPU)
    assert restored.image_format() == pixel_format
    assert restored.copy_payload_bytes() == packed.tobytes()


def test_saved_visual_is_complete_before_result_acknowledgment(nodes, tmp_path):
    import cv2
    runner = ControlledRun()
    runner.release.set()
    node, listener, received = nodes(runner, visualization={
        'save': {'enabled': True, 'directory': str(tmp_path)}})
    for number in (1, 2):
        node.receive(frame(number))
        until(lambda: len(received) == number, listener)
        manifests = list(tmp_path.glob('run-*/frames.jsonl'))
        records = [json.loads(line) for line in manifests[0].read_text().splitlines()]
        record = records[-1]
        assert record['header'] == received[-1]['header']
        assert record['sequence'] == received[-1]['sequence']
        saved = cv2.imread(str(manifests[0].parent / record['image']))
        assert np.all(saved == number)
        assert received[-1]['data'] == {'pixel': number}


def test_visual_failure_stops_node_without_acknowledging_unsaved_frame(nodes, tmp_path, monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'imwrite', lambda *args: False)
    runner = ControlledRun()
    runner.release.set()
    node, listener, received = nodes(runner, visualization={
        'save': {'enabled': True, 'directory': str(tmp_path)}})
    node.receive(frame(1))
    until(lambda: node.error is not None, listener)
    assert 'Could not save' in str(node.error)
    assert node.completed == 0 and not received


def test_auxiliary_video_and_metadata_share_exact_output_clock(monkeypatch):
    from types import SimpleNamespace
    from neat_apps.visual_output import InsightVideo
    sent, metadata = [], []
    video=InsightVideo.__new__(InsightVideo)
    video.shape=(4,8,3)
    video.settings={'host':'example.test','metadata_port_base':9100}
    video.channel=1
    video.sequence=0
    video.started_ns=1_000_000
    video.run=SimpleNamespace(push=lambda inputs: sent.extend(inputs) or True)
    video.metadata=SimpleNamespace(send_raw_json=lambda payload: metadata.append(json.loads(payload)) or True)
    monkeypatch.setattr('neat_apps.visual_output.time.monotonic_ns',lambda:1_500_000)
    payload={'schema_version':1,'renderer':'blazepose-3d','payload':{'poses':[]}}
    video.send(np.zeros(video.shape,np.uint8),payload)
    assert sent[0].pts_ns==metadata[0]['pts_ns']==1_000_500_000
    assert int(metadata[0]['frame_id'])==sent[0].frame_id==1
    assert metadata[0]['rtp_timestamp']==90_045
    assert metadata[0]['data']==payload
