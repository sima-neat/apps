"""BlazePose ROS boundaries and native latest-frame scheduling, without models."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from queue import Queue, Empty
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Image

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "examples/ros2/pose-estimation/multi-stream-blazepose3d/src/python/pose_node.py"


@pytest.fixture
def app():
    from tests.ros2.local_modules import load_node
    module = load_node(ENTRY, "standalone_test_blazepose")
    return module


def message(stamp, camera="camera0"):
    value = Image()
    value.header.frame_id = camera
    value.header.stamp.sec, value.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
    value.width, value.height, value.step = 8, 4, 8
    value.encoding = "nv12"
    value.data = bytes(48)
    return value


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Worker did not make progress"
        time.sleep(.005)


def test_input_keeps_only_newest_frame_and_exact_timestamp(app, monkeypatch):
    monkeypatch.setattr(app, "image_tensor", lambda value: value.data)
    monkeypatch.setattr(app.helpers, "image_input_sample", lambda name, tensor, identity: identity)
    source = app.ImageInput(SimpleNamespace(id="camera0", width=8, height=4), .05)
    for stamp in (1_700_000_000_123456700, 1_700_000_000_123456701, 1_700_000_000_123456702):
        source.receive(message(stamp))
    source.close()
    sample = source.pull("frames", -1)
    assert sample.frame_id == sample.pts_ns == 1_700_000_000_123456702
    assert source.pull("frames", -1) is None
    assert source.received == 3 and source.replaced == 2


def test_input_rejects_bad_geometry_encoding_and_time(app):
    source = app.ImageInput(SimpleNamespace(id="camera0", width=8, height=4), .01)
    source.receive(message(100))
    with pytest.raises(ValueError, match="increasing"):
        source.receive(message(100))
    bad = message(101)
    bad.encoding = "rgb8"
    with pytest.raises(ValueError, match="nv12"):
        source.receive(bad)
    bad = message(101)
    bad.width = 10
    with pytest.raises(ValueError, match="dimensions"):
        source.receive(bad)
    source.pending = None
    with pytest.raises(TimeoutError, match="camera0"):
        source.pull("frames", -1)


def test_blocked_source_wakes_on_close(app):
    source = app.ImageInput(SimpleNamespace(id="camera0", width=8, height=4), 30)
    result = []
    worker = threading.Thread(target=lambda: result.append(source.pull("frames", -1)))
    worker.start()
    source.close()
    worker.join(timeout=1)
    assert not worker.is_alive() and result == [None]


def test_publication_uses_native_smoothing_and_stale_suppression(app):
    native = app.helpers
    native.np = np
    result, reference = [], []
    cfg = native.StreamConfig("camera0", "unused", "h264", 0, 8, 4, 30)
    ros_stream = native.StreamRuntime(0, cfg, None,
        app.ResultOutput("camera0", lambda header, data: result.append((header, json.loads(data)))), 8, 4, 30)
    sender = SimpleNamespace(send_raw_json=lambda payload: reference.append(json.loads(payload)) or True)
    native_stream = native.StreamRuntime(0, cfg, None, sender, 8, 4, 30)
    # A later result overtakes frame 2; frame 2 must neither publish nor smooth.
    for sequence, coordinate in ((1, 0.0), (3, 2.0), (2, 100.0), (4, 4.0)):
        identity = native.FrameIdentity("camera0", 1_000_000_000 + sequence,
                                        1_000_000_000 + sequence, sequence)
        pose = {"box": {"x1": 0, "y1": 0, "x2": 8, "y2": 4, "score": .9, "class_id": 0},
                "presence": .9, "image": np.full((39, 3), coordinate, dtype=np.float32),
                "world": np.full((39, 4), coordinate, dtype=np.float32), "roi_index": 0}
        native.publish_frame(native_stream, identity, [deepcopy(pose)])
        native.publish_frame(ros_stream, identity, [deepcopy(pose)])
    expected = [row["data"] for row in reference if row["type"] == "pose-estimation"]
    assert [data for header, data in result] == expected
    assert [header["stamp"]["nanosec"] for header, data in result] == [1, 3, 4]
    assert ros_stream.frames_out == native_stream.frames_out == 3
    np.testing.assert_array_equal(ros_stream.pose_smoother.previous[0]["image"],
                                  native_stream.pose_smoother.previous[0]["image"])


class HeldDetector:
    """One outstanding input; the test explicitly releases each output."""
    def __init__(self):
        self.active = None
        self.attempted = []
        self.outputs = Queue()
        self.lock = threading.Lock()
        self.closed = False

    def try_push(self, port, samples):
        with self.lock:
            self.attempted.append(samples[0].frame_id)
            if self.active is not None:
                return False
            self.active = samples[0]
            return True

    def release(self):
        with self.lock:
            sample, self.active = self.active, None
        assert sample is not None
        self.outputs.put(sample)

    def pull(self, port, timeout):
        try:
            return self.outputs.get(timeout=timeout / 1000)
        except Empty:
            return None

    def can_push(self):
        return not self.closed

    def can_pull(self):
        return not self.closed

    def last_error(self):
        return ""

    def close(self):
        self.closed = True


def test_native_detector_replacement_drains_and_preserves_headers(app, monkeypatch):
    native = app.helpers
    monkeypatch.setattr(app, "image_tensor", lambda message: message)
    monkeypatch.setattr(native, "require_nv12_tensor", lambda sample: sample.tensor)
    monkeypatch.setattr(native, "image_input_sample", lambda name, tensor, identity:
                        SimpleNamespace(tensor=tensor, frame_id=identity.frame_id,
                                        pts_ns=identity.pts_ns))
    monkeypatch.setattr(native, "select_people", lambda *args: [])
    detector, pose = HeldDetector(), HeldDetector()
    cfg = native.AppConfig("unused", "unused", [native.StreamConfig("camera0", "unused", "h264", 0, 8, 4, 30)])

    def build(cfg, timeout, publish):
        stream = native.StreamRuntime(0, cfg.streams[0], None, app.ResultOutput("camera0", publish), 8, 4, 30,
                                      source_run=app.ImageInput(cfg.streams[0], timeout))
        return native.AppRuntime(None, detector, None, pose, None, None, [stream], native.SharedState(1))

    monkeypatch.setattr(app, "build_runtime", build)
    rclpy.init()
    node = app.BlazePoseNode(cfg, {"result_timeout_seconds": 2})
    received = []
    original_publish = node.publish_result
    node.runtime.streams[0].metadata_sender.publish = lambda header, data: (
        received.append(header), original_publish(header, data))
    node.start()
    try:
        node.receive(message(1))
        wait_for(lambda: detector.active is not None)
        node.receive(message(2))
        wait_for(lambda: 2 in detector.attempted)
        node.receive(message(3))
        wait_for(lambda: node.runtime.streams[0].frames_in == 3)
        node.receive(message(4))
        wait_for(lambda: node.runtime.streams[0].frames_in == 4)
        assert node.runtime.state.detector_mailboxes[0].identity.frame_id == 4
        for stamp in (1, 2, 4):
            wait_for(lambda: detector.active is not None and detector.active.frame_id == stamp)
            detector.release()
            wait_for(lambda: any(row["stamp"]["nanosec"] == stamp for row in received))
    finally:
        node.close()
        report = node.summary()
        node.destroy_node()
        rclpy.try_shutdown()
    assert [row["stamp"]["nanosec"] for row in received] == [1, 2, 4]
    assert report["error"] is None
    assert report["streams"][0]["dropped_after_admission"] == 1
    assert report["streams"][0]["outstanding"] == 0
    assert all(not worker.is_alive() for worker in node.workers)


def test_missing_camera_closes_without_stopping_other_cameras(app, monkeypatch):
    native = app.helpers
    detector, pose = HeldDetector(), HeldDetector()
    cfg = native.AppConfig("unused", "unused", [
        native.StreamConfig(f"camera{i}", "unused", "h264", i, 8, 4, 30) for i in range(2)])

    def build(cfg, timeout, publish):
        streams = [native.StreamRuntime(i, camera, None, app.ResultOutput(camera.id, publish), 8, 4, 30,
                   source_run=app.ImageInput(camera, .02 if i == 0 else 30))
                   for i, camera in enumerate(cfg.streams)]
        return native.AppRuntime(None, detector, None, pose, None, None, streams, native.SharedState(2))

    monkeypatch.setattr(app, "build_runtime", build)
    rclpy.init()
    node = app.BlazePoseNode(cfg, {"result_timeout_seconds": 1})
    try:
        node.start()
        wait_for(lambda: node.runtime.streams[0].closed)
        assert node.runtime.state.error is None
        assert not node.runtime.streams[1].closed
        assert not native.all_streams_done(node.runtime)
        node.inputs["camera1"].close()
        wait_for(lambda: native.all_streams_done(node.runtime))
        with pytest.raises(RuntimeError, match="all source streams stopped"):
            native.require_successful_completion(node.runtime, 0)
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    assert detector.closed and pose.closed
    assert all(not worker.is_alive() for worker in node.workers)


def test_result_bridge_preserves_native_json_and_rejects_unknown_envelopes(app):
    delivered = []
    camera = 'camera_é_"frame_id"'
    output = app.ResultOutput(camera, lambda header, data: delivered.append((header, data)))
    data = {"stream_id": camera, "poses": [{"value": 0.123456789, "label": ',"frame_id":'}]}
    payload = app.helpers.metadata_message("pose-estimation", data, 1234, "1234567890")
    output.send_raw_json(payload)
    assert delivered[0][0] == {"frame_id": camera, "stamp": {"sec": 1, "nanosec": 234567890}}
    assert json.loads(delivered[0][1]) == data
    assert delivered[0][1] == json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    output.send_raw_json(app.helpers.metadata_message("auxiliary-visualization", data, 1234, "1234567890"))
    assert len(delivered) == 1
    with pytest.raises(ValueError, match="envelope"):
        output.send_raw_json('{"poses": []}')
    with pytest.raises(ValueError, match="identity"):
        output.send_raw_json('{"data":{},"type":"pose-estimation"}')


def test_visual_frame_cache_is_bounded_and_matches_exact_timestamp(app, monkeypatch):
    monkeypatch.setattr(app, 'image_tensor', lambda value: value.data)
    monkeypatch.setattr(app.helpers, 'image_input_sample', lambda name, tensor, identity: identity)
    source = app.ImageInput(SimpleNamespace(id='camera0', width=8, height=4), .05)
    source.visual_capacity = 2
    originals = {stamp: message(stamp) for stamp in (100, 101, 102)}
    for stamp, msg in originals.items():
        source.receive(msg)
        source.pull('frames', -1)
    assert len(source.visual_frames) == 2
    assert source.take_visual_frame(100) is None
    assert source.take_visual_frame(102) is originals[102]
    assert not source.visual_frames
