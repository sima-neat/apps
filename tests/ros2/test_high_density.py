"""Live graph result identity, native numerical output and failed-start cleanup."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pyneat
import pytest
import rclpy
import yaml

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "support/ros2"))
ENTRY = ROOT / "examples/ros2/object-detection/high-density-multi-stream-object-detector/src/python/ros_node.py"


@pytest.fixture
def app():
    spec = importlib.util.spec_from_file_location("high_density_ros_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.native.load_runtime_dependencies()
    return module


def detection(payload, **identity):
    return SimpleNamespace(kind=pyneat.SampleKind.Tensor, payload_tag="BBOX",
                           tensor=SimpleNamespace(copy_payload_bytes=lambda: payload),
                           pts_ns=identity.get("pts_ns", 12_345_678_901),
                           frame_id=identity.get("frame_id", 97), stream_id="stream3")


def test_live_result_preserves_source_identity_and_native_boxes(app):
    # The box crosses image bounds; retain native clipping and FP32 confidence.
    payload = struct.pack("<Iiiiifi", 1, -10, 20, 70, 100, 0.75, 0)
    source = SimpleNamespace(index=3, frame_w=80, frame_h=60, labels=["person"])
    actual = app.live_result(detection(payload), source, SimpleNamespace(max_detections=50), 8)
    assert actual == {"schema_version": 1, "sequence": 8,
                      "header": {"frame_id": "camera_3_optical_frame",
                                 "stamp": {"sec": 12, "nanosec": 345678901}},
                      "source_frame_id": 97, "stream_id": "stream3",
                      "data": {"objects": [{"id": "obj_1", "label": "person",
                                            "confidence": 0.75, "bbox": [0.0, 20.0, 60.0, 40.0]}]}}


@pytest.mark.parametrize("field", ["pts_ns", "frame_id"])
def test_live_result_rejects_missing_identity(app, field):
    with pytest.raises(ValueError, match="source timestamp or frame ID"):
        app.live_result(detection(b"", **{field: -1}), None, None, 0)


def test_failed_graph_setup_closes_ros(app, tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"ros": {}}))
    monkeypatch.setattr(app.native, "load_app_config", lambda path:
                        SimpleNamespace(labels_path=tmp_path / "labels", rtsp_urls=["unused"]))
    monkeypatch.setattr(app.native, "load_labels", lambda path: [])
    def fail(config):
        raise RuntimeError("model setup failed")
    monkeypatch.setattr(app.native, "build_detector_graph", fail)
    with pytest.raises(RuntimeError, match="model setup failed"):
        app.run_live(config, [])
    assert not rclpy.ok()


@pytest.mark.parametrize("pts", [0, 12_345_678_901, 100_000_345_678_901])
def test_insight_bridge_matches_native_metadata(app, pts):
    from neat_apps.insight_sink import detection_metadata
    payload = struct.pack("<Iiiiifi", 1, -10, 20, 70, 100, 0.75, 0)
    source = SimpleNamespace(index=3, frame_w=80, frame_h=60, labels=["person"])
    sample = detection(payload, pts_ns=pts)
    index, actual = detection_metadata(app.live_result(sample, source, SimpleNamespace(max_detections=50), 8))
    boxes = app.native.parse_boxes_strict(payload, 80, 60, 50)
    expected = json.loads(app.native.build_metadata_payload(source, sample, boxes))
    assert index == 3
    assert actual == {key: expected[key] for key in actual}


def test_insight_sink_exits_cleanly_when_ros_context_stops(tmp_path, monkeypatch):
    from neat_apps.insight_sink import main
    from rclpy.executors import ExternalShutdownException
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"streams": ["unused"], "output": {
        "insight": {"host": "127.0.0.1", "metadata_port_base": 29900}}}))
    def shutdown(node):
        rclpy.shutdown()
        raise ExternalShutdownException()
    monkeypatch.setattr(rclpy, "spin", shutdown)
    assert main(["--config", str(config)]) == 0
    assert not rclpy.ok()
