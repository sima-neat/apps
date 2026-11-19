"""Live graph result identity, native numerical output and failed-start cleanup."""
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pyneat
import pytest
import rclpy
import yaml

pytestmark = pytest.mark.ros2
ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "examples/ros2/object-detection/high-density-multi-stream-object-detector/src/python/detection_node.py"


@pytest.fixture
def app():
    from tests.ros2.local_modules import load_node
    module = load_node(ENTRY, "standalone_test_high_density")
    module.helpers.load_runtime_dependencies()
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
    monkeypatch.setattr(app.helpers, "load_app_config", lambda path:
                        SimpleNamespace(labels_path=tmp_path / "labels", rtsp_urls=["unused"]))
    monkeypatch.setattr(app.helpers, "load_labels", lambda path: [])
    def fail(config):
        raise RuntimeError("model setup failed")
    monkeypatch.setattr(app.helpers, "build_detector_graph", fail)
    with pytest.raises(RuntimeError, match="model setup failed"):
        app.run_live(config, [])
    assert not rclpy.ok()


@pytest.mark.parametrize("pts", [0, 12_345_678_901, 100_000_345_678_901])
def test_insight_bridge_matches_native_metadata(app, pts):
    from tests.ros2.local_modules import load_node
    detection_metadata = load_node(ENTRY.with_name("insight_sink.py"), "local_metadata_sink").detection_metadata
    payload = struct.pack("<Iiiiifi", 1, -10, 20, 70, 100, 0.75, 0)
    source = SimpleNamespace(index=3, frame_w=80, frame_h=60, labels=["person"])
    sample = detection(payload, pts_ns=pts)
    index, actual = detection_metadata(app.live_result(sample, source, SimpleNamespace(max_detections=50), 8))
    boxes = app.helpers.parse_boxes_strict(payload, 80, 60, 50)
    reference = load_node(
        ROOT / "examples/object-detection/high-density-multi-stream-object-detector/src/python/main.py",
        "native_high_density_reference", resolve_share=False)
    reference.load_runtime_dependencies()
    expected = json.loads(reference.build_metadata_payload(source, sample, boxes))
    assert index == 3
    assert actual == {key: expected[key] for key in actual}


def test_insight_sink_exits_cleanly_when_ros_context_stops(tmp_path, monkeypatch):
    from tests.ros2.local_modules import load_node
    main = load_node(ENTRY.with_name("insight_sink.py"), "local_shutdown_sink").main
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


def test_image_topic_entrypoint_starts_inference(app, monkeypatch):
    calls = []
    monkeypatch.setattr(app, 'run_images', lambda argv: calls.append(argv) or 23)
    argv = ['--config', '/unused/config.yaml']
    assert app.main(argv) == 23
    assert calls == [argv]
