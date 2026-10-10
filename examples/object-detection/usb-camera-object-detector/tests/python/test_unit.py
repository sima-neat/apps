"""Unit tests for usb-camera-object-detector (Python).

These run with no camera, no model, and no board: everything covered here is
configuration handling, metadata, or mocked compressed-camera capture.
"""

import copy
import importlib.util
import re
import struct
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
CONFIG_YAML = COMMON_DIR / "config.yaml"
LABELS_TXT = COMMON_DIR / "coco_label.txt"
SCOPE_YAML = EXAMPLE_DIR / "tests" / "test-scope.yaml"
README_MD = EXAMPLE_DIR / "README.md"

_SPEC = importlib.util.spec_from_file_location("usb_camera_object_detector_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


def valid_config() -> dict:
    """A minimal config that must validate, independent of the shipped one."""
    return {
        "model": {"path": "models/pack.tar.gz", "labels": str(LABELS_TXT)},
        "source": {
            "device": "/dev/video16",
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "flip": "none",
            "override_fragment": "",
        },
        "inference": {"frames": 0, "min_score": 0.30, "nms_iou": 0.50, "max_detections": 100},
        "runtime": {"profile": False, "profile_interval": 100, "queue_depth": 3},
        "output": {
            "insight": {
                "host": "127.0.0.1",
                "video_port": 9000,
                "metadata_port": 9100,
                "bitrate_kbps": 4000,
            }
        },
    }


def config_with(**sections) -> dict:
    raw = valid_config()
    for name, values in sections.items():
        raw.setdefault(name, {}).update(values)
    return raw


def bbox_payload(records, declared=None) -> bytes:
    """Build a BBOX payload; declared overrides the record count in the header."""
    count = len(records) if declared is None else declared
    payload = struct.pack("<I", count)
    for x, y, w, h, score, class_id in records:
        payload += struct.pack(main.BBOX_RECORD_FORMAT, x, y, w, h, score, class_id)
    return payload


@pytest.mark.unit
class TestArgParsing:
    """Validate the CLI surface, which must mirror the C++ twin."""

    def test_help(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"], capture_output=True, text=True, timeout=20
        )
        assert r.returncode == 0
        assert "--config" in r.stdout
        assert "--validate-config-only" in r.stdout

    def test_missing_config_file_exits_nonzero(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/usb-camera.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 1
        assert "config file not found" in r.stderr

    def test_unknown_flag_is_rejected(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"], capture_output=True, text=True, timeout=20
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_config_defaults_to_shared_common_config(self):
        assert main.parse_args([]).config == CONFIG_YAML

    def test_validate_flag_defaults_off(self):
        assert main.parse_args([]).validate_config_only is False
        assert main.parse_args(["--validate-config-only"]).validate_config_only is True


@pytest.mark.unit
class TestConfigLoading:
    """Validate config resolution and every rejection rule."""

    def test_shipped_config_is_valid(self):
        raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
        raw["output"]["insight"]["host"] = "127.0.0.1"  # shipped value is a placeholder
        cfg = main.build_app_config(raw)
        main.validate_config(cfg)

        # Like every other example, the shipped config ships a placeholder the
        # reader replaces after downloading the pack. A concrete path here would
        # mean someone committed a machine-local model location.
        assert cfg.model_path == "<model-path>"
        assert cfg.labels_path.name == "coco_label.txt"
        assert (cfg.width, cfg.height, cfg.fps) == (1920, 1080, 30)
        assert cfg.flip == "none"

    def test_shipped_config_host_is_a_placeholder(self):
        """The committed config must not carry a real lab IP."""
        raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))

        assert raw["output"]["insight"]["host"] == "<insight-host-ip>"
        assert raw["source"]["override_fragment"] == ""
        # The capture node is assigned at plug-in time and differs per camera,
        # port and boot. Shipping a real number invites reusing a stale one that
        # may name a live non-camera device on the board.
        assert raw["source"]["device"] == "<video-capture-node>"

    def test_defaults_apply_to_missing_sections(self):
        cfg = main.build_app_config({"model": {"path": "models/pack.tar.gz", "labels": "l.txt"}})

        assert (cfg.width, cfg.height, cfg.fps) == (1920, 1080, 30)
        assert cfg.device == ""
        assert cfg.max_detections == 100
        assert cfg.queue_depth == 3
        assert cfg.profile is False

    def test_omitted_device_is_rejected_without_override(self):
        raw = valid_config()
        del raw["source"]["device"]
        with pytest.raises(ValueError, match="source.device"):
            main.validate_config(main.build_app_config(raw))
        raw["source"]["override_fragment"] = "videotestsrc ! queue"
        main.validate_config(main.build_app_config(raw))

    @pytest.mark.parametrize(
        "section,key,value,message",
        [
            ("source", "width", 0, "source.width"),
            ("source", "width", -1, "source.width"),
            ("source", "width", 1279, "source.width"),
            ("source", "height", 719, "source.height"),
            ("source", "height", 0, "source.height"),
            ("source", "fps", 0, "source.fps"),
            ("source", "device", "", "source.device"),
            ("inference", "frames", -1, "inference.frames"),
            ("inference", "min_score", 1.5, "inference.min_score"),
            ("inference", "min_score", -0.1, "inference.min_score"),
            ("inference", "nms_iou", 1.2, "inference.nms_iou"),
            ("inference", "max_detections", 0, "inference.max_detections"),
            ("runtime", "profile_interval", 0, "runtime.profile_interval"),
            ("runtime", "queue_depth", 0, "runtime.queue_depth"),
        ],
    )
    def test_out_of_range_values_are_rejected(self, section, key, value, message):
        raw = config_with(**{section: {key: value}})
        with pytest.raises(ValueError, match=message.replace(".", r"\.")):
            main.validate_config(main.build_app_config(raw))

    @pytest.mark.parametrize(
        "key,message",
        [("host", "output.insight.host"), ("video_port", "output.insight.video_port"),
         ("metadata_port", "output.insight.metadata_port"),
         ("bitrate_kbps", "output.insight.bitrate_kbps")],
    )
    def test_insight_settings_are_required(self, key, message):
        raw = valid_config()
        raw["output"]["insight"][key] = "" if key == "host" else 0
        with pytest.raises(ValueError, match=message.replace(".", r"\.")):
            main.validate_config(main.build_app_config(raw))

    def test_missing_model_path_is_rejected(self):
        raw = config_with(model={"path": ""})
        with pytest.raises(ValueError, match="model.path"):
            main.validate_config(main.build_app_config(raw))

    @pytest.mark.parametrize("key", ["video_port", "metadata_port"])
    @pytest.mark.parametrize("port", [-1, 0, 65536, 70000])
    def test_invalid_udp_ports_are_rejected(self, key, port):
        raw = valid_config()
        raw["output"]["insight"][key] = port
        with pytest.raises(ValueError, match=key):
            main.validate_config(main.build_app_config(raw))

    @pytest.mark.parametrize("key", ["video_port", "metadata_port"])
    @pytest.mark.parametrize("port", [1, 65535])
    def test_udp_port_boundaries_are_valid(self, key, port):
        raw = valid_config()
        raw["output"]["insight"][key] = port
        main.validate_config(main.build_app_config(raw))

    def test_device_may_be_empty_when_overridden(self):
        """An override fragment replaces the camera, so the device is not needed."""
        raw = config_with(source={"device": "", "override_fragment": "videotestsrc ! queue"})

        main.validate_config(main.build_app_config(raw))  # must not raise

    def test_non_mapping_root_is_rejected(self):
        with pytest.raises(ValueError, match="mapping"):
            main.build_app_config(["not", "a", "mapping"])

    def test_non_mapping_section_is_rejected(self):
        with pytest.raises(ValueError, match="source must be a mapping"):
            main.build_app_config({"source": "0"})

    @pytest.mark.parametrize("value", ["30", 30.5, True])
    def test_non_integer_fps_is_rejected(self, value):
        with pytest.raises(ValueError, match="fps must be an integer"):
            main.build_app_config(config_with(source={"fps": value}))

    def test_non_boolean_profile_is_rejected(self):
        with pytest.raises(ValueError, match="profile must be true or false"):
            main.build_app_config(config_with(runtime={"profile": "yes"}))


@pytest.mark.unit
class TestFlip:
    """Validate flip parsing, which maps onto videoflip methods."""

    @pytest.mark.parametrize(
        "value", ["none", "rotate-180", "horizontal-flip", "vertical-flip"]
    )
    def test_supported_methods(self, value):
        assert main.parse_flip(value) == value

    @pytest.mark.parametrize("value", ["NONE", " Rotate-180 ", "Vertical-Flip"])
    def test_parsing_is_case_and_space_insensitive(self, value):
        assert main.parse_flip(value) == value.strip().lower()

    @pytest.mark.parametrize("value", ["rotate-90", "flip", "", "mirror"])
    def test_unsupported_methods_are_rejected(self, value):
        with pytest.raises(ValueError, match="source.flip"):
            main.parse_flip(value)

    def test_every_method_has_a_videoflip_mapping(self):
        assert set(main.FLIP_METHODS) == {
            "none", "rotate-180", "horizontal-flip", "vertical-flip"
        }
        assert main.FLIP_METHODS["none"] == ""


class FakeCapture:
    def __init__(self, frames=(), *, opened=True, mismatches=None, reject_raw=False):
        self.frames = iter(frames)
        self.opened = opened
        self.properties = {}
        self.mismatches = mismatches or {}
        self.reject_raw = reject_raw
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.properties[key] = value
        return not (self.reject_raw and key == "CONVERT_RGB")

    def get(self, key):
        return self.mismatches.get(key, self.properties.get(key, 0))

    def read(self):
        data = next(self.frames, None)
        return (False, None) if data is None else (True, SimpleNamespace(tobytes=lambda: data))

    retrieve = read

    def release(self):
        self.released = True


def fake_cv2(capture):
    module = SimpleNamespace(CAP_V4L2=200, VideoCapture=lambda *args: capture,
                             VideoWriter_fourcc=lambda *args: 1196444237)
    module.VideoCapture.waitAny = lambda streams, timeout: (True, [0])
    for key in ("FOURCC", "FRAME_WIDTH", "FRAME_HEIGHT", "FPS", "BUFFERSIZE", "CONVERT_RGB"):
        setattr(module, "CAP_PROP_" + key, key)
    return module


@pytest.mark.unit
class TestUsbCapture:
    def test_preserves_encoded_jpeg_and_disables_cpu_decode(self):
        jpeg = b"\xff\xd8compressed-camera-frame\xff\xd9"
        capture = FakeCapture([jpeg])
        camera = main.UsbCamera(main.build_app_config(valid_config()), fake_cv2(capture))
        assert capture.properties["CONVERT_RGB"] == 0
        assert capture.properties["FOURCC"] == 1196444237
        assert capture.properties["BUFFERSIZE"] == 2
        assert camera.caps == "image/jpeg,width=1920,height=1080,framerate=30/1"
        assert camera.read() == jpeg
        camera.close()
        assert capture.released

    def test_discards_truncated_jpeg_before_decode(self):
        jpeg = b"\xff\xd8complete\xff\xd9"
        capture = FakeCapture([b"\xff\xd8truncated", jpeg])
        camera = main.UsbCamera(main.build_app_config(valid_config()), fake_cv2(capture))
        assert camera.read() == jpeg
        assert camera.dropped_frames == 1

    @pytest.mark.parametrize("frames,message", [
        ([], "stopped delivering"),
        ([b"not a JPEG"], "invalid MJPEG"),
        ([b"\xff\xd8truncated"] * 8, "8 consecutive incomplete"),
    ])
    def test_capture_errors_fail_clearly(self, frames, message):
        camera = main.UsbCamera(main.build_app_config(valid_config()), fake_cv2(FakeCapture(frames)))
        with pytest.raises(RuntimeError, match=message):
            camera.read()

    @pytest.mark.parametrize("kwargs,message", [
        ({"opened": False}, "Cannot open"),
        ({"reject_raw": True}, "without CPU decode"),
        ({"mismatches": {"FOURCC": 0}}, "compressed MJPEG"),
        ({"mismatches": {"FRAME_WIDTH": 640}}, "requested resolution/frame rate"),
        ({"mismatches": {"FRAME_HEIGHT": 480}}, "requested resolution/frame rate"),
        ({"mismatches": {"FPS": 15}}, "requested resolution/frame rate"),
        ({"mismatches": {"FPS": float("nan")}}, "requested resolution/frame rate"),
    ])
    def test_failed_negotiation_releases_camera(self, kwargs, message):
        capture = FakeCapture(**kwargs)
        with pytest.raises(RuntimeError, match=message):
            main.UsbCamera(main.build_app_config(valid_config()), fake_cv2(capture))
        assert capture.released

    def test_stalled_camera_can_be_cancelled_without_concurrent_release(self):
        import threading
        import time

        capture = FakeCapture([b"\xff\xd8seed\xff\xd9"])
        cv2 = fake_cv2(capture)
        waiting = threading.Event()
        def wait_any(streams, timeout):
            assert timeout == 100_000_000
            assert not capture.released
            waiting.set()
            time.sleep(timeout / 1e9)
            return False, []
        cv2.VideoCapture.waitAny = wait_any
        camera = main.UsbCamera(main.build_app_config(valid_config()), cv2)
        stop = threading.Event()
        result = []
        def capture_frame():
            try:
                result.append(camera.read(stop))
            finally:
                camera.close()
        worker = threading.Thread(target=capture_frame)
        worker.start()
        try:
            assert waiting.wait(1)
        finally:
            stop.set()
            worker.join(1)
        assert not worker.is_alive()
        assert result == [None]
        assert capture.released

    def test_stalled_camera_times_out(self, monkeypatch):
        capture = FakeCapture([b"\xff\xd8seed\xff\xd9"])
        cv2 = fake_cv2(capture)
        cv2.VideoCapture.waitAny = lambda *args: (False, [])
        camera = main.UsbCamera(main.build_app_config(valid_config()), cv2)
        times = iter([0, 21])
        monkeypatch.setattr(main.time, "monotonic", lambda: next(times))
        with pytest.raises(RuntimeError, match="timed out"):
            camera.read()
        camera.close()

    def test_override_description_is_preserved(self):
        override = "videotestsrc ! video/x-raw,format=NV12 ! queue"
        cfg = main.build_app_config(config_with(source={"override_fragment": override}))
        assert main.source_description(cfg) == override


@pytest.mark.unit
class TestCaptureLifecycle:
    @pytest.mark.parametrize("fail_metadata", [False, True])
    def test_success_and_send_failure_stop_and_join_capture(self, monkeypatch, fail_metadata):
        import threading

        pushed, closed = threading.Event(), threading.Event()
        push_calls = []

        def push(samples):
            push_calls.append(samples)
            pushed.set()
            return True

        def read(stop=None):
            assert closed.wait(2), "capture was not stopped"
            return b"\xff\xd8\xff\xd9"

        def pull(*args):
            assert pushed.wait(2), "capture did not submit the seed"
            return SimpleNamespace()

        def send(*args):
            if fail_metadata:
                raise RuntimeError("metadata send failed")

        monkeypatch.setattr(main, "pyneat", SimpleNamespace(
            make_encoded_sample=lambda *args, **kwargs: SimpleNamespace()))
        monkeypatch.setattr(main, "extract_bbox_payload", lambda sample: b"")
        monkeypatch.setattr(main, "send_metadata", send)
        runtime = SimpleNamespace(seed=SimpleNamespace(), video_port=9000,
                                  run=SimpleNamespace(push=push, pull=pull, close=closed.set))
        cfg = main.build_app_config(config_with(inference={"frames": 1}))
        camera = SimpleNamespace(read=read, caps="image/jpeg", close=lambda: None)
        if fail_metadata:
            with pytest.raises(RuntimeError, match="metadata send failed"):
                main.run_pipeline(runtime, cfg, camera)
        else:
            assert main.run_pipeline(runtime, cfg, camera) == 1
        assert closed.is_set()
        assert len(push_calls) == 1
        assert not any(t.name == "usb-capture" for t in threading.enumerate())

    def test_camera_failure_reaches_main_loop(self):
        import threading

        failed, closed = threading.Event(), threading.Event()

        def read(stop=None):
            failed.set()
            raise RuntimeError("USB camera stopped delivering frames")

        def pull(*args):
            assert failed.wait(2)
            return None

        runtime = SimpleNamespace(seed=None, run=SimpleNamespace(pull=pull, close=closed.set))
        with pytest.raises(RuntimeError, match="stopped delivering frames"):
            main.run_pipeline(runtime, main.build_app_config(valid_config()),
                              SimpleNamespace(read=read, caps="image/jpeg", close=lambda: None))
        assert closed.is_set()
        assert not any(t.name == "usb-capture" for t in threading.enumerate())


@pytest.mark.unit
class TestLabels:
    """Validate label handling."""

    def test_shipped_labels_are_the_coco_80(self):
        labels = main.load_labels(LABELS_TXT)

        assert len(labels) == 80
        assert labels[0] == "person"

    def test_missing_labels_file_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="labels file does not exist"):
            main.load_labels(tmp_path / "absent.txt")

    def test_empty_labels_file_is_rejected(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("\n \n", encoding="utf-8")
        with pytest.raises(ValueError, match="labels file is empty"):
            main.load_labels(empty)

    def test_class_label_falls_back_for_unknown_ids(self):
        labels = ["person", "bicycle"]

        assert main.class_label(1, labels) == "bicycle"
        assert main.class_label(99, labels) == "unknown"
        assert main.class_label(-1, labels) == "unknown"


@pytest.mark.unit
class TestBboxPayload:
    """Validate BBOX decoding, which must match the C++ twin record for record."""

    def test_record_layout_is_24_bytes(self):
        assert main.BBOX_RECORD_SIZE == 24

    def test_parses_records_into_xyxy(self):
        boxes = main.parse_bbox_payload(bbox_payload([(10, 20, 30, 40, 0.9, 2)]), 1920, 1080, 100)

        assert boxes == [
            {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": pytest.approx(0.9),
             "class_id": 2}
        ]

    def test_boxes_are_clamped_to_the_frame(self):
        payload = bbox_payload([(-20, -30, 100, 100, 0.9, 0), (1900, 1060, 200, 200, 0.9, 1)])
        boxes = main.parse_bbox_payload(payload, 1920, 1080, 100)

        assert boxes[0]["x1"] == 0.0 and boxes[0]["y1"] == 0.0
        assert boxes[1]["x2"] == 1920.0 and boxes[1]["y2"] == 1080.0

    def test_truncated_payload_uses_available_records(self):
        payload = bbox_payload([(10, 10, 20, 20, 0.9, 0)], declared=5)

        assert len(main.parse_bbox_payload(payload, 1920, 1080, 100)) == 1

    def test_max_detections_caps_the_parse(self):
        payload = bbox_payload([(10, 10, 20, 20, 0.9, i) for i in range(10)])

        assert len(main.parse_bbox_payload(payload, 1920, 1080, 3)) == 3
        assert len(main.parse_bbox_payload(payload, 1920, 1080, 0)) == 10

    @pytest.mark.parametrize("payload", [b"", b"\x00", b"\x01\x00\x00"])
    def test_short_payloads_return_no_detections(self, payload):
        assert main.parse_bbox_payload(payload, 1920, 1080, 100) == []

    def test_filters_padding_degenerate_boxes_and_invalid_scores(self):
        payload = bbox_payload([
            (0, 0, 0, 0, 0.0, 0), (2000, 0, 20, 20, 0.9, 0),
            (10, 10, -5, 20, 0.9, 0), (10, 10, 20, 20, 0.1, 0),
            (10, 10, 20, 20, float("nan"), 0), (10, 10, 20, 20, 1.5, 0),
            (10, 10, 20, 20, 0.9, -1), (10, 10, 20, 20, 0.8, 0),
        ])
        boxes = main.parse_bbox_payload(payload, 1920, 1080, 100, 0.3)
        assert len(boxes) == 1 and boxes[0]["score"] == pytest.approx(0.8)

    def test_header_only_payload_returns_no_detections(self):
        assert main.parse_bbox_payload(struct.pack("<I", 0), 1920, 1080, 100) == []


@pytest.mark.unit
class TestMetadata:
    """Validate the Insight object-detection contract."""

    @pytest.mark.parametrize("result", [True, False])
    def test_metadata_send_result(self, result):
        runtime = SimpleNamespace(
            labels=["person"],
            metadata_sender=SimpleNamespace(send_metadata=lambda *args: result),
        )
        cfg = main.build_app_config(valid_config())
        sample = SimpleNamespace(pts_ns=1000000, frame_id=1)
        if result:
            main.send_metadata(runtime, cfg, sample, [])
        else:
            with pytest.raises(RuntimeError, match="metadata send failed"):
                main.send_metadata(runtime, cfg, sample, [])

    def test_metadata_transport_exception_is_propagated(self):
        def fail(*args):
            raise RuntimeError("sendto failed: Network is unreachable")

        runtime = SimpleNamespace(labels=[], metadata_sender=SimpleNamespace(send_metadata=fail))
        with pytest.raises(RuntimeError, match="Network is unreachable"):
            main.send_metadata(runtime, main.build_app_config(valid_config()),
                               SimpleNamespace(pts_ns=-1, frame_id=-1), [])

    def test_boxes_become_xywh_objects(self):
        boxes = [{"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.8, "class_id": 0}]
        objects = main.build_metadata_boxes(boxes, ["person"], 1920, 1080)

        assert objects == [
            {"id": "obj_1", "label": "person", "confidence": pytest.approx(0.8),
             "bbox": [10.0, 20.0, 30.0, 40.0]}
        ]

    def test_ids_are_sequential_from_one(self):
        boxes = [{"x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": 5.0, "score": 0.5, "class_id": 0}] * 3
        objects = main.build_metadata_boxes(boxes, ["person"], 1920, 1080)

        assert [o["id"] for o in objects] == ["obj_1", "obj_2", "obj_3"]

    def test_width_and_height_are_clamped_to_the_frame(self):
        boxes = [{"x1": 1900.0, "y1": 1070.0, "x2": 2400.0, "y2": 1600.0, "score": 0.5,
                  "class_id": 0}]
        objects = main.build_metadata_boxes(boxes, ["person"], 1920, 1080)

        x, y, w, h = objects[0]["bbox"]
        assert x + w <= 1920 and y + h <= 1080

    def test_unknown_class_is_labelled_not_dropped(self):
        boxes = [{"x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": 5.0, "score": 0.5, "class_id": 999}]
        objects = main.build_metadata_boxes(boxes, ["person"], 1920, 1080)

        assert objects[0]["label"] == "unknown"

    def test_empty_detections_produce_an_empty_list(self):
        assert main.build_metadata_boxes([], ["person"], 1920, 1080) == []


@pytest.mark.unit
class TestModelAcquisition:
    """Keep the configured model, the documented download, and the test scope in step."""

    @staticmethod
    def scope() -> dict:
        return yaml.safe_load(SCOPE_YAML.read_text(encoding="utf-8"))

    def test_documented_default_model_is_declared_in_test_scope(self):
        """The Model named in the README metadata must be the one e2e downloads."""
        readme = README_MD.read_text(encoding="utf-8")
        match = re.search(r"^\|\s*Model\s*\|\s*(.+?)\s*\|\s*$", readme, re.MULTILINE)
        assert match, "README metadata table has no Model row"

        documented = f"{match.group(1)}.tar.gz"
        declared = {model["file"] for model in self.scope()["models"].values()}

        assert documented in declared, f"{documented} is not declared in test-scope.yaml"

    def test_scope_models_are_downloadable_artifacts(self):
        for model_id, model in self.scope()["models"].items():
            if model["source"] == "modelzoo":
                assert model["name"] == "yolo_26n"
                assert model["file"] == "yolo_26n_mpk.tar.gz"
                continue
            assert model["source"] == "url", f"{model_id} must be a downloadable artifact"
            assert model["url"].startswith("https://"), f"{model_id} needs an https url"
            assert model["url"].endswith(model["file"]), f"{model_id} url must end with its file"

    def test_scope_model_urls_are_modelzoo_version_agnostic(self):
        """The Model Zoo version is resolved at download time, never hardcoded."""
        for model_id, model in self.scope()["models"].items():
            if model["source"] == "modelzoo":
                continue
            assert "{modelzoo_version}" in model["url"], (
                f"{model_id} url must use the {{modelzoo_version}} placeholder"
            )
            assert "SDK2." not in model["url"].replace("SDK{modelzoo_version}", ""), (
                f"{model_id} url must not pin an SDK version"
            )

    @pytest.mark.parametrize("language", ["python", "cpp"])
    def test_e2e_selects_a_declared_model(self, language):
        scope = self.scope()
        e2e = scope["e2e"][language]

        assert e2e["enabled"] is True, f"{language} e2e should be enabled"
        assert e2e["models"], f"{language} e2e selects no model"
        for model_id in e2e["models"]:
            assert model_id in scope["models"], f"{model_id} is not declared"

    def test_documented_download_matches_the_scope_url(self):
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        for model in self.scope()["models"].values():
            if model["source"] == "modelzoo":
                assert "get " + model["name"] in readme
                continue
            documented = model["url"].replace("{modelzoo_version}", "${MODELZOO_VERSION}")
            assert documented in readme, f"README does not document {documented}"


@pytest.mark.unit
class TestTwinParity:
    """Cheap structural checks that the C++ twin stayed in step."""

    @staticmethod
    def cpp_source() -> str:
        return (EXAMPLE_DIR / "src" / "cpp" / "main.cpp").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "key",
        ["model.path", "model.labels", "source.device", "source.width", "source.height",
         "source.fps", "source.flip", "source.override_fragment", "inference.frames",
         "inference.min_score", "inference.nms_iou", "inference.max_detections",
         "runtime.profile", "runtime.profile_interval", "runtime.queue_depth",
         "output.insight.host", "output.insight.video_port", "output.insight.metadata_port",
         "output.insight.bitrate_kbps"],
    )
    def test_cpp_reads_every_config_key(self, key):
        assert f'"{key}"' in self.cpp_source(), f"C++ twin does not read {key}"

    def test_both_twins_use_the_same_bbox_record_size(self):
        assert "kBboxRecordSize = 24" in self.cpp_source()
        assert main.BBOX_RECORD_SIZE == 24

    def test_both_twins_use_realtime_latest_by_stream(self):
        """One slow branch must not back-pressure the camera in either twin."""
        assert "RealtimeLatestByStream" in self.cpp_source()
        assert "RealtimeLatestByStream" in MAIN_PY.read_text(encoding="utf-8")


@pytest.mark.unit
class TestRepoIntegration:
    """The example must be wired into the repository the way the others are."""

    def test_no_hardcoded_lab_hosts_in_committed_files(self):
        """CONTRIBUTING forbids real board, RTSP, or Insight hosts in examples."""
        for path in (CONFIG_YAML, MAIN_PY, EXAMPLE_DIR / "src" / "cpp" / "main.cpp"):
            text = path.read_text(encoding="utf-8")
            assert "192.168." not in text, f"{path.name} carries a lab IP"
            assert "10.42." not in text, f"{path.name} carries a lab IP"
