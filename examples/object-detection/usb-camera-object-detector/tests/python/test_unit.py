"""Unit tests for usb-camera-object-detector (Python).

These run with no camera, no model, and no board: everything covered here is
either pure configuration handling or the GStreamer fragment builder.
"""

import copy
import importlib.util
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
APPS_ROOT = EXAMPLE_DIR.parents[2]
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
        assert cfg.device == "/dev/video16"
        assert cfg.max_detections == 100
        assert cfg.queue_depth == 3
        assert cfg.profile is False

    @pytest.mark.parametrize(
        "section,key,value,message",
        [
            ("source", "width", 0, "source.width"),
            ("source", "width", -1, "source.width"),
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


@pytest.mark.unit
class TestCameraFragment:
    """Validate the GStreamer fragment. This is the part with no Neat node behind
    it, so it is the part most worth pinning down."""

    @staticmethod
    def fragment(**source) -> str:
        return main.camera_fragment(main.build_app_config(config_with(source=source)))

    def test_pins_mjpeg_not_raw_yuyv(self):
        """Without image/jpeg caps v4l2src negotiates YUYV, capped at ~5 fps at 1080p."""
        assert "image/jpeg" in self.fragment()

    def test_carries_device_resolution_and_rate(self):
        frag = self.fragment(device="/dev/video9", width=1280, height=720, fps=25)

        assert "v4l2src device=/dev/video9" in frag
        assert "width=1280,height=720,framerate=25/1" in frag
        assert "image/jpeg,width=1280,height=720,framerate=25/1" in frag

    def test_uses_mmap_io(self):
        """io-mode=rw memcpys every frame; mmap is zero-copy from the UVC driver."""
        assert "io-mode=mmap" in self.fragment()

    def test_decodes_on_cpu_and_converts_to_nv12(self):
        frag = self.fragment()

        assert "neatdecoder" in frag and "dec-type=mjpeg" in frag
        # jpegparse breaks UVC MJPEG on GStreamer 1.22 (see camera_fragment).
        assert "jpegparse" not in frag
        # The hardware decoder emits NV12 natively; no CPU conversion stage.
        assert "videoconvert" not in frag
        assert "jpegdec" not in frag
        assert "dec-fmt=NV12" in frag

    def test_queues_are_leaky(self):
        """A stalled MLA must drop frames, never back-pressure the camera."""
        assert frag_count(self.fragment(), "leaky=downstream") >= 2

    def test_does_not_end_on_bare_caps(self):
        """gst_parse_launch reads a trailing caps string as an element name and
        fails with `no element "video"`."""
        frag = self.fragment()

        assert not frag.strip().split("!")[-1].strip().startswith("video/")
        assert frag.strip().split("!")[-1].strip().startswith("queue")

    def test_flip_is_absent_by_default(self):
        assert "videoflip" not in self.fragment()

    @pytest.mark.parametrize(
        "flip", ["rotate-180", "horizontal-flip", "vertical-flip"]
    )
    def test_flip_is_inserted_before_conversion(self, flip):
        frag = self.fragment(flip=flip)

        assert f"videoflip method={flip}" in frag
        assert frag.index("neatdecoder") < frag.index("videoflip")
        assert frag.index("videoflip") < frag.rindex("queue")

    def test_override_replaces_the_whole_fragment(self):
        override = "videotestsrc ! video/x-raw,format=NV12 ! queue"
        frag = self.fragment(override_fragment=override)

        assert frag == override
        assert "v4l2src" not in frag

    def test_shipped_test_override_is_a_valid_nv12_source(self):
        """The e2e harness drives this fragment instead of a camera."""
        raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
        override = raw["testing"]["e2e"]["source"]["override_fragment"]

        assert "videotestsrc" in override
        assert "format=NV12" in override
        assert f"width={raw['source']['width']}" in override
        assert f"height={raw['source']['height']}" in override
        assert not override.strip().split("!")[-1].strip().startswith("video/")


def frag_count(fragment: str, needle: str) -> int:
    return fragment.count(needle)


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

    def test_header_only_payload_returns_no_detections(self):
        assert main.parse_bbox_payload(struct.pack("<I", 0), 1920, 1080, 100) == []


@pytest.mark.unit
class TestMetadata:
    """Validate the Insight object-detection contract."""

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
            assert model["source"] == "url", f"{model_id} must be a downloadable artifact"
            assert model["url"].startswith("https://"), f"{model_id} needs an https url"
            assert model["url"].endswith(model["file"]), f"{model_id} url must end with its file"

    def test_scope_model_urls_are_modelzoo_version_agnostic(self):
        """The Model Zoo version is resolved at download time, never hardcoded."""
        for model_id, model in self.scope()["models"].items():
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

    @pytest.mark.parametrize(
        "element",
        ["v4l2src", "io-mode=mmap", "image/jpeg", "neatdecoder", "dec-type=mjpeg",
         "videoflip", "dec-fmt=NV12", "leaky=downstream"],
    )
    def test_cpp_fragment_uses_the_same_elements(self, element):
        assert element in self.cpp_source(), f"C++ fragment is missing {element}"

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

    def test_registered_in_the_category_cmakelists(self):
        cmake = (APPS_ROOT / "examples" / "object-detection" / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )

        assert "usb-camera-object-detector" in cmake

    def test_no_hardcoded_lab_hosts_in_committed_files(self):
        """CONTRIBUTING forbids real board, RTSP, or Insight hosts in examples."""
        for path in (CONFIG_YAML, MAIN_PY, EXAMPLE_DIR / "src" / "cpp" / "main.cpp"):
            text = path.read_text(encoding="utf-8")
            assert "192.168." not in text, f"{path.name} carries a lab IP"
            assert "10.42." not in text, f"{path.name} carries a lab IP"
