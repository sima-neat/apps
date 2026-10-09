"""Unit tests for single-stream-instance-segmenter (Python)."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
PARITY_FIXTURE = EXAMPLE_DIR / "tests" / "fixtures" / "decode_parity.json"

main = load_example_main(EXAMPLE_DIR, "instance_seg_main")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_type", "url", "tcp", "expected_tcp"),
    [
        ("rtsp", "rtsp://camera/live", True, True),
        ("rtsp", "rtsp://camera/live", False, False),
        ("http", "https://camera/live", True, False),
    ],
)
def test_ffprobe_transport_matches_source(monkeypatch, source_type, url, tcp, expected_tcp):
    captured = []

    def fake_run(cmd, **_kwargs):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout="width=1920\nheight=1080\navg_frame_rate=30/1\n")

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    cfg = main.AppConfig(
        model_path="model",
        labels_path=Path("labels"),
        source_url=url,
        source_type=source_type,
        tcp=tcp,
        ssl_strict=False,
    )

    assert main.probe_ffprobe(cfg) == (1920, 1080, 30)
    assert captured[0].count("-rtsp_transport") == int(expected_tcp)
    assert captured[0][-3:] == ["-tls_verify", "0", url]


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the RTSP segmentation pipeline."""

    def test_help(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/config.yaml"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


@pytest.mark.unit
class TestConfig:
    def test_hevc_alias(self):
        assert main.parse_source_codec("hevc") == "h265"

    def test_validate_config_only_accepts_common_config(self):
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )
        assert r.returncode == 0, r.stderr

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("yolo26", "yolo26"),
            ("YOLO26", "yolo26"),
            ("yolov8", "yolov8"),
            ("yolo_v8", "yolov8"),
        ],
    )
    def test_model_family_is_selected_explicitly(self, value, expected):
        assert main.parse_model_family(value) == expected

    def test_unknown_model_family_is_rejected(self):
        # The family is configuration, never inferred from the package file name.
        with pytest.raises(ValueError, match="model.family"):
            main.parse_model_family("yolo_v8n_seg_mpk.tar.gz")

    def test_config_carries_family_and_input_size(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  family: yolov8
  path: model.tar.gz
  input_size: 640
source:
  url: rtsp://127.0.0.1:8554/src1
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )

        cfg = main.load_app_config(config)

        assert cfg.model_family == main.YOLOV8
        assert cfg.input_size == 640
        # Source and output settings stay independent of the selected family.
        assert cfg.source_url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.insight_host == "127.0.0.1"

    def test_invalid_input_size_is_rejected(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  path: model.tar.gz
  input_size: 600
source:
  url: rtsp://127.0.0.1:8554/src1
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="input_size"):
            main.load_app_config(config)

    def test_invalid_mask_alpha_is_rejected(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  path: model.tar.gz
source:
  rtsp_url: rtsp://127.0.0.1:8554/src1
output:
  mask_alpha: 2
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="mask_alpha"):
            main.load_app_config(config)

    def test_an_empty_labels_value_is_rejected(self, tmp_path):
        """Path("") is ".", so the rule has to look at the raw value, not the Path."""
        config = tmp_path / "config.yaml"
        config.write_text(
            """
model:
  path: model.tar.gz
  labels: ""
source:
  url: rtsp://127.0.0.1:8554/src1
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="model.labels must be set"):
            main.load_app_config(config)

    @staticmethod
    def _source_config(tmp_path, url_line, legacy_line):
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""
model:
  path: model.tar.gz
source:
{url_line}
{legacy_line}
output:
  insight:
    host: 127.0.0.1
""",
            encoding="utf-8",
        )
        return config

    def test_an_empty_url_falls_back_to_the_legacy_key(self, tmp_path):
        """config.yaml ships source.url present and says rtsp_url is "used when
        source.url is empty", so the empty value must fall through."""
        config = self._source_config(
            tmp_path, '  url: ""', "  rtsp_url: rtsp://127.0.0.1:8554/legacy"
        )

        cfg = main.load_app_config(config)
        assert cfg.source_url == "rtsp://127.0.0.1:8554/legacy"
        assert cfg.source_key == "source.rtsp_url"

    def test_an_absent_url_falls_back_to_the_legacy_key(self, tmp_path):
        config = self._source_config(tmp_path, "", "  rtsp_url: rtsp://127.0.0.1:8554/legacy")

        assert main.load_app_config(config).source_url == "rtsp://127.0.0.1:8554/legacy"

    def test_a_present_url_wins_over_the_legacy_key(self, tmp_path):
        config = self._source_config(
            tmp_path,
            "  url: rtsp://127.0.0.1:8554/src1",
            "  rtsp_url: rtsp://127.0.0.1:8554/legacy",
        )

        cfg = main.load_app_config(config)
        assert cfg.source_url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.source_key == "source.url"

    def test_both_empty_is_still_rejected(self, tmp_path):
        config = self._source_config(tmp_path, '  url: ""', '  rtsp_url: ""')

        with pytest.raises(ValueError, match="source.url or source.rtsp_url must be set"):
            main.load_app_config(config)

    def test_validate_config_only_names_the_key_and_not_the_url(self, tmp_path):
        """The URL can carry credentials and this line ends up in logs, so the
        validated line reports the key that supplied the source. The C++ binary
        prints the same line and its unit suite asserts the same."""
        config = self._source_config(
            tmp_path, '  url: ""', "  rtsp_url: rtsp://user:secret@127.0.0.1:8554/legacy"
        )
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(EXAMPLE_DIR),
        )

        assert r.returncode == 0, r.stderr
        assert "(source=source.rtsp_url)" in r.stdout
        assert "secret" not in r.stdout
        assert "rtsp://" not in r.stdout


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# Each test starts from VALID_CONFIG and breaks exactly one thing, so a failure
# names the rule that fired rather than "config rejected". The legacy-URL
# fallback, the empty labels value and mask_alpha are owned by TestConfig above.
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "model": {"path": "model.tar.gz", "labels": "coco_label.txt"},
    "source": {
        "type": "rtsp",
        "codec": "h264",
        "url": "rtsp://127.0.0.1:8554/src1",
        "rtsp_url": "",
        "tcp": True,
        "latency_ms": 100,
        "fps": 0,
        "ssl_strict": True,
    },
    "inference": {"frames": 0, "min_score": 0.55, "nms_iou": 0.6, "max_detections": 50},
    "runtime": {"profile": False, "profile_interval": 100},
    "output": {
        "save_dir": "",
        "save_every": 0,
        "mask_alpha": 0.55,
        "mask_threshold": 0.5,
        "draw_boxes": True,
        "insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100},
    },
}


# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_full_config = config_writer(VALID_CONFIG)


@pytest.mark.unit
class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main.load_app_config(write_full_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert cfg.source_url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.source_type == "rtsp"
        assert cfg.source_codec == "h264"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "source": {"url": "rtsp://127.0.0.1:8554/src1"},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main.load_app_config(write_full_config(tmp_path, root=raw))

        assert cfg.source_type == "rtsp"
        assert cfg.source_codec == "h264"
        assert cfg.latency_ms == 200
        assert cfg.tcp is True
        assert cfg.source_fps == 0
        assert cfg.ssl_strict is True
        assert cfg.min_score == pytest.approx(0.55)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_detections == 50
        assert cfg.profile_interval == 100
        assert cfg.video_port == 9000
        assert cfg.metadata_port == 9100
        assert cfg.output.save_every == 0
        assert cfg.output.mask_alpha == pytest.approx(main.MASK_ALPHA)
        assert cfg.output.mask_threshold == pytest.approx(main.MASK_THRESHOLD)
        assert cfg.output.draw_boxes is True
        assert cfg.labels_path.name == "coco_label.txt"


REJECTED = [
    pytest.param(("model", "path"), "", "model.path must be set", id="model-path-empty"),
    pytest.param(
        ("output", "insight", "host"), "", "output.insight.host must be set", id="insight-host-empty"
    ),
    pytest.param(("source", "type"), "webrtc", "source.type must be rtsp or http", id="source-type-unsupported"),
    pytest.param(
        ("source", "codec"), "vp9", "source.codec must be h264/avc, h265/hevc, or mjpeg", id="codec-unsupported"
    ),
    pytest.param(("source", "latency_ms"), -1, "source.latency_ms must be >= 0", id="latency-negative"),
    pytest.param(("source", "fps"), -1, "source.fps must be >= 0", id="fps-negative"),
    pytest.param(("inference", "frames"), -1, "inference.frames must be >= 0", id="frames-negative"),
    pytest.param(
        ("inference", "min_score"), -0.01, "inference.min_score must be between 0 and 1", id="min-score-below"
    ),
    pytest.param(
        ("inference", "min_score"), 1.01, "inference.min_score must be between 0 and 1", id="min-score-above"
    ),
    pytest.param(
        ("inference", "nms_iou"), -0.01, "inference.nms_iou must be between 0 and 1", id="nms-below"
    ),
    pytest.param(
        ("inference", "nms_iou"), 1.01, "inference.nms_iou must be between 0 and 1", id="nms-above"
    ),
    pytest.param(
        ("inference", "max_detections"), 0, "inference.max_detections must be > 0", id="max-detections-zero"
    ),
    pytest.param(
        ("runtime", "profile_interval"), 0, "runtime.profile_interval must be > 0", id="profile-interval-zero"
    ),
    pytest.param(
        ("output", "insight", "video_port"), 0, "output.insight.video_port must be > 0", id="video-port-zero"
    ),
    pytest.param(
        ("output", "insight", "metadata_port"),
        0,
        "output.insight.metadata_port must be > 0",
        id="metadata-port-zero",
    ),
    pytest.param(("output", "save_every"), -1, "output.save_every must be >= 0", id="save-every-negative"),
    pytest.param(
        ("output", "mask_threshold"), 1.01, "output.mask_threshold must be between 0 and 1", id="mask-threshold-above"
    ),
    pytest.param(
        ("output", "mask_threshold"), -0.01, "output.mask_threshold must be between 0 and 1", id="mask-threshold-below"
    ),
]


@pytest.mark.unit
class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main.load_app_config(write_full_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("inference", "min_score"), 0.0),
            (("inference", "min_score"), 1.0),
            (("inference", "nms_iou"), 0.0),
            (("inference", "nms_iou"), 1.0),
            (("output", "mask_alpha"), 0.0),
            (("output", "mask_alpha"), 1.0),
            (("output", "mask_threshold"), 0.0),
            (("output", "mask_threshold"), 1.0),
            (("source", "latency_ms"), 0),
            (("source", "fps"), 0),
            (("inference", "frames"), 0),
            (("output", "save_every"), 0),
            (("inference", "max_detections"), 1),
            (("runtime", "profile_interval"), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main.load_app_config(write_full_config(tmp_path, [(path, value)]))


@pytest.mark.unit
class TestSourceCombinations:
    def test_http_source_requires_mjpeg(self, tmp_path):
        config_path = write_full_config(
            tmp_path, [(("source", "type"), "http"), (("source", "codec"), "h264")]
        )

        with pytest.raises(ValueError, match="source.codec must be mjpeg for source.type=http"):
            main.load_app_config(config_path)

    def test_http_source_with_mjpeg_is_accepted(self, tmp_path):
        cfg = main.load_app_config(
            write_full_config(tmp_path, [(("source", "type"), "http"), (("source", "codec"), "mjpeg")])
        )

        assert cfg.source_type == "http"
        assert cfg.source_codec == "mjpeg"

    @pytest.mark.parametrize(
        ("alias", "codec"),
        [("h264", "h264"), ("avc", "h264"), ("h265", "h265"), ("hevc", "h265"), ("mjpeg", "mjpeg")],
    )
    def test_every_documented_rtsp_codec_is_accepted(self, tmp_path, alias, codec):
        cfg = main.load_app_config(write_full_config(tmp_path, [(("source", "codec"), alias)]))

        assert cfg.source_codec == codec


@pytest.mark.unit
class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main.load_app_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="must be set"):
            main.load_app_config(config_path)

    @pytest.mark.parametrize("key", ["path", "labels"])
    def test_a_non_string_model_value_is_rejected(self, tmp_path, key):
        with pytest.raises(ValueError, match=f"{key} must be a string"):
            main.load_app_config(write_full_config(tmp_path, [(("model", key), 42)]))


@pytest.mark.unit
class TestMaskOverlay:
    def setup_method(self):
        main.load_runtime_dependencies()

    def test_class_color_uses_vivid_palette(self):
        assert main.class_color(0) == (56, 56, 255)
        assert main.class_color(1) == (151, 157, 255)
        assert main.class_color(2) == (31, 112, 255)

    def test_overlay_segmentation_changes_pixels(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[40:120, 40:120] = 255
        dets = [
            {
                "x1": 16.0,
                "y1": 16.0,
                "x2": 48.0,
                "y2": 48.0,
                "score": 0.9,
                "class_id": 1,
                "mask": mask,
            }
        ]
        cfg = main.OutputConfig("", 0, 0.5, 0.5, True)

        out = main.overlay_segmentation(frame, dets, 0.55, cfg, ["person", "bicycle"])

        assert int(out.sum()) > 0

    def test_overlay_segmentation_can_skip_boxes(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        mask[40:120, 40:120] = 255
        dets = [
            {
                "x1": 16.0,
                "y1": 16.0,
                "x2": 48.0,
                "y2": 48.0,
                "score": 0.9,
                "class_id": 2,
                "mask": mask,
            }
        ]
        cfg = main.OutputConfig("", 0, 0.5, 0.5, False)

        out = main.overlay_segmentation(frame, dets, 0.55, cfg, ["person", "bicycle", "car"])

        assert int(out.sum()) > 0


@pytest.mark.unit
class TestSegmentationMetadata:
    def setup_method(self):
        main.load_runtime_dependencies()

    def _segment(self, id_: str, confidence: float, points: int) -> dict:
        return {
            "id": id_,
            "label": "person",
            "confidence": confidence,
            "bbox": [0, 0, 64, 64],
            "mask_format": "polygon",
            "mask": [[i % 64, i // 64] for i in range(points)],
        }

    def test_polygon_is_frame_absolute_and_in_bounds(self):
        mask = np.full((160, 160), 255, dtype=np.uint8)

        polygon = main.mask_polygon(mask, (1080, 1920, 3), (1600, 900, 1900, 1070), 0.5)

        assert len(polygon) >= 3
        assert all(1600 <= x <= 1900 and 900 <= y <= 1070 for x, y in polygon)

    def test_polygon_is_empty_without_foreground(self):
        mask = np.zeros((160, 160), dtype=np.uint8)

        assert main.mask_polygon(mask, (640, 640, 3), (0, 0, 64, 64), 0.5) == []

    def test_metadata_segments_shape(self):
        mask = np.full((160, 160), 255, dtype=np.uint8)
        dets = [{"x1": 8.0, "y1": 8.0, "x2": 56.0, "y2": 56.0, "score": 0.9, "class_id": 0,
                 "mask": mask}]

        segments = main.metadata_segments(dets, ["person"], (64, 64, 3), 0.5)

        assert len(segments) == 1
        assert segments[0]["id"] == "seg_1"
        assert segments[0]["label"] == "person"
        assert segments[0]["mask_format"] == "polygon"
        assert segments[0]["bbox"] == [8, 8, 48, 48]
        assert len(segments[0]["mask"]) >= 3

    def test_encode_segments_emits_segments_array(self):
        data_json, dropped = main.encode_segments([self._segment("seg_1", 0.9, 5)])

        data = json.loads(data_json)
        assert dropped == 0
        assert len(data["segments"]) == 1
        assert data["segments"][0]["mask_format"] == "polygon"
        assert len(data["segments"][0]["mask"]) == 5

    def test_budget_drops_lowest_confidence_first(self):
        segments = [self._segment(f"seg_{i}", 0.5 + 0.01 * i, 1000) for i in range(12)]

        data_json, dropped = main.encode_segments(segments)

        data = json.loads(data_json)
        assert len(data_json) <= main.METADATA_BYTE_BUDGET
        assert 0 < len(data["segments"]) < len(segments)
        assert len(data["segments"]) + dropped == len(segments)
        lowest_kept = min(segment["confidence"] for segment in data["segments"])
        assert lowest_kept == pytest.approx(0.5 + 0.01 * dropped)


@pytest.mark.unit
class TestSampleAccess:
    """The pulled sample is a bundle only when save_dir added a frame branch to combine."""

    class _Kind:
        Bundle = "bundle"
        Tensor = "tensor"
        TensorSet = "tensorset"

    class _Sample:
        def __init__(self, kind, *, tensors=(), fields=(), stream_label=""):
            self.kind = kind
            self.tensors = list(tensors)
            self.tensor = None
            self.fields = list(fields)
            self.stream_label = stream_label

    @pytest.fixture(autouse=True)
    def _stub_pyneat(self, monkeypatch):
        monkeypatch.setattr(main, "pyneat", type("P", (), {"SampleKind": self._Kind}))

    def test_unjoined_sample_is_the_segments_payload(self):
        sample = self._Sample(self._Kind.TensorSet, tensors=["boxes", "masks"])

        assert main.segment_tensors_from_sample(sample) == ["boxes", "masks"]

    def test_bundled_sample_resolves_the_segments_field(self):
        frame = self._Sample(self._Kind.TensorSet, tensors=["frame"], stream_label="frame")
        segments = self._Sample(self._Kind.TensorSet, tensors=["boxes"], stream_label="segments")
        bundle = self._Sample(self._Kind.Bundle, fields=[frame, segments])

        assert main.segment_tensors_from_sample(bundle) == ["boxes"]


def load_parity_fixture() -> dict:
    """The decode contract both implementations are held to."""
    with PARITY_FIXTURE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_yolov8_heads(spec: dict) -> list:
    """Synthetic YOLOv8 head tensors described by the shared fixture."""
    input_size = int(spec["input_size"])
    grids = [input_size // stride for stride in main.YOLOV8_STRIDES]
    boxes = [np.zeros((grid, grid, 4 * main.DFL_BINS), np.float32) for grid in grids]
    scores = [
        np.zeros((grid, grid, int(spec["class_count"])), np.float32) for grid in grids
    ]
    coefficients = [
        np.zeros((grid, grid, main.MASK_COEFFICIENTS), np.float32) for grid in grids
    ]
    proto_grid = input_size // main.MASK_STRIDE
    proto = np.zeros((proto_grid, proto_grid, main.MASK_COEFFICIENTS), np.float32)

    for channel in spec["prototype"]:
        index = int(channel["channel"])
        proto[:, :, index] = channel["background"]
        x0, y0, x1, y1 = channel["rect"]
        proto[y0:y1, x0:x1, index] = channel["foreground"]

    for cell in spec["cells"]:
        level, row, column = int(cell["level"]), int(cell["row"]), int(cell["column"])
        sides = np.full((4 * main.DFL_BINS,), -8.0, np.float32)
        for side in range(4):
            sides[side * main.DFL_BINS + int(cell["dfl_bin"])] = 8.0
        boxes[level][row, column, :] = sides
        scores[level][row, column, int(cell["class_id"])] = cell["score"]
        coefficients[level][row, column, int(cell["coefficient"])] = 1.0

    return boxes + scores + coefficients + [proto]


def decode_fixture(spec: dict) -> list:
    return main.decode_yolov8_segments(
        build_yolov8_heads(spec),
        int(spec["frame"]["width"]),
        int(spec["frame"]["height"]),
        int(spec["input_size"]),
        float(spec["min_score"]),
        float(spec["nms_iou"]),
        int(spec["max_detections"]),
    )


@pytest.mark.unit
class TestYolov8Decode:
    """The YOLOv8 side of the model-specific decoding boundary."""

    def setup_method(self):
        main.load_runtime_dependencies()

    def test_decode_matches_shared_parity_fixture(self):
        fixture = load_parity_fixture()
        spec = fixture["input"]

        detections = decode_fixture(spec)

        expected = fixture["expected"]["detections"]
        assert len(detections) == len(expected)
        for det, want in zip(detections, expected):
            assert det["class_id"] == want["class_id"]
            assert det["score"] == pytest.approx(want["score"], abs=1e-6)
            box = [det["x1"], det["y1"], det["x2"], det["y2"]]
            assert box == pytest.approx(want["box"], abs=1e-3)
            assert det["mask"].shape == (want["mask_grid"], want["mask_grid"])
            assert det["mask"].dtype == np.uint8
            assert int((det["mask"] > 127).sum()) == want["mask_above_threshold"]

    def test_metadata_matches_shared_parity_fixture(self):
        fixture = load_parity_fixture()
        spec = fixture["input"]
        frame_shape = (int(spec["frame"]["height"]), int(spec["frame"]["width"]), 3)
        labels = [f"class_{index}" for index in range(int(spec["class_count"]))]

        segments = main.metadata_segments(
            decode_fixture(spec), labels, frame_shape, float(spec["mask_threshold"])
        )

        expected = fixture["expected"]["segments"]
        assert len(segments) == len(expected)
        for segment, want in zip(segments, expected):
            assert segment["id"] == want["id"]
            assert segment["label"] == want["label"]
            assert segment["confidence"] == pytest.approx(want["confidence"], abs=1e-6)
            assert segment["bbox"] == want["bbox"]
            assert segment["mask_format"] == want["mask_format"]
            assert len(segment["mask"]) >= want["min_polygon_points"]

    def test_confidence_is_the_packaged_class_probability(self):
        spec = load_parity_fixture()["input"]

        detections = decode_fixture(spec)

        # The packaged class head emits probabilities, so no activation is applied here.
        assert [round(det["score"], 6) for det in detections] == [0.9, 0.74]

    def test_scores_below_the_threshold_are_dropped(self):
        spec = dict(load_parity_fixture()["input"], min_score=0.95)

        assert decode_fixture(spec) == []

    def test_letterboxed_boxes_land_in_frame_pixels(self):
        spec = load_parity_fixture()["input"]
        frame_w, frame_h = int(spec["frame"]["width"]), int(spec["frame"]["height"])

        detections = decode_fixture(spec)

        assert detections
        for det in detections:
            assert 0.0 <= det["x1"] < det["x2"] <= frame_w
            assert 0.0 <= det["y1"] < det["y2"] <= frame_h

    def test_head_shapes_are_checked_against_input_size(self):
        spec = load_parity_fixture()["input"]
        heads = build_yolov8_heads(spec)

        with pytest.raises(RuntimeError, match="model.input_size"):
            main.split_yolov8_heads(heads, 320)

    def test_missing_head_tensors_fail_clearly(self):
        spec = load_parity_fixture()["input"]

        with pytest.raises(RuntimeError, match="head tensors"):
            main.split_yolov8_heads(build_yolov8_heads(spec)[:-1], int(spec["input_size"]))


@pytest.mark.unit
class TestDecodeBoundary:
    """Family selection is the only thing that changes between the two decode paths."""

    def setup_method(self):
        main.load_runtime_dependencies()

    def _config(self, family: str) -> object:
        return main.AppConfig(
            model_path="model.tar.gz",
            labels_path=Path("labels.txt"),
            source_url="rtsp://127.0.0.1:8554/src1",
            model_family=family,
            insight_host="127.0.0.1",
        )

    def test_family_selects_the_decoder(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            main, "decode_yolo26_segments", lambda *args: calls.append("yolo26") or []
        )
        monkeypatch.setattr(
            main, "decode_yolov8_segments", lambda *args: calls.append("yolov8") or []
        )
        monkeypatch.setattr(main, "tensor_to_hwc_f32", lambda tensor: tensor)

        main.decode_segments(self._config(main.YOLO26), ["payload"], 1920, 1080)
        main.decode_segments(self._config(main.YOLOV8), ["payload"] * 10, 1920, 1080)

        assert calls == ["yolo26", "yolov8"]

    def test_both_families_produce_the_same_record(self):
        spec = load_parity_fixture()["input"]
        mask = np.full((160, 160), 255, dtype=np.uint8)
        # decode_yolo26_segments() builds its records through detection() as well, so one
        # record from each path is compared field by field.
        yolo26_record = main.detection(8.0, 8.0, 56.0, 56.0, 0.9, 0, mask)
        yolov8_record = decode_fixture(spec)[0]

        assert sorted(yolov8_record) == sorted(yolo26_record)
        for key, value in yolo26_record.items():
            assert type(yolov8_record[key]) is type(value)


@pytest.mark.unit
class TestFramePairing:
    """YOLOv8 pairs frames with segments here, so a saved frame is the one it was decoded from."""

    def _runtime(self, frame_ids):
        return SimpleNamespace(
            frames=[(frame_id, f"frame_{frame_id}") for frame_id in frame_ids],
            frame_output_name="frame",
        )

    def test_returns_the_frame_the_segments_came_from(self):
        runtime = self._runtime([7, 8, 9])

        assert main.frame_for(runtime, 8) == "frame_8"

    def test_returns_none_once_the_frame_has_aged_out(self):
        runtime = self._runtime([7, 8, 9])

        assert main.frame_for(runtime, 3) is None

    def test_unidentified_samples_are_never_paired(self):
        # A sample without a frame identity cannot be matched to a frame; saving is skipped
        # rather than pairing an overlay onto the wrong picture.
        runtime = self._runtime([7, 8, 9])

        assert main.frame_for(runtime, -1) is None

    def test_ring_keeps_the_newest_frames(self, monkeypatch):
        monkeypatch.setattr(main, "host_frame_copy", lambda sample: f"copy_{sample.frame_id}")
        pulled = [SimpleNamespace(frame_id=index) for index in range(main.FRAME_RING_CAPACITY + 4)]
        runtime = SimpleNamespace(frames=[], frame_output_name="frame",
                                  run=SimpleNamespace(pull=lambda name, timeout: (
                                      pulled.pop(0) if pulled else None)))

        main.drain_frames(runtime)

        assert len(runtime.frames) == main.FRAME_RING_CAPACITY
        newest = main.FRAME_RING_CAPACITY + 3
        assert runtime.frames[-1] == (newest, f"copy_{newest}")

    def test_ring_holds_host_copies_not_pulled_samples(self, monkeypatch):
        # A retained sample holds a decoder-buffer loan, and the decoder stalls once its
        # in-flight frames are all on loan, so only the copied pixels may outlive the drain.
        pulled = [SimpleNamespace(frame_id=1)]
        copied = []

        def copy(sample):
            copied.append(sample)
            return np.zeros((6, 4), dtype=np.uint8)

        monkeypatch.setattr(main, "host_frame_copy", copy)
        runtime = SimpleNamespace(frames=[], frame_output_name="frame",
                                  run=SimpleNamespace(pull=lambda name, timeout: (
                                      pulled.pop(0) if pulled else None)))

        main.drain_frames(runtime)

        assert len(copied) == 1
        assert all(retained is not copied[0] for _, retained in runtime.frames)

    def test_retained_nv12_converts_to_bgr_when_saved(self):
        main.load_runtime_dependencies()
        bgr =main.bgr_from_host_frame(np.full((6, 4), 128, dtype=np.uint8))

        assert bgr.shape == (4, 4, 3)

    def test_retained_bgr_is_used_as_it_is(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        assert main.bgr_from_host_frame(frame) is frame


@pytest.mark.unit
class TestModelPreprocess:
    """Both families must request the same input preparation.

    A YOLOv8 package whose normalization is left to chance sees unnormalized pixels and its
    class scores collapse to a fraction of their range, which looks like a decode bug but is
    not one. This pins the request for both families.
    """

    class _Preprocess:
        def __init__(self):
            self.kind = None
            self.enable = None
            self.preset = None
            self.input_max_width = 0
            self.input_max_height = 0
            self.color_convert = SimpleNamespace(input_format=None)
            self.resize = SimpleNamespace(enable=None, mode=None)

    class _Options:
        def __init__(self):
            self.preprocess = TestModelPreprocess._Preprocess()
            self.decode_type = None
            self.score_threshold = 0.0
            self.nms_iou_threshold = 0.0
            self.top_k = 0

    @pytest.fixture
    def stub(self, monkeypatch):
        captured = {}

        def model(path, opt):
            captured["path"] = path
            captured["opt"] = opt
            return object()

        monkeypatch.setattr(
            main,
            "pyneat",
            SimpleNamespace(
                ModelOptions=TestModelPreprocess._Options,
                Model=model,
                InputKind=SimpleNamespace(Image="image"),
                AutoFlag=SimpleNamespace(On="on", Auto="auto"),
                NormalizePreset=SimpleNamespace(COCO_YOLO="coco_yolo"),
                PreprocessColorFormat=SimpleNamespace(NV12="nv12"),
                ResizeMode=SimpleNamespace(Letterbox="letterbox"),
                BoxDecodeType=SimpleNamespace(YoloV26Seg="yolo26seg"),
            ),
        )
        return captured

    def _config(self, family):
        return main.AppConfig(
            model_path="model.tar.gz",
            labels_path=Path("labels.txt"),
            source_url="rtsp://127.0.0.1:8554/src1",
            model_family=family,
            insight_host="127.0.0.1",
        )

    @pytest.mark.parametrize("family", [main.YOLO26, main.YOLOV8])
    def test_normalization_is_requested_for_both_families(self, stub, family):
        main.make_model(self._config(family), 1920, 1080)

        preprocess = stub["opt"].preprocess
        assert preprocess.enable == "on"
        assert preprocess.preset == "coco_yolo"
        assert preprocess.color_convert.input_format == "nv12"
        assert (preprocess.input_max_width, preprocess.input_max_height) == (1920, 1080)

    def test_only_yolo26_decodes_on_device(self, stub):
        main.make_model(self._config(main.YOLO26), 1920, 1080)
        assert stub["opt"].decode_type == "yolo26seg"

        main.make_model(self._config(main.YOLOV8), 1920, 1080)
        # YOLOv8 heads are decoded on the host, and that decode inverts a letterbox.
        assert stub["opt"].decode_type is None
        assert stub["opt"].preprocess.resize.mode == "letterbox"


@pytest.mark.unit
class TestPullOutcomes:
    """The pull loop, driven by a run that yields no sample.

    A timeout is a warning and another pull, and a runtime error raised by the pull ends
    the run without a summary. pyneat's pull returns None for a closed output as for a
    timeout, so Python cannot yet tell a source that ended from a slow one.
    """

    def test_timeout_is_not_a_sample(self):
        run = FakeRun("timeout")
        run.pull("segments", 20000)

        assert main.pull_result_has_sample(run, None, "segments") is False

    def test_run_pipeline_warns_on_timeout_and_stops_on_runtime_error(self, capsys):
        run = FakeRun("timeout", ("error", "queue torn down"))
        runtime = SimpleNamespace(run=run, output_name="segments", frame_output_name="")
        cfg = SimpleNamespace(frames=0, profile=False, profile_interval=1)

        with pytest.raises(RuntimeError, match="queue torn down"):
            main.run_pipeline(runtime, cfg)

        captured = capsys.readouterr()
        assert captured.err.count("[warn] timed out waiting for segmentation output") == 1
        assert "processed=" not in captured.out
        assert run.pulls == [("segments", 20000)] * 2
