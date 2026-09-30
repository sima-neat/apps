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
    cfg = main.AppConfig("model", Path("labels"), url, source_type, tcp=tcp,
                         ssl_strict=False)

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


@pytest.mark.unit
class TestPullOutcomes:
    """The pull loop, driven by a run that yields no sample.

    A timeout is a warning and another pull; a closed output and a runtime error end the
    run with a message, so a dead source is neither a healthy wait nor a completed run.
    """

    def test_timeout_is_not_a_sample(self):
        run = FakeRun("timeout")
        run.pull("segments", 20000)

        assert main.pull_result_has_sample(run, None, "segments") is False

    def test_closed_output_ends_the_run_with_the_reason(self):
        run = FakeRun(("closed", "source reached EOS"))
        run.pull("segments", 20000)

        with pytest.raises(
            RuntimeError, match="segments output closed unexpectedly: source reached EOS"
        ):
            main.pull_result_has_sample(run, None, "segments")

    def test_runtime_error_ends_the_run(self):
        run = FakeRun(("error", "queue torn down"))
        run.pull("segments", 20000)

        with pytest.raises(RuntimeError, match="runtime error: queue torn down"):
            main.pull_result_has_sample(run, None, "segments")

    def test_run_pipeline_warns_on_timeout_and_stops_on_closed_output(self, capsys):
        run = FakeRun("timeout", ("closed", "source reached EOS"))
        runtime = SimpleNamespace(run=run, output_name="segments")
        cfg = SimpleNamespace(frames=0, profile=False, profile_interval=1)

        with pytest.raises(RuntimeError, match="segments output closed unexpectedly"):
            main.run_pipeline(runtime, cfg)

        captured = capsys.readouterr()
        assert captured.err.count("[warn] timed out waiting for segmentation output") == 1
        assert "processed=" not in captured.out
        assert run.pulls == [("segments", 20000)] * 2
