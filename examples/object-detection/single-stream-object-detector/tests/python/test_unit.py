"""Unit tests for single-stream-object-detector (Python)."""
import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

_SPEC = importlib.util.spec_from_file_location("object_detector_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


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
    """Validate CLI argument parsing for the single RTSP Insight detection pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        """Missing config should fail."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/single-rtsp-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode != 0

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


# Every required field set to something acceptable. Each test copies this and
# breaks exactly one thing, so a failure names the rule that fired.
VALID_CONFIG = {
    "model": {"path": "model.tar.gz", "labels": "coco_label.txt"},
    "source": {
        "type": "rtsp",
        "codec": "h264",
        "url": "rtsp://127.0.0.1:8554/src1",
        "latency_ms": 200,
        "fps": 0,
    },
    "inference": {"frames": 0, "min_score": 0.55, "nms_iou": 0.6, "max_detections": 50},
    "runtime": {"profile": False, "profile_interval": 100},
    "output": {
        "save_dir": "",
        "save_every": 0,
        "insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100},
    },
}


def write_config(tmp_path: Path, overrides=None, *, root=None) -> Path:
    """Write the baseline config with `overrides` applied as (section, key, value)."""
    raw = copy.deepcopy(VALID_CONFIG) if root is None else root
    for path, value in (overrides or []):
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


@pytest.mark.unit
class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main.load_app_config(write_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert cfg.source_type == "rtsp"
        assert cfg.source_codec == "h264"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "source": {"url": "rtsp://127.0.0.1:8554/src1"},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main.load_app_config(write_config(tmp_path, root=raw))

        assert cfg.source_type == "rtsp"
        assert cfg.source_codec == "h264"
        assert cfg.latency_ms == 200
        assert cfg.min_score == pytest.approx(0.55)
        assert cfg.nms_iou == pytest.approx(0.60)
        assert cfg.max_detections == 50
        assert cfg.profile_interval == 100
        assert cfg.video_port == 9000
        assert cfg.metadata_port == 9100


REJECTED = [
    pytest.param(("model", "path"), "", "model.path must be set", id="model-path-empty"),
    pytest.param(
        ("output", "insight", "host"), "", "output.insight.host must be set", id="insight-host-empty"
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
]


@pytest.mark.unit
class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        config_path = write_config(tmp_path, [(path, value)])

        with pytest.raises(ValueError) as excinfo:
            main.load_app_config(config_path)

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestBoundariesAreAccepted:
    """The edges of each documented range are valid, not off-by-one rejections."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("inference", "min_score"), 0.0),
            (("inference", "min_score"), 1.0),
            (("inference", "nms_iou"), 0.0),
            (("inference", "nms_iou"), 1.0),
            (("source", "latency_ms"), 0),
            (("source", "fps"), 0),
            (("inference", "frames"), 0),
            (("output", "save_every"), 0),
            (("inference", "max_detections"), 1),
            (("runtime", "profile_interval"), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main.load_app_config(write_config(tmp_path, [(path, value)]))


@pytest.mark.unit
class TestSourceCombinations:
    def test_http_source_requires_mjpeg(self, tmp_path):
        config_path = write_config(
            tmp_path, [(("source", "type"), "http"), (("source", "codec"), "h264")]
        )

        with pytest.raises(ValueError, match="source.codec must be mjpeg for source.type=http"):
            main.load_app_config(config_path)

    def test_http_source_with_mjpeg_is_accepted(self, tmp_path):
        cfg = main.load_app_config(
            write_config(tmp_path, [(("source", "type"), "http"), (("source", "codec"), "mjpeg")])
        )

        assert cfg.source_type == "http"
        assert cfg.source_codec == "mjpeg"

    @pytest.mark.parametrize("codec", ["h264", "h265", "mjpeg"])
    def test_every_documented_rtsp_codec_is_accepted(self, tmp_path, codec):
        cfg = main.load_app_config(write_config(tmp_path, [(("source", "codec"), codec)]))

        assert cfg.source_codec == codec

    def test_an_unsupported_codec_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            main.load_app_config(write_config(tmp_path, [(("source", "codec"), "vp9")]))

    def test_an_unsupported_source_type_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            main.load_app_config(write_config(tmp_path, [(("source", "type"), "webrtc")]))

    def test_rtsp_url_is_read_from_the_legacy_key(self, tmp_path):
        """`source.rtsp_url` predates `source.url` and configs still in the field use it."""
        raw = copy.deepcopy(VALID_CONFIG)
        del raw["source"]["url"]
        raw["source"]["rtsp_url"] = "rtsp://127.0.0.1:8554/legacy"

        cfg = main.load_app_config(write_config(tmp_path, root=raw))

        assert cfg.source_url == "rtsp://127.0.0.1:8554/legacy"


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

        with pytest.raises(ValueError, match="source.url or source.rtsp_url must be set"):
            main.load_app_config(config_path)

    def test_a_missing_source_url_is_rejected(self, tmp_path):
        raw = copy.deepcopy(VALID_CONFIG)
        del raw["source"]["url"]

        with pytest.raises(ValueError, match="source.url or source.rtsp_url must be set"):
            main.load_app_config(write_config(tmp_path, root=raw))

    @pytest.mark.parametrize("key", ["path", "labels"])
    def test_a_non_string_where_a_string_belongs_is_rejected(self, tmp_path, key):
        with pytest.raises(ValueError, match=f"{key} must be a string"):
            main.load_app_config(write_config(tmp_path, [(("model", key), 42)]))


@pytest.mark.unit
class TestLabelsPathHandling:
    """`model.labels` has a rule that the config file cannot actually trigger."""

    def test_an_empty_labels_value_resolves_to_the_current_directory(self, tmp_path):
        # string_or only falls back when the key is absent, so "" survives and
        # Path("") becomes ".", which is truthy. The "model.labels must be set"
        # rule is therefore unreachable from a config file. Recorded here so the
        # behaviour is deliberate rather than assumed; a fix would make this fail.
        cfg = main.load_app_config(write_config(tmp_path, [(("model", "labels"), "")]))

        assert str(cfg.labels_path) == "."

    def test_the_rule_still_fires_when_the_path_really_is_empty(self):
        cfg = main.AppConfig(
            model_path="model.tar.gz",
            labels_path="",
            source_url="rtsp://127.0.0.1:8554/src1",
            insight_host="127.0.0.1",
        )

        with pytest.raises(ValueError, match="model.labels must be set"):
            main.validate_config(cfg)
