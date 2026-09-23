"""Unit tests for configuration handling and option validation.

`validate_config` and the `streams` parsing in `load_app_config` decide whether
a mistyped `config.yaml` becomes a clear error or a confusing runtime failure,
and none of those rules were covered. Each test starts from a valid baseline
and breaks one thing, so a failure names the rule that fired.

No model, no stream, no Neat runtime: `main` defers those imports.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def _load_main():
    """Load this example's main.py under a name unique to the example.

    Every example has a module called `main`, so a plain `import main` binds
    whichever one was imported first. tests/test.sh runs an isolated pytest per
    example and never hits that, but a developer running pytest across examples
    would silently test the wrong application.
    """
    spec = importlib.util.spec_from_file_location(
        f"{EXAMPLE_DIR.name.replace('-', '_')}_main", PYTHON_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


main = _load_main()

pytestmark = pytest.mark.unit


VALID_CONFIG = {
    "model": {"path": "model.tar.gz", "labels": "coco_label.txt"},
    "streams": ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
    "input": {"codec": "h264", "latency_ms": 100, "tcp": True},
    "inference": {
        "frames": 0,
        "fps": 0,
        "max_inflight_per_stream": 4,
        "max_inflight_total": 16,
        "min_score": 0.55,
        "nms_iou": 0.6,
        "max_detections": 50,
    },
    "runtime": {"profile": False, "warmup_frames": 30},
    "output": {
        "save_every": 0,
        "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100},
    },
}


def write_config(tmp_path: Path, overrides=None, *, root=None) -> Path:
    raw = copy.deepcopy(VALID_CONFIG) if root is None else root
    for path, value in (overrides or []):
        target = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return config_path


class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main.load_app_config(write_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert len(cfg.rtsp_urls) == 2
        assert cfg.codec == "h264"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "streams": ["rtsp://127.0.0.1:8554/src1"],
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main.load_app_config(write_config(tmp_path, root=raw))

        assert cfg.codec == "h264"
        assert cfg.latency_ms == 100
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16
        assert cfg.max_detections == 50
        assert cfg.warmup_frames == 30
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100


REJECTED = [
    pytest.param(("model", "path"), "", "model.path must be set", id="model-path-empty"),
    pytest.param(
        ("output", "insight", "host"), "", "output.insight.host must be set", id="insight-host-empty"
    ),
    pytest.param(("input", "latency_ms"), -1, "input.latency_ms must be >= 0", id="latency-negative"),
    pytest.param(("inference", "frames"), -1, "inference.frames must be >= 0", id="frames-negative"),
    pytest.param(("inference", "fps"), -1, "inference.fps must be >= 0", id="fps-negative"),
    pytest.param(
        ("inference", "min_score"), 1.01, "inference.min_score must be between 0 and 1", id="min-score-above"
    ),
    pytest.param(
        ("inference", "nms_iou"), -0.01, "inference.nms_iou must be between 0 and 1", id="nms-below"
    ),
    pytest.param(
        ("inference", "max_detections"), 0, "inference.max_detections must be > 0", id="max-detections-zero"
    ),
    pytest.param(
        ("runtime", "warmup_frames"), -1, "runtime.warmup_frames must be >= 0", id="warmup-negative"
    ),
    pytest.param(
        ("output", "insight", "video_port_base"),
        0,
        "output.insight.video_port_base must be > 0",
        id="video-port-base-zero",
    ),
    pytest.param(
        ("output", "insight", "metadata_port_base"),
        0,
        "output.insight.metadata_port_base must be > 0",
        id="metadata-port-base-zero",
    ),
    pytest.param(("output", "save_every"), -1, "output.save_every must be >= 0", id="save-every-negative"),
]


class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main.load_app_config(write_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


class TestStreamList:
    """`streams` is the option that makes this application multistream."""

    def test_an_empty_stream_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main.load_app_config(write_config(tmp_path, [(("streams",), [])]))

    def test_a_missing_stream_list_is_rejected(self, tmp_path):
        raw = copy.deepcopy(VALID_CONFIG)
        del raw["streams"]

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main.load_app_config(write_config(tmp_path, root=raw))

    def test_a_scalar_instead_of_a_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main.load_app_config(
                write_config(tmp_path, [(("streams",), "rtsp://127.0.0.1:8554/src1")])
            )

    @pytest.mark.parametrize(
        ("streams", "bad_index"),
        [
            ([""], 0),
            (["   "], 0),
            (["rtsp://127.0.0.1:8554/src1", ""], 1),
            (["rtsp://127.0.0.1:8554/src1", 42], 1),
        ],
    )
    def test_a_blank_or_non_string_entry_names_its_index(self, tmp_path, streams, bad_index):
        """The index matters: with four streams, "one of them is wrong" is not actionable."""
        with pytest.raises(ValueError, match=rf"streams\[{bad_index}\] must be a non-empty string"):
            main.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    def test_up_to_four_streams_are_accepted(self, tmp_path, count):
        streams = [f"rtsp://127.0.0.1:8554/src{index}" for index in range(count)]

        cfg = main.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

        assert len(cfg.rtsp_urls) == count

    def test_a_fifth_stream_is_rejected_at_the_documented_cap(self, tmp_path):
        streams = [f"rtsp://127.0.0.1:8554/src{index}" for index in range(5)]

        with pytest.raises(ValueError, match="this phase supports up to four streams"):
            main.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

    def test_stream_order_is_preserved(self, tmp_path):
        """Stream order decides which Insight port each stream publishes on."""
        streams = ["rtsp://host/c", "rtsp://host/a", "rtsp://host/b"]

        cfg = main.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

        assert cfg.rtsp_urls == streams


class TestInflightSentinels:
    """-1 means unbounded; 0 and negatives other than -1 are mistakes."""

    @pytest.mark.parametrize("key", ["max_inflight_per_stream", "max_inflight_total"])
    @pytest.mark.parametrize("value", [-1, 1, 64])
    def test_unbounded_and_positive_values_are_accepted(self, tmp_path, key, value):
        cfg = main.load_app_config(write_config(tmp_path, [(("inference", key), value)]))

        assert getattr(cfg, key) == value

    @pytest.mark.parametrize("key", ["max_inflight_per_stream", "max_inflight_total"])
    @pytest.mark.parametrize("value", [0, -2])
    def test_zero_and_other_negatives_are_rejected(self, tmp_path, key, value):
        with pytest.raises(ValueError, match=f"inference.{key} must be -1 or > 0"):
            main.load_app_config(write_config(tmp_path, [(("inference", key), value)]))


class TestCodecOption:
    @pytest.mark.parametrize("codec", ["h264", "h265"])
    def test_each_documented_codec_is_accepted(self, tmp_path, codec):
        cfg = main.load_app_config(write_config(tmp_path, [(("input", "codec"), codec)]))

        assert cfg.codec == codec

    def test_an_unsupported_codec_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            main.load_app_config(write_config(tmp_path, [(("input", "codec"), "vp9")]))


class TestBoundariesAreAccepted:
    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("inference", "min_score"), 0.0),
            (("inference", "min_score"), 1.0),
            (("inference", "nms_iou"), 0.0),
            (("inference", "nms_iou"), 1.0),
            (("input", "latency_ms"), 0),
            (("inference", "frames"), 0),
            (("inference", "fps"), 0),
            (("runtime", "warmup_frames"), 0),
            (("output", "save_every"), 0),
            (("inference", "max_detections"), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main.load_app_config(write_config(tmp_path, [(path, value)]))


class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main.load_app_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main.load_app_config(config_path)
