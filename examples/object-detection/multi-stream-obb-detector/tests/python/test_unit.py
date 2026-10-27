"""Unit tests for the multi-stream oriented-box detector."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
# The C++ unit test checks the same fixture, so both implementations publish the same objects.
OBJECTS_FIXTURE = EXAMPLE_DIR / "tests" / "fixtures" / "insight_objects.json"

pytestmark = pytest.mark.unit

main_module = load_example_main(EXAMPLE_DIR, "multi_stream_obb_detector_main")
main_module.np = np  # main.py binds numpy lazily at runtime


def detections(boxes: list[dict]):
    rows = [[b["cx"], b["cy"], b["width"], b["height"], b["angle_radians"]] for b in boxes]
    return main_module.Detections(
        boxes=np.array(rows, dtype=np.float64).reshape(-1, 5),
        scores=np.array([b["score"] for b in boxes], dtype=np.float64),
        class_ids=np.array([b["class_id"] for b in boxes], dtype=np.int64),
    )


VALID_CONFIG = {
    "model": {"path": "model.tar.gz", "labels": "dota_label.txt"},
    "streams": ["rtsp://127.0.0.1:8554/src1", "rtsp://127.0.0.1:8554/src2"],
    "input": {"codec": "h264", "tcp": True, "latency_ms": 100},
    "inference": {
        "frames": 0,
        "max_inflight_per_stream": 4,
        "max_inflight_total": 16,
        "min_score": 0.30,
        "max_detections": 300,
    },
    "output": {
        "insight": {"host": "127.0.0.1", "video_port_base": 9000, "metadata_port_base": 9100},
        "debug_dir": "",
        "save_every": 0,
    },
}

# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_config = config_writer(VALID_CONFIG)


def run_main(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLE_DIR),
        timeout=20,
    )


class TestMainEntrypoint:
    def test_help_lists_the_options(self):
        result = run_main("--help")

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--validate-config-only" in result.stdout

    def test_missing_config_file_exits_with_code_2(self):
        result = run_main("--config", "does-not-exist.yaml")

        assert result.returncode == 2
        assert "config file not found" in result.stderr

    def test_validate_config_only_reports_the_stream_count(self, tmp_path):
        result = run_main("--config", str(write_config(tmp_path)), "--validate-config-only")

        assert result.returncode == 0
        assert "streams=2" in result.stdout

    def test_the_shipped_config_parses(self):
        cfg = main_module.load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.model_path == "<model-path>"
        assert len(cfg.rtsp_urls) == 2


class TestConfig:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main_module.load_app_config(write_config(tmp_path))

        assert cfg.model_path == "model.tar.gz"
        assert cfg.rtsp_urls == VALID_CONFIG["streams"]
        assert cfg.codec == "h264"

    def test_omitted_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {"path": "model.tar.gz"},
            "streams": ["rtsp://127.0.0.1:8554/src1"],
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main_module.load_app_config(write_config(tmp_path, root=raw))

        assert cfg.labels_path.endswith("src/common/dota_label.txt")
        assert cfg.codec == "h264"
        assert cfg.tcp is True
        assert cfg.latency_ms == 100
        assert cfg.frames == 0
        assert cfg.max_inflight_per_stream == 4
        assert cfg.max_inflight_total == 16
        assert cfg.min_score == pytest.approx(0.30)
        assert cfg.max_detections == 300
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.debug_dir == ""
        assert cfg.save_every == 0

    @pytest.mark.parametrize(("codec", "expected"), [("avc", "h264"), ("HEVC", "h265")])
    def test_codec_aliases_are_accepted(self, tmp_path, codec, expected):
        cfg = main_module.load_app_config(write_config(tmp_path, [(("input", "codec"), codec)]))

        assert cfg.codec == expected

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("model", "path"), "", "model.path must be set"),
            (("model", "labels"), "", "model.labels must be set"),
            (("input", "codec"), "vp9", "input.codec must be h264/avc or h265/hevc"),
            (("input", "tcp"), "yes", "input.tcp must be true or false"),
            (("input", "latency_ms"), -1, "input.latency_ms must be >= 0"),
            (("inference", "frames"), -1, "inference.frames must be >= 0"),
            (("inference", "frames"), 1.5, "inference.frames must be an integer"),
            (("inference", "max_inflight_per_stream"), 0, "must be -1 or > 0"),
            (("inference", "max_inflight_total"), -2, "must be -1 or > 0"),
            (("inference", "min_score"), 1.01, "inference.min_score must be between 0 and 1"),
            (("inference", "max_detections"), 0, "inference.max_detections must be > 0"),
            (("output", "insight", "host"), "", "output.insight.host must be set"),
            (("output", "insight", "video_port_base"), 0, "video_port_base must be > 0"),
            (("output", "insight", "metadata_port_base"), 0, "metadata_port_base must be > 0"),
            (("output", "save_every"), -1, "output.save_every must be >= 0"),
        ],
    )
    def test_an_invalid_value_is_rejected_with_its_key(self, tmp_path, path, value, message):
        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(write_config(tmp_path, [(path, value)]))

    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main_module.load_app_config(config_path)


class TestStreamList:
    def test_a_missing_list_is_rejected(self, tmp_path):
        raw = copy.deepcopy(VALID_CONFIG)
        del raw["streams"]

        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(write_config(tmp_path, root=raw))

    def test_a_scalar_instead_of_a_list_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="streams must be a non-empty list"):
            main_module.load_app_config(write_config(tmp_path, [(("streams",), "rtsp://host/a")]))

    @pytest.mark.parametrize(("streams", "bad_index"), [([""], 0), (["rtsp://host/a", 42], 1)])
    def test_a_blank_or_non_string_entry_names_its_index(self, tmp_path, streams, bad_index):
        with pytest.raises(ValueError, match=rf"streams\[{bad_index}\] must be a non-empty string"):
            main_module.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

    def test_more_than_four_streams_are_rejected(self, tmp_path):
        streams = [f"rtsp://host/src{index}" for index in range(5)]

        with pytest.raises(ValueError, match="streams supports up to four entries"):
            main_module.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

    def test_stream_order_is_preserved(self, tmp_path):
        """Stream order decides which Insight channel each stream publishes on."""
        streams = ["rtsp://host/c", "rtsp://host/a", "rtsp://host/b", "rtsp://host/d"]

        cfg = main_module.load_app_config(write_config(tmp_path, [(("streams",), streams)]))

        assert cfg.rtsp_urls == streams


class FakeMetadataSender:
    def __init__(self):
        self.calls = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id):
        self.calls.append((metadata_type, data_json, timestamp_ms, frame_id))
        return True


class TestInsightMetadata:
    def test_objects_match_the_shared_fixture(self):
        fixture = json.loads(OBJECTS_FIXTURE.read_text(encoding="utf-8"))
        objects = main_module.insight_objects(detections(fixture["boxes"]), fixture["labels"])

        assert len(objects) == len(fixture["objects"])
        for actual, expected in zip(objects, fixture["objects"]):
            assert actual["id"] == expected["id"]
            assert actual["label"] == expected["label"]
            assert actual["confidence"] == pytest.approx(expected["confidence"])
            assert actual["bbox"] == pytest.approx(expected["bbox"], abs=1e-9)
            assert actual["oriented_bbox"] == pytest.approx(expected["oriented_bbox"], abs=1e-9)

    def test_send_metadata_publishes_one_object_detection_message(self):
        sender = FakeMetadataSender()
        stream = main_module.Stream(0, "rtsp://host/a", 1280, 720, 30, sender)
        box = {
            "cx": 100.0,
            "cy": 50.0,
            "width": 40.0,
            "height": 20.0,
            "angle_radians": 0.0,
            "score": 0.9,
            "class_id": 0,
        }
        sample = SimpleNamespace(frame_id=42, pts_ns=1_234_000_000)

        main_module.send_metadata(stream, sample, detections([box]), ["plane"])

        assert len(sender.calls) == 1
        metadata_type, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert metadata_type == "object-detection"
        assert timestamp_ms == 1234
        assert frame_id == "42"
        objects = json.loads(data_json)["objects"]
        assert [obj["label"] for obj in objects] == ["plane"]
        assert objects[0]["oriented_bbox"] == [100.0, 50.0, 40.0, 20.0, 0.0]

    def test_a_frame_without_detections_still_publishes_an_empty_list(self):
        """Insight clears the previous frame's boxes only when it receives the empty list."""
        sender = FakeMetadataSender()
        stream = main_module.Stream(0, "rtsp://host/a", 1280, 720, 30, sender)

        main_module.send_metadata(
            stream, SimpleNamespace(frame_id=-1, pts_ns=-1), detections([]), []
        )

        _, data_json, timestamp_ms, frame_id = sender.calls[0]
        assert json.loads(data_json) == {"objects": []}
        assert timestamp_ms == -1
        assert frame_id == ""

    def test_a_failed_send_warns_and_keeps_running(self, capsys):
        """Like C++, one lost message must not end the run."""

        class FailingSender:
            def send_metadata(self, *args):
                raise RuntimeError("MetadataSender payload exceeds 65507 bytes")

        stream = main_module.Stream(1, "rtsp://host/a", 1280, 720, 30, FailingSender())

        main_module.send_metadata(stream, SimpleNamespace(frame_id=1, pts_ns=0), detections([]), [])

        assert "stream 1 metadata send failed: MetadataSender payload exceeds 65507 bytes" in (
            capsys.readouterr().err
        )


class TestRunLoop:
    def test_a_timeout_is_not_a_sample(self):
        run = FakeRun("timeout")

        assert main_module.pull_detections(run) is None
        assert run.pulls == [("detections", main_module.PULL_TIMEOUT_MS)]

    def test_a_runtime_error_ends_the_run(self):
        run = FakeRun(("error", "queue torn down"))

        with pytest.raises(RuntimeError, match="queue torn down"):
            main_module.pull_detections(run)

    def test_a_sample_is_routed_to_its_stream(self):
        streams = [main_module.Stream(index, "", 0, 0, 0, None) for index in range(2)]

        stream = main_module.stream_for(SimpleNamespace(stream_id="stream1"), streams)

        assert stream is streams[1]

    def test_a_sample_from_an_unknown_stream_fails(self):
        streams = [main_module.Stream(0, "", 0, 0, 0, None)]

        with pytest.raises(RuntimeError, match="unknown stream id"):
            main_module.stream_for(SimpleNamespace(stream_id="stream9"), streams)

    def test_a_finite_run_ends_when_every_stream_reaches_the_limit(self):
        streams = [main_module.Stream(index, "", 0, 0, 0, None) for index in range(2)]
        streams[0].processed = 10
        streams[1].processed = 9

        assert not main_module.all_streams_done(streams, 10)
        streams[1].processed = 10
        assert main_module.all_streams_done(streams, 10)
        assert not main_module.all_streams_done(streams, 0)
