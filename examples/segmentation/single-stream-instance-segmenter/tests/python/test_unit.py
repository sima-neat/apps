"""Unit tests for single-stream-instance-segmenter (Python)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

_SPEC = importlib.util.spec_from_file_location("instance_seg_main", MAIN_PY)
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
