"""Unit tests for fastsam-multistream (Python)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import config
import tokenizer
import clip
import fastsam
import main

pytestmark = pytest.mark.unit


def _base_config() -> dict:
    """A minimal config that passes every required-field check."""
    return {
        "model": {"path": "/models/fastsam.tar.gz"},
        "clip": {
            "image_encoder_path": "/models/clip_image",
            "text_encoder_path": "/models/clip_text",
            "text_host_consts": "/models/consts.npz",
        },
        "source": {"rtsp_urls": ["rtsp://127.0.0.1:8554/src1"]},
        "prompt": {"text": "the black labrador"},
        "output": {"insight": {"host": "127.0.0.1"}},
    }


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TestConfigLoading:
    def test_accepts_minimal_config(self, tmp_path: Path):
        cfg = config.load_config(_write_config(tmp_path, _base_config()))

        assert cfg.model_path == "/models/fastsam.tar.gz"
        assert cfg.rtsp_urls == ["rtsp://127.0.0.1:8554/src1"]
        assert cfg.text == "the black labrador"
        assert cfg.insight_host == "127.0.0.1"

    def test_applies_defaults(self, tmp_path: Path):
        cfg = config.load_config(_write_config(tmp_path, _base_config()))

        assert cfg.infer_size == 640
        assert cfg.score_threshold == 0.7
        assert cfg.nms_iou == 0.9
        assert cfg.max_detections == 300
        assert cfg.min_score == 0.65
        assert cfg.max_box_frac == 0.8
        assert cfg.queue_depth == 8
        assert cfg.latency_ms == 200
        assert cfg.tcp is True
        assert cfg.frames == 0
        assert cfg.video_port_base == 9000
        assert cfg.metadata_port_base == 9100
        assert cfg.bitrate_kbps == 1000

    def test_rtsp_urls_filters_falsy_entries(self, tmp_path: Path):
        raw = _base_config()
        raw["source"]["rtsp_urls"] = ["", "rtsp://x/src1", None, "rtsp://x/src2"]

        cfg = config.load_config(_write_config(tmp_path, raw))

        assert cfg.rtsp_urls == ["rtsp://x/src1", "rtsp://x/src2"]

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="config file not found"):
            config.load_config(tmp_path / "does-not-exist.yaml")

    @pytest.mark.parametrize(
        ("drop", "message"),
        [
            ("model.path", "model.path must be set"),
            ("clip.image_encoder_path", "clip.image_encoder_path must be set"),
            ("clip.text_encoder_path", "clip.text_encoder_path must be set"),
            ("clip.text_host_consts", "clip.text_host_consts must be set"),
            ("source.rtsp_urls", "source.rtsp_urls must be a non-empty list of RTSP URLs"),
            ("prompt.text", "prompt.text must be set"),
            ("output.insight.host", "output.insight.host must be set"),
        ],
    )
    def test_rejects_missing_required_field(self, tmp_path: Path, drop: str, message: str):
        raw = _base_config()
        node = raw
        *parents, leaf = drop.split(".")
        for key in parents:
            node = node[key]
        del node[leaf]

        with pytest.raises(RuntimeError, match=message):
            config.load_config(_write_config(tmp_path, raw))


@pytest.fixture(scope="module")
def tok():
    return tokenizer.SimpleTokenizer()


class TestTokenizer:
    def test_golden_the_black_dog(self, tok):
        row = tok(["the black dog"])[0]
        assert row[:5].tolist() == [49406, 518, 1449, 1929, 49407]
        assert not row[5:].any()

    def test_golden_a_photo_of_a_cat(self, tok):
        row = tok(["a photo of a cat"])[0]
        assert row[:7].tolist() == [49406, 320, 1125, 539, 320, 2368, 49407]
        assert not row[7:].any()

    def test_shape_and_dtype(self, tok):
        single = tok(["the black dog"])
        assert single.shape == (1, 77)
        assert single.dtype == np.int32
        assert tok(["a", "b", "c"]).shape == (3, 77)

    def test_truncation_preserves_eot(self, tok):
        row = tok(["word " * 100])[0]
        assert row.shape == (77,)
        assert row[0] == 49406
        assert row[76] == 49407


class TestClipScores:
    def test_empty_returns_size_zero(self):
        assert clip._scores([], [1.0, 0.0]).size == 0

    def test_softmax_sums_to_one(self):
        scores = clip._scores([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], [1.0, 0.0])
        assert scores.sum() == pytest.approx(1.0)
        assert np.all(scores >= 0.0)
        assert np.all(np.isfinite(scores))

    def test_argmax_picks_aligned_feature(self):
        scores = clip._scores([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
        assert int(np.argmax(scores)) == 0

    def test_zero_norm_query_is_uniform(self):
        scores = clip._scores([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])
        assert scores.tolist() == pytest.approx([0.5, 0.5])
        assert scores.sum() == pytest.approx(1.0)


class TestFastsamGeometry:
    def test_landscape(self):
        geom = fastsam.letterbox_geometry(1280, 720, 640)
        assert (geom.scale, geom.pad_x, geom.pad_y) == (0.5, 0, 140)

    def test_portrait(self):
        geom = fastsam.letterbox_geometry(720, 1280, 640)
        assert (geom.scale, geom.pad_x, geom.pad_y) == (0.5, 140, 0)

    def test_square(self):
        geom = fastsam.letterbox_geometry(640, 640, 640)
        assert (geom.scale, geom.pad_x, geom.pad_y) == (1.0, 0, 0)


class TestMaskPolygon:
    GEOM = fastsam.Geometry(1.0, 0, 0)

    def test_empty_mask_returns_empty(self):
        assert fastsam.mask_polygon(np.zeros((160, 160), np.uint8), self.GEOM) == []

    def test_degenerate_single_pixel_returns_empty(self):
        mask = np.zeros((160, 160), np.uint8)
        mask[80, 80] = 255
        assert fastsam.mask_polygon(mask, self.GEOM) == []

    def test_rectangle_maps_by_stride(self):
        mask = np.zeros((160, 160), np.uint8)
        mask[20:60, 30:100] = 255

        poly = fastsam.mask_polygon(mask, self.GEOM)

        assert 3 <= len(poly) <= 80
        assert all(isinstance(x, int) and isinstance(y, int) for x, y in poly)
        # MASK_STRIDE=4 scales proto (<=160) coords into frame space (>160).
        assert max(x for x, _ in poly) > 160


class TestObjectCrop:
    GEOM = fastsam.Geometry(1.0, 0, 0)

    def test_none_when_empty_mask(self):
        frame = np.zeros((640, 640, 3), np.uint8)
        proto = np.zeros((160, 160), np.uint8)
        assert fastsam.object_crop(proto, frame, self.GEOM, max_box_frac=0.8) is None

    def test_none_when_too_small(self):
        frame = np.zeros((64, 64, 3), np.uint8)
        proto = np.zeros((160, 160), np.uint8)
        proto[10:20, 10:20] = 255
        assert fastsam.object_crop(proto, frame, self.GEOM, max_box_frac=0.8) is None

    def test_none_when_background_like(self):
        frame = np.zeros((640, 640, 3), np.uint8)
        proto = np.zeros((160, 160), np.uint8)
        proto[0:120, 0:120] = 255  # 14400 px > 0.5 * 160 * 160
        assert fastsam.object_crop(proto, frame, self.GEOM, max_box_frac=0.8) is None

    def test_valid_crop_shapes(self):
        frame = np.zeros((640, 640, 3), np.uint8)
        proto = np.zeros((160, 160), np.uint8)
        proto[40:80, 40:80] = 255

        crop = fastsam.object_crop(proto, frame, self.GEOM, max_box_frac=0.8)

        assert isinstance(crop, fastsam.Crop)
        assert crop.window.shape == (192, 192, 3)
        assert crop.submask.shape == (40, 40)


class TestSegmentsJson:
    def test_none_match_is_empty(self):
        assert json.loads(main._segments_json(None, "the black labrador")) == {"segments": []}

    def test_populated_match(self):
        payload = json.loads(
            main._segments_json((0.87, [(1, 2), (3, 4), (5, 6)]), "the black labrador")
        )
        assert payload == {
            "segments": [
                {
                    "id": "seg_1",
                    "label": "the black labrador",
                    "confidence": 0.87,
                    "mask_format": "polygon",
                    "mask": [[1, 2], [3, 4], [5, 6]],
                }
            ]
        }

    def test_float_polygon_is_int_cast(self):
        payload = json.loads(main._segments_json((0.5, [(10.9, 20.1)]), "x"))
        assert payload["segments"][0]["mask"] == [[10, 20]]
