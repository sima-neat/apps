"""Unit tests for the RF-DETR detection and segmentation example."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"

main_module = load_example_main(EXAMPLE_DIR, "rfdetr_detection_segmentation_main")
postprocess = main_module.postprocess

pytestmark = pytest.mark.unit

# Every key the application reads, with test values.
CONFIG_LINES = {
    "model.id": "model:\n  id: seg-medium",
    "model.path": "  path: models/rf_detr_seg_medium_backbone_mpk.tar.gz",
    "model.labels": "  labels: coco-labels.txt",
    "input.source": "input:\n  source: rtsp://127.0.0.1:8554/src1",
    "input.codec": "  codec: h264",
    "input.width": "  width: 1280",
    "input.height": "  height: 720",
    "input.fps": "  fps: 30",
    "input.rtsp.tcp": "  rtsp:\n    tcp: true",
    "input.rtsp.latency_ms": "    latency_ms: 10",
    "inference.frames": "inference:\n  frames: 0",
    "inference.min_score": "  min_score: 0.3",
    "inference.max_detections": "  max_detections: 24",
    "output.insight.host": "output:\n  insight:\n    host: 127.0.0.1",
    "output.insight.video_port_base": "    video_port_base: 9000",
    "output.insight.metadata_port_base": "    metadata_port_base: 9100",
    "app.mask_threshold": "app:\n  mask_threshold: 0.08",
    "app.mask_grid_size": "  mask_grid_size: 640",
}

LABELS = ["unused", "person"] + ["unused"] * 89

OPTIONS = postprocess.PostprocessOptions(
    width=1280,
    height=720,
    min_score=0.3,
    max_detections=10,
    mask_threshold=0.08,
    mask_grid_size=640,
)


def write_config(tmp_path: Path, extra: str = "", **replace: str) -> Path:
    lines = {**CONFIG_LINES, **replace}
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines.values()) + "\n" + extra, encoding="utf-8")
    return path


def test_help_lists_the_options():
    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--validate-config-only" in result.stdout


def test_missing_config_file_exits_with_code_2(tmp_path, capsys):
    assert main_module.main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert "config file not found" in capsys.readouterr().err


def test_shipped_config_loads():
    cfg = main_module.load_app_config(COMMON_DIR / "config.yaml")
    assert cfg.model_id == "det-small" and not cfg.segmentation
    assert cfg.model_path.name == "rf_detr_small_w8a8_mixed_w16a16_backbone_mpk.tar.gz"
    assert cfg.transformer_path == cfg.model_path.with_name(
        "rf_detr_small_w8a8_mixed_w16a16_decoder_mpk.tar.gz"
    )
    # The packaged path reaches prebuilt-apps/models from the config file.
    assert cfg.model_path.resolve().is_relative_to(EXAMPLE_DIR.parents[2] / "models")


def test_config_reads_every_key(tmp_path):
    cfg = main_module.load_app_config(write_config(tmp_path))
    assert cfg.model_id == "seg-medium" and cfg.segmentation
    assert (
        cfg.model_path
        == tmp_path.resolve() / "models/rf_detr_seg_medium_backbone_mpk.tar.gz"
    )
    assert (
        cfg.transformer_path
        == tmp_path.resolve() / "models/rf_detr_seg_medium_decoder_mpk.tar.gz"
    )
    assert cfg.source == "rtsp://127.0.0.1:8554/src1"
    assert (cfg.width, cfg.height, cfg.fps) == (1280, 720, 30)
    assert cfg.mask_threshold == 0.08 and cfg.mask_grid_size == 640


def test_unknown_key_is_rejected(tmp_path):
    path = write_config(tmp_path, extra="source:\n  rtsp_url: rtsp://old\n")
    with pytest.raises(ValueError, match="unknown keys: source.rtsp_url"):
        main_module.load_app_config(path)


def test_missing_key_is_rejected(tmp_path):
    path = write_config(tmp_path, **{"app.mask_grid_size": ""})
    with pytest.raises(ValueError, match="is missing app.mask_grid_size"):
        main_module.load_app_config(path)


def test_model_id_must_name_a_variant(tmp_path):
    path = write_config(tmp_path, **{"model.id": "model:\n  id: seg-large"})
    with pytest.raises(ValueError, match="model.id must be one of det-nano"):
        main_module.load_app_config(path)


def test_model_path_must_name_a_backbone_package(tmp_path):
    path = write_config(
        tmp_path, **{"model.path": "  path: models/rf_detr_small_mpk.tar.gz"}
    )
    with pytest.raises(ValueError, match="must name an RF-DETR backbone package"):
        main_module.load_app_config(path)


def test_codec_must_be_h264_h265_or_mjpeg(tmp_path):
    path = write_config(tmp_path, **{"input.codec": "  codec: vp9"})
    with pytest.raises(ValueError, match="input.codec must be h264, h265, or mjpeg"):
        main_module.load_app_config(path)


def test_top_proposals_are_stable_and_best_first():
    scores = np.zeros((1, 305), dtype=np.float32)
    scores[0, 3] = scores[0, 4] = 2.0
    proposals = np.zeros((1, 305, 4), dtype=np.float32)
    proposals[0, :, 0] = np.arange(305)
    best = postprocess.top_proposals(scores, proposals, 300)
    assert best.shape == (300, 4)
    assert list(best[:3, 0]) == [3.0, 4.0, 0.0]


def test_detection_skips_unused_classes_and_scales_to_the_frame():
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = boxes[0, 1] = [0.5, 0.5, 0.5, 0.25]
    logits = np.full((1, 300, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0  # person
    logits[0, 1, 2] = 12.0  # an unused class, ranked first
    options = postprocess.PostprocessOptions(1920, 1080, 0.5, 10, 0.08, 640)
    objects = json.loads(postprocess.detection_json(boxes, logits, LABELS, options))[
        "objects"
    ]
    assert len(objects) == 1
    assert objects[0]["id"] == "obj_1" and objects[0]["label"] == "person"
    assert objects[0]["bbox"] == pytest.approx([480.0, 405.0, 960.0, 270.0])


@pytest.mark.parametrize(("queries", "mask_size"), [(200, 108), (100, 78)])
@pytest.mark.parametrize("mask_grid_size", [None, 432, 640])
def test_segmentation_traces_a_polygon_inside_the_box(
    queries, mask_size, mask_grid_size
):
    boxes = np.zeros((1, queries, 4), dtype=np.float32)
    boxes[0, 0] = boxes[0, 1] = [0.5, 0.5, 0.5, 0.5]
    logits = np.full((1, queries, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 11.0  # person on query 0
    logits[0, 1, 1] = 10.0  # person on query 1, whose mask is empty
    masks = np.full((1, mask_size, mask_size, queries), -20.0, dtype=np.float32)
    lo, hi = mask_size * 3 // 8, mask_size * 5 // 8
    masks[0, lo:hi, lo:hi, 0] = 10.0
    options = postprocess.PostprocessOptions(
        1280, 720, 0.3, 24, 0.08, mask_grid_size or mask_size
    )
    payload = postprocess.segmentation_json(
        boxes, logits, masks, mask_size, LABELS, options
    )
    segments = json.loads(payload)["segments"]
    assert len(segments) == 1
    segment = segments[0]
    assert segment["label"] == "person" and segment["mask_format"] == "polygon"
    assert segment["bbox"] == [320, 180, 640, 360]
    assert len(segment["mask"]) >= 3
    assert all(320 <= x < 960 and 180 <= y < 540 for x, y in segment["mask"])
    assert len(payload) <= postprocess.SEGMENTS_BYTE_BUDGET
