"""Unit tests for the RF-DETR example."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
_SPEC = importlib.util.spec_from_file_location("rfdetr_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


@pytest.mark.unit
def test_config_argument_is_required():
    with pytest.raises(SystemExit):
        main.parse_args([])


@pytest.mark.unit
@pytest.mark.parametrize("storage_kind", ["CpuOwned", "CpuExternal", "GstSample"])
def test_transformer_inputs_copy_only_cpu_features(monkeypatch, storage_kind):
    kinds = SimpleNamespace(CpuOwned="CpuOwned", CpuExternal="CpuExternal")
    monkeypatch.setattr(main, "pyneat", SimpleNamespace(StorageKind=kinds), raising=False)
    device_feature = SimpleNamespace(shape=[1, 4, 256])
    feature = SimpleNamespace(
        shape=[1, 4, 256],
        storage=SimpleNamespace(kind=storage_kind),
        cvu=Mock(return_value=device_feature),
    )
    gathered = SimpleNamespace(shape=[1, 2, 4])
    model = SimpleNamespace(input_specs=lambda: [gathered, device_feature])

    inputs = main.transformer_inputs(model, feature, gathered, 2)

    assert inputs[0] is gathered
    if storage_kind == "GstSample":
        assert inputs[1] is feature
        feature.cvu.assert_not_called()
    else:
        assert inputs[1] is device_feature
        feature.cvu.assert_called_once_with()


@pytest.mark.unit
def test_topk_gather_is_stable_and_deterministic():
    scores = np.zeros(305, dtype=np.float32)
    scores[3:5] = 2.0
    proposals = np.zeros((305, 4), dtype=np.float32)
    proposals[:, 0] = np.arange(305)

    gathered = main.stable_topk_gather(scores, proposals, 300)

    assert gathered.shape == (1, 300, 4)
    assert gathered[0, :3, 0].tolist() == [3.0, 4.0, 0.0]


@pytest.mark.unit
def test_postprocess_uses_sparse_coco_ids_and_source_geometry():
    labels = ["unused"] * 91
    labels[1] = "person"
    boxes = np.array([[480, 405, 1440, 675, 1.0, 0],
                      [480, 405, 1440, 675, 0.99, 1]], dtype=np.float32)
    objects = main.postprocess(boxes, 1920, 1080, labels, 0.5, 10)

    assert len(objects) == 1
    assert objects[0]["label"] == "person"
    assert objects[0]["bbox"] == pytest.approx([480.0, 405.0, 960.0, 270.0])


@pytest.mark.unit
@pytest.mark.parametrize(("variant", "size"), [("small", 512), ("medium", 576)])
def test_config_selects_one_model_pair(tmp_path, variant, size):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(f"label-{index}" for index in range(91)) + "\n")
    config = {
        "model": {
            "task": "detection",
            "labels": str(labels),
            "detection": {
                "variant": variant,
                "small": {"backbone": "small-b.tar.gz", "transformer": "small-t.tar.gz"},
                "medium": {"backbone": "medium-b.tar.gz", "transformer": "medium-t.tar.gz"},
            },
        },
        "source": {"rtsp_url": "rtsp://camera/live", "codec": "h265"},
        "inference": {
            "frames": 1,
            "detection": {"min_score": 0.5, "max_detections": 10},
            "segmentation": {"mask_threshold": 2.0},
        },
        "output": {"insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    selected = main.load_config(path)

    assert selected.variant == variant
    assert selected.input_size == size
    assert selected.backbone.startswith(variant)
    assert selected.codec == "h265"
    assert (selected.width, selected.height, selected.fps) == (0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("mask_grid_size", [None, 108, 432, 640, 107])
def test_config_selects_segmentation_model_pair(tmp_path, mask_grid_size):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(f"label-{index}" for index in range(91)) + "\n")
    config = {
        "model": {
            "task": "segmentation",
            "labels": str(labels),
            "segmentation": {
                "backbone": "segmentation-b.tar.gz",
                "transformer": "segmentation-t.tar.gz",
            },
        },
        "source": {"rtsp_url": "rtsp://camera/live", "codec": "h264"},
        "inference": {"segmentation": {"min_score": 0.3, "max_segments": 24}},
        "output": {"insight": {"host": "127.0.0.1"}},
    }
    if mask_grid_size is not None:
        config["inference"]["segmentation"]["mask_grid_size"] = mask_grid_size
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if mask_grid_size == 107:
        with pytest.raises(ValueError, match="mask_grid_size must be >= 108"):
            main.load_config(path)
        return

    selected = main.load_config(path)

    assert selected.mask_grid_size == (mask_grid_size or 640)
    assert selected.task == "segmentation"
    assert selected.backbone == "segmentation-b.tar.gz"
    assert selected.input_size == 432
    assert selected.top_k == 200


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("h264", "h264"),
        ("AVC", "h264"),
        ("h.264", "h264"),
        ("h265", "h265"),
        ("HEVC", "h265"),
        ("mjpeg", "mjpeg"),
        ("JPEG", "mjpeg"),
    ],
)
def test_source_codec_aliases(value, expected):
    assert main.parse_source_codec(value) == expected


@pytest.mark.unit
def test_probed_geometry_uses_configured_fallbacks_and_fps_override():
    assert main.resolve_geometry((1280, 720, 60), (640, 480, 30)) == (1280, 720, 30)
    assert main.resolve_geometry((1280, 0, 0), (640, 480, 30)) == (1280, 480, 30)


@pytest.mark.unit
def test_config_rejects_unknown_model_variant(tmp_path):
    config = {
        "model": {"task": "detection", "detection": {"variant": "large"}},
        "source": {},
        "inference": {},
        "output": {"insight": {}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="small or medium"):
        main.load_config(path)


@pytest.mark.unit
@pytest.mark.parametrize("mask_grid_size", [108, 432, 640])
def test_segmentation_metadata_contains_polygons(mask_grid_size):
    labels = ["unused"] * 91
    labels[1] = "person"
    boxes = np.array([[320, 180, 960, 540, 1.0, 0],
                      [320, 180, 960, 540, 0.99, 1],
                      [320, 180, 960, 540, 0.98, 1]], dtype=np.float32)
    masks = np.zeros((3, 108, 108), dtype=np.float32)
    masks[2, 40:68, 40:68] = 1.0
    payload = main.segmentation_metadata(
        boxes, masks, 1280, 720, labels, 0.3, 1, 0.08, mask_grid_size
    )

    segments = json.loads(payload)["segments"]
    assert len(payload.encode()) <= main.METADATA_BYTE_BUDGET
    assert len(segments) == 1
    segment = segments[0]
    assert segment["label"] == "person"
    assert segment["mask_format"] == "polygon"
    assert len(segment["mask"]) >= 3
    assert all(0 <= x < 1280 and 0 <= y < 720 for x, y in segment["mask"])


@pytest.mark.unit
def test_decoded_boundaries_and_metadata_budget():
    labels = ["unused", "person"]
    boxes = np.array([[-2, -3, 20, 15, 0.5, 1]] * 2, dtype=np.float32)
    masks = np.full((2, 108, 108), 0.5, dtype=np.float32)
    objects = main.postprocess(boxes, 13, 9, labels, 0.5, 2)
    assert objects == [
        {"id": f"obj_{i}", "label": "person", "confidence": 0.5,
         "bbox": [0.0, 0.0, 13.0, 9.0]} for i in (1, 2)
    ]
    assert main.postprocess(boxes[:0], 13, 9, labels, 0.5, 2) == []
    def metadata(rows, probabilities, names):
        return main.segmentation_metadata(rows, probabilities, 13, 9, names,
                                          0.5, 2, 0.5, 108)
    payload = metadata(boxes, masks, labels)
    assert json.loads(payload) == {"segments": [
        {"id": f"seg_{i}", "label": "person", "confidence": 0.5,
         "bbox": [0, 0, 13, 9], "mask_format": "polygon",
         "mask": [[0, 0], [0, 8], [12, 8], [12, 0]]} for i in (1, 2)
    ]}
    assert json.loads(metadata(boxes[:0], masks[:0], labels)) == {"segments": []}
    below = np.nextafter(np.float32(0.5), np.float32(0))
    assert json.loads(metadata(boxes, np.full_like(masks, below), labels)) == {"segments": []}
    long_labels = ["unused", "x" * (main.METADATA_BYTE_BUDGET // 2)]
    limited = metadata(boxes, masks, long_labels)
    assert len(json.loads(limited)["segments"]) == 1
    assert len(limited.encode()) <= main.METADATA_BYTE_BUDGET
