"""Unit tests for the RF-DETR example."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
def test_topk_gather_is_stable_and_deterministic():
    scores = np.zeros(305, dtype=np.float32)
    scores[3:5] = 2.0
    proposals = np.zeros((305, 4), dtype=np.float32)
    proposals[:, 0] = np.arange(305)

    gathered, indices = main.stable_topk_gather(scores, proposals)

    assert gathered.shape == (1, 300, 4)
    assert indices[:3].tolist() == [3, 4, 0]
    assert gathered[0, :3, 0].tolist() == [3.0, 4.0, 0.0]


@pytest.mark.unit
def test_postprocess_uses_sparse_coco_ids_and_source_geometry():
    labels = ["unused"] * 91
    labels[1] = "person"
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.5, 0.25]
    logits = np.full((1, 300, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0

    objects = main.postprocess(boxes, logits, 1920, 1080, labels, 0.5, 10)

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
            "variant": variant,
            "labels": str(labels),
            "small": {"backbone": "small-b.tar.gz", "transformer": "small-t.tar.gz", "input_size": 512},
            "medium": {"backbone": "medium-b.tar.gz", "transformer": "medium-t.tar.gz", "input_size": 576},
        },
        "source": {"rtsp_url": "rtsp://camera/live", "width": 1280, "height": 720, "fps": 30},
        "inference": {"frames": 1, "min_score": 0.5, "max_detections": 10},
        "output": {"insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    selected = main.load_config(path)

    assert selected.variant == variant
    assert selected.input_size == size
    assert selected.backbone.startswith(variant)


@pytest.mark.unit
def test_config_rejects_unknown_model_variant(tmp_path):
    config = {
        "model": {"variant": "large"},
        "source": {},
        "inference": {},
        "output": {"insight": {}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="small or medium"):
        main.load_config(path)
