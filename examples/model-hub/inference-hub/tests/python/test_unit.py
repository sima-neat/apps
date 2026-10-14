"""Unit tests for the inference hub example."""

from __future__ import annotations

import importlib
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.utils.config_cases import load_example_main

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
main = load_example_main(EXAMPLE_DIR, "inference_hub_main")
MODEL_NAMES = sorted(p.parent.name for p in main.MODELS_DIR.glob("*/config.yaml"))


def write_config(tmp_path: Path, **model) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["model"].update(model)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.unit
def test_config_argument_is_required():
    with pytest.raises(SystemExit):
        main.parse_args([])


@pytest.mark.unit
def test_shipped_config_loads():
    assert main.load_config(CONFIG)["model_cfg"]["variants"]


@pytest.mark.unit
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_folder_contract(name):
    model_cfg = main.load_yaml(main.MODELS_DIR / name / "config.yaml")
    assert main.load_labels(model_cfg["labels"])
    assert model_cfg["components"] and model_cfg["variants"] and model_cfg["build"]
    assert hasattr(importlib.import_module(f"hub_models.{name}.model"), "Model")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "message"),
    [({"name": "nope"}, "model.name must be one of"), ({"variant": "nope"}, "model.variant")],
)
def test_unknown_model_or_variant_names_the_choices(tmp_path, model, message):
    with pytest.raises(ValueError, match=message):
        main.load_config(write_config(tmp_path, **model))


@pytest.mark.unit
def test_missing_pack_prints_install_command(tmp_path):
    cfg = main.load_config(write_config(tmp_path, name="rf_detr", variant="nano", dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError) as error:
        main.find_packs(cfg)
    assert f"--id rf_detr_nano --variant modalix_int8_bf16 -o {tmp_path}" in str(error.value)


@pytest.mark.unit
def test_installed_packs_are_found_per_component(tmp_path):
    cfg = main.load_config(write_config(tmp_path, name="rf_detr", variant="nano", dir=str(tmp_path)))
    for component in ("backbone", "transformer"):
        folder = tmp_path / "rf_detr_nano" / "components" / component
        folder.mkdir(parents=True)
        (folder / f"{component}_mpk.tar.gz").touch()
    packs = main.find_packs(cfg)
    assert packs["transformer"].endswith("transformer/transformer_mpk.tar.gz")


@pytest.mark.unit
def test_yolo26_boxes_are_clipped_to_the_frame():
    yolo26 = importlib.import_module("hub_models.yolo26.model")
    payload = struct.pack("<I", 2) + yolo26.BOX.pack(-5, 10, 20, 30, 0.9, 1) + yolo26.BOX.pack(90, 0, 20, 10, 0.5, 99)
    objects = yolo26.parse_boxes(payload, ["person", "bicycle"], 100, 50)
    assert objects[0] == {"label": "bicycle", "confidence": pytest.approx(0.9), "bbox": [0, 10, 15, 30]}
    assert objects[1]["label"] == "99" and objects[1]["bbox"] == [90, 0, 10, 10]


@pytest.mark.unit
def test_yolo26_rejects_a_truncated_payload():
    yolo26 = importlib.import_module("hub_models.yolo26.model")
    with pytest.raises(RuntimeError, match="fewer than 3"):
        yolo26.parse_boxes(struct.pack("<I", 3), ["person"], 100, 100)


@pytest.mark.unit
def test_resnet50_top_classes_are_softmax_ranked():
    resnet50 = importlib.import_module("hub_models.resnet50.model")
    classes = resnet50.top_classes(np.array([1.0, 3.0, 2.0]), ["a", "b", "c"], 2)
    assert [c["label"] for c in classes] == ["b", "c"]
    assert classes[0]["confidence"] == pytest.approx(np.exp(3) / np.exp([1, 2, 3]).sum())


@pytest.mark.unit
def test_rf_detr_top_proposals_are_stable():
    rf_detr = importlib.import_module("hub_models.rf_detr.model")
    scores = np.zeros(400, dtype=np.float32)
    scores[[7, 3]] = 2.0
    proposals = np.repeat(np.arange(400, dtype=np.float32)[:, None], 4, axis=1)
    gathered = rf_detr.top_proposals(scores, proposals)
    assert gathered.shape == (1, rf_detr.TOP_K, 4)
    assert list(gathered[0, :3, 0]) == [3.0, 7.0, 0.0]


@pytest.mark.unit
def test_rf_detr_detections_decode_to_frame_pixels():
    rf_detr = importlib.import_module("hub_models.rf_detr.model")
    labels = ["unused", "person"] + [f"c{i}" for i in range(2, rf_detr.NUM_CLASSES)]
    boxes = np.zeros((rf_detr.TOP_K, 4), dtype=np.float32)
    boxes[5] = [0.5, 0.5, 0.2, 0.4]
    logits = np.full((rf_detr.TOP_K, rf_detr.NUM_CLASSES), -10.0, dtype=np.float32)
    logits[5, 1] = 5.0
    logits[6, 0] = 5.0  # "unused" class is skipped
    objects = rf_detr.detections(boxes, logits, labels, 200, 100, 0.5, 10)
    assert len(objects) == 1
    assert objects[0]["label"] == "person"
    assert objects[0]["bbox"] == pytest.approx([80.0, 30.0, 40.0, 40.0])
