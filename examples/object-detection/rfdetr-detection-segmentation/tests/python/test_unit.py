"""Unit tests for the RF-DETR example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.utils.config_cases import config_writer, load_example_main
from tests.utils.fake_run import FakeRun
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
main = load_example_main(EXAMPLE_DIR, "rfdetr_main")


@pytest.mark.unit
def test_config_argument_is_required():
    with pytest.raises(SystemExit):
        main.parse_args([])


@pytest.mark.unit
@pytest.mark.parametrize("storage_kind", ["CpuOwned", "CpuExternal", "GstSample"])
def test_transformer_inputs_copy_only_cpu_features(monkeypatch, storage_kind):
    kinds = SimpleNamespace(CpuOwned="CpuOwned", CpuExternal="CpuExternal")
    monkeypatch.setattr(main, "pyneat", SimpleNamespace(StorageKind=kinds), raising=False)
    device_feature = SimpleNamespace(shape=[1, 2, 2, 256])
    feature = SimpleNamespace(
        shape=[1, 2, 2, 256],
        storage=SimpleNamespace(kind=storage_kind),
        cvu=Mock(return_value=device_feature),
    )
    gathered = SimpleNamespace(shape=[1, 2, 4])
    input_shapes = (tuple(gathered.shape), tuple(device_feature.shape))

    inputs = main.transformer_inputs(input_shapes, feature, gathered, 2)

    assert inputs[0] is gathered
    assert list(inputs[1].shape) == [1, 2, 2, 256]
    if storage_kind == "GstSample":
        assert inputs[1] is feature
        feature.cvu.assert_not_called()
    else:
        assert inputs[1] is device_feature
        feature.cvu.assert_called_once_with()


def fake_model_pair(monkeypatch, side, top_k, mask_size=0, batch=1):
    """Backbone and transformer specs shaped like an RF-DETR registry model pair."""
    dtypes = SimpleNamespace(Float32="fp32", UInt8="uint8")
    monkeypatch.setattr(main, "pyneat", SimpleNamespace(TensorDType=dtypes), raising=False)

    def specs(shapes, dtype=dtypes.Float32):
        return [SimpleNamespace(shape=shape, dtypes=[dtype]) for shape in shapes]

    backbone = SimpleNamespace(
        input_specs=lambda: specs([[-1, -1, 3]], dtypes.UInt8),
        output_specs=lambda: specs(
            [[1, side, side, 256], [1, side * side], [1, side * side, 4]]
        ),
    )
    outputs = [[1, top_k, 4], [1, top_k, 91]]
    if mask_size:
        outputs.append([1, mask_size, mask_size, top_k])
    transformer = SimpleNamespace(
        input_specs=lambda: specs([[batch, side, side, 256], [1, top_k, 4]]),
        output_specs=lambda: specs(outputs),
    )
    return backbone, transformer


# Feature grid, query count, and mask size of the registry packages.
MODEL_SIZES = [
    pytest.param("detection", 24, 300, 0, id="detection-nano"),
    pytest.param("detection", 44, 300, 0, id="detection-large"),
    pytest.param("segmentation", 26, 100, 78, id="segmentation-nano"),
    pytest.param("segmentation", 32, 100, 96, id="segmentation-small"),
    pytest.param("segmentation", 36, 200, 108, id="segmentation-medium"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("task", "side", "top_k", "mask_size"), MODEL_SIZES)
def test_model_contract_reads_the_sizes_of_the_loaded_pair(
    monkeypatch, task, side, top_k, mask_size
):
    backbone, transformer = fake_model_pair(monkeypatch, side, top_k, mask_size)
    cfg = SimpleNamespace(task=task, mask_grid_size=640)

    contract = main.model_contract(cfg, backbone, transformer)

    assert contract == main.ModelContract(side=side, top_k=top_k, mask_size=mask_size)


@pytest.mark.unit
@pytest.mark.parametrize("task", ["detection", "segmentation"])
def test_model_contract_rejects_a_batched_feature_input(monkeypatch, task):
    mask_size = 108 if task == "segmentation" else 0
    backbone, transformer = fake_model_pair(monkeypatch, 36, 200, mask_size, batch=2)
    cfg = SimpleNamespace(task=task, mask_grid_size=640)

    with pytest.raises(RuntimeError, match="unexpected I/O contract"):
        main.model_contract(cfg, backbone, transformer)


@pytest.mark.unit
def test_mask_grid_below_the_model_mask_size_is_rejected(monkeypatch):
    backbone, transformer = fake_model_pair(monkeypatch, 26, 100, 78)
    cfg = SimpleNamespace(task="segmentation", mask_grid_size=77)

    with pytest.raises(ValueError, match=r"mask_grid_size must be >= the model's mask size \(78\)"):
        main.model_contract(cfg, backbone, transformer)


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
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.5, 0.25]
    logits = np.full((1, 300, 91), -20.0, dtype=np.float32)
    logits[0, 0, 1] = 10.0

    objects = main.postprocess(boxes, logits, 1920, 1080, labels, 0.5, 10)

    assert len(objects) == 1
    assert objects[0]["label"] == "person"
    assert objects[0]["bbox"] == pytest.approx([480.0, 405.0, 960.0, 270.0])


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["nano", "small", "medium", "large"])
def test_config_selects_one_model_pair(tmp_path, variant):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(f"label-{index}" for index in range(91)) + "\n")
    config = {
        "model": {
            "task": "detection",
            "labels": str(labels),
            "detection": {
                "variant": variant,
                **{
                    name: {"backbone": f"{name}-b.tar.gz", "transformer": f"{name}-t.tar.gz"}
                    for name in ("nano", "small", "medium", "large")
                },
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
    assert selected.backbone.startswith(variant)
    assert selected.codec == "h265"
    assert (selected.width, selected.height, selected.fps) == (0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["nano", "small", "medium"])
@pytest.mark.parametrize("mask_grid_size", [None, 78, 432, 640, 0])
def test_config_selects_segmentation_model_pair(tmp_path, variant, mask_grid_size):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n".join(f"label-{index}" for index in range(91)) + "\n")
    config = {
        "model": {
            "task": "segmentation",
            "labels": str(labels),
            "segmentation": {
                "variant": variant,
                **{
                    name: {"backbone": f"{name}-b.tar.gz", "transformer": f"{name}-t.tar.gz"}
                    for name in ("nano", "small", "medium")
                },
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
    if mask_grid_size == 0:
        with pytest.raises(ValueError, match="mask_grid_size must be > 0"):
            main.load_config(path)
        return

    selected = main.load_config(path)

    assert selected.mask_grid_size == (mask_grid_size or 640)
    assert selected.task == "segmentation"
    assert selected.variant == variant
    assert selected.backbone == f"{variant}-b.tar.gz"


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
def test_config_rejects_variant_without_model_pair(tmp_path):
    config = {
        "model": {"task": "detection", "detection": {"variant": "large"}},
        "source": {},
        "inference": {},
        "output": {"insight": {}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="large"):
        main.load_config(path)


@pytest.mark.unit
@pytest.mark.parametrize(("top_k", "mask_size"), [(200, 108), (100, 96), (100, 78)])
@pytest.mark.parametrize("mask_grid_size", [None, 432, 640])
def test_segmentation_metadata_contains_polygons(top_k, mask_size, mask_grid_size):
    labels = ["unused"] * 91
    labels[1] = "person"
    boxes = np.zeros((1, top_k, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.5, 0.5]
    boxes[0, 1] = [0.5, 0.5, 0.5, 0.5]
    logits = np.full((1, top_k, 91), -20.0, dtype=np.float32)
    logits[0, 0, 0] = 12.0
    logits[0, 0, 1] = 11.0
    logits[0, 1, 1] = 10.0
    masks = np.full((1, mask_size, mask_size, top_k), -20.0, dtype=np.float32)
    masks[0, mask_size * 3 // 8 : mask_size * 5 // 8, mask_size * 3 // 8 : mask_size * 5 // 8, 0] = 10.0

    payload = main.segmentation_metadata(
        boxes, logits, masks, 1280, 720, labels, 0.3, 1, 0.08,
        mask_grid_size or mask_size, top_k, mask_size,
    )

    segments = json.loads(payload)["segments"]
    assert len(payload.encode()) <= main.METADATA_BYTE_BUDGET
    assert len(segments) == 1
    segment = segments[0]
    assert segment["label"] == "person"
    assert segment["mask_format"] == "polygon"
    assert len(segment["mask"]) >= 3
    assert all(0 <= x < 1280 and 0 <= y < 720 for x, y in segment["mask"])


# ---------------------------------------------------------------------------
# Configuration handling and option validation (Refs #526).
# Each test starts from VALID_CONFIG and breaks exactly one thing, so a failure
# names the rule that fired rather than "config rejected". The model-variant
# choice and the segmentation mask grid are owned by the tests above.
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "model": {
        "task": "detection",
        "labels": "labels.txt",
        "detection": {
            "variant": "small",
            "small": {"backbone": "small-b.tar.gz", "transformer": "small-t.tar.gz"},
            "medium": {"backbone": "medium-b.tar.gz", "transformer": "medium-t.tar.gz"},
        },
        "segmentation": {"medium": {"backbone": "seg-b.tar.gz", "transformer": "seg-t.tar.gz"}},
    },
    "source": {
        "rtsp_url": "rtsp://127.0.0.1:8554/src1",
        "codec": "h264",
        "tcp": True,
        "latency_ms": 100,
        "width": 0,
        "height": 0,
        "fps": 0,
    },
    "inference": {
        "frames": 0,
        "detection": {"min_score": 0.5, "max_detections": 100},
        "segmentation": {"min_score": 0.3, "max_segments": 24, "mask_threshold": 0.08, "mask_grid_size": 640},
    },
    "output": {"insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100}},
}


# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_full_config = config_writer(VALID_CONFIG)


@pytest.mark.unit
class TestValidBaseline:
    def test_the_baseline_config_loads(self, tmp_path):
        """If this breaks, every rejection test below is testing the wrong thing."""
        cfg = main.load_config(write_full_config(tmp_path))

        assert cfg.task == "detection"
        assert cfg.backbone == "small-b.tar.gz"
        assert cfg.rtsp_url == "rtsp://127.0.0.1:8554/src1"
        assert cfg.insight_host == "127.0.0.1"

    def test_omitted_optional_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": {
                "labels": "labels.txt",
                "detection": {"small": {"backbone": "small-b.tar.gz", "transformer": "small-t.tar.gz"}},
            },
            "source": {"rtsp_url": "rtsp://127.0.0.1:8554/src1"},
            # inference.<task> must exist as a mapping; every value inside it may be omitted.
            "inference": {"detection": {}},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main.load_config(write_full_config(tmp_path, root=raw))

        assert cfg.task == "detection"
        assert cfg.variant == "small"
        assert cfg.codec == "h264"
        assert cfg.tcp == True
        assert cfg.latency_ms == 100
        assert cfg.width == 0
        assert cfg.height == 0
        assert cfg.fps == 0
        assert cfg.frames == 0
        assert cfg.min_score == pytest.approx(0.5)
        assert cfg.max_results == 100
        # Mask settings are read for the segmentation task; see TestSegmentationTask.
        assert cfg.video_port == 9000
        assert cfg.metadata_port == 9100


REJECTED = [
    pytest.param(('model', 'task'), 'tracking', 'model.task must be detection or segmentation', id='task-unsupported'),
    pytest.param(('model', 'detection', 'small', 'backbone'), '', 'model.detection.small backbone and transformer must be set', id='backbone-empty'),
    pytest.param(('model', 'labels'), '', 'model.labels must be set', id='labels-empty'),
    pytest.param(('source', 'rtsp_url'), 'http://127.0.0.1:8080/stream', 'source.rtsp_url must be an RTSP URL', id='rtsp-url-not-rtsp'),
    pytest.param(('source', 'codec'), 'vp9', 'source.codec must be h264/avc, h265/hevc, or mjpeg', id='codec-unsupported'),
    pytest.param(('source', 'latency_ms'), -1, 'source.latency_ms and inference.frames must be >= 0', id='latency-negative'),
    pytest.param(('inference', 'frames'), -1, 'source.latency_ms and inference.frames must be >= 0', id='frames-negative'),
    pytest.param(('source', 'width'), -1, 'source.width, source.height, and source.fps must be >= 0', id='width-negative'),
    pytest.param(('source', 'height'), -1, 'source.width, source.height, and source.fps must be >= 0', id='height-negative'),
    pytest.param(('source', 'fps'), -1, 'source.width, source.height, and source.fps must be >= 0', id='fps-negative'),
    pytest.param(('inference', 'detection', 'min_score'), 1.01, 'inference.detection.min_score must be in [0, 1]', id='min-score-above'),
    pytest.param(('inference', 'detection', 'min_score'), -0.01, 'inference.detection.min_score must be in [0, 1]', id='min-score-below'),
    pytest.param(('inference', 'detection', 'max_detections'), 0, 'inference.detection.max_detections must be > 0', id='max-detections-zero'),
    pytest.param(('output', 'insight', 'host'), '', 'output.insight.host must be set', id='insight-host-empty'),
    pytest.param(('output', 'insight', 'video_port'), 0, 'Insight ports must be in [1, 65535]', id='video-port-zero'),
    pytest.param(('output', 'insight', 'video_port'), 65536, 'Insight ports must be in [1, 65535]', id='video-port-above'),
    pytest.param(('output', 'insight', 'metadata_port'), 0, 'Insight ports must be in [1, 65535]', id='metadata-port-zero'),
]


@pytest.mark.unit
class TestRejectedValues:
    @pytest.mark.parametrize(("path", "value", "message"), REJECTED)
    def test_invalid_value_is_rejected_with_an_actionable_message(
        self, tmp_path, path, value, message
    ):
        with pytest.raises(ValueError) as excinfo:
            main.load_config(write_full_config(tmp_path, [(path, value)]))

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestBoundariesAreAccepted:
    """An off-by-one that rejects a legal value is the failure nobody writes a test for."""

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (('inference', 'detection', 'min_score'), 0.0),
            (('inference', 'detection', 'min_score'), 1.0),
            (('source', 'latency_ms'), 0),
            (('inference', 'frames'), 0),
            (('source', 'width'), 0),
            (('source', 'height'), 0),
            (('source', 'fps'), 0),
            (('inference', 'detection', 'max_detections'), 1),
            (('output', 'insight', 'video_port'), 1),
            (('output', 'insight', 'video_port'), 65535),
            (('output', 'insight', 'metadata_port'), 1),
        ],
    )
    def test_boundary_value_is_accepted(self, tmp_path, path, value):
        main.load_config(write_full_config(tmp_path, [(path, value)]))


@pytest.mark.unit
class TestSegmentationTask:
    """The segmentation task reads its own model pair and inference section."""

    def test_the_segmentation_pair_and_limits_are_selected(self, tmp_path):
        cfg = main.load_config(write_full_config(tmp_path, [(("model", "task"), "segmentation")]))

        assert cfg.task == "segmentation"
        assert cfg.variant == "medium"
        assert cfg.backbone == "seg-b.tar.gz"
        assert cfg.max_results == 24
        assert cfg.min_score == pytest.approx(0.3)

    def test_a_missing_segmentation_pair_is_rejected(self, tmp_path):
        overrides = [(("model", "task"), "segmentation"), (("model", "segmentation", "medium", "transformer"), "")]

        with pytest.raises(ValueError, match="model.segmentation.medium backbone and transformer must be set"):
            main.load_config(write_full_config(tmp_path, overrides))

    def test_mask_defaults_apply_to_the_segmentation_task(self, tmp_path):
        raw = {
            "model": {
                "labels": "labels.txt",
                "task": "segmentation",
                "segmentation": {"medium": {"backbone": "seg-b.tar.gz", "transformer": "seg-t.tar.gz"}},
            },
            "source": {"rtsp_url": "rtsp://127.0.0.1:8554/src1"},
            "inference": {"segmentation": {}},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main.load_config(write_full_config(tmp_path, root=raw))

        assert cfg.mask_threshold == pytest.approx(0.08)
        assert cfg.mask_grid_size == 640

    @pytest.mark.parametrize(("value",), [(0.0,), (1.0,)])
    def test_mask_threshold_boundaries_are_accepted(self, tmp_path, value):
        overrides = [(("model", "task"), "segmentation"), (("inference", "segmentation", "mask_threshold"), value)]

        main.load_config(write_full_config(tmp_path, overrides))

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("min_score", 1.01, "inference.segmentation.min_score must be in [0, 1]"),
            ("max_segments", 0, "inference.segmentation.max_segments must be > 0"),
            ("mask_threshold", 1.01, "inference.segmentation.mask_threshold must be in [0, 1]"),
        ],
    )
    def test_segmentation_limits_are_enforced(self, tmp_path, key, value, message):
        overrides = [(("model", "task"), "segmentation"), (("inference", "segmentation", key), value)]

        with pytest.raises(ValueError) as excinfo:
            main.load_config(write_full_config(tmp_path, overrides))

        assert message in str(excinfo.value)


@pytest.mark.unit
class TestMalformedConfigFiles:
    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        """A list at the root used to escape as an AttributeError from `raw.get`."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main.load_config(config_path)

    def test_an_empty_file_is_reported_as_missing_settings_not_a_crash(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="model must be a mapping"):
            main.load_config(config_path)


# After a pull returns no sample the backbone bridge keeps going; a runtime error raised by
# the pull stops the whole run through bridge_error. pyneat's pull returns None for a
# closed output as for a timeout. The bridge is a thread inside run(), so only the helper
# is tested here.
@pytest.mark.unit
def test_backbone_pull_timeout_is_not_a_sample():
    run = FakeRun("timeout")
    run.pull("backbone", 500)

    assert main.pull_result_has_sample(run, None, "backbone") is False
