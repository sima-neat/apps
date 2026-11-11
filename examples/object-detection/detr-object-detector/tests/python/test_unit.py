"""Unit tests for detr-object-detector (Python)."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("detr_object_detector", MAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves its annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_pyneat(decoded_boxes: list[SimpleNamespace]) -> tuple[SimpleNamespace, list[tuple]]:
    """A pyneat stand-in with the option fields, tensor factory, and BBOX parser the example uses.

    Returns the stand-in and the list that records each decode_bbox_tensor call.
    """
    decode_calls: list[tuple] = []

    def model_options() -> SimpleNamespace:
        return SimpleNamespace(
            preprocess=SimpleNamespace(resize=SimpleNamespace(), color_convert=SimpleNamespace())
        )

    def decode_bbox_tensor(
        tensor: str, width: int, height: int, max_detections: int, strict: bool
    ) -> SimpleNamespace:
        decode_calls.append((tensor, width, height, max_detections, strict))
        return SimpleNamespace(boxes=decoded_boxes)

    fake = SimpleNamespace(
        ModelOptions=model_options,
        InputKind=SimpleNamespace(Image="image"),
        AutoFlag=SimpleNamespace(On="on"),
        ResizeMode=SimpleNamespace(Stretch="stretch"),
        NormalizePreset=SimpleNamespace(ImageNet="imagenet"),
        PreprocessColorFormat=SimpleNamespace(BGR="bgr", RGB="rgb"),
        BoxDecodeType=SimpleNamespace(Detr="detr"),
        PixelFormat=SimpleNamespace(BGR="bgr"),
        TensorMemory=SimpleNamespace(EV74="ev74"),
        Tensor=SimpleNamespace(from_numpy=lambda array, **_: "image-tensor"),
        detections=SimpleNamespace(decode_bbox_tensor=decode_bbox_tensor),
    )
    return fake, decode_calls


def box(class_id: int, score: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x1=80.0, y1=30.0, x2=120.0, y2=70.0, score=score, class_id=class_id)


def write_config(tmp_path: Path, decode: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"decode:\n{decode}", encoding="utf-8")
    return config_path


@pytest.mark.unit
class TestConfig:
    """Config values map to Neat's DETR decoder, and values it would ignore are rejected."""

    def test_config_maps_to_the_detr_decoder(self, tmp_path: Path) -> None:
        detr = load_module()
        detr.pyneat, _ = fake_pyneat([])
        config = detr.load_config(
            write_config(tmp_path, "  confidence_threshold: 0.6\n  max_draw: 30\n")
        )

        options = detr.build_model_options(config)

        assert options.preprocess.kind == "image"
        assert options.preprocess.resize.mode == "stretch"
        assert options.preprocess.preset == "imagenet"
        assert options.preprocess.color_convert.input_format == "bgr"
        assert options.preprocess.color_convert.output_format == "rgb"
        assert options.decode_type == "detr"
        assert options.score_threshold == 0.6
        assert options.top_k == 30

    @pytest.mark.parametrize(
        ("decode", "message"),
        [
            ("  confidence_threshold: 0.0\n", "confidence_threshold"),
            ("  confidence_threshold: 1.5\n", "confidence_threshold"),
            ("  max_draw: 0\n", "max_draw"),
        ],
    )
    def test_values_neat_would_ignore_are_rejected(
        self, tmp_path: Path, decode: str, message: str
    ) -> None:
        detr = load_module()

        with pytest.raises(ValueError, match=message):
            detr.load_config(write_config(tmp_path, decode))


@pytest.mark.unit
class TestDetect:
    """One image goes through the model, and the decoder's BBOX tensor becomes boxes."""

    def run_detect(
        self, tmp_path: Path, decoded_boxes: list[SimpleNamespace], person_only: bool
    ) -> tuple[list[SimpleNamespace], list[tuple], list[list[str]]]:
        detr = load_module()
        detr.np = np
        detr.pyneat, decode_calls = fake_pyneat(decoded_boxes)
        config = detr.load_config(
            write_config(tmp_path, f"  max_draw: 50\n  person_only: {str(person_only).lower()}\n")
        )
        runner_inputs: list[list[str]] = []

        def run(inputs: list[str]) -> list[str]:
            runner_inputs.append(inputs)
            return ["bbox"]

        image_bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        boxes = detr.detect(SimpleNamespace(run=run), image_bgr, config)
        return boxes, decode_calls, runner_inputs

    def test_boxes_are_read_in_original_image_pixels(self, tmp_path: Path) -> None:
        dog = box(class_id=18)

        boxes, decode_calls, runner_inputs = self.run_detect(tmp_path, [dog], person_only=False)

        assert runner_inputs == [["image-tensor"]]
        assert decode_calls == [("bbox", 200, 100, 50, False)]
        assert boxes == [dog]

    def test_person_only_keeps_person_detections(self, tmp_path: Path) -> None:
        detr = load_module()
        person = box(class_id=detr.PERSON_CLASS_ID)

        boxes, _, _ = self.run_detect(tmp_path, [box(class_id=18), person], person_only=True)

        assert boxes == [person]

    def test_rejects_raw_model_heads(self, tmp_path: Path) -> None:
        detr = load_module()
        detr.np = np
        detr.pyneat, _ = fake_pyneat([])
        config = detr.load_config(write_config(tmp_path, "  max_draw: 50\n"))
        runner = SimpleNamespace(run=lambda inputs: ["logits", "boxes"])

        with pytest.raises(RuntimeError, match="one BBOX tensor"):
            detr.detect(runner, np.zeros((100, 200, 3), dtype=np.uint8), config)


@pytest.mark.unit
class TestLabels:
    def test_unused_coco_ids_get_a_numbered_label(self) -> None:
        detr = load_module()

        assert detr.class_name(detr.PERSON_CLASS_ID) == "person"
        assert detr.class_name(18) == "dog"
        assert detr.class_name(12) == "class_12"
        assert detr.class_name(91) == "class_91"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the DETR example."""

    def run_main(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MAIN_PY), *args],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(EXAMPLE_DIR),
        )

    def test_help(self) -> None:
        r = self.run_main("--help")

        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self) -> None:
        r = self.run_main("--config", "/nonexistent/detr-config.yaml")

        assert r.returncode == 2
        assert "failed to open config file" in r.stderr

    def test_unknown_flag(self) -> None:
        r = self.run_main("--bogus")

        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower()

    def test_max_draw_must_be_positive(self, tmp_path: Path) -> None:
        r = self.run_main("--config", str(write_config(tmp_path, "  max_draw: 0\n")))

        assert r.returncode == 2
        assert "decode.max_draw must be >= 1" in r.stderr
