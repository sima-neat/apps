"""Unit tests for the Python MIPI multi-model example."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_DIR = EXAMPLE_DIR / "src" / "python"
MAIN_PY = PYTHON_DIR / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
sys.path.insert(0, str(PYTHON_DIR))


def load_module():
    spec = importlib.util.spec_from_file_location("mipi_multi_model", MAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_package(path: Path, *, sdk: str = "2.1.3", decoder: bool = False) -> None:
    members = {
        "fixture_mpk.json": json.dumps(
            {"name": "fixture", "model_sdk_version": sdk}
        ).encode(),
        "fixture_stage1_mla.elf": b"ELF fixture",
    }
    if decoder:
        members["0_boxdecoder.json"] = json.dumps(
            {
                "decode_type": "yolo",
                "detection_threshold": 0.51,
                "nms_iou_threshold": 0.62,
                "topk": 24,
            }
        ).encode()
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.mark.unit
def test_help_and_profile_listing_do_not_import_pyneat() -> None:
    help_result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    list_result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--list-profiles"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert help_result.returncode == 0
    assert list_result.returncode == 0
    assert all(name in list_result.stdout for name in ("detect", "pose", "segment", "ssd"))


@pytest.mark.unit
def test_default_config_is_minimal_and_valid() -> None:
    module = load_module()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert set(raw) == {"model", "runtime"}
    assert set(raw["model"]) == {"profile", "path"}
    assert module.main(["--config", str(DEFAULT_CONFIG), "--validate-config-only"]) == 0


@pytest.mark.unit
def test_invalid_profile_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["model"]["profile"] = "unknown"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--validate-config-only"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "unknown profile" in result.stderr


@pytest.mark.unit
def test_package_validation_reads_its_decoder_policy(tmp_path: Path) -> None:
    from model_profiles import inspect_package, profile_named

    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path, decoder=True)
    package = inspect_package(package_path, profile_named("detect"))
    assert package.name == "fixture"
    assert package.decoder_policy is not None
    assert package.decoder_policy.score_threshold == pytest.approx(0.51)
    assert package.decoder_policy.nms_iou_threshold == pytest.approx(0.62)
    assert package.decoder_policy.top_k == 24


@pytest.mark.unit
def test_package_validation_rejects_wrong_sdk(tmp_path: Path) -> None:
    from model_profiles import ModelPackageError, inspect_package, profile_named

    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path, sdk="2.1.2", decoder=True)
    with pytest.raises(ModelPackageError, match="expected 2.1.3"):
        inspect_package(package_path, profile_named("detect"))


class FakeOptions:
    def __init__(self) -> None:
        self.decode_type = "auto"
        self.score_threshold = "default-score"
        self.nms_iou_threshold = "default-nms"
        self.top_k = "default-top-k"
        self.num_classes = "default-classes"
        self.untouched = "default"
        self.preprocess = SimpleNamespace(
            kind="auto",
            enable="auto",
            resize=SimpleNamespace(enable="auto", mode="auto"),
            normalize=SimpleNamespace(
                enable="auto", mean=[], stddev=[], has_explicit_stats=False
            ),
            color_convert=SimpleNamespace(input_format="auto", output_format="auto"),
        )


def fake_pyneat():
    return SimpleNamespace(
        ModelOptions=FakeOptions,
        PreprocessColorFormat=SimpleNamespace(NV12="nv12", RGB="rgb"),
        BoxDecodeType=SimpleNamespace(
            YoloV26="yolo26", YoloV26Pose="pose", YoloV26Seg="seg", Ssd="ssd"
        ),
        InputKind=SimpleNamespace(Image="image"),
        AutoFlag=SimpleNamespace(On="on"),
        ResizeMode=SimpleNamespace(Stretch="stretch"),
    )


@pytest.mark.unit
def test_yolo_options_preserve_package_policy_without_extra_overrides(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path, decoder=True)
    profile = module.profile_named("detect")
    package = module.inspect_package(package_path, profile)
    options = module.model_options(fake_pyneat(), profile, package)

    assert options.preprocess.color_convert.input_format == "nv12"
    assert options.decode_type == "yolo26"
    assert options.score_threshold == pytest.approx(0.51)
    assert options.nms_iou_threshold == pytest.approx(0.62)
    assert options.top_k == 24
    assert options.untouched == "default"


@pytest.mark.unit
def test_classification_leaves_negotiated_defaults_alone(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("classify")
    package = module.inspect_package(package_path, profile)
    options = module.model_options(fake_pyneat(), profile, package)

    assert options.preprocess.color_convert.input_format == "nv12"
    assert options.decode_type == "auto"
    assert options.score_threshold == "default-score"
    assert options.preprocess.resize.enable == "auto"
    assert options.untouched == "default"


@pytest.mark.unit
def test_ssd_sets_only_its_required_recipe(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("ssd")
    package = module.inspect_package(package_path, profile)
    options = module.model_options(fake_pyneat(), profile, package)

    assert options.decode_type == "ssd"
    assert options.preprocess.color_convert.input_format == "nv12"
    assert options.preprocess.color_convert.output_format == "rgb"
    assert options.preprocess.resize.mode == "stretch"
    assert options.preprocess.normalize.mean == [0.485, 0.456, 0.406]
    assert options.num_classes == 91
    assert options.score_threshold == "default-score"
    assert options.top_k == "default-top-k"


@pytest.mark.unit
def test_graph_has_named_camera_route_and_manual_output(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("classify")
    package = module.inspect_package(package_path, profile)
    added = []
    route_seen = None

    class Model:
        def __init__(self, path, options) -> None:
            assert path == str(package_path.resolve())
            assert options.preprocess.color_convert.input_format == "nv12"

        def graph(self, route):
            nonlocal route_seen
            route_seen = route
            return "model"

    class Graph:
        def __init__(self, name) -> None:
            assert name == "mipi_classify"

        def add(self, node) -> None:
            added.append(node)

    fake = fake_pyneat()
    fake.Model = Model
    fake.ModelRouteOptions = lambda: SimpleNamespace(upstream_name="")
    fake.Graph = Graph
    fake.nodes = SimpleNamespace(
        camera_input=lambda: "camera",
        output=lambda name: ("output", name),
    )

    module.make_graph(fake, profile, package)
    assert route_seen.upstream_name == "camera"
    assert added == ["camera", "model", ("output", "results")]


@pytest.mark.unit
def test_strict_zero_copy_contract() -> None:
    module = load_module()
    module.require_strict_zero_copy("libcamerasrc simaai-zero-copy-required=true ! output")
    with pytest.raises(RuntimeError, match="does not require"):
        module.require_strict_zero_copy("libcamerasrc ! output")
    with pytest.raises(RuntimeError, match="CPU camera bridge"):
        module.require_strict_zero_copy(
            "libcamerasrc simaai-zero-copy-required=true ! neatcamerabridge ! output"
        )
