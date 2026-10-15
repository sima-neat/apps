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
def test_package_validation_only_resolves_the_archive_path(tmp_path: Path) -> None:
    from model_profiles import inspect_package, profile_named

    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    package = inspect_package(package_path, profile_named("detect"))
    assert package.path == package_path.resolve()


@pytest.mark.unit
def test_package_validation_leaves_sdk_compatibility_to_neat(tmp_path: Path) -> None:
    from model_profiles import inspect_package, profile_named

    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path, sdk="2.1.2", decoder=True)
    package = inspect_package(package_path, profile_named("detect"))
    assert package.path == package_path.resolve()


@pytest.mark.unit
def test_raw_segmentation_accepts_sdk_200_without_packaged_decoder(tmp_path: Path) -> None:
    from model_profiles import inspect_package, profile_named

    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path, sdk="2.0.0")
    profile = profile_named("segment")
    package = inspect_package(package_path, profile)

    assert package.path == package_path.resolve()
    assert profile.decode_type == "YoloV26Seg"


@pytest.mark.unit
def test_registry_fetch_selects_exact_flat_artifact(tmp_path: Path, monkeypatch) -> None:
    import model_profiles

    profile = model_profiles.profile_named("ssd")

    def fake_install(command, *, check) -> None:
        assert check is True
        staging = Path(command[-1])
        write_package(staging / profile.archive)
        write_package(staging / "ssd_mobilenet_v3_modalix_bf16_tess_mla_mpk.tar.gz")

    monkeypatch.setattr(model_profiles.subprocess, "run", fake_install)
    package = model_profiles.fetch_profile(profile, tmp_path / "models")

    assert package.path == (tmp_path / "models" / profile.archive).resolve()


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
            input_max_width=0,
            input_max_height=0,
            input_max_depth=0,
            preset="auto",
            resize=SimpleNamespace(enable="auto", mode="auto"),
            normalize=SimpleNamespace(
                enable="auto", mean=[], stddev=[], has_explicit_stats=False
            ),
            color_convert=SimpleNamespace(input_format="auto", output_format="auto"),
        )
        self.preprocess.resize.width = 0
        self.preprocess.resize.height = 0
        self.preprocess.resize.pad_value = 0
        self.advanced_execution = SimpleNamespace(
            preprocess_target="", postprocess_target=""
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
        ResizeMode=SimpleNamespace(Stretch="stretch", Letterbox="letterbox"),
        NormalizePreset=SimpleNamespace(COCO_YOLO="coco-yolo"),
    )


@pytest.mark.unit
def test_yolo_options_select_runtime_decoder_and_policy(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("detect")
    package = module.inspect_package(package_path, profile)
    options = module.model_options(fake_pyneat(), profile, package)

    assert options.preprocess.color_convert.input_format == "nv12"
    assert options.decode_type == "yolo26"
    assert options.score_threshold == pytest.approx(0.55)
    assert options.nms_iou_threshold == pytest.approx(0.45)
    assert options.top_k == 50
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
    model_added = []
    route_seen = None

    class ModelGraph:
        def add(self, node) -> None:
            model_added.append(node)

    class Model:
        def __init__(self, path, options) -> None:
            assert path == str(package_path.resolve())
            assert options.preprocess.color_convert.input_format == "nv12"

        def graph(self, route):
            nonlocal route_seen
            route_seen = route
            return ModelGraph()

    class Graph:
        def __init__(self, name) -> None:
            assert name == "mipi_classify_camera"

        def add(self, node) -> None:
            added.append(node)

    fake = fake_pyneat()
    fake.Model = Model
    fake.ModelRouteOptions = lambda: SimpleNamespace(
        include_input=True,
        include_output=False,
        upstream_name="",
        buffer_name="",
        name_suffix="",
        advanced_execution=SimpleNamespace(preprocess_target="", postprocess_target=""),
    )
    fake.CameraInputOptions = lambda: SimpleNamespace()
    fake.Graph = Graph
    fake.nodes = SimpleNamespace(
        camera_input=lambda options, **kwargs: ("camera", options, kwargs),
        output=lambda name: ("output", name),
    )

    module.make_graph(fake, profile, package)
    assert route_seen.upstream_name == "camera0"
    assert route_seen.buffer_name == "camera0"
    assert route_seen.include_input is False
    assert route_seen.include_output is False
    assert added[0][0] == "camera"
    assert added[0][1].buffer_name == "camera0"
    assert added[0][1].allow_cpu_fallback is False
    assert added[0][2]["capture_buffer_count"] == 32
    assert isinstance(added[1], ModelGraph)
    assert model_added == [("output", "results")]


@pytest.mark.unit
def test_graph_camera_copy_is_an_explicit_opt_in(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("classify")
    package = module.inspect_package(package_path, profile)
    camera_seen = None

    class ModelGraph:
        def add(self, node) -> None:
            pass

    class Model:
        def __init__(self, path, options) -> None:
            pass

        def graph(self, route):
            return ModelGraph()

    class Graph:
        def __init__(self, name) -> None:
            pass

        def add(self, node) -> None:
            pass

    def camera_input(options, **kwargs):
        nonlocal camera_seen
        camera_seen = options
        return "camera"

    fake = fake_pyneat()
    fake.Model = Model
    fake.CameraInputOptions = lambda: SimpleNamespace()
    fake.ModelRouteOptions = lambda: SimpleNamespace(
        include_input=True,
        include_output=False,
        upstream_name="",
        buffer_name="",
        name_suffix="",
        advanced_execution=SimpleNamespace(preprocess_target="", postprocess_target=""),
    )
    fake.Graph = Graph
    fake.nodes = SimpleNamespace(camera_input=camera_input, output=lambda name: name)

    module.make_graph(fake, profile, package, allow_camera_copy=True)
    assert camera_seen.allow_cpu_fallback is True


@pytest.mark.unit
def test_insight_graph_keeps_model_and_video_branches_realtime(tmp_path: Path) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("segment")
    package = module.inspect_package(package_path, profile)
    graphs = []
    connections = []

    class Graph:
        def __init__(self, name) -> None:
            self.name = name
            self.added = []
            graphs.append(self)

        def add(self, node) -> None:
            self.added.append(node)

        def connect(self, *nodes) -> None:
            connections.append(nodes)

    class Model:
        def __init__(self, path, options) -> None:
            assert path == str(package_path.resolve())

        def graph(self, route):
            assert route.upstream_name == "images"
            return Graph("model")

    fake = fake_pyneat()
    fake.Model = Model
    fake.Graph = Graph
    fake.CameraInputOptions = lambda: SimpleNamespace()
    fake.ModelRouteOptions = lambda: SimpleNamespace(
        include_input=True,
        include_output=False,
        upstream_name="",
        buffer_name="",
        name_suffix="",
        advanced_execution=SimpleNamespace(preprocess_target="", postprocess_target=""),
    )
    fake.InputOptions = lambda: SimpleNamespace()
    fake.InputMemoryPolicy = SimpleNamespace(Ev74="ev74")
    fake.PayloadType = SimpleNamespace(Image="image")
    fake.Format = SimpleNamespace(NV12="nv12")
    fake.GraphLinkOptions = lambda: SimpleNamespace(policy=None)
    fake.GraphLinkPolicy = SimpleNamespace(RealtimeLatestByStream="latest")
    fake.VideoSenderOptions = SimpleNamespace(
        h264_rtp_udp_from_raw=lambda *_args: SimpleNamespace(
            host="",
            channel=0,
            video_port_base=0,
            encoder=SimpleNamespace(bitrate_kbps=0),
        )
    )
    fake.nodes = SimpleNamespace(
        camera_input=lambda options, **kwargs: ("camera", options, kwargs),
        input=lambda name, options: ("input", name, options),
        output=lambda name: ("output", name),
    )
    fake.groups = SimpleNamespace(video_sender=lambda options: ("sender", options))
    fake.graphs = SimpleNamespace(branch=lambda source, outputs: ("branch", source, outputs))

    graph = module.make_graph(fake, profile, package, insight_host="127.0.0.1")

    assert graph.name == "mipi_segment_insight"
    assert len(connections) == 3
    assert connections[1][-1].policy == "latest"
    assert connections[2][-1].policy == "latest"


@pytest.mark.unit
def test_strict_zero_copy_contract() -> None:
    module = load_module()
    module.require_strict_zero_copy("libcamerasrc simaai-zero-copy-required=true ! output")
    module.require_strict_zero_copy(
        "libcamerasrc external-buffer-mode=required ! "
        "neatcamerabridge copy-allowed=false ! output"
    )
    with pytest.raises(RuntimeError, match="does not require"):
        module.require_strict_zero_copy("libcamerasrc ! output")
    with pytest.raises(RuntimeError, match="permits a CPU copy"):
        module.require_strict_zero_copy(
            "libcamerasrc simaai-zero-copy-required=true ! neatcamerabridge ! output"
        )
