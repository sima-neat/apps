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


STRICT_BACKEND = (
    "libcamerasrc name=camera_src0 ! "
    "capsfilter name=camera_caps0 ! "
    "neatcamerabridge name=camera_bridge0 buffer-name=camera0 copy-allowed=false ! output"
)


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
    assert set(raw) == {"model", "runtime", "insight"}
    assert set(raw["model"]) == {"profile", "path"}
    assert raw["insight"]["host"] == ""
    assert module.main(["--config", str(DEFAULT_CONFIG), "--validate-config-only"]) == 0


@pytest.mark.unit
def test_insight_settings_come_from_config_with_cli_overrides(tmp_path: Path) -> None:
    module = load_module()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["insight"] = {
        "host": "192.168.1.127",
        "video_port": 9200,
        "metadata_port": 9300,
        "channel": 2,
        "bitrate_kbps": 4000,
    }
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    loaded = module.load_config(config)
    assert (
        loaded.insight_host,
        loaded.insight_video_port,
        loaded.insight_metadata_port,
        loaded.insight_channel,
        loaded.insight_bitrate_kbps,
    ) == ("192.168.1.127", 9200, 9300, 2, 4000)

    args = SimpleNamespace(
        profile=None,
        model=None,
        frames=None,
        timeout_ms=None,
        insight_host="10.0.0.5",
        insight_video_port=None,
        insight_metadata_port=None,
        insight_channel=0,
        insight_bitrate_kbps=None,
    )
    merged = module._apply_overrides(loaded, args)
    assert (merged.insight_host, merged.insight_video_port, merged.insight_channel) == (
        "10.0.0.5",
        9200,
        0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("insight", "message"),
    [
        ({"host": 5}, "insight.host must be a string"),
        ({"video_port": 0}, "insight-video-port must be a positive integer"),
        ({"channel": -1}, "insight-channel must be a non-negative integer"),
    ],
)
def test_invalid_insight_config_is_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], insight: dict, message: str
) -> None:
    module = load_module()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["insight"].update(insight)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert module.main(["--config", str(config), "--validate-config-only"]) == 2
    assert message in capsys.readouterr().err


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
def test_package_accepts_any_compiled_tar_gz_name(tmp_path: Path) -> None:
    from model_profiles import ModelPackageError, inspect_package, profile_named

    package_path = tmp_path / "yolo26n-det-bf16-mla_tess-b1.tar.gz"
    write_package(package_path)
    package = inspect_package(package_path, profile_named("detect"))
    assert package.path == package_path.resolve()

    onnx_path = tmp_path / "yolo26n.onnx"
    onnx_path.write_bytes(b"onnx")
    with pytest.raises(ModelPackageError, match=r"\.tar\.gz"):
        inspect_package(onnx_path, profile_named("detect"))


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
    assert options.preprocess.normalize.mean == [0.5, 0.5, 0.5]
    assert options.preprocess.normalize.stddev == [0.5, 0.5, 0.5]
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

        def describe_backend(self, _verbose: bool) -> str:
            return STRICT_BACKEND

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
    assert added[0][2] == {}
    assert isinstance(added[1], ModelGraph)
    assert model_added == [("output", "results")]


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

        def describe_backend(self, _verbose: bool) -> str:
            return STRICT_BACKEND if self.name.endswith("_camera") else "ExecutionGraphPlan {}"

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
    module.require_strict_zero_copy(STRICT_BACKEND)
    with pytest.raises(RuntimeError, match="no Neat camera bridge"):
        module.require_strict_zero_copy("libcamerasrc name=camera_src0 ! output")
    with pytest.raises(RuntimeError, match="permits a CPU copy"):
        module.require_strict_zero_copy("libcamerasrc ! neatcamerabridge ! output")
    with pytest.raises(RuntimeError, match="permits a CPU copy"):
        module.require_strict_zero_copy(
            STRICT_BACKEND.replace("copy-allowed=false", "copy-allowed=true")
        )


@pytest.mark.unit
@pytest.mark.parametrize("insight_host", ["", "127.0.0.1"])
def test_graph_rejects_copying_camera_before_composition(
    tmp_path: Path, insight_host: str
) -> None:
    module = load_module()
    package_path = tmp_path / "fixture_mpk.tar.gz"
    write_package(package_path)
    profile = module.profile_named("detect")
    package = module.inspect_package(package_path, profile)
    composed = []

    class Graph:
        def __init__(self, name) -> None:
            self.name = name

        def add(self, node) -> None:
            composed.append(node)

        def describe_backend(self, _verbose: bool) -> str:
            return STRICT_BACKEND.replace("copy-allowed=false", "copy-allowed=true")

    class Model:
        def __init__(self, path, options) -> None:
            pass

        def graph(self, route):
            raise AssertionError("the model must not be composed onto a copying camera")

    fake = fake_pyneat()
    fake.Model = lambda path, options: Model(path, options)
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
    fake.nodes = SimpleNamespace(camera_input=lambda options: ("camera", options))

    with pytest.raises(RuntimeError, match="permits a CPU copy"):
        module.make_graph(fake, profile, package, insight_host=insight_host)
    assert composed == [("camera", composed[0][1])]


@pytest.mark.unit
def test_camera_copy_escape_hatch_is_not_offered() -> None:
    module = load_module()
    with pytest.raises(SystemExit):
        module.main(["--allow-camera-copy", "--validate-config-only"])


@pytest.mark.unit
def test_insight_labels_ship_with_the_example() -> None:
    module = load_module()
    assert module.DEFAULT_LABELS.parent == EXAMPLE_DIR / "src" / "common"
    assert module.load_labels()[0] == "person"


def fake_tensor(values):
    import numpy as np

    array = np.asarray(values)
    return SimpleNamespace(shape=array.shape, to_numpy=lambda copy=True: array)


class RecordingSender:
    def __init__(self) -> None:
        self.sent = []

    def send_metadata(self, metadata_type, data_json, timestamp_ms, frame_id) -> None:
        self.sent.append((metadata_type, json.loads(data_json), timestamp_ms, frame_id))


def insight_sample():
    return SimpleNamespace(
        tensor=None, tensors=["raw"], fields=[], pts_ns=2_500_000_000, frame_id=7
    )


@pytest.mark.unit
def test_detection_metadata_is_sent_as_clamped_insight_objects() -> None:
    module = load_module()
    rows = [[-10.0, 20.0, 2000.0, 300.0, 0.9, 0.0], [100.0, 100.0, 200.0, 250.0, 0.6, 999.0]]
    fake = SimpleNamespace(decode_bbox=lambda tensors: [fake_tensor(rows)])
    sender = RecordingSender()

    count = module.send_insight_metadata(
        sender, fake, module.profile_named("detect"), insight_sample(), ["person"]
    )

    assert count == 2
    assert sender.sent == [
        (
            "object-detection",
            {
                "objects": [
                    {"id": "obj_1", "label": "person", "confidence": pytest.approx(0.9),
                     "bbox": [0.0, 20.0, 1920.0, 280.0]},
                    {"id": "obj_2", "label": "unknown", "confidence": pytest.approx(0.6),
                     "bbox": [100.0, 100.0, 100.0, 150.0]},
                ]
            },
            2500,
            "7",
        )
    ]


@pytest.mark.unit
def test_segmentation_metadata_sends_mask_polygons_in_frame_coordinates() -> None:
    import numpy as np

    module = load_module()
    # A 640x640 letterboxed model with stride-4 masks: frame (480, 270)-(1440, 810)
    # maps to mask columns 40-120 and rows 57.5-102.5.
    mask = np.zeros((1, 160, 160), dtype=np.uint8)
    mask[0, 58:102, 40:120] = 255
    decoded = SimpleNamespace(
        boxes=fake_tensor([[480.0, 270.0, 1440.0, 810.0, 0.8, 0.0]]),
        masks=fake_tensor(mask),
    )
    fake = SimpleNamespace(decode_segmentation=lambda tensors, **kwargs: [decoded])
    sender = RecordingSender()

    count = module.send_insight_metadata(
        sender, fake, module.profile_named("segment"), insight_sample(), ["person"]
    )

    assert count == 1
    metadata_type, data, timestamp_ms, frame_id = sender.sent[0]
    assert (metadata_type, timestamp_ms, frame_id) == ("segmentation", 2500, "7")
    segment = data["segments"][0]
    assert segment["label"] == "person"
    assert segment["bbox"] == [480, 270, 960, 540]
    assert segment["mask_format"] == "polygon"
    polygon = np.asarray(segment["mask"])
    assert len(polygon) >= 4
    assert polygon[:, 0].min() >= 480 and polygon[:, 0].max() <= 1440
    assert polygon[:, 1].min() >= 270 and polygon[:, 1].max() <= 810
    assert polygon[:, 0].max() - polygon[:, 0].min() > 900
    assert polygon[:, 1].max() - polygon[:, 1].min() > 500


@pytest.mark.unit
def test_segment_encoding_keeps_strongest_masks_within_udp_budget() -> None:
    module = load_module()
    polygon = [[index, index] for index in range(400)]
    segments = [
        {"id": f"seg_{index}", "label": "person", "confidence": index / 100,
         "bbox": [0, 0, 1, 1], "mask_format": "polygon", "mask": polygon}
        for index in range(20)
    ]

    data_json, count = module.encode_segments(segments)

    assert 0 < count < len(segments)
    assert len(data_json) <= module.METADATA_BYTE_BUDGET
    kept = json.loads(data_json)["segments"]
    assert [item["id"] for item in kept] == [f"seg_{index}" for index in range(19, 19 - count, -1)]
