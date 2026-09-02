"""Unit tests for pcb-defect-detector (Python)."""

import importlib.util
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
CONFIG_YAML = COMMON_DIR / "config.yaml"
LABELS_TXT = COMMON_DIR / "pcb_label.txt"
SCOPE_YAML = EXAMPLE_DIR / "tests" / "test-scope.yaml"
README_MD = EXAMPLE_DIR / "README.md"

_SPEC = importlib.util.spec_from_file_location("pcb_defect_detector_main", MAIN_PY)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main
_SPEC.loader.exec_module(main)


def bbox_payload(records: list[tuple[int, int, int, int, float, int]], declared: int | None = None) -> bytes:
    """Build a BBOX payload; declared overrides the record count in the header."""
    count = len(records) if declared is None else declared
    payload = struct.pack("<I", count)
    for x, y, w, h, score, class_id in records:
        payload += struct.pack(main.BBOX_RECORD_FORMAT, x, y, w, h, score, class_id)
    return payload


def valid_config() -> dict:
    return {
        "model": {"path": "models/pack.tar.gz", "labels": str(LABELS_TXT), "input_size": 640},
        "io": {"input_dir": "assets/datasets/pcb", "output_dir": "sandbox/pcb-defect-detector"},
        "decode": {"score_threshold": 0.25, "nms_iou": 0.45, "max_detections": 300},
        "runtime": {"timeout_ms": 8000, "num_runs": 1, "queue_depth": 8},
        "output": {"overlay": True},
    }


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the PCB defect detection pipeline."""

    def test_help(self):
        """--help should describe the config-driven CLI."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 0
        assert "--config" in r.stdout
        assert "--validate-config-only" in r.stdout

    def test_bad_config_path(self):
        """A missing config file exits with code 2, as the C++ twin does."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", "/nonexistent/pcb-config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "config file not found" in r.stderr

    def test_unknown_flag(self):
        """An unrecognized flag should cause argparse to exit with code 2."""
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--bogus"],
            capture_output=True, text=True, timeout=20,
        )
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()

    def test_threshold_overrides_are_parsed(self):
        """--score and --nms are optional float overrides."""
        args = main.parse_args(["--score", "0.4", "--nms", "0.55"])
        assert args.score == pytest.approx(0.4)
        assert args.nms == pytest.approx(0.55)

    def test_config_defaults_to_shared_common_config(self):
        """Without --config the packaged src/common/config.yaml is used."""
        assert main.parse_args([]).config == CONFIG_YAML


@pytest.mark.unit
class TestConfigLoading:
    """Validate config loading and value checking."""

    def test_shipped_config_is_valid(self):
        """The packaged config.yaml must load and validate as-is."""
        cfg = main.load_app_config(CONFIG_YAML)
        # Like every other example, the shipped config ships a placeholder the
        # reader replaces after downloading the pack. A concrete path here would
        # mean someone committed a machine-local model location.
        assert cfg.model_path == "<model-path>"
        assert cfg.labels_path.name == "pcb_label.txt"
        assert cfg.input_dir == Path("assets/datasets/pcb")
        assert cfg.input_size == 640
        assert cfg.max_detections == 300

    def test_defaults_apply_to_missing_sections(self):
        cfg = main.build_app_config({"model": {"path": "models/pack.tar.gz", "labels": "l.txt"}})
        assert cfg.score_threshold == pytest.approx(0.25)
        assert cfg.input_size == 640
        assert cfg.max_detections == 300
        assert cfg.profile is False

    def test_missing_model_path_is_rejected(self):
        raw = valid_config()
        raw["model"]["path"] = ""
        with pytest.raises(ValueError, match="model.path"):
            main.validate_config(main.build_app_config(raw))

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("decode", "score_threshold", 1.5),
            ("decode", "score_threshold", -0.1),
            ("decode", "nms_iou", 1.2),
            ("decode", "max_detections", 0),
            ("model", "input_size", 0),
            ("runtime", "timeout_ms", 0),
            ("runtime", "num_runs", 0),
            ("runtime", "queue_depth", 0),
        ],
    )
    def test_out_of_range_values_are_rejected(self, section, key, value):
        raw = valid_config()
        raw[section][key] = value
        with pytest.raises(ValueError, match=key):
            main.validate_config(main.build_app_config(raw))


@pytest.mark.unit
class TestValidateConfigOnly:
    """Validate the --validate-config-only path shared with the C++ twin."""

    @staticmethod
    def run_validate(config_path: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path),
             "--validate-config-only", *extra],
            capture_output=True, text=True, timeout=20, cwd=str(EXAMPLE_DIR.parents[2]),
        )

    def write_config(self, tmp_path: Path, raw: dict) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return config_path

    def test_shipped_config_validates(self):
        """The packaged config validates without loading the model."""
        r = self.run_validate(CONFIG_YAML)

        assert r.returncode == 0, r.stderr
        assert "classes=6" in r.stdout
        assert "max_detections=300" in r.stdout
        assert "configuration OK" in r.stdout

    def test_cli_overrides_reach_the_resolved_config(self, tmp_path: Path):
        config_path = self.write_config(tmp_path, valid_config())
        r = self.run_validate(config_path, "--score", "0.40", "--nms", "0.55")

        assert r.returncode == 0, r.stderr
        assert "score_threshold=0.40" in r.stdout
        assert "nms_iou=0.55" in r.stdout

    def test_out_of_range_override_exits_with_code_1(self, tmp_path: Path):
        config_path = self.write_config(tmp_path, valid_config())
        r = self.run_validate(config_path, "--score", "1.5")

        assert r.returncode == 1
        assert "decode.score_threshold" in r.stderr

    def test_missing_labels_file_exits_with_code_1(self, tmp_path: Path):
        raw = valid_config()
        raw["model"]["labels"] = "/nonexistent/pcb_label.txt"
        r = self.run_validate(self.write_config(tmp_path, raw))

        assert r.returncode == 1
        assert "labels file does not exist" in r.stderr

    def test_missing_input_dir_exits_with_code_2(self, tmp_path: Path):
        raw = valid_config()
        raw["io"]["input_dir"] = "/nonexistent/pcb-images"
        config_path = self.write_config(tmp_path, raw)
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=20, cwd=str(EXAMPLE_DIR.parents[2]),
        )

        assert r.returncode == 2
        assert "Input directory does not exist" in r.stderr

    def test_empty_input_dir_exits_with_code_3(self, tmp_path: Path):
        empty_input = tmp_path / "images"
        empty_input.mkdir()
        raw = valid_config()
        raw["io"]["input_dir"] = str(empty_input)
        config_path = self.write_config(tmp_path, raw)
        r = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True, text=True, timeout=20, cwd=str(EXAMPLE_DIR.parents[2]),
        )

        assert r.returncode == 3
        assert "No images found" in r.stderr


@pytest.mark.unit
class TestModelAcquisition:
    """Keep the configured model, the documented download, and the test scope in step."""

    @staticmethod
    def scope() -> dict:
        return yaml.safe_load(SCOPE_YAML.read_text(encoding="utf-8"))

    def test_documented_model_is_declared_in_test_scope(self):
        """The package the README tells you to download must be the one e2e downloads."""
        documented = set(
            re.findall(r"([A-Za-z0-9._-]+\.tar\.gz)", README_MD.read_text(encoding="utf-8"))
        )
        declared = {model["file"] for model in self.scope()["models"].values()}

        assert documented, "README documents no model package to download"
        assert documented <= declared, (
            f"{sorted(documented - declared)} not declared in test-scope.yaml"
        )

    def test_scope_models_are_downloadable_artifacts(self):
        for model_id, model in self.scope()["models"].items():
            assert model["source"] == "url", f"{model_id} must be a downloadable artifact"
            assert model["url"].startswith("https://"), f"{model_id} needs an https url"
            assert model["url"].endswith(model["file"]), (
                f"{model_id} url must end with its file name"
            )

    def test_scope_model_urls_are_modelzoo_version_agnostic(self):
        """The Model Zoo version is resolved at download time, never hardcoded."""
        for model_id, model in self.scope()["models"].items():
            assert "{modelzoo_version}" in model["url"], (
                f"{model_id} url must use the {{modelzoo_version}} placeholder"
            )
            assert "SDK2." not in model["url"].replace("SDK{modelzoo_version}", ""), (
                f"{model_id} url must not pin an SDK version"
            )

    @pytest.mark.parametrize("language", ["python", "cpp"])
    def test_e2e_selects_a_declared_model(self, language):
        scope = self.scope()
        e2e = scope["e2e"][language]

        assert e2e["enabled"] is True, f"{language} e2e should be enabled"
        assert e2e["models"], f"{language} e2e selects no model"
        for model_id in e2e["models"]:
            assert model_id in scope["models"], f"{model_id} is not declared"

    def test_documented_download_matches_the_scope_url(self):
        """README must document the same artifact the test scope downloads."""
        readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
        for model in self.scope()["models"].values():
            documented = model["url"].replace("{modelzoo_version}", "${MODELZOO_VERSION}")
            assert documented in readme, f"README does not document {documented}"


@pytest.mark.unit
class TestLabels:
    """Validate label handling for the six PCB defect classes."""

    def test_shipped_labels_match_the_color_palette(self):
        labels = main.load_labels(LABELS_TXT)
        assert labels == [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper",
        ]
        assert len(labels) == len(main.DEFECT_COLORS)

    def test_missing_labels_file_is_rejected(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            main.load_labels(tmp_path / "absent.txt")

    def test_empty_labels_file_is_rejected(self, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("\n \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            main.load_labels(empty)

    def test_class_name_falls_back_for_unknown_ids(self):
        labels = ["missing_hole", "mouse_bite"]
        assert main.class_name(1, labels) == "mouse_bite"
        assert main.class_name(9, labels) == "class_9"
        assert main.class_name(-1, labels) == "class_-1"

    def test_class_color_is_stable_and_in_range(self):
        assert main.class_color(0) == main.DEFECT_COLORS[0]
        assert main.class_color(len(main.DEFECT_COLORS)) == main.DEFECT_COLORS[0]
        assert main.class_color(-3) == main.DEFECT_COLORS[0]


@pytest.mark.unit
class TestImageDiscovery:
    """Validate input discovery and output naming."""

    def test_supported_extensions_only(self, tmp_path: Path):
        for name in ("a.jpg", "b.JPEG", "c.png", "d.bmp", "notes.txt", "model.tar.gz"):
            (tmp_path / name).write_bytes(b"x")
        (tmp_path / "nested").mkdir()

        assert [p.name for p in main.discover_images(tmp_path)] == ["a.jpg", "b.JPEG", "c.png", "d.bmp"]

    def test_discovery_is_sorted(self, tmp_path: Path):
        for name in ("pcb_03.jpg", "pcb_01.jpg", "pcb_02.jpg"):
            (tmp_path / name).write_bytes(b"x")

        assert [p.name for p in main.discover_images(tmp_path)] == [
            "pcb_01.jpg",
            "pcb_02.jpg",
            "pcb_03.jpg",
        ]

    def test_output_path_uses_png_extension(self, tmp_path: Path):
        out = main.output_path_for(Path("images/pcb_01_missing_hole.jpg"), tmp_path)
        assert out == tmp_path / "pcb_01_missing_hole.png"


@pytest.mark.unit
class TestOutputCleanup:
    """Validate stale-output removal between runs."""

    def test_stale_images_are_removed(self, tmp_path: Path):
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        (output_dir / "old_a.png").write_bytes(b"x")
        (output_dir / "old_b.jpg").write_bytes(b"x")
        (output_dir / "report.txt").write_bytes(b"x")

        assert main.clear_output_images(output_dir, input_dir) == 2
        assert [p.name for p in output_dir.iterdir()] == ["report.txt"]

    def test_cleanup_is_skipped_when_output_matches_input(self, tmp_path: Path):
        shared = tmp_path / "images"
        shared.mkdir()
        (shared / "pcb_01.jpg").write_bytes(b"x")

        assert main.clear_output_images(shared, shared) == 0
        assert (shared / "pcb_01.jpg").exists()


@pytest.mark.unit
class TestBboxPayload:
    """Validate BBOX payload decoding into original image coordinates."""

    def test_record_layout_is_24_bytes(self):
        assert main.BBOX_RECORD_SIZE == 24

    def test_parses_records_into_xyxy(self):
        payload = bbox_payload([(10, 20, 30, 40, 0.9, 2)])
        boxes = main.parse_bbox_payload(payload, 640, 480, 0.25)

        assert len(boxes) == 1
        assert boxes[0] == {
            "x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0,
            "score": pytest.approx(0.9), "class_id": 2,
        }

    def test_scores_below_threshold_are_dropped(self):
        payload = bbox_payload([(10, 10, 20, 20, 0.10, 0), (10, 10, 20, 20, 0.80, 1)])
        boxes = main.parse_bbox_payload(payload, 640, 480, 0.25)

        assert [b["class_id"] for b in boxes] == [1]

    def test_boxes_are_clamped_to_the_image(self):
        payload = bbox_payload([(-20, -30, 100, 100, 0.9, 0), (90, 90, 100, 100, 0.9, 1)])
        boxes = main.parse_bbox_payload(payload, 100, 100, 0.25)

        assert boxes[0]["x1"] == 0.0 and boxes[0]["y1"] == 0.0
        assert boxes[1]["x2"] == 100.0 and boxes[1]["y2"] == 100.0

    def test_degenerate_boxes_are_dropped(self):
        payload = bbox_payload([(10, 10, 0, 0, 0.9, 0)])

        assert main.parse_bbox_payload(payload, 640, 480, 0.25) == []

    def test_max_detections_caps_the_parse(self):
        """max_detections bounds the host parse the way the C++ expected_topk does."""
        payload = bbox_payload([(10, 10, 20, 20, 0.9, i) for i in range(5)])

        assert len(main.parse_bbox_payload(payload, 640, 480, 0.25, 2)) == 2
        assert len(main.parse_bbox_payload(payload, 640, 480, 0.25, 0)) == 5

    def test_truncated_payload_uses_available_records(self):
        """A header count larger than the payload must not over-read."""
        payload = bbox_payload([(10, 10, 20, 20, 0.9, 0)], declared=5)

        assert len(main.parse_bbox_payload(payload, 640, 480, 0.25)) == 1

    @pytest.mark.parametrize("payload", [b"", b"\x00", b"\x01\x00\x00"])
    def test_short_payloads_return_no_detections(self, payload):
        assert main.parse_bbox_payload(payload, 640, 480, 0.25) == []

    def test_header_only_payload_returns_no_detections(self):
        assert main.parse_bbox_payload(struct.pack("<I", 0), 640, 480, 0.25) == []


@pytest.mark.unit
class TestLetterbox:
    """Validate the letterbox and its inverse, which must mirror main.cpp."""

    @staticmethod
    def frame(width: int, height: int):
        np = pytest.importorskip("numpy")
        return np.full((height, width, 3), 200, dtype=np.uint8)

    def test_rounding_matches_the_cpp_twin(self):
        """std::round is half-away-from-zero; Python's round() is banker's rounding."""
        assert main.round_half_up(160.5) == 161
        assert main.round_half_up(161.5) == 162
        assert main.round_half_up(160.4999) == 160
        assert main.round_half_up(0.5) == 1

    def test_model_sized_input_is_passed_through_untouched(self):
        """A 640x640 frame must not be resampled, so results stay bit-exact."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        frame = self.frame(640, 640)
        lb = main.letterbox(frame, 640)

        assert lb.scale == 1.0 and (lb.pad_x, lb.pad_y) == (0, 0)
        assert lb.image is frame
        assert np.array_equal(lb.image, frame)

    @pytest.mark.parametrize("width,height", [(1280, 960), (320, 480), (1000, 100), (77, 640)])
    def test_letterbox_fills_the_square_and_preserves_aspect(self, width, height):
        pytest.importorskip("cv2")
        lb = main.letterbox(self.frame(width, height), 640)

        assert lb.image.shape[:2] == (640, 640)
        assert lb.scale == pytest.approx(min(640 / width, 640 / height))
        assert round(width * lb.scale) <= 640 and round(height * lb.scale) <= 640
        assert lb.pad_x >= 0 and lb.pad_y >= 0

    def test_padding_uses_the_yolo_grey(self):
        pytest.importorskip("cv2")
        lb = main.letterbox(self.frame(1280, 320), 640)

        assert lb.pad_y > 0, "a wide frame must be padded vertically"
        assert (lb.image[0, 0] == main.PAD_VALUE).all()

    def test_round_trip_maps_a_box_back_onto_the_source(self):
        """A box drawn around the whole letterboxed content maps to the whole frame."""
        pytest.importorskip("cv2")
        width, height = 1280, 960
        lb = main.letterbox(self.frame(width, height), 640)
        content = {
            "x1": float(lb.pad_x),
            "y1": float(lb.pad_y),
            "x2": float(640 - lb.pad_x),
            "y2": float(640 - lb.pad_y),
            "score": 0.9,
            "class_id": 0,
        }

        mapped = main.to_source_coordinates([content], lb, width, height)

        assert len(mapped) == 1
        assert mapped[0]["x1"] == pytest.approx(0.0, abs=1.0)
        assert mapped[0]["y1"] == pytest.approx(0.0, abs=1.0)
        assert mapped[0]["x2"] == pytest.approx(width, abs=1.0)
        assert mapped[0]["y2"] == pytest.approx(height, abs=1.0)
        assert mapped[0]["score"] == pytest.approx(0.9) and mapped[0]["class_id"] == 0

    def test_identity_letterbox_leaves_boxes_untouched(self):
        pytest.importorskip("cv2")
        lb = main.letterbox(self.frame(640, 640), 640)
        box = {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0, "score": 0.5, "class_id": 2}

        assert main.to_source_coordinates([box], lb, 640, 640) == [box]

    def test_mapped_boxes_are_clamped_to_the_source(self):
        pytest.importorskip("cv2")
        lb = main.letterbox(self.frame(1280, 960), 640)
        box = {"x1": -50.0, "y1": -50.0, "x2": 900.0, "y2": 900.0, "score": 0.5, "class_id": 0}

        mapped = main.to_source_coordinates([box], lb, 1280, 960)

        assert mapped[0]["x1"] == 0.0 and mapped[0]["y1"] == 0.0
        assert mapped[0]["x2"] == 1280.0 and mapped[0]["y2"] == 960.0

    def test_degenerate_mapped_boxes_are_dropped(self):
        pytest.importorskip("cv2")
        lb = main.letterbox(self.frame(1280, 960), 640)
        pad_only = {
            "x1": 0.0, "y1": 0.0, "x2": 5.0, "y2": float(lb.pad_y),
            "score": 0.9, "class_id": 0,
        }

        assert main.to_source_coordinates([pad_only], lb, 1280, 960) == []


@pytest.mark.unit
class TestOverlay:
    """Validate the annotation overlay."""

    def test_overlay_changes_pixels(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        before = frame.copy()
        main.draw_boxes(
            frame,
            [{"x1": 20.0, "y1": 30.0, "x2": 90.0, "y2": 100.0, "score": 0.87, "class_id": 4}],
            main.load_labels(LABELS_TXT),
        )

        assert not np.array_equal(before, frame)

    def test_overlay_skips_degenerate_boxes(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")

        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        before = frame.copy()
        main.draw_boxes(
            frame,
            [{"x1": 50.0, "y1": 50.0, "x2": 50.0, "y2": 50.0, "score": 0.9, "class_id": 0}],
            main.load_labels(LABELS_TXT),
        )

        assert np.array_equal(before, frame)
