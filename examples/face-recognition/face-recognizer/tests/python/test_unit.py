"""Unit tests for the face recognizer."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tests.utils.config_cases import config_writer, load_example_main


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
# The C++ unit test reads the same file, so both implementations agree on the format.
GALLERY_FIXTURE = EXAMPLE_DIR / "tests" / "fixtures" / "gallery.bin"

pytestmark = pytest.mark.unit

main_module = load_example_main(EXAMPLE_DIR, "face_recognizer_main")
main_module.np = np  # main.py binds numpy lazily at runtime

VALID_CONFIG = {
    "model": {"scrfd": "models/scrfd.tar.gz", "arcface": "models/arcface.tar.gz"},
    "gallery": {"path": "gallery.bin"},
    "source": {
        "type": "rtsp",
        "rtsp": {"url": "rtsp://127.0.0.1:8554/src1", "tcp": True, "latency_ms": 100, "fps": 0},
        "usb": {"camera": "camera", "width": 1280, "height": 720, "fps": 30},
        "video_file": {"path": "clip.mp4"},
    },
    "inference": {"frames": 0, "min_score": 0.65, "nms_iou": 0.40, "max_faces": 100},
    "match": {"threshold": 0.55, "margin": 0.12, "unknown_label": "Unknown"},
    "enrollment": {
        "images_dir": "people",
        "video_path": "",
        "name": "",
        "sample_every": 5,
        "min_score": 0.75,
        "max_per_person": 0,
    },
    "output": {"insight": {"host": "127.0.0.1", "video_port": 9000, "metadata_port": 9100}},
}

# Writes VALID_CONFIG with overrides applied as ((section, ..., key), value).
write_config = config_writer(VALID_CONFIG)


def run_main(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        cwd=str(EXAMPLE_DIR),
        timeout=20,
    )


def unit_embedding(axis: int, along_next: float = 0.0):
    embedding = np.zeros(main_module.EMBEDDING_DIM, dtype=np.float32)
    embedding[axis] = 1.0
    embedding[axis + 1] = along_next
    return embedding / np.linalg.norm(embedding)


class TestMainEntrypoint:
    def test_help_lists_the_options(self):
        result = run_main("--help")

        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--enroll" in result.stdout
        assert "--validate-config-only" in result.stdout

    def test_missing_config_file_exits_with_code_2(self):
        result = run_main("--config", "does-not-exist.yaml")

        assert result.returncode == 2
        assert "config file not found" in result.stderr

    @pytest.mark.parametrize(("flags", "mode"), [((), "recognize"), (("--enroll",), "enroll")])
    def test_validate_config_only_reports_the_mode(self, tmp_path, flags, mode):
        config_path = str(write_config(tmp_path))

        result = run_main("--config", config_path, *flags, "--validate-config-only")

        assert result.returncode == 0
        assert f"mode={mode}" in result.stdout

    def test_the_shipped_config_parses(self):
        cfg = main_module.load_app_config(EXAMPLE_DIR / "src" / "common" / "config.yaml")

        assert cfg.scrfd_model == "models/scrfd_2.5g_bnkps.mla_mpk.tar.gz"
        assert cfg.arcface_model == "models/w600k_r50.surgery_mpk.tar.gz"
        assert cfg.source_type == "rtsp"


class TestConfig:
    def test_omitted_values_fall_back_to_documented_defaults(self, tmp_path):
        raw = {
            "model": VALID_CONFIG["model"],
            "gallery": VALID_CONFIG["gallery"],
            "source": {"rtsp": {"url": "rtsp://127.0.0.1:8554/src1"}},
            "output": {"insight": {"host": "127.0.0.1"}},
        }

        cfg = main_module.load_app_config(write_config(tmp_path, root=raw))

        assert cfg.source_type == "rtsp"
        assert cfg.rtsp_tcp is True
        assert cfg.rtsp_latency_ms == 100
        assert cfg.frames == 0
        assert cfg.min_score == pytest.approx(0.65)
        assert cfg.nms_iou == pytest.approx(0.40)
        assert cfg.max_faces == 100
        assert cfg.match_threshold == pytest.approx(0.55)
        assert cfg.match_margin == pytest.approx(0.12)
        assert cfg.unknown_label == "Unknown"
        assert cfg.enroll_sample_every == 5
        assert cfg.enroll_min_score == pytest.approx(0.75)
        assert cfg.video_port == 9000
        assert cfg.metadata_port == 9100

    def test_a_video_file_needs_no_insight_host(self, tmp_path):
        overrides = [(("source", "type"), "video_file"), (("output", "insight", "host"), "")]

        cfg = main_module.load_app_config(write_config(tmp_path, overrides))

        assert cfg.live is False

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("model", "scrfd"), "", "model.scrfd must be set"),
            (("model", "arcface"), "", "model.arcface must be set"),
            (("gallery", "path"), "", "gallery.path must be set"),
            (("source", "type"), "http", "source.type must be rtsp, usb or video_file"),
            (("source", "rtsp", "url"), "", "source.rtsp.url must be set"),
            (("source", "rtsp", "tcp"), "yes", "source.rtsp.tcp must be true or false"),
            (("source", "rtsp", "latency_ms"), -1, "source.rtsp.latency_ms must be >= 0"),
            (("inference", "frames"), -1, "inference.frames must be >= 0"),
            (("inference", "frames"), 1.5, "inference.frames must be an integer"),
            (("inference", "min_score"), 1.0, "inference.min_score must be greater than 0"),
            (("inference", "nms_iou"), 1.5, "inference.nms_iou must be between 0 and 1"),
            (("inference", "max_faces"), 0, "inference.max_faces must be > 0"),
            (("match", "threshold"), 1.5, "match.threshold must be between 0 and 1"),
            (("match", "margin"), -0.1, "match.margin must be between 0 and 1"),
            (("output", "insight", "host"), "", "output.insight.host must be set"),
            (("output", "insight", "video_port"), 0, "output.insight.video_port must be > 0"),
            (("output", "insight", "metadata_port"), 0, "metadata_port must be > 0"),
        ],
    )
    def test_an_invalid_value_is_rejected_with_its_key(self, tmp_path, path, value, message):
        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(write_config(tmp_path, [(path, value)]))

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (("camera",), "", "source.usb.camera must be set"),
            (("fps",), 0, "source.usb.width, height and fps must be > 0"),
            (("width",), 1281, "must be multiples of 8 and at most 2040"),
            (("height",), 2048, "must be multiples of 8 and at most 2040"),
        ],
    )
    def test_an_invalid_usb_camera_mode_is_rejected(self, tmp_path, path, value, message):
        overrides = [(("source", "type"), "usb"), (("source", "usb", *path), value)]

        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(write_config(tmp_path, overrides))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ([(("enrollment", "images_dir"), "")], "images_dir or enrollment.video_path must be"),
            (
                [(("enrollment", "video_path"), "person.mp4")],
                "enrollment.name must be set with enrollment.video_path",
            ),
            ([(("enrollment", "sample_every"), 0)], "enrollment.sample_every must be > 0"),
            ([(("enrollment", "min_score"), 1.5)], "enrollment.min_score must be between 0 and 1"),
            ([(("enrollment", "max_per_person"), -1)], "enrollment.max_per_person must be >= 0"),
        ],
    )
    def test_an_invalid_enrollment_is_rejected(self, tmp_path, overrides, message):
        with pytest.raises(ValueError, match=message):
            main_module.load_app_config(write_config(tmp_path, overrides), enroll=True)

    def test_enrollment_does_not_need_a_source(self, tmp_path):
        overrides = [(("source", "rtsp", "url"), ""), (("output", "insight", "host"), "")]

        main_module.load_app_config(write_config(tmp_path, overrides), enroll=True)

    def test_a_non_mapping_root_is_rejected(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="config root must be a mapping"):
            main_module.load_app_config(config_path)


class TestScrfdDecode:
    @pytest.fixture
    def cfg(self, tmp_path):
        return main_module.load_app_config(write_config(tmp_path))

    @pytest.fixture
    def output(self):
        """SCRFD output for a 640x640 input: scores, boxes and landmarks at strides 8, 16
        and 32, two anchors per cell. Two anchors of one cell fire on the same box, and a
        weaker face fires elsewhere."""
        anchors = [(640 // stride) ** 2 * 2 for stride in (8, 16, 32)]
        tensors = [np.full(count, -20.0, dtype=np.float32) for count in anchors]
        tensors += [np.zeros((count, 4), dtype=np.float32) for count in anchors]
        tensors += [np.zeros((count, 10), dtype=np.float32) for count in anchors]
        # Stride 16, cell (column 10, row 12).
        strong = (12 * 40 + 10) * 2
        tensors[1][strong : strong + 2] = (3.0, 2.0)
        tensors[4][strong : strong + 2] = (1.0, 2.0, 3.0, 4.0)
        tensors[7][strong : strong + 2, 0::2] = 0.5 * np.arange(5)
        tensors[7][strong : strong + 2, 1::2] = 1.0
        # Stride 32, cell (column 2, row 15).
        weak = (15 * 20 + 2) * 2
        tensors[2][weak] = 1.0
        tensors[5][weak] = 1.0
        return tensors

    def test_faces_are_in_source_pixels_strongest_first(self, cfg, output):
        # A 1280x720 frame is halved and padded by 140 rows above and below.
        letterbox = main_module.letterbox_for(1280, 720)

        faces = main_module.decode_scrfd(output, cfg, letterbox)

        assert letterbox == (0.5, 0, 140)
        assert len(faces) == 2
        assert faces[0].score == pytest.approx(0.9526, abs=1e-3)
        assert faces[0].box == pytest.approx([288.0, 40.0, 416.0, 232.0])
        assert faces[0].landmarks[0] == pytest.approx([320.0, 136.0])
        assert faces[0].landmarks[4] == pytest.approx([384.0, 136.0])
        assert faces[1].box[:2] == pytest.approx([64.0, 616.0])

    def test_max_faces_limits_the_result(self, tmp_path, output):
        cfg = main_module.load_app_config(
            write_config(tmp_path, [(("inference", "max_faces"), 1)])
        )

        assert len(main_module.decode_scrfd(output, cfg, (0.5, 0, 140))) == 1

    def test_min_score_drops_the_weaker_face(self, tmp_path, output):
        cfg = main_module.load_app_config(
            write_config(tmp_path, [(("inference", "min_score"), 0.8)])
        )

        assert len(main_module.decode_scrfd(output, cfg, (0.5, 0, 140))) == 1

    def test_an_unexpected_tensor_count_is_rejected(self, cfg, output):
        with pytest.raises(RuntimeError, match="expected 9 SCRFD output tensors"):
            main_module.decode_scrfd(output[:6], cfg, (0.5, 0, 140))


class TestAlignment:
    def test_the_transform_moves_the_landmarks_onto_the_template(self):
        template = np.array(main_module.ARCFACE_TEMPLATE)

        # The template doubled in size and shifted maps back by halving and shifting.
        transform = main_module.similarity_transform(template * 2.0 + (50.0, 30.0))

        assert transform == pytest.approx(np.array([[0.5, 0.0, -25.0], [0.0, 0.5, -15.0]]))


class TestMatching:
    @pytest.fixture
    def cfg(self, tmp_path):
        return main_module.load_app_config(write_config(tmp_path))

    @pytest.fixture
    def people(self):
        return [
            main_module.Person("PersonA", unit_embedding(0)),
            main_module.Person("PersonB", unit_embedding(1)),
        ]

    def test_a_close_embedding_is_named(self, cfg, people):
        name, similarity = main_module.match_embedding(unit_embedding(0, 0.2), people, cfg)

        assert name == "PersonA"
        assert similarity == pytest.approx(0.9806, abs=1e-3)

    def test_a_similarity_below_the_threshold_is_unknown(self, cfg, people):
        assert main_module.match_embedding(unit_embedding(2), people, cfg)[0] == "Unknown"

    def test_a_lead_smaller_than_the_margin_is_unknown(self, cfg, people):
        name, similarity = main_module.match_embedding(unit_embedding(0, 0.9), people, cfg)

        assert name == "Unknown"
        assert similarity > cfg.match_threshold

    def test_an_empty_gallery_names_nobody(self, cfg):
        assert main_module.match_embedding(unit_embedding(0), [], cfg) == ("Unknown", 0.0)


class TestGallery:
    def test_the_shared_fixture_loads(self):
        people = main_module.load_gallery(str(GALLERY_FIXTURE))

        assert [(person.name, person.samples) for person in people] == [
            ("PersonA", 3),
            ("PersonB", 1),
        ]
        assert people[0].mean[256] == pytest.approx(0.5)
        assert people[1].mean[0] == pytest.approx(1.0)

    def test_a_saved_gallery_matches_the_shared_fixture(self, tmp_path):
        path = tmp_path / "gallery.bin"

        main_module.save_gallery(main_module.load_gallery(str(GALLERY_FIXTURE)), str(path))

        assert path.read_bytes() == GALLERY_FIXTURE.read_bytes()

    def test_a_second_sample_is_averaged_into_the_same_person(self):
        people = []

        main_module.add_to_gallery(people, "PersonA", unit_embedding(0))
        main_module.add_to_gallery(people, "PersonA", unit_embedding(1))

        assert len(people) == 1
        assert people[0].samples == 2
        assert people[0].mean[:2] == pytest.approx([0.5, 0.5])

    @pytest.mark.parametrize(
        ("content", "message"),
        [(b"not a gallery", "not a face-recognizer gallery"), (None, "gallery is truncated")],
    )
    def test_a_damaged_file_is_rejected(self, tmp_path, content, message):
        path = tmp_path / "gallery.bin"
        path.write_bytes(content if content is not None else GALLERY_FIXTURE.read_bytes()[:-8])

        with pytest.raises(RuntimeError, match=message):
            main_module.load_gallery(str(path))
