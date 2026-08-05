"""Unit tests for ssd-mobilenet-object-detector (Python)."""

import importlib.util
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"

MAIN_SPEC = importlib.util.spec_from_file_location("ssd_mobilenet_main", MAIN_PY)
assert MAIN_SPEC is not None and MAIN_SPEC.loader is not None
ssd_mobilenet_main = importlib.util.module_from_spec(MAIN_SPEC)
MAIN_SPEC.loader.exec_module(ssd_mobilenet_main)


def _run(args, cwd=EXAMPLE_DIR):
    return subprocess.run(
        [sys.executable, str(MAIN_PY), *args],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=str(cwd),
    )


def _write_config(tmp_path: Path, section: str, key: str, value: str) -> Path:
    config = tmp_path / f"{section}_{key}.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: /nonexistent/ssd_model.tar.gz
            io:
              input_dir: assets/datasets/coco
              output_dir: sandbox/ssd-mobilenet-object-detector
            {section}:
              {key}: {value}
            """
        ),
        encoding="utf-8",
    )
    return config


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (True, False, True),
        (False, True, False),
        ("true", False, True),
        ("FALSE", True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_config_bool_or_matches_scalar_config(value, default, expected):
    assert (
        ssd_mobilenet_main.config_bool_or(
            {"overlay": value}, "overlay", default, "output.overlay"
        )
        is expected
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [0, 1, [], {}, "yes", "0", ""])
def test_config_bool_or_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="output.overlay must be true or false"):
        ssd_mobilenet_main.config_bool_or(
            {"overlay": value}, "overlay", True, "output.overlay"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("section", "key", "default", "expected"),
    [
        ({}, "num_runs", 1, 1),
        ({"num_runs": None}, "num_runs", 1, 1),
        ({"num_runs": 7}, "num_runs", 1, 7),
        ({"score_threshold": None}, "score_threshold", 0.55, 0.55),
    ],
)
def test_config_value_or_treats_null_as_omitted(section, key, default, expected):
    assert ssd_mobilenet_main.config_value_or(section, key, default) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("profile", "mean", "stddev"),
    [
        ("tensorflow_ssd", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        (
            "torchvision_ssdlite",
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ],
)
def test_normalization_profiles_are_explicit(profile, mean, stddev):
    assert ssd_mobilenet_main.normalization_for_profile(profile) == (mean, stddev)


@pytest.mark.unit
def test_normalization_profile_rejects_unknown_value():
    with pytest.raises(ValueError, match="tensorflow_ssd or torchvision_ssdlite"):
        ssd_mobilenet_main.normalization_for_profile("guess-from-filename")


@pytest.mark.unit
def test_clear_output_images_unlinks_dangling_symlink(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    missing_target = tmp_path / "outside" / "missing.png"
    output = output_dir / "frame_jpg.png"
    output.symlink_to(missing_target)

    removed = ssd_mobilenet_main.clear_output_images(output_dir, {output.name})

    assert removed == 1
    assert not output.is_symlink()
    assert not missing_target.exists()


@pytest.mark.unit
def test_aggregate_suppression_is_opt_in():
    detections = [
        {"box": [0.0, 0.0, 600.0, 600.0], "score": 0.9, "class_id": 3},
        {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.8, "class_id": 3},
        {"box": [60.0, 60.0, 100.0, 100.0], "score": 0.8, "class_id": 3},
    ]

    assert (
        ssd_mobilenet_main.suppress_aggregate_detections(
            detections, 640, 640, ssd_mobilenet_main.AggregateSuppressionOptions()
        )
        == detections
    )


@pytest.mark.unit
def test_detection_report_preserves_hidden_raw_box():
    raw = [
        {"box": [0.0, 0.0, 600.0, 600.0], "score": 0.9, "class_id": 3},
        {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.8, "class_id": 3},
        {"box": [60.0, 60.0, 100.0, 100.0], "score": 0.8, "class_id": 3},
    ]
    displayed = ssd_mobilenet_main.suppress_aggregate_detections(
        raw,
        640,
        640,
        ssd_mobilenet_main.AggregateSuppressionOptions(enabled=True),
    )

    record = ssd_mobilenet_main.detections_record(
        Path("frame.jpg"),
        SimpleNamespace(shape=(640, 640, 3)),
        raw,
        displayed,
        ["background", "N/A", "N/A", "car"],
    )

    assert len(record["detections"]) == len(raw)
    assert [det["displayed"] for det in record["detections"]] == [False, True, True]


@pytest.mark.unit
def test_suppresses_same_class_crowd_region_only():
    detections = [
        {"box": [43.0, 180.0, 617.0, 467.0], "score": 0.64, "class_id": 3},
        {"box": [270.0, 330.0, 324.0, 374.0], "score": 0.76, "class_id": 3},
        {"box": [306.0, 356.0, 368.0, 409.0], "score": 0.61, "class_id": 3},
        {"box": [420.0, 373.0, 489.0, 440.0], "score": 0.64, "class_id": 3},
        {"box": [20.0, 80.0, 620.0, 500.0], "score": 0.90, "class_id": 6},
    ]

    filtered = ssd_mobilenet_main.suppress_aggregate_detections(
        detections,
        640,
        640,
        ssd_mobilenet_main.AggregateSuppressionOptions(enabled=True),
    )

    assert detections[0] not in filtered
    assert filtered == detections[1:]


@pytest.mark.unit
def test_preserves_large_object_without_multiple_same_class_children():
    detections = [
        {"box": [20.0, 20.0, 620.0, 620.0], "score": 0.95, "class_id": 3},
        {"box": [100.0, 100.0, 180.0, 180.0], "score": 0.80, "class_id": 3},
        {"box": [300.0, 300.0, 380.0, 380.0], "score": 0.80, "class_id": 6},
    ]

    filtered = ssd_mobilenet_main.suppress_aggregate_detections(
        detections,
        640,
        640,
        ssd_mobilenet_main.AggregateSuppressionOptions(enabled=True),
    )

    assert filtered == detections


@pytest.mark.unit
def test_aggregate_suppression_stays_below_one_millisecond():
    # Worst-case max_detections=100 scan: all boxes are candidate parents, but none is a
    # materially smaller child. The timed loop amortizes timer noise.
    detections = [
        {
            "box": [float(i % 5), float(i % 5), 500.0 + i % 5, 500.0 + i % 5],
            "score": 0.80,
            "class_id": 3,
        }
        for i in range(100)
    ]
    options = ssd_mobilenet_main.AggregateSuppressionOptions(enabled=True)
    runs = 500

    start = time.perf_counter()
    for _ in range(runs):
        filtered = ssd_mobilenet_main.suppress_aggregate_detections(
            detections, 640, 640, options
        )
    mean_ms = (time.perf_counter() - start) * 1000.0 / runs

    assert filtered == detections
    assert mean_ms < 1.0, f"aggregate suppression mean={mean_ms:.3f} ms"


@pytest.mark.unit
class TestArgParsing:
    """Validate CLI argument parsing for the SSD example."""

    def test_help(self):
        r = _run(["--help"])
        assert r.returncode == 0
        assert "--config" in r.stdout

    def test_bad_config_path(self):
        r = _run(["--config", "/nonexistent/ssd-config.yaml"])
        assert r.returncode == 2
        assert "failed to open config file" in r.stderr

    def test_unknown_flag(self):
        r = _run(["--bogus"])
        assert r.returncode == 2
        assert "unrecognized" in r.stderr.lower() or "error" in r.stderr.lower()


@pytest.mark.unit
class TestDecodeConfigValidation:
    """The decode section drives the model-managed BoxDecode stage, so it must be validated."""

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("score_threshold", "1.5", "decode.score_threshold must be in [0.0, 1.0]"),
            ("score_threshold", ".nan", "decode.score_threshold must be in [0.0, 1.0]"),
            ("nms_iou", "-0.1", "decode.nms_iou must be in [0.0, 1.0]"),
            ("nms_iou", ".nan", "decode.nms_iou must be in [0.0, 1.0]"),
            ("max_detections", "0", "decode.max_detections must be >= 1"),
        ],
    )
    def test_rejects_out_of_range_values(self, tmp_path, key, value, message):
        config = _write_config(tmp_path, "decode", key, value)
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert message in r.stderr


@pytest.mark.unit
class TestAggregateDisplayConfigValidation:
    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            (
                "aggregate_min_parent_area_fraction",
                "1.1",
                "output.aggregate_min_parent_area_fraction must be in [0.0, 1.0]",
            ),
            (
                "aggregate_min_child_containment",
                "0.0",
                "output.aggregate_min_child_containment must be in (0.0, 1.0]",
            ),
            (
                "aggregate_max_child_area_ratio",
                ".nan",
                "output.aggregate_max_child_area_ratio must be in (0.0, 1.0]",
            ),
            (
                "aggregate_min_children",
                "1",
                "output.aggregate_min_children must be >= 2",
            ),
        ],
    )
    def test_rejects_out_of_range_values(self, tmp_path, key, value, message):
        config = _write_config(tmp_path, "output", key, value)
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert message in r.stderr


@pytest.mark.unit
class TestRuntimeConfigValidation:
    """Runtime limits must be rejected as config errors, not surface as load/profile failures."""

    @pytest.mark.parametrize("key", ["timeout_ms", "num_runs"])
    def test_null_values_use_documented_defaults(self, tmp_path, key):
        config = _write_config(tmp_path, "runtime", key, "null")
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert "Model file does not exist" in r.stderr
        assert "Traceback" not in r.stderr

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("timeout_ms", "0", "runtime.timeout_ms must be > 0"),
            ("num_runs", "0", "runtime.num_runs must be >= 1"),
        ],
    )
    def test_rejects_out_of_range_values(self, tmp_path, key, value, message):
        config = _write_config(tmp_path, "runtime", key, value)
        r = _run(["--config", str(config)])
        assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        assert message in r.stderr


@pytest.mark.unit
def test_rejects_detection_report_that_aliases_an_input_image(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.touch()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "frame.jpg"
    image.touch()
    config = tmp_path / "alias.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
            io:
              input_dir: {input_dir}
              output_dir: {tmp_path / 'output'}
              detections_json: {image}
            output:
              overlay: false
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "io.detections_json must not overwrite an input image" in r.stderr


@pytest.mark.unit
def test_rejects_new_image_named_detection_report_inside_input_dir(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.touch()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "frame.jpg").touch()
    report = input_dir / "detections.png"
    config = tmp_path / "image_report.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
            io:
              input_dir: {input_dir}
              output_dir: {tmp_path / 'output'}
              detections_json: {report}
            output:
              overlay: false
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])

    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert (
        "io.detections_json must not use an image filename inside io.input_dir"
        in r.stderr
    )
    assert not report.exists()


@pytest.mark.unit
def test_rejects_detection_report_hard_linked_to_input_image(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.touch()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "frame.jpg"
    image.touch()
    report = tmp_path / "detections.json"
    os.link(image, report)
    config = tmp_path / "hardlink_alias.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
            io:
              input_dir: {input_dir}
              output_dir: {tmp_path / 'output'}
              detections_json: {report}
            output:
              overlay: false
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "io.detections_json must not overwrite an input image" in r.stderr


@pytest.mark.unit
def test_rejects_overlay_that_aliases_a_consumed_input(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model = output_dir / "frame_jpg.png"
    model.touch()
    labels = tmp_path / "labels.txt"
    labels.write_text("background\nperson\n", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "frame.jpg").touch()
    config = tmp_path / "overlay_alias.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
              labels: {labels}
            io:
              input_dir: {input_dir}
              output_dir: {output_dir}
            output:
              overlay: true
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "generated overlay must not overwrite a consumed input" in r.stderr


@pytest.mark.unit
def test_rejects_overlay_hard_linked_to_consumed_input(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model = tmp_path / "model.tar.gz"
    model.touch()
    labels = tmp_path / "labels.txt"
    labels.write_text("background\nperson\n", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "frame.jpg").touch()
    os.link(model, output_dir / "frame_jpg.png")
    config = tmp_path / "overlay_hardlink_alias.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
              labels: {labels}
            io:
              input_dir: {input_dir}
              output_dir: {output_dir}
            output:
              overlay: true
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "generated overlay must not overwrite a consumed input" in r.stderr


@pytest.mark.unit
def test_profile_ignores_unused_output_collisions(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.touch()
    labels = tmp_path / "labels.txt"
    labels.write_text("background\nperson\n", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "frame.jpg"
    image.touch()
    config = tmp_path / "profile.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
              labels: {labels}
            io:
              input_dir: {input_dir}
              output_dir: {input_dir}
              detections_json: {image}
            runtime:
              profile: true
            output:
              overlay: true
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert "io.output_dir must differ from io.input_dir" not in r.stderr
    assert "io.detections_json must not overwrite" not in r.stderr
    assert "generated overlay must not overwrite" not in r.stderr


@pytest.mark.unit
@pytest.mark.parametrize("collision", ["config", "model", "labels", "overlay"])
def test_rejects_detection_report_that_aliases_run_files(tmp_path, collision):
    model = tmp_path / "model.tar.gz"
    model.touch()
    labels = tmp_path / "labels.txt"
    labels.write_text("background\nperson\n", encoding="utf-8")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image = input_dir / "frame.jpg"
    image.touch()
    config = tmp_path / "alias.yaml"
    collision_paths = {
        "config": config,
        "model": model,
        "labels": labels,
        "overlay": tmp_path / "output" / "frame_jpg.png",
    }
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
              labels: {labels}
            io:
              input_dir: {input_dir}
              output_dir: {tmp_path / 'output'}
              detections_json: {collision_paths[collision]}
            output:
              overlay: true
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    expected = "generated overlay" if collision == "overlay" else "consumed input"
    assert f"io.detections_json must not overwrite a {expected}" in r.stderr


@pytest.mark.unit
def test_missing_custom_coco_labels_path_does_not_use_packaged_fallback(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.touch()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    missing_labels = tmp_path / "custom" / "coco_labels.txt"
    config = tmp_path / "missing_labels.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              path: {model}
              labels: {missing_labels}
            io:
              input_dir: {input_dir}
              output_dir: {tmp_path / 'output'}
            """
        ),
        encoding="utf-8",
    )

    r = _run(["--config", str(config)])
    assert r.returncode == 2, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert f"labels file does not exist: {missing_labels}" in r.stderr
