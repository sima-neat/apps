"""Unit tests for ssd-mobilenet-object-detector (Python)."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


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
class TestRuntimeConfigValidation:
    """Runtime limits must be rejected as config errors, not surface as load/profile failures."""

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
