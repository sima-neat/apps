"""E2E tests for image-classification-explorer (Python)."""

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
COMMON_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"


def _validation_expectations() -> tuple[int, float]:
    """Expected top-1 class id and minimum probability for the bundled goldfish
    fixture, read from the shipped config's `validation` block."""
    validation = yaml.safe_load(COMMON_CONFIG.read_text())["validation"]
    return int(validation["expected_class_id"]), float(validation["min_probability"])


def _assert_goldfish(image_entry: dict, model_names) -> None:
    """A nonempty top-k is not enough: a regression in normalization, output
    interpretation or label mapping would still produce one. Pin the known
    class for the known image."""
    expected_id, min_probability = _validation_expectations()
    for model_name in model_names:
        top1 = image_entry["predictions"][model_name]["top_k"][0]
        assert top1["class_id"] == expected_id, (
            f"{model_name}: expected class {expected_id}, got {top1['class_id']} "
            f"({top1['label']})"
        )
        assert top1["probability"] >= min_probability, (
            f"{model_name}: goldfish probability {top1['probability']:.4f} below "
            f"{min_probability}"
        )


MODEL_NAMES = ("resnet_50", "resnet_18", "efficientnet_b0", "densenet_121")


def _cpp_binary(apps_root: Path) -> str:
    """The built C++ entrypoint, or "" when it is not available."""
    configured = os.environ.get("SIMANEAT_APPS_TEST_CPP_BINARY", "")
    if configured:
        return configured if Path(configured).is_file() else ""
    candidate = (
        apps_root
        / "build/examples/classification/image-classification-explorer"
        / "image-classification-explorer"
    )
    return str(candidate) if candidate.is_file() else ""


def _drop_timing_column(csv_text: str) -> list[list[str]]:
    """CSV rows without the measured inference_ms column."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    return [row[:6] + row[7:] for row in rows]


def _strip_timings(html: str) -> str:
    return re.sub(r"[0-9]+\.[0-9]+ ms", "<ms>", html)


def _normalised_report(output_dir: Path) -> dict:
    payload = json.loads((output_dir / "report.json").read_text())
    payload["timing"]["total_ms"] = "<ms>"
    for image in payload["images"]:
        image["path"] = Path(image["path"]).name
        for prediction in image.get("predictions", {}).values():
            prediction["inference_ms"] = "<ms>"
    payload["skipped"] = [Path(entry.split(":")[0]).name for entry in payload["skipped"]]
    return payload



def _model_paths(models_dir: Path) -> dict[str, Path]:
    return {name: models_dir / f"{name}_mpk.tar.gz" for name in MODEL_NAMES}


@pytest.mark.e2e
class TestE2E:
    def test_single_image_two_models(
        self,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        tmp_output_dir,
    ):
        default_image = apps_root / "assets" / "datasets-test" / "imagenet" / "goldfish.jpeg"
        image_env = Path(
            os.environ.get("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE", str(default_image))
        )
        skip_unless_e2e_ready(
            image_env.exists(),
            "classification image missing; set SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
        )
        paths = _model_paths(models_dir)
        skip_unless_e2e_ready(
            all(p.is_file() for p in paths.values()),
            f"{', '.join(MODEL_NAMES)} model packages not found under {models_dir}",
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = e2e_config_writer(
            {
                "io": {"input": str(image_env), "output_dir": str(output_dir)},
                "models": {name: {"path": str(paths[name])} for name in MODEL_NAMES},
            }
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        report_json = output_dir / "report.json"
        assert report_json.is_file()
        assert (output_dir / "report.csv").is_file()
        assert (output_dir / "report.html").is_file()

        payload = json.loads(report_json.read_text())
        assert set(payload["models"]) == set(MODEL_NAMES)
        assert len(payload["images"]) == 1
        image_entry = payload["images"][0]
        assert "errors" not in image_entry, f"image had errors: {image_entry.get('errors')}"
        for model_name in MODEL_NAMES:
            assert image_entry["predictions"][model_name]["top_k"], (
                f"{model_name} produced no predictions"
            )
        if image_env == default_image:
            _assert_goldfish(image_entry, MODEL_NAMES)

    def test_directory_input(
        self,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        tmp_output_dir,
        test_images_dir,
    ):
        skip_unless_e2e_ready(
            test_images_dir.is_dir() and any(test_images_dir.glob("*.jpg")),
            f"no test images found under {test_images_dir}",
        )
        paths = _model_paths(models_dir)
        skip_unless_e2e_ready(
            all(p.is_file() for p in paths.values()),
            f"{', '.join(MODEL_NAMES)} model packages not found under {models_dir}",
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = e2e_config_writer(
            {
                "io": {"input": str(test_images_dir), "output_dir": str(output_dir)},
                "models": {name: {"path": str(paths[name])} for name in MODEL_NAMES},
            }
        )

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        payload = json.loads((output_dir / "report.json").read_text())
        expected_count = len(list(test_images_dir.glob("*.jpg")))
        assert len(payload["images"]) == expected_count
        assert payload["timing"]["image_count"] == expected_count
        for image_entry in payload["images"]:
            assert "errors" not in image_entry, (
                f"{image_entry['path']} had errors: {image_entry.get('errors')}"
            )
            for model_name in MODEL_NAMES:
                assert image_entry["predictions"][model_name]["top_k"], (
                    f"{image_entry['path']}: {model_name} produced no predictions"
                )

    def test_cpp_and_python_reports_are_identical(
        self,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        e2e_config_writer,
        tmp_output_dir,
        test_images_dir,
    ):
        """The two entrypoints must produce the same report for the same input.

        Nearly every review finding on this application was a divergence
        between them - operator handling, exit codes, CSV line endings, drifted
        copies of the report JavaScript. Comparing the artifacts directly is
        what makes that class visible instead of being found one case at a
        time."""
        binary = _cpp_binary(apps_root)
        skip_unless_e2e_ready(
            bool(binary), "C++ binary not built; set SIMANEAT_APPS_TEST_CPP_BINARY"
        )
        skip_unless_e2e_ready(
            test_images_dir.is_dir() and any(test_images_dir.glob("*.jpg")),
            f"no test images found under {test_images_dir}",
        )
        paths = _model_paths(models_dir)
        skip_unless_e2e_ready(
            all(p.is_file() for p in paths.values()),
            f"{', '.join(MODEL_NAMES)} model packages not found under {models_dir}",
        )

        models = {name: {"path": str(paths[name])} for name in MODEL_NAMES}
        py_out = tmp_output_dir.parent / "report-python"
        cpp_out = tmp_output_dir.parent / "report-cpp"

        # e2e_config_writer always writes the same path, so write and run each
        # configuration in turn rather than holding two at once.
        for output_dir, command in (
            (py_out, lambda cfg: [sys.executable, str(MAIN_PY), "--config", str(cfg)]),
            (cpp_out, lambda cfg: [binary, "--config", str(cfg)]),
        ):
            config = e2e_config_writer(
                {"io": {"input": str(test_images_dir), "output_dir": str(output_dir)},
                 "models": models}
            )
            argv = command(config)
            result = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=test_timeout_ms / 1000, cwd=str(EXAMPLE_DIR),
            )
            assert result.returncode == 0, (
                f"{argv[0]} exited {result.returncode}\n{result.stdout}\n{result.stderr}"
            )

        # report.csv must match byte for byte, including line endings.
        py_csv = (py_out / "report.csv").read_text()
        cpp_csv = (cpp_out / "report.csv").read_text()
        assert _drop_timing_column(py_csv) == _drop_timing_column(cpp_csv), (
            "report.csv differs between the implementations"
        )

        # report.html differs only in the per-image timings it displays.
        py_html = _strip_timings((py_out / "report.html").read_text())
        cpp_html = _strip_timings((cpp_out / "report.html").read_text())
        assert py_html == cpp_html, "report.html differs between the implementations"

        # report.json must match once measured times are removed.
        assert _normalised_report(py_out) == _normalised_report(cpp_out), (
            "report.json differs between the implementations"
        )

    # Every configuration below is rejected by both entrypoints. The successful
    # run is compared by test_cpp_and_python_reports_are_identical; this covers
    # the other half, where the two used to drift unnoticed: several findings on
    # this PR were error paths where one language exited 2 and the other 6.
    FAILURE_CASES = [
        ("missing config file", None, 2),
        ("top-level scalar", "just-a-string\n", 2),
        ("io section is a scalar", "io: /images\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("runtime section is a scalar",
         "runtime: 5000\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("no models", "io:\n  input: null\n", 2),
        ("model without a path", "models:\n  m: {}\n", 2),
        ("unsupported preprocess",
         "models:\n  m:\n    path: m.tar.gz\n    preprocess: bespoke\n", 2),
        ("unsupported output",
         "models:\n  m:\n    path: m.tar.gz\n    output: raw_logits\n", 2),
        ("non-integral top_k",
         "models:\n  m:\n    path: m.tar.gz\n    top_k: 1.9\n", 2),
        ("non-positive top_k",
         "models:\n  m:\n    path: m.tar.gz\n    top_k: 0\n", 2),
        ("non-positive timeout",
         "runtime:\n  timeout_ms: 0\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("unquoted non-string profile name",
         "models:\n  true:\n    path: m.tar.gz\n", 2),
        ("profile name with a dot",
         "models:\n  resnet.v2:\n    path: m.tar.gz\n", 2),
        ("malformed validation block",
         "validation:\n  expected_class_id: abc\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("missing label map",
         "models:\n  m:\n    path: m.tar.gz\n    label_map: /nonexistent/labels.txt\n", 2),
        ("input path does not exist",
         "io:\n  input: /nonexistent/directory\nmodels:\n  m:\n    path: m.tar.gz\n", 3),
        ("non-positive num_classes",
         "models:\n  m:\n    path: m.tar.gz\n    num_classes: 0\n", 2),
        ("non-positive input_width",
         "models:\n  m:\n    path: m.tar.gz\n    input_width: 0\n", 2),
        ("integer beyond int32",
         "models:\n  m:\n    path: m.tar.gz\n    top_k: 2147483648\n", 2),
        ("profile name with a colon",
         'models:\n  "resnet:50":\n    path: m.tar.gz\n', 2),
        ("profile name with a space",
         'models:\n  "res net":\n    path: m.tar.gz\n', 2),
        ("non-numeric min_probability",
         "validation:\n  min_probability: high\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("label map that is too short",
         "models:\n  m:\n    path: m.tar.gz\n    num_classes: 5\n"
         "    label_map: TOO_SHORT_LABELS\n", 2),
        ("label map with a blank line",
         "models:\n  m:\n    path: m.tar.gz\n    num_classes: 3\n"
         "    label_map: BLANK_LINE_LABELS\n", 2),
        # Runtime rather than configuration: the model archive does not exist,
        # so both must fail the same way once the configuration is accepted.
        ("model archive missing",
         "models:\n  m:\n    path: /nonexistent/model.tar.gz\n", 6),

        # Falsy scalars. `x or {}` reads these as "absent" and quietly applies
        # defaults, so testing only a truthy scalar missed them entirely.
        ("io section is false", "io: false\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("io section is zero", "io: 0\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("runtime section is false",
         "runtime: false\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("validation section is false",
         "validation: false\nmodels:\n  m:\n    path: m.tar.gz\n", 2),
        ("models section is false", "models: false\n", 2),
        ("models section is empty", "models: {}\n", 2),
        ("top-level is false", "false\n", 2),

        # Combinations: a configuration error must win over an input error
        # regardless of which is discovered first, or the two entrypoints
        # disagree on which code to report.
        ("bad preprocess and missing input",
         "io:\n  input: /nonexistent/directory\n"
         "models:\n  m:\n    path: m.tar.gz\n    preprocess: bespoke\n", 2),
        ("bad top_k and missing input",
         "io:\n  input: /nonexistent/directory\n"
         "models:\n  m:\n    path: m.tar.gz\n    top_k: 0\n", 2),
        ("bad label map and missing input",
         "io:\n  input: /nonexistent/directory\n"
         "models:\n  m:\n    path: m.tar.gz\n    label_map: /nonexistent/labels.txt\n", 2),

        # `foo:` and `"foo":` are one YAML key and the later definition wins.
        # The invalid top_k is in the FIRST definition, so a run that honours
        # "last wins" never sees it and fails later at the missing archive (6),
        # while one that keeps both entries rejects the configuration (2). The
        # bad value has to be in the discarded definition for this to detect
        # anything - with it in the second, both behaviours exit 2 and the case
        # proves nothing.
        ("quoted and unquoted duplicate key",
         'models:\n  foo:\n    path: /nonexistent/m.tar.gz\n    top_k: 0\n'
         '  "foo":\n    path: /nonexistent/m.tar.gz\n',
         6),
    ]

    def test_cpp_and_python_fail_identically(
        self, apps_root, tmp_output_dir, skip_unless_e2e_ready, test_timeout_ms
    ):
        """Both entrypoints must reject the same configurations the same way."""
        binary = _cpp_binary(apps_root)
        skip_unless_e2e_ready(
            bool(binary), "C++ binary not built; set SIMANEAT_APPS_TEST_CPP_BINARY"
        )

        config_path = tmp_output_dir.parent / "failure-config.yaml"
        too_short = tmp_output_dir.parent / "too-short-labels.txt"
        too_short.write_text("only\none\n")
        blank_line = tmp_output_dir.parent / "blank-line-labels.txt"
        blank_line.write_text("cat\n\nbird\n")

        mismatches = []
        for label, body, expected in self.FAILURE_CASES:
            if body is None:
                argument = str(tmp_output_dir.parent / "does-not-exist.yaml")
            else:
                body = (body.replace("TOO_SHORT_LABELS", str(too_short))
                            .replace("BLANK_LINE_LABELS", str(blank_line)))
                config_path.write_text(body)
                argument = str(config_path)

            outcomes = {}
            for language, command in (
                ("python", [sys.executable, str(MAIN_PY), "--config", argument]),
                ("cpp", [binary, "--config", argument]),
            ):
                result = subprocess.run(
                    command, capture_output=True, text=True,
                    timeout=test_timeout_ms / 1000, cwd=str(EXAMPLE_DIR),
                )
                outcomes[language] = (result.returncode, result.stderr.strip())

            py_code, py_err = outcomes["python"]
            cpp_code, cpp_err = outcomes["cpp"]
            if py_code != cpp_code or py_code != expected:
                mismatches.append(
                    f"{label}: expected {expected}, python={py_code}, cpp={cpp_code}"
                )
            for language, (_, err) in outcomes.items():
                if "Traceback" in err or "terminate called" in err:
                    mismatches.append(f"{label}: {language} crashed instead of reporting: {err[:200]}")

        assert not mismatches, "exit codes differ between the entrypoints:\n  " + "\n  ".join(
            mismatches
        )

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_each_model_individually(
        self,
        model_name,
        apps_root,
        models_dir,
        test_timeout_ms,
        skip_unless_e2e_ready,
        tmp_output_dir,
    ):
        """Each supported model must work on its own, not just alongside the others."""
        default_image = apps_root / "assets" / "datasets-test" / "imagenet" / "goldfish.jpeg"
        image_env = Path(
            os.environ.get("SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE", str(default_image))
        )
        skip_unless_e2e_ready(
            image_env.exists(),
            "classification image missing; set SIMANEAT_APPS_TEST_CLASSIFICATION_IMAGE",
        )
        model_path = _model_paths(models_dir)[model_name]
        skip_unless_e2e_ready(
            model_path.is_file(), f"{model_name} model package not found under {models_dir}"
        )

        output_dir = tmp_output_dir.parent / "report"
        config_path = tmp_output_dir.parent / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "io": {"input": str(image_env), "output_dir": str(output_dir)},
            "models": {model_name: {"path": str(model_path)}},
        }))

        result = subprocess.run(
            [sys.executable, str(MAIN_PY), "--config", str(config_path)],
            capture_output=True,
            text=True,
            timeout=test_timeout_ms / 1000,
            cwd=str(EXAMPLE_DIR),
        )

        assert result.returncode == 0, (
            f"main.py exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        payload = json.loads((output_dir / "report.json").read_text())
        assert payload["models"] == [model_name]
        assert payload["images"][0]["predictions"][model_name]["top_k"]
        if image_env == default_image:
            _assert_goldfish(payload["images"][0], [model_name])
