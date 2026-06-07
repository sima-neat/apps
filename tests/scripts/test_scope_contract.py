"""Regression tests for the apps test-scope contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from tests.utils.test_scope import load_scope
from tests.utils.test_scope import scoped_model_files, scoped_models, validate_scope


APPS_ROOT = Path(__file__).resolve().parents[2]


def _write_example(root: Path) -> str:
    example_key = "classification/demo-example"
    readme = root / "examples" / example_key / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("# Demo Example\n", encoding="utf-8")
    return example_key


def _scope(example_key: str) -> dict:
    return {
        "examples": {
            example_key: {
                "models": {
                    "demo-model": {
                        "source": "modelzoo",
                        "name": "demo_model",
                        "file": "demo_model_mpk.tar.gz",
                    }
                },
                "unit": {"python": False, "cpp": True},
                "e2e": {
                    "python": {"enabled": False, "models": []},
                    "cpp": {"enabled": True, "models": ["demo-model"]},
                },
            }
        }
    }


def _scope_entry(example_key: str) -> dict:
    return _scope(example_key)["examples"][example_key]


def test_validate_scope_accepts_cpp_source_tests(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tests_dir / "test_e2e.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_load_scope_discovers_per_example_scope_files(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tests_dir / "test_e2e.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    scope_file = tmp_path / "examples" / example_key / "test-scope.yaml"
    scope_file.write_text(yaml.safe_dump(_scope_entry(example_key)), encoding="utf-8")

    scope = load_scope(tmp_path / "examples", tmp_path)

    assert scope == _scope(example_key)
    assert validate_scope(scope, tmp_path) == []


def test_load_scope_accepts_single_per_example_scope_file(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tests_dir / "test_e2e.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    scope_file = tmp_path / "examples" / example_key / "test-scope.yaml"
    scope_file.write_text(yaml.safe_dump(_scope_entry(example_key)), encoding="utf-8")

    scope = load_scope(scope_file, tmp_path)

    assert scope == _scope(example_key)
    assert validate_scope(scope, tmp_path) == []


def test_validate_scope_accepts_cpp_packaged_test_binaries(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "demo-example_unit_test").write_text("", encoding="utf-8")
    (tests_dir / "demo-example_e2e_test").write_text("", encoding="utf-8")

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_validate_scope_accepts_cpp_packaged_binary_dir_tests(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = (
        tmp_path / "examples" / "classification" / "demo-example_cpp" / "cpp" / "tests"
    )
    tests_dir.mkdir(parents=True)
    (tests_dir / "demo-example_unit_test").write_text("", encoding="utf-8")
    (tests_dir / "demo-example_e2e_test").write_text("", encoding="utf-8")

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_validate_scope_rejects_enabled_cpp_test_without_artifact(tmp_path):
    example_key = _write_example(tmp_path)

    errors = validate_scope(_scope(example_key), tmp_path)

    assert errors == [
        (
            "classification/demo-example: cpp e2e is enabled but no test artifact exists; "
            "expected one of: examples/classification/demo-example/cpp/tests/test_e2e.cpp, "
            "examples/classification/demo-example/cpp/tests/demo-example_e2e_test, "
            "examples/classification/demo-example_cpp/cpp/tests/demo-example_e2e_test"
        ),
        (
            "classification/demo-example: cpp unit is enabled but no test artifact exists; "
            "expected one of: examples/classification/demo-example/cpp/tests/test_unit.cpp, "
            "examples/classification/demo-example/cpp/tests/demo-example_unit_test, "
            "examples/classification/demo-example_cpp/cpp/tests/demo-example_unit_test"
        ),
    ]


def test_scoped_model_files_returns_selected_model_file():
    example_key = "classification/demo-example"

    assert scoped_model_files(_scope(example_key), "cpp", "e2e") == [
        (example_key, "demo_model_mpk.tar.gz")
    ]


def test_scoped_models_keeps_per_example_model_id_local():
    def entry(file_name: str) -> dict:
        return {
            "models": {
                "default": {
                    "source": "modelzoo",
                    "name": file_name.removesuffix("_mpk.tar.gz"),
                    "file": file_name,
                }
            },
            "unit": {"python": False, "cpp": False},
            "e2e": {
                "python": {"enabled": False, "models": []},
                "cpp": {"enabled": True, "models": ["default"]},
            },
        }

    scope = {
        "examples": {
            "classification/first-example": entry("first_model_mpk.tar.gz"),
            "classification/second-example": entry("second_model_mpk.tar.gz"),
        }
    }

    assert [model["file"] for _, model in scoped_models(scope, ["cpp"], "e2e")] == [
        "first_model_mpk.tar.gz",
        "second_model_mpk.tar.gz",
    ]


def test_download_models_fails_when_scope_resolution_fails(tmp_path):
    bad_scope = tmp_path / "bad-scope.yaml"
    bad_scope.write_text("examples: []\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--scope-file",
            str(bad_scope),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(tmp_path / "models"),
            "TEST_SCOPE_PYTHON_BIN": sys.executable,
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "failed to resolve scoped models" in result.stderr
    assert "No scoped models are required" not in result.stdout
