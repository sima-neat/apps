"""Regression tests for the apps test-scope contract."""

from __future__ import annotations

from pathlib import Path

from tests.utils.test_scope import validate_scope


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


def test_validate_scope_accepts_cpp_source_tests(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (tests_dir / "test_e2e.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_validate_scope_accepts_cpp_packaged_test_binaries(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "cpp" / "tests"
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
            "examples/classification/demo-example/cpp/tests/demo-example_e2e_test"
        ),
        (
            "classification/demo-example: cpp unit is enabled but no test artifact exists; "
            "expected one of: examples/classification/demo-example/cpp/tests/test_unit.cpp, "
            "examples/classification/demo-example/cpp/tests/demo-example_unit_test"
        ),
    ]
