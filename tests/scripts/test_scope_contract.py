"""Regression tests for the apps test-scope contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
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
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_validate_scope_rejects_malformed_sha256(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    scope = _scope(example_key)
    scope["examples"][example_key]["models"]["demo-model"]["sha256"] = "ABC123"

    assert validate_scope(scope, tmp_path) == [
        f"{example_key}: model demo-model sha256 must be 64 lowercase hex characters"
    ]


@pytest.mark.parametrize("source", ["modelzoo", "url"])
def test_validate_scope_requires_file_for_checksummed_download(tmp_path, source):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    scope = _scope(example_key)
    model = scope["examples"][example_key]["models"]["demo-model"]
    model["source"] = source
    model["url"] = "https://example.invalid/demo-model.tar.gz"
    model["sha256"] = "a" * 64
    model.pop("file")

    assert validate_scope(scope, tmp_path) == [
        f"{example_key}: model demo-model sha256 requires file for {source} source"
    ]


def test_load_scope_discovers_per_example_scope_files(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    scope_file = tmp_path / "examples" / example_key / "tests" / "test-scope.yaml"
    scope_file.write_text(yaml.safe_dump(_scope_entry(example_key)), encoding="utf-8")

    scope = load_scope(tmp_path / "examples", tmp_path)

    assert scope == _scope(example_key)
    assert validate_scope(scope, tmp_path) == []


def test_load_scope_accepts_single_per_example_scope_file(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    scope_file = tmp_path / "examples" / example_key / "tests" / "test-scope.yaml"
    scope_file.write_text(yaml.safe_dump(_scope_entry(example_key)), encoding="utf-8")

    scope = load_scope(scope_file, tmp_path)

    assert scope == _scope(example_key)
    assert validate_scope(scope, tmp_path) == []


def test_validate_scope_accepts_cpp_packaged_test_binaries(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = tmp_path / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "demo-example_unit_test").write_text("", encoding="utf-8")
    (tests_dir / "demo-example_e2e_test").write_text("", encoding="utf-8")

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_validate_scope_accepts_cpp_packaged_binary_dir_tests(tmp_path):
    example_key = _write_example(tmp_path)
    tests_dir = (
        tmp_path / "examples" / "classification" / "demo-example_cpp" / "tests" / "cpp"
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
            "expected one of: examples/classification/demo-example/tests/cpp/test_e2e.cpp, "
            "examples/classification/demo-example/tests/cpp/demo-example_e2e_test, "
            "examples/classification/demo-example_cpp/tests/cpp/demo-example_e2e_test"
        ),
        (
            "classification/demo-example: cpp unit is enabled but no test artifact exists; "
            "expected one of: examples/classification/demo-example/tests/cpp/test_unit.cpp, "
            "examples/classification/demo-example/tests/cpp/demo-example_unit_test, "
            "examples/classification/demo-example_cpp/tests/cpp/demo-example_unit_test"
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


def test_download_models_preserves_empty_url_model_name(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'url-demo\\turl\\t\\thttps://example.test/demo-file.tar.gz\\tdemo-file.tar.gz\\t\\t\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "demo-file.tar.gz").write_text(
        "already downloaded\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_dir),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "[skip] url-demo already exists" in result.stdout


def test_download_models_rejects_existing_file_with_wrong_sha256(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'url-demo\\turl\\t\\thttps://example.test/demo-file.tar.gz\\t"
        "demo-file.tar.gz\\t\\t\\t"
        "0000000000000000000000000000000000000000000000000000000000000000\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "demo-file.tar.gz").write_text(
        "not the expected artifact\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_dir),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256 mismatch for demo-file.tar.gz" in result.stderr
    assert "[skip]" not in result.stdout


def test_download_models_rejects_new_file_with_wrong_sha256_before_install(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'url-demo\\turl\\t\\thttps://example.test/demo-file.tar.gz\\t"
        "demo-file.tar.gz\\t\\t\\t"
        "0000000000000000000000000000000000000000000000000000000000000000\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)
    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'not the expected artifact\\n' > demo-file.tar.gz\n",
        encoding="utf-8",
    )
    fake_sima_cli.chmod(0o755)
    models_dir = tmp_path / "models"

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_dir),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
            "SIMA_CLI_BIN": str(fake_sima_cli),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256 mismatch for demo-file.tar.gz" in result.stderr
    assert not (models_dir / "demo-file.tar.gz").exists()


def test_download_models_rejects_existing_modelzoo_file_with_wrong_sha256(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'modelzoo-demo\\tmodelzoo\\tdemo_model\\t\\tdemo-file.tar.gz\\t\\t\\t"
        "0000000000000000000000000000000000000000000000000000000000000000\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "demo-file.tar.gz").write_text(
        "not the expected artifact\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_dir),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256 mismatch for demo-file.tar.gz" in result.stderr
    assert "[skip]" not in result.stdout


def test_download_models_rejects_new_modelzoo_file_with_wrong_sha256_before_install(
    tmp_path,
):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'modelzoo-demo\\tmodelzoo\\tdemo_model\\t\\tdemo-file.tar.gz\\t\\t\\t"
        "0000000000000000000000000000000000000000000000000000000000000000\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)
    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'not the expected artifact\\n' > demo-file.tar.gz\n",
        encoding="utf-8",
    )
    fake_sima_cli.chmod(0o755)
    models_dir = tmp_path / "models"

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(models_dir),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
            "SIMA_CLI_BIN": str(fake_sima_cli),
            "NEAT_APPS_MODELZOO_VERSION": "2.1.1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sha256 mismatch for demo-file.tar.gz" in result.stderr
    assert not (models_dir / "demo-file.tar.gz").exists()


def test_download_models_expands_modelzoo_version_placeholder(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'url-demo\\turl\\t\\thttps://example.test/SDK{modelzoo_version}/demo-file.tar.gz\\tdemo-file.tar.gz\\t\\t\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)

    sima_cli_args = tmp_path / "sima-cli-args.txt"
    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" > "$NEAT_APPS_TEST_SIMA_CLI_ARGS"\n'
        "touch demo-file.tar.gz\n",
        encoding="utf-8",
    )
    fake_sima_cli.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(tmp_path / "models"),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
            "SIMA_CLI_BIN": str(fake_sima_cli),
            "NEAT_APPS_MODELZOO_VERSION": "2.1.1",
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(sima_cli_args),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "https://example.test/SDK2.1.1/demo-file.tar.gz" in sima_cli_args.read_text(
        encoding="utf-8"
    )


def test_download_models_rejects_unknown_url_placeholder(tmp_path):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    fake_scope_python = tmp_path / "fake_scope_python"
    fake_scope_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'url-demo\\turl\\t\\thttps://example.test/SDK{platform_version}/demo-file.tar.gz\\tdemo-file.tar.gz\\t\\t\\n'\n",
        encoding="utf-8",
    )
    fake_scope_python.chmod(0o755)

    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_sima_cli.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(APPS_ROOT / "scripts" / "download_models.sh"),
            "--language",
            "python",
        ],
        cwd=APPS_ROOT,
        env={
            **os.environ,
            "MODELS_DIR": str(tmp_path / "models"),
            "TEST_SCOPE_PYTHON_BIN": str(fake_scope_python),
            "SIMA_CLI_BIN": str(fake_sima_cli),
            "NEAT_APPS_MODELZOO_VERSION": "2.1.1",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unsupported URL placeholder" in result.stderr


@pytest.mark.parametrize(
    ("manifest_json", "expected_version"),
    [
        ('{"platform-version": "2.1.3", "modelzoo-version": "2.1.2"}\n', "2.1.2"),
        ('{"platform-version": "2.1.1"}\n', "2.1.1"),
    ],
    ids=["modelzoo-version-wins", "platform-version-fallback"],
)
def test_download_models_resolves_modelzoo_version_from_manifest(
    tmp_path, manifest_json, expected_version
):
    if (
        subprocess.run(["bash", "-lc", "type mapfile"], capture_output=True).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")

    root = tmp_path / "apps"
    script = root / "scripts" / "download_models.sh"
    manifest = root / "deps" / "manifest.json"
    script.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    script.write_text(
        (APPS_ROOT / "scripts" / "download_models.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    script.chmod(0o755)
    manifest.write_text(manifest_json, encoding="utf-8")

    sima_cli_args = tmp_path / "sima-cli-args.txt"
    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" > "$NEAT_APPS_TEST_SIMA_CLI_ARGS"\n',
        encoding="utf-8",
    )
    fake_sima_cli.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script), "resnet_50"],
        cwd=root,
        env={
            **os.environ,
            "MODELS_DIR": str(tmp_path / "models"),
            "PYTHON_TEST_BIN": sys.executable,
            "SIMA_CLI_BIN": str(fake_sima_cli),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(sima_cli_args),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert sima_cli_args.read_text(encoding="utf-8").strip() == (
        f"modelzoo -v {expected_version} get resnet_50"
    )
