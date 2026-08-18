"""Regression tests for the apps test-scope contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.utils import test_scope
from tests.utils.test_scope import (
    load_scope,
    scoped_model_files,
    scoped_models,
    validate_scope,
)

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


def _write_cpp_tests(root: Path, example_key: str) -> None:
    tests_dir = root / "examples" / example_key / "tests" / "cpp"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_unit.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )
    (tests_dir / "test_e2e.cpp").write_text(
        "int main() { return 0; }\n", encoding="utf-8"
    )


def _registry_scope(example_key: str) -> dict:
    scope = _scope(example_key)
    scope["examples"][example_key]["models"] = {
        "demo-model": {
            "source": "model-registry",
            "name": "demo-models",
            "ref": "main",
            "spec": "latest",
            "file": "demo-model.tar.gz",
        }
    }
    return scope


def _require_modern_bash() -> None:
    if (
        subprocess.run(
            ["bash", "-lc", "type mapfile"], capture_output=True, check=False
        ).returncode
        != 0
    ):
        pytest.skip("download_models.sh requires bash with mapfile support")


def _write_registry_scope_query(
    path: Path, rows: list[tuple[str, str]], ref: str = "main"
) -> Path:
    query = path / "fake_scope_python"
    output = "".join(
        f"{model_id}\\tmodel-registry\\tdemo-models\\t\\t{file_name}\\t\\t\\t{ref}\\tlatest\\n"
        for model_id, file_name in rows
    )
    query.write_text(
        f"#!/usr/bin/env bash\nprintf '{output}'\n",
        encoding="utf-8",
    )
    query.chmod(0o755)
    return query


def _write_registry_cli(path: Path, installed_files: list[str]) -> tuple[Path, Path]:
    calls = path / "sima-cli-calls.txt"
    cli = path / "sima-cli"
    touch_commands = "".join(
        f'touch "$install_dir/{file_name}"\n' for file_name in installed_files
    )
    cli.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$*" >> "$NEAT_APPS_TEST_SIMA_CLI_CALLS"\n'
        'install_dir="${!#:?missing install directory}"\n'
        'mkdir -p "$install_dir"\n'
        f"{touch_commands}",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli, calls


def _run_registry_download(
    models_dir: Path,
    query: Path,
    cli: Path,
    calls: Path,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
            "TEST_SCOPE_PYTHON_BIN": str(query),
            "SIMA_CLI_BIN": str(cli),
            "NEAT_APPS_TEST_SIMA_CLI_CALLS": str(calls),
            **(env_overrides or {}),
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_validate_scope_accepts_cpp_source_tests(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)

    assert validate_scope(_scope(example_key), tmp_path) == []


def test_load_scope_discovers_per_example_scope_files(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
    scope_file = tmp_path / "examples" / example_key / "tests" / "test-scope.yaml"
    scope_file.write_text(yaml.safe_dump(_scope_entry(example_key)), encoding="utf-8")

    scope = load_scope(tmp_path / "examples", tmp_path)

    assert scope == _scope(example_key)
    assert validate_scope(scope, tmp_path) == []


def test_load_scope_accepts_single_per_example_scope_file(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
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


def test_validate_scope_accepts_model_registry_source(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)

    assert validate_scope(_registry_scope(example_key), tmp_path) == []


@pytest.mark.parametrize("missing_field", ["name", "ref", "spec", "file"])
def test_validate_scope_requires_model_registry_fields(tmp_path, missing_field):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
    scope = _registry_scope(example_key)
    scope["examples"][example_key]["models"]["demo-model"][missing_field] = " "

    errors = validate_scope(scope, tmp_path)

    assert (
        f"{example_key}: model-registry model demo-model needs {missing_field}"
        in errors
    )


@pytest.mark.parametrize(
    "file_name",
    ["..", "../outside.tar.gz", "/tmp/outside.tar.gz", "nested/model.tar.gz"],
)
def test_validate_scope_rejects_model_registry_file_paths(tmp_path, file_name):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
    scope = _registry_scope(example_key)
    scope["examples"][example_key]["models"]["demo-model"]["file"] = file_name

    errors = validate_scope(scope, tmp_path)

    assert (
        f"{example_key}: model-registry model demo-model file must be a filename "
        "without directory components"
    ) in errors


def test_validate_scope_rejects_registry_destination_conflicts(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
    scope = _registry_scope(example_key)
    entry = scope["examples"][example_key]
    entry["models"]["other-model"] = {
        "source": "model-registry",
        "name": "other-models",
        "ref": "release",
        "spec": "latest",
        "file": "demo-model.tar.gz",
    }
    entry["e2e"]["cpp"]["models"].append("other-model")

    errors = validate_scope(scope, tmp_path)

    assert (
        f"{example_key}: model other-model conflicts with {example_key}: model "
        "demo-model: model-registry file 'demo-model.tar.gz' maps to different "
        "name/ref/spec"
    ) in errors


def test_validate_scope_allows_shared_destination_for_same_registry_resource(tmp_path):
    example_key = _write_example(tmp_path)
    _write_cpp_tests(tmp_path, example_key)
    scope = _registry_scope(example_key)
    entry = scope["examples"][example_key]
    entry["models"]["model-alias"] = dict(entry["models"]["demo-model"])
    entry["e2e"]["cpp"]["models"].append("model-alias")

    assert validate_scope(scope, tmp_path) == []


def test_models_command_exports_model_registry_ref_and_spec(monkeypatch, capsys):
    example_key = "classification/demo-example"
    scope = _registry_scope(example_key)
    monkeypatch.setattr(test_scope, "load_scope", lambda *_: scope)
    monkeypatch.setattr(test_scope, "validate_scope", lambda *_: [])
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_scope.py", "models", "--kind", "e2e", "--language", "cpp"],
    )

    assert test_scope.main() == 0
    assert capsys.readouterr().out.rstrip("\n").split("\t") == [
        "demo-model",
        "model-registry",
        "demo-models",
        "",
        "demo-model.tar.gz",
        "",
        "",
        "main",
        "latest",
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
        check=False,
    )

    assert result.returncode != 0
    assert "failed to resolve scoped models" in result.stderr
    assert "No scoped models are required" not in result.stdout


def test_download_models_preserves_empty_url_model_name(tmp_path):
    _require_modern_bash()

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
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[skip] url-demo already exists" in result.stdout


def test_download_models_expands_modelzoo_version_placeholder(tmp_path):
    _require_modern_bash()

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
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "https://example.test/SDK2.1.1/demo-file.tar.gz" in sima_cli_args.read_text(
        encoding="utf-8"
    )


def test_download_models_rejects_unknown_url_placeholder(tmp_path):
    _require_modern_bash()

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
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported URL placeholder" in result.stderr


def test_download_models_installs_registry_resource_once_for_multiple_files(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(
        tmp_path,
        [("first", "first.tar.gz"), ("second", "second.tar.gz")],
    )
    cli, calls = _write_registry_cli(tmp_path, ["first.tar.gz", "second.tar.gz"])
    models_dir = tmp_path / "models"

    result = _run_registry_download(models_dir, query, cli, calls)

    assert result.returncode == 0, result.stderr
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert len(call_lines) == 1
    assert call_lines[0].split()[:4] == [
        "neat",
        "install",
        "models/demo-models@main:latest",
        "--install-dir",
    ]
    assert len(call_lines[0].split()) == 5
    assert (models_dir / "first.tar.gz").is_file()
    assert (models_dir / "second.tar.gz").is_file()


def test_download_models_uses_staging_registry_for_non_main_ref(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(
        tmp_path,
        [("demo", "demo.tar.gz")],
        ref="codex/model-branch",
    )
    cli, calls = _write_registry_cli(tmp_path, ["demo.tar.gz"])

    result = _run_registry_download(tmp_path / "models", query, cli, calls)

    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").split()[:5] == [
        "neat",
        "install",
        "--stg",
        "models/demo-models@codex/model-branch:latest",
        "--install-dir",
    ]


def test_download_models_skips_complete_registry_resource(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(
        tmp_path,
        [("first", "first.tar.gz"), ("second", "second.tar.gz")],
    )
    cli, calls = _write_registry_cli(tmp_path, [])
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "first.tar.gz").touch()
    (models_dir / "second.tar.gz").touch()

    result = _run_registry_download(models_dir, query, cli, calls)

    assert result.returncode == 0, result.stderr
    assert not calls.exists()
    assert "[skip] first already exists" in result.stdout
    assert "[skip] second already exists" in result.stdout


def test_download_models_reports_missing_registry_file(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(
        tmp_path,
        [("first", "first.tar.gz"), ("second", "second.tar.gz")],
    )
    cli, calls = _write_registry_cli(tmp_path, ["first.tar.gz"])

    result = _run_registry_download(tmp_path / "models", query, cli, calls)

    assert result.returncode != 0
    assert (
        "second: requested file second.tar.gz was not installed by "
        "models/demo-models@main:latest"
    ) in result.stderr


def test_download_models_rejects_unsafe_registry_file(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(tmp_path, [("demo", "../outside.tar.gz")])
    cli, calls = _write_registry_cli(tmp_path, ["../outside.tar.gz"])
    outside = tmp_path / "outside.tar.gz"
    outside.write_text("keep", encoding="utf-8")

    result = _run_registry_download(tmp_path / "models", query, cli, calls)

    assert result.returncode != 0
    assert "demo has unsafe model-registry file: ../outside.tar.gz" in result.stderr
    assert outside.read_text(encoding="utf-8") == "keep"
    assert not calls.exists()


def test_download_models_stops_when_registry_temp_directory_fails(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(tmp_path, [("demo", "demo.tar.gz")])
    cli, calls = _write_registry_cli(tmp_path, ["demo.tar.gz"])
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_mktemp = fake_bin / "mktemp"
    fake_mktemp.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_mktemp.chmod(0o755)

    result = _run_registry_download(
        tmp_path / "models",
        query,
        cli,
        calls,
        {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert result.returncode != 0
    assert "failed to create temporary directory" in result.stderr
    assert not calls.exists()


def test_download_models_retries_after_failed_registry_copy(tmp_path):
    _require_modern_bash()
    query = _write_registry_scope_query(tmp_path, [("demo", "demo.tar.gz")])
    cli, calls = _write_registry_cli(tmp_path, ["demo.tar.gz"])
    models_dir = tmp_path / "models"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_cp = fake_bin / "cp"
    fake_cp.write_text(
        "#!/usr/bin/env bash\n"
        'for argument in "$@"; do destination="$argument"; done\n'
        "printf 'partial' > \"$destination\"\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_cp.chmod(0o755)

    failed = _run_registry_download(
        models_dir,
        query,
        cli,
        calls,
        {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )

    assert failed.returncode != 0
    assert not (models_dir / "demo.tar.gz").exists()
    assert list(models_dir.glob("demo.tar.gz.tmp.*")) == []

    retried = _run_registry_download(models_dir, query, cli, calls)

    assert retried.returncode == 0, retried.stderr
    assert (models_dir / "demo.tar.gz").is_file()
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 2


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
    _require_modern_bash()

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
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sima_cli_args.read_text(encoding="utf-8").strip() == (
        f"modelzoo -v {expected_version} get resnet_50"
    )
