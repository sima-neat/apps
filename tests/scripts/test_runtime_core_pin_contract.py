"""Tests for the Core dependency pin carried by Apps runtime bundles."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_VALIDATOR = APPS_ROOT / "scripts/ci/validate_apps_runtime_archive.sh"
VULCAN_INSTALLER = APPS_ROOT / "scripts/install-vulcan-apps-package.sh"


def _write_runtime(root: Path, metadata: object | str | None) -> Path:
    runtime_dir = root / "neat-apps-runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "README.md").write_text("runtime\n", encoding="utf-8")
    if metadata is not None:
        content = metadata if isinstance(metadata, str) else json.dumps(metadata)
        (runtime_dir / "neat-core.json").write_text(content, encoding="utf-8")
    return runtime_dir


def _write_archive(tmp_path: Path, metadata: object | str | None) -> Path:
    runtime_dir = _write_runtime(tmp_path / "archive-source", metadata)
    archive = tmp_path / "neat-apps-runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(runtime_dir, arcname="neat-apps-runtime")
    return archive


def _run_installer(
    package_dir: Path,
    install_dir: Path,
    *,
    sima_cli: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NEAT_APPS_INSTALL_DIR"] = str(install_dir)
    if sima_cli is not None:
        env["SIMA_CLI_BIN"] = str(sima_cli)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(VULCAN_INSTALLER)],
        cwd=package_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_archive_validator_accepts_exact_core_pin(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        {"neat-core": {"branch": "develop", "version": "17a7c7788b07"}},
    )

    result = subprocess.run(
        ["bash", str(ARCHIVE_VALIDATOR), str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "metadata",
    [
        {"neat-core": {"branch": "develop", "version": "latest"}},
        {"neat-core": {"branch": "develop"}},
        {"neat-core": "develop"},
        "{not-json",
    ],
)
def test_archive_validator_rejects_invalid_core_pin(
    tmp_path: Path, metadata: object | str
) -> None:
    archive = _write_archive(tmp_path, metadata)

    result = subprocess.run(
        ["bash", str(ARCHIVE_VALIDATOR), str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "valid exact Core dependency pin" in result.stderr


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "{not-json",
        {"neat-core": {"branch": "develop", "version": "latest"}},
    ],
)
def test_installer_rejects_invalid_pin_before_replacing_runtime(
    tmp_path: Path, metadata: object | str | None
) -> None:
    package_dir = tmp_path / "package"
    _write_runtime(package_dir, metadata)
    install_dir = tmp_path / "installed"
    install_dir.mkdir()
    sentinel = install_dir / "existing-runtime"
    sentinel.write_text("keep\n", encoding="utf-8")

    result = _run_installer(package_dir, install_dir)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert (package_dir / "neat-apps-runtime").is_dir()


def test_installer_uses_recorded_exact_core_pin(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    metadata = {"neat-core": {"branch": "develop", "version": "17a7c7788b07"}}
    _write_runtime(package_dir, metadata)
    install_dir = tmp_path / "installed"
    args_log = tmp_path / "sima-cli-args.txt"
    sima_cli = tmp_path / "sima-cli"
    sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" > "${TEST_SIMA_CLI_ARGS}"\n',
        encoding="utf-8",
    )
    sima_cli.chmod(0o755)

    result = _run_installer(
        package_dir,
        install_dir,
        sima_cli=sima_cli,
        extra_env={
            "NEAT_APPS_CORE_INSTALL_DIR": str(tmp_path / "core-install"),
            "TEST_SIMA_CLI_ARGS": str(args_log),
        },
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert args_log.read_text(encoding="utf-8").strip() == (
        "neat install --env production -d . -t minimal "
        "core@develop:17a7c7788b07"
    )
    installed_metadata = install_dir / "neat-apps-runtime/neat-core.json"
    assert json.loads(installed_metadata.read_text(encoding="utf-8")) == metadata
