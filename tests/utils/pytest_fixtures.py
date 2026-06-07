"""Shared pytest fixtures for NEAT Apps example tests."""

import os
import shutil
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.utils.e2e_config import (
    APPS_ROOT,
    common_config_path_for_test,
    prepare_output_dir,
    resolve_configured_model_path,
    section_from_common_config,
    write_merged_config,
)


def _require_env(key: str, description: str) -> str:
    """Return env var value or skip/fail depending on strict mode."""
    value = os.environ.get(key, "").strip()
    if value:
        return value

    strict = os.environ.get("SIMANEAT_APPS_TEST_REQUIRE_E2E", "") == "1"
    if strict:
        pytest.fail(
            f"{key} is required ({description}) but not set. "
            "Set it or unset SIMANEAT_APPS_TEST_REQUIRE_E2E."
        )
    pytest.skip(f"set {key} ({description}) to run this test")


@pytest.fixture
def apps_root() -> Path:
    """Return the apps/ repository root."""
    return APPS_ROOT


@pytest.fixture
def e2e_config_section():
    """Return one section from an example's common/config.yaml."""

    def _get(example_name: str, section: str) -> dict[str, Any]:
        try:
            return section_from_common_config(example_name, section)
        except (FileNotFoundError, TypeError) as exc:
            pytest.fail(str(exc))

    return _get


@pytest.fixture
def models_dir() -> Path:
    """Resolve SIMANEAT_APPS_TEST_MODELS_DIR (default: assets/models)."""
    raw = os.environ.get("SIMANEAT_APPS_TEST_MODELS_DIR", "").strip()
    if raw:
        return Path(raw)
    return APPS_ROOT / "assets" / "models"


@pytest.fixture
def rtsp_url() -> str:
    """Resolve SIMANEAT_APPS_TEST_RTSP_URL; skip if unset."""
    return _require_env("SIMANEAT_APPS_TEST_RTSP_URL", "single RTSP stream URL")


@pytest.fixture
def rtsp_urls() -> list[str]:
    """Resolve SIMANEAT_APPS_TEST_RTSP_URLS (comma-separated); skip if unset."""
    raw = _require_env(
        "SIMANEAT_APPS_TEST_RTSP_URLS", "comma-separated RTSP URLs for multistream"
    )
    return [u.strip() for u in raw.split(",") if u.strip()]


@pytest.fixture
def tmp_output_dir(request) -> Path:
    """Provide a stable per-test out/ directory, cleared before each run."""
    base_raw = os.environ.get("SIMANEAT_APPS_TEST_OUTPUT_DIR", "").strip()
    base_root = Path(base_raw) if base_raw else APPS_ROOT / "sandbox" / "test-runs"
    keep_output = os.environ.get("SIMANEAT_APPS_TEST_KEEP_OUTPUT", "").strip() == "1"
    cleanup_needed = not keep_output
    # python/tests/test_e2e.py -> example directory is parents[2]
    test_file = Path(str(request.node.fspath))
    example_name = test_file.parents[2].name
    test_name = request.node.name.replace("/", "_")
    out = prepare_output_dir(base_root, "python", example_name, test_name)
    run_dir = out.parent

    try:
        yield out
    finally:
        if cleanup_needed:
            shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture
def e2e_config_writer(request, tmp_output_dir, models_dir):
    """Write a per-test config by overlaying overrides onto common/config.yaml."""

    def _write(overrides: dict[str, Any]) -> Path:
        test_file = Path(str(request.node.fspath))
        common_config = common_config_path_for_test(test_file)
        config_path = tmp_output_dir.parent / "config.yaml"
        return write_merged_config(common_config, config_path, overrides, models_dir)

    return _write


@pytest.fixture(autouse=True)
def e2e_subprocess_artifacts(request, monkeypatch):
    """Persist command/stdout/stderr for e2e subprocess runs."""
    if request.node.get_closest_marker("e2e") is None:
        yield
        return

    write_logs = os.environ.get("SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS", "1").strip() != "0"
    base_raw = os.environ.get("SIMANEAT_APPS_TEST_OUTPUT_DIR", "").strip()
    base_root = Path(base_raw) if base_raw else APPS_ROOT / "sandbox" / "test-runs"
    test_file = Path(str(request.node.fspath))
    example_name = test_file.parents[2].name
    test_name = request.node.name.replace("/", "_")
    run_dir = base_root / "python" / example_name / test_name
    real_run = subprocess.run

    def _format_command(args: Any) -> str:
        if isinstance(args, (list, tuple)):
            return " ".join(shlex.quote(str(part)) for part in args)
        return str(args)

    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _run(*args: Any, **kwargs: Any):
        command = args[0] if args else kwargs.get("args", "")
        if write_logs:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "command.txt").write_text(_format_command(command) + "\n", encoding="utf-8")
        try:
            result = real_run(*args, **kwargs)
        except Exception as exc:
            if write_logs:
                (run_dir / "stdout.log").write_text("", encoding="utf-8")
                (run_dir / "stderr.log").write_text(
                    f"[test harness] subprocess.run raised {type(exc).__name__}: {exc}\n",
                    encoding="utf-8",
                )
            raise
        if write_logs:
            (run_dir / "stdout.log").write_text(_text(result.stdout), encoding="utf-8")
            (run_dir / "stderr.log").write_text(_text(result.stderr), encoding="utf-8")
        return result

    monkeypatch.setattr(subprocess, "run", _run)
    yield


def _output_files(output_dir: Path) -> list[Path]:
    return [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "config.yaml"
    ]


def _has_output_files(output_dir: Path, expected_files: int) -> bool:
    files = _output_files(output_dir)
    return len(files) >= expected_files and all(path.stat().st_size > 0 for path in files)


@pytest.fixture
def e2e_model_path(request, models_dir, skip_unless_e2e_ready) -> Path:
    """Resolve the model path named by common/config.yaml under the active models dir."""
    test_file = Path(str(request.node.fspath))
    model_path = resolve_configured_model_path(common_config_path_for_test(test_file), models_dir)
    skip_unless_e2e_ready(
        model_path is not None and model_path.is_file(),
        f"configured model not found under {models_dir}: {model_path}",
    )
    return model_path


@pytest.fixture
def run_until_output_files(request):
    """Run a process until the requested number of output files exists."""

    write_logs = os.environ.get("SIMANEAT_APPS_TEST_WRITE_PROCESS_LOGS", "1").strip() != "0"
    base_raw = os.environ.get("SIMANEAT_APPS_TEST_OUTPUT_DIR", "").strip()
    base_root = Path(base_raw) if base_raw else APPS_ROOT / "sandbox" / "test-runs"
    test_file = Path(str(request.node.fspath))
    example_name = test_file.parents[2].name
    test_name = request.node.name.replace("/", "_")
    run_dir = base_root / "python" / example_name / test_name

    def _format_command(args: list[str]) -> str:
        return " ".join(shlex.quote(str(part)) for part in args)

    def _write_artifacts(command: list[str], stdout: str, stderr: str) -> None:
        if not write_logs:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "command.txt").write_text(_format_command(command) + "\n", encoding="utf-8")
        (run_dir / "stdout.log").write_text(stdout, encoding="utf-8")
        (run_dir / "stderr.log").write_text(stderr, encoding="utf-8")

    def _terminate(process: subprocess.Popen[str]) -> tuple[str, str]:
        process.terminate()
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.communicate()

    def _run(
        command: list[str],
        output_dir: Path,
        expected_files: int,
        timeout_s: float,
        cwd: str,
    ) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + timeout_s
        while True:
            if expected_files > 0 and _has_output_files(output_dir, expected_files):
                stdout, stderr = _terminate(process)
                _write_artifacts(command, stdout, stderr)
                return subprocess.CompletedProcess(command, 0, stdout, stderr)

            returncode = process.poll()
            if returncode is not None:
                stdout, stderr = process.communicate()
                _write_artifacts(command, stdout, stderr)
                return subprocess.CompletedProcess(command, returncode, stdout, stderr)

            if time.monotonic() >= deadline:
                stdout, stderr = _terminate(process)
                stderr += f"\n[test_process] killed after timeout ({int(timeout_s * 1000)}ms)"
                _write_artifacts(command, stdout, stderr)
                return subprocess.CompletedProcess(command, -1, stdout, stderr)

            time.sleep(0.1)

    return _run


@pytest.fixture
def test_images_dir() -> Path:
    """Return the path to bundled test images."""
    return APPS_ROOT / "assets" / "test_images"


@pytest.fixture
def test_timeout_ms() -> int:
    """Resolve SIMANEAT_APPS_TEST_TIMEOUT_MS (default: 180000)."""
    raw = os.environ.get("SIMANEAT_APPS_TEST_TIMEOUT_MS", "180000").strip()
    return int(raw)


@pytest.fixture
def require_e2e() -> bool:
    """True when strict mode is enabled for e2e tests."""
    return os.environ.get("SIMANEAT_APPS_TEST_REQUIRE_E2E", "").strip() == "1"


@pytest.fixture
def skip_unless_e2e_ready(require_e2e):
    """Skip in local mode or fail in strict mode when a prerequisite is missing."""

    def _check(condition: bool, reason: str) -> None:
        if condition:
            return
        if require_e2e:
            pytest.fail(reason)
        pytest.skip(reason)

    return _check
