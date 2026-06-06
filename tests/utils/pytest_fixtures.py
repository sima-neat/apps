"""Shared pytest fixtures for NEAT Apps example tests."""

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from tests.utils.e2e_config import (
    APPS_ROOT,
    load_e2e_config,
    prepare_output_dir,
    required_mapping,
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


def _required_mapping(value, name):
    try:
        return required_mapping(value, name)
    except TypeError as exc:
        pytest.fail(str(exc))


@pytest.fixture
def apps_root() -> Path:
    """Return the apps/ repository root."""
    return APPS_ROOT


@pytest.fixture(scope="session")
def e2e_test_config() -> dict[str, Any]:
    """Return tracked non-secret e2e semantic parameters."""
    try:
        return load_e2e_config()
    except TypeError as exc:
        pytest.fail(str(exc))


@pytest.fixture
def e2e_config_section(e2e_test_config):
    """Return one section from tests/configs/e2e.yaml, failing if it is absent."""

    def _get(example_name: str, section: str) -> dict[str, Any]:
        example = _required_mapping(e2e_test_config.get(example_name), f"e2e.{example_name}")
        value = example
        for part in section.split("."):
            value = _required_mapping(value.get(part), f"e2e.{example_name}.{section}")
        return value

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
def e2e_config_writer(request, tmp_output_dir):
    """Write a per-test config by overlaying overrides onto common/config.yaml."""

    def _write(overrides: dict[str, Any]) -> Path:
        test_file = Path(str(request.node.fspath))
        common_config = test_file.parents[2] / "common" / "config.yaml"
        config_path = tmp_output_dir.parent / "config.yaml"
        return write_merged_config(common_config, config_path, overrides)

    return _write


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
