"""Shared pytest fixtures for NEAT Apps example tests."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parent.parent


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
    """Provide a temporary output directory rooted at env output dir or /tmp."""
    base_raw = os.environ.get("SIMANEAT_APPS_TEST_OUTPUT_DIR", "").strip() or "/tmp"
    keep_output = os.environ.get("SIMANEAT_APPS_TEST_KEEP_OUTPUT", "").strip() == "1"
    out: Path
    cleanup_needed = not keep_output
    # tests/python/test_e2e.py -> example directory is parents[2]
    test_file = Path(str(request.node.fspath))
    example_name = test_file.parents[2].name
    prefix = f"{example_name}_e2e_"

    base = Path(base_raw) / "python"
    base.mkdir(parents=True, exist_ok=True)
    out_str = tempfile.mkdtemp(prefix=prefix, dir=str(base))
    out = Path(out_str)

    try:
        yield out
    finally:
        if cleanup_needed:
            shutil.rmtree(out, ignore_errors=True)


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
    """True when strict e2e mode is enabled."""
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
