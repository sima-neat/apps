"""Regression tests for explicit apps-side modelzoo version pinning."""

from __future__ import annotations

import json
import re
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]

# The runtime bundle ships neither deps/manifest.json nor download_models.sh, so
# customer-facing commands carry a copy-pasteable literal instead of resolving the
# version at run time. These tests are what keeps that literal honest.
DOCUMENTED_VERSION_RE = re.compile(r'export MODELZOO_VERSION="([^"]+)"')
DOCUMENTED_MODEL_URL_LITERAL_RE = re.compile(r"pkg_downloads/SDK\d")
PLATFORM_BADGE_RE = re.compile(r"Neat%20Development%20Environment-([^-]+)-")
# Scope files interpolate the version at download time. A hardcoded SDK<version>
# would still download successfully, so only a static check catches it.
SCOPE_VERSION_LITERAL_RE = re.compile(r"SDK\d")


def _manifest() -> dict:
    manifest = APPS_ROOT / "deps" / "manifest.json"
    return json.loads(manifest.read_text(encoding="utf-8"))


def _manifest_modelzoo_version() -> str:
    manifest = _manifest()
    return (manifest.get("modelzoo-version") or manifest["platform-version"]).strip()


def _scope_files() -> list[Path]:
    return sorted((APPS_ROOT / "examples").glob("*/*/tests/test-scope.yaml"))


def _documented_surfaces() -> list[Path]:
    files = [
        APPS_ROOT / "README.md",
        APPS_ROOT / "examples" / "TEMPLATE_README.md",
        APPS_ROOT / "scripts" / "create_example_scaffold.sh",
    ]
    files.extend(sorted((APPS_ROOT / "examples").glob("*/*/README.md")))
    return files


def test_documented_modelzoo_version_matches_manifest():
    expected = _manifest_modelzoo_version()
    offenders: list[str] = []
    documented = 0

    for path in _documented_surfaces():
        for version in DOCUMENTED_VERSION_RE.findall(path.read_text(encoding="utf-8")):
            documented += 1
            if version != expected:
                name = path.relative_to(APPS_ROOT)
                offenders.append(f"{name}: {version} != {expected}")

    assert offenders == []
    # Examples that install models from Hugging Face document no version, but the
    # pinned surfaces must never all disappear without this test noticing.
    assert documented > 0


def test_scope_files_interpolate_the_version_instead_of_hardcoding_it():
    """READMEs pin a literal; scope files must not. Documenting the asymmetry."""
    offenders = [
        str(path.relative_to(APPS_ROOT))
        for path in _scope_files()
        if SCOPE_VERSION_LITERAL_RE.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_documented_model_urls_do_not_hardcode_versions():
    offenders = [
        str(path.relative_to(APPS_ROOT))
        for path in _documented_surfaces()
        if DOCUMENTED_MODEL_URL_LITERAL_RE.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_readme_platform_badge_matches_manifest():
    expected = _manifest()["platform-version"].strip()
    readme = (APPS_ROOT / "README.md").read_text(encoding="utf-8")
    badge = PLATFORM_BADGE_RE.search(readme)

    assert badge is not None
    assert badge.group(1) == expected


def test_apps_surfaces_do_not_call_modelzoo_without_a_version():
    scanned = [
        *_documented_surfaces(),
        APPS_ROOT / "scripts" / "download_models.sh",
        APPS_ROOT / "support" / "runtime" / "asset_utils.cpp",
    ]
    offenders = [
        str(path.relative_to(APPS_ROOT))
        for path in scanned
        if "modelzoo get" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
