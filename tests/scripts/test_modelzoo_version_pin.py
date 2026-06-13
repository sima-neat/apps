"""Regression tests for explicit apps-side modelzoo version pinning."""

from __future__ import annotations

from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]


def _files_requiring_pinned_modelzoo_calls() -> list[Path]:
    files = [
        APPS_ROOT / "scripts" / "download_models.sh",
        APPS_ROOT / "support" / "runtime" / "asset_utils.cpp",
        APPS_ROOT / "examples" / "TEMPLATE_README.md",
    ]
    files.extend(sorted((APPS_ROOT / "examples").glob("*/*/README.md")))
    files.extend(sorted((APPS_ROOT / "examples").glob("*/*/tests/test-scope.yaml")))
    files.append(
        APPS_ROOT
        / "examples"
        / "object-detection"
        / "single-stream-object-detector"
        / "src"
        / "python"
        / "main.py"
    )
    return files


def test_apps_surfaces_do_not_hardcode_model_platform_version():
    offenders: list[str] = []
    blocked = [
        "modelzoo get",
        "modelzoo -v 2.0.0",
        "SDK2.0.0",
        "{sdk_version}",
        "NEAT_APPS_MODEL_SDK_VERSION",
        "MODEL_SDK_VERSION",
    ]

    for path in _files_requiring_pinned_modelzoo_calls():
        text = path.read_text(encoding="utf-8")
        matches = [pattern for pattern in blocked if pattern in text]
        if matches:
            offenders.append(f"{path.relative_to(APPS_ROOT)}: {', '.join(matches)}")

    assert offenders == []
