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
    files.append(
        APPS_ROOT
        / "examples"
        / "object-detection"
        / "single-rtsp-object-detection-optiview"
        / "python"
        / "main.py"
    )
    return files


def test_apps_surfaces_do_not_use_unpinned_modelzoo_get():
    offenders: list[str] = []

    for path in _files_requiring_pinned_modelzoo_calls():
        text = path.read_text(encoding="utf-8")
        if "modelzoo get" in text:
            offenders.append(str(path.relative_to(APPS_ROOT)))

    assert offenders == []
