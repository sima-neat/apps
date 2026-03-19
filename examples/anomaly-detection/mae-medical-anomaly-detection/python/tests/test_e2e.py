#!/usr/bin/env python3
"""E2E tests for mae-medical-anomaly-detection (Python)."""

import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "python" / "main.py"


def _find_model(models_dir: Path, pattern: str) -> Path | None:
    if not models_dir.exists():
        return None
    for f in models_dir.iterdir():
        if pattern in f.name and f.name.endswith(".tar.gz"):
            return f
    return None


@pytest.mark.e2e
def test_mae_anomaly_detection_end_to_end(
    models_dir,
    apps_root,
    test_timeout_ms,
    skip_unless_e2e_ready,
    tmp_output_dir,
) -> None:
    """Run the full MAE + classifier pipeline on bundled .npy slices if available."""
    mae_model = _find_model(
        models_dir, "mae_brats_deterministic_grid_masking_simplified_mpk"
    )
    cls_model = _find_model(models_dir, "mae_classifier_brats_split_3channels_mpk")

    skip_unless_e2e_ready(
        mae_model is not None and cls_model is not None,
        "MAE/classifier models not found in models_dir",
    )

    npy_root = apps_root / "assets" / "test_images_anomaly_detection"
    inputs = sorted(npy_root.glob("*.npy"))
    skip_unless_e2e_ready(
        len(inputs) >= 1,
        f"No .npy slices found under {npy_root}",
    )

    cmd = [
        sys.executable,
        str(MAIN_PY),
        str(mae_model),
        str(cls_model),
        *(str(p) for p in inputs),
        "--output-dir",
        str(tmp_output_dir),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=test_timeout_ms / 1000,
        cwd=str(EXAMPLE_DIR),
    )

    assert result.returncode == 0, (
        f"main.py exited with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
