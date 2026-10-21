"""Hardware-gated MIPI multi-model smoke test."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"
DEFAULT_CONFIG = EXAMPLE_DIR / "src" / "common" / "config.yaml"
sys.path.insert(0, str(MAIN_PY.parent))


def require_strict_zero_copy(backend: str) -> None:
    """Apply the application's own strict zero-copy check to the described backend."""
    spec = importlib.util.spec_from_file_location("mipi_multi_model_e2e", MAIN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.require_strict_zero_copy(backend)


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("SIMA_NEAT_TEST_MIPI_CAMERA") != "1",
    reason="set SIMA_NEAT_TEST_MIPI_CAMERA=1 on a board with a configured camera",
)
def test_selected_profile_uses_strict_zero_copy(tmp_path: Path) -> None:
    profile = os.environ.get("SIMA_NEAT_MIPI_PROFILE", "detect")
    model = os.environ.get("SIMA_NEAT_MIPI_MODEL", "")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["model"] = {"profile": profile, "path": model}
    raw["runtime"] = {"frames": 2, "timeout_ms": 30_000}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MAIN_PY), "--config", str(config), "--describe"],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"MIPI inference failed with {result.returncode}\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )
    require_strict_zero_copy(result.stdout)
    assert "PASS strict-zero-copy MIPI" in result.stdout
