"""Tests for the DevKit /tmp cleanup that runs before the runtime install."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STEP_NAME = "Remove scratch left in /tmp by interrupted runs"

LEFTOVERS = (
    "neat-apps-core-install.9Qljgt/core.deb",
    "neat-apps-latest.j3fRyu/tags",
    "sima-neat-install.okekYD/bundle.deb",
    "sima-neat-platform-admission-SpscjL/prepared/pkg/file",
    "sima-neat-apt-simulate-fkFVuf",
    "core-scheduler-admission-z4_c1_rg/package/file",
    "neat-sysroot-deps-a1b2c3/file",
    "sima-cli-artifact-install.Q1w2E3/file",
    "install-neat-from-a-branch.Zx98Qw.sh",
    "pytest-of-sima/pytest-117/test_case0/file",
)

KEPT = (
    "sima_neat_model_archive_tests/model.tar.gz",
    "sima-yolov8s-cache/model.bin",
    "simaai-session/file",
    "other-tool.txt",
)


def shard_steps():
    workflow = yaml.load(
        (ROOT / ".github/workflows/vulcan-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    return workflow["jobs"]["test-runtime-shard"]["steps"]


def test_tmp_cleanup_runs_before_the_core_install():
    names = [step.get("name") for step in shard_steps()]

    assert names.index(STEP_NAME) < names.index("Install Apps Core for runtime tests")


def test_tmp_cleanup_removes_interrupted_run_scratch_only(tmp_path):
    for relative in LEFTOVERS + KEPT:
        entry = tmp_path / relative
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("x")
    step = next(step for step in shard_steps() if step.get("name") == STEP_NAME)
    script = step["run"].replace("/tmp", str(tmp_path))

    proc = subprocess.run(["bash", "-e", "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "::warning::" not in proc.stdout
    assert {entry.name for entry in tmp_path.iterdir()} == {Path(kept).parts[0] for kept in KEPT}
