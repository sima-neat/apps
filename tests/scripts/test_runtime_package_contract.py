"""Regression tests for source files included in the Apps runtime archive."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_archive_includes_readme_referenced_stress_helpers():
    build_script = (APPS_ROOT / "build.sh").read_text(encoding="utf-8")
    package_start = build_script.index("package_distribution()")
    package_end = build_script.index("build_portal()", package_start)
    package_body = build_script[package_start:package_end]

    assert "-path '*/stress/*'" in package_body

    for readme in sorted((APPS_ROOT / "examples").glob("*/*/README.md")):
        references = set(re.findall(r"`(stress/[^`\s]+)`", readme.read_text(encoding="utf-8")))
        for reference in references:
            assert (readme.parent / reference).is_file(), (
                f"{readme.relative_to(APPS_ROOT)} references missing runtime helper {reference}"
            )


def test_vulcan_installer_disables_interactive_sima_cli_update_check(tmp_path: Path):
    package_dir = tmp_path / "package"
    runtime_dir = package_dir / "neat-apps-runtime"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "neat-core.json").write_text(
        json.dumps(
            {
                "neat-core": {
                    "branch": "codex/test-core",
                    "version": "deadbeef1234",
                }
            }
        ),
        encoding="utf-8",
    )

    invocation_log = tmp_path / "sima-cli-invocation.txt"
    fake_sima_cli = tmp_path / "sima-cli"
    fake_sima_cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "test \"${SIMA_CLI_CHECK_FOR_UPDATE:-}\" = 0\n"
        "printf '%s\\n' \"$*\" > \"${FAKE_SIMA_CLI_LOG}\"\n",
        encoding="utf-8",
    )
    fake_sima_cli.chmod(0o755)

    install_dir = tmp_path / "install"
    core_scratch = tmp_path / "core-scratch"
    env = os.environ.copy()
    env.pop("SIMA_CLI_CHECK_FOR_UPDATE", None)
    env.update(
        {
            "SIMA_CLI_BIN": str(fake_sima_cli),
            "FAKE_SIMA_CLI_LOG": str(invocation_log),
            "NEAT_APPS_INSTALL_DIR": str(install_dir),
            "NEAT_APPS_CORE_INSTALL_DIR": str(core_scratch),
            "NEAT_VULCAN_ENV": "test",
        }
    )
    result = subprocess.run(
        ["bash", str(APPS_ROOT / "scripts" / "install-vulcan-apps-package.sh")],
        cwd=package_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (install_dir / "neat-apps-runtime" / "neat-core.json").is_file()
    invocation = invocation_log.read_text(encoding="utf-8")
    assert "--env test" in invocation
    assert "core@codex/test-core:deadbeef1234" in invocation
