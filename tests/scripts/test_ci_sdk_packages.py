"""Exercise SDK package preparation without Docker or package mutations."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


APPS_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "missing,operation,status,expected",
    [
        ("", "", 0, ["configure", "build"]),
        ("npm", "", 0, ["configure", "update", "install", "build"]),
        ("rg", "", 0, ["configure", "update", "install", "build"]),
        ("npm rg", "", 0, ["configure", "update", "install", "build"]),
        ("npm rg", "configure", 2, ["configure"]),
        ("npm rg", "update", 100, ["configure", "update"]),
        ("npm rg", "install", 100, ["configure", "update", "install"]),
    ],
)
def test_sdk_package_operations_precede_build(
    tmp_path, missing, operation, status, expected
):
    workflow = yaml.safe_load(
        (APPS_ROOT / ".github/workflows/vulcan-ci.yml").read_text()
    )
    script = workflow["jobs"]["build"]["with"]["build_command"]
    script = re.sub(r"\$\{\{.*?\}\}", "fixture", script)
    log = tmp_path / "commands.jsonl"
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'ps':\n"
        "    print('selected-sdk')\n"
        "elif args[:4] == ['exec', 'selected-sdk', 'bash', '-lc']:\n"
        "    tool = args[4].split()[2]\n"
        "    sys.exit(int(tool in os.environ['MISSING_TOOLS'].split()))\n"
        "elif args[0] == 'exec' and 'selected-sdk' in args:\n"
        "    with open(os.environ['COMMAND_LOG'], 'a') as stream:\n"
        "        stream.write(json.dumps(['build']) + '\\n')\n"
        "else:\n"
        "    raise AssertionError(args)\n"
    )
    docker.chmod(0o755)
    helper = tmp_path / "sdk-package"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "assert args[0] == 'selected-sdk', args\n"
        "with open(os.environ['COMMAND_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(args[1:]) + '\\n')\n"
        "if args[1] == os.environ['FAIL_OPERATION']:\n"
        "    print('fixture package failure', file=sys.stderr)\n"
        "    sys.exit(int(os.environ['FAIL_STATUS']))\n"
    )
    helper.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "SDK_PACKAGE_HELPER": str(helper),
        "COMMAND_LOG": str(log),
        "MISSING_TOOLS": missing,
        "FAIL_OPERATION": operation,
        "FAIL_STATUS": str(status),
    }
    env.pop("GITHUB_WORKSPACE", None)
    result = subprocess.run(
        ["/bin/bash", "-c", script], env=env, text=True, capture_output=True
    )
    assert result.returncode == status, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [call[0] for call in calls] == expected
    if "install" in expected:
        packages = []
        if "npm" in missing.split():
            packages.extend(["nodejs", "npm"])
        if "rg" in missing.split():
            packages.append("ripgrep")
        assert calls[2] == ["install", "-y", "--no-install-recommends", *packages]
    if status:
        assert "fixture package failure" in result.stderr
