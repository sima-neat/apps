"""Exercise nightly dispatch, run tracking and failure propagation without GitHub."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NIGHTLY = yaml.load(
    (ROOT / ".github/workflows/nightly-e2e.yml").read_text(), Loader=yaml.BaseLoader
)
VULCAN = yaml.load(
    (ROOT / ".github/workflows/vulcan-ci.yml").read_text(), Loader=yaml.BaseLoader
)
STEPS = NIGHTLY["jobs"]["integration"]["steps"]


@pytest.fixture
def github_cli(tmp_path):
    """Record CLI calls and supply concurrent/delayed runs or CLI failures."""
    cli = tmp_path / "gh"
    cli.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            log = Path(os.environ["CLI_LOG"])
            calls = json.loads(log.read_text()) if log.exists() else []
            previous_lists = sum(call[:2] == ["run", "list"] for call in calls)
            calls.append(args)
            log.write_text(json.dumps(calls))
            if args[:2] == ["workflow", "run"]:
                sys.exit(int(os.environ.get("DISPATCH_STATUS", "0")))
            if args[:2] == ["run", "list"]:
                batches = json.loads(os.environ["RUN_BATCHES"])
                print(json.dumps(batches[min(previous_lists, len(batches) - 1)]))
            elif args[:2] == ["run", "watch"]:
                sys.exit(int(os.environ.get("WATCH_STATUS", "0")))
            elif args[:2] != ["run", "cancel"]:
                sys.exit("Unexpected CLI call: " + repr(args))
            """
        )
    )
    cli.chmod(0o755)
    sleep = tmp_path / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CLI_LOG": str(tmp_path / "calls.json"),
        "APPS_BRANCH": "develop",
        "NIGHTLY_REQUEST": "123-2",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "sima-neat/apps",
        "NIGHTLY_RUN_ID": "42",
    }


def _run(step, env):
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        env=env, capture_output=True, text=True, timeout=10,
    )


@pytest.mark.parametrize("branch", ["develop", 'feature/quoted"branch'])
def test_dispatch_tracks_its_own_run_after_delayed_visibility(github_cli, branch):
    github_cli["APPS_BRANCH"] = branch
    unrelated = {"databaseId": 99, "displayTitle": f"Apps Nightly Integration {branch} (123-1)"}
    matching = {"databaseId": 42, "displayTitle": f"Apps Nightly Integration {branch} (123-2)"}
    github_cli["RUN_BATCHES"] = json.dumps([[unrelated], [unrelated, matching]])

    result = _run(STEPS[0], github_cli)

    assert result.returncode == 0, result.stderr
    calls = json.loads(Path(github_cli["CLI_LOG"]).read_text())
    assert calls[0] == [
        "workflow", "run", "vulcan-ci.yml", "--ref", branch,
        "-f", "nightly=true", "-f", "nightly_request=123-2",
    ]
    assert len(calls) == 3
    assert Path(github_cli["GITHUB_OUTPUT"]).read_text() == "run_id=42\n"
    assert "/sima-neat/apps/actions/runs/42" in Path(github_cli["GITHUB_STEP_SUMMARY"]).read_text()


def test_missing_dispatched_run_fails_instead_of_watching_another_run(github_cli):
    github_cli["RUN_BATCHES"] = json.dumps([[{"databaseId": 99, "displayTitle": "Other run"}]])
    result = _run(STEPS[0], github_cli)
    assert result.returncode != 0
    assert "Could not find" in result.stderr
    assert not Path(github_cli["GITHUB_OUTPUT"]).exists()


def test_dispatch_failure_stops_before_run_lookup(github_cli):
    github_cli["DISPATCH_STATUS"] = "1"
    result = _run(STEPS[0], github_cli)
    assert result.returncode != 0
    assert len(json.loads(Path(github_cli["CLI_LOG"]).read_text())) == 1


@pytest.mark.parametrize("status", [0, 1, 2])
def test_build_and_test_failure_propagates_to_nightly(github_cli, status):
    github_cli["WATCH_STATUS"] = str(status)
    result = _run(STEPS[1], github_cli)
    assert result.returncode == status
    assert json.loads(Path(github_cli["CLI_LOG"]).read_text()) == [
        ["run", "watch", "42", "--interval", "30", "--exit-status"]
    ]


def test_cancelling_nightly_cancels_its_dispatched_run(github_cli):
    result = _run(STEPS[2], github_cli)
    assert result.returncode == 0
    assert "cancelled()" in STEPS[2]["if"]
    assert json.loads(Path(github_cli["CLI_LOG"]).read_text()) == [["run", "cancel", "42"]]


def test_nightly_disables_all_publication_and_keeps_build_and_test_stages():
    jobs = VULCAN["jobs"]
    for name in (
        "publish-vulcan-artifacts", "promote-latest-tag", "resolve-portal-publish-config",
        "build-sysdoc-examples", "publish-sysdoc-examples", "publish-apps-algolia",
    ):
        assert "!inputs.nightly" in jobs[name]["if"], name
    runtime_steps = {step.get("name"): step for step in jobs["test-runtime"]["steps"]}
    for name in ("Package tested runtime for release", "Upload tested Apps release"):
        assert "!inputs.nightly" in runtime_steps[name]["if"]
    shard_steps = {step.get("name"): step for step in jobs["test-runtime-shard"]["steps"]}
    tests = shard_steps["Run assigned Apps suites"]
    assert "scripts/ci/run_runtime_shard.py" in tests["run"]
    assert tests["env"]["SIMANEAT_APPS_TEST_MODEL_VARIANTS"] == "${{ inputs.nightly && '1' || '0' }}"
    for name in ("build", "contract-tests", "test-runtime-shard"):
        assert "if" not in jobs[name]
    assert "always()" in jobs["test-runtime"]["if"]
    assert jobs["test-runtime-shard"]["strategy"]["fail-fast"] == "false"
    assert "test-runtime-shard" in jobs["test-runtime"]["needs"]
    assert "./build.sh --all --clean" in jobs["build"]["with"]["build_command"]
    assert VULCAN["on"]["workflow_dispatch"]["inputs"]["nightly"]["default"] == "false"
    assert NIGHTLY["on"]["workflow_dispatch"]["inputs"]["branch"]["default"] == "develop"
    assert "inputs.nightly" in VULCAN["concurrency"]["group"]
    assert "always()" in shard_steps["Upload shard result and logs"]["if"]
