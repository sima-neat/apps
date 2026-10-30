"""Coverage, admission, strict execution and cache regression tests for CI shards."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import cache_test_models as cache
import plan_runtime_shards as planner
import run_runtime_shard as execution

from tests.utils.test_scope import discover_scope, validate_scope


def runner(name, *, rtsp=False, busy=False, status="online", extra=(),
           firmware="3.0.0_master_B2372"):
    labels = [
        "self-hosted",
        "Linux",
        "ARM64",
        "modalix",
        "vulcan-production",
        firmware,
    ]
    labels += ["rtsp"] if rtsp else ["model-registry"]
    return {
        "name": name,
        "busy": busy,
        "status": status,
        "labels": [{"name": label} for label in [*labels, *extra]],
    }


@pytest.fixture
def scope():
    return discover_scope(ROOT / "examples", ROOT)


@pytest.fixture
def plan(scope):
    fleet = [runner(f"rtsp-{i}", rtsp=True) for i in range(8)]
    fleet += [runner(f"regular-{i}") for i in range(7)]
    return planner.make_plan(scope, fleet, ["modalix", "vulcan-production"], {}, "abc")


def test_every_enabled_suite_assigned_once_with_caps(scope, plan):
    planner.validate_plan(plan, scope)
    assert sum(s["pool"] == "rtsp" for s in plan["shards"]) == 6
    assert sum(s["pool"] == "regular" for s in plan["shards"]) == 3
    assert sum(s["tasks"].count("python:unit:harness") for s in plan["shards"]) == 1
    for shard in plan["shards"]:
        assert ("rtsp" in shard["labels"]) == (shard["pool"] == "rtsp")
        if shard["pool"] == "rtsp":
            assert all(":e2e:" in t for t in shard["tasks"])


@pytest.mark.parametrize("idle, expected", [(0, 1), (1, 1), (2, 2), (8, 6)])
def test_adapts_to_idle_capacity_and_queues_one_when_busy(idle, expected):
    runners = [runner(f"r-{i}", rtsp=True, busy=i >= idle) for i in range(8)]
    fleet = planner.select_fleet(runners, ["modalix"], 6, "rtsp")
    assert len(fleet) == expected
    assert all(r["idle_at_planning"] for r in fleet) == bool(idle)


def test_does_not_count_offline_or_wrong_firmware():
    runners = [runner("offline", rtsp=True, status="offline"), runner("old", rtsp=True)]
    with pytest.raises(ValueError, match="No online"):
        planner.select_fleet(runners, ["3.0.0_master_B9999"], 6, "rtsp")


def test_planner_keeps_resolver_firmware_for_both_pools(tmp_path, monkeypatch):
    fleet = [runner(f"{version}-{pool}", rtsp=pool == "rtsp", firmware=version)
             for version in ("3.0.0_master_B2499", "3.0.0_master_B2501")
             for pool in ("rtsp", "regular")]
    monkeypatch.setattr(planner, "discover_runners", lambda *args: fleet)
    for key, value in {"RUNNER_DISCOVERY_TOKEN": "test", "GITHUB_REPOSITORY": "sima-neat/apps",
                       "GITHUB_SHA": "abc", "GITHUB_OUTPUT": str(tmp_path / "outputs"),
                       "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
                       "APPS_UNREADY_RUNNERS": "[]"}.items():
        monkeypatch.setenv(key, value)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["planner", "--group", "modalix-devkit-3.0.0",
                                      "--labels", '["modalix", "3.0.0_master_B2501"]',
                                      "--output", str(output)])
    planner.main()
    plan = json.loads(output.read_text())
    assert {s["pool"] for s in plan["shards"]} == {"rtsp", "regular"}
    assert all("3.0.0_master_b2501" in s["labels"] for s in plan["shards"])
    assert all("3.0.0_master_b2499" not in s["labels"] for s in plan["shards"])


@pytest.mark.parametrize(("actual", "expected", "fallback", "accepted"), [
    ("3.0.0_master_B2501", "3.0.0_master_B2501", "false", True),
    ("3.0.0_master_B2499", "3.0.0_master_B2501", "false", False),
    (None, "3.0.0_master_B2501", "false", False),
    ("3.0.0_master_B2501", "3.0.0_master_B2501", "true", False),
    ("3.0.0_master_B2501", "", "false", False),
])
def test_hardware_guard_checks_resolved_firmware(tmp_path, monkeypatch, actual, expected, fallback, accepted):
    workflow = yaml.safe_load((ROOT / ".github/workflows/vulcan-ci.yml").read_text())
    guard = workflow["jobs"]["test-runtime-shard"]["steps"][0]
    assert guard["name"] == "Verify selected DevKit firmware"
    code = guard["run"].split("<<'PYFIRMWARE'\n", 1)[1].rsplit("\nPYFIRMWARE", 1)[0]
    if actual:
        (tmp_path / "buildinfo").write_text(f"SIMA_BUILD_VERSION = {actual}\n")
    monkeypatch.setenv("EXPECTED_ELXR_VERSION", expected)
    monkeypatch.setenv("VERSION_FALLBACK", fallback)
    monkeypatch.setenv("RUNNER_NAME", "test")

    def check():
        with monkeypatch.context() as context:
            context.setattr("pathlib.Path", lambda filename: tmp_path / Path(filename).name)
            exec(compile(code, "workflow-firmware-guard", "exec"), {})

    if accepted:
        check()
    else:
        with pytest.raises(SystemExit, match="Cannot verify exact DevKit firmware"):
            check()


def test_regular_selector_cannot_match_rtsp_even_when_offline():
    regular = runner("regular")
    streaming = runner("stream", rtsp=True, status="offline", extra=("model-registry",))
    with pytest.raises(ValueError, match="excluding RTSP"):
        planner.select_fleet([regular, streaming], ["modalix"], 3, "regular")


def test_read_only_group_discovery_checks_repository_access():
    group = {
        "id": 7,
        "name": "devkits",
        "visibility": "selected",
        "allows_public_repositories": True,
    }

    def get(path):
        if "/repositories?" in path:
            return {"repositories": [{"full_name": "other/repo"}]}
        return {"runner_groups": [group]}

    with pytest.raises(ValueError, match="no access"):
        planner.discover_runners(get, "sima-neat/apps", "devkits", False)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong-pool"])
def test_plan_rejects_coverage_gaps_duplicates_and_wrong_pool(scope, plan, fault):
    if fault == "missing":
        plan["shards"][0]["tasks"].pop()
    elif fault == "duplicate":
        plan["shards"][0]["tasks"].append(plan["expected"][0])
    else:
        plan["shards"][0]["pool"] = "regular"
    with pytest.raises(ValueError):
        planner.validate_plan(plan, scope)


def test_missing_rtsp_metadata_is_conservative(scope):
    example = "classification/image-classification-explorer"
    del scope["examples"][example]["e2e"]["python"]["requires_rtsp"]
    task = next(
        t
        for t in planner.enumerate_suites(scope, {})
        if t["id"] == f"python:e2e:{example}"
    )
    assert task["pool"] == "rtsp"


def test_reject_non_boolean_rtsp_requirement(scope):
    scope["examples"]["classification/image-classification-explorer"]["e2e"]["python"][
        "requires_rtsp"
    ] = "false"
    assert any(
        "requires_rtsp must be boolean" in error
        for error in validate_scope(scope, ROOT)
    )


def test_partial_scope_retains_models_and_nightly_variants(scope, plan):
    for shard in plan["shards"]:
        selected = execution.scope_for_tasks(scope, shard["tasks"])
        assert not validate_scope(selected, ROOT)
        for example, entry in selected["examples"].items():
            assert entry["models"] == scope["examples"][example]["models"]
            for language in ("python", "cpp"):
                assert entry["e2e"][language].get("variants") == scope["examples"][
                    example
                ]["e2e"].get(language, {}).get("variants")


@pytest.mark.parametrize(
    "fault", ["missing", "failed", "different-sha", "missing-task", "duplicate"]
)
def test_release_gate_rejects_incomplete_results(plan, fault):
    results = [
        {"sha": "abc", "shard": s["id"], "tasks": s["tasks"].copy(), "success": True}
        for s in plan["shards"]
    ]
    if fault == "missing":
        results.pop()
    elif fault == "failed":
        results[0]["success"] = False
    elif fault == "different-sha":
        results[0]["sha"] = "other"
    elif fault == "missing-task":
        results[0]["tasks"].pop()
    elif fault == "duplicate":
        results[-1] = results[0]
    with pytest.raises(ValueError):
        execution.verify_results(plan, results)


def test_release_gate_accepts_complete_results(plan):
    results = [
        {"sha": "abc", "shard": s["id"], "tasks": s["tasks"], "success": True}
        for s in plan["shards"]
    ]
    execution.verify_results(plan, results)


def test_log_staging_does_not_follow_symlink_fixtures(tmp_path):
    runtime = tmp_path / "runtime"
    logs = runtime / "sandbox-test/python/example"
    logs.mkdir(parents=True)
    (logs / "summary.log").write_text("test output")
    (logs / "self-symlink").symlink_to("self-symlink")
    (logs / "broken.log").symlink_to("broken.log")
    (logs / "recursive-dir").symlink_to(logs, target_is_directory=True)
    destination = tmp_path / "staged"
    execution.collect_logs(runtime, destination)
    assert [
        str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()
    ] == ["python/example/summary.log"]
    assert (destination / "python/example/summary.log").read_text() == "test output"


def test_shard_execution_uses_strict_harness_and_records_all_tasks(
    scope, plan, tmp_path, monkeypatch
):
    # Use source artifacts in the fake runtime; subprocess execution is captured.
    for source in (ROOT / "examples").glob("*/*/README.md"):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    for source in (ROOT / "examples").glob("*/*/tests/*/test_*.*"):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    (tmp_path / "tests").mkdir(exist_ok=True)
    calls = []
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setattr(
        execution,
        "prepare_models",
        lambda *args: {"hits": 1, "models": 1, "seconds": 0},
    )
    monkeypatch.setattr(
        execution.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    shard = next(s for s in plan["shards"] if "python:unit:harness" in s["tasks"])
    result_path = tmp_path / "result.json"
    execution.run(plan, shard["id"], tmp_path, result_path, tmp_path / "cache")
    result = json.loads(result_path.read_text())
    assert result["success"] and sorted(result["tasks"]) == sorted(shard["tasks"])
    assert all("--strict" in command and kwargs["check"] for command, kwargs in calls)
    assert (
        sum(
            kwargs["env"]["SIMANEAT_APPS_TEST_RUN_HARNESS"] == "1"
            for _, kwargs in calls
        )
        == 1
    )


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps/manifest.json").write_text('{"modelzoo-version":"3.0.0"}')
    model = {
        "source": "url",
        "url": "https://example.invalid/model",
        "file": "model.tar.gz",
    }
    scope = {
        "examples": {
            "example": {
                "models": {"m": model},
                "e2e": {"python": {"enabled": True, "models": ["m"]}},
            }
        }
    }
    identity = {"revision": 1}
    monkeypatch.setattr(cache, "source_identity", lambda *_: identity.copy())
    downloads = []

    def download(command, **kwargs):
        destination = Path(kwargs["env"]["MODELS_DIR"]) / "model.tar.gz"
        if not destination.exists():
            downloads.append(identity.copy())
            destination.write_bytes(b"model bytes")

    monkeypatch.setattr(cache, "run_downloader", download)

    def prepare(index):
        return cache.prepare_models(
            tmp_path,
            scope,
            tmp_path / "scope.yaml",
            tmp_path / f"models-{index}",
            tmp_path / "persistent-cache",
        )

    return prepare, downloads, identity, scope


def test_nvme_cache_reuses_verified_model_without_download(acquisition):
    prepare, downloads, _, _ = acquisition
    assert prepare(1)["hits"] == 0
    assert prepare(2)["hits"] == 1
    assert len(downloads) == 1


def test_source_revision_change_invalidates_cache(acquisition):
    prepare, downloads, identity, _ = acquisition
    prepare(1)
    identity["revision"] = 2
    assert prepare(2)["hits"] == 0
    assert len(downloads) == 2


def test_corrupt_cache_is_reacquired(acquisition, tmp_path):
    prepare, downloads, _, _ = acquisition
    prepare(1)
    next((tmp_path / "persistent-cache").glob("*/model.tar.gz")).write_bytes(b"broken")
    assert prepare(2)["hits"] == 0
    assert len(downloads) == 2


def test_unvalidated_sources_are_never_reused(acquisition, monkeypatch):
    prepare, downloads, _, _ = acquisition
    monkeypatch.setattr(cache, "source_identity", lambda *_: None)
    prepare(1)
    assert prepare(2)["hits"] == 0
    assert len(downloads) == 2


def test_registry_latest_bypasses_persistent_cache():
    assert (
        cache.source_identity({"source": "model-registry", "spec": "latest"}, "3.0.0")
        is None
    )


@pytest.mark.parametrize("pool", ["regular", "rtsp"])
@pytest.mark.parametrize("excluded_status", ["online", "offline"])
@pytest.mark.parametrize("busy", [False, True])
def test_unready_exclusion_is_enforced_by_labels_not_only_candidate_name(
    pool, excluded_status, busy
):
    rtsp = pool == "rtsp"
    unready = runner("unready", rtsp=rtsp, status=excluded_status)
    indistinguishable = runner("healthy-but-same-labels", rtsp=rtsp)
    ready = runner("ready", rtsp=rtsp, busy=busy, extra=("ready-pod",))
    fleet = planner.select_fleet(
        [unready, indistinguishable, ready],
        ["modalix"],
        3,
        pool,
        excluded_names=["unready"],
    )
    assert [r["candidate"] for r in fleet] == ["ready"]
    assert "ready-pod" in fleet[0]["labels"]
    assert fleet[0]["idle_at_planning"] == (not busy)
    assert not set(fleet[0]["labels"]) <= planner.label_set(unready)


@pytest.mark.parametrize("pool", ["regular", "rtsp"])
def test_unready_exclusion_fails_closed_without_distinguishing_labels(pool):
    rtsp = pool == "rtsp"
    with pytest.raises(ValueError, match="No safe label selector"):
        planner.select_fleet(
            [runner("unready", rtsp=rtsp), runner("healthy", rtsp=rtsp)],
            ["modalix"],
            3,
            pool,
            excluded_names=["unready"],
        )


def test_missing_portal_access_fails_without_interactive_downloader(
    tmp_path, monkeypatch
):
    executable = tmp_path / "sima-cli"
    executable.write_text(f"#!{sys.executable}\n")
    monkeypatch.setenv("SIMA_CLI_BIN", str(executable))
    monkeypatch.setattr(
        cache.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 78, stdout=""),
    )
    models = {
        "model.tar.gz": {"source": "url", "url": "https://docs.sima.ai/model.tar.gz"}
    }
    with pytest.raises(RuntimeError, match="not provisioned"):
        cache.probe_sources(models, "3.0.0")


def test_release_and_portal_still_depend_on_all_shards():
    workflow = yaml.load(
        (ROOT / ".github/workflows/vulcan-ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    jobs = workflow["jobs"]
    resolver = jobs["resolve-test-runner"]
    assert "build" in resolver["needs"]
    assert resolver["with"]["manifest_artifact"] == jobs["build"]["with"]["artifact_name"]
    assert resolver["with"]["manifest_path"] in jobs["build"]["with"]["artifact_path"].splitlines()
    assert resolver["with"]["allow-version-fallback"] == "false"
    discovery = next(s for s in jobs["plan-runtime-tests"]["steps"] if s.get("id") == "plan")
    assert discovery["env"]["REQUIRED_LABELS"] == "${{ needs.resolve-test-runner.outputs.runner_labels }}"
    guard = jobs["test-runtime-shard"]["steps"][0]
    assert guard["env"]["EXPECTED_ELXR_VERSION"] == "${{ needs.resolve-test-runner.outputs.devkit_firmware }}"
    assert "test-runtime-shard" in jobs["test-runtime"]["needs"]
    assert "test-runtime" in jobs["publish-vulcan-artifacts"]["needs"]
    assert "contract-tests" in jobs["publish-vulcan-artifacts"]["needs"]
    assert "test-runtime" in jobs["build-sysdoc-examples"]["needs"]
    gate = [s.get("name") for s in jobs["test-runtime"]["steps"]]
    assert (
        gate.index("Require every test shard")
        < gate.index("Verify complete suite coverage")
        < gate.index("Package tested runtime for release")
    )


def test_oversized_model_runs_without_persistent_publication(acquisition, monkeypatch, tmp_path):
    monkeypatch.setenv("SIMANEAT_APPS_MODEL_CACHE_MAX_BYTES", "1")
    prepare, downloads, _, _ = acquisition
    result = prepare(1)
    assert result["hits"] == 0 and result["cache_bytes"] == 0
    assert result["model_sizes_bytes"]["model.tar.gz"] == len(b"model bytes")
    assert (tmp_path / "models-1/model.tar.gz").is_file()
    prepare(2)
    assert len(downloads) == 2
