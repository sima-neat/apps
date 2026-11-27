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

SOM_16 = "modalix-som-16gig"
SOM_8 = "modalix-som-8gig--single-dram"
HIGH_DENSITY = "object-detection/high-density-multi-stream-object-detector"
HIGH_DENSITY_E2E = [f"{language}:e2e:{HIGH_DENSITY}" for language in ("cpp", "python")]
BASE_LABELS = ["modalix", "vulcan-production"]


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
def constrained(scope):
    for language in ("cpp", "python"):
        scope["examples"][HIGH_DENSITY]["e2e"][language]["runner_labels"] = [SOM_16]
    return scope


@pytest.fixture
def plan(constrained):
    fleet = [runner(f"devkit-{i}", extra=(SOM_16,)) for i in range(10)]
    return planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")


def test_every_enabled_suite_assigned_once_with_caps(scope, plan):
    planner.validate_plan(plan, scope)
    assert len({s["candidate"] for s in plan["shards"]}) == len(plan["shards"]) == 8
    assert any(s["needs_media_fixtures"] for s in plan["shards"])
    assert sum(s["tasks"].count("python:unit:harness") for s in plan["shards"]) == 1
    assert all("rtsp" not in s["labels"] for s in plan["shards"])


@pytest.mark.parametrize("idle, expected", [(0, 1), (1, 1), (2, 2), (8, 8)])
def test_adapts_to_idle_capacity_and_queues_one_when_busy(idle, expected):
    runners = [runner(f"r-{i}", rtsp=True, busy=i >= idle) for i in range(8)]
    fleet = planner.select_fleet(runners, ["modalix"], 8)
    assert len(fleet) == expected
    assert all(r["idle_at_planning"] for r in fleet) == bool(idle)


def test_does_not_count_offline_or_wrong_firmware():
    runners = [runner("offline", rtsp=True, status="offline"), runner("old", rtsp=True)]
    with pytest.raises(ValueError, match="No online"):
        planner.select_fleet(runners, ["3.0.0_master_B9999"], 8)


def test_planner_keeps_resolver_firmware(tmp_path, monkeypatch):
    fleet = [runner(f"{version}-{pool}", rtsp=pool == "rtsp", firmware=version,
                    extra=(SOM_16,))
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
    assert plan["unmet_runner_labels"] == {}
    matrix = json.loads((tmp_path / "outputs").read_text().removeprefix("matrix="))
    assert len(matrix["include"]) == 2
    assert all("rtsp" not in s["labels"] for s in matrix["include"])
    assert [s["needs_media_fixtures"] for s in matrix["include"]] == [
        s["needs_media_fixtures"] for s in plan["shards"]
    ]
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

    # Python 3.11 Path construction consults the module-level Path factory.
    concrete_path = type(tmp_path)

    def check():
        with monkeypatch.context() as context:
            context.setattr("pathlib.Path", lambda filename: tmp_path / concrete_path(filename).name)
            exec(compile(code, "workflow-firmware-guard", "exec"), {})

    if accepted:
        check()
    else:
        with pytest.raises(SystemExit, match="Cannot verify exact DevKit firmware"):
            check()


def test_legacy_rtsp_label_cannot_be_required_or_used_to_exclude_an_unready_runner():
    legacy = runner("legacy", rtsp=True, extra=("model-registry",))
    unready = runner("unready", status="offline")
    with pytest.raises(ValueError, match="Base runner selector"):
        planner.select_fleet([legacy], ["modalix", "rtsp"], 8)
    with pytest.raises(ValueError, match="No safe label selector"):
        planner.select_fleet([legacy, unready], ["modalix"], 8, excluded_names=["unready"])


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


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong-fixtures", "rtsp-selector"])
def test_plan_rejects_incorrect_coverage_fixture_setup_and_labels(scope, plan, fault):
    if fault == "missing":
        plan["shards"][0]["tasks"].pop()
    elif fault == "duplicate":
        plan["shards"][0]["tasks"].append(plan["expected"][0])
    elif fault == "wrong-fixtures":
        shard = plan["shards"][0]
        shard["needs_media_fixtures"] = not shard["needs_media_fixtures"]
    else:
        plan["shards"][0]["labels"].append("rtsp")
    with pytest.raises(ValueError):
        planner.validate_plan(plan, scope)


@pytest.mark.parametrize("requires_rtsp, expected", [(None, True), (False, False)])
def test_scope_controls_media_fixture_setup(scope, requires_rtsp, expected):
    example = "classification/image-classification-explorer"
    e2e = scope["examples"][example]["e2e"]["python"]
    if requires_rtsp is None:
        del e2e["requires_rtsp"]
    else:
        e2e["requires_rtsp"] = requires_rtsp
    task = next(
        t
        for t in planner.enumerate_suites(scope, {})
        if t["id"] == f"python:e2e:{example}"
    )
    assert task["needs_media_fixtures"] is expected


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
                original = scope["examples"][example]["e2e"].get(language, {})
                e2e = entry["e2e"][language]
                assert e2e.get("variants") == original.get("variants")
                assert e2e.get("runner_labels") == original.get("runner_labels")


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


REGISTRY_MODEL = {
    "source": "model-registry",
    "name": "demo",
    "ref": "main",
    "spec": "latest",
    "variant": "w8a8",
    "file": "model.tar.gz",
}


def _registry_listing(monkeypatch, builds, *, fail=False):
    commands = []

    def run(command, **_):
        commands.append(command)
        if fail:
            raise subprocess.CalledProcessError(1, command)
        stdout = "notice before the document\n" + json.dumps({"models": builds})
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setenv("SIMA_CLI_BIN", "sima-cli")
    monkeypatch.setattr(cache.subprocess, "run", run)
    return commands


def _build(run_id, variant="w8a8"):
    return {
        "id": run_id,
        "sha256": f"sha-{run_id}",
        "metadata": {"model_id": "demo", "variant_id": variant},
    }


def test_registry_variant_identity_is_the_served_build(monkeypatch):
    commands = _registry_listing(monkeypatch, [_build("run-1"), _build("other", "w16a16")])
    staging = {**REGISTRY_MODEL, "ref": "feat/x", "file": "staging.tar.gz"}

    identities = cache.registry_identities(
        {"model.tar.gz": REGISTRY_MODEL, "staging.tar.gz": staging}
    )

    assert identities["model.tar.gz"] == {
        "model": REGISTRY_MODEL,
        "run_id": "run-1",
        "package_sha256": "sha-run-1",
    }
    assert commands == [
        ["sima-cli", "models", "list", "--stg", "--branch", "feat/x", "--json"],
        ["sima-cli", "models", "list", "--branch", "main", "--json"],
    ]


@pytest.mark.parametrize(
    ("builds", "fail"),
    [([], False), ([_build("run-1"), _build("run-2")], False), ([_build("run-1")], True)],
    ids=["missing", "ambiguous", "listing-failed"],
)
def test_unidentified_registry_variant_bypasses_cache(monkeypatch, builds, fail):
    _registry_listing(monkeypatch, builds, fail=fail)
    assert cache.registry_identities({"model.tar.gz": REGISTRY_MODEL}) == {}


def test_registry_variant_is_reused_until_its_build_changes(acquisition, monkeypatch):
    prepare, downloads, _, scope = acquisition
    scope["examples"]["example"]["models"]["m"] = REGISTRY_MODEL
    builds = [_build("run-1")]
    listings = _registry_listing(monkeypatch, builds)

    assert prepare(1)["hits"] == 0
    # One listing per run: the download is published without asking again.
    assert len(listings) == 1
    assert prepare(2)["hits"] == 1
    builds[0] = _build("run-2")
    assert prepare(3)["hits"] == 0
    assert len(downloads) == 2


@pytest.mark.parametrize("rtsp", [False, True])
@pytest.mark.parametrize("excluded_status", ["online", "offline"])
@pytest.mark.parametrize("busy", [False, True])
def test_unready_exclusion_is_enforced_by_labels_not_only_candidate_name(
    rtsp, excluded_status, busy
):
    unready = runner("unready", rtsp=rtsp, status=excluded_status)
    indistinguishable = runner("healthy-but-same-labels", rtsp=rtsp)
    ready = runner("ready", rtsp=rtsp, busy=busy, extra=("ready-pod",))
    fleet = planner.select_fleet(
        [unready, indistinguishable, ready],
        ["modalix"],
        3,
        excluded_names=["unready"],
    )
    assert [r["candidate"] for r in fleet] == ["ready"]
    assert "ready-pod" in fleet[0]["labels"]
    assert fleet[0]["idle_at_planning"] == (not busy)
    assert not set(fleet[0]["labels"]) <= planner.label_set(unready)


@pytest.mark.parametrize("rtsp", [False, True])
def test_unready_exclusion_fails_closed_without_distinguishing_labels(rtsp):
    with pytest.raises(ValueError, match="No safe label selector"):
        planner.select_fleet(
            [runner("unready", rtsp=rtsp), runner("healthy", rtsp=rtsp)],
            ["modalix"],
            3,
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
    assert resolver["with"]["manifest_path"] == "deps/scheduler-sdk-policy.json"
    assert resolver["with"]["allow-version-fallback"] == "false"
    discovery = next(s for s in jobs["plan-runtime-tests"]["steps"] if s.get("id") == "plan")
    assert discovery["env"]["REQUIRED_LABELS"] == "${{ needs.resolve-test-runner.outputs.runner_labels }}"
    guard = jobs["test-runtime-shard"]["steps"][0]
    assert guard["env"]["EXPECTED_ELXR_VERSION"] == "${{ needs.resolve-test-runner.outputs.devkit_firmware }}"
    assert "rtsp" not in json.loads(resolver["with"]["runner-labels"])
    shard_steps = jobs["test-runtime-shard"]["steps"]
    setup_index = next(
        i for i, step in enumerate(shard_steps) if step.get("name") == "Set up Colo media fixtures"
    )
    setup = shard_steps[setup_index]
    assert setup["if"] == "${{ matrix.needs_media_fixtures }}"
    assert setup["uses"].startswith("sima-neat/.github/actions/setup-colo-media-fixtures@")
    assert setup["with"]["profile"] == "all"
    run = shard_steps[setup_index + 1]
    assert run["name"] == "Run assigned Apps suites"
    assert not any(key.startswith("SIMANEAT_TEST_") for key in run["env"])
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


def runner_label_errors(scope, language):
    prefix = f"{HIGH_DENSITY}: e2e.{language}.runner_labels "
    return [e.removeprefix(prefix) for e in validate_scope(scope, ROOT) if e.startswith(prefix)]


@pytest.mark.parametrize("language", ["cpp", "python"])
def test_validate_scope_accepts_runner_labels_on_a_disabled_suite(scope, language):
    e2e = scope["examples"][HIGH_DENSITY]["e2e"][language]
    e2e["enabled"] = False
    e2e["runner_labels"] = [SOM_16]
    assert not validate_scope(scope, ROOT)


@pytest.mark.parametrize("language", ["cpp", "python"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(("value", "messages"), [
    (None, ["must be a list"]),
    (SOM_16, ["must be a list"]),
    ([], ["must not be empty"]),
    ([""], ["items must be non-empty strings, not ''"]),
    ([16], ["items must be non-empty strings, not 16"]),
    (["  "], ["item '  ' must not contain whitespace"]),
    ([f" {SOM_16}"], [f"item ' {SOM_16}' must not contain whitespace"]),
    (["modalix som"], ["item 'modalix som' must not contain whitespace"]),
    (["modalix-som\t16gig"], ["item 'modalix-som\\t16gig' must not contain whitespace"]),
    ([f"{SOM_16}\n"], [f"item '{SOM_16}\\n' must not contain whitespace"]),
    ([SOM_16, "Modalix-SOM-16GIG"], ["repeats 'Modalix-SOM-16GIG'"]),
    (["Modalix-SOM-16GIG", SOM_16], [f"repeats '{SOM_16}'"]),
    (["RTSP"], ["must not name the legacy rtsp label"]),
    ([SOM_16, "rtsp"], ["must not name the legacy rtsp label"]),
    ([16, " x"], ["items must be non-empty strings, not 16",
                  "item ' x' must not contain whitespace"]),
])
def test_validate_scope_rejects_invalid_runner_labels(scope, value, messages, enabled, language):
    e2e = scope["examples"][HIGH_DENSITY]["e2e"][language]
    e2e["runner_labels"] = value
    e2e["enabled"] = enabled
    assert runner_label_errors(scope, language) == messages


@pytest.mark.parametrize("language", ["cpp", "python"])
@pytest.mark.parametrize(("place", "message"), [
    ("top", "runner_labels belongs under e2e.<language>"),
    ("e2e", "e2e has unknown key 'runner_labels'"),
    ("e2e-capitalized", "e2e has unknown key '{Language}'"),
    ("e2e-bool", "e2e has unknown key True"),
    ("unit", "unit has unknown key 'runner_labels'"),
    ("typo", "e2e.{language} has unknown key 'runner_label'"),
    ("typo-disabled", "e2e.{language} has unknown key 'runner_label'"),
    ("bool", "e2e.{language} has unknown key True"),
])
def test_validate_scope_rejects_misplaced_runner_labels(scope, place, message, language):
    entry = scope["examples"][HIGH_DENSITY]
    e2e = entry["e2e"][language]
    if place == "top":
        entry["runner_labels"] = [SOM_16]
    elif place == "e2e":
        entry["e2e"]["runner_labels"] = [SOM_16]
    elif place == "e2e-capitalized":
        entry["e2e"][language.capitalize()] = {"runner_labels": [SOM_16]}
    elif place == "e2e-bool":
        # YAML reads an "on:" key as True. Sorting it with a str key must not fail.
        entry["e2e"][True] = {}
        entry["e2e"]["runner_labels"] = [SOM_16]
    elif place == "unit":
        entry["unit"]["runner_labels"] = [SOM_16]
    elif place == "bool":
        e2e[True] = [SOM_16]
        e2e["runner_label"] = [SOM_16]
    else:
        e2e["enabled"] = place == "typo"
        e2e["runner_label"] = [SOM_16]
    expected = message.format(language=language, Language=language.capitalize())
    assert f"{HIGH_DENSITY}: {expected}" in validate_scope(scope, ROOT)


def shard_of(plan, task):
    return next(s for s in plan["shards"] if task in s["tasks"])


def candidates(plan):
    return [s["candidate"] for s in plan["shards"]]


def without_runner_labels(scope):
    for entry in scope["examples"].values():
        for config in entry.get("e2e", {}).values():
            config.pop("runner_labels", None)
    return scope


@pytest.mark.parametrize("names", ["abcdefghij", "jihgfedcba"])
def test_unconstrained_scope_keeps_name_ordered_fleet_and_balancing(scope, names):
    scope = without_runner_labels(scope)
    weights = json.loads((ROOT / "tests/ci-durations.json").read_text())["suite_seconds"]
    fleet = [runner(f"runner-{c}", busy=c == "b", extra=(SOM_8,)) for c in names]
    plan = planner.make_plan(scope, fleet, BASE_LABELS, weights, "abc")
    assert candidates(plan) == [f"runner-{c}" for c in "acdefghi"]
    # Without runner_labels, each suite, longest first, goes to the least-loaded shard.
    tasks = {s["id"]: [] for s in plan["shards"]}
    load = dict.fromkeys(tasks, 0)
    for task in sorted(planner.enumerate_suites(scope, weights),
                       key=lambda s: (-s["seconds"], s["id"])):
        shard = min(tasks, key=lambda i: (load[i], i))
        tasks[shard].append(task["id"])
        load[shard] += task["seconds"]
    assert {s["id"]: s["tasks"] for s in plan["shards"]} == tasks


def test_idle_16_gib_runner_replaces_a_spare_shard_in_a_full_fleet(constrained):
    fleet = [runner(f"runner-{c}", extra=(SOM_8,)) for c in "abcdefgh"]
    fleet += [runner("runner-y", busy=True, extra=(SOM_16,)), runner("runner-z", extra=(SOM_16,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert len(plan["shards"]) == planner.MAX_SHARDS
    assert candidates(plan) == [f"runner-{c}" for c in "abcdefg"] + ["runner-z"]
    assert {shard_of(plan, t)["candidate"] for t in HIGH_DENSITY_E2E} == {"runner-z"}
    assigned = [t for s in plan["shards"] for t in s["tasks"]]
    assert sorted(assigned) == plan["expected"]


def test_full_fleet_keeps_a_shard_that_covers_another_label_set(constrained):
    constrained["examples"][HIGH_DENSITY]["e2e"]["python"]["runner_labels"] = ["model-registry"]
    fleet = [runner(f"runner-{c}", rtsp=c != "h", extra=(SOM_8,)) for c in "abcdefgh"]
    fleet.append(runner("runner-z", rtsp=True, extra=(SOM_16,)))
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == [f"runner-{c}" for c in "abcdefh"] + ["runner-z"]
    assert shard_of(plan, HIGH_DENSITY_E2E[1])["candidate"] == "runner-h"


@pytest.mark.parametrize("idle_8_gib", [3, 8])
def test_busy_16_gib_runner_gets_a_queued_shard_with_only_the_suites_that_need_it(
    constrained, idle_8_gib
):
    fleet = [runner(f"runner-{i}", extra=(SOM_8,)) for i in range(idle_8_gib)]
    fleet.append(runner("runner-z", busy=True, extra=(SOM_16,)))
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert len(plan["shards"]) == min(idle_8_gib + 1, planner.MAX_SHARDS)
    queued = shard_of(plan, HIGH_DENSITY_E2E[0])
    assert queued["candidate"] == "runner-z" and not queued["idle_at_planning"]
    assert sorted(queued["tasks"]) == HIGH_DENSITY_E2E


def test_busy_shard_uses_the_selector_that_most_runners_match(constrained):
    # runner-c's selector also matches runner-d, while runner-b's matches only runner-b.
    fleet = [runner("runner-a", extra=(SOM_8,)),
             runner("runner-b", busy=True, extra=(SOM_16, "label-r")),
             runner("runner-c", busy=True, extra=(SOM_16, "label-p", "label-q")),
             runner("runner-d", busy=True, extra=(SOM_16, "label-p", "label-q", "label-s"))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == ["runner-a", "runner-c"]


def test_covered_label_sets_add_no_shards(constrained):
    fleet = [runner(f"runner-{i}", extra=(SOM_16,)) for i in range(10)]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == [f"runner-{i}" for i in range(8)]


def test_an_idle_runner_is_kept_over_a_busy_one_that_covers_more(constrained):
    constrained["examples"][HIGH_DENSITY]["e2e"]["python"]["runner_labels"] = ["label-x"]
    fleet = [runner(f"runner-{c}", extra=(SOM_8,)) for c in "abcdefgh"]
    fleet += [runner("runner-y", extra=(SOM_16,)),
              runner("runner-z", busy=True, extra=(SOM_16, "label-x"))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == [f"runner-{c}" for c in "abcdef"] + ["runner-y", "runner-z"]
    assert shard_of(plan, HIGH_DENSITY_E2E[0])["candidate"] == "runner-y"
    assert shard_of(plan, HIGH_DENSITY_E2E[1])["tasks"] == [HIGH_DENSITY_E2E[1]]


def test_constrained_suites_use_an_idle_shard_before_a_queued_one(constrained):
    detr = "object-detection/detr-object-detector"
    constrained["examples"][HIGH_DENSITY]["e2e"]["python"]["runner_labels"] = ["label-x"]
    constrained["examples"][detr]["e2e"]["cpp"]["runner_labels"] = [SOM_16]
    fleet = [runner("runner-a", extra=(SOM_8,)), runner("runner-b", extra=(SOM_16,)),
             runner("runner-c", busy=True, extra=(SOM_16, "label-x"))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {f"cpp:e2e:{detr}": 10_000}, "abc")
    assert shard_of(plan, HIGH_DENSITY_E2E[0])["candidate"] == "runner-b"
    assert shard_of(plan, HIGH_DENSITY_E2E[1])["candidate"] == "runner-c"


def test_busy_shard_prefers_fewer_labels_between_equally_general_selectors(constrained):
    fleet = [runner("runner-a", extra=(SOM_8,)),
             runner("runner-b", busy=True, extra=(SOM_16, "label-p", "label-q")),
             runner("runner-c", busy=True, extra=(SOM_16, "label-r"))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == ["runner-a", "runner-c"]


def test_shards_without_suites_are_dropped_and_renumbered(scope):
    chosen = [HIGH_DENSITY, "object-detection/detr-object-detector",
              "anomaly-detection/fastflow-anomaly-detector"]
    for example, entry in scope["examples"].items():
        entry["unit"] = {"cpp": False, "python": False}
        for language in ("cpp", "python"):
            e2e = entry.setdefault("e2e", {}).setdefault(language, {})
            e2e["enabled"] = example in chosen
            e2e.pop("runner_labels", None)
            if example in chosen:
                e2e["runner_labels"] = [SOM_16]
    fleet = [runner(f"runner-{c}", extra=("modalix-som",)) for c in "abcdefg"]
    fleet.append(runner("runner-z", extra=(SOM_16,)))
    plan = planner.make_plan(scope, fleet, BASE_LABELS, {}, "abc")
    assert [(s["id"], s["candidate"]) for s in plan["shards"]] == [
        ("devkit-1", "runner-a"), ("devkit-2", "runner-z")
    ]
    assert len(shard_of(plan, HIGH_DENSITY_E2E[0])["tasks"]) == 6


@pytest.mark.parametrize("with_extra_label", ["a", "c"])
def test_all_busy_unconstrained_fleet_queues_the_first_runner_by_name(with_extra_label):
    fleet = [runner(f"runner-{c}", busy=True, extra=("label-p",) if c == with_extra_label else ())
             for c in "cba"]
    assert [s["candidate"] for s in planner.select_fleet(fleet, ["modalix"], 8)] == ["runner-a"]


def test_all_busy_fallback_queues_a_shard_that_covers_the_requirement(constrained):
    fleet = [runner(f"runner-{c}", busy=True, extra=(SOM_8,)) for c in "ab"]
    fleet.append(runner("runner-c", busy=True, extra=(SOM_16,)))
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert candidates(plan) == ["runner-c"]


@pytest.mark.parametrize(("labels", "expected"), [
    ((("x",), ("y",), ("x", "y")), ["runner-c"]),
    ((("x",), ("y",)), ["runner-a", "runner-b"]),
])
def test_all_busy_fallback_prefers_a_shard_that_covers_more_label_sets(labels, expected):
    fleet = [runner(f"runner-{c}", busy=True, extra=extra)
             for c, extra in reversed(list(zip("abc", labels)))]
    suites = [{"id": "x", "runner_labels": ["x"]}, {"id": "y", "runner_labels": ["y"]}]
    selected = planner.select_fleet(fleet, ["modalix"], 8, suites=suites)
    assert [s["candidate"] for s in selected] == expected


def test_label_sets_beyond_the_fleet_cap_fall_back_with_a_reason(scope):
    scope = without_runner_labels(scope)
    e2e = [s["id"] for s in planner.enumerate_suites(scope, {}) if ":e2e:" in s["id"]][:9]
    for i, suite in enumerate(e2e):
        language, _, example = suite.split(":", 2)
        scope["examples"][example]["e2e"][language]["runner_labels"] = [f"label-{i}"]
    fleet = [runner(f"runner-{i}", extra=(f"label-{i}",)) for i in range(9)]
    plan = planner.make_plan(scope, fleet, BASE_LABELS, {}, "abc")
    assert len(plan["shards"]) == planner.MAX_SHARDS
    assert plan["unmet_runner_labels"] == {e2e[8]: ["label-8"]}
    assert planner.unmet_reason(fleet, BASE_LABELS, [], ["label-8"]) == (
        "other label sets take every shard the fleet cap allows"
    )


@pytest.mark.parametrize("unavailable", ["unready", "offline"])
def test_missing_16_gib_runner_falls_back_to_any_shard_and_records_it(constrained, unavailable):
    status = "offline" if unavailable == "offline" else "online"
    fleet = [runner("runner-a", extra=(SOM_8,)), runner("runner-b", extra=(SOM_8,)),
             runner("runner-z", status=status, extra=(SOM_16,))]
    excluded = ["runner-z"] if unavailable == "unready" else []
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc", excluded_names=excluded)
    assert candidates(plan) == ["runner-a", "runner-b"]
    assert plan["unmet_runner_labels"] == dict.fromkeys(HIGH_DENSITY_E2E, [SOM_16])
    assigned = [t for s in plan["shards"] for t in s["tasks"]]
    assert sorted(assigned) == plan["expected"]


@pytest.mark.parametrize(("case", "reason"), [
    ("misspelled", ("no DevKit in the runner group has them, so check the label spelling "
                   "and the DevKit labels")),
    ("offline", "every DevKit with them is offline or lacks this run's platform or "
                "firmware labels"),
    ("other-firmware", "every DevKit with them is offline or lacks this run's platform or "
                       "firmware labels"),
    ("unready", "every selector with them also matches a DevKit in APPS_UNREADY_RUNNERS"),
    ("partial", ("no DevKit in the runner group has them, so check the label spelling "
                 "and the DevKit labels")),
])
def test_unmet_reason_says_why_no_devkit_has_the_labels(case, reason):
    firmware = "3.0.0_master_B2372"
    fleet = [runner("runner-a", extra=(SOM_8,)),
             runner("runner-z", status="offline" if case == "offline" else "online",
                    firmware="other" if case == "other-firmware" else firmware,
                    extra=("label-p",) if case == "partial" else (SOM_16,))]
    labels = {"misspelled": ["modalix-som-16gib"], "partial": ["label-p", SOM_16]}.get(
        case, [SOM_16])
    excluded = ["runner-z"] if case == "unready" else []
    assert planner.unmet_reason(fleet, [*BASE_LABELS, firmware], excluded, labels) == reason


@pytest.mark.parametrize(("case", "reason"), [
    ("misspelled", ("no DevKit in the runner group has them, so check the label spelling "
                   "and the DevKit labels")),
    ("unready", "every selector with them also matches a DevKit in APPS_UNREADY_RUNNERS"),
    ("offline", "every DevKit with them is offline or lacks this run's platform or "
                "firmware labels"),
    ("other-firmware", "every DevKit with them is offline or lacks this run's platform or "
                       "firmware labels"),
])
def test_unmet_runner_labels_warn_in_the_log_and_step_summary(
    constrained, tmp_path, monkeypatch, capsys, case, reason
):
    fleet = [runner("runner-a", extra=(SOM_8,))]
    if case != "misspelled":
        status = "offline" if case == "offline" else "online"
        firmware = "other" if case == "other-firmware" else "3.0.0_master_B2372"
        fleet.append(runner("runner-z", status=status, firmware=firmware, extra=(SOM_16,)))
    unready = '["runner-z"]' if case == "unready" else "[]"
    monkeypatch.setattr(planner, "discover_scope", lambda *args: constrained)
    monkeypatch.setattr(planner, "discover_runners", lambda *args: fleet)
    for key, value in {"RUNNER_DISCOVERY_TOKEN": "test", "GITHUB_REPOSITORY": "sima-neat/apps",
                       "GITHUB_SHA": "abc", "GITHUB_OUTPUT": str(tmp_path / "outputs"),
                       "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
                       "APPS_UNREADY_RUNNERS": unready}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["planner", "--group", "devkits",
                                      "--labels", '["modalix", "3.0.0_master_B2372"]',
                                      "--output", str(tmp_path / "plan.json")])
    planner.main()
    out = capsys.readouterr().out
    summary = (tmp_path / "summary").read_text()
    for suite in HIGH_DENSITY_E2E:
        warning = (f"No planned DevKit has the runner labels {suite} requires ({SOM_16}): "
                   f"{reason}. The suite runs on any DevKit.")
        assert f"::warning::{warning}\n" in out
        assert f"> [!WARNING]\n> {warning}\n\n" in summary


def test_plan_accepts_a_listed_unmet_suite_and_rejects_an_unlisted_one(constrained):
    plan = planner.make_plan(constrained, [runner("runner-a", extra=(SOM_8,))],
                             BASE_LABELS, {}, "abc")
    planner.validate_plan(plan, constrained)
    del plan["unmet_runner_labels"][HIGH_DENSITY_E2E[0]]
    with pytest.raises(ValueError, match=f"lacks the runner labels {HIGH_DENSITY_E2E[0]}"):
        planner.validate_plan(plan, constrained)


@pytest.mark.parametrize(("suite", "labels"), [
    (HIGH_DENSITY_E2E[0], ["other-label"]),
    (HIGH_DENSITY_E2E[0], []),
    ("python:e2e:missing/example", [SOM_16]),
    ("python:unit:harness", [SOM_16]),
])
def test_plan_rejects_unmet_entries_that_do_not_match_the_scope(constrained, suite, labels):
    plan = planner.make_plan(constrained, [runner("runner-a", extra=(SOM_8,))],
                             BASE_LABELS, {}, "abc")
    plan["unmet_runner_labels"][suite] = labels
    with pytest.raises(ValueError, match=f"do not match the scope: {suite}"):
        planner.validate_plan(plan, constrained)


def test_plan_rejects_unmet_labels_that_a_shard_has(plan, constrained):
    plan["unmet_runner_labels"] = {HIGH_DENSITY_E2E[0]: [SOM_16]}
    with pytest.raises(ValueError, match="A shard has the runner labels listed as unmet"):
        planner.validate_plan(plan, constrained)


def test_plan_rejects_unmet_labels_that_one_of_several_shards_has(constrained):
    fleet = [runner("runner-a", extra=(SOM_8,)), runner("runner-b", extra=(SOM_16,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    plan["unmet_runner_labels"] = {HIGH_DENSITY_E2E[0]: [SOM_16]}
    with pytest.raises(ValueError, match="A shard has the runner labels listed as unmet"):
        planner.validate_plan(plan, constrained)


def test_unmet_warning_lists_every_label():
    assert planner.unmet_label_warning("suite", ["label-p", SOM_16], "reason") == (
        "No planned DevKit has the runner labels suite requires "
        f"(label-p, {SOM_16}): reason. The suite runs on any DevKit."
    )


def test_runner_without_a_memory_label_does_not_satisfy_16_gib(constrained):
    plan = planner.make_plan(constrained, [runner("runner-a", extra=("modalix-som",))],
                             BASE_LABELS, {}, "abc")
    assert set(plan["unmet_runner_labels"]) == set(HIGH_DENSITY_E2E)
    fleet = [runner("runner-a", extra=("modalix-som",)), runner("runner-b", extra=(SOM_16,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert {shard_of(plan, t)["candidate"] for t in HIGH_DENSITY_E2E} == {"runner-b"}


@pytest.mark.parametrize(("scope_label", "runner_label"), [
    ("Modalix-SOM-16GIG", SOM_16),
    (SOM_16, "Modalix-SOM-16GIG"),
])
def test_runner_labels_match_without_case(scope, scope_label, runner_label):
    for language in ("cpp", "python"):
        scope["examples"][HIGH_DENSITY]["e2e"][language]["runner_labels"] = [scope_label]
    assert not validate_scope(scope, ROOT)
    fleet = [runner("runner-a", extra=(SOM_8,)), runner("runner-b", extra=(runner_label,))]
    plan = planner.make_plan(scope, fleet, BASE_LABELS, {}, "abc")
    assert {shard_of(plan, t)["candidate"] for t in HIGH_DENSITY_E2E} == {"runner-b"}


def test_suites_with_different_requirements_each_get_a_matching_shard(constrained):
    constrained["examples"][HIGH_DENSITY]["e2e"]["python"]["runner_labels"] = [SOM_8]
    fleet = [runner("runner-a", extra=(SOM_16,)), runner("runner-b", extra=(SOM_8,)),
             runner("runner-c", extra=("modalix-som",))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    assert shard_of(plan, HIGH_DENSITY_E2E[0])["candidate"] == "runner-a"
    assert shard_of(plan, HIGH_DENSITY_E2E[1])["candidate"] == "runner-b"


def test_unit_suites_of_a_constrained_example_may_use_any_shard(constrained):
    weights = dict.fromkeys(HIGH_DENSITY_E2E, 100_000)
    fleet = [runner("runner-a", extra=(SOM_16,)), runner("runner-b", extra=(SOM_8,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, weights, "abc")
    for language in ("cpp", "python"):
        assert shard_of(plan, f"{language}:unit:{HIGH_DENSITY}")["candidate"] == "runner-b"


def test_constrained_suites_are_placed_before_longer_unconstrained_ones(constrained):
    weights = {"python:unit:harness": 100_000}
    fleet = [runner("runner-a", extra=(SOM_16,)), runner("runner-b", extra=(SOM_8,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, weights, "abc")
    assert shard_of(plan, "python:unit:harness")["candidate"] == "runner-b"


def test_plan_rejects_a_constrained_suite_on_a_shard_without_its_labels(constrained):
    fleet = [runner("runner-a", extra=(SOM_16,)), runner("runner-b", extra=(SOM_8,))]
    plan = planner.make_plan(constrained, fleet, BASE_LABELS, {}, "abc")
    source, target = plan["shards"]
    source["tasks"].remove(HIGH_DENSITY_E2E[0])
    target["tasks"].append(HIGH_DENSITY_E2E[0])
    target["needs_media_fixtures"] = True
    with pytest.raises(ValueError, match=f"lacks the runner labels {HIGH_DENSITY_E2E[0]}"):
        planner.validate_plan(plan, constrained)


def test_repository_scope_asks_for_16_gib_devkits_for_high_density_e2e(scope):
    for language in ("cpp", "python"):
        assert scope["examples"][HIGH_DENSITY]["e2e"][language]["runner_labels"] == [SOM_16]
