#!/usr/bin/env python3
"""Run a validated shard through the existing strict Apps test harness."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml
from cache_test_models import prepare_models
from plan_runtime_shards import ROOT, validate_plan

from tests.utils.test_scope import discover_scope, validate_scope


def scope_for_tasks(scope, tasks):
    selected = copy.deepcopy(scope)
    wanted = set(tasks)
    for example, entry in selected["examples"].items():
        for language in ("cpp", "python"):
            entry.setdefault("unit", {})[language] = (
                f"{language}:unit:{example}" in wanted
            )
            entry.setdefault("e2e", {}).setdefault(language, {})["enabled"] = (
                f"{language}:e2e:{example}" in wanted
            )
    return selected


def collect_logs(runtime, destination):
    """Stage regular log files without traversing intentional symlink fixtures."""
    source = runtime / "sandbox-test"
    for directory, directories, files in os.walk(source, followlinks=False):
        directory = Path(directory)
        directories[:] = [
            name for name in directories if not (directory / name).is_symlink()
        ]
        for name in files:
            path = directory / name
            if path.suffix != ".log" or path.is_symlink() or not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def run(plan, shard_id, runtime, result_path, cache_root):
    started = time.monotonic()
    full_scope = discover_scope(ROOT / "examples", ROOT)
    validate_plan(plan, full_scope)
    if plan["sha"] != os.environ["GITHUB_SHA"]:
        raise ValueError("Shard plan does not match the tested commit")
    shard = next(s for s in plan["shards"] if s["id"] == shard_id)
    scope = scope_for_tasks(full_scope, shard["tasks"])
    errors = validate_scope(scope, runtime)
    if errors:
        raise ValueError("; ".join(errors))
    result = {
        "sha": plan["sha"],
        "shard": shard_id,
        "runner": os.environ.get("RUNNER_NAME"),
        "tasks": [],
        "success": False,
        "groups": [],
        "dependencies": {
            name: json.loads((runtime / "deps" / f"{name}.json").read_text())
            for name in ("neat-core", "neat-apps")
            if (runtime / "deps" / f"{name}.json").is_file()
        },
    }
    try:
        scope_file = runtime / "tests/ci-shard-scope.yaml"
        scope_file.write_text(yaml.safe_dump(scope))
        result["cache"] = prepare_models(
            runtime, scope, scope_file, runtime / "models", cache_root
        )
        # Keep the harness in exactly one shard, independent of whether that
        # shard also contains Python example unit tests.
        for language, kind in (
            ("cpp", "unit"),
            ("cpp", "e2e"),
            ("python", "unit"),
            ("python", "e2e"),
        ):
            tasks = [t for t in shard["tasks"] if t.startswith(f"{language}:{kind}:")]
            if not tasks:
                continue
            group_scope = scope_for_tasks(full_scope, tasks)
            scope_file.write_text(yaml.safe_dump(group_scope))
            group_started = time.monotonic()
            subprocess.run(
                ["bash", "tests/test.sh", f"--{language}", f"--{kind}", "--strict"],
                cwd=runtime,
                check=True,
                env={
                    **os.environ,
                    "SIMANEAT_APPS_TEST_SCOPE_FILE": str(scope_file),
                    "SIMANEAT_APPS_TEST_MODELS_DIR": str(runtime / "models"),
                    "NEAT_APPS_SKIP_MODEL_DOWNLOAD": "1",
                    "SIMANEAT_APPS_TEST_RUN_HARNESS": str(
                        int("python:unit:harness" in tasks)
                    ),
                },
            )
            result["groups"].append(
                {
                    "language": language,
                    "kind": kind,
                    "seconds": time.monotonic() - group_started,
                }
            )
            result["tasks"].extend(tasks)
        result["success"] = True
    finally:
        result["seconds"] = time.monotonic() - started
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        collect_logs(runtime, result_path.parent / "logs" / shard_id)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as output:
                output.write(
                    f"### {shard_id}\n\nRunner: `{result['runner']}`. "
                    f"Completed {len(result['tasks'])}/{len(shard['tasks'])} suites "
                    f"in {result['seconds'] / 60:.1f} minutes.\n"
                )
                if "cache" in result:
                    cache = result["cache"]
                    output.write(
                        f"Model cache: {cache['hits']}/{cache['models']} hits; "
                        f"{cache['seconds']:.1f} seconds acquiring models.\n"
                    )


def verify_results(plan, results):
    if not results or len(results) != len(plan["shards"]):
        raise ValueError("Missing shard results")
    by_id = {r["shard"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {s["id"] for s in plan["shards"]}:
        raise ValueError("Duplicate or unexpected shard results")
    for shard in plan["shards"]:
        result = by_id[shard["id"]]
        if not result["success"] or result["sha"] != plan["sha"]:
            raise ValueError(f"Shard failed or used a different commit: {shard['id']}")
        if sorted(result["tasks"]) != sorted(shard["tasks"]):
            raise ValueError(f"Incomplete suite execution: {shard['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--shard")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/media/nvme/kerrigan/neat-apps-model-cache/v1"),
    )
    parser.add_argument("--verify-results", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.verify_results:
        validate_plan(plan, discover_scope(ROOT / "examples", ROOT))
        if plan["sha"] != os.environ["GITHUB_SHA"]:
            raise ValueError("Result plan does not match this commit")
        results = [
            json.loads(p.read_text()) for p in args.verify_results.rglob("*.json")
        ]
        verify_results(plan, results)
        print(
            f"All {len(results)} shards passed; {len(plan['expected'])} suites accounted for"
        )
    else:
        if not args.shard or not args.runtime or not args.result:
            parser.error("--shard, --runtime and --result are required for execution")
        subprocess.run(["mountpoint", "-q", "/media/nvme"], check=True)
        run(plan, args.shard, args.runtime.resolve(), args.result, args.cache_root)


if __name__ == "__main__":
    main()
