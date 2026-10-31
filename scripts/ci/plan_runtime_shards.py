#!/usr/bin/env python3
"""Discover schedulable runners and partition every enabled suite exactly once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.utils.test_scope import discover_scope, is_enabled, validate_scope


def collect(get, path, key):
    result = []
    for page in range(1, 1001):
        data = get(f"{path}?per_page=100&page={page}")[key]
        if not isinstance(data, list):
            raise TypeError(f"Invalid GitHub inventory: {key}")
        result.extend(data)
        if len(data) < 100:
            return result
    raise ValueError("GitHub inventory pagination exceeded limit")


def discover_runners(get, repository, group_name, private):
    base = f"/orgs/{quote(repository.split('/')[0], safe='')}/actions/runner-groups"
    groups = [g for g in collect(get, base, "runner_groups") if g["name"] == group_name]
    if len(groups) != 1:
        raise ValueError(f"Expected one accessible runner group: {group_name}")
    group = groups[0]
    if not private and not group.get("allows_public_repositories", False):
        raise ValueError("Runner group does not allow public repositories")
    if group["visibility"] == "selected":
        repos = collect(get, f"{base}/{group['id']}/repositories", "repositories")
        if not any(r["full_name"].casefold() == repository.casefold() for r in repos):
            raise ValueError("Repository has no access to the runner group")
    if group.get("restricted_to_workflows"):
        raise ValueError(
            "Workflow-restricted runner groups require explicit workflow admission"
        )
    return collect(get, f"{base}/{group['id']}/runners", "runners")


def label_set(runner):
    return {label["name"].casefold() for label in runner["labels"]}


def select_fleet(runners, required, maximum, pool, excluded_names=()):
    required = {label.casefold() for label in required}
    excluded_names = {name.casefold() for name in excluded_names}
    excluded = [r for r in runners if r["name"].casefold() in excluded_names]
    # Consider offline registrations too when proving a selector excludes RTSP.
    rtsp = [r for r in runners if "rtsp" in label_set(r)]
    eligible = [
        r
        for r in runners
        if r["status"] == "online"
        and required <= label_set(r)
        and ("rtsp" in label_set(r)) == (pool == "rtsp")
    ]
    if not eligible:
        raise ValueError(
            f"No online {pool} runner matches the required platform/environment"
        )
    safe = []
    for runner in eligible:
        labels = required | label_set(runner)
        # runs-on has no negative runner-name filter. Prove the complete label
        # conjunction cannot be claimed by an unready registration either.
        if any(labels <= label_set(r) for r in excluded):
            continue
        if pool == "regular" and any(labels <= label_set(r) for r in rtsp):
            continue
        safe.append(runner)
    if not safe:
        raise ValueError(
            "No safe label selector excluding RTSP or unready runners; "
            "configure a dedicated ready-runner label/group"
        )
    eligible = safe
    idle = sorted((r for r in eligible if not r["busy"]), key=lambda r: r["name"])
    # If all are busy, queue one complete shard rather than drop that pool's tests.
    selected = idle[:maximum] or [min(eligible, key=lambda r: r["name"])]
    fleet = []
    for runner in selected:
        labels = required | label_set(runner)
        if pool == "regular" and any(labels <= label_set(r) for r in rtsp):
            raise ValueError(
                "Regular runners need a label combination excluding RTSP runners; "
                "configure a dedicated regular label or runner group"
            )
        fleet.append(
            {
                "labels": sorted(labels),
                "candidate": runner["name"],
                "idle_at_planning": not runner["busy"],
            }
        )
    return fleet


def enumerate_suites(scope, weights):
    suites = [
        {
            "id": "python:unit:harness",
            "pool": "regular",
            "seconds": weights.get("python:unit:harness", 30),
        }
    ]
    for example, entry in sorted(scope["examples"].items()):
        for language in ("cpp", "python"):
            for kind in ("unit", "e2e"):
                if not is_enabled(entry, language, kind):
                    continue
                identifier = f"{language}:{kind}:{example}"
                pool = (
                    "rtsp"
                    if kind == "e2e"
                    and entry["e2e"][language].get("requires_rtsp", True)
                    else "regular"
                )
                suites.append(
                    {
                        "id": identifier,
                        "pool": pool,
                        "seconds": weights.get(
                            identifier, 180 if kind == "e2e" else 30
                        ),
                    }
                )
    return suites


def make_plan(
    scope,
    runners,
    required,
    weights,
    sha,
    rtsp_limit=6,
    regular_limit=3,
    excluded_names=(),
):
    if not 1 <= rtsp_limit <= 6 or not 1 <= regular_limit <= 3:
        raise ValueError("Fleet limits must be 1..6 RTSP and 1..3 regular")
    suites = enumerate_suites(scope, weights)
    shards = []
    for pool, maximum in (("rtsp", rtsp_limit), ("regular", regular_limit)):
        tasks = [s for s in suites if s["pool"] == pool]
        if not tasks:
            continue
        fleet = select_fleet(
            runners, required, min(maximum, len(tasks)), pool, excluded_names
        )
        bins = [
            {
                "id": f"{pool}-{i + 1}",
                "pool": pool,
                **runner,
                "tasks": [],
                "estimated_seconds": 0,
            }
            for i, runner in enumerate(fleet)
        ]
        for task in sorted(tasks, key=lambda s: (-s["seconds"], s["id"])):
            shard = min(bins, key=lambda b: (b["estimated_seconds"], b["id"]))
            shard["tasks"].append(task["id"])
            shard["estimated_seconds"] += task["seconds"]
        shards.extend(bins)
    plan = {"sha": sha, "expected": sorted(s["id"] for s in suites), "shards": shards}
    validate_plan(plan, scope)
    return plan


def validate_plan(plan, scope):
    expected = sorted(s["id"] for s in enumerate_suites(scope, {}))
    assigned = [task for shard in plan["shards"] for task in shard["tasks"]]
    if plan["expected"] != expected or sorted(assigned) != expected:
        raise ValueError(
            "Shard coverage does not match every enabled suite exactly once"
        )
    ids = [shard["id"] for shard in plan["shards"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate shard IDs")
    pools = {s["id"]: s["pool"] for s in enumerate_suites(scope, {})}
    for pool, maximum in (("rtsp", 6), ("regular", 3)):
        if sum(s["pool"] == pool for s in plan["shards"]) > maximum:
            raise ValueError("Shard plan exceeds fleet cap")
    for shard in plan["shards"]:
        if not shard["tasks"] or any(pools[t] != shard["pool"] for t in shard["tasks"]):
            raise ValueError("Suite assigned to the wrong runner pool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--labels", required=True, help="JSON array of required runner labels"
    )
    parser.add_argument("--output", type=Path, default=Path("runtime-shard-plan.json"))
    args = parser.parse_args()
    token = os.environ.get("RUNNER_DISCOVERY_TOKEN")
    if not token:
        raise ValueError(
            "Runner discovery requires the existing App's self-hosted-runners read token"
        )

    def get(path):
        request = Request(
            os.environ.get("GITHUB_API_URL", "https://api.github.com") + path,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    scope = discover_scope(ROOT / "examples", ROOT)
    errors = validate_scope(scope, ROOT)
    if errors:
        raise ValueError("; ".join(errors))
    runners = discover_runners(
        get,
        os.environ["GITHUB_REPOSITORY"],
        args.group,
        os.environ.get("REPOSITORY_PRIVATE", "true") == "true",
    )
    labels = json.loads(args.labels)
    weights = json.loads((ROOT / "tests/ci-durations.json").read_text())[
        "suite_seconds"
    ]
    excluded = json.loads(os.environ.get("APPS_UNREADY_RUNNERS") or "[]")
    if not isinstance(excluded, list) or any(
        not isinstance(name, str) for name in excluded
    ):
        raise ValueError("APPS_UNREADY_RUNNERS must be a JSON array of runner names")
    plan = make_plan(
        scope,
        runners,
        labels,
        weights,
        os.environ["GITHUB_SHA"],
        excluded_names=excluded,
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    matrix = {
        "include": [{"shard": s["id"], "labels": s["labels"]} for s in plan["shards"]]
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"matrix={json.dumps(matrix)}\n")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(
            "| Shard | Idle at planning | Estimated tests | Suites |\n|---|---|---:|---:|\n"
        )
        summary.writelines(
            f"| {shard['id']} | {shard['idle_at_planning']} | "
            f"{shard['estimated_seconds'] / 60:.1f} min | {len(shard['tasks'])} |\n"
            for shard in plan["shards"]
        )
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
