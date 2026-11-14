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

MAX_SHARDS = 8


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


def select_fleet(runners, required, maximum, excluded_names=()):
    required = {label.casefold() for label in required}
    if "rtsp" in required:
        raise ValueError("Base runner selector must not require the legacy RTSP label")
    excluded_names = {name.casefold() for name in excluded_names}
    excluded = [r for r in runners if r["name"].casefold() in excluded_names]
    eligible = [
        r for r in runners if r["status"] == "online" and required <= label_set(r)
    ]
    if not eligible:
        raise ValueError("No online runner matches the required platform/environment")

    safe = []
    for runner in eligible:
        labels = label_set(runner) - {"rtsp"}
        # GitHub runs-on has no negative runner-name filter. Include offline
        # registrations in the check because they can come back while jobs queue.
        if any(labels <= label_set(r) for r in excluded):
            continue
        safe.append((runner, sorted(labels)))
    if not safe:
        raise ValueError(
            "No safe label selector excluding unready runners; "
            "configure a dedicated ready-runner label/group"
        )

    safe.sort(key=lambda candidate: candidate[0]["name"])
    idle = [candidate for candidate in safe if not candidate[0]["busy"]]
    # If all eligible runners are busy, queue one shard with all suites.
    selected = idle[:maximum] or safe[:1]
    return [
        {
            "labels": labels,
            "candidate": runner["name"],
            "idle_at_planning": not runner["busy"],
        }
        for runner, labels in selected
    ]


def enumerate_suites(scope, weights):
    suites = [
        {
            "id": "python:unit:harness",
            "needs_media_fixtures": False,
            "seconds": weights.get("python:unit:harness", 30),
        }
    ]
    for example, entry in sorted(scope["examples"].items()):
        for language in ("cpp", "python"):
            for kind in ("unit", "e2e"):
                if not is_enabled(entry, language, kind):
                    continue
                identifier = f"{language}:{kind}:{example}"
                suites.append(
                    {
                        "id": identifier,
                        "needs_media_fixtures": kind == "e2e"
                        and entry["e2e"][language].get("requires_rtsp", True),
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
    excluded_names=(),
):
    suites = enumerate_suites(scope, weights)
    fleet = select_fleet(runners, required, min(MAX_SHARDS, len(suites)), excluded_names)
    shards = [
        {
            "id": f"devkit-{i + 1}",
            **runner,
            "tasks": [],
            "estimated_seconds": 0,
            "needs_media_fixtures": False,
        }
        for i, runner in enumerate(fleet)
    ]
    for task in sorted(suites, key=lambda s: (-s["seconds"], s["id"])):
        shard = min(shards, key=lambda s: (s["estimated_seconds"], s["id"]))
        shard["tasks"].append(task["id"])
        shard["estimated_seconds"] += task["seconds"]
        shard["needs_media_fixtures"] |= task["needs_media_fixtures"]
    plan = {"sha": sha, "expected": sorted(s["id"] for s in suites), "shards": shards}
    validate_plan(plan, scope)
    return plan


def validate_plan(plan, scope):
    suites = {s["id"]: s["needs_media_fixtures"] for s in enumerate_suites(scope, {})}
    expected = sorted(suites)
    assigned = [task for shard in plan["shards"] for task in shard["tasks"]]
    if plan["expected"] != expected or sorted(assigned) != expected:
        raise ValueError(
            "Shard coverage does not match every enabled suite exactly once"
        )
    ids = [shard["id"] for shard in plan["shards"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate shard IDs")
    if not 1 <= len(ids) <= MAX_SHARDS:
        raise ValueError("Shard plan exceeds fleet cap")
    for shard in plan["shards"]:
        if not shard["tasks"] or not shard["labels"]:
            raise ValueError("Empty shard or runner selector")
        if any(label.casefold() == "rtsp" for label in shard["labels"]):
            raise ValueError("Shard selectors must not require the legacy RTSP label")
        if shard["needs_media_fixtures"] != any(suites[t] for t in shard["tasks"]):
            raise ValueError("Shard media fixture setup does not match its assigned suites")


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
        "include": [
            {
                "shard": s["id"],
                "labels": s["labels"],
                "needs_media_fixtures": s["needs_media_fixtures"],
            }
            for s in plan["shards"]
        ]
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
