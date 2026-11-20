"""Contract for scripts/ci/package-apps.sh: the bundles do not depend on git.

Inside the Vulcan build container git cannot read the checkout (it is owned by
another user, so git refuses it as a dubious directory). Packaging must then
produce the same archives as on a host where git works.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_APPS = APPS_ROOT / "scripts/ci/package-apps.sh"
APPS_SHA = "0123456789abcdef0123456789abcdef01234567"
RUNTIME_ARCHIVE = "neat-apps-contract-0123456.tar.gz"
TESTS_ARCHIVE = "neat-apps-tests.tar.gz"


def _write(path: Path, text: str = "fixture\n", executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _fake_repo(root: Path) -> None:
    (root / "scripts/ci").mkdir(parents=True)
    shutil.copy2(PACKAGE_APPS, root / "scripts/ci/package-apps.sh")
    _write(root / "deps/manifest.json", json.dumps({"neat-core": ""}) + "\n")
    _write(root / "deps/neat-core.json", json.dumps({"neat-core": {"ref": "r", "spec": "s"}}))
    _write(root / "scripts/download_models.sh", executable=True)
    _write(root / "tests/conftest.py")
    _write(root / "tests/test.sh", executable=True)
    _write(root / "support/README.md")
    _write(root / "support/python/sima_examples/helpers.py")
    _write(root / "support/cpp/sima_examples/include/helpers.h")
    _write(root / "support/cpp/sima_examples/testing/fixture.h")
    demo = root / "examples/demo-category/demo"
    _write(demo / "README.md")
    _write(demo / "run.sh", executable=True)
    _write(demo / "src/python/main.py")
    _write(demo / "src/common/labels.txt")
    _write(demo / "src/cpp/main.cpp")
    _write(demo / "src/cpp/CMakeLists.txt")
    _write(demo / "tests/python/test_demo.py")
    _write(demo / "tests/cpp/demo_unit_test.cpp")
    _write(root / "assets/datasets/coco/sample.jpg")
    _write(root / "assets/datasets-test/coco/test.jpg")

    build = root / "build/examples/demo-category/demo"
    _write(build / "demo", executable=True)
    _write(build / "demo_unit_test", executable=True)
    _write(build / "demo_e2e_test", executable=True)
    _write(build / "CMakeFiles/demo.dir/link.txt", executable=True)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )


def _package(root: Path, extra_env: dict[str, str] | None = None) -> dict[str, dict]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GITHUB_", "GIT_", "NEAT_APPS_"))
    }
    env.update(
        {
            "NEAT_APPS_ARTIFACT_BRANCH_KEY": "contract",
            "GITHUB_SHA": APPS_SHA,
            **(extra_env or {}),
        }
    )
    proc = subprocess.run(
        ["bash", str(root / "scripts/ci/package-apps.sh")],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    archives = sorted(path.name for path in root.glob("neat-apps-*.tar.gz"))
    assert archives == sorted([RUNTIME_ARCHIVE, TESTS_ARCHIVE])
    contents = {}
    for name in archives:
        with tarfile.open(root / name, "r:gz") as tar:
            contents[name] = {
                member.name: (
                    member.type,
                    member.mode,
                    hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                    if member.isfile()
                    else None,
                )
                for member in tar.getmembers()
            }
    return contents


@pytest.fixture
def packaged_without_git(tmp_path):
    root = tmp_path / "no-git"
    root.mkdir()
    _fake_repo(root)
    return _package(root)


def test_package_apps_ships_runtime_and_test_layout(packaged_without_git):
    runtime = packaged_without_git[RUNTIME_ARCHIVE]
    tests = packaged_without_git[TESTS_ARCHIVE]
    demo = "prebuilt-apps/examples/demo-category/demo"
    for member in (
        f"{demo}/README.md",
        f"{demo}/run.sh",
        f"{demo}/src/python/main.py",
        f"{demo}/src/common/labels.txt",
        f"{demo}/src/cpp/main.cpp",
        f"{demo}/src/cpp/pre-built/demo",
        "prebuilt-apps/deps/neat-core.json",
        "prebuilt-apps/deps/neat-apps.json",
        "prebuilt-apps/support/README.md",
        "prebuilt-apps/support/python/sima_examples/helpers.py",
        "prebuilt-apps/support/cpp/sima_examples/include/helpers.h",
        "prebuilt-apps/assets/datasets/coco/sample.jpg",
    ):
        assert member in runtime, member
    for member in (
        f"{demo}/src/cpp/CMakeLists.txt",
        f"{demo}/tests/python/test_demo.py",
        "prebuilt-apps/support/cpp/sima_examples/testing/fixture.h",
        "prebuilt-apps/assets/datasets-test/coco/test.jpg",
    ):
        assert member not in runtime, member
    test_demo = "neat-apps-tests/examples/demo-category/demo/tests"
    for member in (
        f"{test_demo}/python/test_demo.py",
        f"{test_demo}/cpp/demo_unit_test",
        f"{test_demo}/cpp/demo_e2e_test",
        "neat-apps-tests/tests/test.sh",
        "neat-apps-tests/examples/conftest.py",
        "neat-apps-tests/deps/manifest.json",
        "neat-apps-tests/scripts/download_models.sh",
        "neat-apps-tests/assets/datasets-test/coco/test.jpg",
    ):
        assert member in tests, member


@pytest.mark.parametrize("git_state", ["repository", "unreadable-repository"])
def test_package_apps_output_does_not_depend_on_git(
    tmp_path, packaged_without_git, git_state
):
    root = tmp_path / git_state
    root.mkdir()
    _fake_repo(root)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture")
    # Inside the build container git refuses the checkout; a GIT_DIR that
    # points nowhere fails the same rev-parse probe the script relies on.
    extra_env = {"GIT_DIR": str(tmp_path / "missing.git")} if git_state != "repository" else {}

    assert _package(root, extra_env) == packaged_without_git
