"""Contract for scripts/ci/package-apps.sh and the files it sources.

With GITHUB_SHA and a branch ref set, as in CI, the archives are the same
whether git can read the checkout or not, for a checkout with no untracked
files under assets/. In a container git can refuse a checkout that another
user owns (a dubious directory), so packaging must not need it.

package-apps.sh names artifacts with sanitize_branch_key and
current_dependency_tag, and install-neat-core.sh selects Core with
current_dependency_tag. A regex check keeps their one definition in
scripts/artifact-refs.sh.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_APPS = APPS_ROOT / "scripts/ci/package-apps.sh"
ARTIFACT_REFS = APPS_ROOT / "scripts/artifact-refs.sh"
VALIDATE_RUNTIME = APPS_ROOT / "scripts/ci/validate_apps_runtime_archive.sh"
APPS_SHA = "0123456789abcdef0123456789abcdef01234567"
CORE_METADATA = json.dumps({"neat-core": {"ref": "r", "spec": "s"}})
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
    shutil.copy2(ARTIFACT_REFS, root / "scripts/artifact-refs.sh")
    _write(root / "deps/manifest.json", json.dumps({"neat-core": ""}) + "\n")
    _write(root / "deps/neat-core.json", CORE_METADATA)
    _write(root / "scripts/download_models.sh", executable=True)
    _write(root / "tests/conftest.py")
    _write(root / "tests/test.sh", executable=True)
    _write(root / "support/README.md")
    _write(root / "support/labels/coco80.txt")
    _write(root / "support/python/sima_examples/helpers.py")
    _write(root / "support/cpp/sima_examples/include/helpers.h")
    _write(root / "support/cpp/sima_examples/testing/fixture.h")
    _write(root / "support/cpp/CMakeLists.txt")
    demo = root / "examples/demo-category/demo"
    _write(demo / "README.md")
    _write(demo / "run.sh", executable=True)
    _write(demo / "src/python/main.py")
    _write(demo / "src/python/model.onnx")
    _write(demo / "src/python/run.log")
    _write(demo / "src/common/labels.txt")
    _write(demo / "src/cpp/main.cpp")
    _write(demo / "src/cpp/CMakeLists.txt")
    _write(demo / "tests/python/test_demo.py")
    _write(demo / "tests/python/.env.local")
    _write(demo / "tests/cpp/demo_unit_test.cpp")
    _write(root / "assets/datasets/coco/sample.jpg")
    _write(root / "assets/datasets-test/coco/test.jpg")

    build = root / "build/examples/demo-category/demo"
    _write(build / "demo", executable=True)
    _write(build / "demo_unit_test", executable=True)
    _write(build / "demo_e2e_test", executable=True)
    # Build-tree files that look like test or example binaries must stay out.
    _write(build / "CMakeFiles/demo.dir/demo_unit_test", executable=True)
    _write(build / "cmTC_0a1b2_unit_test", executable=True)
    _write(build / "stale_e2e_test")
    _write(build / "helper", executable=True)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )


def _run_package_apps(
    root: Path, extra_env: dict[str, str] | None = None, args: tuple[str, ...] = ()
) -> subprocess.CompletedProcess:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GITHUB_", "GIT_", "NEAT_APPS_")) and key != "BUILD_DIR"
    }
    env.update(
        {
            "NEAT_APPS_ARTIFACT_BRANCH_KEY": "contract",
            "GITHUB_SHA": APPS_SHA,
            **(extra_env or {}),
        }
    )
    return subprocess.run(
        ["bash", str(root / "scripts/ci/package-apps.sh"), *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _package(
    root: Path,
    extra_env: dict[str, str] | None = None,
    runtime_archive: str = RUNTIME_ARCHIVE,
    args: tuple[str, ...] = (),
) -> dict[str, dict]:
    proc = _run_package_apps(root, extra_env, args)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    archives = sorted(path.name for path in root.glob("neat-apps-*.tar.gz"))
    assert archives == sorted([runtime_archive, TESTS_ARCHIVE])
    validate = subprocess.run(
        ["bash", str(VALIDATE_RUNTIME), str(root / runtime_archive)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr + validate.stdout
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
    with tarfile.open(root / runtime_archive, "r:gz") as tar:
        deps = {
            name: tar.extractfile(f"prebuilt-apps/deps/{name}").read().decode("utf-8")
            for name in ("neat-apps.json", "neat-core.json")
        }
    apps_metadata = {"neat-apps": {"ref": "contract", "spec": APPS_SHA}}
    assert json.loads(deps["neat-apps.json"]) == apps_metadata
    assert deps["neat-core.json"] == CORE_METADATA
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
        "prebuilt-apps/support/labels/coco80.txt",
        "prebuilt-apps/support/python/sima_examples/helpers.py",
        "prebuilt-apps/support/cpp/sima_examples/include/helpers.h",
        "prebuilt-apps/assets/datasets/coco/sample.jpg",
    ):
        assert member in runtime, member
    assert runtime[f"{demo}/src/cpp/pre-built/demo"][1] & stat.S_IXUSR
    for member in (
        f"{demo}/src/cpp/CMakeLists.txt",
        f"{demo}/src/python/model.onnx",
        f"{demo}/src/python/run.log",
        f"{demo}/tests/python/test_demo.py",
        "prebuilt-apps/support/cpp/CMakeLists.txt",
        "prebuilt-apps/support/cpp/sima_examples/testing/fixture.h",
        "prebuilt-apps/assets/datasets-test/coco/test.jpg",
    ):
        assert member not in runtime, member
    assert not [name for name in runtime if name.endswith(("_unit_test", "_e2e_test"))]
    assert not [name for name in runtime if name.endswith("/helper")]
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
    assert not [name for name in tests if name.endswith(("/.env.local", "/stale_e2e_test"))]
    for archive in (runtime, tests):
        assert not [name for name in archive if "CMakeFiles" in name or "cmTC_" in name]


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
    # Inside the build container git refuses the checkout. A GIT_DIR that
    # points nowhere fails the same rev-parse probe the script relies on.
    extra_env = {"GIT_DIR": str(tmp_path / "missing.git")} if git_state != "repository" else {}

    assert _package(root, extra_env) == packaged_without_git


def test_package_apps_accepts_an_absolute_build_dir(tmp_path, packaged_without_git):
    root = tmp_path / "absolute"
    root.mkdir()
    _fake_repo(root)
    elsewhere = tmp_path / "elsewhere-build"
    (root / "build").rename(elsewhere)

    assert _package(root, args=("--build-dir", str(elsewhere))) == packaged_without_git


def test_package_apps_fails_without_a_build_dir(tmp_path):
    root = tmp_path / "unbuilt"
    root.mkdir()
    _fake_repo(root)

    proc = _run_package_apps(root, args=("--build-dir", "missing"))

    assert proc.returncode == 1
    assert f"build directory not found: {root}/missing" in proc.stderr
    assert not list(root.glob("neat-apps-*.tar.gz"))


def test_package_apps_rejects_a_build_dir_flag_without_a_value(tmp_path):
    root = tmp_path / "no-value"
    root.mkdir()
    _fake_repo(root)

    proc = _run_package_apps(root, args=("--build-dir",))

    assert proc.returncode == 1
    assert "--build-dir needs a directory" in proc.stderr


def test_package_apps_handles_glob_characters_in_the_checkout_path(
    tmp_path, packaged_without_git
):
    root = tmp_path / "check[x]out"
    root.mkdir()
    _fake_repo(root)

    assert _package(root) == packaged_without_git


def test_package_apps_names_a_tag_build_after_the_tag(tmp_path, packaged_without_git):
    root = tmp_path / "tag"
    root.mkdir()
    _fake_repo(root)
    tag_env = {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v1.2.3"}

    packaged = _package(root, tag_env, runtime_archive="neat-apps-contract-1.2.3.tar.gz")

    assert packaged["neat-apps-contract-1.2.3.tar.gz"] == packaged_without_git[RUNTIME_ARCHIVE]


@pytest.mark.parametrize(
    "script", ["scripts/install-neat-core.sh", "scripts/ci/package-apps.sh"]
)
def test_branch_key_and_tag_live_only_in_artifact_refs(script):
    # package-apps.sh uses both functions and install-neat-core.sh uses
    # current_dependency_tag, so both source the one definition.
    text = (APPS_ROOT / script).read_text(encoding="utf-8")
    assert 'source "${ROOT_DIR}/scripts/artifact-refs.sh"' in text
    for function in ("sanitize_branch_key", "current_dependency_tag"):
        definition = re.compile(rf"^\s*(function\s+)?{function}\s*(\(\))?\s*\{{", re.M)
        assert not definition.search(text), f"{script} redefines {function}"
