"""Regression tests for the apps dependency manifest contract."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import tarfile
import textwrap
import time
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
BUILD_SH = APPS_ROOT / "build.sh"
ARCHIVE_INSTALLER = APPS_ROOT / "scripts/install-neat-apps.sh"
VULCAN_INSTALLER = APPS_ROOT / "scripts/install-vulcan-apps-package.sh"
VULCAN_WORKFLOW = APPS_ROOT / ".github/workflows/vulcan-ci.yml"
RELEASE_WORKFLOW = APPS_ROOT / ".github/workflows/release.yml"
RUNTIME_ARCHIVE_VALIDATOR = APPS_ROOT / "scripts/ci/validate_apps_runtime_archive.sh"


def _write_runtime_archive(
    tmp_path: Path,
    *members: str,
    root_name: str = "prebuilt-apps",
    executable_members: tuple[str, ...] = (),
    extra_archive_members: tuple[str, ...] = (),
    member_contents: dict[str, str] | None = None,
) -> Path:
    archive_root = tmp_path / "archive-root" / root_name
    archive_root.mkdir(parents=True)
    for member in members:
        path = archive_root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (member_contents or {}).get(member, "fixture\n")
        path.write_text(content, encoding="utf-8")
        if member in executable_members:
            path.chmod(0o755)

    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(archive_root, arcname=root_name)
        for member in extra_archive_members:
            payload = b"fixture\n"
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            output.addfile(info, io.BytesIO(payload))
    return archive


def _validate_runtime_archive(archive: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNTIME_ARCHIVE_VALIDATOR), str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_runtime_archive_validator_accepts_shipped_datasets(tmp_path):
    archive = _write_runtime_archive(tmp_path, "assets/datasets/coco/example.jpg")
    proc = _validate_runtime_archive(archive)

    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_runtime_archive_validator_rejects_legacy_root(tmp_path):
    archive = _write_runtime_archive(tmp_path, root_name="neat-apps-runtime")
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


@pytest.mark.parametrize("member", ["unexpected.txt", "../escape.txt"])
def test_runtime_archive_validator_rejects_members_outside_root(tmp_path, member):
    archive = _write_runtime_archive(
        tmp_path,
        extra_archive_members=(member,),
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


@pytest.mark.parametrize(
    "member",
    [
        "manifest.json",
        "neat-core.json",
        "assets/datasets-test/coco/example.jpg",
        "models/example.tar.gz",
        "examples/classification/demo/src/common/model.tar.gz",
        "portal/assets/examples/example.png",
        "examples/classification/demo/src/cpp/CMakeLists.txt",
        "examples/classification/demo/src/cpp/pre-built/CTestTestfile.cmake",
    ],
)
def test_runtime_archive_validator_rejects_forbidden_content(tmp_path, member):
    archive = _write_runtime_archive(tmp_path, member)
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


@pytest.mark.parametrize(
    "reference",
    [
        "assets/models/",
        "assets/test_images/",
        "assets/test_images_classification/",
        "neat-apps-runtime/",
        "/usr/bin/fix_devkit_runtime.sh",
    ],
)
def test_runtime_archive_validator_rejects_obsolete_references(tmp_path, reference):
    readme = "examples/classification/demo/README.md"
    archive = _write_runtime_archive(
        tmp_path,
        readme,
        member_contents={readme: reference},
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


def test_runtime_archive_validator_accepts_cpp_reference_and_prebuilt_binary(tmp_path):
    prebuilt_binary = "examples/classification/demo/src/cpp/pre-built/demo"
    archive = _write_runtime_archive(
        tmp_path,
        "examples/classification/demo/src/cpp/main.cpp",
        prebuilt_binary,
        executable_members=(prebuilt_binary,),
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode == 0, proc.stderr + proc.stdout


@pytest.mark.parametrize(
    "misplaced_binary",
    [
        "examples/classification/demo/src/cpp/demo",
        "examples/classification/demo/demo",
        "examples/classification/demo/helper",
        "examples/face-detection/face-detector_cpp/face-detector",
    ],
)
def test_runtime_archive_validator_rejects_misplaced_cpp_executable(
    tmp_path, misplaced_binary
):
    archive = _write_runtime_archive(
        tmp_path,
        misplaced_binary,
        executable_members=(misplaced_binary,),
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


def test_runtime_archive_validator_accepts_example_root_scripts(tmp_path):
    scripts = (
        "examples/genai/multimodal-assistant/run.sh",
        "examples/genai/multimodal-assistant/setup.sh",
    )
    archive = _write_runtime_archive(
        tmp_path,
        *scripts,
        executable_members=scripts,
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode == 0, proc.stderr + proc.stdout


def _write_manifest(
    path: Path,
    neat_core: object = "",
    platform_version: str = "2.0.0",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "neat-core": neat_core,
                "platform-version": platform_version,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_fake_curl(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log_path = tmp_path / "curl.log"
    curl_path = bin_dir / "curl"
    curl_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            from __future__ import annotations

            import os
            import sys
            from pathlib import Path
            from urllib.parse import urlparse

            args = sys.argv[1:]
            output = None
            url = None
            index = 0
            while index < len(args):
                arg = args[index]
                if arg == "-o":
                    output = args[index + 1]
                    index += 2
                elif arg.startswith("-"):
                    index += 1
                else:
                    url = arg
                    index += 1

            if not url:
                sys.exit(2)

            log = os.environ.get("NEAT_APPS_TEST_CURL_LOG")
            if log:
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(url + "\\n")

            installer_url = os.environ.get("NEAT_APPS_TEST_INSTALLER_URL", "")
            if installer_url and url == installer_url:
                body = '''#!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$PWD" > "${NEAT_APPS_TEST_INSTALLER_PWD}"
            printf '%s\\n' "$*" > "${NEAT_APPS_TEST_INSTALLER_ARGS}"
            touch sima-neat-test-Linux-core.deb neat-test.deb pyneat-test.whl
            '''
                if output:
                    Path(output).write_text(body, encoding="utf-8")
                else:
                    print(body, end="")
                sys.exit(0)

            tags: dict[str, str] = {}
            for line in os.environ.get("NEAT_APPS_TEST_LATEST_TAGS", "").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    tags[key] = value

            metadata = set(
                line.strip()
                for line in os.environ.get("NEAT_APPS_TEST_METADATA", "").splitlines()
                if line.strip()
            )

            parts = [part for part in urlparse(url).path.split("/") if part]
            body = None
            if len(parts) >= 2 and parts[-1] == "latest.tag":
                body = tags.get(parts[-2])
            elif len(parts) >= 3 and parts[-1] == "metadata.json":
                branch = parts[-3]
                version = parts[-2]
                if f"{branch}:{version}" in metadata:
                    body = "{}"

            if body is None:
                sys.exit(22)

            if output:
                Path(output).write_text(body, encoding="utf-8")
            else:
                print(body)
            sys.exit(0)
            """
        ),
        encoding="utf-8",
    )
    curl_path.chmod(0o755)
    return bin_dir, log_path


def test_archive_installer_extracts_flat_prebuilt_apps_by_default(tmp_path):
    archive = _write_runtime_archive(tmp_path, "runtime-marker")
    bin_dir = _write_fake_sima_cli(tmp_path)
    env = os.environ.copy()
    env.pop("NEAT_APPS_INSTALL_DIR", None)
    env.update(
        {
            "NEAT_APPS_ARCHIVE": str(archive),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        }
    )

    proc = subprocess.run(
        ["bash", str(ARCHIVE_INSTALLER), "develop"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "prebuilt-apps" / "runtime-marker").is_file()
    assert not (tmp_path / "neat-apps").exists()
    assert (tmp_path / "sima-cli-args.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "neat install -d . -t minimal core",
        "neat install -d . insight",
    ]
    assert "  ./prebuilt-apps" in proc.stdout


def _write_fake_sima_cli(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    sima_cli_path = bin_dir / "sima-cli"
    sima_cli_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$PWD" >> "${NEAT_APPS_TEST_SIMA_CLI_CWD}"
            printf '%s\\n' "$*" >> "${NEAT_APPS_TEST_SIMA_CLI_ARGS}"
            if [[ -n "${NEAT_APPS_TEST_FORBIDDEN_PATH:-}" ]]; then
                [[ ! -e "${NEAT_APPS_TEST_FORBIDDEN_PATH}" ]]
            fi
            if [[ -n "${NEAT_APPS_TEST_SIMA_CLI_READY:-}" ]]; then
                touch "${NEAT_APPS_TEST_SIMA_CLI_READY}"
                while true; do sleep 1; done
            fi
            touch sima-cli-ran.txt
            if [[ -n "${NEAT_APPS_TEST_SIMA_CLI_FAIL_TARGET:-}" \
                && "$*" == *"${NEAT_APPS_TEST_SIMA_CLI_FAIL_TARGET}"* ]]; then
                exit 1
            fi
            exit "${NEAT_APPS_TEST_SIMA_CLI_STATUS:-0}"
            """
        ),
        encoding="utf-8",
    )
    sima_cli_path.chmod(0o755)
    return bin_dir


def _write_vulcan_runtime(package_dir: Path) -> Path:
    runtime_dir = package_dir / "prebuilt-apps"
    runtime_dir.mkdir(parents=True)
    return runtime_dir


def _run_vulcan_installer(
    tmp_path: Path,
    package_dir: Path,
    *,
    install_dir: Path | None = None,
    sima_cli_status: int = 0,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _write_fake_sima_cli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_STATUS": str(sima_cli_status),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        }
    )
    if install_dir is not None:
        env["NEAT_APPS_INSTALL_DIR"] = str(install_dir)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(VULCAN_INSTALLER)],
        cwd=package_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_build(
    tmp_path: Path,
    *,
    neat_core: object = "",
    platform_version: str = "2.0.0",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    deps_dir = tmp_path / "deps"
    manifest_path = deps_dir / "manifest.json"
    _write_manifest(manifest_path, neat_core, platform_version)

    bin_dir, log_path = _write_fake_curl(tmp_path)
    build_dir = tmp_path / "build"
    run_env = os.environ.copy()
    for name in (
        "GITHUB_BASE_REF",
        "GITHUB_HEAD_REF",
        "GITHUB_REF_NAME",
        "GITHUB_REF_TYPE",
        "NEAT_APPS_ARTIFACT_BRANCH_KEY",
        "NEAT_APPS_ARTIFACT_SHORT_SHA",
        "NEAT_APPS_DEPENDENCY_BRANCH",
    ):
        run_env.pop(name, None)

    run_env.update(
        {
            "PATH": f"{bin_dir}:{run_env['PATH']}",
            "NEAT_APPS_TEST_DEPS_DIR": str(deps_dir),
            "NEAT_ARTIFACTS_BASE_URL": "https://core.test",
            "NEAT_APPS_TEST_CURL_LOG": str(log_path),
            "NEAT_APPS_TEST_LATEST_TAGS": "\n".join(
                [
                    "develop=devsha1",
                    "main=mainsha1",
                    "feature%252Fcore-artifact=featsha2",
                    "zz-core-artifact-for-test=featsha1",
                    "scratch-core-for-test=scratchsha1",
                ]
            ),
            "NEAT_APPS_TEST_METADATA": "\n".join(
                [
                    "main:mainsha1",
                    "feature%252Fcore-artifact:pinnedsha2",
                    "zz-core-artifact-for-test:pinnedsha1",
                ]
            ),
        }
    )
    if env:
        run_env.update(env)

    command = [str(BUILD_SH)] + (args or ["--no-cpp", "--build-dir", str(build_dir)])
    return subprocess.run(
        command,
        cwd=APPS_ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _installer_env(tmp_path: Path) -> dict[str, str]:
    return {
        "NEAT_INSTALLER_URL": "https://installer.test/install-neat.sh",
        "NEAT_APPS_TEST_INSTALLER_URL": "https://installer.test/install-neat.sh",
        "NEAT_APPS_TEST_INSTALLER_PWD": str(tmp_path / "installer-pwd.txt"),
        "NEAT_APPS_TEST_INSTALLER_ARGS": str(tmp_path / "installer-args.txt"),
    }


def _installer_args(tmp_path: Path) -> str:
    return (tmp_path / "installer-args.txt").read_text(encoding="utf-8").strip()


def _curl_log(tmp_path: Path) -> list[str]:
    path = tmp_path / "curl.log"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _write_fake_neat_json(tmp_path: Path, *, channel: str, tag: str, env: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    neat_path = bin_dir / "neat"
    payload = {
        "components": {
            "core": {
                "channel": channel,
                "tag": tag,
                "provenance": {"vulcanEnvironment": env},
            }
        }
    }
    neat_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            if [[ "${{1:-}}" == "--json" ]]; then
              cat <<'EOF'
            {json.dumps(payload)}
            EOF
              exit 0
            fi
            exit 1
            """
        ),
        encoding="utf-8",
    )
    neat_path.chmod(0o755)
    return bin_dir


def test_empty_manifest_resolves_from_dependency_branch_and_platform_version(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={**_installer_env(tmp_path), "NEAT_APPS_DEPENDENCY_BRANCH": "develop"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum develop devsha1"
    assert _curl_log(tmp_path).count("https://core.test/develop/latest.tag") == 1


def test_snap_manifest_resolves_from_dependency_branch(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "snap"},
        args=["--only-install-neat-core"],
        env={**_installer_env(tmp_path), "NEAT_APPS_DEPENDENCY_BRANCH": "develop"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum develop devsha1"
    assert _curl_log(tmp_path).count("https://core.test/develop/latest.tag") == 1


def test_empty_manifest_custom_branch_uses_matching_core_artifact_once(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "NEAT_APPS_DEPENDENCY_BRANCH": "zz-core-artifact-for-test",
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum zz-core-artifact-for-test featsha1"
    assert (
        _curl_log(tmp_path).count(
            "https://core.test/zz-core-artifact-for-test/latest.tag"
        )
        == 1
    )


def test_empty_manifest_custom_branch_falls_back_to_develop(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test",
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum develop devsha1"
    assert "using develop-latest" in proc.stderr


def test_snap_manifest_custom_branch_falls_back_to_develop(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "snap"},
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test",
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum develop devsha1"
    assert "using develop-latest" in proc.stderr


def test_explicit_manifest_uses_valid_branch_version(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core="zz-core-artifact-for-test-pinnedsha1",
        args=["--only-install-neat-core"],
        env=_installer_env(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum zz-core-artifact-for-test pinnedsha1"
    assert (
        "https://core.test/zz-core-artifact-for-test/pinnedsha1/metadata.json"
        in _curl_log(tmp_path)
    )


@pytest.mark.parametrize("ref_key", ["branch", "ref"])
def test_dependency_object_manifest_uses_valid_artifact(tmp_path, ref_key):
    proc = _run_build(
        tmp_path,
        neat_core={ref_key: "zz-core-artifact-for-test", "spec": "pinnedsha1"},
        args=["--only-install-neat-core"],
        env=_installer_env(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum zz-core-artifact-for-test pinnedsha1"
    assert (
        "https://core.test/zz-core-artifact-for-test/pinnedsha1/metadata.json"
        in _curl_log(tmp_path)
    )


def test_dependency_object_manifest_url_encodes_slash_branch_for_artifact(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"branch": "feature/core-artifact", "spec": "pinnedsha2"},
        args=["--only-install-neat-core"],
        env=_installer_env(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum feature/core-artifact pinnedsha2"
    assert (
        "https://core.test/feature%252Fcore-artifact/pinnedsha2/metadata.json"
        in _curl_log(tmp_path)
    )


def test_dependency_object_manifest_url_encodes_slash_branch_for_latest(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"branch": "feature/core-artifact"},
        args=["--only-install-neat-core"],
        env=_installer_env(tmp_path),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum feature/core-artifact featsha2"
    assert "https://core.test/feature%252Fcore-artifact/latest.tag" in _curl_log(
        tmp_path
    )


def test_explicit_manifest_fails_when_artifact_is_invalid(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core="zz-missing-core-artifact-for-test-pinnedsha1",
        args=["--only-install-neat-core"],
    )

    assert proc.returncode != 0
    assert "unavailable NEAT core artifact" in proc.stderr


def test_protected_branch_rejects_explicit_manifest_value(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core="main-mainsha1",
        args=["--only-install-neat-core"],
        env={"GITHUB_REF_NAME": "main"},
    )

    assert proc.returncode != 0
    assert "must keep neat-core as policy=snap on main/develop" in proc.stderr


def test_protected_branch_rejects_explicit_dependency_object_manifest(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"branch": "main", "spec": "mainsha1"},
        args=["--only-install-neat-core"],
        env={"GITHUB_REF_NAME": "main"},
    )

    assert proc.returncode != 0
    assert "must keep neat-core as policy=snap on main/develop" in proc.stderr


def test_unsupported_manifest_object_fails(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "latest"},
        args=["--only-install-neat-core"],
    )

    assert proc.returncode != 0
    assert "unsupported neat-core.policy" in proc.stderr


def test_snap_manifest_tag_build_uses_matching_core_tag(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "GITHUB_REF_TYPE": "tag",
            "GITHUB_REF_NAME": "v2.1.0",
            "NEAT_APPS_TEST_LATEST_TAGS": "\n".join(
                [
                    "develop=devsha1",
                    "v2.1.0=tagsha1",
                ]
            ),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum v2.1.0 tagsha1"
    assert "https://core.test/v2.1.0/latest.tag" in _curl_log(tmp_path)


def test_snap_manifest_tag_build_fails_without_matching_core_tag(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "GITHUB_REF_TYPE": "tag",
            "GITHUB_REF_NAME": "v9.9.9",
        },
    )

    assert proc.returncode != 0
    assert "exact tag-snap NEAT core artifact" in proc.stderr
    assert "using develop-latest" not in proc.stderr


def test_core_installer_runs_from_deps_debs_scratch_dir(tmp_path):
    installer_pwd = tmp_path / "installer-pwd.txt"
    installer_args = tmp_path / "installer-args.txt"
    deps_dir = tmp_path / "deps"
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_INSTALLER_URL": "https://installer.test/install-neat.sh",
            "NEAT_APPS_TEST_INSTALLER_URL": "https://installer.test/install-neat.sh",
            "NEAT_APPS_TEST_INSTALLER_PWD": str(installer_pwd),
            "NEAT_APPS_TEST_INSTALLER_ARGS": str(installer_args),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert installer_pwd.read_text(encoding="utf-8").strip() == str(deps_dir / "debs")
    assert installer_args.read_text(encoding="utf-8").strip() == (
        "--minimum scratch-core-for-test scratchsha1"
    )
    assert list((deps_dir / "debs").iterdir()) == []
    assert not (APPS_ROOT / "sima-neat-test-Linux-core.deb").exists()
    assert not (APPS_ROOT / "neat-test.deb").exists()
    assert not (APPS_ROOT / "pyneat-test.whl").exists()


def test_vulcan_core_install_uses_minimal_temp_dir(tmp_path):
    sima_cli_cwd = tmp_path / "sima-cli-cwd.txt"
    sima_cli_args = tmp_path / "sima-cli-args.txt"
    _write_fake_sima_cli(tmp_path)

    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": "production",
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(sima_cli_cwd),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(sima_cli_args),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    install_dir = Path(sima_cli_cwd.read_text(encoding="utf-8").strip())
    assert str(install_dir).startswith("/tmp/neat-apps-core-install.")
    assert sima_cli_args.read_text(encoding="utf-8").strip() == (
        "neat install --env production -d . -t minimal "
        "core@scratch-core-for-test:scratchsha1"
    )
    assert not install_dir.exists()


def test_vulcan_core_install_skips_when_neat_json_matches(tmp_path):
    _write_fake_neat_json(
        tmp_path,
        channel="scratch-core-for-test",
        tag="scratchsha1",
        env="prod",
    )

    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": "production",
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (
        "NEAT core already installed (scratch-core-for-test/scratchsha1)" in proc.stdout
    )


def test_vulcan_installer_creates_flat_prebuilt_apps_directory(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = package_dir / "prebuilt-apps"

    proc = _run_vulcan_installer(tmp_path, package_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert not (install_dir / "neat-apps-runtime").exists()
    assert (tmp_path / "sima-cli-cwd.txt").read_text(encoding="utf-8").splitlines() == [
        str(package_dir / "deps" / "core"),
        str(package_dir / "deps" / "insight"),
    ]
    assert (tmp_path / "sima-cli-args.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "neat install -d . -t minimal core",
        "neat install -d . insight",
    ]
    assert not (package_dir / "deps").exists()
    assert f"  {install_dir}" in proc.stdout


def test_vulcan_installer_removes_downloaded_package_staging(tmp_path):
    package_dir = tmp_path / "package"
    package_name = "neat-apps-integration-apps-runtime-bundle-deadbeef"
    extracted_dir = package_dir / package_name
    runtime_dir = _write_vulcan_runtime(extracted_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    archive = package_dir / f"{package_name}.tar.gz"
    archive.write_text("archive\n", encoding="utf-8")
    install_script = package_dir / "install_vulcan_apps_package.sh"
    install_script.write_text("installer\n", encoding="utf-8")
    legacy_metadata = package_dir / "neat-core.json"
    legacy_metadata.write_text("{}\n", encoding="utf-8")
    unrelated = package_dir / "keep.txt"
    unrelated.write_text("keep\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (package_dir / "prebuilt-apps" / "runtime-marker").read_text(
        encoding="utf-8"
    ) == "new\n"
    assert not extracted_dir.exists()
    assert not archive.exists()
    assert not install_script.exists()
    assert not legacy_metadata.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_vulcan_installer_can_skip_dependency_installation(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = package_dir / "prebuilt-apps"

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        sima_cli_status=1,
        extra_env={"NEAT_APPS_SKIP_DEPENDENCIES": "1"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert not (tmp_path / "sima-cli-args.txt").exists()
    assert not (package_dir / "deps").exists()
    assert "Skipping Core and Insight dependency installation" in proc.stdout


def test_vulcan_runtime_gate_uses_snap_core_without_customer_dependencies():
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")

    assert "./build.sh --only-install-neat-core" in workflow
    assert "NEAT_CORE_INSTALL_MODE=vulcan" in workflow
    assert "NEAT_APPS_SKIP_DEPENDENCIES=1" in workflow


def test_vulcan_uses_one_core_context_for_build_and_runtime_tests():
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")
    dependency_output = "needs.resolve-core-context.outputs.dependency_branch"

    assert "pull_request:\n    branches:\n      - main" in workflow
    assert (
        "if: ${{ github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository }}"
        in workflow
    )
    assert "github.ref_type == 'tag'" in workflow
    assert "github.ref_name == 'main'" in workflow
    assert "startsWith(github.ref_name, 'release-')" in workflow
    assert "github.base_ref == 'main'" in workflow
    assert "&& 'main' || github.ref_name" in workflow
    assert workflow.count(dependency_output) == 2
    assert (
        'NEAT_APPS_ARTIFACT_BRANCH_KEY="${{ github.head_ref || github.ref_name }}"'
        in workflow
    )


def test_release_dispatches_vulcan_for_the_created_tag():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "actions: write" in workflow
    assert "trigger-vulcan:" in workflow
    assert "if: ${{ !inputs.dry_run }}" in workflow
    assert "RELEASE_TAG: ${{ needs.release.outputs.tag_name }}" in workflow
    assert 'gh workflow run vulcan-ci.yml --ref "${RELEASE_TAG}"' in workflow


def test_vulcan_installer_keeps_existing_runtime_when_core_install_fails(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        sima_cli_status=1,
    )

    assert proc.returncode != 0
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert not (install_dir.parent / "deps").exists()


def test_vulcan_installer_keeps_existing_runtime_when_insight_install_fails(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        extra_env={"NEAT_APPS_TEST_SIMA_CLI_FAIL_TARGET": "insight"},
    )

    assert proc.returncode != 0
    assert (tmp_path / "sima-cli-args.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "neat install -d . -t minimal core",
        "neat install -d . insight",
    ]
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert (runtime_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert not (install_dir.parent / "deps").exists()


def test_vulcan_installer_refuses_unowned_dependency_workspace(tmp_path):
    package_dir = tmp_path / "package"
    _write_vulcan_runtime(package_dir)
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    unowned_file = install_dir.parent / "deps" / "user-file"
    unowned_file.parent.mkdir()
    unowned_file.write_text("keep\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode != 0
    assert unowned_file.read_text(encoding="utf-8") == "keep\n"
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "sima-cli-args.txt").exists()
    assert "refusing to replace unowned dependency workspace" in proc.stderr


def test_vulcan_installer_recreates_owned_dependency_workspace(tmp_path):
    package_dir = tmp_path / "package"
    _write_vulcan_runtime(package_dir)
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    deps_dir = install_dir.parent / "deps"
    stale_file = deps_dir / "core" / "stale-resource"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale\n", encoding="utf-8")
    (deps_dir / ".neat-apps-installer-owned").write_text(
        "sima-neat/apps\n", encoding="utf-8"
    )

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        extra_env={"NEAT_APPS_TEST_FORBIDDEN_PATH": str(stale_file)},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert install_dir.is_dir()
    assert not deps_dir.exists()


def test_vulcan_installer_cleans_dependency_workspace_on_termination(tmp_path):
    package_dir = tmp_path / "package"
    _write_vulcan_runtime(package_dir)
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    ready_file = tmp_path / "sima-cli-ready"
    bin_dir = _write_fake_sima_cli(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "NEAT_APPS_INSTALL_DIR": str(install_dir),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_READY": str(ready_file),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        }
    )
    proc = subprocess.Popen(
        ["bash", str(VULCAN_INSTALLER)],
        cwd=package_dir,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    for _ in range(100):
        if ready_file.exists():
            break
        time.sleep(0.02)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=5)
        pytest.fail("sima-cli did not start")

    os.killpg(proc.pid, signal.SIGTERM)
    proc.communicate(timeout=5)

    assert proc.returncode != 0
    assert not (install_dir.parent / "deps").exists()
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"


def test_vulcan_installer_preserves_models_during_reinstall(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    models_dir = install_dir / "models"
    models_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    (models_dir / "user-model.mpk").write_text("model\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert (models_dir / "user-model.mpk").read_text(encoding="utf-8") == "model\n"


def test_vulcan_installer_migrates_legacy_runtime_and_models(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_root = tmp_path / "installed"
    install_dir = install_root / "prebuilt-apps"
    legacy_dir = install_root / "neat-apps" / "neat-apps-runtime"
    models_dir = legacy_dir / "models"
    models_dir.mkdir(parents=True)
    (legacy_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    (models_dir / "user-model.mpk").write_text("model\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert (install_dir / "models" / "user-model.mpk").read_text(
        encoding="utf-8"
    ) == "model\n"
    assert not legacy_dir.exists()


def test_vulcan_installer_restores_runtime_when_promotion_fails(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_root = tmp_path / "installed"
    install_dir = install_root / "prebuilt-apps"
    legacy_dir = install_root / "neat-apps" / "neat-apps-runtime"
    models_dir = legacy_dir / "models"
    models_dir.mkdir(parents=True)
    (legacy_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    (models_dir / "user-model.mpk").write_text("model\n", encoding="utf-8")
    failed_file = tmp_path / "mv-failed"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mv_path = bin_dir / "mv"
    mv_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            destination="${!#}"
            if [[ "${destination}" == "${NEAT_APPS_TEST_MV_FAIL_DEST}" && \
                  ! -e "${NEAT_APPS_TEST_MV_FAILED_FILE}" ]]; then
              touch "${NEAT_APPS_TEST_MV_FAILED_FILE}"
              exit 1
            fi
            exec /bin/mv "$@"
            """
        ),
        encoding="utf-8",
    )
    mv_path.chmod(0o755)
    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        extra_env={
            "NEAT_APPS_TEST_MV_FAIL_DEST": str(install_dir),
            "NEAT_APPS_TEST_MV_FAILED_FILE": str(failed_file),
        },
    )

    assert proc.returncode != 0
    assert failed_file.exists()
    assert not install_dir.exists()
    assert (legacy_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert (models_dir / "user-model.mpk").read_text(encoding="utf-8") == "model\n"
