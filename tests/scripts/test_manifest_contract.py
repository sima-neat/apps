"""Regression tests for the apps dependency manifest contract."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
BUILD_SH = APPS_ROOT / "build.sh"


def _write_manifest(path: Path, neat_core: object = "", platform_version: str = "2.0.0") -> None:
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


def _write_fake_sima_cli(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    sima_cli_path = bin_dir / "sima-cli"
    sima_cli_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$PWD" > "${NEAT_APPS_TEST_SIMA_CLI_CWD}"
            printf '%s\\n' "$*" > "${NEAT_APPS_TEST_SIMA_CLI_ARGS}"
            touch sima-cli-ran.txt
            """
        ),
        encoding="utf-8",
    )
    sima_cli_path.chmod(0o755)
    return bin_dir


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
                    "feature%2Fcore-artifact=featsha2",
                    "zz-core-artifact-for-test=featsha1",
                    "scratch-core-for-test=scratchsha1",
                ]
            ),
            "NEAT_APPS_TEST_METADATA": "\n".join(
                [
                    "main:mainsha1",
                    "feature%2Fcore-artifact:pinnedsha2",
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
        "https://core.test/feature%2Fcore-artifact/pinnedsha2/metadata.json"
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
    assert "https://core.test/feature%2Fcore-artifact/latest.tag" in _curl_log(tmp_path)


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
    assert "NEAT core already installed (scratch-core-for-test/scratchsha1)" in proc.stdout
