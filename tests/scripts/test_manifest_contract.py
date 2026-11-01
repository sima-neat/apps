"""Regression tests for the apps dependency manifest contract."""

from __future__ import annotations

import hashlib
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
import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]
BUILD_SH = APPS_ROOT / "build.sh"
ARCHIVE_INSTALLER = APPS_ROOT / "scripts/install-neat-apps.sh"
VULCAN_INSTALLER = APPS_ROOT / "scripts/install-vulcan-apps-package.sh"
VULCAN_WORKFLOW = APPS_ROOT / ".github/workflows/vulcan-ci.yml"
RELEASE_WORKFLOW = APPS_ROOT / ".github/workflows/release.yml"
RUNTIME_ARCHIVE_VALIDATOR = APPS_ROOT / "scripts/ci/validate_apps_runtime_archive.sh"
APPS_TEST_METADATA = {
    "neat-apps": {
        "ref": "develop",
        "spec": "0123456789abcdef0123456789abcdef01234567",
    }
}


def _write_runtime_archive(
    tmp_path: Path,
    *members: str,
    root_name: str = "prebuilt-apps",
    executable_members: tuple[str, ...] = (),
    extra_archive_members: tuple[str, ...] = (),
    member_contents: dict[str, str] | None = None,
    include_core_metadata: bool = True,
    include_apps_metadata: bool = True,
) -> Path:
    archive_root = tmp_path / "archive-root" / root_name
    archive_root.mkdir(parents=True)
    if include_core_metadata:
        core_metadata = archive_root / "deps/neat-core.json"
        core_metadata.parent.mkdir(parents=True)
        core_metadata.write_text(
            json.dumps({"neat-core": {"ref": "develop", "spec": "core-sha"}}),
            encoding="utf-8",
        )
    if include_apps_metadata:
        apps_metadata = archive_root / "deps/neat-apps.json"
        apps_metadata.parent.mkdir(parents=True, exist_ok=True)
        apps_metadata.write_text(
            json.dumps(APPS_TEST_METADATA),
            encoding="utf-8",
        )
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


def test_runtime_archive_validator_rejects_missing_core_metadata(tmp_path):
    archive = _write_runtime_archive(tmp_path, include_core_metadata=False)
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


def test_runtime_archive_validator_rejects_missing_apps_metadata(tmp_path):
    archive = _write_runtime_archive(tmp_path, include_apps_metadata=False)
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


def test_runtime_archive_validator_rejects_non_exact_apps_commit(tmp_path):
    metadata_path = "deps/neat-apps.json"
    archive = _write_runtime_archive(
        tmp_path,
        metadata_path,
        member_contents={
            metadata_path: json.dumps(
                {"neat-apps": {"ref": "develop", "spec": "short-sha"}}
            )
        },
    )
    proc = _validate_runtime_archive(archive)

    assert proc.returncode != 0


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
        "examples/genai/neat-genai-studio/run.sh",
        "examples/genai/neat-genai-studio/setup.sh",
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

            import json
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

            bodies = json.loads(os.environ.get("NEAT_APPS_TEST_URL_BODIES", "{}"))
            parts = [part for part in urlparse(url).path.split("/") if part]
            body = bodies.get(url)
            if body is None and len(parts) >= 2 and parts[-1] == "latest.tag":
                body = tags.get(parts[-2])
            elif body is None and len(parts) >= 3 and parts[-1] == "metadata.json":
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
            # Record the DevKit sync setting the Apps installer passes to Core's installer.
            sync="${NEAT_INSTALLER_SKIP_DEVKIT_SYNC:-}"
            printf '%s%s\\n' "${sync:+NEAT_INSTALLER_SKIP_DEVKIT_SYNC=${sync} }" "$*" \\
                >> "${NEAT_APPS_TEST_SIMA_CLI_ARGS}"
            if [[ -n "${NEAT_APPS_TEST_EVENT_LOG:-}" ]]; then
                printf 'sima-cli:%s\\n' "$*" >> "${NEAT_APPS_TEST_EVENT_LOG}"
            fi
            if [[ "$*" == *"--json"* ]]; then
                printf '%s\\n' "${NEAT_APPS_TEST_SIMA_CLI_JSON:-}"
                exit "${NEAT_APPS_TEST_SIMA_CLI_JSON_STATUS:-0}"
            fi
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


def _sima_cli_calls(tmp_path: Path) -> list[str]:
    """Return the installs sima-cli ran, without its Core version lookups."""
    path = tmp_path / "sima-cli-args.txt"
    if not path.exists():
        return []
    return [
        call for call in path.read_text(encoding="utf-8").splitlines() if "--json" not in call
    ]


def _write_vulcan_runtime(
    package_dir: Path,
    core: object | None = None,
) -> Path:
    runtime_dir = package_dir / "prebuilt-apps"
    runtime_dir.mkdir(parents=True)
    if core is None:
        core = {"ref": "develop", "spec": "core-sha"}
    deps_dir = runtime_dir / "deps"
    deps_dir.mkdir(parents=True)
    (deps_dir / "neat-core.json").write_text(
        json.dumps({"neat-core": core}),
        encoding="utf-8",
    )
    (deps_dir / "neat-apps.json").write_text(
        json.dumps(APPS_TEST_METADATA),
        encoding="utf-8",
    )
    return runtime_dir


def _run_vulcan_installer(
    tmp_path: Path,
    package_dir: Path,
    *,
    install_dir: Path | None = None,
    runtime_src: Path | None = None,
    allow_overwrite: bool = True,
    input_text: str | None = None,
    sima_cli_status: int = 0,
    compatible_core_version: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _write_fake_sima_cli(tmp_path)
    resolved_core = ""
    if compatible_core_version is not None:
        metadata = tmp_path / "core-metadata.json"
        metadata.write_text(json.dumps({"version": compatible_core_version}), encoding="utf-8")
        resolved_core = "🔧 Environment: test\n" + json.dumps({"metadata_url": metadata.as_uri()})
    # A test's own fake neat in bin_dir wins; otherwise no Core is reported,
    # even on a host that has a real neat installed.
    no_neat_dir = tmp_path / "no-neat-bin"
    no_neat_dir.mkdir(exist_ok=True)
    (no_neat_dir / "neat").write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    (no_neat_dir / "neat").chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_JSON": resolved_core,
            "NEAT_APPS_TEST_SIMA_CLI_STATUS": str(sima_cli_status),
            "PATH": f"{bin_dir}:{no_neat_dir}:{env['PATH']}",
            "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        }
    )
    for name in ("DEVKIT_SYNC_DEVKIT_IP", "DEVKIT_DEPLOY_USER", "NEAT_INSTALLER_SKIP_DEVKIT_SYNC"):
        env.pop(name, None)
    if install_dir is not None:
        env["NEAT_APPS_INSTALL_DIR"] = str(install_dir)
    if allow_overwrite:
        env["NEAT_APPS_OVERWRITE"] = "1"
    else:
        env.pop("NEAT_APPS_OVERWRITE", None)
    if extra_env:
        env.update(extra_env)
    if runtime_src is None:
        runtime_src = package_dir / "prebuilt-apps"
    return subprocess.run(
        ["bash", str(VULCAN_INSTALLER), str(runtime_src)],
        cwd=package_dir,
        env=env,
        input=input_text,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
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
    git_path = bin_dir / "git"
    if not git_path.exists():
        # Branch fixtures must not inherit an exact tag from the real checkout.
        # Local-tag cases provide their own git fixture before calling this helper.
        git_path.write_text(
            '#!/usr/bin/env bash\n'
            'case " $* " in\n'
            '  *" describe --tags --exact-match HEAD "*) exit 1 ;;\n'
            '  *) exec /usr/bin/git "$@" ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        git_path.chmod(0o755)
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
            "NEAT_APPS_TEST_LATEST_TAGS": "\n".join(  # noqa: FLY002
                [
                    "develop=devsha1",
                    "main=mainsha1",
                    "feature%252Fcore-artifact=featsha2",
                    "zz-core-artifact-for-test=featsha1",
                    "scratch-core-for-test=scratchsha1",
                ]
            ),
            "NEAT_APPS_TEST_METADATA": "\n".join(  # noqa: FLY002
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
        capture_output=True,
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


def _write_fake_neat_json(
    tmp_path: Path,
    *,
    channel: str,
    tag: str,
    env: str,
    runtime_channel: str | None = None,
    runtime_version: str | None = None,
    gst_plugins_version: str | None = None,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    neat_path = bin_dir / "neat"
    component_channel = (runtime_channel or channel).replace("/", "-")
    runtime_version = runtime_version or f"0.4.0+{component_channel}.123456789abc"
    payload = {
        "components": {
            "core": {
                "channel": channel,
                "tag": tag,
                "provenance": {"vulcanEnvironment": env},
            },
            "runtime": {"version": runtime_version},
            "gstPlugins": {"version": gst_plugins_version or runtime_version},
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


def _core_metadata_env(
    version: str = "0.4.0+scratch-core-for-test.123456789abc",
    *,
    branch: str = "scratch-core-for-test",
    tag: str = "scratchsha1",
) -> dict[str, str]:
    metadata = {
        "resources": [
            f"neat-runtime_{version}_arm64.deb",
            f"neat-gst-plugins_{version}_arm64.deb",
        ]
    }
    branch_key = branch.replace("/", "%252F")
    return {
        "NEAT_APPS_TEST_URL_BODIES": json.dumps(
            {
                f"https://core.test/{branch_key}/{tag}/metadata.json": json.dumps(
                    metadata
                )
            }
        )
    }


def _write_fake_sysroot(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text(
        "SDK Version = 2.0.0\n"
        "eLXr Version = 12.9.0\n"
        "Platform Base = 2.0.0\n"
        "Platform Version = 2.0.0\n",
        encoding="utf-8",
    )
    id_path = bin_dir / "id"
    id_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            [[ "${1:-}" == -u ]] && echo 0
            """
        ),
        encoding="utf-8",
    )
    id_path.chmod(0o755)
    sysroot_path = bin_dir / "sysroot"
    sysroot_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            printf '%s\\n' "$*" >> "${NEAT_APPS_TEST_SYSROOT_LOG}"
            if [[ -n "${NEAT_APPS_TEST_EVENT_LOG:-}" ]]; then
                printf 'sysroot:%s\\n' "$*" >> "${NEAT_APPS_TEST_EVENT_LOG}"
            fi
            case "${1:-}" in
              update) exit "${NEAT_APPS_TEST_SYSROOT_UPDATE_STATUS:-0}" ;;
              status) exit "${NEAT_APPS_TEST_SYSROOT_STATUS_STATUS:-0}" ;;
            esac
            """
        ),
        encoding="utf-8",
    )
    sysroot_path.chmod(0o755)
    return {
        "ELXR_SDK_RELEASE_FILE": str(sdk_release),
        "NEAT_APPS_TEST_SYSROOT_LOG": str(tmp_path / "sysroot.log"),
        "NEAT_APPS_TEST_EVENT_LOG": str(tmp_path / "events.log"),
    }


def _sysroot_sync_env(
    tmp_path: Path,
    *,
    manifest_text: str = '{"sysroot-version":"2.0.0~pre9999"}\n',
    metadata: dict[str, object] | None = None,
    resolve: dict[str, object] | None = None,
) -> dict[str, str]:
    branch = "scratch-core-for-test"
    version = "scratchsha1"
    try:
        policy = json.loads(manifest_text)
        if isinstance(policy, dict) and isinstance(policy.get("sysroot-version"), str):
            base = policy["sysroot-version"].split("~", 1)[0]
            policy.setdefault("platform-version", base)
            policy.setdefault("elxr_version", f"{base}_master_B9999")
            manifest_text = json.dumps(policy)
    except json.JSONDecodeError:
        pass
    metadata_url = (
        f"https://vulcan.test/core/{branch}/{version}/metadata-minimal.json"
    )
    package_url = f"https://vulcan.test/core/{branch}/{version}/simaai-scheduler.deb"
    package_body = "scheduler package for test\n"
    checksum = hashlib.sha256(package_body.encode()).hexdigest()
    if metadata is None:
        metadata = {
            "resources": ["simaai-scheduler.deb"],
            "resources-checksum": {"simaai-scheduler.deb": checksum},
        }
    if resolve is None:
        resolve = {
            "repository": "core",
            "ref": branch,
            "resolved_spec": version,
            "metadata_url": metadata_url,
        }
    bin_dir = _write_fake_sima_cli(tmp_path)
    dpkg_deb_path = bin_dir / "dpkg-deb"
    dpkg_deb_path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            [[ "$1" == "-x" ]]
            exit_status="${NEAT_APPS_TEST_DPKG_DEB_STATUS:-0}"
            [[ "${exit_status}" == "0" ]] || exit "${exit_status}"
            mkdir -p "$3/usr/share/SimaScheduler"
            printf '%s' "${NEAT_APPS_TEST_SCHEDULER_POLICY}" \\
                > "$3/usr/share/SimaScheduler/dependency-policy.json"
            """
        ),
        encoding="utf-8",
    )
    dpkg_deb_path.chmod(0o755)
    return {
        **_write_fake_sysroot(tmp_path),
        "NEAT_APPS_TEST_SCHEDULER_POLICY": manifest_text,
        "NEAT_APPS_DEPENDENCY_BRANCH": branch,
        "NEAT_CORE_INSTALL_MODE": "vulcan",
        "NEAT_SYNC_SYSROOT": "ON",
        "NEAT_VULCAN_ENV": "production",
        "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
        "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
        "NEAT_APPS_TEST_SIMA_CLI_JSON": json.dumps(resolve),
        "NEAT_APPS_TEST_URL_BODIES": json.dumps(
            {
                **json.loads(_core_metadata_env()["NEAT_APPS_TEST_URL_BODIES"]),
                metadata_url: json.dumps(metadata),
                package_url: package_body,
            }
        ),
        "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
    }


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
    assert json.loads(
        (tmp_path / "deps" / "neat-core.json").read_text(encoding="utf-8")
    ) == {"neat-core": {"ref": "develop", "spec": "devsha1"}}



@pytest.mark.parametrize("policy", [{"policy": "snap"}, ""])
def test_snap_tag_selects_matching_core_release_and_records_revision(tmp_path, policy):
    proc = _run_build(
        tmp_path,
        neat_core=policy,
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "GITHUB_REF_TYPE": "tag",
            "GITHUB_REF_NAME": "v0.6.0",
            "NEAT_APPS_DEPENDENCY_BRANCH": "develop",
            "NEAT_APPS_TEST_LATEST_TAGS": "v0.6.0=release-sha\ndevelop=devsha1",
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum v0.6.0 release-sha"
    assert "https://core.test/develop/latest.tag" not in _curl_log(tmp_path)
    assert json.loads((tmp_path / "deps/neat-core.json").read_text()) == {
        "neat-core": {"ref": "v0.6.0", "spec": "release-sha"}}


def test_snap_tag_missing_core_release_fails_without_develop_fallback(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "snap"},
        args=["--only-install-neat-core"],
        env={**_installer_env(tmp_path), "GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.6.0"},
    )
    assert proc.returncode != 0
    assert "v0.6.0/latest.tag" in proc.stderr
    assert "https://core.test/develop/latest.tag" not in _curl_log(tmp_path)
    assert not (tmp_path / "installer-args.txt").exists()


@pytest.mark.parametrize("selection", ["manifest", "override"])
def test_explicit_core_selection_stays_authoritative_on_apps_tag(tmp_path, selection):
    proc = _run_build(
        tmp_path,
        neat_core={"ref": "main", "spec": "mainsha1"} if selection == "manifest" else {"policy": "snap"},
        args=["--only-install-neat-core"] + (
            ["--neat-core-version", "main-mainsha1"] if selection == "override" else []),
        env={**_installer_env(tmp_path), "GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.6.0"},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum main mainsha1"
    assert not any("/v0.6.0/" in url for url in _curl_log(tmp_path))


def test_snap_local_exact_tag_selects_matching_core_release(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("""#!/usr/bin/env bash
case " $* " in
  *" rev-parse --is-inside-work-tree "*) echo true ;;
  *" describe --tags --exact-match HEAD "*) echo v0.6.0 ;;
  *) exec /usr/bin/git "$@" ;;
esac
""")
    git.chmod(0o755)
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "snap"},
        args=["--only-install-neat-core"],
        env={**_installer_env(tmp_path), "NEAT_APPS_TEST_LATEST_TAGS": "v0.6.0=release-sha"},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum v0.6.0 release-sha"


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


def test_explicit_manifest_is_used_for_pr_into_main(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"ref": "main", "spec": "mainsha1"},
        args=["--only-install-neat-core"],
        env={**_installer_env(tmp_path), "GITHUB_BASE_REF": "main"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == "--minimum main mainsha1"
    assert json.loads(
        (tmp_path / "deps" / "neat-core.json").read_text(encoding="utf-8")
    ) == {"neat-core": {"ref": "main", "spec": "mainsha1"}}


def test_snap_manifest_uses_pr_source_branch(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "snap"},
        args=["--only-install-neat-core"],
        env={
            **_installer_env(tmp_path),
            "GITHUB_BASE_REF": "main",
            "GITHUB_HEAD_REF": "zz-core-artifact-for-test",
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installer_args(tmp_path) == ("--minimum zz-core-artifact-for-test featsha1")


def test_unsupported_manifest_object_fails(tmp_path):
    proc = _run_build(
        tmp_path,
        neat_core={"policy": "latest"},
        args=["--only-install-neat-core"],
    )

    assert proc.returncode != 0
    assert "unsupported neat-core.policy" in proc.stderr


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


@pytest.mark.parametrize(
    ("branch", "tag", "runtime_version"),
    [
        (
            "scratch-core-for-test",
            "scratchsha1",
            "0.4.0+scratch-core-for-test.123456789abc",
        ),
        ("v0.5.0", "release-sha", "0.5.0"),
        ("feature/core-artifact", "pinnedsha2", "0.5.0+internals-branch.abc123"),
    ],
)
@pytest.mark.parametrize("artifact_env", ["production", "staging"])
def test_vulcan_core_install_skips_when_neat_json_matches(
    tmp_path, branch, tag, runtime_version, artifact_env
):
    _write_fake_neat_json(
        tmp_path,
        channel=branch,
        tag=tag,
        env=artifact_env,
        runtime_version=runtime_version,
    )

    proc = _run_build(
        tmp_path,
        neat_core={"ref": branch, "spec": tag},
        args=["--only-install-neat-core"],
        env={
            **_core_metadata_env(runtime_version, branch=branch, tag=tag),
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": artifact_env,
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert f"NEAT core already installed ({branch}/{tag})" in proc.stdout


@pytest.mark.parametrize("artifact_env", ["staging", "development"])
def test_vulcan_core_install_does_not_check_production_metadata_for_other_envs(
    tmp_path, artifact_env
):
    _write_fake_sima_cli(tmp_path)
    _write_fake_neat_json(
        tmp_path,
        channel="scratch-core-for-test",
        tag="scratchsha1",
        env=artifact_env,
    )
    metadata = next(
        iter(json.loads(_core_metadata_env()["NEAT_APPS_TEST_URL_BODIES"]).values())
    )
    metadata_url = (
        "https://artifacts.neat.sima.ai/core/"
        "scratch-core-for-test/scratchsha1/metadata.json"
    )
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "NEAT_ARTIFACTS_BASE_URL": "",
            "NEAT_APPS_TEST_URL_BODIES": json.dumps({metadata_url: metadata}),
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": artifact_env,
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "NEAT core already installed" not in proc.stdout
    assert metadata_url not in _curl_log(tmp_path)
    assert (tmp_path / "sima-cli-args.txt").read_text().strip() == (
        f"neat install --env {artifact_env} -d . -t minimal "
        "core@scratch-core-for-test:scratchsha1"
    )


@pytest.mark.parametrize(
    ("runtime_version", "gst_version"),
    [
        ("0.5.0+different-runtime-branch.new", "0.5.0+scratch-core-for-test.new"),
        ("0.5.0+scratch-core-for-test.old", "0.5.0+scratch-core-for-test.new"),
        ("0.5.0+scratch-core-for-test.new", "0.5.0+scratch-core-for-test.old"),
    ],
)
def test_vulcan_core_install_refreshes_republished_dependencies(
    tmp_path, runtime_version, gst_version
):
    _write_fake_sima_cli(tmp_path)
    _write_fake_neat_json(
        tmp_path,
        channel="scratch-core-for-test",
        tag="scratchsha1",
        env="prod",
        runtime_version=runtime_version,
        gst_plugins_version=gst_version,
    )
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            **_core_metadata_env("0.5.0+scratch-core-for-test.new"),
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": "production",
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "NEAT core already installed" not in proc.stdout
    assert (tmp_path / "sima-cli-args.txt").read_text().strip().endswith(
        "core@scratch-core-for-test:scratchsha1"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "{",
        "[]",
        "{}",
        '{"resources":null}',
        '{"resources":[null]}',
        '{"resources":["neat-runtime_0.5.0_arm64.deb"]}',
        json.dumps(
            {
                "resources": [
                    "neat-runtime__arm64.deb",
                    "neat-gst-plugins_0.5.0_arm64.deb",
                ]
            }
        ),
        json.dumps(
            {
                "resources": [
                    "neat-runtime_0.5.0_arm64.deb",
                    "neat-runtime_0.5.0_arm64.deb",
                    "neat-gst-plugins_0.5.0_arm64.deb",
                ]
            }
        ),
        json.dumps(
            {
                "resources": [
                    "neat-runtime_0.5.0_arm64.deb",
                    "neat-runtime_0.5.1_arm64.deb",
                    "neat-gst-plugins_0.5.0_arm64.deb",
                ]
            }
        ),
    ],
)
def test_vulcan_core_install_refreshes_when_metadata_is_unverifiable(
    tmp_path, metadata
):
    _write_fake_sima_cli(tmp_path)
    _write_fake_neat_json(
        tmp_path,
        channel="scratch-core-for-test",
        tag="scratchsha1",
        env="prod",
        runtime_version="0.5.0",
    )
    bodies = {}
    if metadata is not None:
        bodies["https://core.test/scratch-core-for-test/scratchsha1/metadata.json"] = (
            metadata
        )
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env={
            "NEAT_APPS_TEST_URL_BODIES": json.dumps(bodies),
            "NEAT_APPS_DEPENDENCY_BRANCH": "scratch-core-for-test",
            "NEAT_CORE_INSTALL_MODE": "vulcan",
            "NEAT_VULCAN_ENV": "production",
            "NEAT_APPS_TEST_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.txt"),
            "NEAT_APPS_TEST_SIMA_CLI_ARGS": str(tmp_path / "sima-cli-args.txt"),
        },
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "NEAT core already installed" not in proc.stdout
    assert "Traceback" not in proc.stderr
    assert (tmp_path / "sima-cli-args.txt").read_text().strip().endswith(
        "core@scratch-core-for-test:scratchsha1"
    )


@pytest.mark.parametrize(
    "receipt", ["2.0.0~pre9999", "3.0.0~git202609150138.174a5ae-1416"]
)
def test_vulcan_sysroot_sync_precedes_core_install(tmp_path, receipt):
    proc = _run_build(
        tmp_path,
        platform_version=receipt.split("~", 1)[0],
        args=["--only-install-neat-core"],
        env=_sysroot_sync_env(
            tmp_path, manifest_text=json.dumps({"sysroot-version": receipt})
        ),
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "events.log").read_text(encoding="utf-8").splitlines() == [
        "sima-cli:neat install --env production -t minimal --json "
        "core@scratch-core-for-test:scratchsha1",
        f"sysroot:update {receipt}",
        "sysroot:status",
        "sima-cli:neat install --env production -d . -t minimal "
        "core@scratch-core-for-test:scratchsha1",
    ]


@pytest.mark.parametrize(
    "receipt", ["2.0.0~pre9999", "3.0.0~git202609150138.174a5ae-1416"]
)
def test_vulcan_sysroot_sync_runs_before_installed_core_shortcut(tmp_path, receipt):
    env = _sysroot_sync_env(
        tmp_path, manifest_text=json.dumps({"sysroot-version": receipt})
    )
    _write_fake_neat_json(
        tmp_path,
        channel="scratch-core-for-test",
        tag="scratchsha1",
        env="prod",
    )

    proc = _run_build(
        tmp_path,
        platform_version=receipt.split("~", 1)[0],
        args=["--only-install-neat-core"],
        env=env,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "events.log").read_text(encoding="utf-8").splitlines() == [
        "sima-cli:neat install --env production -t minimal --json "
        "core@scratch-core-for-test:scratchsha1",
        f"sysroot:update {receipt}",
        "sysroot:status",
    ]
    assert "NEAT core already installed" in proc.stdout
    policy = json.loads((tmp_path / "deps/scheduler-sdk-policy.json").read_text())
    assert policy["sysroot-version"] == receipt
    assert policy["elxr_version"] == f"{receipt.split('~', 1)[0]}_master_B9999"


@pytest.mark.parametrize(
    "manifest_text",
    [
        "{",
        "{}",
        '{"sysroot-version":""}',
        '{"sysroot-version":"2.0.0"}',
        '{"sysroot-version":"latest"}',
        '{"sysroot-version":"2.0.0~pre9999; touch /tmp/nope"}',
        '{"sysroot-version":"2.0.0~pre9999\\nstatus"}',
        '{"sysroot-version":"2.0.1"}',
        '{"sysroot-version":"2.0.1~pre9999"}',
        '{"sysroot-version":"2.0.1~git202609150138.174a5ae-1416"}',
        '{"sysroot-version":"2.0.0~git20260915.174a5ae-1416"}',
        '{"sysroot-version":"2.0.0~git202609150138.bad-1416"}',
        '{"sysroot-version":"2.0.0~git202609150138.174a5ae-1416; touch /tmp/nope"}',
    ],
)
def test_vulcan_sysroot_sync_rejects_invalid_receipts(tmp_path, manifest_text):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env=_sysroot_sync_env(tmp_path, manifest_text=manifest_text),
    )

    assert proc.returncode != 0
    assert "Cannot verify the Core Scheduler sysroot-version" in proc.stderr
    assert not (tmp_path / "sysroot.log").exists()


@pytest.mark.parametrize("bad", [None, "receipt", "duplicate-receipt", "inventory",
                                 "duplicate-inventory", "firmware", "overlay",
                                 "inventory-symlink", "headers", "policy-pin",
                                 "missing-firmware", "wrong-platform"])
def test_vulcan_release_sysroot_requires_matching_cached_sdk(tmp_path, bad):
    policy = {"sysroot-version": "3.0.0", "sdk": {"sysroot-version": "3.0.0"},
              "elxr_version": "3.0.0_master_B2499"}
    if bad == "policy-pin":
        policy["sdk"]["sysroot-version"] = "3.0.1"
    env = _sysroot_sync_env(tmp_path, manifest_text=json.dumps(policy))
    if bad in ("missing-firmware", "wrong-platform"):
        selected = json.loads(env["NEAT_APPS_TEST_SCHEDULER_POLICY"])
        if bad == "missing-firmware":
            selected.pop("elxr_version")
        else:
            selected["elxr_version"] = "2.0.0_master_B2499"
        env["NEAT_APPS_TEST_SCHEDULER_POLICY"] = json.dumps(selected)
    receipt = Path(env["ELXR_SDK_RELEASE_FILE"])
    receipt.write_text("SDK Version = 3.0.0\neLXr Version = 3.0.0\n"
                       "Platform Version = 3.0.0\nPlatform Base = 3.0.0\n"
                       "Platform Channel = release\n")
    sysroot = tmp_path / "sysroot"
    env["SYSROOT"] = str(sysroot)
    inventory = sysroot / "var/lib/sima-sdk/sysroot-packages.tsv"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("simaai-palette-modalix\tarm64\t3.0.0\t/usr\n"
                         "simaai-base-files-modalix\tall\t3.0.0\t/etc\n")
    headers = sysroot / "usr/include"
    headers.mkdir(parents=True)
    firmware = sysroot / "etc/packageinfo"
    firmware.parent.mkdir()
    firmware.write_text("SIMA_PACKAGE_VERSION = 3.0.0_master_B2499\n")
    (sysroot / "etc/buildinfo").write_text("SIMA_BUILD_VERSION = 3.0.0_master_B2501\n")
    if bad == "receipt":
        receipt.write_text(receipt.read_text().replace("release", "daily"))
    if bad == "duplicate-receipt":
        receipt.write_text(receipt.read_text() + "Platform Base = 3.0.0\n")
    if bad == "inventory":
        inventory.write_text(inventory.read_text().replace("3.0.0", "3.0.1"))
    if bad == "duplicate-inventory":
        inventory.write_text(inventory.read_text() * 2)
    if bad == "firmware":
        firmware.write_text("SIMA_PACKAGE_VERSION = 3.0.0_master_B2501\n")
    if bad == "overlay":
        (inventory.parent / "sysroot-overlay").symlink_to(tmp_path / "absent-overlay")
    if bad == "inventory-symlink":
        inventory.rename(tmp_path / "inventory.tsv")
        inventory.symlink_to(tmp_path / "inventory.tsv")
    if bad == "headers":
        headers.rmdir()

    proc = _run_build(tmp_path, platform_version="3.0.0",
                      args=["--only-install-neat-core"], env=env)

    if bad:
        assert proc.returncode != 0
        assert "Cannot verify the Core Scheduler sysroot-version" in proc.stderr
        assert not (tmp_path / "sysroot.log").exists()
        assert "SKIP sysroot installation" not in proc.stdout
    else:
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert "SKIP sysroot installation" in proc.stdout
        assert (tmp_path / "sysroot.log").read_text().splitlines() == ["status"]
        assert json.loads((tmp_path / "deps/scheduler-sdk-policy.json").read_text()) == json.loads(
            env["NEAT_APPS_TEST_SCHEDULER_POLICY"]
        )
    calls = (tmp_path / "sima-cli-args.txt").read_text().splitlines()
    assert len(calls) == (1 if bad else 2)


@pytest.mark.parametrize(
    "metadata",
    [
        {"resources": [], "resources-checksum": {}},
        {
            "resources": ["simaai-scheduler.deb", "simaai-scheduler.deb"],
            "resources-checksum": {"simaai-scheduler.deb": "0" * 64},
        },
        {
            "resources": ["simaai-scheduler.deb"],
            "resources-checksum": {},
        },
        {
            "resources": ["simaai-scheduler.deb"],
            "resources-checksum": {"simaai-scheduler.deb": "0" * 64},
        },
    ],
)
def test_vulcan_sysroot_sync_rejects_untrusted_resource(tmp_path, metadata):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env=_sysroot_sync_env(tmp_path, metadata=metadata),
    )

    assert proc.returncode != 0
    assert not (tmp_path / "sysroot.log").exists()


def test_vulcan_sysroot_sync_propagates_package_extraction_failure(tmp_path):
    env = _sysroot_sync_env(tmp_path)
    env["NEAT_APPS_TEST_DPKG_DEB_STATUS"] = "2"

    proc = _run_build(tmp_path, args=["--only-install-neat-core"], env=env)

    assert proc.returncode != 0
    assert "Failed to extract the Core Scheduler package" in proc.stderr
    assert not (tmp_path / "sysroot.log").exists()


def test_vulcan_sysroot_sync_rejects_wrong_core_identity(tmp_path):
    proc = _run_build(
        tmp_path,
        args=["--only-install-neat-core"],
        env=_sysroot_sync_env(
            tmp_path,
            resolve={
                "repository": "core",
                "ref": "develop",
                "resolved_spec": "scratchsha1",
                "metadata_url": "https://vulcan.test/core/metadata-minimal.json",
            },
        ),
    )

    assert proc.returncode != 0
    assert "Cannot verify resolved Core metadata identity" in proc.stderr
    assert not (tmp_path / "sysroot.log").exists()


@pytest.mark.parametrize(
    ("status_env", "expected_calls"),
    [
        ("NEAT_APPS_TEST_SYSROOT_UPDATE_STATUS", ["update {receipt}"]),
        (
            "NEAT_APPS_TEST_SYSROOT_STATUS_STATUS",
            ["update {receipt}", "status"],
        ),
    ],
)
@pytest.mark.parametrize(
    "receipt", ["2.0.0~pre9999", "3.0.0~git202609150138.174a5ae-1416"]
)
def test_vulcan_sysroot_sync_propagates_command_failure(
    tmp_path, status_env, expected_calls, receipt
):
    env = _sysroot_sync_env(
        tmp_path, manifest_text=json.dumps({"sysroot-version": receipt})
    )
    env[status_env] = "23"

    proc = _run_build(
        tmp_path,
        platform_version=receipt.split("~", 1)[0],
        args=["--only-install-neat-core"],
        env=env,
    )

    assert proc.returncode != 0
    assert (tmp_path / "sysroot.log").read_text(encoding="utf-8").splitlines() == (
        [call.format(receipt=receipt) for call in expected_calls]
    )
    assert (tmp_path / "sima-cli-args.txt").read_text().splitlines() == [
        "neat install --env production -t minimal --json "
        "core@scratch-core-for-test:scratchsha1"
    ]


def test_vulcan_sysroot_sync_requires_sysroot_tool(tmp_path):
    env = _sysroot_sync_env(tmp_path)
    (tmp_path / "bin/sysroot").unlink()

    proc = _run_build(tmp_path, args=["--only-install-neat-core"], env=env)

    assert proc.returncode != 0
    assert "requires the sysroot command" in proc.stderr


def test_vulcan_build_enables_ordered_sysroot_sync_without_receipt_pin():
    script = BUILD_SH.read_text(encoding="utf-8")
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")
    manifest = json.loads(
        (APPS_ROOT / "deps/manifest.json").read_text(encoding="utf-8")
    )
    sync = script.index(
        'sync_sysroot_from_core_artifact "${sima_cli_bin}" "${branch}"'
    )
    installed = script.index("if neat_core_installed_matches", sync)
    installer = script.index("run_sima_cli_core_install", installed)
    ensure = script.index("  ensure_neat_core\n")
    configure = script.index("  cmake -S . -B", ensure)

    assert '[[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]] || return 0' in script
    assert "-e NEAT_SYNC_SYSROOT=ON" in workflow
    assert "sysroot-version" not in manifest
    assert "sysroot update --latest" not in script
    assert sync < installed < installer
    assert ensure < configure


def test_vulcan_installer_creates_flat_prebuilt_apps_directory(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = package_dir / "prebuilt-apps"

    proc = _run_vulcan_installer(tmp_path, package_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert not (install_dir / "neat-apps-runtime").exists()
    assert json.loads(
        (install_dir / "deps/neat-core.json").read_text(encoding="utf-8")
    ) == {"neat-core": {"ref": "develop", "spec": "core-sha"}}
    assert (
        json.loads((install_dir / "deps/neat-apps.json").read_text(encoding="utf-8"))
        == APPS_TEST_METADATA
    )
    # The first call looks up the compatible Core version from the package.
    assert (tmp_path / "sima-cli-cwd.txt").read_text(encoding="utf-8").splitlines() == [
        str(package_dir),
        str(runtime_dir / "deps" / "core"),
        str(runtime_dir / "deps" / "insight"),
    ]
    assert _sima_cli_calls(tmp_path) == [
        "neat install -d . -t minimal core@develop:core-sha",
        "neat install -d . insight",
    ]
    assert not (runtime_dir / "deps/core").exists()
    assert not (runtime_dir / "deps/insight").exists()
    assert f"  {install_dir}" in proc.stdout


def test_vulcan_installer_rejects_missing_core_metadata_before_promotion(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "deps" / "neat-core.json").unlink()
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode != 0
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "sima-cli-args.txt").exists()


def test_vulcan_installer_rejects_missing_apps_metadata_before_promotion(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "deps/neat-apps.json").unlink()
    install_dir = tmp_path / "installed/prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode != 0
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "sima-cli-args.txt").exists()


def test_vulcan_installer_rejects_mutable_core_metadata(tmp_path):
    package_dir = tmp_path / "package"
    _write_vulcan_runtime(package_dir, {"ref": "main", "spec": "latest"})

    proc = _run_vulcan_installer(tmp_path, package_dir)

    assert proc.returncode != 0
    assert not (tmp_path / "sima-cli-args.txt").exists()


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
    unrelated = package_dir / "keep.txt"
    unrelated.write_text("keep\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        runtime_src=runtime_dir,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (package_dir / "prebuilt-apps" / "runtime-marker").read_text(
        encoding="utf-8"
    ) == "new\n"
    assert not extracted_dir.exists()
    assert not archive.exists()
    assert not install_script.exists()
    assert not (package_dir / "deps").exists()
    assert (package_dir / "prebuilt-apps/deps/neat-core.json").is_file()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"


def test_vulcan_installer_promotes_explicit_candidate_over_existing_installation(
    tmp_path,
):
    package_dir = tmp_path / "package"
    installed_runtime = _write_vulcan_runtime(package_dir)
    (installed_runtime / "runtime-marker").write_text("old\n", encoding="utf-8")

    package_name = "neat-apps-release-0.4.0-prep-deadbeef"
    extracted_dir = package_dir / package_name
    candidate_runtime = _write_vulcan_runtime(
        extracted_dir,
        {"ref": "v0.3.0", "spec": "release-core-sha"},
    )
    (candidate_runtime / "runtime-marker").write_text("new\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        runtime_src=candidate_runtime,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (installed_runtime / "runtime-marker").read_text(encoding="utf-8") == (
        "new\n"
    )
    assert json.loads(
        (installed_runtime / "deps/neat-core.json").read_text(encoding="utf-8")
    ) == {"neat-core": {"ref": "v0.3.0", "spec": "release-core-sha"}}


def test_vulcan_installer_declines_upgrade_before_installing_dependencies(tmp_path):
    package_dir = tmp_path / "package"
    candidate_runtime = _write_vulcan_runtime(package_dir)
    (candidate_runtime / "runtime-marker").write_text("new\n", encoding="utf-8")

    install_dir = tmp_path / "installed/prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        allow_overwrite=False,
        input_text="n\n",
    )

    assert proc.returncode != 0
    assert "The existing prebuilt-apps installation will be replaced" in proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert (candidate_runtime / "runtime-marker").read_text(encoding="utf-8") == (
        "new\n"
    )
    assert not (tmp_path / "sima-cli-args.txt").exists()


def test_vulcan_installer_cleans_package_staging_when_upgrade_is_declined(tmp_path):
    package_dir = tmp_path / "package"
    package_name = "neat-apps-fix-prebuilt-apps-upgrade-deadbeef"
    extracted_dir = package_dir / package_name
    candidate_runtime = _write_vulcan_runtime(extracted_dir)
    (candidate_runtime / "runtime-marker").write_text("new\n", encoding="utf-8")
    archive = package_dir / f"{package_name}.tar.gz"
    archive.write_text("archive\n", encoding="utf-8")
    install_script = package_dir / "install_vulcan_apps_package.sh"
    install_script.write_text("installer\n", encoding="utf-8")

    install_dir = tmp_path / "installed/prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        runtime_src=candidate_runtime,
        install_dir=install_dir,
        allow_overwrite=False,
        input_text="n\n",
    )

    assert proc.returncode != 0
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert not extracted_dir.exists()
    assert not archive.exists()
    assert not install_script.exists()
    assert not (tmp_path / "sima-cli-args.txt").exists()


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
    assert (install_dir / "deps/neat-core.json").is_file()
    assert not (tmp_path / "sima-cli-args.txt").exists()
    assert not (package_dir / "deps").exists()
    assert "Skipping Core and Insight dependency installation" in proc.stdout


def _neat_status(
    *,
    mode: str = "modalix-board",
    core_version: str | None = "0.6.0+develop.5326276ff765",
    vulcan_ref: str | None = "develop",
    pyneat_version: str | None = None,
    insight_version: str | None = "0.0.0+main.bedca8f",
) -> dict[str, object]:
    """Return the `neat --json --offline` fields the installer reads, as Core reports them."""
    return {
        "schema": "sima.neat.status.v1",
        "environment": {"mode": mode},
        "updateCheck": {"offline": True, "status": "skipped"},
        "components": {
            "core": {
                "version": core_version,
                "provenance": {"vulcanRef": vulcan_ref},
            },
            "pyneat": {"name": "PyNeat", "version": pyneat_version or core_version},
            "insight": {"version": insight_version, "updateAvailable": None},
        },
    }


SDK_STATUS = _neat_status(mode="elxr-sdk")
# A Core installed from main, as `sima-cli neat install core` selects by default.
MAIN_CORE_STATUS = _neat_status(core_version="0.6.0+main.b2001ee23b8c", vulcan_ref="main")
DEVKIT_IP = "192.0.2.10"


def _write_fake_neat_status(tmp_path: Path, status: object) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    neat_path = bin_dir / "neat"
    payload = status if isinstance(status, str) else json.dumps(status)
    neat_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$*" >> "{tmp_path / "neat-args.txt"}"
            if [[ "$*" == "--json --offline" ]]; then
              cat <<'EOF'
            {payload}
            EOF
              exit 0
            fi
            exit 1
            """
        ),
        encoding="utf-8",
    )
    neat_path.chmod(0o755)
    return neat_path


def _write_fake_devkit(tmp_path: Path, status: object | None, exit_code: int = 0) -> None:
    """Write a fake ssh that answers the DevKit's `neat --json --offline` with `status`.

    It exits with `exit_code`; `paired=True` points the installer at it.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    payload = "" if status is None else json.dumps(status)
    ssh_path = bin_dir / "ssh"
    ssh_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$*" >> "{tmp_path / "ssh-args.txt"}"
            # Like ssh, read the caller's input unless -n is given.
            [[ " $* " == *" -n "* ]] || cat >/dev/null
            cat <<'EOF'
            {payload}
            EOF
            exit {exit_code}
            """
        ),
        encoding="utf-8",
    )
    ssh_path.chmod(0o755)


def _ssh_calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "ssh-args.txt"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _run_installer_with_neat_status(
    tmp_path: Path,
    status: object,
    *,
    input_text: str = "",
    core: dict[str, str] | None = None,
    compatible_core_version: str | None = "0.6.0+develop.5326276ff765",
    paired: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    package_dir = tmp_path / "package"
    _write_vulcan_runtime(package_dir, core or {"ref": "develop", "spec": "5326276ff765"})
    _write_fake_neat_status(tmp_path, status)
    env = dict(extra_env or {})
    if paired:
        env["DEVKIT_SYNC_DEVKIT_IP"] = DEVKIT_IP
    return _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=tmp_path / "installed",
        allow_overwrite=False,
        input_text=input_text,
        compatible_core_version=compatible_core_version,
        extra_env=env,
    )


CORE_INSTALL_CALL = "neat install -d . -t minimal core@develop:5326276ff765"
SDK_CORE_INSTALL_CALL = f"NEAT_INSTALLER_SKIP_DEVKIT_SYNC=ON {CORE_INSTALL_CALL}"
INSIGHT_INSTALL_CALL = "neat install -d . insight"
CORE_FOR_APPS = "Core for these Apps: 0.6.0+develop.5326276ff765 (core@develop:5326276ff765)"
SKIPPED_CORE = "Core 0.6.0+develop.5326276ff765 is installed; skipping it."


def test_vulcan_installer_skips_matching_core_and_keeps_insight(tmp_path):
    proc = _run_installer_with_neat_status(tmp_path, _neat_status())

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _sima_cli_calls(tmp_path) == []
    devkit_line = "  DevKit: 0.6.0+develop.5326276ff765, matches"
    assert f"{CORE_FOR_APPS}\n{devkit_line}\n{SKIPPED_CORE}\n" in proc.stdout
    assert "Insight 0.0.0+main.bedca8f is installed; keeping it." in proc.stdout
    assert "[Y/n]" not in proc.stdout
    assert (tmp_path / "installed/deps/neat-core.json").is_file()


def test_vulcan_installer_compares_the_version_a_tag_installs(tmp_path):
    proc = _run_installer_with_neat_status(
        tmp_path,
        _neat_status(core_version="0.6.0", vulcan_ref="v0.6.0"),
        core={"ref": "v0.6.0", "spec": "4404f3fd5a91"},
        compatible_core_version="0.6.0",
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Core for these Apps: 0.6.0 (core@v0.6.0:4404f3fd5a91)" in proc.stdout
    assert "Core 0.6.0 is installed; skipping it." in proc.stdout
    assert _sima_cli_calls(tmp_path) == []


def test_vulcan_installer_matches_pyneat_wheel_spelling_of_a_branch(tmp_path):
    proc = _run_installer_with_neat_status(
        tmp_path,
        _neat_status(
            core_version="0.6.0+feat-x.5326276ff765",
            vulcan_ref="feat-x",
            pyneat_version="0.6.0+feat_x.5326276ff765",
        ),
        core={"ref": "feat-x", "spec": "5326276ff765"},
        compatible_core_version="0.6.0+feat-x.5326276ff765",
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Core 0.6.0+feat-x.5326276ff765 is installed; skipping it." in proc.stdout
    assert _sima_cli_calls(tmp_path) == []


@pytest.mark.parametrize(
    "status",
    [
        MAIN_CORE_STATUS,
        _neat_status(core_version="0.6.0+develop.aaaaaaaaaaaa"),
        # A local Core build keeps the version of the commit it started from.
        _neat_status(vulcan_ref=None),
        _neat_status(pyneat_version="0.5.0"),
    ],
    ids=["other-ref", "other-commit", "local-build", "pyneat-mismatch"],
)
def test_vulcan_installer_installs_different_core_by_default(tmp_path, status):
    # No answer on stdin, as from a script or CI job, installs the compatible Core.
    proc = _run_installer_with_neat_status(tmp_path, status)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Install Core 0.6.0+develop.5326276ff765? [Y/n]" in proc.stdout
    assert _sima_cli_calls(tmp_path) == [CORE_INSTALL_CALL]


@pytest.mark.parametrize(
    ("answer", "calls"), [("y\n", [CORE_INSTALL_CALL]), ("\n", [CORE_INSTALL_CALL]), ("n\n", [])]
)
def test_vulcan_installer_asks_before_installing_core(tmp_path, answer, calls):
    proc = _run_installer_with_neat_status(tmp_path, MAIN_CORE_STATUS, input_text=answer)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "  DevKit: 0.6.0+main.b2001ee23b8c, differs" in proc.stdout
    assert _sima_cli_calls(tmp_path) == calls
    kept = "WARNING: Keeping the installed Core; examples built for Core 0.6.0+develop."
    assert (kept in proc.stdout) == (not calls)


def test_vulcan_installer_overwrite_installs_core_without_asking(tmp_path):
    proc = _run_installer_with_neat_status(
        tmp_path, MAIN_CORE_STATUS, input_text="n\n", extra_env={"NEAT_APPS_OVERWRITE": "1"}
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "[Y/n]" not in proc.stdout
    assert _sima_cli_calls(tmp_path) == [CORE_INSTALL_CALL]


@pytest.mark.parametrize(
    "status",
    [_neat_status(core_version=None, insight_version=None), "not json", "{}", '{"components": []}'],
    ids=["no-core", "not-json", "empty", "wrong-shape"],
)
def test_vulcan_installer_installs_core_and_insight_when_none_is_reported(tmp_path, status):
    proc = _run_installer_with_neat_status(tmp_path, status)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert ": not installed\n" in proc.stdout
    assert "Insight is not installed. Install it? [Y/n]" in proc.stdout
    assert _sima_cli_calls(tmp_path) == [CORE_INSTALL_CALL, INSIGHT_INSTALL_CALL]


def test_vulcan_installer_keeps_missing_insight_when_declined(tmp_path):
    proc = _run_installer_with_neat_status(
        tmp_path, _neat_status(insight_version=None), input_text="n\n"
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert SKIPPED_CORE in proc.stdout
    assert _sima_cli_calls(tmp_path) == []


def test_vulcan_installer_asks_when_the_compatible_version_is_unavailable(tmp_path):
    proc = _run_installer_with_neat_status(tmp_path, _neat_status(), compatible_core_version=None)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Core for these Apps: core@develop:5326276ff765 (version unavailable)" in proc.stdout
    assert "Install Core core@develop:5326276ff765? [Y/n]" in proc.stdout
    assert _sima_cli_calls(tmp_path) == [CORE_INSTALL_CALL]


def test_vulcan_installer_skips_matching_sdk_core_when_unpaired(tmp_path):
    proc = _run_installer_with_neat_status(tmp_path, SDK_STATUS)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (
        f"{CORE_FOR_APPS}\n  SDK   : 0.6.0+develop.5326276ff765, matches\n"
        f"  DevKit: not paired\n{SKIPPED_CORE}\n"
    ) in proc.stdout
    assert "WARNING" not in proc.stdout
    assert _sima_cli_calls(tmp_path) == []


@pytest.mark.parametrize(("answer", "calls"), [("", [SDK_CORE_INSTALL_CALL]), ("n\n", [])])
def test_vulcan_installer_warns_that_an_unpaired_devkit_gets_no_core(tmp_path, answer, calls):
    proc = _run_installer_with_neat_status(
        tmp_path, _neat_status(mode="elxr-sdk", vulcan_ref="main"), input_text=answer
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "WARNING: This SDK is not paired with a DevKit" in proc.stdout
    assert "sima-cli sdk setup --devkit <devkit-ip>" in proc.stdout
    assert "Install Core 0.6.0+develop.5326276ff765 in the SDK only? [Y/n]" in proc.stdout
    assert _sima_cli_calls(tmp_path) == calls


@pytest.mark.parametrize(
    ("local", "devkit", "devkit_line", "question", "calls"),
    [
        (SDK_STATUS, _neat_status(), "0.6.0+develop.5326276ff765, matches", None, []),
        (
            SDK_STATUS,
            MAIN_CORE_STATUS,
            "0.6.0+main.b2001ee23b8c, differs",
            "in the SDK and on the DevKit?",
            [CORE_INSTALL_CALL],
        ),
        (
            _neat_status(mode="elxr-sdk", vulcan_ref="main"),
            _neat_status(),
            "0.6.0+develop.5326276ff765, matches",
            "in the SDK?",
            [SDK_CORE_INSTALL_CALL],
        ),
        (
            _neat_status(mode="elxr-sdk", vulcan_ref="main"),
            MAIN_CORE_STATUS,
            "0.6.0+main.b2001ee23b8c, differs",
            "in the SDK and on the DevKit?",
            [CORE_INSTALL_CALL],
        ),
    ],
    ids=["both-match", "devkit-differs", "sdk-differs", "both-differ"],
)
def test_vulcan_installer_checks_the_paired_devkit_core(
    tmp_path, local, devkit, devkit_line, question, calls
):
    _write_fake_devkit(tmp_path, devkit)

    proc = _run_installer_with_neat_status(tmp_path, local, paired=True)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert f"  DevKit: {devkit_line}\n" in proc.stdout
    assert _ssh_calls(tmp_path) == [
        f"-n -T -o BatchMode=yes -o ConnectTimeout=5 sima@{DEVKIT_IP} neat --json --offline"
    ]
    if question:
        assert f"Install Core 0.6.0+develop.5326276ff765 {question} [Y/n]" in proc.stdout
    else:
        assert SKIPPED_CORE in proc.stdout
    assert _sima_cli_calls(tmp_path) == calls


def test_vulcan_installer_installs_core_on_a_devkit_without_neat(tmp_path):
    _write_fake_devkit(tmp_path, None, exit_code=127)

    proc = _run_installer_with_neat_status(tmp_path, SDK_STATUS, paired=True)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "  DevKit: not installed\n" in proc.stdout
    assert _sima_cli_calls(tmp_path) == [CORE_INSTALL_CALL]


def test_vulcan_installer_keeps_prompt_answers_from_the_devkit_query(tmp_path):
    _write_fake_devkit(tmp_path, MAIN_CORE_STATUS)

    proc = _run_installer_with_neat_status(tmp_path, SDK_STATUS, paired=True, input_text="n\n")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _sima_cli_calls(tmp_path) == []


@pytest.mark.parametrize(
    ("local", "calls"),
    [(SDK_STATUS, []), (_neat_status(mode="elxr-sdk", vulcan_ref="main"), [SDK_CORE_INSTALL_CALL])],
    ids=["sdk-matches", "sdk-differs"],
)
def test_vulcan_installer_installs_sdk_only_when_the_devkit_is_unreachable(tmp_path, local, calls):
    _write_fake_devkit(tmp_path, None, exit_code=255)

    proc = _run_installer_with_neat_status(
        tmp_path, local, paired=True, extra_env={"DEVKIT_DEPLOY_USER": "admin"}
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert f"  DevKit: admin@{DEVKIT_IP} is unreachable over SSH\n" in proc.stdout
    assert ("WARNING: The DevKit is unreachable" in proc.stdout) == bool(calls)
    assert _sima_cli_calls(tmp_path) == calls


def test_vulcan_installer_skip_dependencies_does_not_query_neat(tmp_path):
    _write_fake_devkit(tmp_path, _neat_status())

    proc = _run_installer_with_neat_status(
        tmp_path, _neat_status(), paired=True, extra_env={"NEAT_APPS_SKIP_DEPENDENCIES": "1"}
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not (tmp_path / "neat-args.txt").exists()
    assert _ssh_calls(tmp_path) == []
    assert not (tmp_path / "sima-cli-args.txt").exists()


def test_vulcan_runtime_gate_uses_packaged_core_without_customer_dependencies():
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")

    artifacts = yaml.safe_load(workflow)["jobs"]["build"]["with"]["artifact_path"]
    assert artifacts == "_vulcan-build-artifacts"
    assert "tar -xOzf" in workflow
    assert "prebuilt-apps/deps/neat-core.json" in workflow
    assert "--neat-core-version" in workflow
    assert "NEAT_CORE_INSTALL_MODE=vulcan" in workflow
    assert "NEAT_APPS_SKIP_DEPENDENCIES=1" in workflow
    assert (
        'bash "${GITHUB_WORKSPACE}/scripts/install-vulcan-apps-package.sh" '
        '"./prebuilt-apps"' in workflow
    )


def test_vulcan_package_shell_quotes_exact_runtime_candidate():
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")

    assert 'runtime_extract_dir="${runtime_archive_name%.tar.gz}"' in workflow
    assert 'runtime_install_path="./${runtime_extract_dir}/prebuilt-apps"' in workflow
    assert (
        "printf -v quoted_runtime_install_path '%q' "
        '"${runtime_install_path}"' in workflow
    )
    assert (
        '--install-script "bash ./install_vulcan_apps_package.sh '
        '${quoted_runtime_install_path}"' in workflow
    )


def test_vulcan_uses_push_context_without_manifest_override():
    workflow = VULCAN_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" not in workflow
    assert "github.event.pull_request" not in workflow
    assert "github.head_ref" not in workflow
    assert "GITHUB_HEAD_REF" not in workflow
    assert "NEAT_APPS_ARTIFACT_BRANCH_KEY" not in workflow
    assert "NEAT_APPS_ARTIFACT_SHORT_SHA" not in workflow
    assert '-e GITHUB_REF_NAME="${{ github.ref_name }}"' in workflow
    assert '-e GITHUB_SHA="${{ github.sha }}"' in workflow
    assert "resolve-core-context" not in workflow
    assert "NEAT_APPS_DEPENDENCY_BRANCH" not in workflow


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
    assert _sima_cli_calls(tmp_path) == [
        "neat install -d . -t minimal core@develop:core-sha",
        "neat install -d . insight",
    ]
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert (runtime_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert not (install_dir.parent / "deps").exists()


def test_vulcan_installer_leaves_root_dependency_directory_untouched(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    install_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    unowned_file = install_dir.parent / "deps" / "user-file"
    unowned_file.parent.mkdir()
    unowned_file.write_text("keep\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert unowned_file.read_text(encoding="utf-8") == "keep\n"
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"


def test_vulcan_installer_recreates_candidate_dependency_scratch(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    stale_file = runtime_dir / "deps/core/stale-resource"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale\n", encoding="utf-8")

    proc = _run_vulcan_installer(
        tmp_path,
        package_dir,
        install_dir=install_dir,
        extra_env={"NEAT_APPS_TEST_FORBIDDEN_PATH": str(stale_file)},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert install_dir.is_dir()
    assert not (install_dir / "deps/core").exists()
    assert not (install_dir / "deps/insight").exists()


def test_vulcan_installer_cleans_dependency_workspace_on_termination(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
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
            "NEAT_APPS_OVERWRITE": "1",
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        }
    )
    proc = subprocess.Popen(
        ["bash", str(VULCAN_INSTALLER), str(runtime_dir)],
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
    assert not (runtime_dir / "deps/core").exists()
    assert not (runtime_dir / "deps/insight").exists()
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"


def test_vulcan_installer_preserves_models_during_reinstall(tmp_path):
    package_dir = tmp_path / "package"
    runtime_dir = _write_vulcan_runtime(package_dir)
    (runtime_dir / "runtime-marker").write_text("new\n", encoding="utf-8")
    install_dir = tmp_path / "installed" / "prebuilt-apps"
    models_dir = install_dir / "models"
    models_dir.mkdir(parents=True)
    (install_dir / "runtime-marker").write_text("old\n", encoding="utf-8")
    (install_dir / "custom-config.yaml").write_text("custom\n", encoding="utf-8")
    (models_dir / "user-model.mpk").write_text("model\n", encoding="utf-8")
    previous_backup = install_dir.parent / ".prebuilt-apps-backup"
    previous_backup.mkdir()
    (previous_backup / "obsolete-backup").write_text("obsolete\n", encoding="utf-8")

    proc = _run_vulcan_installer(tmp_path, package_dir, install_dir=install_dir)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (install_dir / "runtime-marker").read_text(encoding="utf-8") == "new\n"
    assert (models_dir / "user-model.mpk").read_text(encoding="utf-8") == "model\n"
    backup_dir = install_dir.parent / ".prebuilt-apps-backup"
    assert (backup_dir / "runtime-marker").read_text(encoding="utf-8") == "old\n"
    assert (backup_dir / "custom-config.yaml").read_text(encoding="utf-8") == (
        "custom\n"
    )
    assert not (backup_dir / "models").exists()
    assert not (backup_dir / "obsolete-backup").exists()
    assert "Existing backup : will be replaced" in proc.stdout


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
