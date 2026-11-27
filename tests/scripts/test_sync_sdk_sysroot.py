"""Contract for scripts/ci/sync-sdk-sysroot.sh.

Fakes stand in for sima-cli, curl, dpkg-deb, sysroot, id and sudo. A local bare git
repository stands in for Core: its stub tools/ci/scheduler_sdk_policy.py records
how the script calls it, because the policy checks themselves belong to Core.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
SYNC = APPS_ROOT / "scripts/ci/sync-sdk-sysroot.sh"
CORE_URL = "https://github.com/sima-neat/core.git"
DEFAULT_SYSROOT = "/opt/toolchain/aarch64/modalix"
REF = "develop"
METADATA_URL = "https://vulcan.test/core/selected/metadata-minimal.json"
PACKAGE_URL = "https://vulcan.test/core/selected/simaai-scheduler.deb"
PACKAGE = b"scheduler package for test\n"
RELEASE = "3.0.1"
PRE_RELEASE = "3.0.1~pre9999"
GIT_RELEASE = "3.0.1~git202609150138.174a5ae-1416"
# Values that Core's scheduler_sdk_policy.py accepts for each kind of version.
FIRMWARE = {
    RELEASE: "3.0.1_master_B9999",
    PRE_RELEASE: "3.0.1_master_B9999",
    GIT_RELEASE: "3.0.1_develop_B1416",
}

STUB_VALIDATOR = """\
import json
import os
import sys
from pathlib import Path

import published_artifact  # noqa: F401  Core's validator imports it from tools/ci.

args = sys.argv[1:]
with open(os.environ["NEAT_APPS_TEST_EVENTS"], "a", encoding="utf-8") as log:
    log.write("validator:" + json.dumps(args) + "\\n")
root = Path(__file__).resolve().parents[2]
if (root / "src").exists() or (root / "tools/lint").exists():
    raise SystemExit("the Core checkout holds more than tools/ci")
release = "--check-release-sdk" in args
status = os.environ.get("NEAT_APPS_TEST_RELEASE_STATUS" if release else "NEAT_APPS_TEST_POLICY_STATUS")
if status:
    print("ValueError: " + os.environ.get("NEAT_APPS_TEST_VALIDATOR_ERROR", "stub"), file=sys.stderr)
    sys.exit(int(status))
policy = json.loads(args[args.index("--policy-json") + 1])
if release:
    print("SKIP sysroot installation: stub")
if "--field" in args:
    print(policy["sysroot-version"])
"""

FAKES = {
    "sima-cli": """\
        #!/usr/bin/env bash
        printf 'sima-cli:%s\\n' "$*" >> "${NEAT_APPS_TEST_EVENTS}"
        echo "Environment: host (linux)"
        printf '%s\\n' "${NEAT_APPS_TEST_RESOLVE}"
        exit "${NEAT_APPS_TEST_SIMA_CLI_STATUS:-0}"
        """,
    "curl": """\
        #!/usr/bin/env python3
        import json
        import os
        import shutil
        import sys

        args = sys.argv[1:]
        output = args[args.index("-o") + 1]
        url = [arg for arg in args if arg.startswith("https://")][0]
        fail = any(arg.startswith("-") and not arg.startswith("--") and "f" in arg for arg in args)
        files = json.loads(os.environ["NEAT_APPS_TEST_URL_FILES"])
        if url in files:
            shutil.copyfile(files[url], output)
        elif fail:
            sys.exit(22)
        else:
            with open(output, "w", encoding="utf-8") as handle:
                handle.write("<html>404 Not Found</html>\\n")
        """,
    "dpkg-deb": """\
        #!/usr/bin/env bash
        set -euo pipefail
        [[ "$1" == "-x" ]]
        printf 'dpkg-deb:%s\n' "$3" >> "${NEAT_APPS_TEST_EVENTS}"
        [[ "${NEAT_APPS_TEST_DPKG_STATUS:-0}" == 0 ]] || exit "${NEAT_APPS_TEST_DPKG_STATUS}"
        mkdir -p "$3/usr/share/SimaScheduler"
        if [[ -z "${NEAT_APPS_TEST_NO_POLICY:-}" ]]; then
          printf '%s' "${NEAT_APPS_TEST_POLICY}" > "$3/usr/share/SimaScheduler/dependency-policy.json"
        fi
        """,
    "sysroot": """\
        #!/usr/bin/env bash
        printf 'sysroot:SDK_PKG_LIST=%s SYSROOT=%s %s\\n' "${SDK_PKG_LIST-<unset>}" \\
          "${SYSROOT-<unset>}" "$*" >> "${NEAT_APPS_TEST_EVENTS}"
        overlay="${NEAT_APPS_TEST_EVENTS}.overlay"
        case "$1" in
          update)
            [[ "${NEAT_APPS_TEST_UPDATE_STATUS:-0}" == 0 ]] || exit "${NEAT_APPS_TEST_UPDATE_STATUS}"
            printf '%s' "${@: -1}" > "${overlay}" ;;
          status)
            if [[ -f "${overlay}" ]]; then
              state="${NEAT_APPS_TEST_STATUS_STATE-active}"
              revision="${NEAT_APPS_TEST_STATUS_REVISION-$(<"${overlay}")}"
            else
              state=inactive
              revision=none
            fi
            echo "SDK image platform:       ${NEAT_APPS_TEST_STATUS_IMAGE:-3.0.1}"
            [[ -z "${state}" ]] || echo "Sysroot overlay state:    ${state}"
            [[ -z "${revision}" ]] || echo "Sysroot overlay revision: ${revision}"
            exit "${NEAT_APPS_TEST_STATUS_STATUS:-0}" ;;
        esac
        """,
    "id": """\
        #!/usr/bin/env bash
        if [[ "$*" == "-u" ]]; then echo "${NEAT_APPS_TEST_UID:-0}"; else echo 4242; fi
        """,
    "sudo": """\
        #!/usr/bin/env bash
        printf 'sudo:%s\\n' "$*" >> "${NEAT_APPS_TEST_EVENTS}"
        exec "$@"
        """,
    # Logs each call, fails the one whose argument is NEAT_APPS_TEST_GIT_FAIL, and sends
    # the Core URL to the local bare repository.
    "git": """\
        #!/usr/bin/env bash
        printf 'git:GIT_TERMINAL_PROMPT=%s %s\\n' "${GIT_TERMINAL_PROMPT-<unset>}" "$*" \\
          >> "${NEAT_APPS_TEST_EVENTS}"
        args=()
        for arg in "$@"; do
          [[ "${arg}" != "${NEAT_APPS_TEST_GIT_FAIL:-}" ]] || exit 1
          [[ "${arg}" != "https://github.com/sima-neat/core.git" ]] || arg="${NEAT_APPS_TEST_CORE_URL}"
          args+=("${arg}")
        done
        exec "${NEAT_APPS_TEST_REAL_GIT}" "${args[@]}"
        """,
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _publish(tmp_path: Path) -> None:
    _git(tmp_path / "core", "push", "-q", "--mirror", str(tmp_path / "core.git"))


def _core_repo(tmp_path: Path) -> str:
    """Publish a bare Core and return its selected spec, which is not the tip of REF."""
    work = tmp_path / "core"
    tools = work / "tools/ci"
    tools.mkdir(parents=True)
    _git(work, "init", "-q", "-b", REF)
    (tools / "published_artifact.py").write_text("", encoding="utf-8")
    (tools / "scheduler_sdk_policy.py").write_text(STUB_VALIDATOR, encoding="utf-8")
    (work / "src").mkdir()
    (work / "src/core.cpp").write_text("", encoding="utf-8")
    (work / "tools/lint").mkdir()
    (work / "tools/lint/check.py").write_text("", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "-q", "-m", "selected")
    spec = _git(work, "rev-parse", "HEAD")[:12]
    (tools / "scheduler_sdk_policy.py").write_text(
        'raise SystemExit("validator from a later Core commit")\n', encoding="utf-8"
    )
    _git(work, "commit", "-q", "-am", "later")
    _git(tmp_path, "init", "-q", "--bare", "core.git")
    _git(tmp_path / "core.git", "config", "uploadpack.allowFilter", "true")
    _publish(tmp_path)
    return spec


def _policy(sysroot_version: str) -> dict[str, object]:
    base = sysroot_version.split("~", 1)[0]
    return {
        "sdk": {"channel": "develop", "sysroot-version": sysroot_version},
        "platform-version": base,
        "elxr_version": FIRMWARE[sysroot_version],
        "sysroot-version": sysroot_version,
    }


def _select(env: dict[str, str], ref: str, spec: str) -> None:
    """Make deps/neat-core.json and sima-cli name another Core."""
    core_file = Path(env["NEAT_APPS_TEST_DEPS_DIR"]) / "neat-core.json"
    core_file.write_text(json.dumps({"neat-core": {"ref": ref, "spec": spec}}), encoding="utf-8")
    resolve = json.loads(env["NEAT_APPS_TEST_RESOLVE"])
    resolve.update({"ref": ref, "resolved_spec": spec})
    env["NEAT_APPS_TEST_RESOLVE"] = json.dumps(resolve)


def _sync_env(
    tmp_path: Path,
    *,
    version: str = PRE_RELEASE,
    metadata: object = None,
) -> dict[str, str]:
    spec = _core_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        (bin_dir / name).write_text(textwrap.dedent(body), encoding="utf-8")
        (bin_dir / name).chmod(0o755)

    deps = tmp_path / "deps"
    deps.mkdir()
    (deps / "manifest.json").write_text(
        json.dumps({"neat-core": {"ref": REF}, "sdk": {"channel": "develop"},
                    "platform-version": "3.0.1"}),
        encoding="utf-8",
    )
    (deps / "neat-core.json").write_text(
        json.dumps({"neat-core": {"ref": REF, "spec": spec}}), encoding="utf-8"
    )
    # A policy from an earlier run, which every failure must remove.
    (deps / "scheduler-sdk-policy.json").write_text("{}", encoding="utf-8")
    sdk_release = tmp_path / "sdk-release"
    sdk_release.write_text("SDK Version = 3.0.1\neLXr Version = 3.0.1\n", encoding="utf-8")

    if metadata is None:
        metadata = {
            "resources": ["simaai-scheduler.deb"],
            "resources-checksum": {"simaai-scheduler.deb": hashlib.sha256(PACKAGE).hexdigest()},
        }
    metadata_file = tmp_path / "metadata-minimal.json"
    metadata_file.write_text(
        metadata if isinstance(metadata, str) else json.dumps(metadata), encoding="utf-8"
    )
    package_file = tmp_path / "simaai-scheduler.deb"
    package_file.write_bytes(PACKAGE)

    (tmp_path / "tmp").mkdir()
    env = os.environ.copy()
    env.pop("SDK_PKG_LIST", None)
    env.update({
        "PATH": f"{bin_dir}:{env['PATH']}",
        "TMPDIR": str(tmp_path / "tmp"),
        "NEAT_APPS_TEST_DEPS_DIR": str(deps),
        "NEAT_APPS_TEST_CORE_URL": str(tmp_path / "core.git"),
        "NEAT_APPS_TEST_REAL_GIT": shutil.which("git"),
        "ELXR_SDK_RELEASE_FILE": str(sdk_release),
        "SYSROOT": str(tmp_path / "sysroot"),
        "SIMA_CLI_BIN": str(bin_dir / "sima-cli"),
        "NEAT_VULCAN_ENV": "production",
        "NEAT_APPS_TEST_EVENTS": str(tmp_path / "events.log"),
        "NEAT_APPS_TEST_RESOLVE": json.dumps({
            "repository": "core", "ref": REF, "resolved_spec": spec,
            "metadata_url": METADATA_URL,
        }),
        "NEAT_APPS_TEST_POLICY": json.dumps(_policy(version), indent=2) + "\n",
        "NEAT_APPS_TEST_URL_FILES": json.dumps({
            METADATA_URL: str(metadata_file),
            PACKAGE_URL: str(package_file),
        }),
    })
    return env


def _run_sync(
    tmp_path: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [str(SYNC), *args], cwd=APPS_ROOT, env=env, text=True, capture_output=True, check=False
    )
    # The script removes its downloads and Core checkout on success and on failure.
    assert list((tmp_path / "tmp").iterdir()) == [], proc.stderr
    return proc


def _events(tmp_path: Path, prefix: str = "") -> list[str]:
    path = tmp_path / "events.log"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [line for line in lines if line.startswith(prefix)]


def _validator_calls(tmp_path: Path) -> list[list[str]]:
    return [json.loads(line.split(":", 1)[1]) for line in _events(tmp_path, "validator:")]


def _spec(env: dict[str, str]) -> str:
    return json.loads(env["NEAT_APPS_TEST_RESOLVE"])["resolved_spec"]


def _policy_output(tmp_path: Path) -> Path:
    return tmp_path / "deps/scheduler-sdk-policy.json"


def _assert_written(tmp_path: Path, proc: subprocess.CompletedProcess[str], env) -> None:
    assert proc.returncode == 0, proc.stderr
    assert _policy_output(tmp_path).read_bytes() == env["NEAT_APPS_TEST_POLICY"].encode()
    # Every git call runs without a credential prompt, and the package unpacks under TMPDIR.
    assert all(line.startswith("git:GIT_TERMINAL_PROMPT=0 ") for line in _events(tmp_path, "git:"))
    assert all(line.startswith(f"dpkg-deb:{tmp_path / 'tmp'}/") for line in _events(tmp_path, "dpkg-deb:"))


def _assert_failed(tmp_path: Path, proc: subprocess.CompletedProcess[str], message: str) -> None:
    assert proc.returncode != 0, proc.stdout
    assert f"ERROR: {message}" in proc.stderr.splitlines(), proc.stderr
    assert not _policy_output(tmp_path).exists()


def _assert_rejected(tmp_path: Path, proc: subprocess.CompletedProcess[str], message: str) -> None:
    _assert_failed(tmp_path, proc, message)
    assert _events(tmp_path, "sysroot:") == []


@pytest.mark.parametrize("version", [PRE_RELEASE, GIT_RELEASE])
def test_pin_updates_sysroot_as_root_with_sdk_package_list(tmp_path, version):
    env = _sync_env(tmp_path, version=version)
    # SDK images set SDK_PKG_LIST with whitespace and empty entries.
    env["SDK_PKG_LIST"] = ",\tlibgrpc-dev,,\tprotobuf-compiler-grpc,\t"

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    sysroot = env["SYSROOT"]
    update, status = _events(tmp_path, "sysroot:")
    assert update == (
        "sysroot:SDK_PKG_LIST=libgrpc-dev,protobuf-compiler-grpc "
        f"SYSROOT={sysroot} update --sysroot {sysroot} {version}"
    )
    assert status.endswith(f" status --sysroot {sysroot}")
    assert (f"Updating the SDK sysroot to {version}, the version Core's Scheduler was built for."
            in proc.stdout.splitlines())
    assert f"Sysroot overlay revision: {version}" in proc.stdout
    assert _events(tmp_path, "sudo:") == []


def test_pin_updates_sysroot_through_sudo_when_not_root(tmp_path):
    env = _sync_env(tmp_path)
    env["NEAT_APPS_TEST_UID"] = "1000"

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    sysroot = env["SYSROOT"]
    assert _events(tmp_path, "sudo:") == [
        f"sudo:env SDK_PKG_LIST= SYSROOT={sysroot} sysroot update --sysroot {sysroot} {PRE_RELEASE}"
    ]


def test_pin_without_root_or_sudo_fails(tmp_path):
    env = _sync_env(tmp_path)
    env["NEAT_APPS_TEST_UID"] = "1000"
    tools = tmp_path / "tools-without-sudo"
    tools.mkdir()
    for directory in (Path("/usr/bin"), Path("/bin")):
        for tool in directory.iterdir():
            if tool.name != "sudo" and not (tools / tool.name).exists():
                (tools / tool.name).symlink_to(tool)
    env["PATH"] = f"{tmp_path / 'bin'}:{tools}"
    (tmp_path / "bin/sudo").unlink()

    proc = _run_sync(tmp_path, env)

    _assert_rejected(tmp_path, proc, f"Updating the SDK sysroot to {PRE_RELEASE} needs root or sudo.")


@pytest.mark.parametrize("uid", ["0", "1000"])
def test_failed_sysroot_update_writes_no_policy(tmp_path, uid):
    env = _sync_env(tmp_path)
    env["NEAT_APPS_TEST_UID"] = uid
    env["NEAT_APPS_TEST_UPDATE_STATUS"] = "23"

    proc = _run_sync(tmp_path, env)

    _assert_failed(tmp_path, proc, f"Failed to update the SDK sysroot to {PRE_RELEASE}.")
    assert [line.rsplit(" ", 1)[-1] for line in _events(tmp_path, "sysroot:")] == [PRE_RELEASE]


@pytest.mark.parametrize("version", [PRE_RELEASE, GIT_RELEASE])
@pytest.mark.parametrize(
    ("state", "revision", "image", "reported"),
    [
        ("active", "{version}0", None, "{version}0 (active)"),
        ("incomplete", "{version}", None, "{version} (incomplete)"),
        ("active-partial", "{version}", None, "{version} (active-partial)"),
        ("inactive", None, "3.0.1", "3.0.1 (inactive)"),
        ("", "", None, "unknown (unknown)"),
    ],
)
def test_pin_fails_when_the_sysroot_is_not_at_the_pin_after_updating(
    tmp_path, version, state, revision, image, reported
):
    env = _sync_env(tmp_path, version=version)
    env["NEAT_APPS_TEST_STATUS_STATE"] = state
    if revision is not None:
        env["NEAT_APPS_TEST_STATUS_REVISION"] = revision.format(version=version)
    if image is not None:
        env["NEAT_APPS_TEST_STATUS_IMAGE"] = image

    proc = _run_sync(tmp_path, env)

    _assert_failed(
        tmp_path, proc,
        f"The SDK sysroot is at {reported.format(version=version)} after updating to {version}.",
    )


@pytest.mark.parametrize("version", [PRE_RELEASE, GIT_RELEASE])
def test_pin_passes_when_the_sdk_image_already_ships_it(tmp_path, version):
    env = _sync_env(tmp_path, version=version)
    # sysroot update writes no overlay when the image already is at the pin.
    env["NEAT_APPS_TEST_STATUS_STATE"] = "inactive"
    env["NEAT_APPS_TEST_STATUS_REVISION"] = "none"
    env["NEAT_APPS_TEST_STATUS_IMAGE"] = version

    _assert_written(tmp_path, _run_sync(tmp_path, env), env)


@pytest.mark.parametrize("version", [PRE_RELEASE, RELEASE])
def test_failed_sysroot_status_writes_no_policy(tmp_path, version):
    env = _sync_env(tmp_path, version=version)
    env["NEAT_APPS_TEST_STATUS_STATUS"] = "23"

    _assert_failed(tmp_path, _run_sync(tmp_path, env), "Failed to read the SDK sysroot status.")


@pytest.mark.parametrize("vulcan_env", ["production", "dev"])
def test_validator_runs_from_selected_core_commit_with_apps_manifest(tmp_path, vulcan_env):
    env = _sync_env(tmp_path)
    env["NEAT_VULCAN_ENV"] = vulcan_env

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    manifest = str(Path(env["NEAT_APPS_TEST_DEPS_DIR"]) / "manifest.json")
    assert _validator_calls(tmp_path) == [[
        "--policy-json", env["NEAT_APPS_TEST_POLICY"].rstrip("\n"),
        "--core-manifest", manifest,
        "--field", "sysroot-version",
    ]]
    assert _events(tmp_path, "sima-cli:") == [
        f"sima-cli:neat install --env {vulcan_env} -t minimal --json core@{REF}:{_spec(env)}"
    ]


def test_defaults_are_production_the_sdk_sysroot_and_the_public_core(tmp_path):
    env = _sync_env(tmp_path, version=RELEASE)
    for name in ("NEAT_VULCAN_ENV", "SYSROOT"):
        del env[name]

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    assert _events(tmp_path, "sima-cli:")[0].startswith("sima-cli:neat install --env production ")
    assert _validator_calls(tmp_path)[-1][-3:] == ["--sysroot", DEFAULT_SYSROOT, "--check-release-sdk"]
    assert _events(tmp_path, "sysroot:")[-1].endswith(f" status --sysroot {DEFAULT_SYSROOT}")
    assert f"git:GIT_TERMINAL_PROMPT=0 -C {tmp_path}/tmp/" in _events(tmp_path, "git:")[1]
    assert _events(tmp_path, "git:")[1].endswith(f" remote add origin {CORE_URL}")


def test_official_release_changes_nothing_after_core_checks_the_sysroot(tmp_path):
    env = _sync_env(tmp_path, version=RELEASE)

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    policy = env["NEAT_APPS_TEST_POLICY"].rstrip("\n")
    manifest = str(Path(env["NEAT_APPS_TEST_DEPS_DIR"]) / "manifest.json")
    assert _validator_calls(tmp_path) == [
        ["--policy-json", policy, "--core-manifest", manifest, "--field", "sysroot-version"],
        ["--policy-json", policy, "--core-manifest", manifest,
         "--sysroot", env["SYSROOT"], "--check-release-sdk"],
    ]
    assert "SKIP sysroot installation: stub" in proc.stdout
    assert [line.split(" ", 2)[-1] for line in _events(tmp_path, "sysroot:")] == [
        f"status --sysroot {env['SYSROOT']}"
    ]


def test_official_release_fails_when_active_sysroot_differs(tmp_path):
    env = _sync_env(tmp_path, version=RELEASE)
    env["NEAT_APPS_TEST_RELEASE_STATUS"] = "1"
    env["NEAT_APPS_TEST_VALIDATOR_ERROR"] = "Active SDK simaai-palette-modalix inventory differs"

    proc = _run_sync(tmp_path, env)

    _assert_rejected(
        tmp_path, proc,
        "The active SDK sysroot is not official release 3.0.1. Use an SDK image of that release.",
    )
    assert "ValueError: Active SDK simaai-palette-modalix inventory differs" in proc.stderr


@pytest.mark.parametrize("status", ["1", "3"])
def test_rejected_policy_changes_nothing(tmp_path, status):
    env = _sync_env(tmp_path)
    env["NEAT_APPS_TEST_POLICY_STATUS"] = status
    env["NEAT_APPS_TEST_VALIDATOR_ERROR"] = "Scheduler SDK channel differs from Core"

    proc = _run_sync(tmp_path, env)

    _assert_rejected(
        tmp_path, proc,
        f"Core's scheduler_sdk_policy.py rejected the Scheduler SDK policy of core@{REF}:{_spec(env)}.",
    )
    assert "ValueError: Scheduler SDK channel differs from Core" in proc.stderr


@pytest.mark.parametrize(
    ("version", "status_env"),
    [(PRE_RELEASE, "NEAT_APPS_TEST_POLICY_STATUS"), (RELEASE, "NEAT_APPS_TEST_RELEASE_STATUS")],
)
def test_validator_usage_error_names_the_core_validator(tmp_path, version, status_env):
    env = _sync_env(tmp_path, version=version)
    env[status_env] = "2"

    proc = _run_sync(tmp_path, env)

    commit = _git(tmp_path / "core", "rev-parse", _spec(env))
    _assert_rejected(
        tmp_path, proc,
        f"Core {commit}'s scheduler_sdk_policy.py does not accept the arguments this script passes.",
    )
    assert len(proc.stderr.splitlines()) == 2


SCHEDULER = "simaai-scheduler.deb"
SCHEDULER_SHA = hashlib.sha256(PACKAGE).hexdigest()
SHAPE_ERROR = (f"Core metadata at {METADATA_URL} must hold a resources list and a "
               "resources-checksum map.")


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, f"Core metadata lists {SCHEDULER} 0 times, not once."),
        ({"resources": [], "resources-checksum": {}}, f"Core metadata lists {SCHEDULER} 0 times, not once."),
        ({"resources": [SCHEDULER] * 2, "resources-checksum": {SCHEDULER: SCHEDULER_SHA}},
         f"Core metadata lists {SCHEDULER} 2 times, not once."),
        ({"resources": [SCHEDULER], "resources-checksum": {}},
         f"Core metadata has no SHA-256 checksum for {SCHEDULER}."),
        ({"resources": [SCHEDULER]}, f"Core metadata has no SHA-256 checksum for {SCHEDULER}."),
        ({"resources": [SCHEDULER], "resources-checksum": {SCHEDULER: SCHEDULER_SHA.upper()}},
         f"Core metadata has no SHA-256 checksum for {SCHEDULER}."),
        ({"resources": [SCHEDULER], "resources-checksum": {SCHEDULER: "0" * 64}},
         f"The Scheduler package's SHA-256 is {SCHEDULER_SHA}, but Core metadata says {'0' * 64}."),
        ([SCHEDULER], SHAPE_ERROR),
        ({"resources": SCHEDULER, "resources-checksum": {SCHEDULER: SCHEDULER_SHA}}, SHAPE_ERROR),
        ({"resources": [SCHEDULER], "resources-checksum": [SCHEDULER_SHA]}, SHAPE_ERROR),
        ("<html>404 Not Found</html>\n", f"Core metadata at {METADATA_URL} is not JSON."),
    ],
)
def test_untrusted_scheduler_package_is_rejected(tmp_path, metadata, message):
    proc = _run_sync(tmp_path, _sync_env(tmp_path, metadata=metadata))

    _assert_rejected(tmp_path, proc, message)
    assert "Traceback" not in proc.stderr
    assert _validator_calls(tmp_path) == []


@pytest.mark.parametrize(
    ("env_name", "message"),
    [
        ("NEAT_APPS_TEST_DPKG_STATUS", "Failed to extract the Core Scheduler package."),
        ("NEAT_APPS_TEST_NO_POLICY", "The Core Scheduler package has no dependency-policy.json."),
    ],
)
def test_unreadable_scheduler_package_is_rejected(tmp_path, env_name, message):
    env = _sync_env(tmp_path)
    env[env_name] = "2"

    _assert_rejected(tmp_path, _run_sync(tmp_path, env), message)


@pytest.mark.parametrize("url", [METADATA_URL, PACKAGE_URL])
def test_failed_download_is_rejected(tmp_path, url):
    env = _sync_env(tmp_path)
    files = json.loads(env["NEAT_APPS_TEST_URL_FILES"])
    del files[url]
    env["NEAT_APPS_TEST_URL_FILES"] = json.dumps(files)

    _assert_rejected(tmp_path, _run_sync(tmp_path, env), f"Failed to download {url}.")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"NEAT_APPS_TEST_SIMA_CLI_STATUS": "1"}, "sima-cli could not resolve core@develop:{spec}."),
        ({"NEAT_APPS_TEST_RESOLVE": "no JSON here"}, "sima-cli printed no Core metadata JSON."),
        ({"NEAT_APPS_TEST_RESOLVE": '[{"repository": "core"}]'},
         "sima-cli printed no Core metadata JSON."),
        ({"metadata_url": None}, "sima-cli printed no Core metadata URL."),
        ({"metadata_url": ""}, "sima-cli printed no Core metadata URL."),
        ({"metadata_url": 7}, "sima-cli printed no Core metadata URL."),
        ({"repository": "other"},
         "sima-cli resolved ('other', 'develop', '{spec}'), not core@develop:{spec}."),
        ({"ref": "other"}, "sima-cli resolved ('core', 'other', '{spec}'), not core@develop:{spec}."),
        ({"resolved_spec": "other"},
         "sima-cli resolved ('core', 'develop', 'other'), not core@develop:{spec}."),
    ],
)
def test_sima_cli_failure_or_identity_mismatch_is_rejected(tmp_path, change, message):
    env = _sync_env(tmp_path)
    spec = _spec(env)
    resolve = json.loads(env["NEAT_APPS_TEST_RESOLVE"])
    for key, value in change.items():
        if key.startswith("NEAT_APPS_TEST_"):
            env[key] = value
        else:
            resolve.pop(key) if value is None else resolve.update({key: value})
            env["NEAT_APPS_TEST_RESOLVE"] = json.dumps(resolve)

    proc = _run_sync(tmp_path, env)

    _assert_rejected(tmp_path, proc, message.format(spec=spec))
    assert _validator_calls(tmp_path) == []


@pytest.mark.parametrize(
    ("core_json", "message"),
    [
        (None, "{path} is missing. Run ./build.sh --only-resolve-neat-core first."),
        ("{", "{path} must hold neat-core.ref and neat-core.spec."),
        ('{"neat-core": []}', "{path} must hold neat-core.ref and neat-core.spec."),
        ('{"neat-core": {"ref": "develop"}}', "{path} must hold neat-core.ref and neat-core.spec."),
        ('{"neat-core": {"ref": "develop", "spec": "latest"}}', None),
        ('{"neat-core": {"ref": "develop", "spec": "0123456"}}', None),
        ('{"neat-core": {"ref": "develop", "spec": "' + "a" * 41 + '"}}', None),
        ('{"neat-core": {"ref": "--upload-pack=x", "spec": "0123456789ab"}}', None),
        ('{"neat-core": {"ref": ".hidden", "spec": "0123456789ab"}}', None),
        ('{"neat-core": {"ref": "a:b", "spec": "0123456789ab"}}', None),
        ('{"neat-core": {"ref": 7, "spec": "0123456789ab"}}', None),
    ],
)
def test_missing_or_malformed_core_selection_is_rejected(tmp_path, core_json, message):
    env = _sync_env(tmp_path)
    core_file = Path(env["NEAT_APPS_TEST_DEPS_DIR"]) / "neat-core.json"
    if core_json is None:
        core_file.unlink()
    else:
        core_file.write_text(core_json, encoding="utf-8")
    if message is None:
        core = json.loads(core_json)["neat-core"]
        message = ("{path} must name a Core ref and an exact commit spec, "
                   f"not {core!r}.")

    proc = _run_sync(tmp_path, env)

    _assert_rejected(tmp_path, proc, message.replace("{path}", str(core_file)))
    assert _events(tmp_path, "sima-cli:") == []


def test_commit_of_a_deleted_branch_is_found_through_its_pull_request_head(tmp_path):
    env = _sync_env(tmp_path)
    work = tmp_path / "core"
    _git(work, "checkout", "-q", "-b", "ci/merged-feature", _spec(env))
    _git(work, "commit", "-q", "--allow-empty", "-m", "feature")
    spec = _git(work, "rev-parse", "HEAD")[:12]
    _git(work, "update-ref", "refs/pull/7/head", "HEAD")
    _git(work, "checkout", "-q", REF)
    _git(work, "branch", "-q", "-D", "ci/merged-feature")
    _publish(tmp_path)
    _select(env, "ci/merged-feature", spec)

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    assert (f"Core ref ci/merged-feature is missing or lacks commit {spec}. "
            "Searching the Core pull request heads." in proc.stdout.splitlines())
    assert len(_validator_calls(tmp_path)) == 1


def test_tag_ref_selects_its_commit(tmp_path):
    env = _sync_env(tmp_path)
    work = tmp_path / "core"
    _git(work, "tag", "v0.6.2", _spec(env))
    _publish(tmp_path)
    _select(env, "v0.6.2", _spec(env))

    proc = _run_sync(tmp_path, env)

    _assert_written(tmp_path, proc, env)
    assert "Searching the Core pull request heads" not in proc.stdout


def test_spec_outside_the_ref_and_every_pull_request_head_is_rejected(tmp_path):
    env = _sync_env(tmp_path)
    work = tmp_path / "core"
    _git(work, "checkout", "-q", "-b", "other")
    _git(work, "commit", "-q", "--allow-empty", "-m", "other")
    spec = _git(work, "rev-parse", "HEAD")[:12]
    _git(work, "checkout", "-q", REF)
    _publish(tmp_path)
    _select(env, REF, spec)

    proc = _run_sync(tmp_path, env)

    _assert_rejected(
        tmp_path, proc,
        f"Core commit {spec} is in neither {REF} nor any Core pull request head at {CORE_URL}.",
    )


@pytest.mark.parametrize(
    ("ref", "fail", "message"),
    [
        (REF, "+refs/pull/*/head:refs/pull/*",
         f"Could not fetch the Core pull request heads from {CORE_URL}."),
        ("gone", "+refs/pull/*/head:refs/pull/*", f"Could not fetch Core from {CORE_URL}."),
        (REF, "remote", f"Failed to add {CORE_URL} as the Core remote."),
        (REF, "sparse-checkout", "Failed to limit the Core checkout to tools/ci."),
        (REF, "checkout", "Failed to check out Core tools/ci at {commit}."),
    ],
)
def test_failed_core_git_step_is_rejected(tmp_path, ref, fail, message):
    env = _sync_env(tmp_path)
    spec = _spec(env)
    if fail.startswith("+refs/pull"):
        # The spec is in no fetched ref, so the script falls back to the pull request heads.
        _git(tmp_path / "core", "checkout", "-q", "-b", "other")
        _git(tmp_path / "core", "commit", "-q", "--allow-empty", "-m", "other")
        spec = _git(tmp_path / "core", "rev-parse", "HEAD")[:12]
        _publish(tmp_path)
    _select(env, ref, spec)
    env["NEAT_APPS_TEST_GIT_FAIL"] = fail

    proc = _run_sync(tmp_path, env)

    commit = _git(tmp_path / "core", "rev-parse", spec)
    _assert_rejected(tmp_path, proc, message.format(commit=commit))


def test_core_commit_without_the_validator_is_rejected(tmp_path):
    env = _sync_env(tmp_path)
    work = tmp_path / "core"
    _git(work, "checkout", "-q", "-b", "no-validator", _spec(env))
    _git(work, "rm", "-q", "tools/ci/scheduler_sdk_policy.py")
    _git(work, "commit", "-q", "-m", "no validator")
    commit = _git(work, "rev-parse", "HEAD")
    _git(work, "checkout", "-q", REF)
    _publish(tmp_path)
    _select(env, "no-validator", commit[:12])

    proc = _run_sync(tmp_path, env)

    _assert_rejected(tmp_path, proc, f"Core {commit} has no tools/ci/scheduler_sdk_policy.py.")


@pytest.mark.parametrize("release", [None, "SDK Version = 3.0.1\n", "eLXr Version = 3.0.1\n"])
def test_non_elxr_sdk_is_rejected(tmp_path, release):
    env = _sync_env(tmp_path)
    sdk_release = Path(env["ELXR_SDK_RELEASE_FILE"])
    if release is None:
        sdk_release.unlink()
    else:
        sdk_release.write_text(release, encoding="utf-8")

    _assert_rejected(
        tmp_path, _run_sync(tmp_path, env),
        f"The SDK sysroot sync needs an eLxr SDK, and {sdk_release} names none.",
    )


def test_missing_sysroot_command_is_rejected(tmp_path):
    env = _sync_env(tmp_path)
    bin_dir = tmp_path / "bin"
    (bin_dir / "sysroot").unlink()
    # An SDK image has a real sysroot in /usr/local/bin.
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"

    _assert_rejected(
        tmp_path, _run_sync(tmp_path, env), "The SDK sysroot sync needs the sysroot command."
    )


def test_missing_sima_cli_is_rejected(tmp_path):
    env = _sync_env(tmp_path)
    env["SIMA_CLI_BIN"] = str(tmp_path / "no-sima-cli")

    _assert_rejected(
        tmp_path, _run_sync(tmp_path, env),
        "The SDK sysroot sync needs sima-cli on PATH or SIMA_CLI_BIN.",
    )


def test_unknown_argument_is_rejected(tmp_path):
    env = _sync_env(tmp_path)

    proc = _run_sync(tmp_path, env, "--bogus")

    assert proc.returncode == 1
    assert proc.stderr == "Unknown argument: --bogus\n"
    assert not _policy_output(tmp_path).exists()
    assert _events(tmp_path) == []


def test_help_prints_the_usage_and_changes_nothing(tmp_path):
    env = _sync_env(tmp_path)

    proc = _run_sync(tmp_path, env, "--help")

    assert proc.returncode == 0
    assert proc.stdout.splitlines() == [
        "Aligns the eLxr SDK sysroot with the Scheduler package that the selected Core ships,",
        "and writes deps/scheduler-sdk-policy.json, which runner selection reads.",
        "",
        "Usage: scripts/ci/sync-sdk-sysroot.sh",
        "Run it after ./build.sh --only-resolve-neat-core, which writes deps/neat-core.json, and",
        "before ./build.sh --all, because a sysroot update replaces a Core install in the sysroot.",
        "Needs an eLxr SDK, sima-cli, curl and git.",
    ]
    assert _events(tmp_path) == []
