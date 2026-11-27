"""Contract for scripts/install-neat-core.sh, run through build.sh.

A fake sima-cli answers `neat install --json core@REF:SPEC` from a table of
published branches, as the real one does, and records every call with its
update check and base URL. A local HTTP server holds the metadata files, so
the existence check of an exact revision sees a real 404.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import pty
import shutil
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]
BUILD_SH = APPS_ROOT / "build.sh"
INSTALL_NEAT_CORE = APPS_ROOT / "scripts/install-neat-core.sh"

# Branch -> the revision its latest.tag names.
PUBLISHED = {
    "develop": "devsha1",
    "main": "mainsha1",
    "v0.6.0": "releasesha1",
    "feature/core-artifact": "featsha2",
    "zz-core-artifact-for-test": "featsha1",
}
# Exact revisions whose metadata exists, besides each latest one.
PINNED = {("zz-core-artifact-for-test", "pinnedsha1"), ("feature/core-artifact", "pinnedsha2")}
ARTIFACT_BASES = ("production", "prod", "staging", "dev", "custom")
ARTIFACT_BASE = ""

FAKE_SIMA_CLI = r'''#!/usr/bin/env python3
import json
import os
import sys
import urllib.parse

args = sys.argv[1:]
check = os.environ.get("SIMA_CLI_CHECK_FOR_UPDATE", "1")
with open(os.environ["FAKE_SIMA_CLI_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\n")
with open(os.environ["FAKE_SIMA_CLI_ENV_LOG"], "a", encoding="utf-8") as log:
    log.write(f"check={check} base={os.environ.get('SIMA_NEAT_BASE_URL', '-')}\n")
# Like sima-cli 2.1.22 while 2.1.23 is published: it asks first and aborts on EOF.
auto_accept = os.environ.get("SIMA_CLI_AUTO_ACCEPT_UPDATE", "").lower() in ("1", "true", "yes", "on")
if check == "1" and os.environ.get("FAKE_UPDATE_AVAILABLE") == "1" and not auto_accept:
    print("Update sima-cli now? [y/N]: ", end="", flush=True)
    if not sys.stdin.readline():
        print("Aborted!", file=sys.stderr)
        sys.exit(1)
target = args[-1]
env = args[args.index("--env") + 1]
ref, spec = target.removeprefix("core@").rsplit(":", 1)
if "--json" in args:
    if os.environ.get("FAKE_LOOKUP_ERROR"):
        print(os.environ["FAKE_LOOKUP_ERROR"], file=sys.stderr)
        sys.exit(1)
    if "FAKE_LOOKUP_STDOUT" in os.environ:
        print(os.environ["FAKE_LOOKUP_STDOUT"])
        sys.exit(0)
    published = json.loads(os.environ["FAKE_PUBLISHED"])
    key = urllib.parse.quote(urllib.parse.quote(ref, safe=""), safe="")
    base = os.environ.get("SIMA_NEAT_BASE_URL", f"{os.environ['FAKE_ARTIFACT_BASE']}/{env}")
    if spec == "latest":
        if ref not in published:
            status = os.environ.get("FAKE_MISSING_STATUS", "403")
            print(f"Error: GET {base}/core/{key}/latest.tag failed: {status} Client Error",
                  file=sys.stderr)
            sys.exit(1)
        spec = published[ref]
    payload = {"repository": "core", "ref": ref, "resolved_spec": spec,
               "metadata_url": f"{base}/core/{key}/{spec}/metadata-minimal.json"}
    payload.update(json.loads(os.environ.get("FAKE_SIMA_CLI_JSON_PATCH", "{}")))
    print("Environment: sdk")
    print(json.dumps(payload, indent=2))
    sys.exit(0)
with open(os.environ["FAKE_SIMA_CLI_CWD"], "a", encoding="utf-8") as log:
    log.write(os.getcwd() + "\n")
print(os.environ.get("FAKE_SIMA_CLI_INSTALL_OUTPUT", "Installation completed successfully."))
sys.exit(int(os.environ.get("FAKE_SIMA_CLI_INSTALL_STATUS", "0")))
'''

# Branch fixtures must not inherit an exact tag or branch from the real checkout.
FAKE_GIT = r'''#!/usr/bin/env bash
case " $* " in
  *" rev-parse --is-inside-work-tree "*) [[ -z "${FAKE_GIT_NOT_REPO:-}" ]] && echo true || exit 128 ;;
  *" describe --tags --exact-match HEAD "*) [[ -n "${FAKE_GIT_TAG:-}" ]] && echo "${FAKE_GIT_TAG}" || exit 1 ;;
  *" rev-parse --abbrev-ref HEAD "*) [[ -n "${FAKE_GIT_BRANCH:-}" ]] && echo "${FAKE_GIT_BRANCH}" || exit 128 ;;
  *) exit 2 ;;
esac
'''


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(autouse=True)
def artifact_server(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(sys.modules[__name__], "ARTIFACT_BASE",
                        f"http://127.0.0.1:{server.server_port}")
    for name in ARTIFACT_BASES:
        for ref, spec in [*PINNED, *PUBLISHED.items()]:
            path = root / name / "core" / ref.replace("/", "%2F") / spec / "metadata-minimal.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
    yield
    server.shutdown()
    server.server_close()


def _metadata_url(ref: str, spec: str, base: str = "production") -> str:
    return f"{ARTIFACT_BASE}/{base}/core/{ref.replace('/', '%252F')}/{spec}/metadata-minimal.json"


def _write_tool(bin_dir: Path, name: str, text: str) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    path = bin_dir / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _run(
    tmp_path: Path,
    *,
    neat_core: object = "",
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    command: list[str] | None = None,
    manifest_text: str | None = None,
    stdin: str | int = "",
    cwd: Path = APPS_ROOT,
) -> subprocess.CompletedProcess[str]:
    deps_dir = tmp_path / "deps"
    deps_dir.mkdir(exist_ok=True)
    if manifest_text is None:
        manifest_text = json.dumps({"neat-core": neat_core, "platform-version": "2.0.0"})
    if manifest_text:
        (deps_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    for name, text in (("sima-cli", FAKE_SIMA_CLI), ("git", FAKE_GIT)):
        if not (bin_dir / name).exists():
            _write_tool(bin_dir, name, text)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir(exist_ok=True)

    run_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GITHUB_", "NEAT_", "SIMA_"))
    }
    run_env.update(
        {
            "PATH": f"{bin_dir}:{run_env['PATH']}",
            "TMPDIR": str(temp_dir),
            "NEAT_APPS_TEST_DEPS_DIR": str(deps_dir),
            "NEAT_APPS_DEPENDENCY_BRANCH": "develop",
            "FAKE_PUBLISHED": json.dumps(PUBLISHED),
            "FAKE_ARTIFACT_BASE": ARTIFACT_BASE,
            "FAKE_SIMA_CLI_LOG": str(tmp_path / "sima-cli.log"),
            "FAKE_SIMA_CLI_ENV_LOG": str(tmp_path / "sima-cli-env.log"),
            "FAKE_SIMA_CLI_CWD": str(tmp_path / "sima-cli-cwd.log"),
        }
    )
    run_env.update(env or {})
    if command is None:
        command = [str(BUILD_SH), *(args or ["--only-install-neat-core"])]
    stdin_args = {"stdin": stdin} if isinstance(stdin, int) else {"input": stdin}
    return subprocess.run(
        command, cwd=cwd, env=run_env, text=True, capture_output=True, check=False, **stdin_args
    )


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def _lookups(tmp_path: Path) -> list[str]:
    return [line.split()[-1] for line in _lines(tmp_path / "sima-cli.log") if "--json" in line]


def _installs(tmp_path: Path) -> list[str]:
    return [line for line in _lines(tmp_path / "sima-cli.log") if "--json" not in line]


def _checks(tmp_path: Path) -> list[str]:
    return [line.split()[0] for line in _lines(tmp_path / "sima-cli-env.log")]


def _install(target: str, env: str = "production") -> str:
    return f"neat install --env {env} -d . -t minimal {target}"


def _record(tmp_path: Path) -> dict[str, object]:
    return json.loads((tmp_path / "deps/neat-core.json").read_text(encoding="utf-8"))


def _core(ref: str, spec: str) -> dict[str, object]:
    return {"neat-core": {"ref": ref, "spec": spec}}


# Snap policy


@pytest.mark.parametrize("neat_core", ["", {"policy": "snap"}, {"policy": " SNAP "},
                                       {"policy": "snap", "ref": "main"}],
                         ids=["empty", "snap", "snap-case-and-space", "snap-with-ref"])
@pytest.mark.parametrize("branch", ["develop", "main"])
def test_snap_on_develop_or_main_installs_that_branchs_latest_revision(
    tmp_path, neat_core, branch
):
    proc = _run(tmp_path, neat_core=neat_core, env={"NEAT_APPS_DEPENDENCY_BRANCH": branch})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == [f"core@{branch}:latest"]
    assert _installs(tmp_path) == [_install(f"core@{branch}:{PUBLISHED[branch]}")]
    assert _record(tmp_path) == _core(branch, PUBLISHED[branch])
    assert f"Selected NEAT core core@{branch}:{PUBLISHED[branch]} (production)." in proc.stderr


@pytest.mark.parametrize("neat_core", ["", {"policy": "snap"}])
@pytest.mark.parametrize(
    "tag_env",
    [{"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.6.0"}, {"FAKE_GIT_TAG": "v0.6.0"}],
    ids=["ci-tag", "checkout-tag"],
)
def test_snap_on_apps_tag_installs_core_release_of_same_name(tmp_path, tag_env, neat_core):
    proc = _run(tmp_path, neat_core=neat_core, env=tag_env)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == ["core@v0.6.0:latest"]
    assert _installs(tmp_path) == [_install("core@v0.6.0:releasesha1")]
    assert _record(tmp_path) == _core("v0.6.0", "releasesha1")


def test_snap_on_apps_tag_without_core_release_fails_without_fallback(tmp_path):
    # A tag build checks out a detached HEAD, which is no main or develop branch.
    proc = _run(
        tmp_path,
        env={"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v9.9.9", "NEAT_APPS_DEPENDENCY_BRANCH": "",
             "FAKE_GIT_BRANCH": "HEAD"},
    )

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "records no Core" not in proc.stderr
    assert "v9.9.9/latest.tag failed" in proc.stderr
    assert "Cannot resolve NEAT core artifact core@v9.9.9:latest in production." in proc.stderr
    assert _lookups(tmp_path) == ["core@v9.9.9:latest"]
    assert _installs(tmp_path) == []
    assert not (tmp_path / "deps/neat-core.json").exists()


@pytest.mark.parametrize(
    ("branch_env", "branch"),
    [
        ({"NEAT_APPS_DEPENDENCY_BRANCH": "zz-core-artifact-for-test"}, "zz-core-artifact-for-test"),
        ({"NEAT_APPS_DEPENDENCY_BRANCH": "feature/core-artifact"}, "feature/core-artifact"),
        (
            {"NEAT_APPS_DEPENDENCY_BRANCH": "", "GITHUB_HEAD_REF": "zz-core-artifact-for-test",
             "GITHUB_REF_NAME": "develop"},
            "zz-core-artifact-for-test",
        ),
        ({"NEAT_APPS_DEPENDENCY_BRANCH": "", "GITHUB_REF_NAME": "feature/core-artifact",
          "FAKE_GIT_BRANCH": "develop"}, "feature/core-artifact"),
        ({"NEAT_APPS_DEPENDENCY_BRANCH": "", "FAKE_GIT_BRANCH": "feature/core-artifact"},
         "feature/core-artifact"),
        ({"NEAT_APPS_DEPENDENCY_BRANCH": "zz-core-artifact-for-test", "GITHUB_HEAD_REF": "main"},
         "zz-core-artifact-for-test"),
    ],
    ids=["branch", "slash-branch", "pr-source-branch", "ci-branch", "checkout-branch",
         "dependency-branch-wins"],
)
def test_snap_on_branch_core_publishes_installs_that_branch(tmp_path, branch_env, branch):
    proc = _run(tmp_path, env=branch_env)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == [f"core@{branch}:latest"]
    assert _installs(tmp_path) == [_install(f"core@{branch}:{PUBLISHED[branch]}")]
    assert "using develop-latest" not in proc.stderr


@pytest.mark.parametrize("status", ["403", "404"])
@pytest.mark.parametrize(
    ("neat_core", "branch_env", "branch"),
    [
        ("", {"NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test"},
         "zz-missing-core-artifact-for-test"),
        ({"policy": "snap"}, {"NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test"},
         "zz-missing-core-artifact-for-test"),
        ("", {"NEAT_APPS_DEPENDENCY_BRANCH": "", "FAKE_GIT_BRANCH": "HEAD"}, "HEAD"),
    ],
    ids=["empty", "snap", "detached-head"],
)
def test_snap_on_unpublished_branch_falls_back_to_develop(
    tmp_path, neat_core, branch_env, branch, status
):
    proc = _run(tmp_path, neat_core=neat_core, env={**branch_env, "FAKE_MISSING_STATUS": status})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (
        f"No NEAT core artifact found for dependency branch '{branch}'; using develop-latest."
        in proc.stderr
    )
    assert "latest.tag failed" not in proc.stderr
    assert _lookups(tmp_path) == [f"core@{branch}:latest", "core@develop:latest"]
    assert _installs(tmp_path) == [_install("core@develop:devsha1")]
    assert _record(tmp_path) == _core("develop", "devsha1")


@pytest.mark.parametrize("branch", ["develop", "main"])
def test_snap_on_develop_or_main_never_falls_back(tmp_path, branch):
    published = {ref: spec for ref, spec in PUBLISHED.items() if ref != branch}
    proc = _run(
        tmp_path,
        env={"NEAT_APPS_DEPENDENCY_BRANCH": branch, "FAKE_PUBLISHED": json.dumps(published)},
    )

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "records no Core" not in proc.stderr
    assert f"Cannot resolve NEAT core artifact core@{branch}:latest" in proc.stderr
    assert "using develop-latest" not in proc.stderr
    assert _lookups(tmp_path) == [f"core@{branch}:latest"]


def test_snap_fallback_shows_why_the_develop_lookup_failed(tmp_path):
    published = {ref: spec for ref, spec in PUBLISHED.items() if ref != "develop"}
    proc = _run(
        tmp_path,
        env={"NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test",
             "FAKE_PUBLISHED": json.dumps(published)},
    )

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "records no Core" not in proc.stderr
    assert "core/develop/latest.tag failed: 403" in proc.stderr
    assert "Cannot resolve NEAT core artifact core@develop:latest in production." in proc.stderr
    assert _installs(tmp_path) == []


def test_snap_outside_a_git_checkout_uses_develop(tmp_path):
    proc = _run(tmp_path, env={"NEAT_APPS_DEPENDENCY_BRANCH": "", "FAKE_GIT_NOT_REPO": "1"})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == ["core@develop:latest"]


def test_snap_without_a_git_branch_asks_for_one(tmp_path):
    proc = _run(tmp_path, env={"NEAT_APPS_DEPENDENCY_BRANCH": ""})

    assert proc.returncode != 0
    assert "Set NEAT_APPS_DEPENDENCY_BRANCH." in proc.stderr
    assert _lines(tmp_path / "sima-cli.log") == []


@pytest.mark.parametrize(
    "error",
    ["Max retries exceeded (Connection refused)", "401 Client Error: Unauthorized",
     "500 Server Error: Internal Server Error"],
    ids=["no-connection", "401", "500"],
)
def test_snap_lookup_error_other_than_missing_branch_fails_without_fallback(tmp_path, error):
    proc = _run(
        tmp_path,
        env={
            "NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test",
            "FAKE_LOOKUP_ERROR": f"Error: GET https://artifacts.test/core/x/latest.tag failed: {error}",
        },
    )

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "records no Core" not in proc.stderr
    assert error in proc.stderr
    assert "Cannot resolve NEAT core artifact core@zz-missing-core-artifact-for-test:latest" in proc.stderr
    assert "using develop-latest" not in proc.stderr
    assert _lookups(tmp_path) == ["core@zz-missing-core-artifact-for-test:latest"]


def test_base_url_reaches_the_probe_the_fallback_lookup_and_the_install(tmp_path):
    base = f"{ARTIFACT_BASE}/custom"
    proc = _run(
        tmp_path,
        env={"NEAT_APPS_DEPENDENCY_BRANCH": "zz-missing-core-artifact-for-test",
             "SIMA_NEAT_BASE_URL": base},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert len(_lines(tmp_path / "sima-cli.log")) == 3
    assert _lines(tmp_path / "sima-cli-env.log") == [
        f"check=0 base={base}", f"check=0 base={base}", f"check=0 base={base}"
    ]


# Explicit selection


@pytest.mark.parametrize(
    ("neat_core", "target"),
    [
        ("zz-core-artifact-for-test-pinnedsha1", "core@zz-core-artifact-for-test:pinnedsha1"),
        ("zz-core-artifact-for-test:pinnedsha1", "core@zz-core-artifact-for-test:pinnedsha1"),
        ({"branch": "zz-core-artifact-for-test", "spec": "pinnedsha1"},
         "core@zz-core-artifact-for-test:pinnedsha1"),
        ({"ref": "feature/core-artifact", "spec": "pinnedsha2"}, "core@feature/core-artifact:pinnedsha2"),
        ({"branch": "main", "ref": "develop", "spec": "mainsha1"}, "core@main:mainsha1"),
        (" main-mainsha1 ", "core@main:mainsha1"),
        ({"ref": " main ", "spec": " mainsha1 "}, "core@main:mainsha1"),
    ],
    ids=["dash", "colon", "branch", "slash-ref", "branch-wins-over-ref", "string-strip",
         "object-strip"],
)
def test_explicit_revision_is_checked_and_installed(tmp_path, neat_core, target):
    proc = _run(tmp_path, neat_core=neat_core)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    ref, spec = target.removeprefix("core@").rsplit(":", 1)
    assert _lookups(tmp_path) == [target]
    assert _installs(tmp_path) == [_install(target)]
    assert _record(tmp_path) == _core(ref, spec)


@pytest.mark.parametrize(
    "neat_core", [{"ref": "feature/core-artifact"}, "feature/core-artifact-latest",
                  {"ref": "feature/core-artifact", "spec": " "}],
    ids=["object", "string", "blank-spec"],
)
def test_explicit_latest_resolves_to_exact_revision_without_fallback(tmp_path, neat_core):
    proc = _run(tmp_path, neat_core=neat_core)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == ["core@feature/core-artifact:latest"]
    assert _installs(tmp_path) == [_install("core@feature/core-artifact:featsha2")]


@pytest.mark.parametrize(
    "command",
    [[str(BUILD_SH), "--only-install-neat-core"], [str(BUILD_SH), "--only-resolve-neat-core"],
     [str(INSTALL_NEAT_CORE), "resolve"]],
    ids=["only-install", "only-resolve", "script-resolve"],
)
def test_explicit_unknown_revision_fails_and_removes_an_earlier_record(tmp_path, command):
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps/neat-core.json").write_text(json.dumps(_core("develop", "devsha1")))
    proc = _run(tmp_path, neat_core={"ref": "develop", "spec": "deadbeef00"}, command=command)

    # build.sh stops at the failed resolve, before it reads or installs a record.
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "records no Core" not in proc.stderr
    assert "Traceback" not in proc.stderr
    assert "HTTP 404\nERROR: unavailable NEAT core artifact" in proc.stderr
    assert (
        "unavailable NEAT core artifact 'develop-deadbeef00': "
        + _metadata_url("develop", "deadbeef00")
        in proc.stderr
    )
    assert _installs(tmp_path) == []
    assert not (tmp_path / "deps/neat-core.json").exists()


def test_unreachable_artifact_host_is_not_called_a_missing_artifact(tmp_path):
    proc = _run(
        tmp_path,
        neat_core={"ref": "develop", "spec": "devsha1"},
        env={"FAKE_SIMA_CLI_JSON_PATCH": json.dumps({"metadata_url": "http://127.0.0.1:1/x.json"})},
    )

    assert proc.returncode == 1
    assert "cannot reach artifact host:" in proc.stderr
    assert "Connection refused" in proc.stderr
    assert "ERROR: cannot check NEAT core artifact 'develop-devsha1'" in proc.stderr
    assert "unavailable" not in proc.stderr
    assert "HTTP " not in proc.stderr


@pytest.mark.parametrize(
    "command", [[str(BUILD_SH), "--only-install-neat-core"], [str(INSTALL_NEAT_CORE), "resolve"]],
    ids=["build", "script-resolve"],
)
def test_explicit_unknown_ref_fails_with_its_name_and_no_fallback(tmp_path, command):
    proc = _run(tmp_path, neat_core={"ref": "no-such-core-branch"}, command=command)

    assert proc.returncode == 1
    assert "Traceback" not in proc.stderr
    assert "records no Core" not in proc.stderr

    assert "Cannot resolve NEAT core artifact core@no-such-core-branch:latest" in proc.stderr
    assert _lookups(tmp_path) == ["core@no-such-core-branch:latest"]
    assert _installs(tmp_path) == []


@pytest.mark.parametrize("selection", ["manifest", "override"])
def test_explicit_selection_wins_over_apps_tag(tmp_path, selection):
    manifest_core = {"ref": "main", "spec": "mainsha1"} if selection == "manifest" else ""
    args = ["--only-install-neat-core"]
    if selection == "override":
        args += ["--neat-core-version", "main-mainsha1"]
    proc = _run(
        tmp_path,
        neat_core=manifest_core,
        args=args,
        env={"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.6.0"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lookups(tmp_path) == ["core@main:mainsha1"]
    assert _installs(tmp_path) == [_install("core@main:mainsha1")]


@pytest.mark.parametrize(
    ("override", "target"),
    [
        ("feature/core-artifact-pinnedsha2", "core@feature/core-artifact:pinnedsha2"),
        ("zz-core-artifact-for-test-latest", "core@zz-core-artifact-for-test:featsha1"),
    ],
)
def test_neat_core_version_overrides_manifest(tmp_path, override, target):
    proc = _run(
        tmp_path,
        neat_core={"ref": "main", "spec": "mainsha1"},
        args=["--only-install-neat-core", "--neat-core-version", override],
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installs(tmp_path) == [_install(target)]
    assert all("core@main:" not in lookup for lookup in _lookups(tmp_path))


@pytest.mark.parametrize("override", ["develop", "develop:devsha1", "-devsha1", "develop-"])
def test_neat_core_version_must_be_branch_version(tmp_path, override):
    proc = _run(tmp_path, args=["--only-install-neat-core", "--neat-core-version", override])

    assert proc.returncode != 0
    assert "--neat-core-version must" in proc.stderr
    assert _lines(tmp_path / "sima-cli.log") == []


@pytest.mark.parametrize(
    ("manifest_text", "error"),
    [
        (json.dumps({"neat-core": {"policy": "latest"}, "platform-version": "2.0.0"}),
         "unsupported neat-core.policy"),
        (json.dumps({"neat-core": {"policy": "pin"}, "platform-version": "2.0.0"}),
         "unsupported neat-core.policy"),
        (json.dumps({"neat-core": {"spec": "abc"}, "platform-version": "2.0.0"}),
         "neat-core object must define policy=snap or branch/ref."),
        (json.dumps({"neat-core": "develop", "platform-version": "2.0.0"}),
         "neat-core must use branch-version"),
        (json.dumps({"neat-core": "develop-", "platform-version": "2.0.0"}),
         "neat-core must provide exactly one non-empty branch and version."),
        (json.dumps({"neat-core": 42, "platform-version": "2.0.0"}),
         "must define neat-core as a string"),
        (json.dumps({"neat-core": [], "platform-version": "2.0.0"}),
         "must define neat-core as a string"),
        (json.dumps({"neat-core": ""}), "must define platform-version"),
        (json.dumps({"neat-core": "", "platform-version": "  "}), "must define platform-version"),
        ("", "Missing manifest"),
        ("{", "Invalid JSON in"),
    ],
    ids=["policy-latest", "policy-pin", "object-without-ref", "string-without-spec",
         "empty-spec", "number", "list", "no-platform", "blank-platform", "missing", "invalid-json"],
)
@pytest.mark.parametrize("override", [[], ["--neat-core-version", "main-mainsha1"]],
                         ids=["manifest", "with-override"])
def test_invalid_manifest_fails_before_sima_cli(tmp_path, manifest_text, error, override):
    proc = _run(tmp_path, manifest_text=manifest_text, args=["--only-install-neat-core", *override])

    assert proc.returncode != 0
    assert error in proc.stderr
    assert "Traceback" not in proc.stderr
    assert _lines(tmp_path / "sima-cli.log") == []


# sima-cli's answer


@pytest.mark.parametrize(
    ("neat_core", "patch"),
    [
        ("", {"repository": "models"}),
        ("", {"ref": "develop"}),
        ("", {"resolved_spec": ""}),
        ({"ref": "zz-core-artifact-for-test", "spec": "pinnedsha1"}, {"resolved_spec": "othersha"}),
        ("", {"metadata_url": ""}),
    ],
    ids=["repository", "ref", "empty-spec", "other-spec", "no-url"],
)
def test_unverified_sima_cli_answer_fails_without_fallback(tmp_path, neat_core, patch):
    proc = _run(
        tmp_path,
        neat_core=neat_core,
        env={
            "NEAT_APPS_DEPENDENCY_BRANCH": "zz-core-artifact-for-test",
            "FAKE_SIMA_CLI_JSON_PATCH": json.dumps(patch),
        },
    )

    assert proc.returncode != 0
    assert "Cannot verify resolved Core metadata identity." in proc.stderr
    assert "using develop-latest" not in proc.stderr
    assert _installs(tmp_path) == []


@pytest.mark.parametrize("stdout", ["Environment: sdk", "Environment: sdk\n{not json"],
                         ids=["no-json", "broken-json"])
def test_sima_cli_answer_without_json_is_named(tmp_path, stdout):
    proc = _run(tmp_path, env={"FAKE_LOOKUP_STDOUT": stdout})

    assert proc.returncode != 0
    assert "sima-cli printed no Core metadata." in proc.stderr
    assert "Traceback" not in proc.stderr
    assert _installs(tmp_path) == []


@pytest.mark.parametrize(
    ("env_value", "base_url"),
    [(None, None), ("prod", None), ("staging", None), ("dev", "custom")],
)
def test_artifact_environment_and_base_url_reach_every_sima_cli_call(tmp_path, env_value, base_url):
    env = {"NEAT_APPS_DEPENDENCY_BRANCH": "develop"}
    if env_value:
        env["NEAT_VULCAN_ENV"] = env_value
    if base_url:
        env["SIMA_NEAT_BASE_URL"] = f"{ARTIFACT_BASE}/{base_url}"
    proc = _run(
        tmp_path,
        args=["--only-install-neat-core", "--neat-core-version", "main-mainsha1"],
        env=env,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    expected_env = env_value or "production"
    assert _lines(tmp_path / "sima-cli.log") == [
        f"neat install --env {expected_env} -t minimal --json core@main:mainsha1",
        _install("core@main:mainsha1", expected_env),
    ]
    expected_base = env.get("SIMA_NEAT_BASE_URL", "-")
    assert _lines(tmp_path / "sima-cli-env.log") == [
        f"check=0 base={expected_base}", f"check=0 base={expected_base}"
    ]


# sima-cli's update check


def test_calls_without_a_terminal_skip_the_update_question_that_would_abort_them(tmp_path):
    # stdin is a pipe here. Only an install with a terminal keeps sima-cli's question.
    proc = _run(tmp_path, env={"FAKE_UPDATE_AVAILABLE": "1"}, stdin="n\n")

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _checks(tmp_path) == ["check=0", "check=0"]
    assert _installs(tmp_path) == [_install("core@develop:devsha1")]


@pytest.mark.parametrize("answer", [b"n\n", b"y\n"])
def test_install_from_a_terminal_keeps_the_update_question(tmp_path, answer):
    leader, follower = pty.openpty()
    try:
        os.write(leader, answer)
        proc = _run(tmp_path, env={"FAKE_UPDATE_AVAILABLE": "1"}, stdin=follower)
    finally:
        os.close(follower)
        os.close(leader)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _checks(tmp_path) == ["check=0", "check=1"]
    assert _installs(tmp_path) == [_install("core@develop:devsha1")]


@pytest.mark.parametrize(
    ("caller_env", "checks"),
    [({"SIMA_CLI_CHECK_FOR_UPDATE": "0"}, ["check=0", "check=0"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "1"}, ["check=1", "check=1"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "True"}, ["check=1", "check=1"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "yes"}, ["check=1", "check=1"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "ON"}, ["check=1", "check=1"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "0"}, ["check=0", "check=0"]),
     ({"SIMA_CLI_AUTO_ACCEPT_UPDATE": "1x"}, ["check=0", "check=0"])],
    ids=["check-off", "auto-accept-1", "auto-accept-true", "auto-accept-yes", "auto-accept-on",
         "auto-accept-0", "auto-accept-1x"],
)
def test_lookups_keep_the_callers_update_setting(tmp_path, caller_env, checks):
    proc = _run(tmp_path, env={"FAKE_UPDATE_AVAILABLE": "1", **caller_env})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _checks(tmp_path) == checks


def test_caller_who_turns_the_update_check_on_sees_its_abort(tmp_path):
    # The lookups read /dev/null, so an answer on stdin cannot reach them.
    proc = _run(tmp_path, env={"FAKE_UPDATE_AVAILABLE": "1", "SIMA_CLI_CHECK_FOR_UPDATE": "1"},
                stdin="y\n")

    assert proc.returncode != 0
    assert "Aborted!" in proc.stderr
    assert _installs(tmp_path) == []


# Install


def test_install_leaves_no_scratch_files(tmp_path):
    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    install_dir = Path(_lines(tmp_path / "sima-cli-cwd.log")[0])
    assert install_dir.parent == tmp_path / "tmp"
    assert install_dir.name.startswith("neat-apps-core-install.")
    assert list((tmp_path / "tmp").iterdir()) == []
    assert "NEAT core installed successfully (core@develop:devsha1)." in proc.stdout


@pytest.mark.parametrize(
    "output",
    [
        # sima-cli prints this and exits 0 when Core's installer script exits 255.
        "Installation script exited with -1. Aborted.\nInstallation completed successfully.",
        # sima-cli exits 0 without installing, for example for low disk space.
        "Not enough disk space!",
    ],
    ids=["installer-abort", "no-completion"],
)
def test_install_that_sima_cli_reports_with_exit_zero_fails(tmp_path, output):
    proc = _run(tmp_path, env={"FAKE_SIMA_CLI_INSTALL_OUTPUT": output})

    assert proc.returncode != 0
    assert "sima-cli did not complete the NEAT core install." in proc.stderr
    assert "installed successfully" not in proc.stdout


def test_sima_cli_install_failure_keeps_its_exit_status(tmp_path):
    proc = _run(tmp_path, env={"FAKE_SIMA_CLI_INSTALL_STATUS": "7"})

    assert proc.returncode == 7
    assert "installed successfully" not in proc.stdout


def test_missing_sima_cli_is_named(tmp_path):
    bin_dir = tmp_path / "bin"
    # A directory keeps _run from writing the fake and is no command.
    (bin_dir / "sima-cli").mkdir(parents=True)
    proc = _run(tmp_path, env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:/usr/bin:/bin"})

    assert proc.returncode != 0
    assert "requires sima-cli on PATH or SIMA_CLI_BIN" in proc.stderr


def test_missing_sima_cli_exits_one_before_any_lookup(tmp_path):
    bin_dir = tmp_path / "bin"
    (bin_dir / "sima-cli").mkdir(parents=True)
    proc = _run(tmp_path, command=[str(INSTALL_NEAT_CORE), "resolve"],
                env={"HOME": str(tmp_path), "PATH": f"{bin_dir}:/usr/bin:/bin"})

    assert proc.returncode == 1
    assert "Cannot resolve" not in proc.stderr
    assert not (tmp_path / "deps/neat-core.json").exists()


def test_sima_cli_is_found_in_the_user_install_path(tmp_path):
    bin_dir = tmp_path / "bin"
    (bin_dir / "sima-cli").mkdir(parents=True)
    _write_tool(tmp_path / "home/.local/bin", "sima-cli", FAKE_SIMA_CLI)
    proc = _run(tmp_path, env={"HOME": str(tmp_path / "home"), "PATH": f"{bin_dir}:/usr/bin:/bin"})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installs(tmp_path) == [_install("core@develop:devsha1")]


def test_sima_cli_lookup_names_the_devkit_install_path():
    # No test can create /data, so this checks the path the DevKit image installs to.
    assert "    /data/sima-cli/.venv/bin/sima-cli \\\n" in INSTALL_NEAT_CORE.read_text()


def test_non_executable_sima_cli_bin_falls_back_to_path(tmp_path):
    other = tmp_path / "other-sima-cli"
    other.write_text("#!/bin/sh\nexit 9\n")
    proc = _run(tmp_path, env={"SIMA_CLI_BIN": str(other)})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert len(_lines(tmp_path / "sima-cli.log")) == 2


def test_sima_cli_bin_wins_over_path(tmp_path):
    chosen = _write_tool(tmp_path / "chosen", "sima-cli", FAKE_SIMA_CLI.replace(
        'os.environ["FAKE_SIMA_CLI_LOG"]', 'os.environ["FAKE_SIMA_CLI_LOG"] + ".chosen"'))
    proc = _run(tmp_path, env={"SIMA_CLI_BIN": str(chosen)})

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lines(tmp_path / "sima-cli.log") == []
    assert len(_lines(tmp_path / "sima-cli.log.chosen")) == 2


# Record


def test_record_is_the_format_packaging_ships(tmp_path):
    proc = _run(tmp_path)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (tmp_path / "deps/neat-core.json").read_text(encoding="utf-8") == textwrap.dedent(
        """\
        {
          "neat-core": {
            "ref": "develop",
            "spec": "devsha1"
          }
        }
        """
    )


def test_only_resolve_prints_only_the_target_and_neither_installs_nor_builds(tmp_path):
    build_dir = tmp_path / "build"
    proc = _run(
        tmp_path,
        args=["--only-resolve-neat-core", "--build-dir", str(build_dir)],
        env={"NEAT_APPS_DEPENDENCY_BRANCH": "feature/core-artifact"},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    # CI reads this line as --neat-core-version.
    assert proc.stdout == "feature/core-artifact-featsha2\n"
    assert _record(tmp_path) == _core("feature/core-artifact", "featsha2")
    assert _installs(tmp_path) == []
    assert not build_dir.exists()
    assert list((tmp_path / "tmp").iterdir()) == []


def test_only_install_installs_without_build(tmp_path):
    build_dir = tmp_path / "build"
    proc = _run(tmp_path, args=["--only-install-neat-core", "--build-dir", str(build_dir)])

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert len(_installs(tmp_path)) == 1
    assert not build_dir.exists()


def test_recorded_revision_installs_after_latest_moves(tmp_path):
    # A caller can resolve, do other work, and then install that exact record.
    resolved = _run(tmp_path, args=["--only-resolve-neat-core"])
    assert resolved.returncode == 0, resolved.stderr + resolved.stdout

    proc = _run(
        tmp_path,
        args=["--only-install-neat-core", "--neat-core-version", resolved.stdout.strip()],
        env={"FAKE_PUBLISHED": json.dumps({**PUBLISHED, "develop": "devsha2"})},
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _installs(tmp_path) == [_install("core@develop:devsha1")]
    assert _record(tmp_path) == _core("develop", "devsha1")


def test_script_install_installs_the_recorded_revision(tmp_path):
    (tmp_path / "deps").mkdir()
    (tmp_path / "deps/neat-core.json").write_text(json.dumps(_core("release-0.6", "recordsha1")))
    proc = _run(tmp_path, command=[str(INSTALL_NEAT_CORE), "install"])

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert _lines(tmp_path / "sima-cli.log") == [_install("core@release-0.6:recordsha1")]


@pytest.mark.parametrize("record", [None, "{", '{"neat-core": {"ref": "develop"}}', '{"neat-core": "x"}'],
                         ids=["missing", "invalid-json", "no-spec", "string"])
def test_script_install_without_a_valid_record_names_the_missing_step(tmp_path, record):
    (tmp_path / "deps").mkdir()
    if record is not None:
        (tmp_path / "deps/neat-core.json").write_text(record)
    proc = _run(tmp_path, command=[str(INSTALL_NEAT_CORE), "install"])

    assert proc.returncode != 0
    assert "records no Core" in proc.stderr
    assert "Traceback" not in proc.stderr
    assert _lines(tmp_path / "sima-cli.log") == []


# Command line


@pytest.mark.parametrize(
    ("command", "error"),
    [(["resolve", "--force"], "Unknown argument: --force"), (["update"], "Usage:"),
     ([], "Usage:"), (["install", "--neat-core-version", "main-x"], "works only with resolve"),
     (["resolve", "--neat-core-version"], "takes a <ref>-<spec> value"),
     (["resolve", "--neat-core-version", ""], "takes a <ref>-<spec> value")],
    ids=["unknown-option", "unknown-command", "no-command", "override-on-install",
         "override-without-value", "empty-override"],
)
def test_script_rejects_misuse(tmp_path, command, error):
    proc = _run(tmp_path, command=[str(INSTALL_NEAT_CORE), *command])

    assert proc.returncode == 1
    assert error in proc.stderr
    assert "unbound variable" not in proc.stderr
    assert _lines(tmp_path / "sima-cli.log") == []


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_script_help_prints_usage(tmp_path, flag):
    proc = _run(tmp_path, command=[str(INSTALL_NEAT_CORE), flag])

    assert proc.returncode == 0
    assert proc.stdout.startswith("Usage: scripts/install-neat-core.sh resolve")


@pytest.mark.parametrize("value", [[], [""]], ids=["no-value", "empty-value"])
def test_build_neat_core_version_without_value_is_named(tmp_path, value):
    proc = _run(tmp_path, args=["--only-install-neat-core", "--neat-core-version", *value])

    assert proc.returncode != 0
    assert "--neat-core-version takes a <ref>-<spec> value" in proc.stderr
    assert "unbound variable" not in proc.stderr


def test_build_banner_names_the_recorded_core(tmp_path):
    # A failing cmake stops the build right after the banner.
    _write_tool(tmp_path / "bin", "cmake", "#!/bin/sh\nexit 3\n")
    proc = _run(tmp_path, args=["--all", "--no-cpp", "--build-dir", str(tmp_path / "build")])

    assert proc.returncode == 3
    assert "  NEAT core target      : develop:devsha1\n" in proc.stdout


def test_default_paths_record_in_the_checkouts_deps(tmp_path):
    tree = tmp_path / "tree"
    for name in ("build.sh", "scripts/install-neat-core.sh", "scripts/artifact-refs.sh"):
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(APPS_ROOT / name, tree / name)
    (tree / "deps").mkdir()
    (tree / "deps/manifest.json").write_text(json.dumps({"neat-core": {"ref": "main"},
                                                          "platform-version": "2.0.0"}))
    proc = _run(tmp_path, command=[str(tree / "build.sh"), "--only-resolve-neat-core"],
                env={"NEAT_APPS_TEST_DEPS_DIR": ""}, cwd=tree)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout == "main-mainsha1\n"
    assert json.loads((tree / "deps/neat-core.json").read_text()) == _core("main", "mainsha1")
    assert _lookups(tmp_path) == ["core@main:latest"]
    assert _lines(tmp_path / "sima-cli.log")[0].startswith("neat install --env production ")


def _workflow_step_text(job: str, step: str | None) -> str:
    jobs = yaml.safe_load((APPS_ROOT / ".github/workflows/vulcan-ci.yml").read_text())["jobs"]
    if step is None:
        return jobs[job]["with"]["build_command"]
    return next(item["run"] for item in jobs[job]["steps"] if item.get("name") == step)


@pytest.mark.parametrize(
    ("job", "step", "line", "env_line"),
    [("build", None, '-e SIMA_NEAT_BASE_URL="${artifact_base_url%/}" \\',
      '-e NEAT_VULCAN_ENV="${{ needs.resolve-vulcan-config.outputs.vulcan_env }}" \\'),
     ("test-runtime-shard", "Install Apps Core for runtime tests",
      'SIMA_NEAT_BASE_URL="${artifact_base_url%/}" \\',
      'NEAT_VULCAN_ENV="${{ needs.resolve-vulcan-config.outputs.vulcan_env }}" \\')],
    ids=["build", "devkit-core-install"],
)
def test_ci_passes_the_artifact_root_to_sima_cli(job, step, line, env_line):
    text = _workflow_step_text(job, step)
    lines = [entry.strip() for entry in text.splitlines()]

    assert line in lines
    assert env_line in lines
    assert not any("SIMA_NEAT_BASE_URL" in entry and "/core" in entry for entry in lines)
    assert "NEAT_ARTIFACTS_BASE_URL" not in text
