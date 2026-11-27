"""scripts/get-example.sh against a local copy of the published bundle index."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = APPS_ROOT / "scripts" / "get-example.sh"
BUNDLE = "neat-apps-team-demo-abc1234.tar.gz"
BUILD = "abc1234def56"
FILES = {
    "examples/demo/sample/README.md": "Images are in assets/datasets/coco\n",
    "examples/demo/sample/src/common/config.yaml": "images: assets/datasets/coco\n",
    "examples/demo/dotted/README.md": "The boards are in assets/datasets/pcb.\n",
    "examples/demo/big/README.md": "Images are in assets/datasets/coco2017\n",
    "examples/demo/mentions/README.md": "Run examples/demo/sample first.\n",
    "examples/demo/combo/README.md": "combo\n",
    "examples/demo/combo/src/common/run-all.sh": "for app in demo/sample demo/other; do\n",
    "examples/demo/near/README.md": "near\n",
    "examples/demo/near/src/common/config.yaml": "notes: demo/samples is not an example\n",
    "examples/demo/xnear/README.md": "xnear\n",
    "examples/demo/xnear/src/common/config.yaml": "notes: xdemo/sample is not an example\n",
    "examples/demo/dotname/README.md": "Images are in assets/datasets/coco.v2\n",
    "examples/demo/dashname/README.md": "Images are in assets/datasets/coco-v2\n",
    "examples/demo/twin/README.md": "twin\n",
    "examples/other/twin/README.md": "twin\n",
    "examples/demo/other/README.md": "other\n",
    "examples/demo/other/data/sample/x.txt": "a folder named like an example\n",
    "examples/genai/neat-genai-studio/README.md": "studio\n",
    "support/python/sima_examples/config.py": "",
    "scripts/get-example.sh": "#!/usr/bin/env bash\n",
    "deps/neat-apps.json": "{}",
    "assets/datasets/coco/a.jpg": "jpg",
    "assets/datasets/coco2017/c.jpg": "jpg",
    "assets/datasets/pcb/b.jpg": "jpg",
    "assets/datasets/coco.v2/d.jpg": "jpg",
    "assets/datasets/coco-v2/e.jpg": "jpg",
}
OLDER_BUNDLE = ("support/", "deps/", "scripts/")


def make_index(
    tmp_path: Path,
    ref: str = "team/demo",
    checksum: Callable[[str], str] | None = None,
    skip: tuple[str, ...] = (),
    reverse: bool = False,
    links: dict[str, str] | None = None,
    extra: dict[str, str] | None = None,
) -> Path:
    """Lay out <base>/apps/<ref key>/... as artifacts.neat.sima.ai does for one ref."""
    tree = tmp_path / "tree" / "prebuilt-apps"
    for name, text in {**FILES, **(extra or {})}.items():
        if name.startswith(skip):
            continue
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    for name, target in (links or {}).items():
        (tree / name).symlink_to(target)
    # The ref is URL-encoded, then encoded again in the path: team/demo -> team%252Fdemo.
    build = tmp_path / "index" / "apps" / ref.replace("/", "%252F") / BUILD
    build.mkdir(parents=True)
    with tarfile.open(build / BUNDLE, "w:gz") as tar:
        if reverse:
            # Members in reverse order, so a listing is sorted only if the script sorts it.
            for path in sorted(tree.parent.rglob("*"), reverse=True):
                tar.add(path, arcname=str(path.relative_to(tree.parent)), recursive=False)
        else:
            tar.add(tree, arcname="prebuilt-apps")
    digest = hashlib.sha256((build / BUNDLE).read_bytes()).hexdigest()
    # A key that differs only where BUNDLE has dots, so a pattern that does not escape them
    # also reads the wrong checksum.
    decoy = BUNDLE.replace(".", "_")
    # CI also uploads the test overlay. The runtime bundle is the first neat-apps-*.tar.gz.
    metadata = {
        "resources": [BUNDLE, "neat-apps-tests.tar.gz"],
        "resources-checksum": {decoy: "f" * 64, BUNDLE: checksum(digest) if checksum else digest,
                               "neat-apps-tests.tar.gz": "e" * 64},
    }
    (build / "metadata.json").write_text(json.dumps(metadata, indent=4))
    (build.parent / "latest.tag").write_text(f"{BUILD}\n")
    return tmp_path / "index"


class IndexServer:
    """Serves an index folder over HTTP: 403 for a missing key, as the published index does."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: dict[str, list[int]] = {}
        self.truncated: set[str] = set()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                planned = server.failures.get(self.path)
                code = (planned.pop(0) if len(planned) > 1 else planned[0]) if planned else 0
                if code:
                    self.send_error(code)
                    return
                path = server.root / self.path.lstrip("/")
                if not path.is_file():
                    self.send_error(403)
                    return
                body = path.read_bytes()
                self.send_response(200)
                cut = self.path in server.truncated
                self.send_header("Content-Length", str(len(body) + (1000 if cut else 0)))
                self.end_headers()
                self.wfile.write(body[: len(body) // 2] if cut else body)

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def fail(self, path: str, *codes: int) -> None:
        """Answer path with these codes in turn, 0 to serve it. The last one repeats."""
        self.failures[path] = list(codes)


@pytest.fixture
def server(tmp_path: Path):
    index = make_index(tmp_path)
    served = IndexServer(index)
    yield served
    served.httpd.shutdown()


def path_without(tmp_path: Path, *tools: str, extra: dict[str, str] | None = None) -> str:
    """A PATH with every system tool except the named ones, plus extra name -> target links."""
    bin_dir = tmp_path / ("bin-" + "-".join(tools + tuple(extra or {})))
    if not bin_dir.exists():
        bin_dir.mkdir()
        for folder in ("/usr/bin", "/bin"):
            for tool in Path(folder).iterdir():
                link = bin_dir / tool.name
                if tool.name not in tools and not os.path.lexists(link):
                    link.symlink_to(tool)
        for name, target in (extra or {}).items():
            (bin_dir / name).unlink(missing_ok=True)
            (bin_dir / name).symlink_to(target)
    return str(bin_dir)


def logging_tool(tmp_path: Path, tool: str) -> tuple[Path, Path]:
    """A wrapper that logs its arguments to a file, then runs the real tool."""
    log, script = tmp_path / f"{tool}.log", tmp_path / f"log-{tool}"
    script.write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nexec {shutil.which(tool)} "$@"\n')
    script.chmod(0o755)
    return script, log


def fetched_before(folder: Path, *files: str) -> Path:
    """A folder as an earlier fetch left it, with deps/neat-apps.json and the given files."""
    for name in ("deps/neat-apps.json", *files):
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text("user data")
    return folder


TOOLS = {
    "curl": lambda tmp: "/usr/bin:/bin",
    "wget": lambda tmp: path_without(tmp, "curl"),
    "busybox-wget": lambda tmp: path_without(tmp, "curl", extra={"wget": shutil.which("busybox") or ""}),
}


def tool_path(tmp_path: Path, tool: str) -> str:
    needed = {"curl": "curl", "wget": "wget", "busybox-wget": "busybox"}[tool]
    if not shutil.which(needed):
        pytest.skip(f"{needed} is not installed")
    return TOOLS[tool](tmp_path)


def get_example(
    tmp_path: Path, index: Path | str, *args: str, path: str = "/usr/bin:/bin",
    cwd: Path | None = None, **env: str,
) -> subprocess.CompletedProcess:
    run_env = {"PATH": path, "HOME": str(tmp_path), "NEAT_APPS_ARTIFACTS_URL": str(index), **env}
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=cwd or tmp_path, env=run_env, capture_output=True, text=True
    )


def files_under(folder: Path) -> set[str]:
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


SAMPLE_FILES = {
    "examples/demo/sample/README.md",
    "examples/demo/sample/src/common/config.yaml",
    "support/python/sima_examples/config.py",
    "scripts/get-example.sh",
    "deps/neat-apps.json",
    "assets/datasets/coco/a.jpg",
}


@pytest.mark.parametrize("ref", ["team/demo", "v1.0"])
def test_fetches_the_example_in_the_installed_layout(tmp_path: Path, ref: str) -> None:
    index = make_index(tmp_path, ref=ref)

    result = get_example(tmp_path, index, "--ref", ref, "sample")

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "sample") == SAMPLE_FILES
    assert "Next: cd sample; APP_DIR=examples/demo/sample\n" in result.stdout


@pytest.mark.parametrize(
    ("example", "dataset"),
    [("dotted", "assets/datasets/pcb/b.jpg"), ("big", "assets/datasets/coco2017/c.jpg")],
)
def test_a_dataset_is_matched_by_its_whole_name(tmp_path: Path, example: str, dataset: str) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", example)

    assert result.returncode == 0, result.stderr
    assert {name for name in files_under(tmp_path / example) if name.startswith("assets/")} == {dataset}


def test_an_older_bundle_without_support_deps_or_the_script_still_fetches(tmp_path: Path) -> None:
    index = make_index(tmp_path, skip=OLDER_BUNDLE)

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample")

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "sample") == {
        "examples/demo/sample/README.md",
        "examples/demo/sample/src/common/config.yaml",
        "assets/datasets/coco/a.jpg",
    }


def test_an_example_that_runs_another_needs_the_whole_bundle(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "combo")

    assert result.returncode == 1
    assert "combo needs the whole bundle: sima-cli neat install apps" in result.stderr
    assert not (tmp_path / "combo").exists()


@pytest.mark.parametrize("example", ["mentions", "near"])
def test_a_readme_mention_or_a_longer_name_is_not_another_example(tmp_path: Path, example: str) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", example)

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / example / "examples/demo/sample").exists()


def test_a_pinned_full_commit_skips_latest_tag(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    (index / "apps/team%252Fdemo/latest.tag").unlink()

    result = get_example(tmp_path, index, "--ref", f"team/demo:{BUILD}" + "0" * 28, "demo/sample")

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("args", "env", "dest"),
    [
        (("--branch", "team/demo", "sample"), {}, "sample"),
        (("sample",), {"NEAT_APPS_BRANCH": "team/demo"}, "sample"),
        (("--ref", "team/demo", "examples/demo/sample"), {}, "sample"),
        (("--ref", "team/demo", "demo/sample/"), {}, "sample"),
        (("--ref", "team/demo", "--dest", "out", "demo/sample"), {}, "out"),
        (("--ref", "team/demo", "demo/sample"), {"NEAT_APPS_EXAMPLE_DEST_DIR": "out"}, "out"),
    ],
)
def test_keeps_the_options_of_the_source_archive_version(
    tmp_path: Path, args: tuple[str, ...], env: dict[str, str], dest: str
) -> None:
    result = get_example(tmp_path, make_index(tmp_path), *args, **env)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / dest / "examples/demo/sample/README.md").is_file()


@pytest.mark.parametrize(
    "alias",
    ["genai-studio", "multimodal-assistant", "multimodal-assist", "examples/genai/multimodal-assistant"],
)
def test_genai_studio_aliases_fetch_the_studio(tmp_path: Path, alias: str) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", alias)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "neat-genai-studio/examples/genai/neat-genai-studio/README.md").is_file()


def test_an_archive_url_skips_the_index(tmp_path: Path) -> None:
    make_index(tmp_path)
    bundle = tmp_path / "index/apps/team%252Fdemo" / BUILD / BUNDLE

    result = get_example(tmp_path, tmp_path / "no-index", "sample", NEAT_APPS_ARCHIVE_URL=str(bundle))

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "sample/examples/demo/sample/README.md").is_file()


def test_help_prints_the_usage(tmp_path: Path) -> None:
    result = get_example(tmp_path, tmp_path, "--help")

    assert result.returncode == 0
    assert result.stdout.startswith("Usage: get-example.sh [--ref <branch|tag>")


def test_refuses_the_repository_url_it_no_longer_reads(tmp_path: Path) -> None:
    result = get_example(
        tmp_path, make_index(tmp_path), "--ref", "team/demo", "sample",
        NEAT_APPS_REPO_URL="https://github.com/sima-neat/apps.git",
    )

    assert result.returncode == 2
    assert "NEAT_APPS_REPO_URL is not read" in result.stderr
    assert not (tmp_path / "sample").exists()


@pytest.mark.parametrize(
    ("args", "rc", "message"),
    [
        (("--ref", "team/demo", "--verbose", "sample"), 2, "Unknown argument: --verbose"),
        (("--ref", "team/demo", "sample", "other"), 2, "Only one example at a time."),
        (("--ref", "team/demo", "sample", "--dest"), 2, "--dest needs a value."),
        (("--ref", "team/demo", "../sample"), 2, "Not an example name: ../sample"),
        (("--ref", "team/demo", "--dest", "x/../y", "sample"), 1, "Refusing destination: x/../y"),
        (("--ref", "a b", "sample"), 2, "Unsupported ref: a b"),
        (("--ref", "team/demo:abc1234", "sample"), 2, "give at least 12 characters of the commit"),
        (("--ref", "team/demo:" + "a" * 41, "sample"), 2, "give at least 12 characters of the commit"),
        (("--ref", "team/demo", "--dest", "file", "sample"), 1, "Destination exists: file"),
        (("sample",), 1, "No published Apps bundle for main ("),
        (("--ref", "team/demo", "a/b/c"), 2, "Not an example name: a/b/c"),
        (("--ref", "main:xyz", "sample"), 2, "Unsupported ref: main:xyz"),
        ((), 2, "Usage: get-example.sh [--ref <branch|tag>"),
        (("--ref", "team/demo", "missing"), 1, "Example not in the bundle: missing. It has:"),
        (("--ref", "team/demo", "demo"), 1, "Example not in the bundle: demo"),
        (("--ref", "no/such", "sample"), 1, "Published branches: "),
        (("--ref", "team/demo:" + "b" * 12, "sample"), 1, f"No build {'b' * 12} of team/demo ("),
    ],
)
def test_rejects_with_a_clear_message(tmp_path: Path, args: tuple[str, ...], rc: int, message: str) -> None:
    (tmp_path / "file").write_text("")

    result = get_example(tmp_path, make_index(tmp_path), *args)

    assert result.returncode == rc
    assert message in result.stderr
    assert not (tmp_path / "sample").exists()


def test_an_unknown_example_lists_the_bundle(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "missing")

    listed = result.stderr.splitlines()[1:]
    assert listed == sorted(listed)
    assert {"  demo/sample", "  demo/twin", "  other/twin", "  genai/neat-genai-studio"} <= set(listed)


def test_an_ambiguous_name_asks_for_the_category(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "twin")

    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "More than one example is named twin. Give <category>/<name>:",
        "  demo/twin",
        "  other/twin",
    ]


@pytest.mark.parametrize(
    "dest", ["link", "link/", "link//", "link/.", "link/./", "./link/.", "x/../link/."]
)
def test_never_replaces_a_symlink(tmp_path: Path, dest: str) -> None:
    (tmp_path / "x").mkdir()
    # The link's target looks like an earlier fetch, so only the symlink check refuses it.
    fetched_before(tmp_path / "real", "keep")
    (tmp_path / "link").symlink_to("real")

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--dest", dest, "--force", "sample")

    assert result.returncode == 1
    assert "Refusing" in result.stderr
    assert files_under(tmp_path / "real") == {"deps/neat-apps.json", "keep"}


@pytest.mark.parametrize("dest", ["{home}", "{top}", ".", "..", "{cwd}", "{work}", "/"])
def test_force_never_replaces_home_cwd_or_a_folder_above_them(tmp_path: Path, dest: str) -> None:
    home, cwd = tmp_path / "top" / "home", tmp_path / "work" / "sub"
    home.mkdir(parents=True)
    cwd.mkdir(parents=True)
    (home / "keep").write_text("")
    index = make_index(tmp_path)

    result = get_example(
        tmp_path, index, "--ref", "team/demo", "--force", "--dest",
        dest.format(home=home, top=home.parent, cwd=cwd, work=cwd.parent), "sample",
        cwd=cwd, HOME=str(home),
    )

    assert result.returncode == 1
    assert "Refusing destination" in result.stderr
    assert (home / "keep").exists() and cwd.is_dir()


def test_force_replaces_the_folder_and_keeps_its_models(tmp_path: Path) -> None:
    old = fetched_before(tmp_path / "sample", "models/m.tar.gz", "config.local.yaml", "locked/f")
    (old / "locked").chmod(0o555)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--force", "sample")

    assert result.returncode == 0, result.stderr
    assert files_under(old) == SAMPLE_FILES | {"models/m.tar.gz"}
    assert (old / "models/m.tar.gz").read_text() == "user data"
    assert not list(tmp_path.glob(".get-example.*"))


@pytest.mark.parametrize("old", ["file", "empty folder", "one fetched example", "locked subfolder", "linked models"])
def test_force_replaces_a_file_an_empty_folder_or_one_fetch(tmp_path: Path, old: str) -> None:
    if old == "file":
        (tmp_path / "sample").write_text("")
    elif old == "empty folder":
        (tmp_path / "sample").mkdir()
    elif old == "one fetched example":
        fetched_before(tmp_path / "sample", "examples/demo/sample/README.md")
    elif old == "locked subfolder":
        fetched_before(tmp_path / "sample", "sandbox/out.png")
        (tmp_path / "sample" / "sandbox").chmod(0o000)
    else:
        # models/ may be a link to a read-only store. The swap moves the link, not the store.
        (tmp_path / "store").mkdir()
        (tmp_path / "store" / "m").write_text("")
        (tmp_path / "store").chmod(0o555)
        fetched_before(tmp_path / "sample")
        (tmp_path / "sample" / "models").symlink_to(tmp_path / "store")

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--force", "sample")

    (tmp_path / "store").exists() and (tmp_path / "store").chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert {f for f in files_under(tmp_path / "sample") if not f.startswith("models/")} == SAMPLE_FILES
    assert not list(tmp_path.glob(".get-example.*"))
    if old == "linked models":
        assert (tmp_path / "sample" / "models").is_symlink()
        assert (tmp_path / "sample" / "models" / "m").exists()


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("plain", "Refusing to replace sample: it has no deps/neat-apps.json, so it is not a fetched example. "
                  "If an older get-example.sh fetched it, move it away and fetch again."),
        ("install", "Refusing to replace sample: it holds more than one example, such as a sima-cli install."),
        ("one category", "Refusing to replace sample: it holds more than one example, such as a sima-cli install."),
        ("unreadable", "Refusing to replace sample: it is not readable."),
        ("not searchable", "Refusing to replace sample: it is not readable."),
        ("read-only models", "Make sample/models writable, so --force can keep it."),
    ],
)
def test_force_refuses_a_folder_it_cannot_replace_safely(tmp_path: Path, setup: str, message: str) -> None:
    if setup == "plain":
        (tmp_path / "sample").mkdir()
        (tmp_path / "sample" / "notes.txt").write_text("user data")
    elif setup == "install":
        fetched_before(tmp_path / "sample", "examples/a/one/README.md", "examples/b/two/README.md")
    elif setup == "one category":
        fetched_before(tmp_path / "sample", "examples/a/one/README.md", "examples/a/two/README.md")
    elif setup in ("unreadable", "not searchable"):
        fetched_before(tmp_path / "sample")
        (tmp_path / "sample").chmod(0o311 if setup == "unreadable" else 0o644)
    else:
        fetched_before(tmp_path / "sample", "models/m")
        (tmp_path / "sample" / "models").chmod(0o555)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--force", "sample")

    (tmp_path / "sample").chmod(0o755)
    (tmp_path / "sample" / "models").chmod(0o755) if (tmp_path / "sample" / "models").exists() else None
    assert result.returncode == 1
    assert message in result.stderr
    assert "user data" in [p.read_text() for p in (tmp_path / "sample").rglob("*") if p.is_file()]


def test_refuses_a_folder_right_under_the_root(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--dest", "/no-such-top-folder", "sample")

    assert result.returncode == 1
    assert "Refusing destination: /no-such-top-folder" in result.stderr


@pytest.mark.parametrize(
    ("tool", "rule", "kept"),
    [
        ("cp", 'case "$*" in *.get-example.*) exit 1 ;; esac', True),
        ("mv", 'case "$2:$3" in */sample:*/old) exit 1 ;; esac', True),
        ("mv", 'case "$2:$3" in */old/models:*/new/models) exit 1 ;; esac', True),
        ("mv", 'case "$2:$3" in */new:*/sample) exit 1 ;; esac', True),
        ("mv", 'case "$2:$3" in */new:*/sample|*/old:*/sample) exit 1 ;; esac', False),
    ],
)
def test_a_failed_copy_or_swap_keeps_one_copy_and_says_where(
    tmp_path: Path, tool: str, rule: str, kept: bool
) -> None:
    fetched_before(tmp_path / "sample", "models/m", "old")
    fake_bin = Path(path_without(tmp_path, tool))
    (fake_bin / tool).write_text(f'#!/bin/sh\n{rule}\nexec {shutil.which(tool)} "$@"\n')
    (fake_bin / tool).chmod(0o755)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--force", "sample", path=str(fake_bin))

    assert result.returncode == 1
    stages = list(tmp_path.glob(".get-example.*"))
    if kept:
        assert "It is unchanged." in result.stderr
        assert files_under(tmp_path / "sample") == {"deps/neat-apps.json", "old", "models/m"}
        assert stages == []
    else:
        assert result.stderr.endswith(f"Could not replace sample.\nThe old folder is kept in {stages[0]}/old.\n")
        assert files_under(stages[0] / "old") == {"deps/neat-apps.json", "old", "models/m"}


@pytest.mark.parametrize("signal_name", ["INT", "TERM", "HUP"])
def test_a_signal_during_the_swap_puts_the_old_folder_back(tmp_path: Path, signal_name: str) -> None:
    fetched_before(tmp_path / "sample", "models/m", "old")
    fake_bin = Path(path_without(tmp_path, "mv"))
    (fake_bin / "mv").write_text(
        f'#!/bin/sh\ncase "$2:$3" in */new:*/sample) kill -{signal_name} $PPID; exit 1 ;; esac\n'
        f'exec {shutil.which("mv")} "$@"\n'
    )
    (fake_bin / "mv").chmod(0o755)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--force", "sample", path=str(fake_bin))

    assert result.returncode != 0
    assert files_under(tmp_path / "sample") == {"deps/neat-apps.json", "old", "models/m"}
    assert not list(tmp_path.glob(".get-example.*"))


def test_keeps_an_existing_destination_without_force(tmp_path: Path) -> None:
    (tmp_path / "sample").mkdir()

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "sample")

    assert result.returncode == 1
    assert "Destination exists: sample" in result.stderr


def test_a_bundle_whose_checksum_differs_is_never_read(tmp_path: Path) -> None:
    # Only the first 8 characters agree, so a check of a prefix passes.
    index = make_index(tmp_path, checksum=lambda digest: digest[:8] + "0" * 56)
    fake_bin = Path(path_without(tmp_path, "tar"))
    (fake_bin / "tar").write_text(f'#!/bin/sh\ntouch "{tmp_path}/tar-ran"\nexec {shutil.which("tar")} "$@"\n')
    (fake_bin / "tar").chmod(0o755)

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample", path=str(fake_bin))

    assert result.returncode == 1
    assert "Checksum mismatch" in result.stderr
    assert not (tmp_path / "tar-ran").exists()
    assert not (tmp_path / "sample").exists()


@pytest.mark.parametrize(("hidden", "ok"), [(("sha256sum",), True), (("sha256sum", "shasum"), False)])
def test_checks_the_checksum_with_shasum_or_says_what_is_missing(
    tmp_path: Path, hidden: tuple[str, ...], ok: bool
) -> None:
    if not shutil.which("shasum"):
        pytest.skip("shasum is not installed")

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "sample", path=path_without(tmp_path, *hidden))

    assert (result.returncode == 0) is ok, result.stderr
    assert ok or "sha256sum or shasum is required." in result.stderr


def test_a_cut_off_download_runs_nothing(tmp_path: Path) -> None:
    (tmp_path / "sample").mkdir()
    (tmp_path / "sample" / "old").write_text("")
    text = SCRIPT.read_text()
    index = make_index(tmp_path)
    for end in [i + 1 for i, char in enumerate(text) if char == "\n"][:-1]:
        result = subprocess.run(
            ["bash", "-s", "--", "--ref", "team/demo", "--force", "sample"], input=text[:end], cwd=tmp_path,
            env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "NEAT_APPS_ARTIFACTS_URL": str(index)},
            capture_output=True, text=True,
        )
        assert result.stdout == "", text[:end].splitlines()[-1]
        assert files_under(tmp_path / "sample") == {"old"}


@pytest.mark.parametrize("answer", ["<html>Sign in</html>\n", "abc123\n"])
def test_a_page_instead_of_a_build_name_is_refused(tmp_path: Path, answer: str) -> None:
    index = make_index(tmp_path)
    (index / "apps/team%252Fdemo/latest.tag").write_text(answer)

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample")

    assert result.returncode == 1
    assert "Not a build name in " in result.stderr


@pytest.mark.parametrize("tool", list(TOOLS))
def test_fetches_over_http(tmp_path: Path, server: IndexServer, tool: str) -> None:
    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=tool_path(tmp_path, tool))

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "sample") == SAMPLE_FILES


@pytest.mark.parametrize("code", [403, 404])
@pytest.mark.parametrize("tool", list(TOOLS))
def test_a_missing_ref_over_http_is_not_published(tmp_path: Path, server: IndexServer, tool: str, code: int) -> None:
    server.fail("/apps/gone/latest.tag", code)

    result = get_example(tmp_path, server.url, "--ref", "gone", "sample", path=tool_path(tmp_path, tool))

    assert result.returncode == 1
    assert f"No published Apps bundle for gone ({server.url}/apps/gone)." in result.stderr


@pytest.mark.parametrize(("tool", "ok"), [("curl", True), ("wget", True), ("busybox-wget", False)])
def test_retries_a_server_error(tmp_path: Path, server: IndexServer, tool: str, ok: bool) -> None:
    server.fail("/apps/team%252Fdemo/latest.tag", 503, 0)

    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=tool_path(tmp_path, tool))

    assert (result.returncode == 0) is ok, result.stderr
    assert ok or "Cannot read the index" in result.stderr


@pytest.mark.parametrize("tool", list(TOOLS))
def test_a_server_error_is_not_a_missing_bundle(tmp_path: Path, server: IndexServer, tool: str) -> None:
    server.fail(f"/apps/team%252Fdemo/{BUILD}/metadata.json", 500)

    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=tool_path(tmp_path, tool))

    assert result.returncode == 1
    assert f"Cannot read the index: {server.url}/apps/team%252Fdemo/{BUILD}/metadata.json" in result.stderr
    assert "No build" not in result.stderr


def test_reads_the_published_index_by_default(tmp_path: Path) -> None:
    curl, log = logging_tool(tmp_path, "curl")
    fake_bin = tmp_path / "fake"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text(f'#!/bin/sh\necho "$*" >> "{log}"\nprintf 404\n')
    (fake_bin / "curl").chmod(0o755)
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin", "HOME": str(tmp_path)}

    result = subprocess.run(["bash", str(SCRIPT), "sample"], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 1
    assert log.read_text().split()[-1] == "https://artifacts.neat.sima.ai/apps/main/latest.tag"


@pytest.mark.parametrize("through", ["cwd", "home"])
def test_force_guard_resolves_symlinks(tmp_path: Path, through: str) -> None:
    real = tmp_path / "real"
    (real / "home" / "keep").parent.mkdir(parents=True)
    (real / "home" / "keep").write_text("")
    (real / "work").mkdir()
    (tmp_path / "home-link").symlink_to(real / "home")
    (tmp_path / "work-link").symlink_to(real / "work")
    # Only the guard under test protects dest: $HOME sits outside it in the cwd case.
    (tmp_path / "elsewhere").mkdir()
    home, cwd, dest = (
        (tmp_path / "elsewhere", tmp_path / "work-link", real)
        if through == "cwd" else (tmp_path / "home-link", real / "work", real / "home")
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--ref", "team/demo", "--force", "--dest", str(dest), "sample"], cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "HOME": str(home), "PWD": str(cwd),
             "NEAT_APPS_ARTIFACTS_URL": str(make_index(tmp_path))},
        capture_output=True, text=True,
    )

    assert result.returncode == 1
    assert f"Refusing destination: {dest}" in result.stderr
    assert (real / "home" / "keep").exists()


def test_lists_the_examples_sorted(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path, reverse=True), "--ref", "team/demo", "missing")

    listed = result.stderr.splitlines()[1:]
    assert listed[:3] == ["  demo/big", "  demo/combo", "  demo/dashname"]
    assert listed == sorted(listed)


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_lists_every_option(tmp_path: Path, flag: str) -> None:
    result = get_example(tmp_path, tmp_path, flag)

    assert result.returncode == 0
    for line in (
        "  <example>  name (single-stream-object-detector) or path (object-detection/<name>)",
        "  --ref      branch or tag the bundle was published from (default: main, latest build);",
        "             --branch is the same option",
        "  --dest     destination folder (default: ./<example name>)",
        "  --force    replace the destination folder, keeping only its models/",
        "Environment: NEAT_APPS_BRANCH (--ref), NEAT_APPS_EXAMPLE_DEST_DIR (--dest),",
    ):
        assert line in result.stdout.splitlines()


def test_passes_the_retry_and_timeout_options(tmp_path: Path, server: IndexServer) -> None:
    curl, curl_log = logging_tool(tmp_path, "curl")
    wget, wget_log = logging_tool(tmp_path, "wget")

    for path in (path_without(tmp_path, extra={"curl": str(curl)}), path_without(tmp_path, "curl", extra={"wget": str(wget)})):
        (tmp_path / "sample").exists() and shutil.rmtree(tmp_path / "sample")
        result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=path)
        assert result.returncode == 0, result.stderr

    assert "-sSL --retry 3 --connect-timeout 20 --speed-time 60 --speed-limit 1000 -w %{http_code}" in curl_log.read_text()
    calls = [line for line in wget_log.read_text().splitlines() if line != "--help"]
    assert all("-S -q --tries=3 --read-timeout=60 --retry-on-http-error=500,502,503,504 -O" in line for line in calls)


def test_says_which_download_tool_is_missing(tmp_path: Path, server: IndexServer) -> None:
    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=path_without(tmp_path, "curl", "wget"))

    assert result.returncode == 1
    assert result.stderr.startswith("curl or wget is required.\n")


@pytest.mark.parametrize("tool", list(TOOLS))
def test_a_failed_bundle_download_says_so(tmp_path: Path, server: IndexServer, tool: str) -> None:
    server.fail(f"/apps/team%252Fdemo/{BUILD}/{BUNDLE}", 500)

    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=tool_path(tmp_path, tool))

    assert result.returncode == 1
    assert f"Download failed: {server.url}/apps/team%252Fdemo/{BUILD}/{BUNDLE}" in result.stderr


@pytest.mark.parametrize("tool", ["curl", "wget"])
def test_a_body_cut_off_is_a_failed_download(tmp_path: Path, server: IndexServer, tool: str) -> None:
    server.truncated.add(f"/apps/team%252Fdemo/{BUILD}/{BUNDLE}")

    result = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=tool_path(tmp_path, tool))

    assert result.returncode == 1
    assert "Download failed:" in result.stderr
    assert not (tmp_path / "sample").exists()


def test_reports_the_http_status_and_the_wget_log(tmp_path: Path, server: IndexServer) -> None:
    server.fail(f"/apps/team%252Fdemo/{BUILD}/metadata.json", 500)
    url = f"{server.url}/apps/team%252Fdemo/{BUILD}/metadata.json"

    curl = get_example(tmp_path, server.url, "--ref", "team/demo", "sample")
    wget = get_example(tmp_path, server.url, "--ref", "team/demo", "sample", path=path_without(tmp_path, "curl"))

    assert f"HTTP 500: {url}" in curl.stderr
    assert f"wget failed with HTTP 500: {url}" in wget.stderr
    # The three lines after it are the last response headers in wget's own log.
    tail = wget.stderr.split(f"wget failed with HTTP 500: {url}\n")[1].splitlines()[:3]
    assert "  Connection: close" in tail


@pytest.mark.parametrize(
    ("args", "folder", "files"),
    [
        (("--dest", "a/b/c", "sample"), "a/b/c", SAMPLE_FILES),
        (("twin",), None, None),
        (("dotname",), "dotname", {"examples/demo/dotname/README.md", "assets/datasets/coco.v2/d.jpg"}),
        (("dashname",), "dashname", {"examples/demo/dashname/README.md", "assets/datasets/coco-v2/e.jpg"}),
        (("xnear",), "xnear", {"examples/demo/xnear/README.md", "examples/demo/xnear/src/common/config.yaml"}),
    ],
)
def test_fetches_into_the_folder_with_exactly_its_files(
    tmp_path: Path, args: tuple[str, ...], folder: str | None, files: set[str] | None
) -> None:
    older = folder != "a/b/c"
    index = make_index(tmp_path, skip=OLDER_BUNDLE if older else ())

    result = get_example(tmp_path, index, "--ref", "team/demo", *args)

    assert result.returncode == 1 if folder is None else result.returncode == 0, result.stderr
    if folder is None:
        assert "More than one example is named twin." in result.stderr
    else:
        assert files_under(tmp_path / folder) == files


def test_an_older_bundle_without_named_datasets_extracts_only_the_example(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path, skip=OLDER_BUNDLE), "--ref", "team/demo", "demo/twin")

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "twin") == {"examples/demo/twin/README.md"}


def test_quotes_the_next_step(tmp_path: Path) -> None:
    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--dest", "my dest", "sample")

    assert result.returncode == 0, result.stderr
    assert "Next: cd my\\ dest; APP_DIR=examples/demo/sample\n" in result.stdout


def test_the_whole_bundle_check_works_with_busybox_grep(tmp_path: Path) -> None:
    if not shutil.which("busybox"):
        pytest.skip("busybox is not installed")
    path = path_without(tmp_path, extra={"grep": shutil.which("busybox")})
    index = make_index(tmp_path)

    combo = get_example(tmp_path, index, "--ref", "team/demo", "combo", path=path)
    sample = get_example(tmp_path, index, "--ref", "team/demo", "sample", path=path)

    assert combo.stderr == "combo needs the whole bundle: sima-cli neat install apps\n"
    assert sample.returncode == 0, sample.stderr
    assert "Usage" not in sample.stderr


def test_refuses_a_bundle_with_a_symlink(tmp_path: Path) -> None:
    index = make_index(tmp_path, links={"examples/demo/sample/src/common/etc": "/etc"})

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample")

    assert result.returncode == 1
    assert "The bundle holds a symlink: prebuilt-apps/examples/demo/sample/src/common/etc" in result.stderr
    assert not (tmp_path / "sample").exists()


def test_an_archive_url_may_start_with_file(tmp_path: Path) -> None:
    make_index(tmp_path)
    bundle = tmp_path / "index/apps/team%252Fdemo" / BUILD / BUNDLE

    result = get_example(tmp_path, tmp_path / "no-index", "sample", NEAT_APPS_ARCHIVE_URL=f"file://{bundle}")

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "sample") == SAMPLE_FILES


def test_a_grep_error_stops_the_whole_bundle_check(tmp_path: Path) -> None:
    fake_bin = Path(path_without(tmp_path, "grep"))
    (fake_bin / "grep").write_text(f'#!/bin/sh\ncase "$1" in -rlE) exit 2 ;; esac\nexec {shutil.which("grep")} "$@"\n')
    (fake_bin / "grep").chmod(0o755)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "sample", path=str(fake_bin))

    assert result.returncode == 1
    assert "grep failed on prebuilt-apps/examples/demo/sample." in result.stderr


@pytest.mark.parametrize("where", ["support/python/sima_examples/link", "assets/datasets/coco/link"])
def test_refuses_a_symlink_anywhere_in_the_fetched_folder(tmp_path: Path, where: str) -> None:
    index = make_index(tmp_path, links={where: "/etc"})

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample")

    assert result.returncode == 1
    assert f"The bundle holds a symlink: prebuilt-apps/{where}" in result.stderr


def test_a_sibling_whose_name_starts_like_the_cwd_is_accepted(tmp_path: Path) -> None:
    cwd = tmp_path / "work" / "sub-x"
    cwd.mkdir(parents=True)

    result = get_example(tmp_path, make_index(tmp_path), "--ref", "team/demo", "--dest", "../sub", "sample", cwd=cwd)

    assert result.returncode == 0, result.stderr
    assert files_under(tmp_path / "work" / "sub") == SAMPLE_FILES


@pytest.mark.parametrize(("answer", "ok"), [(f"{BUILD}\r\n", True), (f"  {BUILD} \n", True), ("a" * 40 + "\n", False)])
def test_reads_a_padded_build_name_and_refuses_a_full_commit(tmp_path: Path, answer: str, ok: bool) -> None:
    index = make_index(tmp_path)
    (index / "apps/team%252Fdemo/latest.tag").write_text(answer)

    result = get_example(tmp_path, index, "--ref", "team/demo", "sample")

    assert (result.returncode == 0) is ok, result.stderr
    assert ok or "Not a build name in " in result.stderr


@pytest.mark.parametrize(
    ("files", "refused"),
    [
        # demo/samples is its own example, so naming it is not naming oneself.
        ({"examples/demo/samples/README.md": "", "examples/demo/sample/src/common/run.sh": "cd demo/samples\n"}, True),
        # Only README.md is a mention. Any other file, Markdown too, runs the other example.
        ({"examples/demo/sample/docs/usage.md": "Start demo/other first.\n"}, True),
        ({"examples/demo/sample/src/README.md": "Start demo/other first.\n"}, False),
    ],
)
def test_the_whole_bundle_check_drops_only_itself_and_readmes(
    tmp_path: Path, files: dict[str, str], refused: bool
) -> None:
    result = get_example(tmp_path, make_index(tmp_path, extra=files), "--ref", "team/demo", "demo/sample")

    assert (result.returncode == 1) is refused, result.stderr
    assert not refused or "needs the whole bundle" in result.stderr
