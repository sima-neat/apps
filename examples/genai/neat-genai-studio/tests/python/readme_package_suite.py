"""README and launcher <-> packaged-file consistency for neat-genai-studio.

The customer README, setup.sh and run.sh must only point at files that ship
in the installed ``prebuilt-apps`` bundle. Run from a source checkout this checks the tracked
tree; run from the installed package (``./tests/test.sh --all`` overlays the
tests onto the runtime dir on the target) it checks the real install, so it
fails if ``sima-cli neat install apps`` shipped a package that is missing a file
the README tells the customer to use.

Named ``*_suite.py`` and collected through ``test_unit.py`` like the other
suites; no hardware, no models, no live services. Run in place with
``python -m unittest readme_package_suite`` from this directory.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
README = EXAMPLE_DIR / "README.md"

# The first path segment of anything build.sh ships in the customer bundle. A
# referenced token whose first segment is not one of these is a model repo,
# config key, URL, runtime directory, or a repository-only path such as
# ``tests/`` (named in the README's development section, and overlaid onto the
# install only by CI), so it is not checked for existence here.
PACKAGED_TOP = {
    "run.sh",
    "setup.sh",
    "src",
    "README.md",
}

# Paths the README legitimately names that setup.sh / run.sh create at install
# or runtime, so they are absent from the packaged tree by design. Matched
# against every path segment, so ``src/python/ui/milvus.db`` is covered too.
GENERATED_TOP = {
    ".venv",
    ".venv-pipertts",
    "config.local.yaml",
    "milvus.db",
    "milvus.meta.json",
    "models",
}

_APP_DIR_PREFIXES = ("${APP_DIR}/", "$APP_DIR/")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_FENCE = re.compile(r"```.*?\n(.*?)```", re.DOTALL)


def _strip_app_dir(token: str) -> str:
    for prefix in _APP_DIR_PREFIXES:
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def _looks_like_repo_path(token: str) -> bool:
    """A relative, packaged-looking path — not a URL, shell expansion,
    assignment, flag, absolute path, or config key."""
    if not token or token.startswith(("/", "-", "#", "~")):
        return False
    if any(ch in token for ch in (" ", "$", "=", ":", "*", "<", ">")):
        return False
    if "://" in token:
        return False
    first = token.split("/", 1)[0]
    return first in PACKAGED_TOP


def _referenced_paths(text: str) -> set[str]:
    tokens: set[str] = set()
    # Pull fenced blocks out first: their triple backticks otherwise confuse the
    # single-backtick inline regex and swallow the inline path references.
    fenced = _FENCE.findall(text)
    prose = _FENCE.sub("\n", text)
    fragments = list(_INLINE_CODE.findall(prose))
    for block in fenced:
        fragments.extend(block.split())
    for raw in fragments:
        token = _strip_app_dir(raw.strip().strip("`").rstrip("\\"))
        if _looks_like_repo_path(token):
            tokens.add(token)
    return tokens


# What build.sh packages for an example (its find(1) filter): everything under
# src/python/ and src/common/, plus README.md, run.sh and setup.sh.
_BUILD_PACKAGED = ("src/python/", "src/common/")
_BUILD_PACKAGED_FILES = {"README.md", "run.sh", "setup.sh"}
_EXAMPLE_DIR_REF = re.compile(r"\$\{?EXAMPLE_DIR\}?/([A-Za-z0-9_./-]+)")


def _launcher_paths(text: str) -> set[str]:
    """``${EXAMPLE_DIR}/...`` paths a launcher script uses, minus the venvs,
    config, pid/log/token files it creates itself (dot-files and
    config.local.yaml)."""
    paths = set()
    for path in _EXAMPLE_DIR_REF.findall(text):
        path = path.rstrip("/.")
        if not path or path.startswith(".") or path == "config.local.yaml":
            continue
        paths.add(path)
    return paths


class LauncherPackagePathsTests(unittest.TestCase):
    def test_launcher_paths_are_packaged(self) -> None:
        for script in ("setup.sh", "run.sh"):
            text = (EXAMPLE_DIR / script).read_text(encoding="utf-8")
            paths = _launcher_paths(text)
            self.assertTrue(paths, f"No ${{EXAMPLE_DIR}} paths found in {script}")
            unpackaged = sorted(
                p for p in paths
                if p not in _BUILD_PACKAGED_FILES
                and not (p + "/").startswith(_BUILD_PACKAGED)
            )
            self.assertEqual(
                unpackaged, [],
                f"{script} uses files build.sh does not package: {unpackaged}",
            )
            missing = sorted(p for p in paths if not (EXAMPLE_DIR / p).exists())
            self.assertEqual(missing, [], f"{script} uses missing files: {missing}")


class ReadmePackagePathsTests(unittest.TestCase):
    def test_readme_present(self) -> None:
        self.assertTrue(README.is_file(), f"README.md missing at {README}")

    def test_referenced_packaged_paths_exist(self) -> None:
        text = README.read_text(encoding="utf-8")
        referenced = _referenced_paths(text)
        # A packaged README always cites at least its own source layout; an empty
        # set means the parser or the README changed shape and the check went
        # blind, so treat that as a failure rather than a silent pass.
        self.assertTrue(
            referenced,
            "No packaged paths were found in README.md; the check would not "
            "verify anything.",
        )
        missing = sorted(
            token
            for token in referenced
            if GENERATED_TOP.isdisjoint(token.rstrip("/").split("/"))
            and not (EXAMPLE_DIR / token).exists()
        )
        self.assertEqual(
            missing,
            [],
            "README.md references packaged paths that do not exist in the "
            f"bundle: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
