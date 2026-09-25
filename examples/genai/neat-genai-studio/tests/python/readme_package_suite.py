"""README <-> packaged-file consistency for neat-genai-studio.

The customer README must only point at files that ship in the installed
``prebuilt-apps`` bundle. Run from a source checkout this checks the tracked
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

# The first path segment of anything the packaged bundle actually ships. A
# referenced token whose first segment is not one of these is a model repo,
# config key, URL, or runtime directory — not a packaged file — so it is not
# checked for existence here.
PACKAGED_TOP = {
    "run.sh",
    "setup.sh",
    "src",
    "tests",
    "README.md",
    "THIRD_PARTY_TTS_MODELS.md",
}

# Paths the README legitimately names that setup.sh / run.sh create at install
# or runtime, so they are absent from the packaged tree by design. Matched
# against the first path segment after any ``${APP_DIR}/`` prefix is stripped.
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
            if token.split("/", 1)[0] not in GENERATED_TOP
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
