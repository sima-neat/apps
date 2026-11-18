"""The examples' vendored helper files match their templates under support/templates/.

An example holds copies of the common helpers it uses in its own src/<lang>/utils/ folder, so
it works without the rest of the repository. scripts/sync_utils.py finds each copy by its
"Copied from support/templates/..." header and rewrites it from the template. The first test
runs its check on the repository; the others pin down what the script does on a small tree.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = APPS_ROOT / "scripts" / "sync_utils.py"


def _load_sync_utils():
    spec = importlib.util.spec_from_file_location("sync_utils", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before it runs: its dataclass resolves its module through sys.modules.
    sys.modules["sync_utils"] = module
    spec.loader.exec_module(module)
    return module


sync_utils = _load_sync_utils()


def test_every_vendored_copy_matches_its_template():
    assert sync_utils.find_copies(APPS_ROOT), (
        "no vendored copies found under examples/, so this check would pass without checking"
    )

    problems = sync_utils.check(APPS_ROOT)

    assert not problems, (
        "Vendored helpers differ from their templates:\n"
        + "\n".join(f"  {problem}" for problem in problems)
        + f"\nEdit the template, not the copy, then run: {sync_utils.FIX_COMMAND}"
    )


PY_TEMPLATE = "support/templates/python/helper.py"
PY_COPY = "examples/demo/app/src/python/utils/helper.py"
CPP_TEMPLATE = "support/templates/cpp/helper.h"
CPP_COPY = "examples/demo/app/src/cpp/utils/helper.h"
PY_HEADER = (
    f"# Copied from {PY_TEMPLATE};\n# edit the template and run scripts/sync_utils.py.\n"
)
CPP_HEADER = (
    f"// Copied from {CPP_TEMPLATE};\n// edit the template and run scripts/sync_utils.py.\n"
)


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _tree(root: Path) -> None:
    """Two templates; the Python copy has drifted and the C++ copy holds only its header."""
    _write(root, PY_TEMPLATE, '"""Helper."""\n\nVALUE = 1\n')
    _write(root, PY_COPY, PY_HEADER + '"""Helper."""\n\nVALUE = 2\n')
    _write(root, CPP_TEMPLATE, "#pragma once\n\ninline int value() { return 1; }\n")
    _write(root, CPP_COPY, f"// Copied from {CPP_TEMPLATE};\n")
    _write(root, "examples/demo/app/src/python/main.py", "print('not a copy')\n")


def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_check_names_each_drifted_copy_and_the_fix_without_writing(tmp_path):
    _tree(tmp_path)
    before = (tmp_path / PY_COPY).read_text(encoding="utf-8")

    result = _run(tmp_path, "--check")

    assert result.returncode == 1
    assert f"{PY_COPY}: differs from {PY_TEMPLATE}" in result.stderr
    assert f"{CPP_COPY}: differs from {CPP_TEMPLATE}" in result.stderr
    assert "python3 scripts/sync_utils.py" in result.stderr
    assert "main.py" not in result.stderr
    assert (tmp_path / PY_COPY).read_text(encoding="utf-8") == before


def test_sync_rewrites_each_copy_as_its_header_and_template(tmp_path):
    _tree(tmp_path)

    result = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / PY_COPY).read_text(encoding="utf-8") == (
        PY_HEADER + (tmp_path / PY_TEMPLATE).read_text(encoding="utf-8")
    )
    assert (tmp_path / CPP_COPY).read_text(encoding="utf-8") == (
        CPP_HEADER + (tmp_path / CPP_TEMPLATE).read_text(encoding="utf-8")
    )
    assert (tmp_path / "examples/demo/app/src/python/main.py").read_text() == (
        "print('not a copy')\n"
    )
    assert _run(tmp_path, "--check").returncode == 0


def test_a_copy_that_names_a_missing_template_fails_both_modes(tmp_path):
    _write(
        tmp_path,
        PY_COPY,
        "# Copied from support/templates/python/gone.py;\n"
        "# edit the template and run scripts/sync_utils.py.\n",
    )

    for args in (("--check",), ()):
        result = _run(tmp_path, *args)
        assert result.returncode == 1
        assert f"{PY_COPY}: names support/templates/python/gone.py, which does not exist" in (
            result.stderr
        )


def test_a_copy_cannot_name_a_file_outside_the_templates(tmp_path):
    _write(tmp_path, "scripts/other.py", "VALUE = 1\n")
    _write(tmp_path, PY_COPY, "# Copied from scripts/other.py;\n")

    result = _run(tmp_path)

    assert result.returncode == 1
    assert "which is not under support/templates/" in result.stderr
    assert (tmp_path / PY_COPY).read_text(encoding="utf-8") == "# Copied from scripts/other.py;\n"


def test_a_utils_file_named_like_a_template_without_the_header_is_reported(tmp_path):
    """A copy is found by its first line; one that lost it must not drop out of the check."""
    _write(tmp_path, PY_TEMPLATE, "VALUE = 1\n")
    _write(tmp_path, PY_COPY, "# Licensed under ...\n" + PY_HEADER + "VALUE = 2\n")

    for args in (("--check",), ()):
        result = _run(tmp_path, *args)
        assert result.returncode == 1
        assert f"{PY_COPY}: has the name of {PY_TEMPLATE} but does not start with" in (
            result.stderr
        )


def test_a_copy_with_bytes_that_are_not_utf8_is_reported_not_a_crash(tmp_path):
    _write(tmp_path, PY_TEMPLATE, "VALUE = 1\n")
    (tmp_path / PY_COPY).parent.mkdir(parents=True)
    (tmp_path / PY_COPY).write_bytes(PY_HEADER.encode() + b"NAME = '\xe9'\n")

    result = _run(tmp_path, "--check")

    assert result.returncode == 1
    assert f"{PY_COPY}: differs from {PY_TEMPLATE}" in result.stderr
    assert "Traceback" not in result.stderr
