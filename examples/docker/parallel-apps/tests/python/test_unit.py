"""Unit tests for the Docker parallel-apps example.

parallel-apps.sh names other examples in its app table. These tests check that
every row resolves to an existing C++ example and its config, in a checkout and
in the installed bundle, so a renamed or removed example fails here instead of
during an image build in the SDK.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMMON_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]
SCRIPT = COMMON_DIR / "parallel-apps.sh"

pytestmark = pytest.mark.unit


def read_app_table() -> dict[str, str]:
    """Return {container name: example path} from the script's APPS heredoc."""
    table = re.search(r"<<'APPS'\n(.*?)\nAPPS\n", SCRIPT.read_text(encoding="utf-8"), re.S)
    assert table, "parallel-apps.sh has no APPS table"
    return dict(line.split() for line in table.group(1).splitlines())


APPS = read_app_table()


def test_app_table_lists_six_distinct_examples() -> None:
    assert len(APPS) == 6
    assert len(set(APPS.values())) == 6


@pytest.mark.parametrize("name", sorted(APPS))
def test_app_is_an_existing_cpp_example(name: str) -> None:
    example = APPS_ROOT / "examples" / APPS[name]
    cpp = example / "src" / "cpp"
    assert cpp.is_dir(), f"{APPS[name]} has no C++ implementation"
    assert (example / "src" / "common" / "config.yaml").is_file()

    # parallel-apps.sh looks for an executable named after the example's
    # directory. A checkout declares it in CMake; the installed bundle drops the
    # CMake file and ships the executable under src/cpp/pre-built.
    cmake = cpp / "CMakeLists.txt"
    if cmake.is_file():
        module = re.search(r'sima_neat_apps_module\(\s*"([^"]+)"', cmake.read_text(encoding="utf-8"))
        assert module and module.group(1) == example.name
    else:
        assert (cpp / "pre-built" / example.name).is_file(), f"{APPS[name]} has no pre-built binary"


def test_dockerfile_exists() -> None:
    assert (COMMON_DIR / "Dockerfile").is_file()
