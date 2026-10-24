"""Unit tests for the Docker parallel-apps example.

parallel-apps.sh names other examples in its app table. These tests check that
every row resolves to an existing C++ example and its config, so a renamed or
removed example fails here instead of during an image build in the SDK.
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


def test_app_table_lists_four_distinct_examples() -> None:
    assert len(APPS) == 4
    assert len(set(APPS.values())) == 4


@pytest.mark.parametrize("name", sorted(APPS))
def test_app_is_an_existing_cpp_example(name: str) -> None:
    example = APPS_ROOT / "examples" / APPS[name]
    cmake = (example / "src" / "cpp" / "CMakeLists.txt").read_text(encoding="utf-8")

    # parallel-apps.sh copies build/examples/<example>/<directory name>, so the
    # example's executable must carry its directory name.
    module = re.search(r'sima_neat_apps_module\(\s*"([^"]+)"', cmake)
    assert module and module.group(1) == example.name
    assert (example / "src" / "common" / "config.yaml").is_file()


def test_dockerfile_exists() -> None:
    assert (COMMON_DIR / "Dockerfile").is_file()
