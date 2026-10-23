"""Unit tests for the Docker parallel-apps example.

compose.yaml names other examples by path. These tests check that every path
resolves to the example, binary, config, and models it is meant to, both in the
repository and in the installed prebuilt-apps/ bundle.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parents[2]
COMPOSE_DIR = EXAMPLE_DIR / "src" / "common"
APPS_ROOT = EXAMPLE_DIR.parents[2]

COMPOSE = yaml.safe_load((COMPOSE_DIR / "compose.yaml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]

pytestmark = pytest.mark.unit


def compose_path(path: str) -> Path:
    """Resolve a path the way Compose does: from the compose file's directory."""
    return (COMPOSE_DIR / path).resolve()


def example_dir(service: dict) -> Path:
    return APPS_ROOT / "examples" / service["build"]["args"]["EXAMPLE"]


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_builds_from_the_apps_root(name: str) -> None:
    build = SERVICES[name]["build"]

    assert compose_path(build["context"]) == APPS_ROOT
    assert (APPS_ROOT / build["dockerfile"]).is_file()


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_runs_an_existing_example_binary(name: str) -> None:
    service = SERVICES[name]

    # The bundle installs a C++ example's binary as src/cpp/pre-built/<example>.
    assert (example_dir(service) / "src" / "cpp").is_dir()
    assert service["build"]["args"]["BINARY"] == example_dir(service).name


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_reads_its_example_config(name: str) -> None:
    service = SERVICES[name]
    (mount,) = service["configs"]
    config = compose_path(COMPOSE["configs"][mount["source"]]["file"])

    assert config == example_dir(service) / "src" / "common" / "config.yaml"
    assert config.is_file()
    assert mount["target"] == "/apps/config.yaml"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_mounts_the_bundle_models(name: str) -> None:
    host_paths = {
        container: host
        for host, container, *_ in (v.split(":") for v in SERVICES[name]["volumes"])
    }

    assert compose_path(host_paths["/apps/models"]) == APPS_ROOT / "models"
