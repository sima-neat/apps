"""Check the configure contract of cmake/YamlCpp.cmake without a network or the SDK."""

import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


APPS_ROOT = Path(__file__).resolve().parents[2]
YAML_CPP_CMAKE = APPS_ROOT / "cmake" / "YamlCpp.cmake"
OFFLINE_VARIABLES = ("FETCHCONTENT_SOURCE_DIR_YAML_CPP", "YAML_CPP_SOURCE_URL")

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="cmake is not on PATH"
)


@pytest.fixture
def project(tmp_path):
    """A project that includes YamlCpp.cmake, a fake libyaml-cpp.so.0.8, and a fake source."""
    source = tmp_path / "project"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\n"
        "project(yaml_cpp_contract NONE)\n"
        f'include("{YAML_CPP_CMAKE.as_posix()}")\n'
        "sima_neat_apps_ensure_yaml_cpp()\n"
        "get_target_property(include_dir SimaNeatApps::yaml_cpp INTERFACE_INCLUDE_DIRECTORIES)\n"
        'message(STATUS "yaml-cpp include: ${include_dir}")\n',
        encoding="utf-8",
    )
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libyaml-cpp.so.0.8").write_bytes(b"")
    return {
        "source": source,
        "build": tmp_path / "build",
        "library_dir": library_dir,
        "tmp": tmp_path,
    }


def _yaml_cpp_tree(root: Path) -> Path:
    header = root / "include" / "yaml-cpp" / "yaml.h"
    header.parent.mkdir(parents=True)
    header.write_text("#pragma once\n", encoding="utf-8")
    return root


def _configure(project, *definitions, env=None):
    environment = {
        key: value for key, value in os.environ.items() if key not in OFFLINE_VARIABLES
    }
    environment.update(env or {})
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(project["source"]),
            "-B",
            str(project["build"]),
            f"-DCMAKE_LIBRARY_PATH={project['library_dir']}",
            *definitions,
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _output(result) -> str:
    """The configure output with CMake's line wrapping undone."""
    return " ".join((result.stdout + result.stderr).split())


def _include_dir(result) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("-- yaml-cpp include: "):
            return line.removeprefix("-- yaml-cpp include: ")
    raise AssertionError(f"no include line in:\n{result.stdout}")


def test_absolute_source_dir_sets_the_include_directory(project):
    tree = _yaml_cpp_tree(project["tmp"] / "yaml-cpp-0.8.0")

    result = _configure(project, f"-DFETCHCONTENT_SOURCE_DIR_YAML_CPP={tree}")

    assert _include_dir(result) == f"{tree}/include"


def test_relative_source_dir_fails(project):
    result = _configure(project, env={"FETCHCONTENT_SOURCE_DIR_YAML_CPP": "yaml-cpp-0.8.0"})

    assert result.returncode != 0
    assert "FETCHCONTENT_SOURCE_DIR_YAML_CPP must be an absolute path" in _output(result)


def test_source_dir_without_yaml_h_fails(project):
    empty = project["tmp"] / "empty"
    empty.mkdir()

    result = _configure(project, f"-DFETCHCONTENT_SOURCE_DIR_YAML_CPP={empty}")

    assert result.returncode != 0
    assert f"{empty} has no include/yaml-cpp/yaml.h" in _output(result)


def test_relative_source_url_fails(project):
    result = _configure(project, env={"YAML_CPP_SOURCE_URL": "0.8.0.tar.gz"})

    assert result.returncode != 0
    assert "YAML_CPP_SOURCE_URL must be a URL or an absolute path" in _output(result)


def test_local_tarball_with_another_hash_fails(project):
    tree = _yaml_cpp_tree(project["tmp"] / "yaml-cpp-0.8.0")
    tarball = project["tmp"] / "0.8.0.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(tree, arcname="yaml-cpp-0.8.0")
    assert hashlib.sha256(tarball.read_bytes()).hexdigest() not in YAML_CPP_CMAKE.read_text()

    result = _configure(project, env={"YAML_CPP_SOURCE_URL": str(tarball)})

    assert result.returncode != 0
    assert "SHA256 hash" in _output(result)
    assert "does not match expected value" in _output(result)


def test_environment_replaces_and_build_directory_keeps_the_source_dir(project):
    first = _yaml_cpp_tree(project["tmp"] / "first")
    second = _yaml_cpp_tree(project["tmp"] / "second")
    assert _include_dir(
        _configure(project, f"-DFETCHCONTENT_SOURCE_DIR_YAML_CPP={first}")
    ) == f"{first}/include"

    replaced = _configure(project, env={"FETCHCONTENT_SOURCE_DIR_YAML_CPP": str(second)})
    kept = _configure(project)

    assert _include_dir(replaced) == f"{second}/include"
    assert _include_dir(kept) == f"{second}/include"
