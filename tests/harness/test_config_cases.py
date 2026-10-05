"""Harness self-tests for the shared configuration-test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.utils.config_cases import config_writer, load_example_main, write_config

pytestmark = pytest.mark.unit


def _write_example(root: Path, example: str, tracker_name: str) -> Path:
    example_dir = root / "examples" / "tracking" / example
    python_dir = example_dir / "src" / "python"
    (python_dir / "utils").mkdir(parents=True)
    (python_dir / "utils" / "__init__.py").write_text("", encoding="utf-8")
    (python_dir / "utils" / "tracker.py").write_text(
        f"class {tracker_name}:\n    pass\n", encoding="utf-8"
    )
    (python_dir / "main.py").write_text(
        f"from utils.tracker import {tracker_name}\n\nTRACKER = {tracker_name}\n",
        encoding="utf-8",
    )
    return example_dir


def test_each_example_imports_its_own_packages(tmp_path):
    # Two examples ship a package of the same name. Loading the second must not hand it
    # the first one's cached copy, and loading is idempotent.
    first_dir = _write_example(tmp_path, "first-tracker", "FirstTracker")
    second_dir = _write_example(tmp_path, "second-tracker", "SecondTracker")

    first = load_example_main(first_dir, "harness_first_tracker_main")
    second = load_example_main(second_dir, "harness_second_tracker_main")

    assert first.TRACKER.__name__ == "FirstTracker"
    assert second.TRACKER.__name__ == "SecondTracker"
    assert load_example_main(first_dir, "harness_first_tracker_main") is first
    assert first.TRACKER.__name__ == "FirstTracker"


def test_write_config_applies_overrides_to_a_copy(tmp_path):
    base = {"model": {"path": "model.tar.gz"}, "source": {"type": "rtsp"}}

    path = write_config(tmp_path, base, [(("source", "type"), "http"), (("model", "path"), "")])

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "model": {"path": ""},
        "source": {"type": "http"},
    }
    assert base == {"model": {"path": "model.tar.gz"}, "source": {"type": "rtsp"}}


def test_write_config_root_replaces_the_baseline(tmp_path):
    write = config_writer({"model": {"path": "model.tar.gz"}})

    assert yaml.safe_load(write(tmp_path, root=["not", "a", "mapping"]).read_text()) == [
        "not",
        "a",
        "mapping",
    ]
    assert yaml.safe_load(write(tmp_path).read_text()) == {"model": {"path": "model.tar.gz"}}
