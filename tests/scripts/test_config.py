"""Contract tests for support/python/sima_examples/config.py, the examples' config reader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.config import Config


def config_file(tmp_path: Path, text: str) -> Config:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Config.load(path)


def test_reads_dotted_keys_by_type(tmp_path):
    config = config_file(
        tmp_path,
        "input:\n  fps: 30\n  rtsp:\n    tcp: true\n  source: [a, b]\n"
        "inference:\n  min_score: 0.25\nmodel:\n  path: m.tar.gz\n",
    )
    assert config.integer("input.fps") == 30
    assert config.boolean("input.rtsp.tcp") is True
    assert config.string_list("input.source") == ["a", "b"]
    assert config.number("inference.min_score") == 0.25
    assert config.string("model.path") == "m.tar.gz"


def test_one_string_reads_as_a_list_of_one(tmp_path):
    config = config_file(tmp_path, "input:\n  source: rtsp://one\n")
    assert config.string_list("input.source") == ["rtsp://one"]


def test_missing_key_names_the_key(tmp_path):
    config = config_file(tmp_path, "input:\n  fps: 30\n")
    with pytest.raises(ValueError, match="is missing input.width"):
        config.integer("input.width")


def test_wrong_type_names_the_key(tmp_path):
    config = config_file(tmp_path, "input:\n  fps: thirty\n")
    with pytest.raises(ValueError, match="input.fps must be an integer"):
        config.integer("input.fps")


def test_bool_is_not_an_integer(tmp_path):
    config = config_file(tmp_path, "input:\n  fps: true\n")
    with pytest.raises(ValueError, match="input.fps must be an integer"):
        config.integer("input.fps")
