"""Contract tests for support/python/sima_examples/config.py, the examples' config reader."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APPS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APPS_ROOT / "support" / "python"))

from sima_examples.config import Config, lookup


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
    config = config_file(tmp_path, "input:\n  fps: thirty\n  scale: half\n")
    with pytest.raises(ValueError, match="^input.fps must be an integer, got 'thirty'$"):
        config.integer("input.fps")
    with pytest.raises(ValueError, match="^input.scale must be a number, got 'half'$"):
        config.number("input.scale")


def test_wrong_type_quotes_a_number_as_cpp_does(tmp_path):
    config = config_file(tmp_path, "input:\n  fps: 1.5\n")
    with pytest.raises(ValueError, match="^input.fps must be an integer, got '1.5'$"):
        config.integer("input.fps")


def test_bool_is_not_an_integer(tmp_path):
    config = config_file(tmp_path, "input:\n  fps: true\n")
    with pytest.raises(ValueError, match="input.fps must be an integer"):
        config.integer("input.fps")


def test_choice_returns_the_entry_the_value_names(tmp_path):
    config = config_file(tmp_path, "input:\n  codec: h265\n")
    assert config.choice("input.codec", {"h264": 1, "h265": 2}) == 2


def test_unknown_value_names_the_key_the_value_and_the_choices(tmp_path):
    config = config_file(tmp_path, "input:\n  codec: vp8\n")
    expected = "input.codec: unknown value 'vp8' (expected h264, h265, mjpeg)"
    with pytest.raises(ValueError) as error:
        config.choice("input.codec", {"mjpeg": 3, "h264": 1, "h265": 2})
    assert str(error.value) == expected
    with pytest.raises(ValueError) as error:
        lookup("input.codec", "vp8", {"h264": 1, "h265": 2, "mjpeg": 3})
    assert str(error.value) == expected


MODELS = (
    "model:\n"
    "  - id: yolov8\n    path: a.tar.gz\n    labels: coco.txt\n"
    "  - id: 5\n    path: b.tar.gz\n"
)


def test_mapping_list_reads_each_entry_by_field(tmp_path):
    entries = config_file(tmp_path, MODELS).mapping_list("model")
    assert len(entries) == 2
    assert entries[0].string("id") == "yolov8"
    assert entries[0].string("labels") == "coco.txt"
    assert entries[1].string("path") == "b.tar.gz"
    assert entries[0].choice("id", {"yolov8": 8}) == 8


def test_mapping_list_errors_name_the_entry_and_field(tmp_path):
    entries = config_file(tmp_path, MODELS).mapping_list("model")
    with pytest.raises(ValueError) as error:
        entries[1].string("labels")
    assert str(error.value) == f"{tmp_path / 'config.yaml'} is missing model[1].labels"
    with pytest.raises(ValueError) as error:
        entries[1].string("id")
    assert str(error.value) == "model[1].id must be a string, got '5'"
    with pytest.raises(ValueError) as error:
        entries[0].choice("id", {"yolo26": 26})
    assert str(error.value) == "model[0].id: unknown value 'yolov8' (expected yolo26)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("input:\n  fps: 30\n", "{config} is missing model"),
        ("model:\n  path: a.tar.gz\n", "model must be a list of mappings, got a mapping"),
        ("model: a.tar.gz\n", "model must be a list of mappings, got 'a.tar.gz'"),
        ("model: []\n", "model must be a non-empty list of mappings"),
        ("model:\n", "model must be a list of mappings, got null"),
        ("model:\n  - path: a.tar.gz\n  -\n", "model[1] must be a mapping, got null"),
        ("model:\n  - path: a.tar.gz\n  - [b, c]\n", "model[1] must be a mapping, got a list"),
        (
            "model:\n  - path: a.tar.gz\n  - b.tar.gz\n",
            "model[1] must be a mapping, got 'b.tar.gz'",
        ),
    ],
)
def test_mapping_list_rejects_what_is_not_a_list_of_mappings(tmp_path, text, expected):
    config = config_file(tmp_path, text)
    with pytest.raises(ValueError) as error:
        config.mapping_list("model")
    assert str(error.value) == expected.format(config=tmp_path / "config.yaml")
