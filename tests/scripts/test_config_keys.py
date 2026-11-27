"""Converted examples name their config settings as examples/config-keys.yaml lists them."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = APPS_ROOT / "examples" / "config-keys.yaml"
EXAMPLE_ONLY_PREFIXES = ("app.",)
TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    ),
    "boolean": lambda value: isinstance(value, bool),
    "string_list": lambda value: (
        isinstance(value, str)
        or (isinstance(value, list) and all(isinstance(item, str) for item in value))
    ),
    "mapping_list": lambda value: (
        isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)
    ),
}

# Examples that still use their own key names. Remove an example from this list in the pull
# request that converts it (#770).
PENDING_EXAMPLES = {
    "depth-estimator",
    "detection-to-vlm-assistant",
    "efficientsam3-promptable-segmenter",
    "face-detector",
    "face-recognizer",
    "fastflow-anomaly-detector",
    "image-classification-explorer",
    "mipi-multi-model",
    "multi-stream-blazepose3d",
    "multi-stream-object-detector",
    "multi-stream-pose-estimator",
    "multi-stream-tracker",
    "neat-genai-studio",
    "patchcore-anomaly-detector",
    "pcb-defect-detector",
    "pcie-high-density-multi-stream-object-detector",
    "single-stream-instance-segmenter",
    "single-stream-thermal-face-detector",
    "ssd-mobilenet-object-detector",
}


def shared_keys() -> dict[str, dict[str, str]]:
    return yaml.safe_load(KEY_FILE.read_text(encoding="utf-8"))


def config_files_by_example() -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for path in sorted(APPS_ROOT.glob("examples/*/*/src/common/config*.yaml")):
        files.setdefault(path.parents[2].name, []).append(path)
    return files


def key_paths(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Every leaf of a config as {"input.rtsp.tcp": value}. A list is one leaf."""
    paths: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            paths.update(key_paths(value, f"{prefix}{key}."))
        else:
            paths[f"{prefix}{key}"] = value
    return paths


def entry_problems(
    name: Path, path: str, entries: list[dict[str, Any]], keys: dict[str, dict[str, Any]]
) -> list[str]:
    """Each field of each entry of the mapping_list at `path` is a field the key lists, of the
    type of the registered key it names."""
    problems = []
    for index, entry in enumerate(entries):
        for field, value in key_paths(entry).items():
            key = f"{path}.{field}"
            if key not in keys[path]["fields"]:
                problems.append(f"{name}: {path}[{index}].{field} is not a field of {path}")
            elif not TYPE_CHECKS[keys[key]["type"]](value):
                problems.append(
                    f"{name}: {path}[{index}].{field} must be a {keys[key]['type']}, "
                    f"got {value!r}"
                )
    return problems


def naming_problems(config_file: Path, keys: dict[str, dict[str, Any]]) -> list[str]:
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    problems = []
    for path, value in key_paths(config).items():
        if path.startswith(EXAMPLE_ONLY_PREFIXES):
            continue
        name = config_file.relative_to(APPS_ROOT)
        if path not in keys:
            problems.append(
                f"{name}: {path} is not in examples/config-keys.yaml; use its name "
                "for this setting, or move it under app."
            )
        elif not TYPE_CHECKS[keys[path]["type"]](value):
            problems.append(
                f"{name}: {path} must be a {keys[path]['type']}, got {value!r}"
            )
        elif keys[path]["type"] == "mapping_list":
            problems += entry_problems(name, path, value, keys)
    return problems


def test_every_shared_key_has_a_type_and_a_meaning():
    for key, entry in shared_keys().items():
        assert entry.get("type") in TYPE_CHECKS, (
            f"{key}: unknown type {entry.get('type')!r}"
        )
        assert entry.get("meaning"), f"{key}: missing meaning"


def test_every_mapping_list_field_is_a_registered_key_under_it():
    keys = shared_keys()
    lists = {key: entry for key, entry in keys.items() if entry["type"] == "mapping_list"}
    assert lists, "no mapping_list key, so this check would pass without checking"
    for key, entry in lists.items():
        assert entry.get("fields"), f"{key}: a mapping_list lists its fields"
        for field in entry["fields"]:
            assert field.startswith(f"{key}.") and field in keys, (
                f"{key}: field {field} is not a registered {key}.<field> key"
            )
            assert keys[field]["type"] != "mapping_list", f"{key}: field {field} is a list"


def test_a_list_of_entries_is_checked_field_by_field(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "APPS_ROOT", tmp_path)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "model:\n"
        "  - id: yolov8\n    path: a.tar.gz\n    labels: coco.txt\n"
        "  - id: 5\n    path: b.tar.gz\n    size: 640\n",
        encoding="utf-8",
    )
    assert naming_problems(config_file, shared_keys()) == [
        "config.yaml: model[1].id must be a string, got 5",
        "config.yaml: model[1].size is not a field of model",
    ]
    config_file.write_text("model: []\n", encoding="utf-8")
    assert naming_problems(config_file, shared_keys()) == [
        "config.yaml: model must be a mapping_list, got []"
    ]


def test_converted_examples_use_the_shared_key_names():
    keys = shared_keys()
    converted = {
        example: files
        for example, files in config_files_by_example().items()
        if example not in PENDING_EXAMPLES
    }
    assert converted, (
        "no converted example found, so this check would pass without checking"
    )

    problems = [
        problem
        for files in converted.values()
        for config_file in files
        for problem in naming_problems(config_file, keys)
    ]
    assert not problems, "\n".join(problems)


def test_pending_examples_still_need_converting():
    keys = shared_keys()
    examples = config_files_by_example()
    unknown = sorted(PENDING_EXAMPLES - set(examples))
    assert not unknown, (
        f"PENDING_EXAMPLES names examples without a config file: {unknown}"
    )

    done = sorted(
        example
        for example in PENDING_EXAMPLES
        if not any(
            naming_problems(config_file, keys) for config_file in examples[example]
        )
    )
    assert not done, (
        f"these examples already use the shared names; remove them from PENDING_EXAMPLES: {done}"
    )
