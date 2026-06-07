"""Shared e2e test config and artifact helpers."""

import copy
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]


def required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"missing mapping: {name}")
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return required_mapping(raw, str(path))


def common_config_path_for_example(example_name: str) -> Path:
    matches = sorted(APPS_ROOT.glob(f"examples/*/{example_name}/common/config.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one common/config.yaml for example '{example_name}', found {len(matches)}"
        )
    return matches[0]


def common_config_path_for_test(test_file: Path) -> Path:
    return test_file.parents[2] / "common" / "config.yaml"


def load_common_config(common_config: Path) -> dict[str, Any]:
    return load_yaml_mapping(common_config)


def section_from_common_config(example_name: str, section: str) -> dict[str, Any]:
    config = load_common_config(common_config_path_for_example(example_name))
    value: Any = config
    for part in section.split("."):
        value = required_mapping(value, section).get(part)
    return required_mapping(value, f"{example_name}.{section}")


def testing_e2e_from_config(config: dict[str, Any]) -> dict[str, Any]:
    testing = config.get("testing", {})
    if not testing:
        return {}
    return required_mapping(testing.get("e2e", {}), "testing.e2e")


def runtime_e2e_overrides(config: dict[str, Any]) -> dict[str, Any]:
    overrides = copy.deepcopy(testing_e2e_from_config(config))
    output = overrides.get("output")
    if isinstance(output, dict):
        output.pop("total_saved_frames", None)
    return overrides


def configured_model_path(config: dict[str, Any]) -> Path | None:
    model = config.get("model")
    if isinstance(model, dict):
        raw = model.get("path")
    elif isinstance(model, str):
        raw = model
    else:
        raw = None
    if not raw:
        return None
    return Path(str(raw))


def resolve_configured_model_path(common_config: Path, models_dir: Path) -> Path | None:
    scoped = resolve_scoped_model_path(common_config, models_dir)
    if scoped is not None:
        return scoped

    model_path = configured_model_path(load_common_config(common_config))
    if model_path is None:
        return None
    return models_dir / model_path.name


def resolve_scoped_model_path(common_config: Path, models_dir: Path) -> Path | None:
    scope_file = Path(
        str(
            os.environ.get(
                "SIMANEAT_APPS_TEST_SCOPE_FILE",
                APPS_ROOT / "tests" / "configs" / "test-scope.yaml",
            )
        )
    )
    if not scope_file.exists():
        return None

    scope = load_yaml_mapping(scope_file)
    examples = required_mapping(scope.get("examples"), "examples")
    example_dir = common_config.parents[1]
    example_key = f"{example_dir.parent.name}/{example_dir.name}"
    entry = examples.get(example_key)
    if not isinstance(entry, dict):
        return None

    e2e = entry.get("e2e", {})
    python_e2e = e2e.get("python", {}) if isinstance(e2e, dict) else {}
    selected = python_e2e.get("models", []) if isinstance(python_e2e, dict) else []
    if not selected:
        return None

    models = entry.get("models", {})
    model = models.get(str(selected[0])) if isinstance(models, dict) else None
    if not isinstance(model, dict):
        return None

    file_name = str(model.get("file", "") or "").strip()
    return models_dir / file_name if file_name else None


def apply_model_path(config: dict[str, Any], model_path: Path | None) -> None:
    if model_path is None:
        return
    if isinstance(config.get("model"), dict):
        config["model"]["path"] = str(model_path)
    elif isinstance(config.get("model"), str):
        config["model"] = str(model_path)


def _repo_path(value: Any) -> Any:
    if value is None:
        return None
    raw = str(value)
    if not raw or "://" in raw:
        return value
    path = Path(raw)
    if path.is_absolute():
        return raw
    return str(APPS_ROOT / path)


def materialize_repo_paths(config: dict[str, Any]) -> None:
    model = config.get("model")
    if isinstance(model, dict):
        if "path" in model:
            model["path"] = _repo_path(model["path"])
        if "labels" in model:
            model["labels"] = _repo_path(model["labels"])

    io = config.get("io")
    if isinstance(io, dict):
        for key in ("image", "image1", "image2", "input_dir", "output", "output_dir"):
            if key in io:
                io[key] = _repo_path(io[key])


def prepare_output_dir(base_root: Path, language: str, example_name: str, test_name: str) -> Path:
    run_dir = base_root / language / example_name / test_name
    out_dir = run_dir / "out"
    shutil.rmtree(run_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_merged_config(
    common_config: Path,
    config_path: Path,
    overrides: dict[str, Any],
    models_dir: Path | None = None,
) -> Path:
    config = load_common_config(common_config)
    e2e_overrides = runtime_e2e_overrides(config)
    config.pop("testing", None)
    deep_update(config, e2e_overrides)
    materialize_repo_paths(config)
    if models_dir is not None:
        apply_model_path(config, resolve_configured_model_path(common_config, models_dir))

    config_path.write_text(
        yaml.safe_dump(deep_update(config, overrides), sort_keys=False),
        encoding="utf-8",
    )
    return config_path
