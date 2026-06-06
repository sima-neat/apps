"""Shared e2e test config and artifact helpers."""

import shutil
from pathlib import Path
from typing import Any

import yaml

APPS_ROOT = Path(__file__).resolve().parents[2]
E2E_CONFIG_PATH = APPS_ROOT / "tests" / "configs" / "e2e.yaml"


def required_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{E2E_CONFIG_PATH} missing mapping: {name}")
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_e2e_config() -> dict[str, Any]:
    with E2E_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return required_mapping(raw.get("e2e"), "e2e")


def prepare_output_dir(base_root: Path, language: str, example_name: str, test_name: str) -> Path:
    run_dir = base_root / language / example_name / test_name
    out_dir = run_dir / "out"
    shutil.rmtree(run_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_merged_config(common_config: Path, config_path: Path, overrides: dict[str, Any]) -> Path:
    with common_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    config_path.write_text(
        yaml.safe_dump(deep_update(config, overrides), sort_keys=False),
        encoding="utf-8",
    )
    return config_path
