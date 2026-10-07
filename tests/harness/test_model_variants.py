"""Harness self-tests for running an e2e suite once per scoped model variant."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.utils.e2e_config import (
    resolve_configured_model_path,
    resolve_configured_model_paths,
    write_merged_config,
)
from tests.utils.test_scope import MODEL_VARIANTS_ENV

pytestmark = pytest.mark.unit


def _write_example(root: Path) -> Path:
    example_dir = root / "examples" / "classification" / "demo-example"
    (example_dir / "src" / "common").mkdir(parents=True)
    (example_dir / "tests").mkdir()
    (example_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    (example_dir / "src" / "common" / "config.yaml").write_text(
        "model:\n  path: models/demo_model_mpk.tar.gz\n", encoding="utf-8"
    )
    (example_dir / "tests" / "test-scope.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "demo-model": {"source": "modelzoo", "name": "demo_model", "file": "demo_model_mpk.tar.gz"},
                    "small-model": {"source": "modelzoo", "name": "small_model", "file": "small_model_mpk.tar.gz"},
                },
                "unit": {"python": False, "cpp": False},
                "e2e": {
                    "python": {"enabled": True, "models": ["demo-model"], "variants": ["small-model"]},
                    "cpp": {"enabled": False, "models": []},
                },
            }
        ),
        encoding="utf-8",
    )
    return example_dir


def test_variants_follow_the_suite_model_only_when_asked(tmp_path, monkeypatch):
    example_dir = _write_example(tmp_path)
    monkeypatch.setenv("SIMANEAT_APPS_TEST_SCOPE_FILE", str(tmp_path / "examples"))
    common_config = example_dir / "src" / "common" / "config.yaml"
    models_dir = tmp_path / "models"

    monkeypatch.delenv(MODEL_VARIANTS_ENV, raising=False)
    assert resolve_configured_model_paths(common_config, models_dir) == [
        models_dir / "demo_model_mpk.tar.gz"
    ]

    monkeypatch.setenv(MODEL_VARIANTS_ENV, "1")
    assert resolve_configured_model_paths(common_config, models_dir) == [
        models_dir / "demo_model_mpk.tar.gz",
        models_dir / "small_model_mpk.tar.gz",
    ]
    # The single-model resolver keeps answering with the suite's model.
    assert resolve_configured_model_path(common_config, models_dir) == (
        models_dir / "demo_model_mpk.tar.gz"
    )


def test_config_writer_uses_the_model_a_test_was_generated_for(tmp_path, monkeypatch):
    example_dir = _write_example(tmp_path)
    monkeypatch.setenv("SIMANEAT_APPS_TEST_SCOPE_FILE", str(tmp_path / "examples"))
    common_config = example_dir / "src" / "common" / "config.yaml"
    models_dir = tmp_path / "models"

    default = write_merged_config(common_config, tmp_path / "default.yaml", {}, models_dir)
    variant = write_merged_config(
        common_config,
        tmp_path / "variant.yaml",
        {},
        models_dir,
        model_path=models_dir / "small_model_mpk.tar.gz",
    )

    assert yaml.safe_load(default.read_text())["model"]["path"] == str(models_dir / "demo_model_mpk.tar.gz")
    assert yaml.safe_load(variant.read_text())["model"]["path"] == str(models_dir / "small_model_mpk.tar.gz")
