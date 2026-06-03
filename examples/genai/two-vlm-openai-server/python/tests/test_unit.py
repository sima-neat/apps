"""Unit tests for the two-VLM OpenAI server example."""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config  # noqa: E402


def test_load_config_requires_two_models(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        textwrap.dedent(
            """
            models:
              - name: vlm-1
                path: model-a
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two models"):
        load_config(config)


def test_load_config_resolves_models_and_names(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text(
        textwrap.dedent(
            """
            server:
              host: 127.0.0.1
              port: 9999
            models:
              - name: vlm-1
                path: model-a
              - name: vlm-2
                path: /models/model-b
            request:
              max_tokens: 32
              system_prompt: Use short answers.
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(work_dir)
    cfg = load_config(config)

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 9999
    assert cfg.model_names() == ("vlm-1", "vlm-2")
    assert cfg.models[0].path == work_dir / "model-a"
    assert cfg.models[1].path == Path("/models/model-b")
    assert cfg.max_tokens == 32
    assert cfg.system_prompt == "Use short answers."
    assert cfg.require_model("vlm-2").name == "vlm-2"


def test_load_config_rejects_duplicate_model_names(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        textwrap.dedent(
            """
            models:
              - name: vlm
                path: model-a
              - name: vlm
                path: model-b
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_config(config)
