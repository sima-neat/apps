"""YAML-backed configuration for the two-VLM OpenAI server example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ServedModel:
    name: str
    path: Path


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    models: tuple[ServedModel, ...]
    max_tokens: int
    system_prompt: str

    def model_names(self) -> tuple[str, ...]:
        return tuple(model.name for model in self.models)

    def require_model(self, name: str) -> ServedModel:
        for model in self.models:
            if model.name == name:
                return model
        available = ", ".join(self.model_names())
        raise ValueError(f"unknown model '{name}'. Available models: {available}")


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    server = raw.get("server", {})
    request = raw.get("request", {})
    models = tuple(_load_model(item, path.parent) for item in raw.get("models", []))
    if len(models) < 2:
        raise ValueError("config requires at least two models")
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    return Config(
        host=server.get("host", "0.0.0.0") or "0.0.0.0",
        port=int(server.get("port", 9998)),
        models=models,
        max_tokens=int(request.get("max_tokens", 96)),
        system_prompt=request.get(
            "system_prompt", "Answer from visual evidence only. Be concise."
        ),
    )


def _load_model(raw: dict, config_dir: Path) -> ServedModel:
    name = str(raw.get("name", "")).strip()
    path_text = str(raw.get("path", "")).strip()
    if not name:
        raise ValueError("each model requires a non-empty name")
    if not path_text:
        raise ValueError(f"model '{name}' requires a path")
    model_path = Path(path_text).expanduser()
    if not model_path.is_absolute():
        model_path = (config_dir / model_path).resolve()
    return ServedModel(name=name, path=model_path)
