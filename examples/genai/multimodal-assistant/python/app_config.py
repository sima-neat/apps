"""Configuration for the Multimodal Assistant example."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


APPS_ROOT = Path(__file__).resolve().parents[4]
COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
DEFAULT_CONFIG = COMMON_DIR / "config.yaml"
DEFAULT_SERVER_CONFIG = DEFAULT_CONFIG
DEFAULT_WEB_CONFIG = DEFAULT_CONFIG


@dataclass(frozen=True)
class ServedModel:
    name: str
    path: Path | None


@dataclass(frozen=True)
class OpenAIConfig:
    host: str
    client_host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.client_host}:{self.port}"


@dataclass(frozen=True)
class RequestConfig:
    max_tokens: int
    system_prompt: str


@dataclass(frozen=True)
class WebConfig:
    host: str
    port: int
    https: bool


@dataclass(frozen=True)
class RagConfig:
    enabled: bool


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig
    chat_model: ServedModel
    asr_model: ServedModel
    request: RequestConfig
    web: WebConfig
    rag: RagConfig


def load_config(path: Path = DEFAULT_CONFIG, apps_root: Path = APPS_ROOT) -> AppConfig:
    return load_server_config(path, apps_root)


def load_server_config(path: Path = DEFAULT_SERVER_CONFIG, apps_root: Path = APPS_ROOT) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = raw.get("server", raw)

    openai = raw.get("openai", {})
    models = raw.get("models", {})
    request = raw.get("request", {})
    web = raw.get("web", {})
    rag = raw.get("rag", {})

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=_load_required_model(models, "chat", apps_root),
        asr_model=_load_required_model(models, "asr", apps_root),
        request=RequestConfig(
            max_tokens=int(request.get("max_tokens", 128)),
            system_prompt=str(request.get("system_prompt", "") or ""),
        ),
        web=WebConfig(
            host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
            port=int(web.get("port", 5000)),
            https=_load_bool(web.get("https", True)),
        ),
        rag=RagConfig(enabled=_load_bool(rag.get("enabled", False))),
    )


def load_web_config(path: Path = DEFAULT_WEB_CONFIG) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = raw.get("app", raw)

    openai = raw.get("openai", {})
    models = raw.get("models", {})
    request = raw.get("request", {})
    web = raw.get("web", {})
    rag = raw.get("rag", {})

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=_load_model_name(models, "chat"),
        asr_model=_load_model_name(models, "asr"),
        request=RequestConfig(
            max_tokens=int(request.get("max_tokens", 128)),
            system_prompt=str(request.get("system_prompt", "") or ""),
        ),
        web=WebConfig(
            host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
            port=int(web.get("port", 5000)),
            https=_load_bool(web.get("https", True)),
        ),
        rag=RagConfig(enabled=_load_bool(rag.get("enabled", False))),
    )


def _load_required_model(models: dict, key: str, apps_root: Path) -> ServedModel:
    raw = models.get(key, {})
    if not isinstance(raw, dict):
        raise ValueError(f"models.{key} must be a mapping")

    name = str(raw.get("name", "")).strip()
    path_text = str(raw.get("path", "")).strip()
    if not name:
        raise ValueError(f"models.{key}.name is required")
    if not path_text:
        raise ValueError(f"models.{key}.path is required")

    model_path = Path(path_text).expanduser()
    if not model_path.is_absolute():
        model_path = apps_root / model_path
    return ServedModel(name=name, path=model_path)


def _load_model_name(models: dict, key: str) -> ServedModel:
    raw = models.get(key, "")
    if isinstance(raw, dict):
        raw = raw.get("name", "")
    name = str(raw).strip()
    if not name:
        raise ValueError(f"models.{key} is required")
    return ServedModel(name=name, path=None)


def _load_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
