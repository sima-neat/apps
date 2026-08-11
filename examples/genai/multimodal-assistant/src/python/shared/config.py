"""Configuration for the Multimodal Assistant example."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


EXAMPLE_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
DEFAULT_CONFIG = COMMON_DIR / "config.yaml"
DEFAULT_SERVER_CONFIG = DEFAULT_CONFIG
DEFAULT_UI_CONFIG = DEFAULT_CONFIG


def _detect_apps_root() -> Path | None:
    maybe_apps_root = EXAMPLE_ROOT.parent.parent.parent
    if EXAMPLE_ROOT.parent.parent.name == "examples" and (maybe_apps_root / "examples").is_dir():
        return maybe_apps_root
    return None


APPS_ROOT = _detect_apps_root()
PATH_ROOT = APPS_ROOT or EXAMPLE_ROOT


@dataclass(frozen=True)
class ServedModel:
    name: str
    path: Path | None
    supports_vision: bool = True
    vision_image_size: dict[str, int] | None = None


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
    embedding_model_dir: str


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig
    chat_model: ServedModel
    chat_models: tuple[ServedModel, ...]
    asr_model: ServedModel
    request: RequestConfig
    web: WebConfig
    rag: RagConfig
    tts_streaming: bool = True


def load_config(path: Path = DEFAULT_CONFIG, apps_root: Path = PATH_ROOT) -> AppConfig:
    return load_server_config(path, apps_root)


def load_server_config(path: Path = DEFAULT_SERVER_CONFIG, apps_root: Path = PATH_ROOT) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = raw.get("server", raw)

    openai = raw.get("openai", {})
    models = raw.get("models", {})
    request = raw.get("request", {})
    web = raw.get("web", {})
    rag = raw.get("rag", {})
    chat_models = _load_required_chat_models(models, apps_root)

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=chat_models[0],
        chat_models=chat_models,
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
        rag=_load_rag_config(rag, apps_root),
    )


def load_ui_config(path: Path = DEFAULT_UI_CONFIG, apps_root: Path = PATH_ROOT) -> AppConfig:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = config.get("app", config)
    server = config.get("server", {})

    openai = raw.get("openai", {})
    server_models = server.get("models", {})
    request = raw.get("request", {})
    web = raw.get("web", {})
    rag = raw.get("rag", {})
    tts = raw.get("tts", {})
    chat_models = _load_chat_model_names(server_models, apps_root)

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=chat_models[0],
        chat_models=chat_models,
        asr_model=_load_model_name(server_models, "asr"),
        request=RequestConfig(
            max_tokens=int(request.get("max_tokens", 128)),
            system_prompt=str(request.get("system_prompt", "") or ""),
        ),
        web=WebConfig(
            host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
            port=int(web.get("port", 5000)),
            https=_load_bool(web.get("https", True)),
        ),
        rag=_load_rag_config(rag, apps_root),
        tts_streaming=_load_bool(tts.get("streaming", True)),
    )


def _load_required_chat_models(models: dict, apps_root: Path) -> tuple[ServedModel, ...]:
    raw = models.get("chat", {})
    if isinstance(raw, list):
        if not raw:
            raise ValueError("models.chat must contain at least one model")
        return tuple(
            _load_required_model_entry(entry, f"models.chat[{index}]", apps_root)
            for index, entry in enumerate(raw)
        )

    return (_load_required_model(models, "chat", apps_root),)


def _load_required_model(models: dict, key: str, apps_root: Path) -> ServedModel:
    raw = models.get(key, {})
    return _load_required_model_entry(raw, f"models.{key}", apps_root)


def _load_required_model_entry(raw: object, label: str, apps_root: Path) -> ServedModel:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")

    name = str(raw.get("name", "")).strip()
    path_text = str(raw.get("path", "")).strip()
    if not name:
        raise ValueError(f"{label}.name is required")
    if not path_text:
        raise ValueError(f"{label}.path is required")

    model_path = Path(path_text).expanduser()
    if not model_path.is_absolute():
        model_path = _resolve_relative_path(model_path, apps_root)
    supports_vision = _load_supports_vision(raw, model_path)
    return ServedModel(
        name=name,
        path=model_path,
        supports_vision=supports_vision,
        vision_image_size=_load_vision_image_size(raw, model_path) if supports_vision else None,
    )


def _load_chat_model_names(models: dict, apps_root: Path) -> tuple[ServedModel, ...]:
    raw = models.get("chat", {})
    entries = raw if isinstance(raw, list) else [raw]

    chat_models = tuple(
        model for model in (_load_chat_model_entry(entry, apps_root) for entry in entries)
        if model.name
    )
    if not chat_models:
        raise ValueError("models.chat requires at least one model name")
    return chat_models


def _load_chat_model_entry(entry: object, apps_root: Path) -> ServedModel:
    if not isinstance(entry, dict):
        return ServedModel(name=str(entry).strip(), path=None)

    name = _load_model_entry_name(entry)
    path_text = str(entry.get("path", "") or "").strip()
    model_path = Path(path_text).expanduser() if path_text else None
    if model_path is not None and not model_path.is_absolute():
        model_path = _resolve_relative_path(model_path, apps_root)
    supports_vision = (
        _load_supports_vision(entry, model_path)
        if model_path is not None
        else _load_bool(entry.get("supports_vision", True))
    )
    return ServedModel(
        name=name,
        path=model_path,
        supports_vision=supports_vision,
        vision_image_size=_load_vision_image_size(entry, model_path) if supports_vision else None,
    )


def _load_model_entry_name(entry: object) -> str:
    if isinstance(entry, dict):
        entry = entry.get("name", "")
    return str(entry).strip()


def _load_supports_vision(raw: dict, model_path: Path | None) -> bool:
    if "supports_vision" in raw:
        return _load_bool(raw.get("supports_vision"))

    if model_path is None:
        return True

    config_path = model_path / "devkit" / "vlm_config.json"
    if not config_path.is_file():
        return True

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return True

    return bool(str(cfg.get("vision_model_name", "") or "").strip())


def _load_vision_image_size(raw: dict, model_path: Path | None) -> dict[str, int] | None:
    raw_size = raw.get("vision_image_size") or raw.get("image_size")

    if raw_size is None and model_path is not None:
        config_path = model_path / "devkit" / "vlm_config.json"
        if config_path.is_file():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                raw_size = cfg.get("vm_cfg", {}).get("image_size")
            except Exception:
                raw_size = None

    return _parse_vision_image_size(raw_size)


def _parse_vision_image_size(raw_size: object) -> dict[str, int] | None:
    height = width = None

    if isinstance(raw_size, int):
        height = width = raw_size
    elif isinstance(raw_size, float):
        height = width = int(raw_size)
    elif isinstance(raw_size, str):
        normalized = raw_size.strip().lower().replace(" ", "")
        if "x" in normalized:
            parts = normalized.split("x", 1)
            if len(parts) == 2:
                try:
                    height = int(float(parts[0]))
                    width = int(float(parts[1]))
                except ValueError:
                    return None
        else:
            try:
                height = width = int(float(normalized))
            except ValueError:
                return None
    elif isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
        try:
            height = int(float(raw_size[0]))
            width = int(float(raw_size[1]))
        except (TypeError, ValueError):
            return None

    if not height or not width or height <= 0 or width <= 0:
        return None

    return {"height": height, "width": width}


def _load_model_name(models: dict, key: str) -> ServedModel:
    raw = models.get(key, "")
    if isinstance(raw, dict):
        raw = raw.get("name", "")
    name = str(raw).strip()
    if not name:
        raise ValueError(f"models.{key} is required")
    return ServedModel(name=name, path=None)


def _load_rag_config(raw: object, apps_root: Path) -> RagConfig:
    rag = raw if isinstance(raw, dict) else {}
    return RagConfig(
        enabled=_load_bool(rag.get("enabled", False)),
        embedding_model_dir=_load_optional_path(rag.get("embedding_model_dir", ""), apps_root),
    )


def _load_optional_path(value: object, apps_root: Path) -> str:
    path_text = str(value or "").strip()
    if not path_text:
        return ""

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = _resolve_relative_path(path, apps_root)
    return str(path)


def _resolve_relative_path(path: Path, apps_root: Path) -> Path:
    candidate_roots = (apps_root, EXAMPLE_ROOT)
    for root in candidate_roots:
        candidate = root / path
        if candidate.exists():
            return candidate

    return apps_root / path


def _load_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
