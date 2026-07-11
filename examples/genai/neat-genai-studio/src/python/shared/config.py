"""Configuration for the Neat GenAI Studio example."""

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
class ControlConfig:
    host: str
    client_host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.client_host}:{self.port}"


@dataclass(frozen=True)
class HubConfig:
    allow_download: bool
    orgs: tuple[str, ...] = ("simaai", "TDoSiMa")

    @property
    def org(self) -> str:
        return self.orgs[0] if self.orgs else ""


@dataclass(frozen=True)
class UIConfig:
    font_family: str
    font_size: int


@dataclass(frozen=True)
class AppConfig:
    openai: OpenAIConfig
    # chat/VLM and ASR models are optional: the server can start with none resident
    # and models are loaded on demand from the UI catalog.
    chat_model: ServedModel | None
    chat_models: tuple[ServedModel, ...]
    asr_model: ServedModel | None
    request: RequestConfig
    web: WebConfig
    rag: RagConfig
    control: ControlConfig = ControlConfig(host="127.0.0.1", client_host="127.0.0.1", port=9997)
    catalog_dir: Path | None = None
    max_resident_chat_models: int = 1
    hub: HubConfig = HubConfig(allow_download=True, orgs=("simaai", "TDoSiMa"))
    ui: UIConfig = UIConfig(font_family="Inter", font_size=15)


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
    control = raw.get("control", {})
    hub = raw.get("hub", {})
    ui = raw.get("ui", {})
    chat_models = _load_startup_chat_models(models, apps_root)

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=chat_models[0] if chat_models else None,
        chat_models=chat_models,
        asr_model=_load_optional_model(models, "asr", apps_root),
        request=RequestConfig(
            max_tokens=int(request.get("max_tokens", 256)),
            system_prompt=str(request.get("system_prompt", "") or ""),
        ),
        web=WebConfig(
            host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
            port=int(web.get("port", 5000)),
            https=_load_bool(web.get("https", True)),
        ),
        rag=_load_rag_config(rag, apps_root),
        control=_load_control_config(control),
        catalog_dir=_load_optional_dir(models.get("catalog_dir"), apps_root),
        max_resident_chat_models=max(1, int(models.get("max_resident_chat_models", 1) or 1)),
        hub=_load_hub_config(hub),
        ui=_load_ui_section(ui),
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
    control = raw.get("control", {})
    hub = server.get("hub", {})
    ui = raw.get("ui", {})
    chat_models = _load_chat_model_names(server_models, apps_root)

    return AppConfig(
        openai=OpenAIConfig(
            host=str(openai.get("host", "0.0.0.0") or "0.0.0.0"),
            client_host=str(openai.get("client_host", "127.0.0.1") or "127.0.0.1"),
            port=int(openai.get("port", 9998)),
        ),
        chat_model=chat_models[0] if chat_models else None,
        chat_models=chat_models,
        asr_model=_load_optional_model_name(server_models, "asr"),
        request=RequestConfig(
            max_tokens=int(request.get("max_tokens", 256)),
            system_prompt=str(request.get("system_prompt", "") or ""),
        ),
        web=WebConfig(
            host=str(web.get("host", "0.0.0.0") or "0.0.0.0"),
            port=int(web.get("port", 5000)),
            https=_load_bool(web.get("https", True)),
        ),
        rag=_load_rag_config(rag, apps_root),
        control=_load_control_config(control),
        catalog_dir=_load_optional_dir(server_models.get("catalog_dir"), apps_root),
        max_resident_chat_models=max(1, int(server_models.get("max_resident_chat_models", 1) or 1)),
        hub=_load_hub_config(hub),
        ui=_load_ui_section(ui),
    )


def _load_startup_chat_models(models: dict, apps_root: Path) -> tuple[ServedModel, ...]:
    """Chat/VLM models to load at startup. May be empty (load on demand instead)."""
    raw = models.get("chat")
    if isinstance(raw, list):
        return tuple(
            _load_required_model_entry(entry, f"models.chat[{index}]", apps_root)
            for index, entry in enumerate(raw)
        )
    if not raw:
        return ()
    return (_load_required_model_entry(raw, "models.chat", apps_root),)


def _load_optional_model(models: dict, key: str, apps_root: Path) -> ServedModel | None:
    """A single model (e.g. ASR) to load at startup, or None if not configured."""
    raw = models.get(key)
    if not raw:
        return None
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
    raw = models.get("chat")
    if not raw:
        return ()
    entries = raw if isinstance(raw, list) else [raw]

    return tuple(
        model for model in (_load_chat_model_entry(entry, apps_root) for entry in entries)
        if model.name
    )


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


def classify_model_dir(model_dir: Path) -> dict | None:
    """Classify a directory as a servable model, or return None if incompatible.

    A compatible precompiled model directory contains a ``devkit/`` folder. Chat
    and vision-language models expose ``devkit/vlm_config.json``; ASR (Whisper)
    models expose a ``whisper_config.json``. The returned dict mirrors the fields
    used by the UI catalog: ``name``, ``path``, ``type`` (``chat`` | ``vlm`` |
    ``asr``), ``supports_vision``, and ``image_size``.
    """
    try:
        if not model_dir.is_dir():
            return None
    except OSError:
        return None

    devkit = model_dir / "devkit"
    vlm_cfg = devkit / "vlm_config.json"
    whisper_cfg_candidates = (
        devkit / "whisper_config.json",
        model_dir / "whisper_config.json",
    )

    if vlm_cfg.is_file():
        supports_vision = _load_supports_vision({}, model_dir)
        return {
            "name": model_dir.name,
            "path": model_dir,
            "type": "vlm" if supports_vision else "chat",
            "supports_vision": supports_vision,
            "image_size": _load_vision_image_size({}, model_dir) if supports_vision else None,
        }

    if any(candidate.is_file() for candidate in whisper_cfg_candidates):
        return {
            "name": model_dir.name,
            "path": model_dir,
            "type": "asr",
            "supports_vision": False,
            "image_size": None,
        }

    return None


def model_dir_complete(path) -> tuple[bool, str]:
    """Best-effort check that a model directory holds a COMPLETE set of weights
    (not a partial or interrupted download). Returns (complete, reason).

    Signals, cheapest first: our own `.neat-complete` marker (written after a
    successful in-app download); huggingface leftover `*.incomplete` temp files
    (an interrupted download); and, for compiled models, that `elf_files/` holds
    at least one non-empty `.elf`. Errs on the side of "complete" if it can't
    tell, so it never blocks a legitimately-present model.
    """
    try:
        p = path if isinstance(path, Path) else Path(path)
    except Exception:
        return True, ""
    if not p or not p.is_dir():
        return False, "model directory is missing"
    # Definitive: marker written only after a fully successful download.
    if (p / ".neat-complete").exists():
        return True, ""
    # An interrupted huggingface_hub download leaves .incomplete temp files.
    try:
        for _ in p.rglob("*.incomplete"):
            return False, "download is incomplete (partial files present)"
    except OSError:
        pass
    # A compiled model needs its ELF weight files, none of them zero-byte.
    elf_dir = p / "elf_files"
    if elf_dir.is_dir():
        elfs = list(elf_dir.glob("*.elf"))
        if not elfs:
            return False, "no weight (.elf) files were found"
        for e in elfs:
            try:
                if e.stat().st_size == 0:
                    return False, f"incomplete weight file: {e.name}"
            except OSError:
                return False, f"unreadable weight file: {e.name}"
    return True, ""


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


def _load_optional_model_name(models: dict, key: str) -> ServedModel | None:
    raw = models.get(key, "")
    if isinstance(raw, dict):
        raw = raw.get("name", "")
    name = str(raw).strip()
    if not name:
        return None
    return ServedModel(name=name, path=None)


def _load_rag_config(raw: object, apps_root: Path) -> RagConfig:
    rag = raw if isinstance(raw, dict) else {}
    return RagConfig(
        enabled=_load_bool(rag.get("enabled", False)),
        embedding_model_dir=_load_optional_path(rag.get("embedding_model_dir", ""), apps_root),
    )


def _load_control_config(raw: object) -> ControlConfig:
    control = raw if isinstance(raw, dict) else {}
    return ControlConfig(
        host=str(control.get("host", "127.0.0.1") or "127.0.0.1"),
        client_host=str(control.get("client_host", "127.0.0.1") or "127.0.0.1"),
        port=int(control.get("port", 9997)),
    )


def _load_hub_config(raw: object) -> HubConfig:
    hub = raw if isinstance(raw, dict) else {}
    raw_orgs = hub.get("orgs", hub.get("org", ["simaai", "TDoSiMa"]))
    if isinstance(raw_orgs, str):
        orgs = tuple(o.strip() for o in raw_orgs.replace(",", " ").split() if o.strip())
    elif isinstance(raw_orgs, (list, tuple)):
        orgs = tuple(str(o).strip() for o in raw_orgs if str(o).strip())
    else:
        orgs = ()
    if not orgs:
        orgs = ("simaai", "TDoSiMa")
    return HubConfig(
        allow_download=_load_bool(hub.get("allow_download", True)),
        orgs=orgs,
    )


def _load_ui_section(raw: object) -> UIConfig:
    ui = raw if isinstance(raw, dict) else {}
    return UIConfig(
        font_family=str(ui.get("font_family", "Inter") or "Inter").strip(),
        font_size=int(ui.get("font_size", 15) or 15),
    )


def _load_optional_dir(value: object, apps_root: Path) -> Path | None:
    path_text = str(value or "").strip()
    if not path_text or path_text.startswith("<"):
        return None

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = _resolve_relative_path(path, apps_root)
    return path


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
