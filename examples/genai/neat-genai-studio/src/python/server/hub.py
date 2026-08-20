#!/usr/bin/env python3
"""Hugging Face Hub helpers for Neat GenAI Studio.

These functions search for and download compatible precompiled models into the
on-disk catalog. They only touch the filesystem and the Hub API (never the
running GenAIServer), so they run in the UI process where ``huggingface_hub`` is
installed. ``internet_reachable`` / ``hub_enabled`` use only the standard
library, so they are also safe to call from the model-server process.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

from shared.chat_template import repair_chat_template_files
from shared.config import HubConfig, classify_model_dir, model_dir_complete


_internet_cache: dict = {"ok": None, "at": 0.0, "refreshing": False}
_internet_lock = threading.Lock()
_REPO_ID_MARKER = ".neat-hub-repo"


def _valid_repo_component(value: str) -> bool:
    """Return whether ``value`` is a safe Hugging Face namespace/name part."""
    if not value or value in {".", ".."} or value[0] in ".-" or value[-1] in ".-":
        return False
    if ".." in value or "--" in value:
        return False
    return all(char.isascii() and (char.isalnum() or char in "-._") for char in value)


def validated_repo_id(repo_id: str, hub: HubConfig | None = None) -> str:
    """Validate and normalize a Hub model id without accepting path syntax."""
    value = (repo_id or "").strip()
    parts = value.split("/")
    if len(value) > 192 or len(parts) != 2 or not all(_valid_repo_component(p) for p in parts):
        raise ValueError("Invalid Hugging Face repository id")
    if hub is not None:
        allowed = {str(org).casefold() for org in (hub.orgs or ("simaai",)) if org}
        if parts[0].casefold() not in allowed:
            raise ValueError("Repository organization is not allowed")
    return value


def safe_name(repo_id: str) -> str:
    """Local catalog directory name for a validated ``org/name`` Hub id."""
    value = validated_repo_id(repo_id)
    org, name = value.split("/", 1)
    # Preserve the existing on-disk layout for official SiMa models. Other
    # allowed organizations include their namespace; '@' cannot occur in a
    # validated component, so this mapping cannot collide with another repo.
    return name if org.casefold() == "simaai" else f"{org}@{name}"


def _catalog_target(catalog_dir: Path, repo_id: str) -> Path:
    """Resolve a model target and prove it remains inside the catalog root."""
    root = Path(catalog_dir).resolve()
    target = (root / safe_name(repo_id)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Model target escapes the catalog directory") from exc
    return target


def _existing_catalog_target(catalog_dir: Path, repo_id: str) -> Path:
    """Prefer the canonical target; trust a legacy path only with ownership metadata."""
    target = _catalog_target(catalog_dir, repo_id)
    if target.exists() or safe_name(repo_id) == repo_id.rsplit("/", 1)[-1]:
        return target

    root = Path(catalog_dir).resolve()
    legacy = (root / repo_id.rsplit("/", 1)[-1]).resolve()
    try:
        legacy.relative_to(root)
    except ValueError as exc:
        raise ValueError("Model target escapes the catalog directory") from exc
    try:
        owner = (legacy / _REPO_ID_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return target
    return legacy if owner.casefold() == repo_id.casefold() else target


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _probe_internet() -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=3):
            return True
    except OSError:
        return False


def _refresh_internet_async() -> None:
    """Re-probe reachability off-thread so callers never block on the socket."""
    with _internet_lock:
        if _internet_cache["refreshing"]:
            return
        _internet_cache["refreshing"] = True

    def _run() -> None:
        ok = _probe_internet()
        with _internet_lock:
            _internet_cache["ok"] = ok
            _internet_cache["at"] = time.monotonic()
            _internet_cache["refreshing"] = False

    threading.Thread(target=_run, name="hub-reachability", daemon=True).start()


def internet_reachable() -> bool:
    now = time.monotonic()
    cached = _internet_cache["ok"]
    if cached is None:
        # Explicit Hub operations need an accurate first result.
        ok = _probe_internet()
        _internet_cache["ok"] = ok
        _internet_cache["at"] = now
        return ok
    if (now - _internet_cache["at"]) >= 30:
        # Stale: refresh in the background and return the last known value now,
        # so frequently-polled callers (e.g. /models/status) never stall on the
        # 3s socket timeout.
        _refresh_internet_async()
    return bool(cached)


def hub_enabled(hub: HubConfig) -> bool:
    return bool(hub.allow_download) and internet_reachable()


# Hub metadata that identifies a repo's modality. ``pipeline_tag`` is the
# reliable signal on SiMa precompiled repos (VLMs are ``image-text-to-text``,
# multimodal Gemma-4 builds are ``any-to-any``); repo-name hints are only a
# fallback for repos with no metadata at all.
_VLM_PIPELINES = {"image-text-to-text", "visual-question-answering",
                  "video-text-to-text", "image-to-text", "any-to-any"}
_ASR_PIPELINES = {"automatic-speech-recognition"}
_VLM_NAME_HINTS = ("vlm", "-vl", "vl-", "_vl", "vision", "multimodal", "llava",
                   "paligemma", "internvl", "pixtral", "molmo", "minicpm-v",
                   "idefics", "siglip")


def classify_hub_repo(repo_id: str, pipeline_tag=None, tags=None) -> str:
    """Best-effort repo modality: ``"llm"`` | ``"vlm"`` | ``"asr"``."""
    tag = (pipeline_tag or "").strip().lower()
    tagset = {str(t).strip().lower() for t in (tags or [])}
    if tag in _ASR_PIPELINES or tagset & _ASR_PIPELINES:
        return "asr"
    if tag in _VLM_PIPELINES or tagset & _VLM_PIPELINES:
        return "vlm"
    if tag == "text-generation" or "text-generation" in tagset:
        return "llm"
    name = (repo_id.rsplit("/", 1)[-1] or "").lower()
    if "whisper" in name:
        return "asr"
    if any(hint in name for hint in _VLM_NAME_HINTS):
        return "vlm"
    return "llm"


def hub_search(catalog_dir: Path | None, hub: HubConfig, query: str, limit: int = 200) -> dict:
    if not hub_enabled(hub):
        return {"enabled": False, "results": []}
    try:
        from huggingface_hub import HfApi
    except Exception:
        return {"enabled": False, "results": []}

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    orgs = hub.orgs or ("simaai",)
    seen: set[str] = set()
    results = []
    errors = []
    for author in orgs:
        try:
            kwargs = dict(
                author=author or None,
                search=(query or None),
                sort="downloads",
                direction=-1,
                limit=limit,
            )
            # Ask for the metadata we render (type badge, size, popularity) in
            # the same call. The Hub API's accepted ``expand`` fields change
            # over time (``usedStorage`` was retired server-side), so fall back
            # progressively rather than losing the whole org listing.
            models = None
            last_exc: Exception | None = None
            for expand in (
                ["pipeline_tag", "tags", "downloads", "likes", "siblings", "usedStorage"],
                ["pipeline_tag", "tags", "downloads", "likes", "siblings"],
                ["pipeline_tag", "tags", "downloads", "likes"],
                None,
            ):
                try:
                    if expand is None:
                        models = list(api.list_models(**kwargs))
                    else:
                        models = list(api.list_models(expand=expand, **kwargs))
                    break
                except Exception as exc:  # noqa: BLE001 - try the next shape
                    last_exc = exc
            if models is None:
                raise last_exc if last_exc else RuntimeError("model listing failed")
            for m in models:
                repo_id = getattr(m, "id", None) or getattr(m, "modelId", "")
                if not repo_id or repo_id in seen:
                    continue
                seen.add(repo_id)
                # Only precompiled repos (with a devkit/ folder) can run on the
                # board. The org's *-Safetensors repos are quantized compile
                # sources — downloading one wastes gigabytes and then fails
                # classification — so keep them out of the catalog. Fall back
                # to the naming convention when file lists weren't returned.
                siblings = getattr(m, "siblings", None)
                if siblings:
                    if not any((getattr(s, "rfilename", "") or "").startswith("devkit/")
                               for s in siblings):
                        continue
                elif "safetensors" in repo_id.rsplit("/", 1)[-1].lower():
                    continue
                try:
                    local = _existing_catalog_target(catalog_dir, repo_id) if catalog_dir else None
                except ValueError:
                    logging.warning("Ignoring unsafe Hub repository id %r", repo_id)
                    continue
                in_catalog = bool(local and local.is_dir())
                results.append({
                    "repoId": repo_id,
                    "org": author,
                    "type": classify_hub_repo(repo_id, getattr(m, "pipeline_tag", None),
                                              getattr(m, "tags", None)),
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "sizeBytes": getattr(m, "used_storage", None) or getattr(m, "usedStorage", None),
                    "alreadyInCatalog": in_catalog,
                    "catalogComplete": (model_dir_complete(local)[0] if in_catalog else None),
                })
        except Exception:  # noqa: BLE001 - one organization must not hide the others
            logging.exception("Hugging Face search failed for organization %s", author)
            errors.append(str(author))

    # Most-downloaded first across all orgs.
    results.sort(key=lambda r: (r.get("downloads") or 0), reverse=True)
    out = {"enabled": True, "results": results[:limit]}
    if errors and not results:
        out["error"] = "Hugging Face search failed"
    return out


def hub_card(hub: HubConfig, repo_id: str) -> dict:
    """Metadata + model card (README) for a Hugging Face repo."""
    try:
        repo_id = validated_repo_id(repo_id, hub)
    except ValueError:
        return {"enabled": False, "error": "Invalid or disallowed repository id"}
    if not hub_enabled(hub):
        return {"enabled": False, "error": "Hugging Face is offline or disabled"}
    try:
        from huggingface_hub import HfApi
    except Exception:
        return {"enabled": False, "error": "huggingface_hub is not installed"}

    # Parse params/quant without importing the manager (avoid a cycle).
    from server.model_manager import parse_param_count, parse_quantization

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    size = None
    downloads = likes = None
    tags = []
    try:
        info = api.model_info(repo_id, files_metadata=True)
        size = sum(int(getattr(s, "size", 0) or 0) for s in (info.siblings or [])) or None
        downloads = getattr(info, "downloads", None)
        likes = getattr(info, "likes", None)
        tags = list(getattr(info, "tags", []) or [])
    except Exception:  # noqa: BLE001
        logging.exception("Hugging Face model metadata lookup failed")
        return {"enabled": True, "error": "Could not load model metadata"}

    card = None
    try:
        from huggingface_hub import ModelCard
        card = ModelCard.load(repo_id, token=token).text[:40000]
    except Exception:
        card = None

    short = repo_id.split("/")[-1]
    return {
        "enabled": True,
        "repoId": repo_id,
        "sizeBytes": size,
        "downloads": downloads,
        "likes": likes,
        "tags": tags[:20],
        "params": parse_param_count(short),
        "quantization": parse_quantization(short),
        "card": card,
        "source": "huggingface",
    }


def hub_download_stream(catalog_dir: Path | None, hub: HubConfig, repo_id: str) -> Iterator[str]:
    """Download a repo into the catalog, yielding JSON progress lines."""
    def event(**payload) -> str:
        return json.dumps(payload) + "\n"

    try:
        repo_id = validated_repo_id(repo_id, hub)
    except ValueError:
        yield event(state="error", message="Invalid or disallowed repository id")
        return
    if not hub_enabled(hub):
        yield event(state="error", message="Hugging Face downloads are disabled or offline")
        return
    if not catalog_dir:
        yield event(state="error", message="No catalog_dir configured")
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except Exception:
        yield event(state="error", message="huggingface_hub is not installed")
        return

    token = os.environ.get("HF_TOKEN")
    yield event(state="resolving", repoId=repo_id)

    # Resolve the server's canonical repository id before deriving a local
    # path.  The request value is validated above, but it is deliberately not
    # used in a filesystem expression: the path is based on Hub metadata and
    # is then confined to the configured catalog root by _catalog_target().
    try:
        info = HfApi(token=token).model_info(repo_id, files_metadata=True)
        canonical_repo_id = validated_repo_id(str(getattr(info, "id", "")), hub)
        if canonical_repo_id.casefold() != repo_id.casefold():
            raise ValueError("Hub returned a different repository id")
        target = _existing_catalog_target(Path(catalog_dir), canonical_repo_id)
        target_existed = target.exists()
        total = sum(int(getattr(s, "size", 0) or 0) for s in (info.siblings or []))
    except Exception:  # noqa: BLE001 - details stay in server logs
        logging.exception("Could not resolve Hugging Face repository %s", repo_id)
        yield event(state="error", repoId=repo_id, message="Could not resolve model repository")
        return

    error: dict = {}

    def _download() -> None:
        try:
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=canonical_repo_id, local_dir=str(target), token=token)
        except Exception:  # noqa: BLE001 - reported generically; details stay in logs
            logging.exception("Hugging Face model download failed for %s", repo_id)
            error["message"] = "Model download failed"

    worker = threading.Thread(target=_download, daemon=True)
    worker.start()

    prev_dl = _dir_size(target)
    prev_t = time.monotonic()
    avg_speed = 0.0  # bytes/sec, exponential moving average
    while worker.is_alive():
        downloaded = _dir_size(target)
        now = time.monotonic()
        dt = now - prev_t
        if dt > 0:
            inst = max(0.0, (downloaded - prev_dl) / dt)
            avg_speed = inst if avg_speed == 0 else 0.6 * avg_speed + 0.4 * inst
        prev_dl, prev_t = downloaded, now
        pct = int(min(99, downloaded * 100 / total)) if total else None
        eta = int((total - downloaded) / avg_speed) if (avg_speed > 0 and total and downloaded < total) else None
        yield event(state="downloading", repoId=repo_id, downloaded=downloaded,
                    total=total, pct=pct, speedBps=int(avg_speed), etaS=eta)
        time.sleep(1.0)
    worker.join()

    if error:
        yield event(state="error", repoId=repo_id, message=error["message"])
        return

    info = classify_model_dir(target)
    if info is None:
        if not target_existed:
            shutil.rmtree(target, ignore_errors=True)
        yield event(state="error", repoId=repo_id,
                    message="Downloaded repo is not a compatible model (no devkit/ config)")
        return

    # Some repos ship a broken chat_template.jinja (an HTML page) or the new
    # root-level template layout the runtime doesn't look at — fix that now so
    # the first load doesn't fail inside minja.
    try:
        for fix in repair_chat_template_files(target):
            yield event(state="repairing", repoId=repo_id, message=fix)
    except Exception:  # noqa: BLE001 - repair is best-effort
        pass

    # The download succeeded, so any leftover hf temp files are orphans from
    # earlier interrupted attempts (partials are keyed by etag, so a retry
    # after the repo changed upstream strands them). Sweep them so the model
    # never reads as incomplete and the dead bytes don't count against disk.
    try:
        for leftover in (target / ".cache" / "huggingface").rglob("*.incomplete"):
            try:
                leftover.unlink()
            except OSError:
                pass
    except OSError:
        pass

    # Mark the download complete so completeness checks trust it definitively.
    try:
        (target / _REPO_ID_MARKER).write_text(canonical_repo_id + "\n", encoding="utf-8")
        (target / ".neat-complete").write_text("ok\n")
    except OSError:
        pass

    yield event(state="done", repoId=repo_id, name=info["name"], total=total, pct=100)
