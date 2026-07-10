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
import os
import socket
import threading
import time
from pathlib import Path
from typing import Iterator

from shared.config import HubConfig, classify_model_dir, model_dir_complete


_internet_cache: dict = {"ok": None, "at": 0.0, "refreshing": False}
_internet_lock = threading.Lock()


def safe_name(repo_id: str) -> str:
    """Local catalog directory name for a Hub repo id (``org/name`` -> ``name``)."""
    return repo_id.rstrip("/").split("/")[-1]


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
        # Never probed — do it once synchronously so the first hubEnabled value
        # is accurate (this runs on the initial status poll, not repeatedly).
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
            # Ask for repo storage size in the same call when supported; fall
            # back gracefully on older huggingface_hub versions.
            try:
                models = list(api.list_models(expand=["usedStorage"], **kwargs))
            except (TypeError, ValueError):
                models = list(api.list_models(**kwargs))
            for m in models:
                repo_id = getattr(m, "id", None) or getattr(m, "modelId", "")
                if not repo_id or repo_id in seen:
                    continue
                seen.add(repo_id)
                local = catalog_dir / safe_name(repo_id) if catalog_dir else None
                in_catalog = bool(local and local.is_dir())
                results.append({
                    "repoId": repo_id,
                    "org": author,
                    "downloads": getattr(m, "downloads", None),
                    "likes": getattr(m, "likes", None),
                    "sizeBytes": getattr(m, "used_storage", None) or getattr(m, "usedStorage", None),
                    "alreadyInCatalog": in_catalog,
                    "catalogComplete": (model_dir_complete(local)[0] if in_catalog else None),
                })
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            errors.append(f"{author}: {exc}")

    # Most-downloaded first across all orgs.
    results.sort(key=lambda r: (r.get("downloads") or 0), reverse=True)
    out = {"enabled": True, "results": results[:limit]}
    if errors and not results:
        out["error"] = "; ".join(errors)
    return out


def hub_card(hub: HubConfig, repo_id: str) -> dict:
    """Metadata + model card (README) for a Hugging Face repo."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return {"enabled": False, "error": "repo_id is required"}
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
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "error": str(exc)}

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

    repo_id = (repo_id or "").strip()
    if not repo_id:
        yield event(state="error", message="repo_id is required")
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
    target = Path(catalog_dir) / safe_name(repo_id)
    yield event(state="resolving", repoId=repo_id)

    total = 0
    try:
        info = HfApi(token=token).model_info(repo_id, files_metadata=True)
        total = sum(int(getattr(s, "size", 0) or 0) for s in (info.siblings or []))
    except Exception:
        total = 0

    error: dict = {}

    def _download() -> None:
        try:
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=repo_id, local_dir=str(target), token=token)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            error["message"] = str(exc)

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
        yield event(state="error", repoId=repo_id,
                    message="Downloaded repo is not a compatible model (no devkit/ config)")
        return

    # Mark the download complete so completeness checks trust it definitively.
    try:
        (target / ".neat-complete").write_text("ok\n")
    except OSError:
        pass

    yield event(state="done", repoId=repo_id, name=info["name"], total=total, pct=100)
