#!/usr/bin/env python3
"""Runtime model management for Neat GenAI Studio.

Wraps a live ``pyneat.GenAIServer`` so chat/VLM models can be loaded and
unloaded on the fly (no restart), scans an on-disk catalog of compatible
models, keeps a bounded set resident in RAM (LRU eviction), and can download
additional compatible models from the Hugging Face Hub when the board is
online.

pyneat's ``add_model`` / ``remove_model`` are thread-safe and may be called
after ``server.start()``; this class serializes catalog mutations under a lock.
ASR (speech-to-text) models get their own slot: exactly one is resident at a
time and switching to another evicts the previous one, so an ASR switch never
disturbs the resident chat/VLM model (and vice versa).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from shared.chat_template import repair_chat_template_files
from shared.config import HubConfig, classify_model_dir, model_dir_complete


def parse_param_count(name: str) -> str | None:
    """Extract a parameter count from a model name, e.g. '4B', '1.5B', '500M'."""
    text = name or ""
    for wanted_suffix in ("b", "m"):
        i = 0
        while i < len(text):
            if not text[i].isdigit():
                i += 1
                continue
            start = i
            while i < len(text) and text[i].isdigit():
                i += 1
            if i < len(text) and text[i] == ".":
                decimal = i
                i += 1
                fraction_start = i
                while i < len(text) and text[i].isdigit():
                    i += 1
                if i == fraction_start:
                    i = decimal
            number_end = i
            while i < len(text) and text[i].isspace():
                i += 1
            if i < len(text) and text[i].lower() == wanted_suffix:
                after = i + 1
                if after == len(text) or not text[after].isascii() or not text[after].isalpha():
                    return f"{text[start:number_end]}{wanted_suffix.upper()}"
            i = max(i, start + 1)
    return None


def parse_quantization(name: str) -> str | None:
    """Human-readable quantization scheme from a model name."""
    text = name or ""
    m = re.search(r"[aA](\d+)[wW](\d+)", text)
    if m:
        return f"a{m.group(1)}w{m.group(2)} · activations {m.group(1)}-bit, weights {m.group(2)}-bit"
    for scheme in ("GPTQ", "AWQ", "GGUF", "int8", "int4", "fp16", "bf16", "fp8"):
        if scheme.lower() in text.lower():
            return scheme
    return None

# Substrings that identify an MLA (accelerator) load failure in an error message.
# Whisper builds are ~1.5 GB at most — an order of magnitude below a 7B VLM —
# so they do not need the chat warm-up's 600s ceiling.
_ASR_WARM_TIMEOUT_S = 300

# Seed for the load-time estimate, in seconds per GB of ELF stages. Measured on a
# Modalix DevKit, where the observed rate spans roughly 1.2-2.7 s/GB depending on
# how a model's stages are structured. Deliberately seeded at the SLOW end: an
# over-estimate makes the bar advance conservatively and finish early, whereas an
# under-estimate pins it at 99% and looks hung. Only used until the board has
# timed that model once — _sec_per_gb and the per-model history then take over
# and are accurate (a re-load of a 7 GB model estimated 19 s and took 19.8 s).
# Without any seed the FIRST load of a session has no bar and no countdown at
# all, which is exactly when a user is most likely to be watching.
_DEFAULT_SEC_PER_GB = 2.7

_MLA_FAILURE_MARKERS = (
    "mlashm",
    "mla_load",
    "mla load",
    "bulk load",
    "dispatcher is unavailable",
    "dispatcher_unavailable",
    "warmup failed",
    "failed to acquire",
)


def _is_mla_failure(detail: str) -> bool:
    text = (detail or "").lower()
    return any(marker in text for marker in _MLA_FAILURE_MARKERS)


class ModelManager:
    """Loads/unloads models on a running GenAIServer and tracks the catalog."""

    def __init__(
        self,
        server,
        *,
        catalog_dir: Path | None,
        max_resident_chat_models: int,
        asr_name: str | None,
        hub: HubConfig,
        openai_base_url: str,
        warmup: bool = True,
        asr_warmup: bool = True,
        mla_reset_exit_code: int = 75,
        switch_settle_s: float = 0.6,
        log_tap=None,
    ) -> None:
        self._server = server
        self._catalog_dir = Path(catalog_dir) if catalog_dir else None
        self._max_resident = max(1, int(max_resident_chat_models))
        # What config pins at startup (immutable) vs what serves transcriptions
        # right now (mutable — the UI can switch it). The active name is only a
        # cache: eviction victims are DERIVED from the server's loaded set, so a
        # stale pointer can never strand a resident ASR model.
        self._configured_asr = asr_name
        self._active_asr = asr_name
        self._hub = hub
        self._openai_base_url = openai_base_url.rstrip("/")
        self._warmup = warmup
        self._asr_warmup = asr_warmup
        # Sentinel exit code that asks the supervisor (run.sh) to reset the MLA
        # dispatcher and relaunch. Only ever used for an explicit user request.
        self._mla_reset_exit_code = int(mla_reset_exit_code)
        # After unloading the outgoing model, wait briefly so its RAII free
        # (which returns MLA memory to the dispatcher) completes before loading
        # the replacement — avoids transient double-residency (MLA_LOAD_FAILED).
        self._switch_settle_s = max(0.0, float(switch_settle_s))
        self._lock = threading.RLock()
        # Serializes mutating operations (load/unload/delete) end-to-end WITHOUT
        # holding _lock across the slow accelerator/IO calls (remove_model's MLA
        # free, the /stop call, the settle sleep, rmtree). _lock is then only ever
        # held for microseconds, so status/catalog reads — which the UI polls
        # every ~0.9s during a load — never block behind a switch.
        self._op_lock = threading.Lock()
        # Chat/VLM models currently resident, ordered most-recently-used first.
        self._resident: list[str] = []
        # Catalog by served name -> classification dict (name, path, type, ...).
        self._catalog: dict[str, dict] = {}
        # On-disk weight size per model dir (cached; dirs are static once present).
        self._size_cache: dict[str, int] = {}
        # Weight-completeness per model dir (cached; cleared on each rescan).
        self._complete_cache: dict = {}
        # Live load progress (for the UI) — the MLA gives no true progress signal,
        # so we expose elapsed time + a learned ETA from weight size / past loads.
        self._loading: dict | None = None
        self._load_history: dict[str, float] = {}  # name -> last load seconds
        self._sec_per_gb: float | None = _DEFAULT_SEC_PER_GB   # learned rate for ETA
        # Optional stdout tap: real per-ELF load progress + a live loading log.
        self._log_tap = log_tap
        # Last load failure, retained so the UI can surface it prominently.
        self._last_error: dict | None = None
        # Performance benchmark (web MoLE `perf`): TTFT/TPS over N passes.
        self._bench: dict | None = None
        self._bench_lock = threading.Lock()

        self.scan_catalog()
        # Only claim the configured ASR is active if the server really loaded it.
        # main.py skips one whose directory is missing — which is exactly what a
        # user sees after switching away from the startup default and deleting
        # it — and advertising an unregistered model sends transcriptions to a
        # name the server does not serve.
        if self._active_asr and self._active_asr not in self._server_model_names():
            logging.info(
                "configured ASR model %r is not loaded; starting with none active",
                self._active_asr,
            )
            self._active_asr = None
        self._sync_resident_from_server()

    # -- catalog ---------------------------------------------------------------

    def scan_catalog(self) -> list[dict]:
        """Rescan ``catalog_dir`` for compatible models and refresh the index."""
        self._complete_cache = {}   # re-check completeness on an explicit rescan
        catalog: dict[str, dict] = {}
        if self._catalog_dir and self._catalog_dir.is_dir():
            for child in sorted(self._catalog_dir.iterdir()):
                info = classify_model_dir(child)
                if info is not None:
                    catalog[info["name"]] = info
        with self._lock:
            # Preserve entries for already-loaded models even if not under
            # catalog_dir (e.g. startup models configured with absolute paths).
            for name in self._server_model_names():
                if name not in catalog and name in self._catalog:
                    catalog[name] = self._catalog[name]
            self._catalog = catalog
        return self.catalog()

    def register_startup_model(self, name: str, path: Path | None, model_type: str,
                               supports_vision: bool, image_size: dict | None) -> None:
        """Record a model loaded at startup so it appears in the catalog."""
        with self._lock:
            self._catalog[name] = {
                "name": name,
                "path": path,
                "type": model_type,
                "supports_vision": supports_vision,
                "image_size": image_size,
            }

    def _size_of(self, path) -> int | None:
        """On-disk size (bytes) of a model directory, cached by path."""
        if not path:
            return None
        key = str(path)
        cached = self._size_cache.get(key)
        if cached is not None:
            return cached
        total = 0
        try:
            for root, _dirs, files in os.walk(key):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
        except OSError:
            return None
        self._size_cache[key] = total
        return total

    def _is_model_complete(self, path) -> tuple[bool, str]:
        """Cached weight-completeness check for a model dir (cleared on rescan)."""
        key = str(path) if path else ""
        cached = self._complete_cache.get(key)
        if cached is not None:
            return cached
        result = model_dir_complete(path)
        self._complete_cache[key] = result
        return result

    def catalog(self) -> list[dict]:
        loaded = set(self._server_model_names())
        active_asr = self._active_asr
        with self._lock:
            infos = list(self._catalog.values())   # snapshot; then walk the FS
        entries = []                                # outside the lock (walks are slow)
        for info in infos:
            complete, reason = self._is_model_complete(info.get("path"))
            path = info.get("path")
            entries.append({
                "name": info["name"],
                "type": info.get("type", "chat"),
                "supportsVision": bool(info.get("supports_vision")),
                "imageSize": info.get("image_size"),
                "loaded": info["name"] in loaded,
                # "pinned" = the ASR model config re-selects on restart;
                # "activeAsr" = the one serving transcriptions right now.
                "pinned": info["name"] == self._configured_asr,
                "activeAsr": (info.get("type") == "asr"
                              and info["name"] == active_asr),
                "sizeBytes": self._size_of(path),
                # How long this model is expected to take to load, so the client
                # can run its own countdown. add_model() does the bulk MLA load
                # in native code without releasing the GIL, which freezes this
                # whole process — control API included — for the duration, so
                # nothing can be polled while a load is actually in flight.
                "estimatedLoadS": self._estimate_load_seconds(
                    info["name"], self._elf_bytes(path) or self._size_of(path)),
                "stagesTotal": self._count_elf_stages(path),
                "complete": complete,
                "incompleteReason": reason or None,
            })
        entries.sort(key=lambda e: (e["type"] == "asr", e["name"].lower()))
        return entries

    # -- resident set ----------------------------------------------------------

    def close(self) -> None:
        """Drop the reference to the GenAIServer so it can be destructed at
        shutdown (its C++ destructor releases the MLA models)."""
        with self._lock:
            self._server = None
            self._resident = []

    def _server_model_names(self) -> list[str]:
        server = self._server
        if server is None:
            return []
        try:
            return list(server.model_names())
        except Exception:
            return []

    def _catalog_type(self, name: str) -> str:
        with self._lock:
            info = self._catalog.get(name)
        return (info or {}).get("type", "chat")

    def _loaded_asr_names(self) -> list[str]:
        """ASR models currently registered on the server.

        Derived from the catalog type rather than from ``_active_asr`` so a
        stale pointer can never leave an ASR model resident forever.
        """
        loaded = self._server_model_names()
        names = [n for n in loaded if self._catalog_type(n) == "asr"]
        active = self._active_asr
        if active and active in loaded and active not in names:
            names.append(active)
        return names

    def active_asr(self) -> str | None:
        """The ASR model currently serving transcriptions."""
        return self._active_asr

    def _sync_resident_from_server(self) -> None:
        # Exclude ASR models by TYPE, not by name: after a switch the active ASR
        # has a different name than the configured one, and letting it leak into
        # _resident would make the next chat load evict it.
        loaded = self._server_model_names()
        asr = set(self._loaded_asr_names())
        with self._lock:
            self._resident = [
                n for n in self._resident if n in loaded and n not in asr
            ]
            for name in loaded:
                if name not in asr and name not in self._resident:
                    self._resident.append(name)

    def touch(self, name: str) -> None:
        """Mark a resident model most-recently-used."""
        with self._lock:
            if name in self._resident:
                self._resident.remove(name)
                self._resident.insert(0, name)

    # -- load / unload ---------------------------------------------------------

    def load(self, name: str) -> dict:
        """Load a model, clearing every other chat/VLM model first.

        Only one chat/VLM model is kept resident at a time. Switching evicts the
        others cleanly (cancel their in-flight streams, then unload, then wait so
        the MLA memory is actually returned before the new model loads), then
        warms the new model synchronously so an MLA load failure is caught here
        and surfaced during the load rather than on the user's first chat.
        """
        name = (name or "").strip()
        with self._op_lock:
            if name not in self._catalog:
                self.scan_catalog()
            with self._lock:
                info = self._catalog.get(name)
                if info is None:
                    raise ValueError(f"Unknown model: {name}")
                path = info.get("path")
                is_asr = info.get("type") == "asr"

            if name in self._server_model_names():
                if not is_asr:
                    self.touch(name)
                    return {"name": name, "state": "ready", "evicted": [],
                            "cold_start": False, "load_seconds": 0.0}
                # An ASR model can be registered yet NOT active: a warm-up that
                # failed leaves the registration behind when remove_model also
                # fails. Declaring it active on that basis would report ready
                # without ever re-checking, and the next transcription would fail
                # the same way — so prove it works before adopting it.
                if self._active_asr == name or not (self._warmup and self._asr_warmup):
                    with self._lock:
                        self._active_asr = name
                    return {"name": name, "state": "ready", "evicted": [],
                            "cold_start": False, "load_seconds": 0.0}
                probe_started = time.monotonic()
                self._log_note(f"Re-checking {name} (already registered)…")
                ok, detail = self._warm_check_asr(name)
                if not ok and _is_mla_failure(detail):
                    with self._lock:
                        self._active_asr = None
                    return self._handle_mla_failure(name, detail)
                with self._lock:
                    self._active_asr = name
                result = {"name": name, "state": "ready", "evicted": [],
                          "cold_start": False,
                          "load_seconds": round(time.monotonic() - probe_started, 1)}
                if not ok:
                    logging.warning("ASR re-check for '%s' did not complete: %s",
                                    name, detail)
                    result["warm_warning"] = detail[:500]
                return result

            if not path or not Path(path).is_dir():
                raise ValueError(f"Model directory not found for: {name}")

            # Refuse to load a model whose weights are missing/partial — otherwise
            # it fails deep in the MLA with a confusing error.
            complete, reason = self._is_model_complete(path)
            if not complete:
                raise ValueError(
                    f"'{name}' cannot be loaded — {reason}. Re-download it from Add Model."
                )

            # Self-heal chat-template files (corrupt HTML pages, new root-level
            # Hub layout) before the runtime's minja parser ever sees them —
            # this also fixes models downloaded before the repair existed.
            try:
                for fix in repair_chat_template_files(path):
                    logging.info("model '%s': %s", name, fix)
            except Exception:  # noqa: BLE001 - repair is best-effort
                pass

            # ASR models have their own slot: switching evicts the previous ASR
            # and leaves the resident chat/VLM model alone (and vice versa).
            if is_asr:
                victims = [v for v in self._loaded_asr_names() if v != name]
            else:
                with self._lock:
                    victims = [v for v in self._resident if v != name]

            size_bytes = self._size_of(path)
            # Estimate and learn against the ELF bytes actually transferred, so
            # the rate stays comparable across models whose directories carry
            # very different amounts of non-ELF baggage.
            load_bytes = self._elf_bytes(path) or size_bytes
            started = time.monotonic()
            self._last_error = None
            self._loading = {
                "name": name,
                "startedAt": started,
                "estTotalS": self._estimate_load_seconds(name, load_bytes),
                # Real progress: count completed ELF-stage loads (from the
                # stdout tap) minus the baseline against the on-disk total.
                "elfTotal": self._count_elf_stages(path),
                "elfBase": self._log_tap.loaded_count if self._log_tap else 0,
            }
            evicted: list = []
            stages = self._loading.get("elfTotal")
            self._log_note(
                f"Loading {name} — {self._fmt_size(size_bytes)}"
                + (f", {stages} stages" if stages else "")
                + (f", est {self._loading['estTotalS']:.0f}s" if self._loading.get("estTotalS") else "")
            )
            try:
                # Clear every other chat/VLM model. _stop_model_streams (an HTTP
                # /stop) and remove_model (which triggers the outgoing model's RAII
                # free of MLA memory) can each block for seconds — do them WITHOUT
                # holding _lock so status polls stay responsive during the switch.
                for victim in victims:
                    self._log_note(f"Unloading {victim}")
                    self._stop_model_streams(victim)
                    try:
                        self._server.remove_model(victim)
                        evicted.append(victim)
                    except Exception as exc:
                        if not is_asr:
                            continue
                        # There is only one ASR slot. If the outgoing model will
                        # not free, adding the replacement leaves two resident —
                        # which can fail the load outright on double residency —
                        # and the state below would report no active ASR while
                        # the old one is still serving. Abort with it untouched.
                        raise RuntimeError(
                            f"Could not unload the current speech-to-text model "
                            f"'{victim}': {exc}"
                        ) from exc
                with self._lock:
                    if is_asr:
                        # Honest during the eviction window: nothing can serve a
                        # transcription until the replacement is registered.
                        self._active_asr = None
                    else:
                        self._resident = []
                # Let the free complete before loading the replacement, so the
                # old and new model are never briefly co-resident.
                if evicted and self._switch_settle_s:
                    time.sleep(self._switch_settle_s)

                # add_model returns the name the server actually served it under,
                # which may differ from the requested one — that is the truth.
                self._log_note(f"Registering {name} with the runtime…")
                served = self._server.add_model(str(path), name)
                self._log_note(
                    f"Registered {served}; transferring weights to the accelerator…"
                )
                with self._lock:
                    if is_asr:
                        self._active_asr = served
                    else:
                        self._resident = [served]

                # add_model only registers; the real MLA load is deferred to first
                # inference. Warm synchronously so a load failure is catchable and
                # so we can time the load (this is where the wait actually happens).
                if is_asr:
                    if not (self._warmup and self._asr_warmup):
                        return {"name": served, "state": "ready", "evicted": evicted,
                                "cold_start": True,
                                "load_seconds": round(time.monotonic() - started, 1)}
                    self._log_note("Warming up (first transcription forces the load)…")
                    ok, detail = self._warm_check_asr(served)
                    if ok:
                        self._record_load_duration(name, load_bytes,
                                                   time.monotonic() - started)
                        self._log_note(f"Ready: {served} in {time.monotonic() - started:.1f}s")
                        return {"name": served, "state": "ready", "evicted": evicted,
                                "cold_start": True,
                                "load_seconds": round(time.monotonic() - started, 1)}
                    if _is_mla_failure(detail):
                        with self._lock:
                            self._active_asr = None
                        return self._handle_mla_failure(served, detail)
                    # Soft failure: unlike the chat probe, this one depends on the
                    # runtime's multipart handling and on how a given Whisper
                    # artifact reacts to pure silence. A non-MLA error here most
                    # likely means the model is fine, so keep it active — it will
                    # load on first use, exactly as it did before warming existed.
                    logging.warning("ASR warm-up for '%s' did not complete: %s",
                                    served, detail)
                    return {"name": served, "state": "ready", "evicted": evicted,
                            "cold_start": True,
                            "load_seconds": round(time.monotonic() - started, 1),
                            "warm_warning": detail[:500]}

                if not self._warmup:
                    return {"name": served, "state": "ready", "evicted": evicted,
                            "cold_start": True, "load_seconds": round(time.monotonic() - started, 1)}

                self._log_note("Warming up (first inference forces the load)…")
                ok, detail = self._warm_check(name)
                if ok:
                    self._record_load_duration(name, load_bytes, time.monotonic() - started)
                    self._log_note(f"Ready: {served} in {time.monotonic() - started:.1f}s")
                    return {"name": served, "state": "ready", "evicted": evicted,
                            "cold_start": True, "load_seconds": round(time.monotonic() - started, 1)}

                if _is_mla_failure(detail):
                    return self._handle_mla_failure(name, detail)

                # Non-MLA warm failure: roll back so manager state stays consistent.
                try:
                    self._server.remove_model(name)
                except Exception:
                    pass
                with self._lock:
                    self._resident = []
                self._record_error(name, detail, kind="load")
                raise RuntimeError(f"Model '{name}' failed to load: {detail}")
            finally:
                self._loading = None

    def set_active_asr(self, name: str) -> dict:
        """Make ``name`` the ASR model that serves transcriptions.

        Only one ASR model is resident at a time, so this evicts the previous
        one. The resident chat/VLM model is untouched.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("An ASR model name is required")
        if name not in self._catalog:
            self.scan_catalog()
        with self._lock:
            info = self._catalog.get(name)
        if info is None:
            raise ValueError(f"Unknown model: {name}")
        # Refuse anything that is not speech-to-text, rather than quietly
        # pointing transcription at a chat model that cannot serve it.
        if info.get("type") != "asr":
            raise ValueError(f"'{name}' is not a speech-to-text (ASR) model")

        previous = self._active_asr
        result = self.load(name)
        active = self._active_asr
        result["activeAsr"] = active
        result["previous"] = previous if previous != active else None
        return result

    def _log_note(self, text: str) -> None:
        """Add a studio-side line to the load log (no-op without a tap)."""
        if self._log_tap is not None:
            try:
                self._log_tap.note(text)
            except Exception:
                pass

    @staticmethod
    def _fmt_size(size_bytes: int | None) -> str:
        if not size_bytes:
            return "size unknown"
        gb = size_bytes / 1e9
        return f"{gb:.1f} GB" if gb >= 1 else f"{size_bytes / 1e6:.0f} MB"

    def _elf_bytes(self, path) -> int | None:
        """Total size of the ELF stages — the bytes the accelerator actually
        loads. A model directory can be twice this (safetensors, tokenizer,
        configs), so estimating from the directory size overshoots badly."""
        if not path:
            return None
        try:
            total = sum(f.stat().st_size
                        for f in Path(path).glob("elf_files/*_mla.elf"))
        except Exception:
            return None
        return total or None

    def _estimate_load_seconds(self, name: str, size_bytes: int | None) -> float | None:
        """Best-effort ETA: the model's own last load time, else a learned rate."""
        if name in self._load_history:
            return self._load_history[name]
        if self._sec_per_gb and size_bytes:
            return (size_bytes / 1e9) * self._sec_per_gb
        return None

    def _record_load_duration(self, name: str, size_bytes: int | None, seconds: float) -> None:
        if seconds <= 0:
            return
        self._load_history[name] = seconds
        if size_bytes and size_bytes > 0:
            rate = seconds / (size_bytes / 1e9)
            self._sec_per_gb = rate if self._sec_per_gb is None else 0.5 * self._sec_per_gb + 0.5 * rate

    def loading_status(self) -> dict | None:
        """Live progress for an in-flight load, or None.

        Prefers REAL progress — completed ELF stages / total ELFs on disk (from
        the stdout tap) — and derives the ETA from the observed per-stage rate.
        Falls back to an elapsed/size-based ESTIMATE when the tap is unavailable.
        """
        info = self._loading
        if not info:
            return None
        elapsed = max(0.0, time.monotonic() - info["startedAt"])

        elf_total = info.get("elfTotal")
        elf_done = None
        if self._log_tap and elf_total:
            elf_done = max(0, min(self._log_tap.loaded_count - info.get("elfBase", 0), elf_total))

        # Only trust the stage counter once it has actually moved. Current
        # runtimes load every stage in a single bulk call and print no per-stage
        # lines, so the counter sits at 0 for the whole load — and taking this
        # branch on the strength of "a tap exists and there are ELFs on disk"
        # pinned the bar to 0% and threw away the time-based estimate below.
        if elf_done and elf_total:
            pct = int(elf_done / elf_total * 100)
            if elf_done > 0:
                eta_total = elapsed / elf_done * elf_total   # observed per-stage rate
                estimated = False
            else:
                eta_total = info.get("estTotalS")
                estimated = True
            return {
                "name": info["name"],
                "elapsedS": round(elapsed, 1),
                "etaS": round(eta_total, 1) if eta_total else None,
                "remainingS": round(max(0.0, eta_total - elapsed), 1) if eta_total else None,
                "pct": pct,
                "filesDone": elf_done,
                "filesTotal": elf_total,
                "stagesTotal": elf_total,
                "estimated": estimated,
            }

        # Fallback: elapsed / learned-size estimate (no per-stage signal). Report
        # filesTotal=None too, so the UI shows elapsed time rather than a frozen
        # "stage 0 of N" against a total it can never make progress against.
        # No per-stage signal (current runtimes load every stage in one bulk call
        # and print nothing per stage). Drive the bar from elapsed/estimate, and
        # hold at 99 rather than 100 so it never claims to be finished early.
        est = info.get("estTotalS")
        pct = int(min(99, elapsed / est * 100)) if est and est > 0 else None
        return {
            "name": info["name"],
            "elapsedS": round(elapsed, 1),
            "etaS": round(est, 1) if est else None,
            "remainingS": round(max(0.0, est - elapsed), 1) if est else None,
            "pct": pct,
            "filesDone": None,
            "filesTotal": None,
            # The stage count is still worth showing as scale, even when the
            # runtime gives us no way to count them off.
            "stagesTotal": info.get("elfTotal"),
            "estimated": True,
        }

    def _count_elf_stages(self, path) -> int | None:
        """Number of ELF stages on disk for a model (the load-progress total)."""
        try:
            elf_dir = Path(path) / "elf_files"
            if not elf_dir.is_dir():
                return None
            n = sum(1 for f in elf_dir.iterdir() if f.name.endswith("_mla.elf"))
            return n or None
        except Exception:
            return None

    def _record_error(self, name: str, detail: str, kind: str) -> None:
        elapsed = None
        if self._loading and self._loading.get("name") == name:
            elapsed = round(max(0.0, time.monotonic() - self._loading["startedAt"]), 1)
        self._last_error = {
            "name": name,
            "error": (detail or "").strip(),
            "kind": kind,          # 'mla' | 'load'
            "elapsedS": elapsed,
        }

    def load_logs(self, after_seq: int = 0) -> dict:
        """Recent model-load log lines (from the stdout tap) + the last error."""
        if self._log_tap:
            snap = self._log_tap.snapshot(after_seq)
            snap["available"] = True
        else:
            snap = {"loadedCount": 0, "seq": 0, "lines": [], "available": False}
        snap["lastError"] = self._last_error
        snap["loading"] = self.loading_status()
        return snap

    # -- performance benchmark (web MoLE `perf`) -------------------------------

    _BENCH_PROMPT = (
        "Write a detailed, technical explanation of how a transformer-based "
        "language model generates text, covering tokenization, self-attention, "
        "and autoregressive decoding."
    )

    def benchmark_start(self, num_samples: int = 5, max_new_tokens: int = 128,
                        prompt: str = "", model: str = "") -> dict:
        """Start a background TTFT/TPS benchmark against the loaded model."""
        num_samples = max(1, min(50, int(num_samples or 5)))
        max_new_tokens = max(8, min(2048, int(max_new_tokens or 128)))
        prompt = (prompt or "").strip() or self._BENCH_PROMPT
        model = (model or "").strip()
        with self._lock:
            if not model:
                model = self._resident[0] if self._resident else ""
            loaded = model in self._server_model_names()
        if not model or not loaded:
            raise ValueError("No model is loaded to benchmark. Load a model first.")
        with self._bench_lock:
            if self._bench and self._bench.get("running"):
                raise ValueError("A benchmark is already running")
            self._bench = {
                "running": True, "cancel": False, "model": model,
                "total": num_samples, "maxNewTokens": max_new_tokens,
                "prompt": prompt, "runs": [], "summary": None, "current": None,
            }
        threading.Thread(target=self._benchmark_loop,
                         args=(model, prompt, max_new_tokens, num_samples),
                         name="benchmark", daemon=True).start()
        return self.benchmark_status()

    def _benchmark_loop(self, model, prompt, max_new_tokens, num_samples) -> None:
        for i in range(num_samples):
            with self._bench_lock:
                if not self._bench or self._bench.get("cancel"):
                    break
                self._bench["current"] = {"index": i, "tokens": 0, "ttftMs": None, "text": ""}
            try:
                result = self._bench_run_once(i, model, prompt, max_new_tokens)
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            with self._bench_lock:
                if not self._bench:
                    return
                self._bench["runs"].append(result)
                self._bench["current"] = None
        with self._bench_lock:
            if self._bench:
                self._bench["running"] = False
                self._bench["current"] = None
                self._bench["summary"] = self._benchmark_summary(self._bench["runs"])

    def _bench_run_once(self, index, model, prompt, max_new_tokens) -> dict:
        """One streamed generation. Prefers the runtime's own TTFT/TPS metrics
        (streamed by the server as `ttft`/`tps` fields — the same numbers the
        chat shows and the datasheet quotes, free of serving/client overhead) and
        falls back to client-side timing only if the server doesn't provide them.
        Updates the shared `current` state live for the UI."""
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "stream": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._openai_base_url}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.monotonic()
        srv_ttft = None      # seconds, runtime-measured
        srv_tps = None       # tokens/sec, runtime-measured
        t_first = None
        t_last = None
        tokens = 0
        parts: list[str] = []
        last_live = 0.0
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                # Runtime metrics streamed by the server (authoritative).
                if "ttft" in obj:
                    try: srv_ttft = float(obj["ttft"])
                    except (TypeError, ValueError): pass
                if "tps" in obj:
                    try: srv_tps = float(obj["tps"])
                    except (TypeError, ValueError): pass
                choices = obj.get("choices") or []
                delta = choices[0].get("delta", {}).get("content") if choices else None
                if not delta:
                    continue
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                tokens += 1
                parts.append(delta)
                # Throttle the live update so it never perturbs the timing.
                if now - last_live > 0.12:
                    last_live = now
                    text = "".join(parts)[-4000:]
                    ttft_ms = round((srv_ttft if srv_ttft is not None else (t_first - t0)) * 1000, 1)
                    with self._bench_lock:
                        cur = self._bench.get("current") if self._bench else None
                        if cur and cur.get("index") == index:
                            cur["tokens"] = tokens
                            cur["ttftMs"] = ttft_ms
                            cur["tps"] = round(srv_tps, 2) if srv_tps is not None else None
                            cur["text"] = text
        # Prefer runtime metrics; else derive from client-side arrival times.
        if srv_ttft is not None:
            ttft = srv_ttft
        else:
            ttft = (t_first - t0) if t_first is not None else 0.0
        if srv_tps is not None:
            tps = srv_tps
        else:
            gen = (t_last - t_first) if (t_first is not None and t_last is not None and t_last > t_first) else 0.0
            tps = ((tokens - 1) / gen) if gen > 0 else 0.0
        with self._bench_lock:
            cur = self._bench.get("current") if self._bench else None
            if cur and cur.get("index") == index:
                cur["tokens"] = tokens
                cur["text"] = "".join(parts)[-4000:]
        return {
            "ttftMs": round(ttft * 1000, 1), "tps": round(tps, 2), "tokens": tokens,
            "serverMetrics": srv_tps is not None, "text": "".join(parts)[:4000],
        }

    @staticmethod
    def _benchmark_summary(runs) -> dict | None:
        ok = [r for r in runs if "error" not in r and r.get("tokens")]
        if not ok:
            return None

        def _stats(vals):
            s = sorted(float(v) for v in vals)
            n = len(s)
            mid = n // 2
            median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
            mean = sum(s) / n
            # Sample standard deviation (Bessel's n-1); 0 with a single sample.
            stdev = (sum((v - mean) ** 2 for v in s) / (n - 1)) ** 0.5 if n > 1 else 0.0

            def _pct(p):
                if n == 1:
                    return s[0]
                rank = (p / 100.0) * (n - 1)   # linear interpolation between ranks
                lo = int(rank)
                hi = min(lo + 1, n - 1)
                return s[lo] + (s[hi] - s[lo]) * (rank - lo)

            return {
                "mean": round(mean, 2), "median": round(median, 2),
                "min": round(s[0], 2), "max": round(s[-1], 2),
                "stdev": round(stdev, 2), "cv": round(stdev / mean * 100, 1) if mean else 0.0,
                "p90": round(_pct(90), 2), "p95": round(_pct(95), 2),
            }

        return {
            "count": len(ok), "errors": len(runs) - len(ok),
            "ttftMs": _stats([r["ttftMs"] for r in ok]),
            "tps": _stats([r["tps"] for r in ok]),
            "tokens": _stats([r["tokens"] for r in ok]),
            "totalTokens": sum(int(r["tokens"]) for r in ok),
        }

    def benchmark_status(self) -> dict:
        with self._bench_lock:
            b = self._bench
            if not b:
                return {"running": False, "done": 0, "total": 0, "runs": [],
                        "summary": None, "prompt": "", "current": None}
            runs = list(b["runs"])
            return {
                "running": b["running"], "model": b["model"],
                "done": len(runs), "total": b["total"], "maxNewTokens": b["maxNewTokens"],
                "prompt": b.get("prompt", ""), "current": b.get("current"),
                "runs": runs, "summary": b.get("summary") or self._benchmark_summary(runs),
            }

    def benchmark_stop(self) -> dict:
        with self._bench_lock:
            if self._bench:
                self._bench["cancel"] = True
                model = self._bench.get("model", "")
            else:
                model = ""
        if model:
            self._stop_model_streams(model)
        return {"stopped": True}

    def _refuse_if_active_asr(self, name: str, verb: str) -> None:
        """Guard against touching the ASR model that is serving transcriptions.

        Must be called while holding ``_op_lock``: checking before taking it
        races a concurrent ``set_active_asr``, which could make ``name`` active
        while this operation waits, and the model would then be unloaded or
        deleted out from under transcription.
        """
        if name and name == self._active_asr:
            raise ValueError(
                f"The active speech-to-text model cannot be {verb} — "
                "switch to another ASR model first."
            )

    def unload(self, name: str) -> dict:
        name = (name or "").strip()
        with self._op_lock:
            self._refuse_if_active_asr(name, "unloaded")
            # remove_model frees MLA memory and can block for seconds — keep it
            # out of _lock so concurrent status polls are not held up.
            self._stop_model_streams(name)
            removed = bool(self._server.remove_model(name))
            with self._lock:
                if name in self._resident:
                    self._resident.remove(name)
        return {"name": name, "state": "unloaded" if removed else "absent"}

    def delete(self, name: str) -> dict:
        """Unload (if loaded) and delete a model's files from the catalog.

        Guarded: refuses to delete the active ASR model or anything outside
        ``catalog_dir``. An ASR model that is merely installed (not active) is
        deletable — that is how you reclaim the space of a model you switched
        away from.
        """
        name = (name or "").strip()
        if not self._catalog_dir:
            raise ValueError("No catalog_dir configured; refusing to delete")
        with self._op_lock:
            self._refuse_if_active_asr(name, "deleted")
            with self._lock:
                info = self._catalog.get(name)
            if info is None:
                self.scan_catalog()
                with self._lock:
                    info = self._catalog.get(name)
            if info is None:
                raise ValueError(f"Unknown model: {name}")
            raw_path = info.get("path")
            if not raw_path:
                raise ValueError(f"No path recorded for model: {name}")

            path = Path(raw_path).resolve()
            catdir = Path(self._catalog_dir).resolve()
            if path == catdir or catdir not in path.parents:
                raise ValueError(f"Refusing to delete a model outside the catalog: {path}")

            # Unload first so the MLA/registry no longer references it. The /stop,
            # remove_model (MLA free) and rmtree can all block for seconds — keep
            # them out of _lock so status polls are not held up.
            if name in self._server_model_names():
                self._stop_model_streams(name)
                try:
                    self._server.remove_model(name)
                except Exception:
                    pass
                with self._lock:
                    if name in self._resident:
                        self._resident.remove(name)

            shutil.rmtree(path, ignore_errors=True)
            with self._lock:
                self._size_cache.pop(str(raw_path), None)
                self._catalog.pop(name, None)

        self.scan_catalog()
        return {"name": name, "state": "deleted"}

    def model_card(self, name: str) -> dict:
        """Return metadata + the model card (README) for a catalog model."""
        name = (name or "").strip()
        info = self._catalog.get(name)
        if info is None:
            self.scan_catalog()
            info = self._catalog.get(name)
        if info is None:
            raise ValueError(f"Unknown model: {name}")

        path = info.get("path")
        card = None
        config = {}
        if path:
            p = Path(path)
            for candidate in ("README.md", "readme.md", "MODEL_CARD.md", "modelcard.md"):
                f = p / candidate
                if f.is_file():
                    try:
                        card = f.read_text(encoding="utf-8", errors="replace")[:40000]
                        break
                    except OSError:
                        continue
            vlm_cfg = p / "devkit" / "vlm_config.json"
            if vlm_cfg.is_file():
                try:
                    raw = json.loads(vlm_cfg.read_text(encoding="utf-8"))
                    for key in ("model_type", "vision_model_name", "max_seq_len",
                                "context_length", "max_position_embeddings", "vm_cfg"):
                        if key in raw:
                            config[key] = raw[key]
                except Exception:
                    pass

        return {
            "name": name,
            "type": info.get("type", "chat"),
            "supportsVision": bool(info.get("supports_vision")),
            "imageSize": info.get("image_size"),
            "sizeBytes": self._size_of(path),
            "params": parse_param_count(name),
            "quantization": parse_quantization(name),
            "config": config,
            "card": card,
            "source": "catalog",
        }

    def _stop_model_streams(self, name: str) -> None:
        """Cancel in-flight generation for a model via the server's /stop."""
        payload = json.dumps({"model": name}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._openai_base_url}/stop",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass

    def _warm_check(self, name: str) -> tuple[bool, str]:
        """Send one throwaway request to trigger + verify the MLA load.

        Returns (ok, detail). ``ok`` is True on success; otherwise ``detail``
        carries the server's error text so the caller can classify it.
        """
        payload = json.dumps({
            "model": name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self._openai_base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                resp.read()
            return True, ""
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:
                body = ""
            return False, f"HTTP {exc.code}: {body}".strip()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            return False, str(exc)

    @staticmethod
    def _silence_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
        """16 kHz mono 16-bit PCM silence — the shape Whisper preprocessing wants."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(b"\x00\x00" * int(rate * seconds))
        return buf.getvalue()

    @staticmethod
    def _multipart_body(fields: dict, file_field: str, filename: str,
                        content: bytes, content_type: str) -> tuple[bytes, str]:
        """Encode a multipart/form-data body (this process is stdlib-only)."""
        boundary = "----neat" + os.urandom(16).hex()
        body = bytearray()
        for key, value in fields.items():
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                     f"{value}\r\n").encode("utf-8")
        body += (f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{file_field}"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n").encode("utf-8")
        body += content + b"\r\n" + f"--{boundary}--\r\n".encode("utf-8")
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    def _warm_check_asr(self, name: str) -> tuple[bool, str]:
        """Force an ASR model's deferred MLA load by transcribing silence.

        The chat ``_warm_check`` sends a completion an ASR model cannot serve,
        so speech models need their own probe. A silent clip is a legitimate
        request: Whisper answers 200 with empty/low-confidence text, which is
        all we need to know the weights reached the accelerator.
        """
        body, content_type = self._multipart_body(
            {"model": name, "language": "en"},
            "file", "warmup.wav", self._silence_wav(), "audio/wav",
        )
        req = urllib.request.Request(
            f"{self._openai_base_url}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_ASR_WARM_TIMEOUT_S) as resp:
                resp.read()
            return True, ""
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            return False, f"HTTP {exc.code}: {detail}".strip()
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            return False, str(exc)

    def _request_supervised_reset(self, reason: str, message: str) -> dict:
        """Ask the supervisor (run.sh) to reset the MLA dispatcher and relaunch.

        A dispatcher restart cannot be done from inside this process — it frees
        the models this process still holds and the runtime cannot reconnect — so
        exit with the sentinel code once the HTTP response has had a moment to
        flush. run.sh resets the dispatcher and relaunches just the model server;
        the UI stays up and reconnects.
        """
        logging.error(
            "requesting supervised MLA reset + relaunch (%s; exit code %d)",
            reason, self._mla_reset_exit_code,
        )

        def _exit_soon() -> None:
            time.sleep(1.5)
            os._exit(self._mla_reset_exit_code)

        threading.Thread(target=_exit_soon, daemon=True).start()
        return {"state": "resetting", "reset": True, "message": message}

    def reset_mla(self) -> dict:
        """Explicit, user-triggered accelerator reset.

        Drains in-flight streams best-effort and deliberately WITHOUT taking
        ``_op_lock``: the whole point is that it still works when a load has
        wedged and is holding that lock. Nothing else in the studio calls this —
        startup and load failures leave the board runtime alone.
        """
        for victim in list(self._resident):
            try:
                self._stop_model_streams(victim)
            except Exception:
                pass
        return self._request_supervised_reset(
            "user requested MLA reset",
            "Resetting the accelerator and restarting. Reconnecting shortly…",
        )

    def _handle_mla_failure(self, name: str, detail: str) -> dict:
        """Roll back a failed MLA load and report it without runtime recovery."""
        logging.error("MLA load failed for '%s': %s", name, detail)
        self._record_error(name, detail, kind="mla")
        try:
            self._server.remove_model(name)
        except Exception:
            pass
        with self._lock:
            if name in self._resident:
                self._resident = []
            if name == self._active_asr:
                self._active_asr = None
        raise RuntimeError(
            f"Model '{name}' could not be loaded because the accelerator (MLA) "
            "reported an error. Check the board runtime before retrying."
        )

    # -- status ----------------------------------------------------------------

    def hub_enabled(self) -> bool:
        # Status polling reports configuration only. Reachability is checked by
        # explicit Hub search/card/download operations, never during UI startup.
        return bool(self._hub.allow_download)

    def disk_info(self) -> dict | None:
        """Free / total bytes of the filesystem that holds the model catalog (the
        NVMe on the board), so the UI can show how much room is left for
        downloads. Walks up to an existing directory when the catalog dir has not
        been created yet."""
        probe = self._catalog_dir or Path.cwd()
        while probe and not probe.exists():
            parent = probe.parent
            probe = parent if parent != probe else None
        if probe is None:
            return None
        try:
            usage = shutil.disk_usage(str(probe))
        except Exception:  # noqa: BLE001
            return None
        return {
            "freeBytes": usage.free,
            "totalBytes": usage.total,
            "usedBytes": usage.used,
            "path": str(self._catalog_dir or probe),
        }

    def status(self) -> dict:
        return {
            "catalog": self.catalog(),
            "loaded": self._server_model_names(),
            "maxResident": self._max_resident,
            # The ASR model serving transcriptions now, and the one a restart
            # re-selects from config (they differ after a runtime switch).
            "asrModel": self._active_asr,
            "configuredAsrModel": self._configured_asr,
            "hubEnabled": self.hub_enabled(),
            "loading": self.loading_status(),
            "disk": self.disk_info(),
        }
