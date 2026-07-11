#!/usr/bin/env python3
"""Runtime model management for Neat GenAI Studio.

Wraps a live ``pyneat.GenAIServer`` so chat/VLM models can be loaded and
unloaded on the fly (no restart), scans an on-disk catalog of compatible
models, keeps a bounded set resident in RAM (LRU eviction), and can download
additional compatible models from the Hugging Face Hub when the board is
online.

pyneat's ``add_model`` / ``remove_model`` are thread-safe and may be called
after ``server.start()``; this class serializes catalog mutations under a lock
and treats the ASR model as pinned (never evicted).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from shared.config import HubConfig, classify_model_dir, model_dir_complete
from server import hub as hub_helpers


def parse_param_count(name: str) -> str | None:
    """Extract a parameter count from a model name, e.g. '4B', '1.5B', '500M'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name or "")
    if m:
        return f"{m.group(1)}B"
    m = re.search(r"(\d+(?:\.\d+)?)\s*[mM](?![a-zA-Z])", name or "")
    if m:
        return f"{m.group(1)}M"
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
        switch_settle_s: float = 0.6,
        mla_reset_on_switch: bool = True,
        mla_reset_exit_code: int = 75,
        log_tap=None,
    ) -> None:
        self._server = server
        self._catalog_dir = Path(catalog_dir) if catalog_dir else None
        self._max_resident = max(1, int(max_resident_chat_models))
        self._asr_name = asr_name
        self._hub = hub
        self._openai_base_url = openai_base_url.rstrip("/")
        self._warmup = warmup
        # After unloading the outgoing model, wait briefly so its RAII free
        # (which returns MLA memory to the dispatcher) completes before loading
        # the replacement — avoids transient double-residency (MLA_LOAD_FAILED).
        self._switch_settle_s = max(0.0, float(switch_settle_s))
        # On a genuine MLA load failure, request a supervised restart (run.sh
        # resets the dispatcher and relaunches) instead of an unsafe in-process
        # dispatcher restart. Gated so the behavior is explicit.
        self._mla_reset_on_switch = bool(mla_reset_on_switch)
        self._mla_reset_exit_code = int(mla_reset_exit_code)

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
        self._sec_per_gb: float | None = None       # learned rate for ETA
        # Optional stdout tap: real per-ELF load progress + a live loading log.
        self._log_tap = log_tap
        # Last load failure, retained so the UI can surface it prominently.
        self._last_error: dict | None = None
        # Performance benchmark (web MoLE `perf`): TTFT/TPS over N passes.
        self._bench: dict | None = None
        self._bench_lock = threading.Lock()

        self.scan_catalog()
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
        with self._lock:
            infos = list(self._catalog.values())   # snapshot; then walk the FS
        entries = []                                # outside the lock (walks are slow)
        for info in infos:
            complete, reason = self._is_model_complete(info.get("path"))
            entries.append({
                "name": info["name"],
                "type": info.get("type", "chat"),
                "supportsVision": bool(info.get("supports_vision")),
                "imageSize": info.get("image_size"),
                "loaded": info["name"] in loaded,
                "pinned": info["name"] == self._asr_name,
                "sizeBytes": self._size_of(info.get("path")),
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

    def _sync_resident_from_server(self) -> None:
        with self._lock:
            loaded = self._server_model_names()
            self._resident = [
                n for n in self._resident if n in loaded and n != self._asr_name
            ]
            for name in loaded:
                if name != self._asr_name and name not in self._resident:
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
        and escalated to a supervised reset rather than surfacing on the user's
        first chat.
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
                self.touch(name)
                return {"name": name, "state": "ready", "evicted": [], "cold_start": False,
                        "load_seconds": 0.0}

            if not path or not Path(path).is_dir():
                raise ValueError(f"Model directory not found for: {name}")

            # Refuse to load a model whose weights are missing/partial — otherwise
            # it fails deep in the MLA with a confusing error.
            complete, reason = self._is_model_complete(path)
            if not complete:
                raise ValueError(
                    f"'{name}' cannot be loaded — {reason}. Re-download it from Add Model."
                )

            with self._lock:
                # The ASR model is pinned and not tracked in _resident.
                victims = [] if is_asr else [v for v in self._resident if v != name]

            size_bytes = self._size_of(path)
            started = time.monotonic()
            self._last_error = None
            if not is_asr:
                self._loading = {
                    "name": name,
                    "startedAt": started,
                    "estTotalS": self._estimate_load_seconds(name, size_bytes),
                    # Real progress: count completed ELF-stage loads (from the
                    # stdout tap) minus the baseline against the on-disk total.
                    "elfTotal": self._count_elf_stages(path),
                    "elfBase": self._log_tap.loaded_count if self._log_tap else 0,
                }
            evicted: list = []
            try:
                # Clear every other chat/VLM model. _stop_model_streams (an HTTP
                # /stop) and remove_model (which triggers the outgoing model's RAII
                # free of MLA memory) can each block for seconds — do them WITHOUT
                # holding _lock so status polls stay responsive during the switch.
                for victim in victims:
                    self._stop_model_streams(victim)
                    try:
                        self._server.remove_model(victim)
                        evicted.append(victim)
                    except Exception:
                        pass
                if not is_asr:
                    with self._lock:
                        self._resident = []
                    # Let the free complete before loading the replacement, so the
                    # old and new model are never briefly co-resident.
                    if evicted and self._switch_settle_s:
                        time.sleep(self._switch_settle_s)

                served = self._server.add_model(str(path), name)
                if not is_asr:
                    with self._lock:
                        self._resident = [served]

                # add_model only registers; the real MLA load is deferred to first
                # inference. Warm synchronously so a load failure is catchable and
                # so we can time the load (this is where the wait actually happens).
                if is_asr or not self._warmup:
                    return {"name": name, "state": "ready", "evicted": evicted,
                            "cold_start": True, "load_seconds": round(time.monotonic() - started, 1)}

                ok, detail = self._warm_check(name)
                if ok:
                    self._record_load_duration(name, size_bytes, time.monotonic() - started)
                    return {"name": name, "state": "ready", "evicted": evicted,
                            "cold_start": True, "load_seconds": round(time.monotonic() - started, 1)}

                if _is_mla_failure(detail):
                    return self._handle_mla_failure(name, evicted, detail)

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

        if elf_done is not None and elf_total:
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
                "pct": pct,
                "filesDone": elf_done,
                "filesTotal": elf_total,
                "estimated": estimated,
            }

        # Fallback: elapsed / learned-size estimate (no per-stage signal). Report
        # filesTotal=None too, so the UI shows elapsed time rather than a frozen
        # "stage 0 of N" against a total it can never make progress against.
        est = info.get("estTotalS")
        pct = int(min(97, elapsed / est * 100)) if est and est > 0 else None
        return {
            "name": info["name"],
            "elapsedS": round(elapsed, 1),
            "etaS": round(est, 1) if est else None,
            "pct": pct,
            "filesDone": None,
            "filesTotal": None,
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
        return {"stopped": True}

    def unload(self, name: str) -> dict:
        name = (name or "").strip()
        if name == self._asr_name:
            raise ValueError("The ASR model cannot be unloaded")
        with self._op_lock:
            # remove_model frees MLA memory and can block for seconds — keep it
            # out of _lock so concurrent status polls are not held up.
            removed = bool(self._server.remove_model(name))
            with self._lock:
                if name in self._resident:
                    self._resident.remove(name)
        return {"name": name, "state": "unloaded" if removed else "absent"}

    def delete(self, name: str) -> dict:
        """Unload (if loaded) and delete a model's files from the catalog.

        Guarded: refuses to delete the pinned ASR model or anything outside
        ``catalog_dir``.
        """
        name = (name or "").strip()
        if name == self._asr_name:
            raise ValueError("The ASR model cannot be deleted")
        if not self._catalog_dir:
            raise ValueError("No catalog_dir configured; refusing to delete")
        with self._op_lock:
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

    def _handle_mla_failure(self, name: str, evicted: list, detail: str) -> dict:
        """A model failed to load with an MLA error. Escalate to a reset."""
        logging.error("MLA load failed for '%s': %s", name, detail)
        self._record_error(name, detail, kind="mla")
        if not self._mla_reset_on_switch:
            with self._lock:
                try:
                    self._server.remove_model(name)
                except Exception:
                    pass
                self._resident = []
            raise RuntimeError(
                f"Model '{name}' could not be loaded because the accelerator (MLA) "
                "is busy or wedged. Restart the studio (run.sh) to clear it."
            )

        return self._request_supervised_reset(
            f"MLA load failed for '{name}'",
            "The accelerator was busy, so the studio is resetting it and "
            "restarting. Reconnecting shortly…",
            extra={"name": name, "evicted": evicted},
        )

    def _request_supervised_reset(self, reason: str, message: str,
                                  extra: dict | None = None) -> dict:
        """Ask the supervisor (run.sh) to reset the MLA dispatcher and relaunch.

        A dispatcher restart cannot be done safely in-process (it would free the
        pinned ASR model and the runtime cannot reconnect), so we exit with the
        sentinel code after the HTTP response has had a moment to flush; run.sh
        resets the dispatcher and relaunches just the model server.
        """
        logging.error(
            "requesting supervised MLA reset + relaunch (%s; exit code %d)",
            reason, self._mla_reset_exit_code,
        )

        def _exit_soon() -> None:
            time.sleep(1.5)
            os._exit(self._mla_reset_exit_code)

        threading.Thread(target=_exit_soon, daemon=True).start()
        result = {"state": "resetting", "reset": True, "message": message}
        if extra:
            result.update(extra)
        return result

    def reset_mla(self) -> dict:
        """User-triggered accelerator reset: drain in-flight streams (best-effort,
        WITHOUT taking the op-lock so it works even if a load is wedged), then ask
        the supervisor to reset the MLA dispatcher and relaunch the server."""
        for victim in list(self._resident):
            try:
                self._stop_model_streams(victim)
            except Exception:
                pass
        return self._request_supervised_reset(
            "user requested MLA reset",
            "Resetting the accelerator and restarting. Reconnecting shortly…",
        )

    # -- status ----------------------------------------------------------------

    def hub_enabled(self) -> bool:
        return hub_helpers.hub_enabled(self._hub)

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
            "asrModel": self._asr_name,
            "hubEnabled": self.hub_enabled(),
            "loading": self.loading_status(),
            "disk": self.disk_info(),
        }
