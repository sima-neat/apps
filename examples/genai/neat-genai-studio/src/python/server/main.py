#!/usr/bin/env python3
"""Start Neat OpenAI hosting and the model-management control API for Neat GenAI Studio."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import signal
import socket
import sys
import time


def _request_shutdown(signum, frame):
    """Turn SIGTERM into a KeyboardInterrupt so the finally: server.stop() runs
    (which releases the models held on the MLA) instead of dying abruptly."""
    raise KeyboardInterrupt()

PYTHON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_DIR))

from shared.config import AppConfig, DEFAULT_SERVER_CONFIG, load_server_config
from server.control_api import serve_control_api
from server.load_log import LoadLogTap
from server.model_manager import ModelManager

# How many catalogued speech models to try when the configured one fails.
_ASR_FALLBACK_LIMIT = 3


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SERVER_CONFIG)
    return parser


def port_accepts_connections(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    try:
        with socket.create_connection((probe_host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_openai_server(cfg: AppConfig):
    try:
        import pyneat
    except ImportError as exc:
        raise RuntimeError(
            "pyneat is required. Run this example from the Python environment "
            "provided by the installed Neat Development Environment."
        ) from exc

    if len(cfg.chat_models) > 1:
        raise RuntimeError("Only one startup chat/VLM model may be configured")

    options = pyneat.GenAIServerOptions()
    options.host = cfg.openai.host
    options.port = cfg.openai.port

    server = None
    try:
        server = pyneat.GenAIServer(options)
        # Startup models are optional. Add only those whose directories exist;
        # everything else can be loaded on demand from the UI catalog.
        for model in cfg.chat_models:
            if model.path is None or not model.path.is_dir():
                print(f"skipping chat model (missing dir): {model.name} -> {model.path}", flush=True)
                continue
            chat_name = server.add_model(model.path, model.name)
            print(f"added chat model: {chat_name} -> {model.path}", flush=True)
            break

        served_asr_name = None
        if cfg.asr_model and cfg.asr_model.path and cfg.asr_model.path.is_dir():
            asr_name = server.add_model(cfg.asr_model.path, cfg.asr_model.name)
            served_asr_name = asr_name
            print(f"added ASR model: {asr_name} -> {cfg.asr_model.path}", flush=True)
        elif cfg.asr_model:
            print(f"skipping ASR model (missing dir): {cfg.asr_model.path}", flush=True)

        loaded = server.model_names()
        print(f"available models: {', '.join(loaded) if loaded else '(none — load from the UI)'}", flush=True)
        print(
            f"serving OpenAI-compatible API on http://{cfg.openai.host}:{cfg.openai.port}",
            flush=True,
        )

        try:
            server.start()
        except Exception as exc:
            # start() warms every registered model, so ONE unusable startup
            # model takes the whole studio down with it — and the ASR model is
            # the likely culprit, because a runtime update can change which
            # Whisper build layout it accepts. Drop it and start without
            # speech-to-text rather than leaving the user with nothing: every
            # other model is still loadable, and a working one can be selected
            # from the UI.
            if served_asr_name is None:
                raise
            print(f"startup ASR model '{served_asr_name}' cannot be used: {exc}",
                  file=sys.stderr, flush=True)
            print("dropped it; looking for another speech model in the catalog",
                  file=sys.stderr, flush=True)
            # remove_model's return value is authoritative: retrying start()
            # with the model still registered just warms it again and fails
            # identically, so the recovery would not actually recover.
            removed = False
            try:
                removed = bool(server.remove_model(served_asr_name))
            except Exception as rm_exc:  # noqa: BLE001 - reported below
                print(f"could not unregister it: {rm_exc}", file=sys.stderr, flush=True)
            if not removed:
                raise RuntimeError(
                    f"the startup speech model '{served_asr_name}' could not be "
                    "unregistered, so the server cannot start without it. Remove "
                    "server.models.asr from the config, or point it at a model "
                    "this runtime accepts."
                ) from exc
            served_asr_name = None
            server.start()
        return server, served_asr_name
    except BaseException:
        if server is not None:
            server.stop()
        raise


def main() -> int:
    signal.signal(signal.SIGTERM, _request_shutdown)
    args = build_arg_parser().parse_args()
    if not args.config.is_file():
        print(f"config does not exist: {args.config}", file=sys.stderr)
        return 2

    try:
        cfg = load_server_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    # Missing startup-model directories are a warning, not an error: the server
    # starts anyway and the UI can load available catalog models on demand.
    configured = [m for m in (*cfg.chat_models, cfg.asr_model) if m is not None]
    missing = [str(m.path) for m in configured if m.path is None or not m.path.is_dir()]
    if missing:
        print("warning: configured model directories not found (they will be skipped):",
              file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)

    server = None
    manager = None
    control_httpd = None
    log_tap = None
    try:
        if port_accepts_connections(cfg.openai.host, cfg.openai.port):
            raise RuntimeError(
                f"port {cfg.openai.port} is already accepting connections. "
                "Stop the old model server before starting a new one."
            )

        # Tee stdout to capture the accelerator's per-ELF load lines (real load
        # progress + a live loading log for the UI). Best-effort: if it can't
        # install it leaves stdout untouched. Disable with STUDIO_LOAD_LOG_TAP=0.
        if os.environ.get("STUDIO_LOAD_LOG_TAP", "1") != "0":
            # NEAT's native device-log filter (SIMA_GST_SUPPRESS_DEVICE_LOGS, on
            # by default) drops exactly the "Loading model …/Done loading …_mla.elf"
            # lines we count — and it sits upstream of our pipe, so it would blind
            # the tap. Keep those lines flowing (respect an explicit user override).
            # Must be set before pyneat is imported (in start_openai_server).
            os.environ.setdefault("SIMA_GST_SUPPRESS_DEVICE_LOGS", "0")
            tap = LoadLogTap()
            log_tap = tap if tap.install() else None

        server, served_asr_name = start_openai_server(cfg)

        manager = ModelManager(
            server,
            catalog_dir=cfg.catalog_dir,
            max_resident_chat_models=cfg.max_resident_chat_models,
            # The runtime may serve the model under a different name than the
            # configured one; the manager must track what is actually loaded, or
            # it reports no active ASR and a later switch fails to evict it.
            asr_name=served_asr_name or (cfg.asr_model.name if cfg.asr_model else None),
            # ... while the configured alias stays what a restart re-selects,
            # so status and the "startup default" marker keep naming it.
            configured_asr_name=cfg.asr_model.name if cfg.asr_model else None,
            # Switching ASR models warms the new one (a short silent clip) so a
            # bad load surfaces during the switch, not on the next transcription.
            asr_warmup=os.environ.get("STUDIO_ASR_WARMUP", "1") != "0",
            mla_reset_exit_code=int(os.environ.get("MLA_RESET_EXIT_CODE", "75")),
            # run.sh exports MLA_RESET; refuse the reset here so a disabled
            # board is never torn down for a reset that will not happen.
            mla_reset_enabled=os.environ.get("MLA_RESET", "1") != "0",
            hub=cfg.hub,
            openai_base_url=cfg.openai.base_url,
            log_tap=log_tap,
        )
        # Make startup models (loaded from absolute paths) visible in the catalog.
        for model in cfg.chat_models:
            manager.register_startup_model(
                model.name, model.path,
                "vlm" if model.supports_vision else "chat",
                model.supports_vision, model.vision_image_size,
            )
        if cfg.asr_model:
            # Register the configured alias (what config re-selects) and, when
            # the runtime normalized it, the served name too, so the catalog
            # entry, the type lookup and the active-ASR pointer all agree.
            manager.register_startup_model(
                cfg.asr_model.name, cfg.asr_model.path, "asr", False, None
            )
            if served_asr_name and served_asr_name != cfg.asr_model.name:
                manager.register_startup_model(
                    served_asr_name, cfg.asr_model.path, "asr", False, None
                )
        manager.scan_catalog()

        # No speech model active — either none was configured, or the runtime
        # refused the configured one above. Try the other speech models already
        # in the catalog and keep the first that loads. Encoder layout
        # requirements have changed between runtime builds in both directions, so
        # trying is the only reliable test; a user whose runtime moved under them
        # gets working transcription instead of silence. Bounded, and
        # STUDIO_ASR_FALLBACK=0 turns it off.
        # Only when a CONFIGURED model failed. An omitted `asr:` is a documented
        # choice — the fully decoupled mode starts with nothing resident — so
        # inventing a model there would override the user and take accelerator
        # memory they did not ask to spend.
        if (cfg.asr_model is not None and manager.active_asr() is None
                and os.environ.get("STUDIO_ASR_FALLBACK", "1") != "0"):
            # Exclude the model that just failed BY PATH, not only by name: a
            # configured alias and the directory basename are two catalog entries
            # for one directory, so a name-only check retries the failure and
            # burns one of the few attempts a third candidate needs.
            failed_name = cfg.asr_model.name if cfg.asr_model else None
            failed_path = None
            if cfg.asr_model and cfg.asr_model.path:
                try:
                    failed_path = Path(cfg.asr_model.path).resolve()
                except Exception:  # noqa: BLE001 - name check still applies
                    failed_path = None
            tried = 0
            for entry in manager.catalog():
                if tried >= _ASR_FALLBACK_LIMIT:
                    break
                if entry.get("type") != "asr" or entry.get("complete") is False:
                    continue
                if entry["name"] == failed_name:
                    continue          # just failed; do not retry it
                if failed_path is not None and \
                        manager.resolved_model_path(entry["name"]) == failed_path:
                    continue          # same directory under its other name
                tried += 1
                try:
                    manager.set_active_asr(entry["name"])
                except Exception as exc:  # noqa: BLE001 - try the next candidate
                    print(f"speech model '{entry['name']}' did not load: "
                          f"{str(exc).splitlines()[0][:200]}", file=sys.stderr, flush=True)
                    continue
                print(f"using '{entry['name']}' for speech-to-text "
                      f"(the configured model is unavailable)", flush=True)
                break
            else:
                print(
                    "no speech model in the catalog loads on this runtime"
                    if tried else
                    "no other speech model in the catalog to fall back to",
                    file=sys.stderr, flush=True)
                print("starting without speech-to-text — download one from "
                      "Settings -> Add Model", file=sys.stderr, flush=True)

        control_httpd = serve_control_api(manager, cfg.control.host, cfg.control.port)
        print(
            f"serving model-management control API on "
            f"http://{cfg.control.host}:{cfg.control.port}",
            flush=True,
        )

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopping OpenAI-compatible API server...", flush=True)
        return 0
    except Exception as exc:
        print(f"server failed: {exc}", file=sys.stderr)
        return 2
    finally:
        # Deterministic teardown so pyneat's nanobind objects are released before
        # the interpreter finalizes (otherwise it prints "leaked instances"): stop
        # the control API (ends the daemon thread that holds the manager -> server),
        # stop the server, drop every reference, and collect. Dropping the last
        # GenAIServer reference runs its C++ destructor, which frees the MLA models.
        if control_httpd is not None:
            try:
                control_httpd.shutdown()
                control_httpd.server_close()
            except Exception:
                pass
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass
        if manager is not None:
            manager.close()
        # Restore the real stdout before the interpreter's final output.
        if log_tap is not None:
            try:
                log_tap.uninstall()
            except Exception:
                pass
        server = None
        manager = None
        control_httpd = None
        log_tap = None
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
