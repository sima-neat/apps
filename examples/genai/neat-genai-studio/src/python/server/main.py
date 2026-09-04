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

        if cfg.asr_model and cfg.asr_model.path and cfg.asr_model.path.is_dir():
            asr_name = server.add_model(cfg.asr_model.path, cfg.asr_model.name)
            print(f"added ASR model: {asr_name} -> {cfg.asr_model.path}", flush=True)
        elif cfg.asr_model:
            print(f"skipping ASR model (missing dir): {cfg.asr_model.path}", flush=True)

        loaded = server.model_names()
        print(f"available models: {', '.join(loaded) if loaded else '(none — load from the UI)'}", flush=True)
        print(
            f"serving OpenAI-compatible API on http://{cfg.openai.host}:{cfg.openai.port}",
            flush=True,
        )

        server.start()
        return server
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

        server = start_openai_server(cfg)

        manager = ModelManager(
            server,
            catalog_dir=cfg.catalog_dir,
            max_resident_chat_models=cfg.max_resident_chat_models,
            asr_name=cfg.asr_model.name if cfg.asr_model else None,
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
            manager.register_startup_model(
                cfg.asr_model.name, cfg.asr_model.path, "asr", False, None
            )
        manager.scan_catalog()

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
