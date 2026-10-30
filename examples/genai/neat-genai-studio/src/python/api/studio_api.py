#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
import ipaddress
import json
import logging
import math
import os
import requests
import re
import sys
import threading
import time

# Flask imports
from flask import Flask, Response, jsonify, request, stream_with_context

from pathlib import Path, PurePath
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile
import atexit
import signal
import socket
import subprocess

from shared.config import HubConfig
from shared.chat_limits import accelerator_busy, first_chat_event, sse_error, whole_reply
from server import hub as hub_helpers
from asr_metadata import (
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_NO_SPEECH_THRESHOLD,
    analyze_transcription,
)
import sys as _sys, os as _os
_TTS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "tts")
if _TTS_DIR not in _sys.path:
    _sys.path.insert(0, _TTS_DIR)
import supertonic_tts
from talk_controller import TalkController
import backend_mode
import rag_chat
from shared.board_camera import (
    capture_camera_frame,
    default_camera_device,
    list_camera_devices,
    normalize_camera_device,
)
from rag.event_loop import close_thread_event_loop
from audio_api import (
    MAX_TRANSCRIPTION_BYTES,
    AudioApiError,
    format_transcription,
    parse_speech_request,
    parse_transcription_form,
)
from voice_catalog import (
    asset_paths as catalog_asset_paths,
    catalog_voices,
    install_voice as install_catalog_voice,
    load_catalog,
    voice_by_id,
)

APP_DIR = Path(__file__).resolve().parent
VOICE_CATALOG = load_catalog()

_VERSION_CACHE = None


def _https_credentials():
    """Return configured TLS files or generate a unique local pair once."""
    cert_override = os.environ.get('NEAT_TLS_CERT')
    key_override = os.environ.get('NEAT_TLS_KEY')
    if bool(cert_override) != bool(key_override):
        raise RuntimeError('Set both NEAT_TLS_CERT and NEAT_TLS_KEY')
    if cert_override and key_override:
        cert_path = Path(cert_override).expanduser()
        key_path = Path(key_override).expanduser()
        if not cert_path.is_file() or not key_path.is_file():
            raise RuntimeError('Configured TLS certificate or key does not exist')
        return str(cert_path), str(key_path)

    tls_dir = APP_DIR.parents[2] / '.local-certs'
    cert_path = tls_dir / 'server.crt'
    key_path = tls_dir / 'server.key'
    if cert_path.is_file() and key_path.is_file():
        return str(cert_path), str(key_path)

    tls_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    hostname = re.sub(r'[^A-Za-z0-9.-]', '', socket.gethostname()) or 'localhost'
    sans = {'DNS:localhost', f'DNS:{hostname}', 'IP:127.0.0.1', 'IP:::1'}
    try:
        for item in socket.getaddrinfo(hostname, None):
            address = item[4][0].split('%', 1)[0]
            sans.add(f'IP:{ipaddress.ip_address(address)}')
    except OSError:
        pass
    try:
        subprocess.run(
            [
                'openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-sha256',
                '-nodes', '-days', '825', '-keyout', str(key_path),
                '-out', str(cert_path), '-subj', f'/CN={hostname}',
                '-addext', f"subjectAltName={','.join(sorted(sans))}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'Could not generate local HTTPS credentials: {exc}') from exc
    key_path.chmod(0o600)
    cert_path.chmod(0o644)
    logging.info('Generated per-install HTTPS certificate at %s', cert_path)
    return str(cert_path), str(key_path)


def _studio_version():
    """Best-effort version info for / and /health: git commit, branch and
    date of the example checkout, or 'unknown' when run from a standalone (non-git)
    fetch. Cached after the first call."""
    global _VERSION_CACHE
    if _VERSION_CACHE is not None:
        return _VERSION_CACHE
    info = {"name": "Neat GenAI Studio", "commit": "unknown", "branch": "unknown",
            "date": "", "dirty": False}
    example_dir = str(Path(__file__).resolve().parents[3])

    def _git(*args):
        try:
            r = subprocess.run(["git", "-C", example_dir, *args],
                               capture_output=True, text=True, timeout=3)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:  # noqa: BLE001
            return ""

    commit = _git("rev-parse", "--short", "HEAD")
    if commit:
        info["commit"] = commit
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
        info["date"] = _git("show", "-s", "--format=%cs", "HEAD")
        info["dirty"] = bool(_git("status", "--porcelain"))
    _VERSION_CACHE = info
    return info


def _supervisor_pid():
    """PID of the run.sh supervisor so /shutdown can stop the Studio gracefully.
    run.sh exports NEAT_RUN_PID and also writes its PID to a file."""
    val = (os.environ.get('NEAT_RUN_PID') or '').strip()
    if val.isdigit() and int(val) > 0:
        return int(val)
    candidates = [os.environ.get('RUN_PID_FILE'),
                  str(Path(__file__).resolve().parents[3] / '.neat-genai-studio.pid')]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                pid = int(Path(p).read_text().strip())
                if pid > 0:
                    return pid
            except (OSError, ValueError):
                pass
    return None


genai_app = None

vectodb_proc = None
rag_db_client = None
RAG_DB_PATH = None
start_service = None
create_markdown_vectordb = None
wait_for_rag_service = None

# One document update at a time: upload, import, reset and clear each stop the
# RAG service and replace or remove its database, so two at once would race.
RAG_UPDATE_LOCK = threading.Lock()
RAG_UPDATE_BUSY = "A document update is already running; try again when it finishes."
# Largest Markdown file /rag/upload accepts, and the request body allowed
# around it (multipart envelope and form fields).
MAX_RAG_UPLOAD_BYTES = 10 * 1024 * 1024
RAG_UPLOAD_REQUEST_LIMIT = MAX_RAG_UPLOAD_BYTES + 1024 * 1024


def ensure_rag_modules_loaded():
    global rag_db_client
    global RAG_DB_PATH
    global start_service
    global create_markdown_vectordb
    global wait_for_rag_service

    if rag_db_client is not None:
        return rag_db_client

    from rag.vectordb import (
        RAG_DB_PATH as _RAG_DB_PATH,
        start_service as _start_service,
        wait_for_service as _wait_for_service,
        RagDbClient,
    )
    from rag.create_db import create_markdown_vectordb as _create_markdown_vectordb

    RAG_DB_PATH = _RAG_DB_PATH
    start_service = _start_service
    create_markdown_vectordb = _create_markdown_vectordb
    wait_for_rag_service = _wait_for_service
    rag_db_client = RagDbClient()
    return rag_db_client

def studio_rag_client():
    """(client, other): the RAG client, and the database that the service on
    its port serves when it is not this Studio's (e.g. one started by another
    process), else None. Callers answer ``other`` with
    rag_chat.mismatch_message() instead of searching. A service that does not
    answer is left to the caller's request to report, as before."""
    client = ensure_rag_modules_loaded()
    return client, rag_chat.other_database(client.served_db_path(), RAG_DB_PATH)


def stop_service():
    global vectodb_proc
    if vectodb_proc and vectodb_proc.poll() is None:
        try:
            pgid = os.getpgid(vectodb_proc.pid)
        except ProcessLookupError:
            vectodb_proc = None
            return
        os.killpg(pgid, signal.SIGTERM)
        try:
            vectodb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            vectodb_proc.wait(timeout=5)
    vectodb_proc = None

def rag_service_failed():
    """Whether the document search service this Studio started has exited
    (a missing embedding model, a database it can't open): then it is not
    "starting" and won't come up by waiting."""
    proc = vectodb_proc
    return proc is not None and proc.poll() is not None


def restore_rag_database(backup_path, replacement_installed):
    """After a failed or abandoned update: put the previous database back and
    restart the service on it (no restart when there was none)."""
    global vectodb_proc
    stop_service()
    if replacement_installed and os.path.exists(RAG_DB_PATH):
        os.unlink(RAG_DB_PATH)
    if backup_path and os.path.exists(backup_path):
        os.replace(backup_path, RAG_DB_PATH)
    if os.path.exists(RAG_DB_PATH):
        try:
            vectodb_proc = start_service()
            wait_for_rag_service(vectodb_proc)
        except Exception:  # noqa: BLE001
            logging.exception("Failed to restore the previous RAG service")


def rag_update_response(stream):
    """Stream a document update that holds ``RAG_UPDATE_LOCK``, releasing it
    once: when the stream ends, or when the response closes because the
    client went away (possibly before the stream started)."""
    released = []

    def release():
        if not released:
            released.append(True)
            RAG_UPDATE_LOCK.release()

    def run():
        try:
            yield from stream
        finally:
            close_thread_event_loop()
            release()

    response = Response(stream_with_context(run()), mimetype="text/plain")
    response.call_on_close(release)
    return response


def rag_document_name():
    """The file the current database was built from, for passages that do not
    name one (empty when unknown)."""
    from rag.inspect_db import read_rag_meta
    return rag_chat.document_name(read_rag_meta())


def handle_shutdown_signal(signum, _frame):
    logging.info("Received signal %s; stopping RAG database service", signum)
    stop_service()
    # Let the Supertonic worker close its MLA runners instead of dying with
    # the process group.
    supertonic_tts.terminate_worker()
    raise KeyboardInterrupt

atexit.register(stop_service)
atexit.register(supertonic_tts.terminate_worker)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

class AppConstants:
    DEFAULT_SIMA_SERVER_IP = "127.0.0.1:9998"


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, default))
        if not math.isfinite(value):
            raise ValueError
        return value
    except (TypeError, ValueError):
        logging.warning("Invalid %s; using %s", name, default)
        return default


class AppContext:
    def __init__(self):
        self.app = None
        self.current_voice_by_lang = {}
        self.talk_ctrl = None
        self.cors_origins = ()      # BACKEND_CORS_ORIGINS / app.web.cors_origins (backend_mode.py)
        self.asr_model_name = ""
        self.max_tokens = None
        self.web_host = "0.0.0.0"
        self.web_port = 5000
        self.rag_enabled = True
        self.rag_embedding_model_dir = ""

        # Runtime model management + appearance
        self.control_base_url = "http://127.0.0.1:9997"
        self.catalog_dir = None
        self.hub_config = HubConfig(allow_download=True,
                                    orgs=("simaai", "TDoSiMa", "florianvoss"))
        self._asr_name_cache = (0.0, "")

    def set_asr_model_name(self, name):
        """Point transcription at a new ASR model after a successful switch."""
        self.asr_model_name = (name or "").strip()
        self._asr_name_cache = (time.monotonic(), self.asr_model_name)
        if self.app is not None:
            self.app.config['ASR_MODEL_NAME'] = self.asr_model_name
        return self.asr_model_name

    def resolve_asr_model(self):
        """The ASR model serving transcriptions, read fresh from the server.

        Nothing this process knows is authoritative: the active model can change
        without passing through here — an operator calling /control/asr directly,
        as the README documents — and sending an evicted name just fails the
        transcription. Any cache window is a window in which we get it wrong, so
        ask every time. This runs once per recording, against a localhost
        endpoint, next to an MLA transcription that costs orders of magnitude
        more; if it cannot be reached we keep the last known name rather than
        caching the failure.
        """
        _, cached = self._asr_name_cache
        try:
            resp = requests.get(
                f"{self.control_base_url.rstrip('/')}/control/status", timeout=5
            )
            # An HTTP error carries a JSON error body, not a status: a
            # transient 500 must not read as "no model active".
            resp.raise_for_status()
            name = str((resp.json() or {}).get("asrModel") or "")
            reachable = True
        except Exception:
            name, reachable = "", False
        if not reachable:
            # Say nothing new: keep using the last known name rather than
            # caching a failure that says nothing about what is loaded.
            return cached or self.asr_model_name
        if not name:
            # A successful status with no active model is real — a switch that
            # failed after eviction leaves exactly this. Forget the stale name
            # so recordings report "unavailable" instead of being sent to a
            # model the server no longer serves.
            logging.warning("No speech-to-text model is active on the server.")
            return self.set_asr_model_name("")
        if name != self.asr_model_name:
            logging.info("Active ASR model changed out from under the UI: %r", name)
            return self.set_asr_model_name(name)
        self._asr_name_cache = (time.monotonic(), name)
        return name

    def update_settings(self, model_server_ip, httponly):
        self.model_server_ip = AppConstants.DEFAULT_SIMA_SERVER_IP if model_server_ip is None else model_server_ip
        self.httponly = httponly
        self.update_config()

        self.talk_ctrl = TalkController([
            'en', 'fr', 'es', 'de', 'it', 'zh', 'vi', 'ja', 'pt', 'ko', 'no'
        ])

    def update_from_config(self, app_cfg):
        model_server = f"{app_cfg.openai.client_host}:{app_cfg.openai.port}"
        self.asr_model_name = app_cfg.asr_model.name if app_cfg.asr_model else ""
        self.max_tokens = app_cfg.request.max_tokens
        self.web_host = app_cfg.web.host
        self.web_port = app_cfg.web.port
        self.rag_enabled = app_cfg.rag.enabled
        self.rag_embedding_model_dir = app_cfg.rag.embedding_model_dir
        if self.rag_embedding_model_dir:
            os.environ["VDB_EMBED_MODEL_DIR"] = self.rag_embedding_model_dir
        self.control_base_url = app_cfg.control.base_url
        self.catalog_dir = app_cfg.catalog_dir
        self.hub_config = app_cfg.hub
        self.update_settings(model_server_ip=model_server, httponly=not app_cfg.web.https)

    def update_config(self):
        self.app.config['SIMAAI_IP_ADDR'] = self.model_server_ip
        self.app.config['ASR_MODEL_NAME'] = self.asr_model_name

    def get_config(self):
        return self.app.config
        
    def initialize(self):
        self.app = Flask(__name__, static_folder=None)

        def _cors_allowed_origin():
            """The request's Origin when the CORS allowlist allows it for this path."""
            if not self.cors_origins:
                return None
            origin = request.headers.get("Origin") or ""
            if origin and backend_mode.cors_path(request.path) \
                    and backend_mode.origin_allowed(origin, self.cors_origins):
                return origin
            return None

        @self.app.before_request
        def _backend_only_gate():
            """Only the API surface is served (backend_mode.py): any other path
            answers 404, and an allowed CORS preflight is answered here."""
            if not backend_mode.path_allowed(request.path):
                return jsonify({"error": "Not available in backend-only mode.",
                                "health": "/health"}), 404
            if request.path == "/" and request.method in ("GET", "HEAD"):
                return jsonify({"studio": "backend-only", "version": _studio_version(),
                                "health": "/health"})
            origin = _cors_allowed_origin()
            if request.method == "OPTIONS" and origin:
                return ("", 204, backend_mode.cors_headers(
                    origin, request.headers.get("Access-Control-Request-Headers")))
            return None

        @self.app.after_request
        def _backend_cors_headers(response):
            origin = _cors_allowed_origin()
            if origin:
                for key, value in backend_mode.cors_headers(
                        origin, request.headers.get("Access-Control-Request-Headers")).items():
                    response.headers[key] = value
            return response

        @self.app.before_request
        def _reject_cross_origin_mutation():
            """Keep browser mutations on the Studio's own origin.

            Headerless local API clients remain supported. Browsers provide
            Origin, Referer, or Sec-Fetch-Site for cross-origin requests.
            Origins on the BACKEND_CORS_ORIGINS allowlist may call the API
            paths.
            """
            if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
                return None
            if _cors_allowed_origin():
                return None

            expected_origin = request.host_url.rstrip("/")
            origin = (request.headers.get("Origin") or "").rstrip("/")
            if origin and origin != expected_origin:
                return jsonify({"error": "Cross-origin request rejected"}), 403

            fetch_site = request.headers.get("Sec-Fetch-Site", "").lower()
            if fetch_site in {"cross-site", "same-site"}:
                return jsonify({"error": "Cross-origin request rejected"}), 403

            referer = request.headers.get("Referer", "")
            if not origin and referer and not referer.startswith(request.host_url):
                return jsonify({"error": "Cross-origin request rejected"}), 403
            return None

    def run(self):
        ssl_context = None if self.httponly else _https_credentials()
        self.app.run(host=self.web_host, port=self.web_port,
                     ssl_context=ssl_context, debug=False, threaded=True)

    def setup_router(self):

        @self.app.route('/tts/engine', methods=['GET'])
        def tts_engine_get():
            if self.talk_ctrl is None:
                return jsonify({'engines': [], 'current': None})
            lang = (request.args.get('lang') or '').strip() or None
            return jsonify(self.talk_ctrl.voice_engine(lang))

        @self.app.route('/tts/engine', methods=['POST'])
        def tts_engine_set():
            data = request.get_json(silent=True) or {}
            engine = (data.get('engine') or '').strip()
            lang = (data.get('lang') or '').strip() or None
            if self.talk_ctrl is None or not self.talk_ctrl.set_voice_engine(engine, lang):
                return jsonify({'status': 'error', 'error': 'unknown or unavailable engine'}), 400
            return jsonify({'status': 'ok', 'current': self.talk_ctrl.voice_engine()['current']})

        @self.app.route('/piperplus/voices', methods=['GET'])
        def piperplus_voices():
            if self.talk_ctrl is None:
                return jsonify({'voices': [], 'current': None})
            return jsonify(self.talk_ctrl.piper_plus_voices())

        @self.app.route('/piperplus/select', methods=['POST'])
        def piperplus_select():
            data = request.get_json(silent=True) or {}
            key = (data.get('key') or '').strip()
            if not key or self.talk_ctrl is None:
                return jsonify({'status': 'error', 'error': 'no voice key'}), 400
            if not self.talk_ctrl.set_piper_plus_voice(key):
                return jsonify({'status': 'error',
                                'error': f"could not load piper-plus voice '{key}'"}), 400
            return jsonify({'status': 'ok', 'current': self.talk_ctrl.pp_current})

        @self.app.route('/supertonic/voices', methods=['GET'])
        def supertonic_voices():
            if self.talk_ctrl is None:
                return jsonify({'voices': [], 'current': None})
            return jsonify(self.talk_ctrl.supertonic_voices())

        @self.app.route('/supertonic/select', methods=['POST'])
        def supertonic_select():
            data = request.get_json(silent=True) or {}
            key = (data.get('key') or data.get('voice') or '').strip()
            if self.talk_ctrl is None or not key:
                return jsonify({'status': 'error', 'error': 'no voice key'}), 400
            if not self.talk_ctrl.set_supertonic_voice(key):
                return jsonify({'status': 'error',
                                'error': f"unknown Supertonic voice '{key}'"}), 400
            return jsonify({'status': 'ok',
                            'current': self.talk_ctrl.supertonic_voices()['current']})

        @self.app.route('/voices', methods=['GET'])
        def list_voices():
            lang = request.args.get('lang', 'en')
            assets_dir = Path('assets')
            voices = []
            for voice in catalog_voices(
                VOICE_CATALOG, engine="piper-tts", language=lang
            ):
                paths = catalog_asset_paths(voice, assets_dir)
                voices.append({
                    'id': voice['id'],
                    'label': voice['label'],
                    'installed': all(path.is_file() for path in paths),
                })
            current = self.current_voice_by_lang.get(lang)
            if not current:
                if self.talk_ctrl != None:
                    inst = self.talk_ctrl.pipers.get(lang)
                    try:
                        if inst is not None and hasattr(inst, 'model_path'):
                            current = Path(getattr(inst, 'model_path')).stem
                    except Exception:
                        current = None
            return jsonify({'lang': lang, 'voices': voices, 'current': current})

        @self.app.route('/voices/select', methods=['POST'])
        def select_voice():
            try:
                if self.talk_ctrl is None:
                    return jsonify({'error': 'TTS engine not initialized.'}), 503

                data = request.get_json() or {}
                lang = data.get('lang', 'en')
                voice_id = data.get('voiceId')
                if not voice_id:
                    return jsonify({'error': 'voiceId is required'}), 400
                assets_dir = Path('assets')
                voice_name = str(voice_id)
                catalog_voice = voice_by_id(voice_name.removesuffix('.onnx'), VOICE_CATALOG)
                if (
                    catalog_voice is None
                    or catalog_voice['engine'] != 'piper-tts'
                    or lang not in catalog_voice['languages']
                ):
                    return jsonify({'error': 'Voice is not in the catalog for this language'}), 400
                paths = catalog_asset_paths(catalog_voice, assets_dir)
                # Always run through the verified installer. Existing files are
                # retained only when their SHA-256 matches the pinned catalog.
                install_catalog_voice(catalog_voice, assets_path=assets_dir)
                candidate = next(path for path in paths if path.suffix == '.onnx')

                from pipertts import PiperTTS, prepare_voice_for_streaming
                if not prepare_voice_for_streaming(candidate):
                    logging.warning(
                        "Voice %s could not be prepared for split synthesis; keeping current voice",
                        catalog_voice['id'],
                    )
                    return jsonify({
                        'error': 'Voice could not be prepared for streaming; current voice unchanged'
                    }), 500
                with self.talk_ctrl.lock:
                    old = self.talk_ctrl.pipers.get(lang)
                    new_voice = PiperTTS(model_path=str(candidate))
                    new_voice.set_utterance_speed(self.talk_ctrl.utterance_speed)
                    self.talk_ctrl.pipers[lang] = new_voice
                    self.current_voice_by_lang[lang] = catalog_voice['id']
                    self.talk_ctrl.set_voice_engine('piper-tts')
                    # Drop reference to old instance; GC will reclaim when unreferenced
                    del old

                return jsonify({
                    'status': 'ok',
                    'lang': lang,
                    'current': self.current_voice_by_lang.get(lang),
                    'engine': 'piper-tts',
                })
            except Exception:
                logging.exception("Voice selection failed")
                return jsonify({'error': 'Voice selection failed'}), 500

        # ---- Runtime model management (proxied to the model-server control API) ----
        def _control_url(path):
            return f"{self.control_base_url.rstrip('/')}{path}"

        # The routes in backend_mode.CONTROL_ROUTES relay to the control API as they are.
        def _relay_to_control(method, route):
            target, timeout = backend_mode.CONTROL_ROUTES[(method, route)]

            def relay():
                try:
                    params, body = backend_mode.control_payload(
                        method, request.args.items(multi=True), request.get_json(silent=True))
                except ValueError:
                    return jsonify({'error': backend_mode.NOT_A_JSON_OBJECT}), 400
                try:
                    resp = requests.request(method, _control_url(target), params=params,
                                            json=body, timeout=timeout)
                except requests.RequestException as exc:
                    logging.warning("Control API %s %s failed: %s", method, target, exc)
                    return jsonify({'error': 'control API unreachable'}), 502
                return Response(resp.content, status=resp.status_code, mimetype='application/json')
            return relay

        for method, route in backend_mode.CONTROL_ROUTES:
            self.app.add_url_rule(route, endpoint=f"control {method} {route}",
                                  view_func=_relay_to_control(method, route), methods=[method])

        @self.app.route('/models/logs/stream', methods=['GET'])
        def models_logs_stream():
            # Server-sent events: push new model-load log lines (and live load
            # status) as they appear, by tailing the control API. Ends a couple
            # polls after the load finishes, or a safety cap if none starts.
            try:
                after = int(request.args.get('after', '0') or 0)
            except ValueError:
                after = 0
            # On an EventSource reconnect the browser replays Last-Event-ID; resume
            # from it so lines already delivered are not re-sent.
            try:
                after = max(after, int(request.headers.get('Last-Event-ID', 0) or 0))
            except (TypeError, ValueError):
                pass

            def _sse(cursor):
                seen_loading = False
                idle = 0
                errors = 0
                while True:
                    try:
                        r = requests.get(_control_url('/control/logs'),
                                         params={'after': cursor}, timeout=8)
                        d = r.json()
                        errors = 0
                    except requests.RequestException:
                        # A transient blip (timeout / reset under board load) must
                        # not kill the whole stream — retry a few times first.
                        errors += 1
                        if errors >= 20:
                            yield 'event: done\ndata: {"error": "control API unreachable"}\n\n'
                            return
                        time.sleep(0.5)
                        continue
                    for ln in (d.get('lines') or []):
                        seq = ln.get('seq') or 0
                        if seq > cursor:
                            cursor = seq
                    # id: lets the browser resume from `cursor` on reconnect.
                    yield f'id: {cursor}\ndata: ' + json.dumps(d) + '\n\n'
                    if d.get('loading'):
                        seen_loading, idle = True, 0
                    else:
                        idle += 1
                        # End shortly after a load finishes, or give up if no load
                        # ever starts (e.g. one queued behind another op ~30s).
                        if (seen_loading and idle >= 3) or (not seen_loading and idle >= 120):
                            yield 'event: done\ndata: {}\n\n'
                            return
                    time.sleep(0.25)

            return Response(
                stream_with_context(_sse(after)),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',   # don't let a proxy buffer the stream
                    'Connection': 'keep-alive',
                },
            )

        @self.app.route('/models/asr', methods=['POST'])
        def models_asr():
            # Not a plain relay: the response body carries the name the server
            # actually served the model under, and transcription must follow it.
            name = (request.get_json(silent=True) or {}).get('name', '')
            try:
                resp = requests.post(
                    _control_url('/control/asr'), json={'name': name}, timeout=600
                )
            except requests.RequestException as exc:
                logging.warning("Control API ASR switch failed: %s", exc)
                return jsonify({'error': 'control API unreachable'}), 502
            if resp.status_code < 300:
                try:
                    active = (resp.json() or {}).get('activeAsr') or name
                except ValueError:
                    active = name
                self.set_asr_model_name(active)
                logging.info("Active ASR model is now %r", active)
            return Response(resp.content, status=resp.status_code,
                            mimetype='application/json')

        @self.app.route('/shutdown', methods=['POST'])
        def shutdown_studio():
            # Stop the whole studio. run.sh writes its PID (and exports
            # NEAT_RUN_PID); a SIGTERM to it runs run.sh's cleanup, which
            # gracefully stops both the backend and the model server — the same path as
            # `./run.sh stop`. The kill is deferred so the HTTP response flushes.
            def _shutdown():
                time.sleep(0.6)
                pid = _supervisor_pid()
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        return
                    except OSError:
                        pass
                # No supervisor PID — fall back to our own process group / self.
                try:
                    os.killpg(os.getpgrp(), signal.SIGTERM)
                except OSError:
                    os._exit(0)
            threading.Thread(target=_shutdown, name="gui-shutdown", daemon=True).start()
            return jsonify({'status': 'shutting-down'})

        @self.app.route('/models/hub/search', methods=['GET'])
        def models_hub_search():
            query = request.args.get('q', '')
            try:
                return jsonify(hub_helpers.hub_search(self.catalog_dir, self.hub_config, query))
            except Exception:
                logging.exception("Hub search failed")
                return jsonify({'enabled': False, 'results': [], 'error': 'Hub search failed'}), 500

        @self.app.route('/models/hub/card', methods=['GET'])
        def models_hub_card():
            repo_id = request.args.get('repoId', '')
            try:
                return jsonify(hub_helpers.hub_card(self.hub_config, repo_id))
            except Exception:
                logging.exception("Hub card failed")
                return jsonify({'enabled': False, 'error': 'Could not load model metadata'}), 500

        @self.app.route('/models/hub/download', methods=['POST'])
        def models_hub_download():
            repo_id = (request.get_json(silent=True) or {}).get('repoId', '')

            def generate():
                for line in hub_helpers.hub_download_stream(
                    self.catalog_dir, self.hub_config, repo_id
                ):
                    yield line
                # Ask the model server to rescan so the new model becomes loadable.
                try:
                    requests.post(_control_url('/control/rescan'), timeout=30)
                except requests.RequestException:
                    pass

            return Response(
                stream_with_context(generate()), mimetype='application/x-ndjson'
            )

        @self.app.route('/v1/chat/completions', methods=['POST'])
        def proxy_chat_completions():
            # Streaming proxy to the model server's OpenAI API: the backend is
            # HTTPS while the model server is HTTP on :9998, so browser callers
            # would be blocked as mixed content (and lack CORS headers). The
            # caller keeps the conversation history.
            payload = request.get_json(silent=True)
            # The whole shape is checked once, before anything reads the request.
            invalid = rag_chat.chat_request_error(payload)
            if invalid:
                message, param = invalid
                return jsonify({'error': {'message': message, 'param': param,
                                          'type': 'invalid_request_error'}}), 400
            if self.max_tokens:
                payload.setdefault('max_tokens', int(self.max_tokens))
            _normalize_openai_image_parts(payload)
            # Extension: "neat_rag" answers from the user's documents (rag_chat.py).
            rag_headers = {}
            rag = rag_chat.rag_options(payload)
            if rag is not None:
                if not self.rag_enabled:
                    return jsonify({'error': {'message': 'Document search (RAG) is disabled on this board. '
                                                         'Run the Studio\'s setup.sh with SETUP_RAG=1 to install it.'}}), 409
                question = rag_chat.last_user_text(payload['messages'])
                if not question:
                    # Nothing to search with (no user text, e.g. only a picture).
                    return jsonify({'error': {'message': 'With neat_rag, the last user message must contain text',
                                              'param': 'messages', 'type': 'invalid_request_error'}}), 400
                other = None
                try:
                    client, other = studio_rag_client()
                    if not other:
                        # Only what fits the model's small window is used and reported.
                        hits = rag_chat.fit_passages(client.search(question, k=rag['k']))
                except Exception as exc:  # noqa: BLE001
                    logging.error(f"RAG search failed: {exc}")
                    from rag.inspect_db import default_db_path
                    message, status = rag_chat.search_failed_reply(os.path.isfile(default_db_path()), rag_service_failed())
                    return jsonify({'error': {'message': message}}), status
                if other:
                    return jsonify({'error': {'message': rag_chat.mismatch_message(other)}}), 409
                payload['messages'] = rag_chat.with_passages(payload['messages'], hits)
                rag_headers = {'X-RAG-Hits': str(len(hits)),
                               'X-RAG-Sources': rag_chat.sources_header(hits, rag_document_name())}
            model = payload.get('model')
            url = f"http://{self.app.config['SIMAAI_IP_ADDR']}/v1/chat/completions"

            def open_upstream():
                return requests.post(url, json=payload, stream=True, timeout=(10, 600))

            try:
                upstream = open_upstream()
            except requests.RequestException as exc:
                logging.error(f"Chat proxy: model server unreachable: {exc}")
                return jsonify({'error': 'model server unreachable'}), 502
            streaming = bool(payload.get('stream'))
            if not streaming:
                # A whole reply, retried once if the accelerator was busy.
                try:
                    status, content_type, body = whole_reply(upstream, open_upstream)
                except requests.RequestException as exc:
                    logging.error(f"Chat proxy: model server unreachable: {exc}")
                    return jsonify({'error': 'model server unreachable'}), 502
                return Response(body, status=status, content_type=content_type,
                                headers=rag_headers)

            def relay():
                nonlocal upstream
                try:
                    if streaming and upstream.ok:
                        # Busy accelerator: the 3.0 driver now and then refuses a
                        # job while several models run ("no free bank", rc=-11).
                        # If that comes before any text, one more try is safe.
                        held, lines, error = first_chat_event(upstream)
                        if error and accelerator_busy(error):
                            logging.warning("Chat proxy: accelerator busy for %s; retrying once", model)
                            upstream.close()
                            time.sleep(0.4)
                            # The client already has a 200 stream: a retry that
                            # fails is reported in it as an error event.
                            try:
                                upstream = open_upstream()
                            except requests.RequestException as exc:
                                yield sse_error(f"The model server could not be reached: {exc}")
                                return
                            if not upstream.ok:
                                yield b"data: " + upstream.content + b"\n\ndata: [DONE]\n\n"
                                return
                            held, lines, error = first_chat_event(upstream)
                        for line in held:
                            yield line + b"\n"
                        for line in lines:
                            yield line + b"\n"
                        return
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                except GeneratorExit:
                    # The caller aborted: free the accelerator instead of letting
                    # generation run out. Note /stop is per-model, so a concurrent
                    # generation on the same model would also stop.
                    try:
                        post_stop_to_sima(payload.get('model'))
                    except Exception:
                        pass
                    raise
                finally:
                    upstream.close()

            return Response(
                stream_with_context(relay()),
                status=upstream.status_code,
                content_type=upstream.headers.get('Content-Type', 'text/event-stream'),
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no', **rag_headers},
            )

        @self.app.route('/rag/status', methods=['GET'])
        def rag_status():
            """Whether document search is enabled, its service answers, and what
            the database was built from (the build-metadata sidecar)."""
            if not self.rag_enabled:
                return jsonify({"enabled": False, "service": "disabled", "database": False, "meta": None})
            from rag.inspect_db import read_rag_meta, default_db_path
            db_path = default_db_path()
            mismatch = None
            try:
                client, other = studio_rag_client()
                if other:
                    up, mismatch = False, rag_chat.mismatch_message(other)
                else:
                    up = client.is_server_up()
            except Exception:  # noqa: BLE001
                up = False
            exists = os.path.isfile(db_path)
            if mismatch:
                service = "failed"
            elif up:
                service = "ok"
            elif not exists:
                service = "no-database"
            else:
                service = "failed" if rag_service_failed() else "starting"
            return jsonify({
                "enabled": True,
                "service": service,
                "database": exists,
                "meta": read_rag_meta(db_path) if exists else None,
                **({"error": mismatch} if mismatch else {}),
            })

        @self.app.route('/rag/search', methods=['GET'])
        def rag_search():
            """The passages chat would add for ``query`` (``k`` up to 10)."""
            if not self.rag_enabled:
                return jsonify({"error": "RAG is disabled"}), 409
            query = (request.args.get("query") or "").strip()
            if not query:
                return jsonify({"error": "query is required"}), 400
            try:
                k = max(1, min(rag_chat.MAX_K, int(request.args.get("k", rag_chat.DEFAULT_K))))
            except (TypeError, ValueError):
                k = rag_chat.DEFAULT_K
            other = None
            try:
                client, other = studio_rag_client()
                if not other:
                    hits = client.search(query, k=k)
            except Exception as exc:  # noqa: BLE001
                logging.error(f"RAG search failed: {exc}")
                from rag.inspect_db import default_db_path
                message, status = rag_chat.search_failed_reply(os.path.isfile(default_db_path()), rag_service_failed())
                return jsonify({"error": message}), status
            if other:
                return jsonify({"error": rag_chat.mismatch_message(other)}), 409
            source = rag_document_name()
            return jsonify({"results": [dict(rag_chat.passage_source(h, source), content=h.get("content", ""))
                                        for h in hits]})

        @self.app.route('/rag/inspect', methods=['GET'])
        def rag_inspect():
            """Inspect the RAG database: the build-metadata sidecar plus every
            ingested chunk (source, headers, text). The sidecar is read directly;
            chunks come through the running VectorDB service (which owns the DB
            file). Degrades gracefully — the summary shows even if enumeration
            fails."""
            if not self.rag_enabled:
                return jsonify({"enabled": False, "error": "RAG is disabled"}), 200
            from rag.inspect_db import read_rag_meta, default_db_path
            try:
                limit = max(1, min(16383, int(request.args.get("limit", 16383))))
            except (TypeError, ValueError):
                limit = 16383
            db_path = default_db_path()
            result = {
                "enabled": True,
                "path": db_path,
                "exists": os.path.isfile(db_path),
                "meta": read_rag_meta(db_path),
                "documents": [],
                "count": 0,
                "collection": None,
                "error": None,
            }
            try:
                client, other = studio_rag_client()
                if other:
                    result["error"] = rag_chat.mismatch_message(other)
                else:
                    data = client.list_documents(limit=limit)
                    result["documents"] = data.get("documents", []) or []
                    result["count"] = data.get("count", len(result["documents"]))
                    result["collection"] = data.get("collection")
            except Exception as e:  # noqa: BLE001
                logging.error(f"RAG inspect error: {e}")
                result["error"] = "Could not read the RAG database"
            return jsonify(result), 200

        @self.app.route('/v1/audio/speech', methods=['POST'])
        @self.app.route('/audio/speech', methods=['POST'])
        def openai_tts():
            """OpenAI-compatible text-to-speech. See audio_api.parse_speech_request
            for the contract; WAV only, `language` is the Studio's extension."""
            try:
                try:
                    req = parse_speech_request(request.get_json(silent=True))
                except AudioApiError as exc:
                    return jsonify(exc.payload()), exc.status
                if self.talk_ctrl is None:
                    return jsonify({'error': 'TTS engine not initialized.'}), 503
                if req.used_speed_alias:
                    logging.info("speech request used a deprecated speed field; use 'speed'")
                logging.info("Received TTS request: model=%s, voice=%s, language=%s, speed=%s",
                             req.model, req.voice, req.language, req.speed)
                try:
                    result = self.talk_ctrl.tts_on_demand(
                        req.text, language=req.language, engine=req.model,
                        voice=req.voice, speed=req.speed)
                except AudioApiError as exc:          # a voice the engine cannot honour
                    return jsonify(exc.payload()), exc.status
                except TalkController.EngineUnavailable as exc:
                    # An explicitly requested engine that cannot serve this
                    # request is refused, never silently swapped. The message is
                    # a fixed string chosen by the reason code.
                    message = TalkController.ENGINE_REFUSALS.get(exc.reason, TalkController.ENGINE_REFUSALS['none'])
                    logging.info("TTS request refused (%s): %s", exc.reason, exc.engine or 'router')
                    return jsonify({'error': message, 'reason': exc.reason,
                                    'engine': exc.engine or None}), 503

                headers = {
                    'Content-Disposition': 'attachment; filename="speech.wav"',
                    'X-RTF': str(result['rtf']),
                    'X-Audio-Duration': str(result['audio_duration']),
                    'X-Elapsed-Time': str(result['elapsed_time']),
                    'X-Engine': str(result.get('engine') or ''),
                    'X-Voice': str(result.get('voice') or ''),
                    'X-Language': str(result.get('language') or ''),
                }
                if result.get('speed') is not None:
                    headers['X-Speed'] = f"{float(result['speed']):.2f}"
                return Response(result['audio_bytes'], mimetype='audio/wav', headers=headers)
            except Exception:
                logging.exception("TTS endpoint error")
                return jsonify({'error': 'TTS request failed'}), 500

        @self.app.route('/health', methods=['GET'])
        def health():
            """Readiness for integrating front ends: model server reachability,
            active ASR model, loaded chat models and TTS engines. Always 200;
            `ok` is false while the model server is unreachable."""
            status, error = None, None
            try:
                resp = requests.get(
                    f"{self.control_base_url.rstrip('/')}/control/status", timeout=3)
                resp.raise_for_status()
                status = resp.json()
            except Exception as exc:  # noqa: BLE001
                error = type(exc).__name__
            engines, failures = [], {}
            if self.talk_ctrl is not None:
                try:
                    engines = self.talk_ctrl.voice_engine()['engines']
                    failures = self.talk_ctrl.engine_failures()
                except Exception:  # noqa: BLE001
                    engines = []
            return jsonify(backend_mode.health_payload(
                mode='backend-only',
                version=_studio_version(), status=status, engines=engines, error=error,
                engine_failures=failures, rag_enabled=self.rag_enabled))

        # Cameras attached to the board (/dev/video*): a front end lists them and
        # attaches a snapshot to its next chat message.
        @self.app.route('/board-camera/devices', methods=['GET'])
        def board_camera_devices():
            return jsonify({
                'devices': list_camera_devices(),
                'default': default_camera_device(),
            })

        @self.app.route('/board-camera/snapshot', methods=['GET'])
        def board_camera_snapshot():
            device = (request.args.get('device') or '').strip()
            try:
                normalize_camera_device(device) if device else None
            except ValueError:
                return jsonify({'error': 'Invalid board camera device'}), 400
            try:
                jpeg, tool = capture_camera_frame(device or None)
            except RuntimeError as exc:
                logging.error(f"Board camera capture failed: {exc}")
                return jsonify({'error': 'Board camera capture failed'}), 503
            return Response(jpeg, mimetype='image/jpeg',
                            headers={'Cache-Control': 'no-store', 'X-Capture-Tool': tool})

        @self.app.route('/v1/audio/voices', methods=['GET'])
        @self.app.route('/audio/voices', methods=['GET'])
        def openai_voices():
            """Extension: which engines, voices and languages the speech API
            offers (OpenAI has no discovery call). Engines are listed whether or
            not they are loaded; selecting one loads it."""
            if self.talk_ctrl is None:
                return jsonify({'error': 'TTS engine not initialized.'}), 503
            try:
                return jsonify(self.talk_ctrl.voices_listing())
            except Exception:
                logging.exception("voices listing error")
                return jsonify({'error': 'could not list voices'}), 500

        @self.app.route('/v1/audio/transcriptions', methods=['POST'])
        @self.app.route('/audio/transcriptions', methods=['POST'])
        def openai_transcriptions():
            """OpenAI-compatible speech-to-text, proxied to the model server's
            /v1/audio/transcriptions so an HTTPS page and plain clients reach it
            on this origin. `response_format`: json (default) | verbose_json |
            text. `model` defaults to the active ASR model."""
            return _audio_to_text('transcribe')

        @self.app.route('/v1/audio/translations', methods=['POST'])
        @self.app.route('/audio/translations', methods=['POST'])
        def openai_translations():
            """OpenAI-compatible speech-to-English (Whisper's translate task),
            proxied to the model server's /v1/audio/translations. Same form and
            formats as transcriptions; the text is always English, and
            verbose_json reports the detected source `language`."""
            return _audio_to_text('translate')

        def _audio_to_text(task):
            started = time.time()
            # Refuse oversized bodies before Werkzeug parses (and spools) the
            # multipart data: the file limit plus headroom for the envelope and
            # the other fields. The exact per-file check follows after parsing.
            declared = request.content_length
            if declared is not None and declared > TRANSCRIPTION_REQUEST_LIMIT:
                return jsonify({'error': f'audio upload exceeds {MAX_TRANSCRIPTION_BYTES // (1024 * 1024)} MiB',
                                'param': 'file'}), 413
            try:
                request.max_content_length = TRANSCRIPTION_REQUEST_LIMIT   # bounds a chunked body too
            except AttributeError:
                # Flask < 3.1 (read-only there): a body without a length cannot
                # be bounded before parsing, so it is refused outright.
                if declared is None:
                    return jsonify({'error': 'Content-Length is required for audio uploads.',
                                    'param': 'file'}), 411
            try:
                upload = request.files.get('file')
            except RequestEntityTooLarge:
                return jsonify({'error': f'audio upload exceeds {MAX_TRANSCRIPTION_BYTES // (1024 * 1024)} MiB',
                                'param': 'file'}), 413
            try:
                req = parse_transcription_form(
                    request.form, has_file=upload is not None,
                    size=_upload_size(upload))
            except AudioApiError as exc:
                return jsonify(exc.payload()), exc.status
            # Bounded read: the file's own bytes are what the 25 MiB limit is
            # about (the multipart envelope around them does not count).
            audio_bytes = upload.stream.read(MAX_TRANSCRIPTION_BYTES + 1)
            if not audio_bytes:
                return jsonify({'error': 'Uploaded file is empty.', 'param': 'file'}), 400
            if len(audio_bytes) > MAX_TRANSCRIPTION_BYTES:
                return jsonify({'error': 'audio upload exceeds the size limit', 'param': 'file'}), 413
            filename = secure_filename(upload.filename or '') or 'audio.wav'
            content_type = upload.mimetype or 'application/octet-stream'
            try:
                result, model = transcribe_audio(
                    audio_bytes, language=req.language, model=req.model,
                    filename=filename, content_type=content_type, task=task)
            except TranscriptionError as exc:
                return jsonify({'error': exc.message}), exc.status
            asr = None
            if req.response_format == 'verbose_json':
                asr = analyze_transcription(
                    result,
                    requested_language=req.language,
                    supported_tts_languages=(self.talk_ctrl.supported_langs
                                             if self.talk_ctrl is not None else ()),
                    no_speech_threshold=_env_float('ASR_NO_SPEECH_THRESHOLD', DEFAULT_NO_SPEECH_THRESHOLD),
                    logprob_threshold=_env_float('ASR_LOGPROB_THRESHOLD', DEFAULT_LOGPROB_THRESHOLD),
                )
            body, mimetype = format_transcription(result, asr, req.response_format, model=model, task=task)
            elapsed = time.time() - started
            headers = {'X-ASR-Model': model or '', 'X-Task': task, 'X-Elapsed-Time': f"{elapsed:.3f}"}
            if isinstance(body, str):
                # content_type, not mimetype: the value already carries its
                # charset and Flask would append a second one.
                return Response(body, content_type=mimetype, headers=headers)
            resp = jsonify(body)
            for key, value in headers.items():
                resp.headers[key] = value
            return resp

        def rag_route_error(message, status):
            """The /rag/ API answers errors as JSON."""
            return jsonify({"error": message}), status

        def rag_disabled_reply():
            # 409, like /rag/search and chat with neat_rag.
            return jsonify({"error": "RAG is disabled"}), 409

        @self.app.route("/rag/upload", methods=["POST"])
        def upload_to_rag():
            if not self.rag_enabled:
                return rag_disabled_reply()

            too_large = f"The document is larger than {MAX_RAG_UPLOAD_BYTES // (1024 * 1024)} MiB."
            declared = request.content_length
            if declared is not None and declared > RAG_UPLOAD_REQUEST_LIMIT:
                return rag_route_error(too_large, 413)
            try:
                request.max_content_length = RAG_UPLOAD_REQUEST_LIMIT   # bounds a chunked body too
            except AttributeError:
                pass  # Flask < 3.1: the file's own size is still checked below
            try:
                uploaded_file = request.files.get("file")
            except RequestEntityTooLarge:
                return rag_route_error(too_large, 413)
            if not uploaded_file:
                return rag_route_error("No file provided", 400)

            # Checked on the name as sent: secure_filename drops non-Latin
            # letters ("日本語.md" -> "md"). The name is shown as the source.
            original = PurePath(str(uploaded_file.filename or "").replace("\\", "/")).name
            if not original.lower().endswith(".md"):
                return rag_route_error("Only Markdown RAG upload is supported locally.", 400)
            filename = original[-200:]
            size = _upload_size(uploaded_file)
            if size is not None and size > MAX_RAG_UPLOAD_BYTES:
                return rag_route_error(too_large, 413)
            # Read now: the request's files are closed once the response starts
            # streaming. Bounded, so at most the limit (plus one byte) is held.
            data = uploaded_file.stream.read(MAX_RAG_UPLOAD_BYTES + 1)
            if len(data) > MAX_RAG_UPLOAD_BYTES:
                return rag_route_error(too_large, 413)
            if not RAG_UPDATE_LOCK.acquire(blocking=False):
                return rag_route_error(RAG_UPDATE_BUSY, 409)

            def stream_response():
                global vectodb_proc
                yield "⏳ Starting...\n"
                tmp_path = None
                new_db_path = None
                backup_path = None
                service_stopped = False
                replacement_installed = False
                try:
                    # Written here rather than before streaming, so a client that
                    # goes away before the first line leaves no temporary file.
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
                        tmp_path = tmp.name
                        tmp.write(data)

                    ensure_rag_modules_loaded()
                    yield "📚 Creating VectorDB from Markdown...\n"
                    fd, new_db_path = tempfile.mkstemp(
                        prefix=f".{Path(RAG_DB_PATH).name}.new-",
                        suffix=".db",
                        dir=str(Path(RAG_DB_PATH).parent),
                    )
                    os.close(fd)
                    os.unlink(new_db_path)
                    create_markdown_vectordb(
                        input_path=tmp_path,
                        output_db=new_db_path,
                        embedding_model=self.rag_embedding_model_dir,
                    )

                    yield "🛑 Stopping existing vectordb service...\n"
                    stop_service()
                    service_stopped = True
                    if os.path.exists(RAG_DB_PATH):
                        fd, backup_path = tempfile.mkstemp(
                            prefix=f".{Path(RAG_DB_PATH).name}.backup-",
                            dir=str(Path(RAG_DB_PATH).parent),
                        )
                        os.close(fd)
                        os.unlink(backup_path)
                        os.replace(RAG_DB_PATH, backup_path)
                    os.replace(new_db_path, RAG_DB_PATH)
                    replacement_installed = True

                    new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                    yield "🚀 Starting vectordb service with new database...\n"
                    vectodb_proc = start_service()
                    wait_for_rag_service(vectodb_proc)
                    meta_path = str(Path(RAG_DB_PATH).with_suffix(".meta.json"))
                    if os.path.exists(new_meta_path):
                        os.replace(new_meta_path, meta_path)
                        # The database was built from a temporary copy; record
                        # the name the user uploaded instead.
                        try:
                            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
                            meta["input"] = filename
                            Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        except (OSError, ValueError) as exc:
                            logging.warning(f"Could not record the uploaded file name: {exc}")
                    elif os.path.exists(meta_path):
                        os.unlink(meta_path)
                    if backup_path:
                        os.unlink(backup_path)
                        backup_path = None
                    service_stopped = False   # done: nothing to roll back
                    yield "✅ Markdown RAG database is ready.\n"

                except BaseException as e:
                    # BaseException too: a client that disconnects mid-update
                    # raises GeneratorExit at a yield, and the previous
                    # database must still come back.
                    if isinstance(e, Exception):
                        logging.exception("RAG document upload failed: %s", e)
                    else:
                        logging.warning("RAG document upload interrupted (%s); restoring the previous database",
                                        type(e).__name__)
                    if service_stopped:
                        restore_rag_database(backup_path, replacement_installed)
                        backup_path = None
                    if not isinstance(e, Exception):
                        raise   # the client is gone; nothing more can be sent
                    yield "❌ RAG document upload failed.\n"
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    if new_db_path and os.path.exists(new_db_path):
                        os.unlink(new_db_path)
                    if new_db_path:
                        new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                        if os.path.exists(new_meta_path):
                            os.unlink(new_meta_path)

            return rag_update_response(stream_response())
        
        @self.app.route("/rag/reset", methods=["POST"])
        def reset_rag():
            """Rebuild the RAG database from the bundled default Markdown."""
            if not self.rag_enabled:
                return rag_disabled_reply()
            if not RAG_UPDATE_LOCK.acquire(blocking=False):
                return rag_route_error(RAG_UPDATE_BUSY, 409)
            default_md = Path(__file__).resolve().parents[2] / "common" / "rag" / "neat.md"

            def _stream():
                global vectodb_proc
                yield "⏳ Resetting to the default RAG database...\n"
                new_db_path = None
                backup_path = None
                service_stopped = False
                replacement_installed = False
                try:
                    ensure_rag_modules_loaded()
                    if not default_md.is_file():
                        yield f"❌ Default RAG source not found: {default_md}\n"
                        return
                    yield "📚 Rebuilding from the bundled document...\n"
                    fd, new_db_path = tempfile.mkstemp(
                        prefix=f".{Path(RAG_DB_PATH).name}.new-",
                        suffix=".db",
                        dir=str(Path(RAG_DB_PATH).parent),
                    )
                    os.close(fd)
                    os.unlink(new_db_path)
                    create_markdown_vectordb(
                        input_path=str(default_md),
                        output_db=new_db_path,
                        embedding_model=self.rag_embedding_model_dir,
                    )

                    yield "🛑 Stopping existing vectordb service...\n"
                    stop_service()
                    service_stopped = True
                    if os.path.exists(RAG_DB_PATH):
                        fd, backup_path = tempfile.mkstemp(
                            prefix=f".{Path(RAG_DB_PATH).name}.backup-",
                            dir=str(Path(RAG_DB_PATH).parent),
                        )
                        os.close(fd)
                        os.unlink(backup_path)
                        os.replace(RAG_DB_PATH, backup_path)
                    os.replace(new_db_path, RAG_DB_PATH)
                    replacement_installed = True

                    yield "🚀 Starting vectordb service with the default database...\n"
                    vectodb_proc = start_service()
                    wait_for_rag_service(vectodb_proc)
                    new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                    meta_path = str(Path(RAG_DB_PATH).with_suffix(".meta.json"))
                    if os.path.exists(new_meta_path):
                        os.replace(new_meta_path, meta_path)
                    elif os.path.exists(meta_path):
                        os.unlink(meta_path)
                    if backup_path:
                        os.unlink(backup_path)
                        backup_path = None
                    service_stopped = False   # done: nothing to roll back
                    yield "✅ RAG database reset to default.\n"
                except BaseException as e:  # noqa: BLE001
                    # BaseException too: see upload_to_rag.
                    if isinstance(e, Exception):
                        logging.exception("RAG database reset failed: %s", e)
                    else:
                        logging.warning("RAG database reset interrupted (%s); restoring the previous database",
                                        type(e).__name__)
                    if service_stopped:
                        restore_rag_database(backup_path, replacement_installed)
                        backup_path = None
                    if not isinstance(e, Exception):
                        raise
                    yield "❌ RAG database reset failed.\n"
                finally:
                    if new_db_path and os.path.exists(new_db_path):
                        os.unlink(new_db_path)
                    if new_db_path:
                        new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                        if os.path.exists(new_meta_path):
                            os.unlink(new_meta_path)

            return rag_update_response(_stream())

        @self.app.route("/rag/clear", methods=["POST"])
        def clear_rag():
            """Clear the RAG database — stop the service and remove the DB files."""
            if not self.rag_enabled:
                return rag_disabled_reply()
            if not RAG_UPDATE_LOCK.acquire(blocking=False):
                return rag_route_error(RAG_UPDATE_BUSY, 409)

            def _stream():
                global vectodb_proc
                yield "⏳ Clearing the RAG database...\n"
                try:
                    ensure_rag_modules_loaded()
                    yield "🛑 Stopping the vectordb service...\n"
                    stop_service()
                    vectodb_proc = None
                    meta_path = os.path.splitext(RAG_DB_PATH)[0] + ".meta.json"
                    removed = 0
                    for p in (RAG_DB_PATH, meta_path):
                        try:
                            os.remove(p)
                            removed += 1
                        except FileNotFoundError:
                            pass
                    yield (f"✅ RAG database cleared ({removed} file(s) removed). "
                           "Upload a document or reset to default to use RAG again.\n")
                except Exception as e:  # noqa: BLE001
                    logging.exception("RAG database cleanup failed: %s", e)
                    yield "❌ RAG database cleanup failed.\n"

            return rag_update_response(_stream())




def _normalize_openai_image_parts(payload):
    """Rewrite OpenAI-standard image parts to the pyneat server's schema.

    OpenAI clients send {"type": "image_url", "image_url": {"url": u}} while
    pyneat expects {"type": "image", "image": u}. Mutates payload in place. The payload
    has passed rag_chat.chat_request_error.
    """
    for message in payload['messages']:
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if part['type'] == 'image_url' and part['image_url']['url']:
                content[i] = {'type': 'image', 'image': part['image_url']['url']}


def post_stop_to_sima(model_name=None):
    cfg = genai_app.get_config()
    url = f"http://{str(cfg['SIMAAI_IP_ADDR']).strip()}/stop"
    normalized_model = str(model_name).strip() if model_name else ""
    request_kwargs = {"timeout": 5}
    if normalized_model:
        request_kwargs["json"] = {"model": normalized_model}

    logging.debug(f'Posting STOP signal to SIMA model server at {url} for model {normalized_model or "*"}')
    try:
        response = requests.post(url, **request_kwargs)
        response.raise_for_status()
        logging.info("Successfully sent stop signal to SiMa.ai server.")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to send stop signal: {e}")
        return False

class TranscriptionError(Exception):
    """The model server could not transcribe: ``status`` is the HTTP status the
    API should answer with, ``message`` a client-safe explanation."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# Hard limit on a transcription request body: the file cap plus headroom for
# the multipart envelope and the form fields (model, language, response_format).
TRANSCRIPTION_REQUEST_LIMIT = MAX_TRANSCRIPTION_BYTES + 1024 * 1024


def _upload_size(upload):
    """Byte length of a werkzeug ``FileStorage`` without consuming it (``None``
    when the stream cannot be measured); used for the 25 MiB upload limit."""
    if upload is None:
        return None
    try:
        stream = upload.stream
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell() - position
        stream.seek(position)
        return size
    except (AttributeError, OSError, ValueError):
        return None


def transcribe_audio(audio_bytes, *, language='auto', model=None, filename='audio.wav',
                     content_type='audio/wav', timeout=60, task='transcribe'):
    """Send audio to the model server's /v1/audio/transcriptions (``task``
    "transcribe") or /v1/audio/translations ("translate": Whisper's
    speech-to-English) and return ``(result_dict, model_name)``. ``model``
    defaults to the active ASR model. The model server decodes the upload with
    libavformat, so the browser's WebM/Opus and other container formats work;
    pass the real name and type."""
    cfg = genai_app.get_config()
    route = 'translations' if task == 'translate' else 'transcriptions'
    url = f"http://{str(cfg['SIMAAI_IP_ADDR']).strip()}/v1/audio/{route}"
    model = (model or '').strip() or genai_app.resolve_asr_model()
    if not model:
        logging.error("No speech-to-text model is active; cannot transcribe.")
        raise TranscriptionError(503, 'No speech-to-text model is active on the server.')
    files = {'file': (filename, audio_bytes, content_type)}
    # The server treats 'auto' as detection; it is sent as-is, exactly as the
    # chat upload path always has.
    data = {'model': model, 'language': language or 'auto'}
    logging.debug('Posting AUDIO to SIMA model server at %s (model=%s, %s)', url, model, filename)
    try:
        response = requests.post(url, files=files, data=data, timeout=timeout)
    except requests.RequestException as exc:
        logging.error("Failed to send audio for transcription: %s", exc)
        raise TranscriptionError(503, 'Speech transcription is unavailable. Please try again.')
    if response.status_code >= 400:
        detail = ''
        try:
            error = (response.json() or {}).get('error')
            # The model server answers OpenAI-style: {"error": {"message", "type"}}.
            detail = str(error.get('message') or error.get('type') or '') if isinstance(error, dict) else str(error or '')
        except ValueError:
            pass
        logging.error("Model server rejected the transcription (%s): %s", response.status_code, detail[:200])
        status = response.status_code if 400 <= response.status_code < 600 else 502
        raise TranscriptionError(status, detail[:300] or f'transcription failed (HTTP {response.status_code})')
    try:
        result = response.json()
    except ValueError:
        raise TranscriptionError(502, 'The model server returned a malformed transcription.')
    logging.info("Successfully sent audio for transcription to SiMa.ai server.")
    return result, model


def configure_logging(log_filename='server.log'):
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_api(app_cfg):
    global genai_app
    global vectodb_proc

    configure_logging()
    logging.info('Initializing the Neat GenAI Studio backend (API endpoints and TTS), please wait....')
    # Supertonic paths, before the TTS engines initialize: an explicit
    # environment override (run.sh exports one only when set) wins, then the
    # persisted config, then the module defaults.
    if app_cfg.supertonic.models_root:
        os.environ.setdefault("SUPERTONIC_MODELS_ROOT", app_cfg.supertonic.models_root)
    if app_cfg.supertonic.venv and not os.environ.get("SUPERTONIC_PYTHON"):
        os.environ.setdefault("SUPERTONIC_VENV", app_cfg.supertonic.venv)
    genai_app = AppContext()
    # The environment overrides the persisted allowlist (app.web.cors_origins).
    cors_raw = backend_mode.effective_cors_setting(os.environ, app_cfg.web.cors_origins)
    genai_app.cors_origins = backend_mode.parse_cors_origins(cors_raw)
    logging.info("CORS: %s", genai_app.cors_origins or "off")
    genai_app.initialize()
    genai_app.update_from_config(app_cfg)
    genai_app.setup_router()

    if genai_app.rag_enabled:
        try:
            ensure_rag_modules_loaded()
        except ImportError as exc:
            # A SETUP_RAG=0 install has no RAG packages; app.rag.enabled was
            # turned on by hand afterwards.
            raise SystemExit(
                f"app.rag.enabled is true but the RAG packages are not installed ({exc.name}). "
                f"Run SETUP_RAG=1 {Path(__file__).resolve().parents[3] / 'setup.sh'}, "
                "or set app.rag.enabled: false.") from exc
        # Without a database (cleared, or never built) the service stays off,
        # so /rag/status keeps reporting "no-database" and chat with neat_rag
        # answers 409 until a document is uploaded or reset.
        if os.path.isfile(RAG_DB_PATH):
            logging.info("Starting RAG database service")
            vectodb_proc = start_service()
        else:
            logging.info("No RAG database at %s; upload a document or reset to start the service", RAG_DB_PATH)
    else:
        logging.info("RAG database service disabled")

    try:
        genai_app.run()
    finally:
        stop_service()
