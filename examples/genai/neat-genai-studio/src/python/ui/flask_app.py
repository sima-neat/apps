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
import base64
import ipaddress
import json
import logging
import math
import os
import requests
import shutil
import re
import sys
import threading
import queue
import time

# Flask imports
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, stream_with_context
from flask_socketio import SocketIO

from collections import deque
import wave
import traceback
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import atexit
import signal
import socket
import subprocess

from shared.config import HubConfig
from shared.board_camera import (
    capture_camera_frame,
    default_camera_device,
    list_camera_devices,
    normalize_camera_device,
)
from server import hub as hub_helpers
from tts_text import sanitize_for_tts   # Markdown/LaTeX → speakable text (stdlib only)
from asr_metadata import (
    DEFAULT_LOGPROB_THRESHOLD,
    DEFAULT_NO_SPEECH_THRESHOLD,
    analyze_transcription,
    normalize_language_code,
)
from voice_catalog import (
    asset_paths as catalog_asset_paths,
    catalog_voices,
    install_voice as install_catalog_voice,
    installed_voices,
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
    """Best-effort version info for the About/version modal: git commit, branch and
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
    """PID of the run.sh supervisor so the GUI can trigger a graceful shutdown.
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
ttfs = 0

vectodb_proc = None
rag_db_client = None
RAG_DB_PATH = None
start_service = None
create_markdown_vectordb = None
validate_rag_database = None
wait_for_rag_service = None


def save_image_upload_as_base64(image_file, upload_folder):
    filename = secure_filename(image_file.filename or "")
    if not filename:
        raise ValueError("Invalid image file name")

    upload_dir = Path(upload_folder).resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = (upload_dir / filename).resolve()
    # Raises ValueError if filename resolution escapes upload_dir.
    image_path.relative_to(upload_dir)

    image_file.save(str(image_path))
    with image_path.open('rb') as img_file:
        return image_path, base64.b64encode(img_file.read()).decode('utf-8')


def ensure_rag_modules_loaded():
    global rag_db_client
    global RAG_DB_PATH
    global start_service
    global create_markdown_vectordb
    global validate_rag_database
    global wait_for_rag_service

    if rag_db_client is not None:
        return rag_db_client

    from rag.vectordb import (
        RAG_DB_PATH as _RAG_DB_PATH,
        start_service as _start_service,
        validate_database as _validate_database,
        wait_for_service as _wait_for_service,
        RagDbClient,
    )
    from rag.create_db import create_markdown_vectordb as _create_markdown_vectordb

    RAG_DB_PATH = _RAG_DB_PATH
    start_service = _start_service
    create_markdown_vectordb = _create_markdown_vectordb
    validate_rag_database = _validate_database
    wait_for_rag_service = _wait_for_service
    rag_db_client = RagDbClient()
    return rag_db_client

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

def handle_shutdown_signal(signum, _frame):
    logging.info("Received signal %s; stopping RAG database service", signum)
    stop_service()
    raise KeyboardInterrupt

atexit.register(stop_service)
signal.signal(signal.SIGTERM, handle_shutdown_signal)

class AppConstants:
    DEFAULT_SIMA_SERVER_IP = "127.0.0.1:9998"
    DEFAULT_CAMERA_IDX = 0
    DEFAULT_MODEL_QUERY_STR='Describe what you see in the picture.'
    DEFAULT_HTTP_PORT = 8081
    DEFAULT_UPLOADS_DIR = 'uploads'


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, default))
        if not math.isfinite(value):
            raise ValueError
        return value
    except (TypeError, ValueError):
        logging.warning("Invalid %s; using %s", name, default)
        return default

class TalkController:
    def __init__(self, supported_langs=None):
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        self.pipers = {}            # rhasspy piper-tts voices, keyed by language
        self.pp = None              # active piper-plus multilingual engine
        self.pp_models = []         # installed piper-plus voices (selectable)
        self.pp_current = None      # key of the active piper-plus voice
        self.pp_lock = threading.Lock()   # guards runtime piper-plus voice switches
        self.prefer_piper_plus = False  # default engine: dedicated piper-tts voices
        self.browser_tts = False    # when True, no server synthesis — the client speaks via Web Speech
        self.utterance_speed = 1.0
        self.missing_voice_warnings = set()
        # Load Piper Plus first so the dedicated voice loader can report which
        # allowlisted languages already have a fallback.
        self._init_piper_plus()
        self._init_pipers_threaded(supported_langs or ['en'])
        
        self.lock = threading.Lock()
        self._reset_tps_counters()

        self.current_language = 'en'
        self.supported_langs = supported_langs or ['en']
        # When False, spoken responses (PiperTTS) are skipped entirely — no
        # synthesis compute and no audio_chunk emitted.
        self.tts_enabled = True
        # Create a thread-safe queue for incoming subwords
        self.queue = queue.Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.full_response = []

        # Chunking state
        self.chunk_count = 0
        self.permissive_chunks = 3  # first N chunks use permissive boundaries
        self.min_chars_first_chunks = 20  # require at least this many chars before first N flushes

    def set_language(self, lang):
        self.current_language = lang if lang in self.supported_langs else 'xx'

    def set_tts_enabled(self, enabled):
        self.tts_enabled = bool(enabled)

    def set_utterance_speed(self, speed):
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        self.utterance_speed = max(0.5, min(2.0, speed))
        for eng in list(self.pipers.values()) + [self.pp]:
            if eng is not None and hasattr(eng, 'set_utterance_speed'):
                eng.set_utterance_speed(self.utterance_speed)

    def _init_pipers_threaded(self, langs):
        threads = []
        assets = Path("assets")

        def load_piper(lang_code, onnx_file):
            from pipertts import PiperTTS # import here if needed locally
            try:
                piper = PiperTTS(model_path=str(onnx_file))
                piper.set_utterance_speed(self.utterance_speed)
                self.pipers[lang_code] = piper
                logging.info("Loaded Piper model for %s (streaming)", lang_code)
            except Exception as e:
                logging.warning(f"Failed to load Piper model for {lang_code}: {e}")

        for lang in langs:
            found = False
            voices = installed_voices(
                VOICE_CATALOG, assets_path=assets, engine="piper-tts", language=lang
            )
            voices.sort(key=lambda voice: not voice.get("default", False))
            if voices:
                model_path = next(
                    path for path in catalog_asset_paths(voices[0], assets)
                    if path.suffix == ".onnx"
                )
                thread = threading.Thread(target=load_piper, args=(lang, model_path))
                thread.start()
                threads.append(thread)
                found = True
            if not found:
                if self.pp is not None and self.pp.supports(lang):
                    logging.info("No dedicated voice installed for '%s' — using Piper Plus.", lang)
                elif lang == 'ko':
                    logging.info("No Korean server voice — browser/text only.")
                else:
                    logging.warning("No server TTS voice installed for language: %s", lang)

        for t in threads:
            t.join()

    def _init_piper_plus(self):
        """Load only installed Piper Plus models from the reviewed catalog."""
        assets = Path("assets")
        voices = installed_voices(
            VOICE_CATALOG, assets_path=assets, engine="piper-plus"
        )
        voices.sort(key=lambda voice: not voice.get("default", False))
        self.pp_models = []
        for voice in voices:
            paths = catalog_asset_paths(voice, assets)
            onnx = next(path for path in paths if path.suffix == ".onnx")
            cfg = next((path for path in paths if path.name == "config.json"), None)
            self.pp_models.append({
                "key": voice["id"],
                "label": voice["label"],
                "onnx": onnx,
                "config": cfg,
                "license": voice["license"],
            })

        self.pp = None
        self.pp_current = None
        if not self.pp_models:
            logging.info("No piper-plus voice found under assets/piper-plus/ — "
                         "multilingual alternative unavailable (run voice_install.sh).")
            return
        self._load_piper_plus(self.pp_models[0]["key"])

    def _load_piper_plus(self, key):
        """Load the piper-plus voice `key` into self.pp. Returns True on success."""
        entry = next((m for m in getattr(self, "pp_models", []) if m["key"] == key), None)
        if entry is None:
            return False
        try:
            from piperplus_tts import PiperPlusTTS
            pp = PiperPlusTTS(entry["onnx"], config_path=entry["config"])
            pp.set_utterance_speed(self.utterance_speed)
            self.pp = pp
            self.pp_current = key
            logging.info("piper-plus voice '%s' ready (languages: %s)",
                         key, sorted(pp.languages))
            return True
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to load piper-plus voice '%s': %s", key, e)
            return False

    def set_piper_plus_voice(self, key):
        """Install a catalogued model when needed and switch to it."""
        with self.pp_lock:
            catalog_voice = voice_by_id(key, VOICE_CATALOG)
            if catalog_voice is None or catalog_voice["engine"] != "piper-plus":
                return False
            try:
                # The installer also verifies existing files against the pinned
                # checksums, so a corrupt or manually replaced model is repaired.
                install_catalog_voice(catalog_voice, assets_path=Path("assets"))
                self._init_piper_plus()
                return self._load_piper_plus(key)
            except Exception:  # noqa: BLE001
                logging.exception("Could not install/load Piper Plus voice '%s'", key)
                return False

    def piper_plus_voices(self):
        """Available piper-plus voices + the current one, for the UI picker."""
        installed = {model["key"] for model in getattr(self, "pp_models", [])}
        return {
            "voices": [
                {
                    "key": voice["id"],
                    "label": voice["label"],
                    "installed": voice["id"] in installed,
                }
                for voice in catalog_voices(VOICE_CATALOG, engine="piper-plus")
            ],
            "current": getattr(self, "pp_current", None),
        }

    def set_voice_engine(self, engine):
        """Choose the preferred TTS engine for languages both can speak.
        'piper-plus' -> prefer piper-plus; 'piper-tts' -> prefer rhasspy piper.
        Languages only one engine supports are unaffected."""
        engine = (engine or "").strip().lower()
        if engine in ("browser", "web", "webspeech", "web-speech"):
            self.browser_tts = True
        elif engine in ("piper-plus", "piperplus", "pp"):
            self.browser_tts = False
            self.prefer_piper_plus = True
        elif engine in ("piper-tts", "pipertts", "piper", "rhasspy"):
            self.browser_tts = False
            self.prefer_piper_plus = False
        else:
            return False
        logging.info("Preferred TTS engine set to %s",
                     "browser" if self.browser_tts
                     else ("piper-plus" if self.prefer_piper_plus else "piper-tts"))
        return True

    def voice_engine(self, language=None):
        """Engines that can speak `language` (all loaded engines when language is
        None), plus the one actually used for it. Only engines that support the
        language are returned, so the UI never offers an incompatible engine."""
        engines = []
        pp_ok = self.pp is not None and (language is None or self.pp.supports(language))
        tts_ok = bool(self.pipers) and (language is None or language in self.pipers)
        if pp_ok:
            engines.append({"key": "piper-plus", "label": "piper-plus (multilingual)"})
        if tts_ok:
            engines.append({"key": "piper-tts", "label": "piper-tts (rhasspy voices)"})
        # Browser TTS runs entirely in the client via the Web Speech API, so it is
        # always available and works for any language the device has a voice for.
        engines.append({"key": "browser", "label": "Browser (device voices)"})
        # The engine the router actually uses for this language.
        if self.browser_tts:
            current = "browser"
        else:
            current = "piper-plus" if self.prefer_piper_plus else "piper-tts"
            if language:
                _, eng = self._get_piper(language)
                if eng is not None and eng is self.pp:
                    current = "piper-plus"
                elif eng is not None and eng in self.pipers.values():
                    current = "piper-tts"
        keys = [e["key"] for e in engines]
        if current not in keys:
            current = keys[0]
        return {"engines": engines, "current": current}

    def _get_piper(self, language):
        """Route a language to the best available TTS engine, returning
        ``(effective_language, engine)`` or ``(None, None)`` when nothing can
        speak it. Preference (with prefer_piper_plus): piper-plus for the
        languages it supports → a dedicated catalogued voice → English as a
        *latin-script only* fallback.
        CJK/Korean never fall back to an English voice — better silent than
        mispronounced."""
        if self.prefer_piper_plus and self.pp is not None and self.pp.supports(language):
            return language, self.pp
        if language in self.pipers:
            return language, self.pipers[language]
        if self.pp is not None and self.pp.supports(language):
            return language, self.pp
        if language not in ('ja', 'ko', 'zh') and 'en' in self.pipers:
            return 'en', self.pipers['en']
        return None, None

    def has_voice(self, language='en'):
        _, piper = self._get_piper(language)
        return piper is not None

    def _warn_missing_voice_once(self, language):
        if language in self.missing_voice_warnings:
            return
        self.missing_voice_warnings.add(language)
        logging.warning(f"No Piper voice loaded for language '{language}'. Skipping TTS audio.")

    def _reset_tps_counters(self):
        self.tps = 0.0
        self.tps_history = []
        self.max_tps_history = 10

    def _worker_loop(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.05)
                if item == "__STOP__":
                    logging.info("TTS worker loop stopping.")
                    break
                try:
                    self._process_subword(item)
                except Exception as e:
                    logging.error(f"Error in _process_subword: {e}")
                    logging.debug(traceback.format_exc())
            except queue.Empty:
                continue
            except Exception as e:
                # Catch anything unexpected (e.g., bad queue state)
                logging.error(f"Unexpected error in TTS worker loop: {e}")
                logging.debug(traceback.format_exc())

    def enqueue(self, data):
        """
        Enqueue incoming data containing the subword and tokens-per-second (tps).
        Smooths tps over the last N samples.
        Expected data format: {'text': subword, 'tokens_per_sec': tps}
        """
        subword = data.get('text', '')
        new_tps = data.get('tokens_per_sec', 0.0)
        generation_id = data.get('generation_id')

        subword = subword.replace('*', '').replace('＊', '')

        self.queue.put({
            'text': subword,
            'generation_id': generation_id,
        })

        # Update TPS history and compute average
        self.tps_history.append(new_tps)
        if len(self.tps_history) > self.max_tps_history:
            self.tps_history.pop(0)

        # Compute the average TPS
        if self.tps_history:
            self.tps = sum(self.tps_history) / len(self.tps_history)
            logging.debug(f'Current average TPS is {self.tps}')
        else:
            self.tps = 0.0

    def stop(self):
        self.running = False
        self.queue.put("__STOP__")
        self.worker_thread.join()

    def _get_wav_duration(self, buffer):
        logging.info("Calculating WAV duration")
        buffer.seek(0)
        with wave.open(buffer, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        
        logging.info(f'WAV duration: {duration:.2f} seconds')
        return duration

    def _flush_talk(self, generation_id, force=False):
        """Synthesize buffered text, bypassing the early-chunk minimum at END."""
        if force and self._next is not None:
            self.talk.append(self._next)
            self._next = None
        if not self.talk:
            return

        self.totalk = self.generate_talk()
        start_time = time.time()
        sanitized_sentence = self._sanitize_for_tts(self.totalk)
        logging.info(f"{self.full_response} : {sanitized_sentence}")

        if (
            not force
            and self.tts_enabled
            and self.current_language != 'xx'
            and self.chunk_count < self.permissive_chunks
            and len(sanitized_sentence) < self.min_chars_first_chunks
        ):
            # Keep the entire fragment in order until more text arrives. When
            # END arrives, force=True flushes it regardless of its length.
            if self._next is not None:
                self.talk.append(self._next)
                self._next = None
            return

        if not sanitized_sentence.strip():
            pass
        elif sanitized_sentence in self.full_response:
            logging.info(f"Skipping synthesis for repeated text: {self.totalk}")
            self._reset_tps_counters()
        elif self.tts_enabled and self.current_language != 'xx':
            if self.browser_tts:
                if generation_id is not None and not genai_app.is_generation_current(generation_id):
                    return
                self.full_response.append(sanitized_sentence)
                genai_app.emit('audio_chunk', {
                    'text': sanitized_sentence.strip(),
                    'browser': True,
                    'lang': self.current_language,
                    'tps': round(self.tps, 2),
                })
                self.chunk_count += 1
            else:
                _, piper = self._get_piper(self.current_language)
                if piper is None:
                    self._warn_missing_voice_once(self.current_language)
                else:
                    self.full_response.append(sanitized_sentence)
                    audio_duration = 0.0
                    elapsed_time = 0.0
                    if hasattr(piper, 'synthesize_stream'):
                        buffers = piper.synthesize_stream(
                            sanitized_sentence, language=self.current_language
                        )
                    else:
                        buffers = [piper.synthesize(
                            sanitized_sentence, language=self.current_language
                        )]
                    try:
                        for buffer in buffers:
                            if generation_id is not None and not genai_app.is_generation_current(generation_id):
                                return
                            elapsed_time = time.time() - start_time
                            audio_duration += self._get_wav_duration(buffer)
                            rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
                            genai_app.emit('audio_chunk', {
                                'text': sanitized_sentence.strip(),
                                'audio': buffer.getvalue(),
                                'tps': round(self.tps, 2),
                                'rtf': round(rtf, 2)
                            })
                    finally:
                        close = getattr(buffers, 'close', None)
                        if close is not None:
                            close()
                    logging.info(f"[Timing] self.piper.synthesize took {elapsed_time:.3f} seconds for [{sanitized_sentence}]")
                    self.chunk_count += 1

        self.talk = []
        if self._next is not None:
            self.talk.append(self._next)
            self._next = None

    def _process_subword(self, item):
        subword = item.get('text', '')
        generation_id = item.get('generation_id')
        if generation_id is not None and not genai_app.is_generation_current(generation_id):
            return

        mod_subword = ''

        if subword != 'END':  # End of streaming
            if '<0x0A>' in subword:
                mod_subword = re.sub(r"<0x([0-9A-Fa-f]+)>", "", subword)
            if '</s>' in subword:
                mod_subword = re.sub(r"</s>", "", subword)

            if self.check_punctuation(subword):
                if mod_subword:
                    # Split on both English and CJK punctuation marks
                    parts = re.split(r'([。！？\.?!])', mod_subword)
                    if len(parts) > 1:
                        tmp = parts[0] + (parts[1] if len(parts) > 1 else '')
                        self._next = ''.join(parts[2:]) if len(parts) > 2 else ''
                        self.talk.append(tmp)
                    else:
                        self.talk.append(mod_subword)
                else:
                    self.talk.append(subword)

                self._flush_talk(generation_id)
            else:
                self.talk.append(subword)
        else:
            # A response can end with a short sentence that did not satisfy the
            # initial chunk-size threshold. Never discard that final fragment.
            self._flush_talk(generation_id, force=True)
            avg_tps = (sum(self.tps_history) / len(self.tps_history)) if getattr(self, 'tps_history', None) else 0.0
            logging.info(f"[TTS] Average tokens/sec for answer: {avg_tps:.2f}")
            genai_app.emit('end', {})
            self.chunk_count = 0
            self._reset_tps_counters()
            if generation_id is not None:
                genai_app.complete_generation(generation_id)

    def reset(self):
        with self.lock:
            logging.info("TalkController reset called. Clearing state and queue.")
            self._next = None
            self.prefix = ''
            self.totalk = ''
            self.talk = []
            self.full_response = []

            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break

    def check_punctuation(self, word):
        # Dynamic boundaries: first N chunks permissive, then strict
        if self.chunk_count < self.permissive_chunks:
            return bool(re.search(r"[。！？\.,?!:;]", word))
        else:
            return bool(re.search(r"[。！？\.?!]", word))

    def generate_talk(self):
        # Simply join tokens as-is, preserving the model's intended spacing
        return "".join(self.talk)

    def _sanitize_for_tts(self, text: str) -> str:
        # Strip Markdown and LaTeX so Piper utters the prose, not the formatting
        # ("star star", "dollar x caret 2", raw URLs). See ui/tts_text.py.
        return sanitize_for_tts(text)

    
    def tts_on_demand(self, text, language='en'):
        """
        Perform TTS synthesis on-demand for the given text and return audio bytes and timing info.
        """
        if not text:
            raise ValueError("No text provided for TTS synthesis.")

        start_time = time.time()
        language, piper = self._get_piper(language)
        if piper is None:
            raise RuntimeError("No Piper TTS voice model is loaded. Install Piper .onnx voice assets under assets/.")

        sanitized_text = self._sanitize_for_tts(text)
        buffer = piper.synthesize(sanitized_text, language=language)
        elapsed_time = time.time() - start_time
        audio_duration = self._get_wav_duration(buffer)
        rtf = elapsed_time / audio_duration if audio_duration > 0 else 0

        logging.info(f"[On-Demand TTS] Synthesized audio in {elapsed_time:.3f} sec (RTF: {rtf:.2f})")

        return {
            'audio_bytes': buffer.getvalue(),
            'elapsed_time': elapsed_time,
            'audio_duration': audio_duration,
            'rtf': rtf
        }

class AppContext:
    def __init__(self):
        self.app = None
        self.socketio = None
        self.current_voice_by_lang = {}
        self.talk_ctrl = None
        self.llm_only = False
        self.system_prompt = None
        self.model_display_name = ""
        self.chat_model_name = "model"
        self.chat_model_names = ("model",)
        self.chat_model_capabilities = {"model": {"supportsVision": True, "imageSize": None}}
        self.active_chat_model_name = "model"
        self.asr_model_name = ""
        self.max_tokens = None
        self.web_host = "0.0.0.0"
        self.web_port = 5000
        self.rag_enabled = True
        self.rag_embedding_model_dir = ""
        self.vision_image_size = None

        # Runtime model management + appearance
        self.control_base_url = "http://127.0.0.1:9997"
        self.catalog_dir = None
        self.hub_config = HubConfig(allow_download=True, orgs=("simaai", "TDoSiMa"))
        self.ui_font_family = "Inter"
        self.ui_font_size = 15
        self._catalog_names_cache = (0.0, frozenset())
        self._asr_name_cache = (0.0, "")

        # Conversation history for OpenAI-style chat
        self.conversation_history = []
        self.current_response = ""  # Accumulate raw LLM response during streaming
        self._state_lock = threading.RLock()
        self._generation_counter = 0
        self._active_generation_id = None

    def _build_system_prompt_message(self):
        if not self.system_prompt:
            return None

        if self.llm_only:
            content = self.system_prompt
        else:
            content = [{"type": "text", "text": self.system_prompt}]
        return {"role": "system", "content": content}

    def set_system_prompt(self, prompt=None):
        normalized = (prompt or '').strip()
        self.system_prompt = normalized if normalized else None
        # Update or remove the system message in-place
        existing_index = next(
            (idx for idx, message in enumerate(self.conversation_history) if message.get('role') == 'system'),
            None
        )
        system_message = self._build_system_prompt_message()
        if system_message:
            if existing_index is not None:
                self.conversation_history[existing_index] = system_message
            else:
                self.conversation_history.insert(0, system_message)
        elif existing_index is not None:
            self.conversation_history.pop(existing_index)

        if self.system_prompt:
            logging.info("System prompt updated.")
        else:
            logging.info("System prompt cleared.")

    def get_system_prompt(self):
        return self.system_prompt or ""

    def add_user_message(self, content, has_image=False, image_base64=None):
        """Add a user message to conversation history, with optional image support."""
        if has_image and image_base64:
            # Construct the full data URI required by the server's backend.
            # This assumes a default MIME type of "image/jpeg".
            image_data_uri = f"data:image/jpeg;base64,{image_base64}"
            
            # Multi-modal format with the correctly formatted image data
            message_content = [
                {"type": "image", "image": image_data_uri},
                {"type": "text", "text": content}
            ]
        else:
            # Simple text format
            if self.llm_only:
                message_content = content
            else:
                message_content = [{"type": "text", "text": content}]

        with self._state_lock:
            self.conversation_history.append({"role": "user", "content": message_content})
            total_messages = len(self.conversation_history)
        logging.info(
            f"Added user message (image: {has_image}). "
            f"Total messages: {total_messages}"
        )


    def _append_current_response_to_history_locked(self):
        if not self.current_response:
            return False

        # Store only the answer in history — the model's <think> reasoning should
        # not be fed back on later turns (it bloats context and, for reasoning
        # models like Qwen3, degrades quality).
        stored = _answer_part(self.current_response).strip() or self.current_response
        if self.llm_only:
            assistant_content = stored
        else:
            assistant_content = [{"type": "text", "text": stored}]
        self.conversation_history.append({"role": "assistant", "content": assistant_content})
        self.current_response = ""
        logging.info(f"Added assistant response to history. Total messages: {len(self.conversation_history)}")
        return True

    def start_assistant_response(self, generation_id):
        """Start accumulating a new assistant response."""
        with self._state_lock:
            if self._active_generation_id != generation_id:
                return False
            self.current_response = ""
            return True

    def add_to_current_response(self, generation_id, text):
        """Add text to the current assistant response being accumulated."""
        with self._state_lock:
            if self._active_generation_id != generation_id:
                return False
            self.current_response += text
            return True

    def finish_assistant_response(self, generation_id):
        """Finish the current assistant response and add it to history."""
        with self._state_lock:
            if self._active_generation_id != generation_id:
                return False
            self._append_current_response_to_history_locked()
            return True

    def complete_generation(self, generation_id):
        with self._state_lock:
            if self._active_generation_id != generation_id:
                return False
            self._active_generation_id = None
            return True

    def fail_generation(self, generation_id, message):
        """Terminate one failed generation and notify the browser exactly once."""
        with self._state_lock:
            if self._active_generation_id != generation_id:
                return False
            self._append_current_response_to_history_locked()
            self._active_generation_id = None
        if self.talk_ctrl is not None:
            self.talk_ctrl.reset()
        self.emit('generation_error', {'message': message})
        self.emit('end', {})
        return True

    def interrupt_active_generation(self, preserve_partial):
        with self._state_lock:
            had_active_generation = self._active_generation_id is not None
            if preserve_partial:
                self._append_current_response_to_history_locked()
            self.current_response = ""
            self._active_generation_id = None
            return had_active_generation

    def begin_generation(self, chat_model_name=None):
        with self._state_lock:
            self._generation_counter += 1
            generation_id = self._generation_counter
            self._active_generation_id = generation_id
            if chat_model_name:
                self.active_chat_model_name = chat_model_name
            self.current_response = ""
            return generation_id

    def is_generation_current(self, generation_id):
        with self._state_lock:
            return self._active_generation_id == generation_id

    def clear_conversation_history(self):
        """Clear the conversation history."""
        with self._state_lock:
            self.conversation_history = []
            self.current_response = ""
            self._active_generation_id = None
        logging.info("Conversation history cleared")
        # Reinsert system prompt if present
        system_message = self._build_system_prompt_message()
        if system_message:
            with self._state_lock:
                self.conversation_history.insert(0, system_message)

    def _known_catalog_names(self):
        """Names the control API currently reports as available (short cache)."""
        cached_at, names = self._catalog_names_cache
        if (time.monotonic() - cached_at) < 3 and names:
            return names
        try:
            resp = requests.get(
                f"{self.control_base_url.rstrip('/')}/control/status", timeout=5
            )
            data = resp.json()
            names = frozenset(entry.get("name", "") for entry in data.get("catalog", []))
        except Exception:
            names = frozenset()
        self._catalog_names_cache = (time.monotonic(), names)
        return names

    def set_asr_model_name(self, name):
        """Point transcription at a new ASR model after a successful switch."""
        self.asr_model_name = (name or "").strip()
        self._asr_name_cache = (time.monotonic(), self.asr_model_name)
        if self.app is not None:
            self.app.config['ASR_MODEL_NAME'] = self.asr_model_name
        return self.asr_model_name

    def resolve_asr_model(self):
        """The ASR model serving transcriptions, asking the control API if the
        UI has not been told one yet (short cache, like the catalog names)."""
        if self.asr_model_name:
            return self.asr_model_name
        cached_at, cached = self._asr_name_cache
        if (time.monotonic() - cached_at) < 5:
            return cached
        try:
            resp = requests.get(
                f"{self.control_base_url.rstrip('/')}/control/status", timeout=5
            )
            name = str((resp.json() or {}).get("asrModel") or "")
        except Exception:
            name = ""
        self._asr_name_cache = (time.monotonic(), name)
        return name

    def resolve_chat_model(self, requested_model=None):
        model_name = str(requested_model or self.chat_model_name).strip()
        if not model_name:
            raise ValueError("No model is loaded. Select or download a model to start chatting.")
        if model_name in self.chat_model_names:
            return model_name
        if model_name in self._known_catalog_names():
            return model_name
        raise ValueError(f"Unknown chat model: {model_name}")

    def get_conversation_history(self):
        """Get the current conversation history."""
        with self._state_lock:
            return self.conversation_history.copy()


    def update_settings(self, camidx, model_server_ip, ragserver, httponly, apionly, llm_only, model_name=None, vision_image_size=None):
        self.camidx = AppConstants.DEFAULT_CAMERA_IDX if camidx is None else camidx
        self.model_server_ip = AppConstants.DEFAULT_SIMA_SERVER_IP if model_server_ip is None else model_server_ip
        self.ragserver = ragserver
        self.httponly = httponly
        self.apionly = apionly
        self.llm_only = llm_only
        self.model_display_name = model_name or ""
        self.vision_image_size = vision_image_size
        self.update_config()
        # Align system prompt format with current mode
        if self.system_prompt:
            self.set_system_prompt(self.system_prompt)

        if not self.apionly:
            self.talk_ctrl = TalkController([
                'en', 'fr', 'es', 'de', 'it', 'zh', 'vi', 'ja', 'pt', 'ko', 'no'
            ])

    def update_from_config(self, app_cfg):
        model_server = f"{app_cfg.openai.client_host}:{app_cfg.openai.port}"
        self.chat_model_names = tuple(model.name for model in app_cfg.chat_models)
        self.chat_model_capabilities = {
            model.name: {
                "supportsVision": bool(model.supports_vision),
                "imageSize": model.vision_image_size,
            }
            for model in app_cfg.chat_models
        }
        # Models are optional at startup; the UI loads them on demand from the catalog.
        self.chat_model_name = self.chat_model_names[0] if self.chat_model_names else ""
        self.active_chat_model_name = self.chat_model_name
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
        self.ui_font_family = app_cfg.ui.font_family
        self.ui_font_size = app_cfg.ui.font_size
        self.update_settings(
            camidx=None,
            model_server_ip=model_server,
            ragserver=None,
            httponly=not app_cfg.web.https,
            apionly=False,
            llm_only=False,
            model_name=app_cfg.chat_model.name if app_cfg.chat_model else "",
            vision_image_size=None
        )
        if app_cfg.request.system_prompt:
            self.set_system_prompt(app_cfg.request.system_prompt)

    def update_config(self):
        self.app.config['CAMERA_IDX'] = self.camidx
        self.app.config['SIMAAI_IP_ADDR'] = self.model_server_ip
        self.app.config['SIMAAI_IP_PORT'] =  AppConstants.DEFAULT_HTTP_PORT
        self.app.config['UPLOAD_FOLDER'] = AppConstants.DEFAULT_UPLOADS_DIR
        self.app.config['MODEL_DISPLAY_NAME'] = self.model_display_name
        self.app.config['CHAT_MODEL_NAME'] = self.chat_model_name
        self.app.config['CHAT_MODEL_NAMES'] = list(self.chat_model_names)
        self.app.config['CHAT_MODEL_CAPABILITIES'] = self.chat_model_capabilities
        self.app.config['ASR_MODEL_NAME'] = self.asr_model_name
        self.app.config['MAX_TOKENS'] = self.max_tokens
        self.app.config['RAG_ENABLED'] = self.rag_enabled
        self.app.config['RAG_FILE_PROCESSING_URL'] = self.ragserver or ""
        self.app.config['RAG_EMBEDDING_MODEL_DIR'] = self.rag_embedding_model_dir
        self.app.config['VISION_IMAGE_SIZE'] = self.vision_image_size

    def get_config(self):
        return self.app.config
        
    def initialize(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)

        @self.app.before_request
        def _reject_cross_origin_mutation():
            """Keep browser mutations on the Studio's own origin.

            Headerless local API/CLI clients remain supported. Browsers provide
            Origin, Referer, or Sec-Fetch-Site for cross-origin requests.
            """
            if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
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

        # Cache-bust static assets: append each file's mtime to its URL so the
        # browser refetches newui.js / newui.css after an update instead of
        # serving a stale cached copy (the cause of "I don't see your change").
        @self.app.url_defaults
        def _static_cache_bust(endpoint, values):
            if endpoint == 'static' and 'filename' in values:
                try:
                    fp = os.path.join(self.app.static_folder, values['filename'])
                    values['v'] = int(os.path.getmtime(fp))
                except OSError:
                    pass

        if not os.path.exists(AppConstants.DEFAULT_UPLOADS_DIR):
            os.makedirs(AppConstants.DEFAULT_UPLOADS_DIR)

        # Note: setup_router() is now called after update_settings()

        # Setup config.js route early so it's always available
        @self.app.route('/config.js')
        def config_js():
            llm_only_val = 'true' if self.llm_only else 'false'
            height_val = ''
            width_val = ''
            if isinstance(self.vision_image_size, dict):
                height_val = str(self.vision_image_size.get('height', ''))
                width_val = str(self.vision_image_size.get('width', ''))
            # Use quotes around the values to make them strings in JavaScript
            hub_org = self.hub_config.org if self.hub_config else "simaai"
            js = (
                "window.SIMA_CONFIG=window.SIMA_CONFIG||{};"
                f"window.SIMA_CONFIG.llmOnly='{llm_only_val}';"
                f"window.SIMA_CONFIG.ragEnabled='{str(self.rag_enabled).lower()}';"
                f"window.SIMA_CONFIG.chatModels={json.dumps(list(self.chat_model_names))};"
                f"window.SIMA_CONFIG.defaultChatModel={json.dumps(self.chat_model_name)};"
                f"window.SIMA_CONFIG.chatModelCapabilities={json.dumps(self.chat_model_capabilities)};"
                f"window.SIMA_CONFIG.visionImageHeight='{height_val}';"
                f"window.SIMA_CONFIG.visionImageWidth='{width_val}';"
                "window.SIMA_CONFIG.controlEnabled='true';"
                f"window.SIMA_CONFIG.hubOrg={json.dumps(hub_org)};"
                f"window.SIMA_CONFIG.defaultFontFamily={json.dumps(self.ui_font_family)};"
                f"window.SIMA_CONFIG.defaultFontSize={json.dumps(self.ui_font_size)};"
                f"window.SIMA_CONFIG.defaultMaxTokens={json.dumps(self.max_tokens)};"
                f"window.SIMA_CONFIG.version={json.dumps(_studio_version())};"
            )
            return self.app.response_class(js, mimetype='application/javascript')

    def emit(self, ep, obj):
        self.socketio.emit(ep, obj)

    def run(self):
        if not self.httponly:
            ssl_context = _https_credentials()
            self.socketio.run(self.app, host=self.web_host, port=self.web_port,
                            ssl_context=ssl_context,
                            debug=False, allow_unsafe_werkzeug=True)
        else:
            self.socketio.run(self.app, host=self.web_host, port=self.web_port,
                            debug=False, allow_unsafe_werkzeug=True)
            
    def run_stop(self):
        logging.info('Stopping processing...')
        self.interrupt_active_generation(preserve_partial=True)

        try:
            success = post_stop_to_sima(self.active_chat_model_name)
            if not success:
                return jsonify({'status': 'error', 'message': 'Failed to send stop signal to SiMa.ai server.'}), 500
        except Exception as e:
            logging.error(f"Failed to post stop to SiMa.ai: {e}")
            return jsonify({'status': 'error', 'message': 'Exception while contacting SiMa.ai server.'}), 500

        try:
            if self.talk_ctrl != None:
                self.talk_ctrl.reset()
        
        except Exception as e:
            logging.error(f"Failed to reset talk controller: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to reset talk controller.'}), 500


    def setup_router(self):

        @self.app.route('/')
        def newui():
            self.socketio.emit('update', {"hello" : "world"})
            if not self.apionly:
                return render_template('newui.html',
                                     llm_only=self.llm_only,
                                     model_name=self.model_display_name)

            return render_template('apionly.html')

        @self.app.route('/showcase')
        def showcase():
            # Conference / booth "present mode": a marketing slideshow about SiMa
            # Modalix + LLiMa + Palette Neat. Its Launch buttons return to "/".
            return render_template('showcase.html')

        # SiMaSentry Solutions: vendored static harness suite (Mission Control
        # portal + Med/Safe/Sec verticals) under ui/harnesses/. Served as plain
        # files — the pages are self-contained and chat with the loaded model
        # through the same-origin /v1/chat/completions proxy below.
        harness_dir = str(APP_DIR / 'harnesses')

        @self.app.route('/solutions/')
        def solutions_portal():
            return send_from_directory(harness_dir, 'index.html')

        @self.app.route('/solutions/<path:filename>')
        def solutions_asset(filename):
            if filename.endswith('/'):
                filename += 'index.html'
            return send_from_directory(harness_dir, filename)

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
            if self.talk_ctrl is None or not self.talk_ctrl.set_voice_engine(engine):
                return jsonify({'status': 'error', 'error': 'unknown engine'}), 400
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
                    return jsonify({'error': 'TTS engine not initialized, start the app with --apionly disabled.'}), 503

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

                # Play a short confirmation line with the newly selected voice
                try:
                    confirm_text = "Successfully switched to this voice and fully loaded now."
                    # Prefer the controller's sanitizer if available
                    if hasattr(self.talk_ctrl, '_sanitize_for_tts'):
                        confirm_text = self.talk_ctrl._sanitize_for_tts(confirm_text)
                    buffer = new_voice.synthesize(confirm_text)
                    genai_app.emit('system_audio', {
                        'text': confirm_text,
                        'audio': buffer.getvalue()
                    })
                except Exception as ee:
                    logging.warning(f"Voice switch confirmation audio failed: {ee}")

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

        def _proxy_control(method, path, timeout, json_body=None):
            try:
                resp = requests.request(
                    method, _control_url(path), json=json_body, timeout=timeout
                )
                return Response(
                    resp.content, status=resp.status_code, mimetype='application/json'
                )
            except requests.RequestException as exc:
                logging.warning("Control API request failed: %s", exc)
                return jsonify({'error': 'control API unreachable'}), 502

        @self.app.route('/favicon.ico')
        def favicon():
            # Serve the Neat logo for bare /favicon.ico requests too.
            return self.app.send_static_file('icons/neat-logo.png')

        @self.app.route('/models/status', methods=['GET'])
        def models_status():
            return _proxy_control('GET', '/control/status', 10)

        @self.app.route('/models/catalog', methods=['GET'])
        def models_catalog():
            return _proxy_control('GET', '/control/catalog', 30)

        @self.app.route('/models/logs', methods=['GET'])
        def models_logs():
            after = request.args.get('after', '0')
            try:
                resp = requests.get(
                    _control_url('/control/logs'), params={'after': after}, timeout=10
                )
                return Response(resp.content, status=resp.status_code, mimetype='application/json')
            except requests.RequestException as exc:
                logging.warning("Control API logs request failed: %s", exc)
                return jsonify({'error': 'control API unreachable'}), 502

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

        @self.app.route('/models/load', methods=['POST'])
        def models_load():
            name = (request.get_json(silent=True) or {}).get('name', '')
            return _proxy_control('POST', '/control/load', 600, {'name': name})

        @self.app.route('/models/asr', methods=['POST'])
        def models_asr():
            # Not _proxy_control: the response body carries the name the server
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

        @self.app.route('/models/unload', methods=['POST'])
        def models_unload():
            name = (request.get_json(silent=True) or {}).get('name', '')
            return _proxy_control('POST', '/control/unload', 60, {'name': name})

        @self.app.route('/models/delete', methods=['POST'])
        def models_delete():
            name = (request.get_json(silent=True) or {}).get('name', '')
            return _proxy_control('POST', '/control/delete', 120, {'name': name})

        @self.app.route('/benchmark/run', methods=['POST'])
        def benchmark_run():
            body = request.get_json(silent=True) or {}
            return _proxy_control('POST', '/control/benchmark', 30, {
                'num_samples': body.get('num_samples', 5),
                'max_new_tokens': body.get('max_new_tokens', 128),
                'prompt': body.get('prompt', ''),
                'model': body.get('model', ''),
            })

        @self.app.route('/benchmark/status', methods=['GET'])
        def benchmark_status():
            return _proxy_control('GET', '/control/benchmark/status', 10)

        @self.app.route('/benchmark/stop', methods=['POST'])
        def benchmark_stop():
            return _proxy_control('POST', '/control/benchmark/stop', 10)

        @self.app.route('/shutdown', methods=['POST'])
        def shutdown_studio():
            # Stop the whole studio from the GUI. run.sh writes its PID (and
            # exports NEAT_RUN_PID); a SIGTERM to it runs run.sh's cleanup, which
            # gracefully stops both the UI and the model server — the same path as
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

        @self.app.route('/models/card', methods=['GET'])
        def models_card():
            name = request.args.get('name', '')
            try:
                resp = requests.get(
                    _control_url('/control/card'), params={'name': name}, timeout=15
                )
                return Response(resp.content, status=resp.status_code, mimetype='application/json')
            except requests.RequestException as exc:
                logging.warning("Control API model-card request failed: %s", exc)
                return jsonify({'error': 'control API unreachable'}), 502

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
            # Same-origin streaming proxy to the model server's OpenAI API,
            # used by the Solutions harnesses: the Studio page is HTTPS while
            # the model server is HTTP on :9998, so direct browser calls would
            # be blocked as mixed content (and lack CORS headers).
            # These generations bypass the Studio's conversation history and
            # TTS by design — the harness keeps its own history client-side.
            payload = request.get_json(silent=True) or {}
            if self.max_tokens:
                payload.setdefault('max_tokens', int(self.max_tokens))
            _normalize_openai_image_parts(payload)
            url = f"http://{self.app.config['SIMAAI_IP_ADDR']}/v1/chat/completions"
            try:
                upstream = requests.post(url, json=payload, stream=True, timeout=(10, 600))
            except requests.RequestException as exc:
                logging.error(f"Chat proxy: model server unreachable: {exc}")
                return jsonify({'error': 'model server unreachable'}), 502

            def relay():
                try:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                except GeneratorExit:
                    # Browser aborted (harness Stop button or iframe unloaded):
                    # free the accelerator instead of letting generation run out.
                    # Note /stop is per-model, so a concurrent Studio generation
                    # on the same model would also stop (single-user semantics).
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
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )

        @self.app.route('/stop', methods=['POST'])
        def stop_processing():
            logging.info('Received /stop request. Attempting to stop processing...')
            result = self.run_stop()
            if result is not None:
                return result
            return jsonify({'status': 'stopped'}), 200

        @self.app.route('/clear-history', methods=['POST'])
        def clear_history():
            logging.info('Received /clear-history request. Clearing conversation history...')
            self.clear_conversation_history()
            return jsonify({'status': 'history cleared'}), 200

        @self.app.route('/system-prompt', methods=['GET', 'POST'])
        def system_prompt():
            if request.method == 'GET':
                return jsonify({'system_prompt': self.get_system_prompt()})

            try:
                data = request.get_json(silent=True) or {}
                prompt = data.get('system_prompt', '')
                self.set_system_prompt(prompt)
                self.clear_conversation_history()
                return jsonify({'system_prompt': self.get_system_prompt()})
            except Exception as e:
                logging.error(f"System prompt update failed: {e}")
                return jsonify({'error': 'Failed to update system prompt'}), 500
            
        @self.app.route('/upload', methods=['POST'])
        def upload():
            global ttfs
            audio_file = None
            image_file = None
            elapsed_time = 0
            query_str = ''
            result = None
            asr = None

            # Handle file uploads
            if 'audio_data' in request.files:
                audio_file = request.files['audio_data']
            if 'image_data' in request.files:
                image_file = request.files['image_data']

            # Handle form fields
            language = request.form.get('language', 'auto')
            # For typed prompts while ASR is in auto mode, keep using the most
            # recently detected UI language (English before the first recording).
            response_language = normalize_language_code(
                request.form.get('responseLanguage', '')
            ) or 'en'
            utterance_speed = request.form.get('utteranceSpeed', '1.0')
            chat = request.form.get('textchat', '').strip()
            search_rag = request.form.get('searchRag', 'false').lower() == 'true'
            include_chat_history = request.form.get('includeChatHistory', 'true').lower() == 'true'
            enable_tts = request.form.get('enableTts', 'true').lower() == 'true'
            use_board_camera = request.form.get('useBoardCamera', 'false').lower() == 'true'
            board_camera_device = request.form.get('boardCameraDevice', '').strip()
            gen_params = _read_gen_params(request.form)
            cfg = genai_app.get_config()
            try:
                selected_model = self.resolve_chat_model(request.form.get('chatModel'))
            except ValueError:
                return jsonify({'error': 'Invalid chat model'}), 400

            self.talk_ctrl.reset()
            initial_tts_language = (
                response_language
                if str(language).strip().lower() == 'auto'
                else normalize_language_code(language)
            )
            self.talk_ctrl.set_language(initial_tts_language)
            # Set before any tokens are enqueued so the worker never synthesizes.
            self.talk_ctrl.set_tts_enabled(enable_tts)
            self.talk_ctrl.set_utterance_speed(utterance_speed)
            had_active_generation = self.interrupt_active_generation(
                preserve_partial=include_chat_history
            )
            if had_active_generation:
                try:
                    post_stop_to_sima(self.active_chat_model_name)
                except Exception as e:
                    logging.error(f"Failed to stop previous generation before new request: {e}")

            # Save image if available and convert to base64
            image_path = None
            image_base64 = None
            audio_path = None

            if image_file:
                try:
                    image_path, image_base64 = save_image_upload_as_base64(image_file, cfg['UPLOAD_FOLDER'])
                except ValueError:
                    return jsonify({'error': 'Invalid image file name'}), 400
            elif use_board_camera and not self.llm_only:
                # Grab a fresh frame from the camera attached to the devkit
                # board itself (not the browser's webcam) for this prompt.
                try:
                    node = normalize_camera_device(board_camera_device) if board_camera_device else None
                    jpeg, tool = capture_camera_frame(node)
                except ValueError:
                    return jsonify({'error': 'Invalid board camera device'}), 400
                except RuntimeError as exc:
                    logging.error(f"Board camera capture failed: {exc}")
                    return jsonify({'error': 'Board camera capture failed'}), 503
                image_base64 = base64.b64encode(jpeg).decode('utf-8')
                logging.info(f"Attached board-camera frame (via {tool})")

            # If textchat is present, use it directly
            if chat:
                query_str = chat
            elif audio_file:
                try:
                    # Use MLA Backend transcription - read directly from FileStorage
                    start_time = time.time()
                    audio_bytes = audio_file.read()
                    result = post_audio_to_mla(audio_bytes, language=language)

                    if result:
                        elapsed_time = round(time.time() - start_time, 2)
                        asr = analyze_transcription(
                            result,
                            requested_language=language,
                            supported_tts_languages=self.talk_ctrl.supported_langs,
                            no_speech_threshold=_env_float(
                                'ASR_NO_SPEECH_THRESHOLD', DEFAULT_NO_SPEECH_THRESHOLD
                            ),
                            logprob_threshold=_env_float(
                                'ASR_LOGPROB_THRESHOLD', DEFAULT_LOGPROB_THRESHOLD
                            ),
                        )
                        query_str = asr['text']
                        logging.info(
                            "ASR result text=%r requested_language=%s detected_language=%s "
                            "resolved_language=%s tts_language=%s no_speech_prob=%s "
                            "avg_logprob=%s ignored=%s",
                            query_str,
                            language,
                            asr['language'] if asr['language_detected'] else None,
                            asr['language'],
                            asr['tts_language'],
                            asr['no_speech_prob'],
                            asr['avg_logprob'],
                            asr['reason'],
                        )

                        # Route this response through the voice matching the
                        # detected language. Unsupported languages intentionally
                        # produce text only instead of using a mismatched voice.
                        self.talk_ctrl.set_language(asr['tts_language'] or 'xx')

                        if asr['ignored']:
                            message = (
                                'No clear speech detected. Please try again.'
                                if asr['reason'] in ('no_speech', 'low_confidence')
                                else 'No transcription was produced. Please try again.'
                            )
                            return jsonify({
                                'ignored': True,
                                'message': message,
                                'question': '',
                                'ttt': elapsed_time,
                                'asr': asr,
                                'rag_used': False,
                                'rag_hits': 0,
                            })

                        self.socketio.emit('transcription', asr)
                    else:
                        logging.warning("Backend transcription returned no result")
                        return jsonify({
                            'error': 'Speech transcription is unavailable. Please try again.'
                        }), 503
                except Exception as e:
                    logging.exception("Speech transcription failed: %s", e)
                    return jsonify({
                        'error': 'Speech transcription is unavailable. Please try again.'
                    }), 503

            # Handle polite endings
            if ('Thank you' in query_str) or ('Thanks' in query_str):
                query_str = AppConstants.DEFAULT_MODEL_QUERY_STR

            full_query_str = query_str
            rag_used = False
            rag_hits = 0

            # if we have the query and search_rag flag is on, search the rag database
            if search_rag and self.rag_enabled and len(full_query_str) > 0:
                ensure_rag_modules_loaded()
                rag = rag_db_client.search(query_str)
                rag_used = True
                rag_hits = len(rag)
                full_query_str = f"{query_str}\n\nThe context is:\n\n{[entry['content'] for entry in rag]}"

            # Add user message to conversation history (detect if image is present)
            has_image = image_base64 is not None
            self.add_user_message(full_query_str, has_image=has_image, image_base64=image_base64)
            generation_id = self.begin_generation(selected_model)

            ttfs = 0
            logging.info(f"Query string: {full_query_str}")

            # Build messages list based on includeChatHistory flag
            if include_chat_history:
                # Pass full conversation history
                conversation_history = self.get_conversation_history()
                logging.info(f"Using full conversation history ({len(conversation_history)} messages)")
            else:
                # Single-shot mode: only current message + optional system prompt
                conversation_history = []
                system_message = self._build_system_prompt_message()
                if system_message:
                    conversation_history.append(system_message)
                # Add only the current user message (last one added)
                current_messages = self.get_conversation_history()
                if current_messages:
                    # Find the last user message
                    for msg in reversed(current_messages):
                        if msg.get('role') == 'user':
                            conversation_history.append(msg)
                            break
                logging.info(f"Using single-shot mode (history disabled, {len(conversation_history)} messages)")

            thread = threading.Thread(
                target=stream_chat_request,
                args=[conversation_history, selected_model, cfg, generation_id],
                kwargs={'gen_params': gen_params}
            )
            thread.start()

            return jsonify({
                'question': query_str,
                'ttt': elapsed_time,
                'asr': asr,
                'rag_used': rag_used,
                'rag_hits': rag_hits,
            })

        @self.app.route('/raghealth', methods=['GET'])
        def check_rag_server():
            if not self.rag_enabled:
                return {
                    "rag_db": "disabled",
                    "message": "RAG is disabled"
                }, 200

            try:
                client = ensure_rag_modules_loaded()
                rag_db_status = client.is_server_up()
            except Exception as e:
                logging.error(f"Exception checking RAG DB server: {e}")
                rag_db_status = False

            if rag_db_status:
                return {
                    "rag_db": "ok",
                    "message": "RAG DB is available"
                }, 200
            else:
                return {
                    "rag_db": "unavailable",
                    "message": "RAG DB is not available"
                }, 503

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
                client = ensure_rag_modules_loaded()
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
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'Missing JSON payload.'}), 400

                if self.talk_ctrl == None:
                    return jsonify({'error': 'TTS engine not initialized, start the app with --apionly disabled.'}), 500

                text = data.get('input')
                model = data.get('model', 'piper-tts')
                voice = data.get('voice', 'default')
                language = data.get('language', 'en')
                utterance_speed = data.get('utterance_speed', data.get('utteranceSpeed', 1.0))
                response_format = data.get('response_format', 'wav')

                if not text:
                    return jsonify({'error': 'Missing "input" text field.'}), 400
                if not self.talk_ctrl.has_voice(language):
                    return jsonify({
                        'error': 'No Piper TTS voice model is loaded. Install Piper .onnx voice assets under assets/.'
                    }), 503

                logging.info(f"Received TTS request: model={model}, voice={voice}, format={response_format}")

                self.talk_ctrl.set_utterance_speed(utterance_speed)
                result = self.talk_ctrl.tts_on_demand(text, language=language)

                # Default to WAV for Piper. If mp3 is requested, you need ffmpeg to convert.
                audio_bytes = result['audio_bytes']
                content_type = 'audio/wav'
                filename = 'output.wav'

                # Optional: Add MP3 conversion logic if needed here

                return Response(
                    audio_bytes,
                    mimetype=content_type,
                    headers={
                        'Content-Disposition': f'attachment; filename="{filename}"',
                        'X-RTF': str(result['rtf']),
                        'X-Audio-Duration': str(result['audio_duration']),
                        'X-Elapsed-Time': str(result['elapsed_time'])
                    }
                )

            except Exception:
                logging.exception("TTS endpoint error")
                return jsonify({'error': 'TTS request failed'}), 500

        # Board camera: a /dev/video* device plugged into the devkit board
        # itself, as opposed to the in-browser camera (the *client's* webcam).
        # These are plain hardware reads, so they stay registered even in
        # LLM-only mode; vision gating applies where a frame enters a chat.
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
                node = normalize_camera_device(device) if device else None
            except ValueError:
                return jsonify({'error': 'Invalid board camera device'}), 400
            try:
                jpeg, tool = capture_camera_frame(node)
            except RuntimeError as exc:
                logging.error(f"Board camera capture failed: {exc}")
                return jsonify({'error': 'Board camera capture failed'}), 503
            return Response(jpeg, mimetype='image/jpeg',
                            headers={'Cache-Control': 'no-store', 'X-Capture-Tool': tool})

        # Only register image upload route if not in LLM-only mode
        if not self.llm_only:
            @self.app.route('/upload_image', methods=['POST'])
            def upload_image():
                image_file = None
                query_str = AppConstants.DEFAULT_MODEL_QUERY_STR

                if 'image_data' in request.files:
                    image_file = request.files['image_data']

                cfg = genai_app.get_config()
                try:
                    selected_model = self.resolve_chat_model(request.form.get('chatModel'))
                except ValueError:
                    return jsonify({'error': 'Invalid chat model'}), 400

                had_active_generation = self.interrupt_active_generation(preserve_partial=True)
                if had_active_generation:
                    try:
                        post_stop_to_sima(self.active_chat_model_name)
                    except Exception as e:
                        logging.error(f"Failed to stop previous generation before image request: {e}")

                image_path = None
                image_base64 = None
                if image_file:
                    try:
                        image_path, image_base64 = save_image_upload_as_base64(image_file, cfg['UPLOAD_FOLDER'])
                    except ValueError:
                        return jsonify({'error': 'Invalid image file name'}), 400

                # Add to conversation history and start assistant response (always has image)
                self.add_user_message(query_str, has_image=True, image_base64=image_base64)
                generation_id = self.begin_generation(selected_model)

                logging.info(f"Query string {query_str}")
                conversation_history = self.get_conversation_history()
                thread = threading.Thread(
                    target=stream_chat_request,
                    args=[conversation_history, selected_model, cfg, generation_id],
                    kwargs={'gen_params': _read_gen_params(request.form)}
                )
                thread.start()
                return {'question' : query_str}
        else:
            # Return error for image upload when in LLM-only mode
            @self.app.route('/upload_image', methods=['POST'])
            def upload_image_disabled():
                return jsonify({'error': 'Image upload is disabled in LLM-only mode'}), 400

        @self.app.route("/upload-to-rag", methods=["POST"])
        def upload_to_rag():
            if not self.rag_enabled:
                return "RAG is disabled", 403

            uploaded_file = request.files.get("file")
            if not uploaded_file:
                return "No file provided", 400

            filename = secure_filename(uploaded_file.filename)
            if not filename.lower().endswith(".md"):
                return "Only Markdown RAG upload is supported locally.", 400

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                uploaded_file.save(tmp)
                tmp_path = tmp.name

            def stream_response():
                global vectodb_proc
                yield "⏳ Starting...\n"
                new_db_path = None
                backup_path = None
                service_stopped = False
                replacement_installed = False
                try:
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
                    elif os.path.exists(meta_path):
                        os.unlink(meta_path)
                    if backup_path:
                        os.unlink(backup_path)
                        backup_path = None
                    yield "✅ Markdown RAG database is ready.\n"

                except Exception as e:
                    logging.exception("RAG document upload failed: %s", e)
                    if service_stopped:
                        stop_service()
                        if replacement_installed and os.path.exists(RAG_DB_PATH):
                            os.unlink(RAG_DB_PATH)
                        if backup_path and os.path.exists(backup_path):
                            os.replace(backup_path, RAG_DB_PATH)
                            backup_path = None
                        if os.path.exists(RAG_DB_PATH):
                            try:
                                vectodb_proc = start_service()
                                wait_for_rag_service(vectodb_proc)
                            except Exception:  # noqa: BLE001
                                logging.exception("Failed to restore the previous RAG service")
                    yield "❌ RAG document upload failed.\n"
                finally:
                    os.unlink(tmp_path)
                    if new_db_path and os.path.exists(new_db_path):
                        os.unlink(new_db_path)
                    if new_db_path:
                        new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                        if os.path.exists(new_meta_path):
                            os.unlink(new_meta_path)

            return Response(stream_with_context(stream_response()), mimetype="text/plain")
        
        @self.app.route("/import-rag-db", methods=["POST"])
        def import_rag_db():
            if not self.rag_enabled:
                return "RAG is disabled", 403

            def _stream():
                global vectodb_proc
                db_file = request.files.get("dbfile")
                if not db_file:
                    yield "❌ No file provided\n"
                    return

                yield "📥 Uploading database file...\n"
                backup_path = None
                service_stopped = False
                replacement_installed = False
                try:
                    ensure_rag_modules_loaded()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                        db_file.save(tmp)
                        tmp_path = tmp.name

                    yield "🛑 Stopping existing vectordb service for validation...\n"
                    stop_service()
                    service_stopped = True

                    yield "🔎 Validating uploaded database...\n"
                    validate_rag_database(tmp_path)

                    if os.path.exists(RAG_DB_PATH):
                        fd, backup_path = tempfile.mkstemp(
                            prefix=f".{Path(RAG_DB_PATH).name}.backup-",
                            dir=str(Path(RAG_DB_PATH).parent),
                        )
                        os.close(fd)
                        os.unlink(backup_path)
                        os.replace(RAG_DB_PATH, backup_path)

                    yield "✅ Database validated. Installing...\n"
                    shutil.move(tmp_path, RAG_DB_PATH)
                    replacement_installed = True

                    yield "🚀 Starting vectordb service with new database...\n"
                    vectodb_proc = start_service()
                    wait_for_rag_service(vectodb_proc)

                    meta_path = str(Path(RAG_DB_PATH).with_suffix(".meta.json"))
                    if os.path.exists(meta_path):
                        os.unlink(meta_path)

                    if backup_path:
                        os.unlink(backup_path)
                        backup_path = None

                    yield "✅ Import completed and service restarted.\n"
                except Exception as e:
                    logging.exception("RAG database import failed: %s", e)
                    if service_stopped:
                        stop_service()
                        if replacement_installed and os.path.exists(RAG_DB_PATH):
                            os.unlink(RAG_DB_PATH)
                        if backup_path and os.path.exists(backup_path):
                            os.replace(backup_path, RAG_DB_PATH)
                            backup_path = None
                        if os.path.exists(RAG_DB_PATH):
                            try:
                                vectodb_proc = start_service()
                                wait_for_rag_service(vectodb_proc)
                            except Exception:  # noqa: BLE001
                                logging.exception("Failed to restore the previous RAG service")
                    yield "❌ RAG database import failed.\n"
                finally:
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            return Response(stream_with_context(_stream()), mimetype="text/plain")

        @self.app.route("/reset-rag", methods=["POST"])
        def reset_rag():
            """Rebuild the RAG database from the bundled default Markdown."""
            if not self.rag_enabled:
                return "RAG is disabled", 403
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
                    yield "✅ RAG database reset to default.\n"
                except Exception as e:  # noqa: BLE001
                    logging.exception("RAG database reset failed: %s", e)
                    if service_stopped:
                        stop_service()
                        if replacement_installed and os.path.exists(RAG_DB_PATH):
                            os.unlink(RAG_DB_PATH)
                        if backup_path and os.path.exists(backup_path):
                            os.replace(backup_path, RAG_DB_PATH)
                            backup_path = None
                        if os.path.exists(RAG_DB_PATH):
                            try:
                                vectodb_proc = start_service()
                                wait_for_rag_service(vectodb_proc)
                            except Exception:  # noqa: BLE001
                                logging.exception("Failed to restore the previous RAG service")
                    yield "❌ RAG database reset failed.\n"
                finally:
                    if new_db_path and os.path.exists(new_db_path):
                        os.unlink(new_db_path)
                    if new_db_path:
                        new_meta_path = str(Path(new_db_path).with_suffix(".meta.json"))
                        if os.path.exists(new_meta_path):
                            os.unlink(new_meta_path)

            return Response(stream_with_context(_stream()), mimetype="text/plain")

        @self.app.route("/clear-rag", methods=["POST"])
        def clear_rag():
            """Clear the RAG database — stop the service and remove the DB files."""
            if not self.rag_enabled:
                return "RAG is disabled", 403

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

            return Response(stream_with_context(_stream()), mimetype="text/plain")




token_timestamps = deque()

def send_talk_text(in_text, generation_id=None):
    now = time.time()
    token_timestamps.append(now)

    # Remove timestamps older than 3 seconds
    while token_timestamps and (now - token_timestamps[0]) > 3:
        token_timestamps.popleft()

    # Calculate tokens per second
    token_count = len(token_timestamps)
    tokens_per_sec = token_count / 3.0  # Window size is 3 seconds

    # Pass both the text and the calculated tokens_per_sec into the queue
    if genai_app.talk_ctrl != None:
        genai_app.talk_ctrl.enqueue({
            'text': in_text,
            'generation_id': generation_id,
            'tokens_per_sec': tokens_per_sec, 
            'ttfs': ttfs
        })

    # Optional debug log
    logging.debug(f"Enqueued text: '{in_text}' | Tokens/sec: {tokens_per_sec:.2f}")
            


def cleanup():
    shutil.rmtree('./uploads')
    os.mkdir('./uploads')

# Function to post the file to another server
def _read_gen_params(form):
    """Per-request generation settings from the UI. The SiMa runtime honors
    max_tokens (response length); it decodes greedily (no temperature) and the
    context window is fixed by the compiled model."""
    params = {}
    raw = form.get('maxTokens')
    try:
        value = int(raw)
        if value > 0:
            params['max_tokens'] = value
    except (TypeError, ValueError):
        pass
    if str(form.get('noThink', '')).lower() in ('true', '1', 'yes'):
        params['no_think'] = True
    return params


def _apply_no_think(messages):
    """Return a copy of ``messages`` with ``/no_think`` appended to the last user
    turn (Qwen3's soft switch to disable reasoning). The shared history is not
    mutated, so it only affects this request."""
    if not messages:
        return messages
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if not (isinstance(m, dict) and m.get('role') == 'user'):
            continue
        m = dict(m)
        content = m.get('content')
        if isinstance(content, str):
            m['content'] = (content + ' /no_think').strip()
        elif isinstance(content, list):
            new = list(content)
            for j in range(len(new) - 1, -1, -1):
                part = new[j]
                if isinstance(part, dict) and part.get('type') == 'text':
                    part = dict(part)
                    part['text'] = (part.get('text', '') + ' /no_think').strip()
                    new[j] = part
                    break
            else:
                new.append({'type': 'text', 'text': '/no_think'})
            m['content'] = new
        out[i] = m
        break
    return out


def _answer_part(raw):
    """The portion of a streamed reply outside <think>…</think> — i.e. what should
    be spoken (reasoning is never sent to TTS)."""
    open_i = raw.find('<think>')
    if open_i == -1:
        close_i = raw.find('</think>')          # template pre-filled <think>
        return raw[close_i + 8:] if close_i != -1 else raw
    pre = raw[:open_i]
    after = raw[open_i + 7:]
    close_i = after.find('</think>')
    return pre if close_i == -1 else pre + after[close_i + 8:]


def _normalize_openai_image_parts(payload):
    """Rewrite OpenAI-standard image parts to the pyneat server's schema.

    The Solutions harnesses send {"type": "image_url", "image_url": {"url": u}}
    while pyneat expects {"type": "image", "image": u} (the format the Studio
    itself sends in add_user_message). Mutates payload in place.
    """
    for message in payload.get('messages') or []:
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if isinstance(part, dict) and part.get('type') == 'image_url':
                url = (part.get('image_url') or {}).get('url')
                if url:
                    content[i] = {'type': 'image', 'image': url}


def stream_chat_request(messages, model, config, generation_id, socketio_event='update', gen_params=None):
    """
    Streams chat completions from backend to frontend, handling TTS and history.
    """
    gen_params = gen_params or {}
    url = f"http://{config['SIMAAI_IP_ADDR']}/v1/chat/completions"
    no_think = bool(gen_params.get('no_think'))
    payload = {
        "model": model,
        "messages": _apply_no_think(messages) if no_think else messages,
        "stream": True
    }
    if no_think:
        # Also pass the chat-template switch, for runtimes that honor it.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    # Per-request max response tokens overrides the config default.
    max_tokens = gen_params.get('max_tokens') or config.get('MAX_TOKENS')
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    _full_reply = ""   # accumulated content, to keep reasoning out of TTS
    _tts_spoken = 0    # length of the answer already sent to TTS

    try:
        if not genai_app.start_assistant_response(generation_id):
            return None
        
        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
            
            for line in resp.iter_lines():
                if not genai_app.is_generation_current(generation_id):
                    break
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            chunk_json = json.loads(data_str)
                            if not genai_app.is_generation_current(generation_id):
                                break
                            
                            # Handle Metrics
                            if 'ttft' in chunk_json:
                                genai_app.emit('ttfs', round(float(chunk_json['ttft']), 2))
                            
                            if 'tps' in chunk_json:
                                genai_app.emit('tps', round(float(chunk_json['tps']), 2))

                            # Handle Content
                            if 'choices' in chunk_json and len(chunk_json['choices']) > 0:
                                delta = chunk_json['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    if not genai_app.add_to_current_response(generation_id, content):
                                        break
                                    # Update UI (the client splits <think> itself)
                                    genai_app.emit(socketio_event, {"results": content})

                                    # Trigger TTS on the ANSWER only — never speak
                                    # the model's <think> reasoning aloud.
                                    _full_reply += content
                                    answer_so_far = _answer_part(_full_reply)
                                    # A </think> can retroactively move earlier text
                                    # into reasoning, shrinking the answer — reset so
                                    # the real answer still gets spoken.
                                    if len(answer_so_far) < _tts_spoken:
                                        _tts_spoken = 0
                                    if len(answer_so_far) > _tts_spoken:
                                        send_talk_text(answer_so_far[_tts_spoken:], generation_id)
                                        _tts_spoken = len(answer_so_far)
                                
                        except Exception: 
                            pass
            
            # Finalize
            if genai_app.finish_assistant_response(generation_id):
                if genai_app.talk_ctrl is not None:
                    send_talk_text('END', generation_id)
                else:
                    if genai_app.complete_generation(generation_id):
                        genai_app.emit('end', {})

    except Exception as e:
        if genai_app.is_generation_current(generation_id):
            logging.error(f"Backend Stream Error: {e}")
            genai_app.fail_generation(
                generation_id,
                'Response generation failed. Please try again.',
            )
        return None

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

def post_audio_to_mla(audio_bytes, language="auto"):
    """
    Sends an audio file (as bytes) to the SIMA model server for transcription.
    """
    cfg = genai_app.get_config()
    url = f"http://{str(cfg['SIMAAI_IP_ADDR']).strip()}/v1/audio/transcriptions"

    files = {
        'file': ('audio.wav', audio_bytes, 'audio/wav')
    }
    model = genai_app.resolve_asr_model()
    if not model:
        logging.error("No speech-to-text model is active; cannot transcribe.")
        return None
    data = {
        'model': model,
        'language': language
    }

    logging.debug(f'Posting AUDIO to SIMA model server at {url}')
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        response.raise_for_status()
        logging.info("Successfully sent audio for transcription to SiMa.ai server.")

        result = response.json()
        return result  # Should include 'text' field with transcription text

    except requests.RequestException as e:
        logging.error(f"Failed to send audio for transcription: {e}")
        return None

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

def run_ui(app_cfg):
    global genai_app
    global vectodb_proc

    configure_logging()
    logging.info('Initializing Neat GenAI Studio app (frontend and TTS) please wait....')
    genai_app = AppContext()
    genai_app.initialize()
    genai_app.update_from_config(app_cfg)
    genai_app.setup_router()
    cleanup()

    if not genai_app.apionly and genai_app.rag_enabled:
        logging.info("Starting RAG database service")
        ensure_rag_modules_loaded()
        vectodb_proc = start_service()
    elif not genai_app.apionly:
        logging.info("RAG database service disabled")

    try:
        genai_app.run()
    finally:
        stop_service()
