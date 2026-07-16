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
import json
import logging
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
from flask_cors import CORS

from collections import deque
import wave
import traceback
from pathlib import Path
from werkzeug.utils import secure_filename
import tempfile
import atexit
import signal
import subprocess

from shared.config import HubConfig
from server import hub as hub_helpers
from tts_text import sanitize_for_tts   # Markdown/LaTeX → speakable text (stdlib only)

APP_DIR = Path(__file__).resolve().parent

_VERSION_CACHE = None


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

    if rag_db_client is not None:
        return rag_db_client

    from rag.vectordb import (
        RAG_DB_PATH as _RAG_DB_PATH,
        start_service as _start_service,
        RagDbClient,
    )
    from rag.create_db import create_markdown_vectordb as _create_markdown_vectordb

    RAG_DB_PATH = _RAG_DB_PATH
    start_service = _start_service
    create_markdown_vectordb = _create_markdown_vectordb
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

class TalkController:
    def __init__(self, supported_langs=None):
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        self.pipers = {}            # rhasspy piper-tts voices, keyed by language
        self.pp = None              # active piper-plus engine (multilingual; adds Japanese)
        self.pp_models = []         # installed piper-plus voices (selectable)
        self.pp_current = None      # key of the active piper-plus voice
        self.pp_lock = threading.Lock()   # guards runtime piper-plus voice switches
        self.mms = None             # optional MMS engine (Korean; opt-in, CC-BY-NC)
        self.prefer_piper_plus = False  # default engine: rhasspy piper-tts (ja still uses piper-plus)
        self.browser_tts = False    # when True, no server synthesis — the client speaks via Web Speech
        self.utterance_speed = 1.0
        self.missing_voice_warnings = set()
        # Load piper-plus (and optional MMS) first so the rhasspy-voice loader
        # knows which languages are already covered by another engine.
        self._init_piper_plus()
        self._init_mms_optional()
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
        for eng in list(self.pipers.values()) + [self.pp, self.mms]:
            if eng is not None and hasattr(eng, 'set_utterance_speed'):
                eng.set_utterance_speed(self.utterance_speed)

    def _init_pipers_threaded(self, langs):
        threads = []
        assets = Path("assets")

        def load_piper(lang_code, onnx_file):
            from pipertts import PiperTTS # import here if needed locally
            try:
                piper = PiperTTS(
                    model_path=str(onnx_file)
                )
                piper.set_utterance_speed(self.utterance_speed)
                self.pipers[lang_code] = piper
                logging.info(f"Loaded Piper model for {lang_code}")
            except Exception as e:
                logging.warning(f"Failed to load Piper model for {lang_code}: {e}")

        for lang in langs:
            found = False
            voice_files = list(assets.glob(f"{lang}_*-*.onnx"))

            # For English, prioritize voices with fallback order
            if lang == 'en' and voice_files:
                def voice_priority(voice_path):
                    name = voice_path.name
                    if name == "en_US-hfc_female-medium.onnx":
                        return 0  # Highest priority
                    elif name == "en_US-kristin-medium.onnx":
                        return 1  # Second choice
                    elif name == "en_US-hfc_male-medium.onnx":
                        return 2  # Third choice
                    elif name == "en_GB-northern_english_male-medium.onnx":
                        return 3  # Fourth choice
                    else:
                        return 4  # Any other voices

                voice_files.sort(key=voice_priority)

            for model_path in voice_files:
                config_path = assets / f"{model_path.name}.json"
                if config_path.exists():
                    lang_code = lang
                    thread = threading.Thread(
                        target=load_piper, args=(lang_code, model_path)
                    )
                    thread.start()
                    threads.append(thread)
                    found = True
                    break
            if not found:
                # No rhasspy voice for this language — expected when piper-plus
                # covers it (ja/en/zh/es/fr/pt) or when it is served by the
                # opt-in MMS engine (ko). Only warn when nothing can speak it.
                if (self.pp is not None and self.pp.supports(lang)) or lang == 'ko':
                    logging.info(f"No rhasspy voice for '{lang}' — served by piper-plus/MMS.")
                else:
                    logging.warning(f"No TTS voice available for language: {lang}")

        for t in threads:
            t.join()

    # Selectable piper-plus voices. Every checkpoint is the same 6-language
    # model (ja/en/zh/es/fr/pt); these differ only by voice. Discovered under
    # assets/piper-plus/<key>/{model.onnx,config.json} (see voice_install.sh);
    # order sets the default (first available).
    PIPER_PLUS_VOICES = [
        {"key": "css10", "label": "CSS10 (neutral)"},
        {"key": "tsukuyomi", "label": "Tsukuyomi-chan"},
        {"key": "mera", "label": "MERA multilingual"},
    ]

    def _init_piper_plus(self):
        """Discover installed piper-plus voices and load the default one. Each
        voice is a full 6-language model; users switch between them at runtime.
        Best-effort — no voices / unbuilt deps just leaves piper-plus off and the
        router falls back to piper-tts."""
        base = Path("assets") / "piper-plus"
        self.pp_models = []          # [{key, label, onnx: Path, config: Path|None}]
        for v in self.PIPER_PLUS_VOICES:
            onnx = base / v["key"] / "model.onnx"
            cfg = base / v["key"] / "config.json"
            if onnx.exists():
                self.pp_models.append({"key": v["key"], "label": v["label"],
                                       "onnx": onnx, "config": cfg if cfg.exists() else None})
        # Back-compat: a flat assets/piper-plus.onnx counts as one unnamed voice.
        flat = Path("assets") / "piper-plus.onnx"
        if not self.pp_models and flat.exists():
            flat_cfg = Path("assets") / "piper-plus.config.json"
            self.pp_models.append({"key": "default", "label": "piper-plus",
                                   "onnx": flat, "config": flat_cfg if flat_cfg.exists() else None})

        self.pp = None
        self.pp_current = None
        if not self.pp_models:
            logging.info("No piper-plus voice found under assets/piper-plus/ — "
                         "Japanese TTS disabled (run voice_install.sh).")
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
        """Switch the active piper-plus voice at runtime (thread-safe)."""
        with self.pp_lock:
            return self._load_piper_plus(key)

    def piper_plus_voices(self):
        """Available piper-plus voices + the current one, for the UI picker."""
        return {
            "voices": [{"key": m["key"], "label": m["label"]}
                       for m in getattr(self, "pp_models", [])],
            "current": getattr(self, "pp_current", None),
        }

    def set_voice_engine(self, engine):
        """Choose the preferred TTS engine for languages both can speak.
        'piper-plus' -> prefer piper-plus; 'piper-tts' -> prefer rhasspy piper.
        Languages only one engine supports are unaffected (ja is always
        piper-plus; de/it/no/vi are always piper-tts)."""
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
            engines.append({"key": "piper-plus", "label": "piper-plus (multilingual · Japanese)"})
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

    def _init_mms_optional(self):
        """Optionally load the MMS Korean engine, ONLY when explicitly enabled
        (ENABLE_KOREAN_TTS=1) and the model is present. MMS is CC-BY-NC and pulls
        in torch, so it is off by default."""
        flag = os.environ.get("ENABLE_KOREAN_TTS", "0").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        model_dir = Path("assets") / "mms-tts-kor"
        if not model_dir.is_dir():
            logging.warning("ENABLE_KOREAN_TTS set but %s is missing — Korean TTS off.",
                            model_dir)
            return
        try:
            from mms_tts import MmsTTS
            self.mms = MmsTTS(model_dir)
            self.mms.set_utterance_speed(self.utterance_speed)
            logging.info("MMS Korean TTS ready (CC-BY-NC 4.0, non-commercial).")
        except Exception as e:  # noqa: BLE001
            self.mms = None
            logging.warning("Failed to load MMS Korean TTS: %s", e)

    def _init_mms_optional(self):
        """Optionally load the MMS Korean engine, ONLY when explicitly enabled
        (ENABLE_KOREAN_TTS=1) and the model is present. MMS is CC-BY-NC and pulls
        in torch, so it is off by default."""
        flag = os.environ.get("ENABLE_KOREAN_TTS", "0").strip().lower()
        if flag not in ("1", "true", "yes", "on"):
            return
        model_dir = Path("assets") / "mms-tts-kor"
        if not model_dir.is_dir():
            logging.warning("ENABLE_KOREAN_TTS set but %s is missing — Korean TTS off.",
                            model_dir)
            return
        try:
            from mms_tts import MmsTTS
            self.mms = MmsTTS(model_dir)
            self.mms.set_utterance_speed(self.utterance_speed)
            logging.info("MMS Korean TTS ready (CC-BY-NC 4.0, non-commercial).")
        except Exception as e:  # noqa: BLE001
            self.mms = None
            logging.warning("Failed to load MMS Korean TTS: %s", e)

    def _get_piper(self, language):
        """Route a language to the best available TTS engine, returning
        ``(effective_language, engine)`` or ``(None, None)`` when nothing can
        speak it. Preference (with prefer_piper_plus): piper-plus for the
        languages it supports (incl. Japanese) → MMS for Korean (if enabled) →
        a rhasspy piper-tts voice → English as a *latin-script only* fallback.
        CJK/Korean never fall back to an English voice — better silent than
        mispronounced."""
        if self.prefer_piper_plus and self.pp is not None and self.pp.supports(language):
            return language, self.pp
        if language == 'ko' and self.mms is not None:
            return language, self.mms
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

                self.totalk = self.generate_talk()
                start_time = time.time()

                sanitized_sentence = self._sanitize_for_tts(self.totalk)
                logging.info(f"{self.full_response} : {sanitized_sentence}")
                if sanitized_sentence in self.full_response:
                    logging.info(f"Skipping synthesis for repeated text: {self.totalk}")
                    self._reset_tps_counters()
                    
                elif self.tts_enabled and self.current_language != 'xx':
                    sanitized_sentence = self._sanitize_for_tts(self.totalk)
                    # Enforce minimum length for first N chunks
                    if self.chunk_count < self.permissive_chunks and len(sanitized_sentence) < self.min_chars_first_chunks:
                        # Not enough content yet; keep buffering
                        return
                    if self.browser_tts:
                        # Browser TTS: skip server synthesis and hand the clean
                        # sentence to the client, which speaks it with the Web
                        # Speech API. Reuses the same chunking + sanitization.
                        if generation_id is not None and not genai_app.is_generation_current(generation_id):
                            return
                        if sanitized_sentence.strip():
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
                            buffer = piper.synthesize(sanitized_sentence, language=self.current_language)
                            if generation_id is not None and not genai_app.is_generation_current(generation_id):
                                return
                            self.full_response.append(sanitized_sentence)

                            elapsed_time = time.time() - start_time
                            audio_duration = self._get_wav_duration(buffer)
                            rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
                            logging.info(f"[Timing] self.piper.synthesize took {elapsed_time:.3f} seconds for [{sanitized_sentence}]")
                            genai_app.emit('audio_chunk', {
                                'text': sanitized_sentence.strip(),
                                'audio': buffer.getvalue(),
                                'tps': round(self.tps, 2),
                                'rtf': round(rtf, 2)
                            })
                            self.chunk_count += 1

                self.talk = []
                if self._next is not None:
                    self.talk.append(self._next)
                    self._next = None
            else:
                self.talk.append(subword)
        else:
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
        self.asr_model_name = "whisper-small"
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
            self.talk_ctrl = TalkController(['en', 'fr', 'es', 'de', 'it', 'zh', 'vi', 'ja', 'pt', 'ko'])

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
        CORS(self.app)
        self.socketio = SocketIO(self.app)

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
            ssl_context = (
                str(APP_DIR / 'certs/server.crt'),
                str(APP_DIR / 'certs/server.key')
            )
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
            for onnx_path in assets_dir.glob(f"{lang}_*-*.onnx"):
                cfg = assets_dir / f"{onnx_path.name}.json"
                if cfg.exists():
                    voices.append({
                        'id': onnx_path.name,
                        'label': onnx_path.stem
                    })
            current = self.current_voice_by_lang.get(lang)
            if not current:
                if self.talk_ctrl != None:
                    inst = self.talk_ctrl.pipers.get(lang)
                    try:
                        if inst is not None and hasattr(inst, 'model_path'):
                            current = Path(getattr(inst, 'model_path')).name
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
                voice_names = [voice_name]
                if not voice_name.endswith('.onnx'):
                    voice_names.append(f"{voice_name}.onnx")
                available_voices = {path.name: path for path in assets_dir.glob('*.onnx')}
                candidate = next((available_voices[name] for name in voice_names if name in available_voices), None)
                if candidate is None:
                    return jsonify({'error': 'Voice model not found'}), 404

                from pipertts import PiperTTS
                with self.talk_ctrl.lock:
                    old = self.talk_ctrl.pipers.get(lang)
                    new_voice = PiperTTS(model_path=str(candidate))
                    new_voice.set_utterance_speed(self.talk_ctrl.utterance_speed)
                    self.talk_ctrl.pipers[lang] = new_voice
                    self.current_voice_by_lang[lang] = candidate.name
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

                return jsonify({'status': 'ok', 'lang': lang, 'current': self.current_voice_by_lang.get(lang)})
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
                return jsonify({'error': f'control API unreachable: {exc}'}), 502

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
                return jsonify({'error': f'control API unreachable: {exc}'}), 502

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

        @self.app.route('/models/unload', methods=['POST'])
        def models_unload():
            name = (request.get_json(silent=True) or {}).get('name', '')
            return _proxy_control('POST', '/control/unload', 60, {'name': name})

        @self.app.route('/models/delete', methods=['POST'])
        def models_delete():
            name = (request.get_json(silent=True) or {}).get('name', '')
            return _proxy_control('POST', '/control/delete', 120, {'name': name})

        @self.app.route('/models/reset-mla', methods=['POST'])
        def models_reset_mla():
            return _proxy_control('POST', '/control/reset_mla', 30)

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
                return jsonify({'error': f'control API unreachable: {exc}'}), 502

        @self.app.route('/models/hub/search', methods=['GET'])
        def models_hub_search():
            query = request.args.get('q', '')
            try:
                return jsonify(hub_helpers.hub_search(self.catalog_dir, self.hub_config, query))
            except Exception as exc:
                logging.exception("Hub search failed")
                return jsonify({'enabled': False, 'results': [], 'error': str(exc)}), 500

        @self.app.route('/models/hub/card', methods=['GET'])
        def models_hub_card():
            repo_id = request.args.get('repoId', '')
            try:
                return jsonify(hub_helpers.hub_card(self.hub_config, repo_id))
            except Exception as exc:
                logging.exception("Hub card failed")
                return jsonify({'enabled': False, 'error': str(exc)}), 500

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
                return jsonify({'error': f'model server unreachable: {exc}'}), 502

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

            # Handle file uploads
            if 'audio_data' in request.files:
                audio_file = request.files['audio_data']
            if 'image_data' in request.files:
                image_file = request.files['image_data']

            # Handle form fields
            language = request.form.get('language', 'en')
            utterance_speed = request.form.get('utteranceSpeed', '1.0')
            chat = request.form.get('textchat', '').strip()
            search_rag = request.form.get('searchRag', 'false').lower() == 'true'
            include_chat_history = request.form.get('includeChatHistory', 'true').lower() == 'true'
            enable_tts = request.form.get('enableTts', 'true').lower() == 'true'
            gen_params = _read_gen_params(request.form)
            cfg = genai_app.get_config()
            try:
                selected_model = self.resolve_chat_model(request.form.get('chatModel'))
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400

            self.talk_ctrl.reset()
            self.talk_ctrl.set_language(language)
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
                        # The text is in the 'text' field
                        query_str = result.get('text', '')
                        logging.info(f"Transcribed results from backend: {query_str}")
                        
                        if query_str:
                            self.socketio.emit('transcription', query_str)
                    else:
                        logging.warning("Backend transcription returned no result")
                        query_str = AppConstants.DEFAULT_MODEL_QUERY_STR
                except Exception as e:
                    logging.error(e)

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
            has_image = image_file is not None
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

            return {
                'question': query_str,
                'ttt': elapsed_time,
                'rag_used': rag_used,
                'rag_hits': rag_hits,
            }

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
                result["error"] = f"Could not read the RAG database: {e}"
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
                except ValueError as exc:
                    return jsonify({'error': str(exc)}), 400

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
                try:
                    ensure_rag_modules_loaded()
                    yield "🛑 Stopping existing vectordb service...\n"
                    stop_service()

                    yield "📚 Creating VectorDB from Markdown...\n"
                    create_markdown_vectordb(
                        input_path=tmp_path,
                        output_db=RAG_DB_PATH,
                        embedding_model=self.rag_embedding_model_dir,
                    )

                    yield "🚀 Starting vectordb service with new database...\n"
                    vectodb_proc = start_service()
                    yield "✅ Markdown RAG database is ready.\n"

                except Exception as e:
                    yield f"❌ {str(e)}\n"
                finally:
                    os.unlink(tmp_path)

            return Response(stream_with_context(stream_response()), mimetype="text/plain")
        
        @self.app.route("/import-rag-db", methods=["POST"])
        def import_rag_db():
            if not self.rag_enabled:
                return "RAG is disabled", 403

            def _stream():
                db_file = request.files.get("dbfile")
                if not db_file:
                    yield "❌ No file provided\n"
                    return

                yield "📥 Uploading database file...\n"
                try:
                    ensure_rag_modules_loaded()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                        db_file.save(tmp)
                        tmp_path = tmp.name

                    yield "✅ File uploaded. Replacing existing DB...\n"
                    shutil.move(tmp_path, RAG_DB_PATH)

                    yield "🛑 Stopping existing vectordb service...\n"
                    stop_service()
                    time.sleep(1)

                    yield "🚀 Starting vectordb service with new database...\n"
                    global vectodb_proc
                    vectodb_proc = start_service()
                    time.sleep(30)

                    yield "✅ Import completed and service restarted.\n"
                except Exception as e:
                    yield f"❌ Failed: {str(e)}\n"
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
                try:
                    ensure_rag_modules_loaded()
                    if not default_md.is_file():
                        yield f"❌ Default RAG source not found: {default_md}\n"
                        return
                    yield "🛑 Stopping existing vectordb service...\n"
                    stop_service()
                    yield "📚 Rebuilding from the bundled document...\n"
                    create_markdown_vectordb(
                        input_path=str(default_md),
                        output_db=RAG_DB_PATH,
                        embedding_model=self.rag_embedding_model_dir,
                    )
                    yield "🚀 Starting vectordb service with the default database...\n"
                    vectodb_proc = start_service()
                    yield "✅ RAG database reset to default.\n"
                except Exception as e:  # noqa: BLE001
                    yield f"❌ {str(e)}\n"

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
                    yield f"❌ {str(e)}\n"

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
        # genai_app.emit('error', str(e))
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

def post_audio_to_mla(audio_bytes, language="en"):
    """
    Sends an audio file (as bytes) to the SIMA model server for transcription.
    """
    cfg = genai_app.get_config()
    url = f"http://{str(cfg['SIMAAI_IP_ADDR']).strip()}/v1/audio/transcriptions"

    files = {
        'file': ('audio.wav', audio_bytes, 'audio/wav')
    }
    data = {
        'model': cfg.get('ASR_MODEL_NAME', 'whisper-small'),
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
