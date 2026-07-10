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
from flask import Flask, Response, render_template, jsonify, request, stream_with_context
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

APP_DIR = Path(__file__).resolve().parent

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
        self.pipers = {}
        self.utterance_speed = 1.0
        self.missing_voice_warnings = set()
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
        for piper in self.pipers.values():
            if hasattr(piper, 'set_utterance_speed'):
                piper.set_utterance_speed(self.utterance_speed)

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
                logging.warning(f"No model found for language: {lang}")

        for t in threads:
            t.join()

    def _get_piper(self, language):
        if language in self.pipers:
            return language, self.pipers[language]
        if language != 'en' and 'en' in self.pipers:
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
                    _, piper = self._get_piper(self.current_language)
                    if piper is None:
                        self._warn_missing_voice_once(self.current_language)
                    else:
                        buffer = piper.synthesize(sanitized_sentence)
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
        # Remove newlines and asterisks first
        text = text.replace('\n', ' ').replace('*', '').replace('＊', '')
        # Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix apostrophe spacing for contractions - remove spaces around apostrophes
        # This handles cases like "don ' t" -> "don't", "can ' t" -> "can't", etc.
        # Handle multiple spaces and various apostrophe characters
        text = re.sub(r"(\w)\s*['']\s*(\w)", r"\1'\2", text)
        # Remove space before punctuation .,;:!? and CJK equivalents where applicable
        text = re.sub(r'\s+([\.,;:!\?])', r'\1', text)
        # Join hyphenated words: letter - letter -> letter-letter
        text = re.sub(r'(?<=\w)\s*-\s*(?=\w)', '-', text)
        return text

    
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
        buffer = piper.synthesize(sanitized_text)
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

        if self.llm_only:
            assistant_content = self.current_response
        else:
            assistant_content = [{"type": "text", "text": self.current_response}]
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
            self.talk_ctrl = TalkController(['en', 'fr', 'es', 'de', 'it', 'zh', 'vi'])

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
    return params


def stream_chat_request(messages, model, config, generation_id, socketio_event='update', gen_params=None):
    """
    Streams chat completions from backend to frontend, handling TTS and history.
    """
    gen_params = gen_params or {}
    url = f"http://{config['SIMAAI_IP_ADDR']}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    # Per-request max response tokens overrides the config default.
    max_tokens = gen_params.get('max_tokens') or config.get('MAX_TOKENS')
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    
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
                                    # Update UI
                                    genai_app.emit(socketio_event, {"results": content})

                                    # Trigger TTS
                                    send_talk_text(content, generation_id)
                                
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
