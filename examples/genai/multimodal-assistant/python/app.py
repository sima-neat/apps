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
import argparse
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
from vectordb.gradioclient import upload_and_process_file, is_rag_fps_available
from vectordb.vectordb import RAG_DB_PATH, start_service, RagDbClient
from werkzeug.utils import secure_filename
import tempfile
import atexit
import signal
import subprocess

APP_DIR = Path(__file__).resolve().parent

genai_app = None
ttfs = 0
rag_db_client = RagDbClient()

vectodb_proc = None

def stop_service():
    global vectodb_proc
    if vectodb_proc and vectodb_proc.poll() is None:
        pgid = os.getpgid(vectodb_proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            vectodb_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            vectodb_proc.wait(timeout=5)
    vectodb_proc = None

atexit.register(stop_service)

class AppConstants:
    DEFAULT_SIMA_SERVER_IP = "127.0.0.1:9998"
    DEFAULT_CAMERA_IDX = 0
    DEFAULT_MODEL_QUERY_STR='Describe what you see in the picture.'
    DEFAULT_HTTP_PORT = 8081
    DEFAULT_UPLOADS_DIR = 'uploads'

def parse_vision_image_size(size_str):
    if not size_str:
        return None

    normalized = str(size_str).strip().lower().replace(' ', '')
    if not normalized:
        return None

    height = width = None

    if 'x' in normalized:
        parts = normalized.split('x', 1)
        if len(parts) == 2:
            try:
                height = int(float(parts[0]))
                width = int(float(parts[1]))
            except ValueError:
                return None
    else:
        try:
            value = int(float(normalized))
            height = width = value
        except ValueError:
            return None

    if not height or not width or height <= 0 or width <= 0:
        return None

    return {'height': height, 'width': width}

class TalkController:
    def __init__(self, supported_langs=None):
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        self.pipers = {}
        self._init_pipers_threaded(supported_langs or ['en'])
        
        self.lock = threading.Lock()
        self._reset_tps_counters()

        self.current_language = 'en'
        self.supported_langs = supported_langs or ['en']
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

    def _init_pipers_threaded(self, langs):
        threads = []
        assets = Path("assets")

        def load_piper(lang_code, onnx_file):
            from pipertts import PiperTTS # import here if needed locally
            try:
                self.pipers[lang_code] = PiperTTS(
                    model_path=str(onnx_file)
                )
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
                    
                elif self.current_language != 'xx':
                    sanitized_sentence = self._sanitize_for_tts(self.totalk)
                    # Enforce minimum length for first N chunks
                    if self.chunk_count < self.permissive_chunks and len(sanitized_sentence) < self.min_chars_first_chunks:
                        # Not enough content yet; keep buffering
                        return
                    buffer = self.pipers.get(self.current_language).synthesize(sanitized_sentence)
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
        language = language if language in self.pipers else 'en'

        sanitized_text = self._sanitize_for_tts(text)
        buffer = self.pipers.get(language).synthesize(sanitized_text)
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
        self.asr_model_name = "whisper-small"
        self.max_tokens = None
        self.vision_image_size = None

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

    def begin_generation(self):
        with self._state_lock:
            self._generation_counter += 1
            generation_id = self._generation_counter
            self._active_generation_id = generation_id
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
            self.talk_ctrl = TalkController(['en', 'fr', 'es', 'de', 'it', 'zh']) 

    def update_from_config(self, app_cfg):
        model_server = f"{app_cfg.openai.client_host}:{app_cfg.openai.port}"
        self.chat_model_name = app_cfg.chat_model.name
        self.asr_model_name = app_cfg.asr_model.name
        self.max_tokens = app_cfg.request.max_tokens
        self.update_settings(
            camidx=None,
            model_server_ip=model_server,
            ragserver=None,
            httponly=not app_cfg.web.https,
            apionly=False,
            llm_only=False,
            model_name=app_cfg.chat_model.name,
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
        self.app.config['ASR_MODEL_NAME'] = self.asr_model_name
        self.app.config['MAX_TOKENS'] = self.max_tokens
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
            js = (
                "window.SIMA_CONFIG=window.SIMA_CONFIG||{};"
                f"window.SIMA_CONFIG.llmOnly='{llm_only_val}';"
                f"window.SIMA_CONFIG.visionImageHeight='{height_val}';"
                f"window.SIMA_CONFIG.visionImageWidth='{width_val}';"
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
            self.socketio.run(self.app, host='0.0.0.0', port="5000",
                            ssl_context=ssl_context,
                            debug=False, allow_unsafe_werkzeug=True)
        else:
            self.socketio.run(self.app, host='0.0.0.0', port="5000",
                            debug=False, allow_unsafe_werkzeug=True)
            
    def run_stop(self):
        logging.info('Stopping processing...')
        self.interrupt_active_generation(preserve_partial=True)

        try:
            success = post_stop_to_sima()
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
                data = request.get_json() or {}
                lang = data.get('lang', 'en')
                voice_id = data.get('voiceId')
                if not voice_id:
                    return jsonify({'error': 'voiceId is required'}), 400
                assets_dir = Path('assets')
                candidate = assets_dir / voice_id
                if not candidate.exists():
                    # try adding .onnx
                    candidate = assets_dir / f"{voice_id}.onnx"
                if not candidate.exists():
                    return jsonify({'error': f'Voice model not found: {voice_id}'}), 404

                from pipertts import PiperTTS
                with self.talk_ctrl.lock:
                    old = self.talk_ctrl.pipers.get(lang)
                    new_voice = PiperTTS(model_path=str(candidate))
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
            except Exception as e:
                logging.error(f"Voice selection failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/stop', methods=['POST'])
        def stop_processing():
            logging.info('Received /stop request. Attempting to stop processing...')
            self.run_stop()
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
            chat = request.form.get('textchat', '').strip()
            search_rag = request.form.get('searchRag', 'false').lower() == 'true'
            include_chat_history = request.form.get('includeChatHistory', 'true').lower() == 'true'
            self.talk_ctrl.reset()
            self.talk_ctrl.set_language(language)
            had_active_generation = self.interrupt_active_generation(
                preserve_partial=include_chat_history
            )
            if had_active_generation:
                try:
                    post_stop_to_sima()
                except Exception as e:
                    logging.error(f"Failed to stop previous generation before new request: {e}")

            cfg = genai_app.get_config()

            # Save image if available and convert to base64
            image_path = None
            image_base64 = None
            audio_path = None

            if image_file:
                image_path = os.path.join(cfg['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)

                # Convert image to base64 for conversation history
                with open(image_path, 'rb') as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

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

            # if we have the query and search_rag flag is on, search the rag database
            if search_rag and len(full_query_str) > 0:
                rag = rag_db_client.search(query_str)
                full_query_str = f"{query_str}\n\nThe context is:\n\n{[entry['content'] for entry in rag]}"

            # Add user message to conversation history (detect if image is present)
            has_image = image_file is not None
            self.add_user_message(full_query_str, has_image=has_image, image_base64=image_base64)
            generation_id = self.begin_generation()

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
                args=[conversation_history, cfg.get('CHAT_MODEL_NAME', 'model'), cfg, generation_id]
            )
            thread.start()

            return {'question': query_str, 'ttt': elapsed_time}

        @self.app.route('/raghealth', methods=['GET'])
        def check_rag_server():
            try:
                rag_db_status = rag_db_client.is_server_up()
            except Exception as e:
                logging.error(f"Exception checking RAG DB server: {e}")
                rag_db_status = False

            try:
                rag_fps_status = is_rag_fps_available(self.ragserver)
            except Exception as e:
                logging.warning(f"Optional RAG File Processing Server check failed: {e}")
                rag_fps_status = False

            if rag_db_status:
                return {
                    "rag_db": "ok",
                    "rag_fps": "ok" if rag_fps_status else "unavailable",
                    "message": "RAG DB is available"
                }, 200
            else:
                return {
                    "rag_db": "unavailable",
                    "rag_fps": "ok" if rag_fps_status else "unavailable",
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
                response_format = data.get('response_format', 'wav')

                if not text:
                    return jsonify({'error': 'Missing "input" text field.'}), 400

                logging.info(f"Received TTS request: model={model}, voice={voice}, format={response_format}")

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

            except Exception as e:
                logging.error(f"TTS endpoint error: {e}")
                return jsonify({'error': str(e)}), 500

        # Only register image upload route if not in LLM-only mode
        if not self.llm_only:
            @self.app.route('/upload_image', methods=['POST'])
            def upload_image():
                image_file = None
                query_str = AppConstants.DEFAULT_MODEL_QUERY_STR

                if 'image_data' in request.files:
                    image_file = request.files['image_data']

                cfg = genai_app.get_config()
                had_active_generation = self.interrupt_active_generation(preserve_partial=True)
                if had_active_generation:
                    try:
                        post_stop_to_sima()
                    except Exception as e:
                        logging.error(f"Failed to stop previous generation before image request: {e}")

                image_path = None
                image_base64 = None
                if image_file:
                    image_path = os.path.join(cfg['UPLOAD_FOLDER'], image_file.filename)
                    image_file.save(image_path)

                    # Convert image to base64 for conversation history
                    with open(image_path, 'rb') as img_file:
                        image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                # Add to conversation history and start assistant response (always has image)
                self.add_user_message(query_str, has_image=True, image_base64=image_base64)
                generation_id = self.begin_generation()

                logging.info(f"Query string {query_str}")
                conversation_history = self.get_conversation_history()
                thread = threading.Thread(
                    target=stream_chat_request,
                    args=[conversation_history, cfg.get('CHAT_MODEL_NAME', 'model'), cfg, generation_id]
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
            uploaded_file = request.files.get("file")
            if not uploaded_file:
                return "No file provided", 400

            filename = secure_filename(uploaded_file.filename)
            if not filename.lower().endswith((".pdf", ".txt", ".md", ".xml")):
                return "Unsupported file type. Only PDF and TXT are allowed.", 400

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                uploaded_file.save(tmp)
                tmp_path = tmp.name

            def stream_response():
                yield "⏳ Starting...\n"
                try:
                    for line in upload_and_process_file(tmp_path, self.ragserver):
                        yield line + "\n"

                        if 'Database synchronized' in line:
                            time.sleep(2)
                            yield "🛑 Stopping existing vectordb service...\n"
                            stop_service()
                            time.sleep(5)

                            yield "🚀 Starting vectordb service with new database...\n"
                            global vectodb_proc
                            vectodb_proc = start_service()
                            time.sleep(20)

                except Exception as e:
                    yield f"❌ {str(e)}\n"
                finally:
                    os.unlink(tmp_path)

            return Response(stream_with_context(stream_response()), mimetype="text/plain")
        
        @self.app.route("/import-rag-db", methods=["POST"])
        def import_rag_db():
            def _stream():
                db_file = request.files.get("dbfile")
                if not db_file:
                    yield "❌ No file provided\n"
                    return

                yield "📥 Uploading database file...\n"
                try:
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
def stream_chat_request(messages, model, config, generation_id, socketio_event='update'):
    """
    Streams chat completions from backend to frontend, handling TTS and history.
    """
    url = f"http://{config['SIMAAI_IP_ADDR']}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    max_tokens = config.get('MAX_TOKENS')
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

def post_stop_to_sima():
    cfg = genai_app.get_config()
    url = f"http://{str(cfg['SIMAAI_IP_ADDR']).strip()}/stop"

    logging.debug(f'Posting STOP signal to SIMA model server at {url}')
    try:
        response = requests.post(url, timeout=5)
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

def run_app(app_cfg):
    global genai_app
    global vectodb_proc

    configure_logging()
    logging.info('Initializing multimodal assistant app (frontend and TTS) please wait....')
    genai_app = AppContext()
    genai_app.initialize()
    genai_app.update_from_config(app_cfg)
    genai_app.setup_router()
    cleanup()

    if not genai_app.apionly:
        logging.info("Starting RAG database service")
        vectodb_proc = start_service()

    genai_app.run()

if __name__ == '__main__':
    configure_logging()
    logging.info('Initializing genai-demo app (frontend and TTS) please wait....')
    genai_app = AppContext()
    genai_app.initialize()
    
    parser = argparse.ArgumentParser(description ='SiMa.ai GenAI demo application args')
    parser.add_argument('--camidx', type=int, required=False)
    parser.add_argument('--ip', type=str, required=False)
    parser.add_argument('--ragserver', type=str, required=False)
    parser.add_argument('--httponly', action='store_true', help='Run app server in http only')
    parser.add_argument('--apionly', action='store_true', help='Run app server with only OpenAI APIs, so we disable UI and Piper TTS functionalities')
    parser.add_argument('--llm-only', action='store_true', help='Disable vision features when using LLM-only models')
    parser.add_argument('--system-prompt-file', type=str, required=False, help='Provide a text file containing custom system prompt')
    parser.add_argument('--model-config', type=str, required=False, help='Path to model vlm_config.json')

    args = parser.parse_args()
    config_model_name = None
    vision_image_size_arg = None

    if args.model_config:
        try:
            with open(args.model_config, 'r', encoding='utf-8') as fh:
                cfg = json.load(fh)
            config_model_name = cfg.get('model_name')

            if isinstance(config_model_name, str):
                config_model_name = config_model_name.strip() or None
            else:
                config_model_name = None
            if not args.llm_only:
                raw_size = cfg.get('vm_cfg', {}).get('image_size')
                if raw_size is not None:
                    if isinstance(raw_size, int):
                        vision_image_size_arg = f"{raw_size}x{raw_size}"
                    elif isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
                        vision_image_size_arg = f"{raw_size[0]}x{raw_size[1]}"
        except Exception as exc:
            logging.warning(f"Failed to read model config '{args.model_config}': {exc}")

    vision_image_size = parse_vision_image_size(vision_image_size_arg)
    genai_app.update_settings(
        args.camidx,
        args.ip,
        args.ragserver,
        args.httponly,
        args.apionly,
        args.llm_only,
        config_model_name,
        None if args.llm_only else vision_image_size
    )

    if args.system_prompt_file:
        prompt_path = Path(args.system_prompt_file).expanduser()
        try:
            prompt_text = prompt_path.read_text(encoding='utf-8')
            genai_app.set_system_prompt(prompt_text)
        except Exception as exc:
            logging.error(f"Failed to load system prompt file '{prompt_path}': {exc}")

    # Setup routes after settings are configured
    genai_app.setup_router()
    cleanup()

    if not args.apionly:
        logging.info("Starting RAG database service")
        vectodb_proc = start_service()
    

    genai_app.run()
