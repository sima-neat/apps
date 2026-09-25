"""The Studio's text-to-speech controller.

Extracted from ui/flask_app.py to keep the web app focused on routing. Owns the
Supertonic (MLA) and CPU (piper-plus, piper-tts) engines, the browser-TTS mode,
engine/voice selection, and turning streamed reply text into spoken audio. The
owning AppContext sets ``self.app`` after construction; this controller calls
back through it to emit ``audio_chunk``/``end`` events and to check or complete
the active generation.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
import traceback
import wave
from pathlib import Path

import supertonic_tts
from tts_text import sanitize_for_tts
from voice_catalog import (
    asset_paths as catalog_asset_paths,
    catalog_voices,
    install_voice as install_catalog_voice,
    installed_voices,
    load_catalog,
    voice_by_id,
)

VOICE_CATALOG = load_catalog()


class TalkController:
    def __init__(self, supported_langs=None):
        # Set by AppContext after construction; used to emit audio chunks and
        # to check/complete the active generation. None until wired.
        self.app = None
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        self.pipers = {}            # rhasspy piper-tts voices, keyed by language
        self.pp = None              # active piper-plus multilingual engine
        self.pp_models = []         # installed piper-plus voices (selectable)
        self.pp_current = None      # key of the active piper-plus voice
        self.pp_lock = threading.Lock()   # guards runtime piper-plus voice switches
        self.prefer_piper_plus = False  # prefer piper-plus over dedicated piper-tts voices
        self.st = None              # Supertonic 3 MLA engine (all voices/languages)
        self.st_lock = threading.Lock()   # guards Supertonic voice switches
        # Supertonic is preferred whenever it loaded: it runs on the MLA and
        # speaks every language the Studio offers except Norwegian.
        self.prefer_supertonic = False
        self.browser_tts = False    # when True, no server synthesis — the client speaks via Web Speech
        self.utterance_speed = 1.0
        self.missing_voice_warnings = set()
        self._cpu_engine_lock = threading.Lock()   # serializes deferred CPU engine loads
        langs = supported_langs or ['en']
        # Supertonic is the default engine. When it loads, the CPU engines stay
        # out of RAM until a user selects them; only languages Supertonic cannot
        # speak (Chinese, Norwegian) get their dedicated Piper voice at startup.
        # Without Supertonic every installed CPU engine loads as before.
        self._init_supertonic()
        if self.st is None:
            self._init_piper_plus()
            self._init_pipers_threaded(langs)
        else:
            uncovered = [lang for lang in langs if not self.st.supports(lang)]
            if uncovered:
                self._init_pipers_threaded(uncovered)
            logging.info("CPU TTS engines (piper-plus, piper-tts) load on demand "
                         "when selected under Settings -> Voice engine.")
        
        self.lock = threading.Lock()
        self._reset_tps_counters()

        self.current_language = 'en'
        self.supported_langs = supported_langs or ['en']
        self._log_tts_coverage()
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

    def _engine_name(self, eng):
        if eng is None:
            return None
        if eng is self.st:
            return 'supertonic'
        if eng is self.pp:
            return 'piper-plus'
        return 'piper-tts'

    def _engine_voice(self, eng):
        """Speaker label for the metrics strip, where an engine has one."""
        if eng is not None and eng is self.st:
            return self.st.voice
        if eng is not None and eng is self.pp:
            return self.pp_current
        return None

    def _log_tts_coverage(self):
        """One line per UI language naming the engine that will speak it. The
        per-engine loaders run concurrently, so only this summary is
        authoritative about what has no server voice."""
        for lang in self.supported_langs:
            effective, eng = self._get_piper(lang)
            name = self._engine_name(eng)
            if name is None:
                logging.info("TTS coverage: %s -> browser/text only", lang)
            elif effective != lang:
                logging.info("TTS coverage: %s -> %s (via '%s' voice)", lang, name, effective)
            else:
                logging.info("TTS coverage: %s -> %s", lang, name)

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
        for eng in list(self.pipers.values()) + [self.pp, self.st]:
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
                elif self.st is not None and self.st.supports(lang):
                    logging.info("No dedicated voice installed for '%s' — using Supertonic.", lang)
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
        installed = {
            voice["id"] for voice in installed_voices(
                VOICE_CATALOG, assets_path=Path("assets"), engine="piper-plus")
        }
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

    def _init_supertonic(self):
        """Load the Supertonic 3 MLA engine when its runtime is installed."""
        if not supertonic_tts.available():
            logging.info("Supertonic runtime not installed under %s — MLA TTS unavailable "
                         "(clone supertonic-sima and re-run setup.sh).",
                         supertonic_tts.app_root())
            return
        try:
            st = supertonic_tts.SupertonicTTS(voice=self.st_voice_default())
            st.set_utterance_speed(self.utterance_speed)
            self.st = st
            self.prefer_supertonic = True
            logging.info("Supertonic 3 (MLA) ready: voice %s, %d steps, languages: %s",
                         st.voice, st.steps, sorted(st.languages))
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to load Supertonic 3 engine: %s", e)
            supertonic_tts.shutdown_worker()

    @staticmethod
    def st_voice_default():
        voice = (os.environ.get("SUPERTONIC_VOICE") or supertonic_tts.DEFAULT_VOICE).strip()
        return voice if voice in supertonic_tts.VOICES else supertonic_tts.DEFAULT_VOICE

    def set_supertonic_voice(self, key):
        """Switch the Supertonic speaker (F1-F5, M1-M5). Returns True on success."""
        with self.st_lock:
            if self.st is None:
                return False
            return self.st.set_voice((key or "").strip().upper())

    def supertonic_voices(self):
        """Supertonic speakers + the current one, for the UI picker."""
        if self.st is None:
            return {"voices": [], "current": None}
        return {
            "voices": [
                {"key": voice, "label": supertonic_tts.VOICE_LABELS.get(voice, voice)}
                for voice in self.st.voices
            ],
            "current": self.st.voice,
        }

    # -- deferred CPU engine loading -------------------------------------------
    def _installed_piper_plus(self, language=None):
        """Catalogued piper-plus models on disk (optionally speaking `language`)."""
        return installed_voices(
            VOICE_CATALOG, assets_path=Path("assets"), engine="piper-plus",
            language=language)

    def _installed_pipers(self, language=None):
        """Catalogued dedicated Piper voices on disk (optionally for `language`)."""
        return installed_voices(
            VOICE_CATALOG, assets_path=Path("assets"), engine="piper-tts",
            language=language)

    def _ensure_piper_plus_loaded(self):
        """Load piper-plus on selection. A failed attempt (transient worker or
        memory error) is retried on the next selection rather than remembered,
        so the user can recover without restarting. Returns True when an
        engine is up."""
        with self._cpu_engine_lock:
            if self.pp is None:
                self._init_piper_plus()
            return self.pp is not None

    def _load_piper_language(self, language):
        """Load the dedicated Piper voice for one language if it is installed
        and not resident yet. Voices load one at a time in the Piper worker, so
        this is deliberately per language: loading everything on selection would
        queue tens of seconds of loads ahead of the first utterance."""
        if not language or language in self.pipers or not self._installed_pipers(language):
            return language in self.pipers
        with self._cpu_engine_lock:
            if language not in self.pipers:
                self._init_pipers_threaded([language])
        return language in self.pipers

    def _ensure_pipers_loaded(self, language=None):
        """Make piper-tts selectable: load the voice for `language` now; other
        languages load on their first utterance. Returns True when the engine
        has at least one usable voice (loaded or installed)."""
        self._load_piper_language(language)
        return bool(self.pipers) or bool(self._installed_pipers())

    def set_voice_engine(self, engine, language=None):
        """Choose the preferred TTS engine for languages several can speak.
        'supertonic' -> the MLA engine; 'piper-plus' -> prefer piper-plus;
        'piper-tts' -> prefer rhasspy piper; 'browser' -> client-side speech.
        Languages only one engine supports are unaffected. CPU engines that
        were deferred at startup load here, on first selection."""
        engine = (engine or "").strip().lower()
        if engine in ("browser", "web", "webspeech", "web-speech"):
            self.browser_tts = True
        elif engine in ("supertonic", "supertonic-tts", "st", "mla"):
            if self.st is None:
                return False
            self.browser_tts = False
            self.prefer_supertonic = True
        elif engine in ("piper-plus", "piperplus", "pp"):
            if not self._ensure_piper_plus_loaded():
                return False
            self.browser_tts = False
            self.prefer_supertonic = False
            self.prefer_piper_plus = True
        elif engine in ("piper-tts", "pipertts", "piper", "rhasspy"):
            if not self._ensure_pipers_loaded(language):
                return False
            self.browser_tts = False
            self.prefer_supertonic = False
            self.prefer_piper_plus = False
        else:
            return False
        logging.info("Preferred TTS engine set to %s", self._preferred_engine_key())
        return True

    def _preferred_engine_key(self):
        if self.browser_tts:
            return "browser"
        if self.prefer_supertonic and self.st is not None:
            return "supertonic"
        return "piper-plus" if self.prefer_piper_plus else "piper-tts"

    def voice_engine(self, language=None):
        """Engines that can speak `language` (all loaded engines when language is
        None), plus the one actually used for it. Only engines that support the
        language are returned, so the UI never offers an incompatible engine."""
        engines = []
        st_ok = self.st is not None and (language is None or self.st.supports(language))
        pp_loaded = self.pp is not None and (language is None or self.pp.supports(language))
        # Deferred engines are offered when their models are installed; selecting
        # one loads it.
        pp_ok = pp_loaded or (self.pp is None and bool(self._installed_piper_plus(language)))
        tts_loaded = bool(self.pipers) and (language is None or language in self.pipers)
        tts_ok = tts_loaded or (
            (language is None or language not in self.pipers)
            and bool(self._installed_pipers(language)))
        if st_ok:
            engines.append({"key": "supertonic", "label": "Supertonic 3 (MLA, multilingual)",
                            "loaded": True})
        if pp_ok:
            engines.append({"key": "piper-plus", "label": "piper-plus (multilingual)",
                            "loaded": pp_loaded})
        if tts_ok:
            engines.append({"key": "piper-tts", "label": "piper-tts (rhasspy voices)",
                            "loaded": tts_loaded})
        # Browser TTS runs entirely in the client via the Web Speech API, so it is
        # always available and works for any language the device has a voice for.
        engines.append({"key": "browser", "label": "Browser (device voices)", "loaded": True})
        # The engine the router actually uses for this language.
        if self.browser_tts:
            current = "browser"
        else:
            current = self._preferred_engine_key()
            if language and current == "piper-tts" and language not in self.pipers \
                    and self._installed_pipers(language):
                pass   # deferred dedicated voice: it loads on the first utterance
            elif language:
                _, eng = self._get_piper(language)
                if eng is not None and eng is self.st:
                    current = "supertonic"
                elif eng is not None and eng is self.pp:
                    current = "piper-plus"
                elif eng is not None and eng in self.pipers.values():
                    current = "piper-tts"
        keys = [e["key"] for e in engines]
        if current not in keys:
            current = keys[0]
        return {"engines": engines, "current": current}

    def _get_piper(self, language, load=False):
        """Route a language to the best available TTS engine, returning
        ``(effective_language, engine)`` or ``(None, None)`` when nothing can
        speak it. Preference: Supertonic (when preferred) → piper-plus (when
        preferred) → a dedicated catalogued voice → piper-plus → Supertonic →
        English as a *latin-script only* fallback.
        CJK/Korean never fall back to an English voice — better silent than
        mispronounced. With ``load=True`` (synthesis paths) an installed but
        deferred dedicated voice is loaded before it is chosen."""
        if self.prefer_supertonic and self.st is not None and self.st.supports(language):
            return language, self.st
        if self.prefer_piper_plus and self.pp is not None and self.pp.supports(language):
            return language, self.pp
        if load and language not in self.pipers:
            self._load_piper_language(language)
        if language in self.pipers:
            return language, self.pipers[language]
        if self.pp is not None and self.pp.supports(language):
            return language, self.pp
        # Supertonic covers languages no CPU voice is installed for (incl. Korean).
        if self.st is not None and self.st.supports(language):
            return language, self.st
        if language not in ('ja', 'ko', 'zh'):
            if 'en' in self.pipers:
                return 'en', self.pipers['en']
            if self.st is not None and self.st.supports('en'):
                return 'en', self.st
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
                if generation_id is not None and not self.app.is_generation_current(generation_id):
                    return
                self.full_response.append(sanitized_sentence)
                self.app.emit('audio_chunk', {
                    'text': sanitized_sentence.strip(),
                    'browser': True,
                    'lang': self.current_language,
                    'tps': round(self.tps, 2),
                    'engine': 'browser',
                })
                self.chunk_count += 1
            else:
                _, piper = self._get_piper(self.current_language, load=True)
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
                            if generation_id is not None and not self.app.is_generation_current(generation_id):
                                return
                            elapsed_time = time.time() - start_time
                            audio_duration += self._get_wav_duration(buffer)
                            rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
                            self.app.emit('audio_chunk', {
                                'text': sanitized_sentence.strip(),
                                'audio': buffer.getvalue(),
                                'tps': round(self.tps, 2),
                                'rtf': round(rtf, 2),
                                'engine': self._engine_name(piper),
                                'voice': self._engine_voice(piper),
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
        if generation_id is not None and not self.app.is_generation_current(generation_id):
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
            self.app.emit('end', {})
            self.chunk_count = 0
            self._reset_tps_counters()
            if generation_id is not None:
                self.app.complete_generation(generation_id)

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

    
    class EngineUnavailable(Exception):
        """An explicitly requested engine cannot serve the request. ``reason``
        is one of the keys of ENGINE_REFUSALS: the API answers with that fixed
        text, never with exception details."""
        def __init__(self, reason, engine=""):
            super().__init__(reason)
            self.reason, self.engine = reason, engine

    ENGINE_REFUSALS = {
        'not-installed': 'that engine is not installed on this Studio',
        'no-voice': 'that engine has no voice for the requested language',
        'none': 'no TTS engine can speak the requested language',
    }

    ENGINE_KEYS = {
        'supertonic': 'supertonic', 'supertonic-tts': 'supertonic', 'mla': 'supertonic',
        'piper-plus': 'piper-plus', 'piperplus': 'piper-plus',
        'piper-tts': 'piper-tts', 'pipertts': 'piper-tts', 'piper': 'piper-tts', 'rhasspy': 'piper-tts',
    }

    def engine_for_request(self, engine, language):
        """Resolve an explicit ``model`` from the speech API to one engine.

        A recognized engine name is honored or refused: it never falls back to
        the UI's preference. Anything else (``default``, ``tts-1``, empty)
        goes through the router. Returns ``(effective_language, engine)``;
        raises ``EngineUnavailable`` when nothing can speak ``language``.
        """
        key = self.ENGINE_KEYS.get((engine or '').strip().lower())
        if key is None:
            language, eng = self._get_piper(language, load=True)
            if eng is None:
                raise self.EngineUnavailable('none')
            return language, eng
        if key == 'supertonic':
            if self.st is None:
                raise self.EngineUnavailable('not-installed', key)
            if self.st.supports(language):
                return language, self.st
            raise self.EngineUnavailable('no-voice', key)
        if key == 'piper-plus':
            if not self._installed_piper_plus():
                raise self.EngineUnavailable('not-installed', key)
            if self._ensure_piper_plus_loaded() and self.pp.supports(language):
                return language, self.pp
            raise self.EngineUnavailable('no-voice', key)
        # piper-tts: the dedicated voice for this language, loaded on demand.
        if self._load_piper_language(language):
            return language, self.pipers[language]
        raise self.EngineUnavailable('no-voice', key)

    def tts_on_demand(self, text, language='en', engine=None, voice=None):
        """
        Perform TTS synthesis on-demand for the given text and return audio bytes and timing info.

        ``engine`` is the request's ``model``: a recognized engine name is
        dispatched to exactly that engine (``EngineUnavailable`` when it cannot
        serve ``language``); other values use the router, which raises the same
        when no engine can speak the language. ``voice`` selects a Supertonic
        speaker (F1-F5, M1-M5) for this request only.
        """
        if not text:
            raise ValueError("No text provided for TTS synthesis.")

        start_time = time.time()
        language, piper = self.engine_for_request(engine, language)

        sanitized_text = self._sanitize_for_tts(text)
        if piper is self.st:
            buffer = piper.synthesize(sanitized_text, language=language, voice=voice)
        else:
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
