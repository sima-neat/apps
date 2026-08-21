"""PiperPlusTTS — wrapper around piper-plus for multilingual neural TTS.

piper-plus (https://github.com/ayutaz/piper-plus) is the MIT-licensed Piper fork
with a custom G2P (no espeak-ng). It is installed in the MAIN venv and owns the
top-level ``piper`` package, exposing ``piper.PiperVoice``. Its trained
multilingual model covers **Japanese, English, Mandarin (zh), Spanish, French,
Portuguese** (the ``language_id_map`` in the checkpoint). It is the selectable
multilingual alternative to the dedicated piper-tts voices.

Published-wheel API (piper-plus 1.13.0):

    from piper import PiperVoice
    voice = PiperVoice.load(model_path, config_path=..., use_cuda=False)
    voice.synthesize(text, wav_file, length_scale=..., language_id=<int|None>)

The language is selected with ``language_id`` (an int from the config's
``language_id_map``, e.g. ja=0); ``None`` lets piper-plus auto-detect.

Runtime: onnxruntime (CPU). Verification note: piper-plus and pyopenjtalk-plus
are aarch64/board-only; this wrapper is syntax-checked on the dev host and
exercised end-to-end on the Modalix board.
"""

from __future__ import annotations

import io
import json
import logging
import wave
from pathlib import Path


class PiperPlusTTS:
    """Wraps piper-plus's ``piper.PiperVoice`` with the app's uniform TTS
    interface (``synthesize(text, language=None) -> BytesIO`` WAV)."""

    # Languages an actual trained piper-plus checkpoint supports.
    FALLBACK_LANGS = ("ja", "en", "zh", "es", "fr", "pt")

    def __init__(self, model_path, config_path=None):
        from piper import PiperVoice   # piper-plus's PiperVoice (main venv owns `piper`)

        self.model_path = str(model_path)
        cfg_path = str(config_path) if config_path else None
        self.voice = PiperVoice.load(self.model_path, config_path=cfg_path)

        # language code -> language_id, read from the model config.
        self.lang_ids = {}
        try:
            resolved = cfg_path or f"{self.model_path}.json"
            with open(resolved, "r", encoding="utf-8") as fh:
                self.lang_ids = json.load(fh).get("language_id_map", {}) or {}
        except Exception:  # noqa: BLE001
            self.lang_ids = {}
        self.languages = set(self.lang_ids) or set(self.FALLBACK_LANGS)
        self.length_scale = None   # None -> model default
        logging.info("PiperPlusTTS loaded %s (languages: %s)",
                     self.model_path, sorted(self.languages))

    def set_utterance_speed(self, speed):
        """Map a 0.5–2.0x speed multiplier to piper's ``length_scale``
        (inverse: higher = slower)."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))
        self.length_scale = 1.0 / speed

    def supports(self, language):
        return bool(language) and language in self.languages

    def synthesize(self, text, language=None):
        """Synthesize ``text`` and return a WAV ``BytesIO``. ``language`` is
        mapped to the model's ``language_id`` (``None`` -> auto-detect)."""
        language_id = self.lang_ids.get(language) if language else None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self.voice.synthesize(
                text, wav_file,
                length_scale=self.length_scale,
                language_id=language_id,
            )
        buf.seek(0)
        return buf
