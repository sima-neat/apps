"""MmsTTS — optional Korean (and other) TTS via Meta's MMS-TTS VITS models.

Used only for languages that neither piper-plus nor rhasspy piper-tts provide a
trained voice for — in practice **Korean** (``facebook/mms-tts-kor``, VITS,
16 kHz, Hangul input directly, no uroman romanization needed).

⚠️  LICENSE: ``facebook/mms-tts-*`` is **CC-BY-NC-4.0 (NON-COMMERCIAL)**. This
engine is therefore **opt-in and disabled by default** (see ``ENABLE_KOREAN_TTS``
/ the ``korean`` tts config) so the app's default footprint stays MIT-licensed
and torch-free. Enable it only if the non-commercial license is acceptable for
your use.

Runtime: transformers + torch (CPU). Heavier than the onnxruntime piper path, so
kept out of the default requirements — install ``requirements-korean.txt``.

Verification note: torch/transformers inference is board-only; this wrapper is
syntax-checked on the dev host and exercised on the Modalix board.
"""

from __future__ import annotations

import io
import logging
import wave


class MmsTTS:
    """Wrapper over a Meta MMS-TTS VITS model with the app's uniform TTS
    interface (``synthesize(text, language=None) -> BytesIO`` WAV)."""

    def __init__(self, model_dir, device="cpu"):
        # Lazy, guarded imports: torch/transformers are optional deps, and this
        # engine is only constructed when Korean TTS is explicitly enabled.
        import numpy as np
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._np = np
        self._torch = torch
        self.device = device
        self.model = VitsModel.from_pretrained(str(model_dir)).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.sample_rate = int(getattr(self.model.config, "sampling_rate", 16000))
        self.languages = {"ko"}
        logging.info("MmsTTS loaded %s (%d Hz)", str(model_dir), self.sample_rate)

    def set_utterance_speed(self, speed):
        """MMS ``VitsModel.speaking_rate``: higher = faster (1.0 = normal)."""
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.5, min(2.0, speed))
        try:
            self.model.speaking_rate = speed
        except Exception:                 # noqa: BLE001
            pass

    def supports(self, language):
        return language in self.languages

    def synthesize(self, text, language=None):
        """Synthesize ``text`` (Hangul) and return a WAV ``BytesIO`` (int16 PCM)."""
        np = self._np
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            waveform = self.model(**inputs).waveform  # float tensor [1, T]
        audio = waveform.squeeze().detach().cpu().numpy().astype("float32")
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)            # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)
        return buf
