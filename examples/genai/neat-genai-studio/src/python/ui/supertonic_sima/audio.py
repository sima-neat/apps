# Vendored from https://github.com/florianvoss-commit/supertonic-sima
# (app/supertonic_sima/audio.py) at commit 3b837b3e1b6a378ab8c24c3c04b079429b67e237,
# included in Neat GenAI Studio at the upstream author's request. See README.md
# in this directory for how to refresh it and THIRD_PARTY_TTS_MODELS.md for
# attribution. Local changes: none.
"""WAV serialization helpers shared by the example applications."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import wave

import numpy as np


def _pcm16(waveform: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(waveform).reshape(-1), -1.0, 1.0)
    return np.rint(samples * 32767.0).astype("<i2").tobytes()


def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(_pcm16(waveform))
    return output.getvalue()


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(waveform, sample_rate))
