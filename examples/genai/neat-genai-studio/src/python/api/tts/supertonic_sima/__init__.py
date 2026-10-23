# Vendored from https://github.com/florianvoss-commit/supertonic-sima
# (app/supertonic_sima/__init__.py) at commit 3b837b3e1b6a378ab8c24c3c04b079429b67e237,
# included in Neat GenAI Studio at the upstream author's request. See README.md
# in this directory for how to refresh it and THIRD_PARTY_TTS_MODELS.md for
# attribution. Local changes: none.
"""Hybrid Supertonic 3 runtime for SiMa Modalix."""

from .audio import save_wav, wav_bytes
from .engine import SynthesisResult, SupertonicModalix, benchmark_summary
from .text import AVAILABLE_LANGUAGES, AVAILABLE_VOICES, MAX_SPEED, MIN_SPEED

__all__ = (
    "AVAILABLE_LANGUAGES",
    "AVAILABLE_VOICES",
    "MAX_SPEED",
    "MIN_SPEED",
    "SynthesisResult",
    "SupertonicModalix",
    "benchmark_summary",
    "save_wav",
    "wav_bytes",
)
