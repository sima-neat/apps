# Vendored from https://github.com/florianvoss-commit/supertonic-sima
# (app/supertonic_sima/text.py) at commit 3b837b3e1b6a378ab8c24c3c04b079429b67e237,
# included in Neat GenAI Studio at the upstream author's request. See README.md
# in this directory for how to refresh it and THIRD_PARTY_TTS_MODELS.md for
# attribution. Local changes: none.
"""Text normalization and supported Supertonic language/voice identifiers."""

from __future__ import annotations

import re
from unicodedata import normalize


AVAILABLE_LANGUAGES = frozenset(
    {
        "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
        "hi", "hr", "hu", "id", "it", "ja", "ko", "lt", "lv", "na", "nl",
        "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
    }
)
AVAILABLE_VOICES = tuple(f"F{index}" for index in range(1, 6)) + tuple(
    f"M{index}" for index in range(1, 6)
)
MIN_SPEED = 0.7
MAX_SPEED = 2.0

# Local change: the upstream single character class listed adjacent ranges in
# an order CodeQL reports as overlapping; the same code points are matched here
# as a few ordered, non-overlapping ranges.
_EMOJI_PATTERN = re.compile(
    "(?:[\u2600-\u26ff\u2700-\u27bf]"
    "|[\U0001f1e6-\U0001f1ff]"
    "|[\U0001f300-\U0001f5ff]"
    "|[\U0001f600-\U0001f64f]"
    "|[\U0001f680-\U0001f6ff]"
    "|[\U0001f700-\U0001f77f]"
    "|[\U0001f780-\U0001f7ff]"
    "|[\U0001f800-\U0001f8ff]"
    "|[\U0001f900-\U0001f9ff]"
    "|[\U0001fa00-\U0001fa6f]"
    "|[\U0001fa70-\U0001faff])+",
    flags=re.UNICODE,
)
_SYMBOL_REPLACEMENTS = {
    "\u2013": "-", "\u2011": "-", "\u2014": "-", "\u00af": " ", "_": " ",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u00b4": "'",
    "`": "'", "[": " ", "]": " ", "|": " ", "/": " ", "#": " ",
    "→": " ", "←": " ",
}
_SPECIAL_SYMBOLS = re.compile(r"[♥☆♡©\\]")
_DUPLICATE_QUOTES = re.compile(r'(["\'`])\1+')
_WHITESPACE = re.compile(r"\s+")
_ENDING_PUNCTUATION = re.compile(r"[.!?;:,'\"')\]}…。」】〉》›»]$")


def preprocess_text(text: str, language: str) -> str:
    if language not in AVAILABLE_LANGUAGES:
        raise ValueError(f"unsupported language {language!r}")
    text = normalize("NFKD", text)
    text = _EMOJI_PATTERN.sub("", text)
    for old, new in _SYMBOL_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = _SPECIAL_SYMBOLS.sub("", text)
    text = text.replace("@", " at ")
    text = text.replace("e.g.,", "for example, ")
    text = text.replace("i.e.,", "that is, ")
    for old, new in (
        (" ,", ","), (" .", "."), (" !", "!"), (" ?", "?"),
        (" ;", ";"), (" :", ":"), (" '", "'"),
    ):
        text = text.replace(old, new)
    text = _DUPLICATE_QUOTES.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        raise ValueError("text is empty after preprocessing")
    if not _ENDING_PUNCTUATION.search(text):
        text += "."
    return f"<{language}>{text}</{language}>"
