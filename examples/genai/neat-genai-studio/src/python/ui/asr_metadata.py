"""Helpers for consuming Whisper transcription metadata.

This module deliberately has no Flask or Neat dependencies so the confidence
policy can be unit-tested without starting the Studio server.
"""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_LOGPROB_THRESHOLD = -1.0


_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "norwegian": "no",
    "portuguese": "pt",
    "spanish": "es",
    "vietnamese": "vi",
}


def normalize_language_code(value: Any) -> str:
    """Return a lowercase ISO-style base language code when possible."""
    language = str(value or "").strip().lower().replace("_", "-")
    if not language:
        return ""
    language = _LANGUAGE_ALIASES.get(language, language)
    return language.split("-", 1)[0]


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_transcription(
    payload: Any,
    *,
    requested_language: str = "auto",
    supported_tts_languages: Iterable[str] = (),
    no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
) -> dict[str, Any]:
    """Normalize a Whisper result and decide whether it should reach the LLM.

    This mirrors Whisper's no-speech decision: a high no-speech probability is
    ignored when the generated tokens still have a sufficiently good average
    log probability. If avg_logprob is unavailable, no_speech_prob is used on
    its own.
    """
    result = payload if isinstance(payload, dict) else {}
    text = str(result.get("text") or "").strip()
    no_speech_prob = _optional_float(result.get("no_speech_prob"))
    avg_logprob = _optional_float(result.get("avg_logprob"))

    requested = normalize_language_code(requested_language)
    automatic = str(requested_language or "").strip().lower() in ("", "auto")
    language = normalize_language_code(result.get("language"))
    if not language and not automatic:
        language = requested

    likely_no_speech = (
        no_speech_prob is not None
        and no_speech_prob > no_speech_threshold
        and (avg_logprob is None or avg_logprob <= logprob_threshold)
    )
    reason = ""
    if likely_no_speech:
        reason = "no_speech"
    elif not text:
        reason = "empty_transcription"

    supported = {normalize_language_code(item) for item in supported_tts_languages}
    tts_language = language if language and language in supported else None

    return {
        "text": text,
        "language": language or None,
        "language_detected": automatic and bool(language),
        "tts_language": tts_language,
        "no_speech_prob": no_speech_prob,
        "avg_logprob": avg_logprob,
        "ignored": bool(reason),
        "reason": reason or None,
    }
