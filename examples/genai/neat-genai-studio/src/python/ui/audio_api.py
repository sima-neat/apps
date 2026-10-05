"""Request parsing and response shaping for the Studio's OpenAI-compatible
audio API (``/v1/audio/speech``, ``/v1/audio/transcriptions``,
``/v1/audio/voices``).

Pure Python on purpose: no Flask, no requests, no engine imports, so the unit
suite can exercise every contract decision here without the web app or the
board. The routes in ``flask_app.py`` are thin adapters over these helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# OpenAI's documented bounds for ``speed``. Engines clamp further to their own
# range (Supertonic 0.7-2.0, Piper 0.5-2.0); the effective value is reported.
SPEED_MIN = 0.25
SPEED_MAX = 4.0
SPEED_DEFAULT = 1.0
MAX_INPUT_CHARS = 4096            # OpenAI's limit for `input`
MAX_TRANSCRIPTION_BYTES = 25 * 1024 * 1024   # OpenAI's upload limit
SPEECH_FORMATS = ("wav",)         # this server produces WAV only
TRANSCRIPTION_FORMATS = ("json", "verbose_json", "text")
SPEED_ALIASES = ("utterance_speed", "utteranceSpeed")   # pre-standard field names
# `model` on the speech route: an engine name (aliases accepted) is used or
# refused; the router names (OpenAI's own model ids included, so OpenAI
# clients work unchanged) let the Studio pick the engine. Anything else is a
# client error rather than a silent swap to another engine.
SPEECH_ENGINE_ALIASES = {
    "supertonic": "supertonic", "supertonic-tts": "supertonic", "mla": "supertonic",
    "piper-plus": "piper-plus", "piperplus": "piper-plus",
    "piper-tts": "piper-tts", "pipertts": "piper-tts", "piper": "piper-tts", "rhasspy": "piper-tts",
}
SPEECH_ROUTER_MODELS = ("default", "tts-1", "tts-1-hd", "gpt-4o-mini-tts")


def speech_model(value: Any) -> str:
    """Canonical `model` for the speech route: an engine key or "default".
    Raises AudioApiError(400, param=model) for an unknown name."""
    name = str(value or "").strip().lower()
    if not name or name in SPEECH_ROUTER_MODELS:
        return "default"
    if name in SPEECH_ENGINE_ALIASES:
        return SPEECH_ENGINE_ALIASES[name]
    accepted = ", ".join(("default", "supertonic", "piper-plus", "piper-tts"))
    raise AudioApiError(400, f"model '{value}' is not a speech engine; use one of {accepted}", "model")


class AudioApiError(ValueError):
    """A client error with the HTTP status and offending parameter attached."""

    def __init__(self, status: int, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.param = param

    def payload(self) -> dict:
        body = {"error": self.message}
        if self.param:
            body["param"] = self.param
        return body


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    model: str
    voice: str
    language: str
    speed: float
    response_format: str
    used_speed_alias: bool


def _float_param(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise AudioApiError(400, f"'{name}' must be a number", name)
    if number != number:   # NaN
        raise AudioApiError(400, f"'{name}' must be a number", name)
    return number


def _speech_language(value: Any) -> str:
    language = str(value or "").strip().lower()
    return "en" if language in ("", "auto") else language


def parse_speech_request(data: Any) -> SpeechRequest:
    """Validate a ``POST /v1/audio/speech`` JSON body."""
    if not isinstance(data, Mapping):
        raise AudioApiError(400, "Missing JSON payload.")
    text = data.get("input")
    if not isinstance(text, str) or not text.strip():
        raise AudioApiError(400, 'Missing "input" text field.', "input")
    if len(text) > MAX_INPUT_CHARS:
        raise AudioApiError(
            400, f"'input' is longer than {MAX_INPUT_CHARS} characters", "input")

    response_format = str(data.get("response_format") or "wav").strip().lower()
    if response_format not in SPEECH_FORMATS:
        raise AudioApiError(
            400,
            f"response_format '{response_format}' is not supported; this server "
            f"produces 'wav' only",
            "response_format")

    used_alias = False
    if "speed" in data and data.get("speed") is not None:
        speed = _float_param(data.get("speed"), "speed")
    else:
        speed = SPEED_DEFAULT
        for alias in SPEED_ALIASES:
            if data.get(alias) is not None:
                speed = _float_param(data.get(alias), alias)
                used_alias = True
                break
    if not SPEED_MIN <= speed <= SPEED_MAX:
        raise AudioApiError(
            400, f"'speed' must be between {SPEED_MIN} and {SPEED_MAX}", "speed")

    return SpeechRequest(
        text=text,
        model=speech_model(data.get("model")),
        voice=str(data.get("voice") or "default").strip() or "default",
        # "auto" means "not stated": speech needs one language, so it is English
        # (the Studio's default) rather than a refusal for an unknown code.
        language=_speech_language(data.get("language")),
        speed=speed,
        response_format=response_format,
        used_speed_alias=used_alias,
    )


@dataclass(frozen=True)
class TranscriptionRequest:
    model: str | None
    language: str
    response_format: str


def parse_transcription_form(form: Mapping, *, has_file: bool, size: int | None) -> TranscriptionRequest:
    """Validate a ``POST /v1/audio/transcriptions`` multipart form."""
    if not has_file:
        raise AudioApiError(400, 'Missing "file" upload field.', "file")
    if size is not None and size > MAX_TRANSCRIPTION_BYTES:
        raise AudioApiError(
            413, f"audio upload exceeds {MAX_TRANSCRIPTION_BYTES // (1024 * 1024)} MiB", "file")
    response_format = str(form.get("response_format") or "json").strip().lower()
    if response_format not in TRANSCRIPTION_FORMATS:
        raise AudioApiError(
            400,
            f"response_format '{response_format}' is not supported; use one of "
            + ", ".join(TRANSCRIPTION_FORMATS),
            "response_format")
    model = str(form.get("model") or "").strip() or None
    language = str(form.get("language") or "auto").strip().lower() or "auto"
    return TranscriptionRequest(model=model, language=language, response_format=response_format)


TRANSCRIPTION_TASKS = ("transcribe", "translate")


def format_transcription(result: Mapping, asr: Mapping | None, response_format: str,
                         model: str | None = None, *, task: str = "transcribe") -> tuple[Any, str]:
    """Shape the model server's transcription/translation result for the
    requested format.

    Returns ``(body, mimetype)``; ``body`` is a dict for the JSON formats and a
    string for ``text``. ``asr`` is the Studio's ``analyze_transcription``
    output and is only consulted for ``verbose_json``. For ``task="translate"``
    (Whisper's speech-to-English) ``language`` stays the detected *source*
    language and ``tts_language`` is English, the language of the text.
    """
    text = str((result or {}).get("text") or "").strip()
    if response_format == "text":
        return text + "\n", "text/plain; charset=utf-8"
    if response_format == "verbose_json":
        asr = asr or {}
        body = {
            "text": text,
            "language": asr.get("language") or (result or {}).get("language") or None,
            "language_detected": bool(asr.get("language_detected", False)),
            "tts_language": asr.get("tts_language"),
            "no_speech_prob": asr.get("no_speech_prob", (result or {}).get("no_speech_prob")),
            "avg_logprob": asr.get("avg_logprob", (result or {}).get("avg_logprob")),
            "ignored": bool(asr.get("ignored", False)),
            "reason": asr.get("reason"),
            "model": model,
            "task": task,
        }
        if task == "translate":
            body["tts_language"] = "en"
        return body, "application/json"
    return {"text": text}, "application/json"


def build_voices_listing(engines: Iterable[Mapping], *, supertonic: Mapping | None = None,
                         piper_plus: Mapping | None = None, piper_tts: Mapping | None = None,
                         default_engine: str | None = None) -> dict:
    """Assemble ``GET /v1/audio/voices``.

    ``engines`` is the router's list (``key``, ``label``, ``loaded``); the keyword
    arguments carry each engine's ``languages`` (iterable) and ``voices`` (list
    of dicts with at least ``id`` and ``label``). Engines without a server-side
    voice (the browser) are omitted.
    """
    details = {"supertonic": supertonic, "piper-plus": piper_plus, "piper-tts": piper_tts}
    listed = []
    all_languages: set[str] = set()
    for engine in engines:
        key = str(engine.get("key") or "")
        info = details.get(key)
        if info is None:
            continue
        languages = sorted({str(lang) for lang in (info.get("languages") or [])})
        voices = [dict(v) for v in (info.get("voices") or [])]
        all_languages.update(languages)
        listed.append({
            "key": key,
            "label": str(engine.get("label") or key),
            "loaded": bool(engine.get("loaded", False)),
            "languages": languages,
            "voices": voices,
        })
    keys = {e["key"] for e in listed}
    return {
        "default_engine": default_engine if default_engine in keys else None,
        "languages": sorted(all_languages),
        "engines": listed,
    }


def curl_for_speech(base_url: str, request: Mapping) -> str:
    """A copy-pasteable curl line equivalent to a speech request (for docs/harness)."""
    body = json.dumps(dict(request), ensure_ascii=False)
    return (f"curl -k -X POST {base_url}/v1/audio/speech -H 'Content-Type: application/json' "
            f"-d '{body}' -o speech.wav")
