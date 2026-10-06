"""Contract tests for the OpenAI-compatible audio API helpers (audio_api.py).

Pure Python: no Flask, no engines, no board.
"""

from __future__ import annotations

import unittest

from asr_metadata import analyze_transcription
from audio_api import (
    MAX_INPUT_CHARS,
    MAX_TRANSCRIPTION_BYTES,
    AudioApiError,
    build_voices_listing,
    curl_for_speech,
    format_transcription,
    parse_speech_request,
    parse_transcription_form,
)


class SpeechRequestTests(unittest.TestCase):
    def test_defaults(self):
        req = parse_speech_request({"input": "Hello"})
        self.assertEqual((req.text, req.model, req.voice, req.language, req.speed,
                          req.response_format, req.used_speed_alias),
                         ("Hello", "default", "default", "en", 1.0, "wav", False))

    def test_auto_or_blank_language_means_english(self):
        for value in ("auto", "AUTO", "", None, "  "):
            self.assertEqual(parse_speech_request({"input": "a", "language": value}).language, "en")
        self.assertEqual(parse_speech_request({"input": "a", "language": "DE"}).language, "de")

    def test_model_is_an_engine_or_the_router(self):
        for value, expect in (("supertonic", "supertonic"), ("MLA", "supertonic"), ("piper", "piper-tts"),
                              ("piperplus", "piper-plus"), ("tts-1", "default"), ("gpt-4o-mini-tts", "default"),
                              ("", "default"), (None, "default"), ("Default", "default")):
            self.assertEqual(parse_speech_request({"input": "a", "model": value}).model, expect, value)

    def test_unknown_model_is_400_on_model(self):
        for value in ("supertonik", "whisper-small", "piper_tts"):
            with self.assertRaises(AudioApiError) as ctx:
                parse_speech_request({"input": "a", "model": value})
            self.assertEqual((ctx.exception.status, ctx.exception.param), (400, "model"), value)

    def test_missing_or_blank_input_is_400_on_input(self):
        for body in ({}, {"input": ""}, {"input": "   "}, {"input": 5}, None, "text"):
            with self.assertRaises(AudioApiError) as ctx:
                parse_speech_request(body)
            self.assertEqual(ctx.exception.status, 400)
            if isinstance(body, dict) and "input" in body:
                self.assertEqual(ctx.exception.param, "input")

    def test_input_length_limit(self):
        parse_speech_request({"input": "x" * MAX_INPUT_CHARS})
        with self.assertRaises(AudioApiError) as ctx:
            parse_speech_request({"input": "x" * (MAX_INPUT_CHARS + 1)})
        self.assertEqual(ctx.exception.param, "input")

    def test_speed_bounds_and_types(self):
        self.assertEqual(parse_speech_request({"input": "a", "speed": 4.0}).speed, 4.0)
        self.assertEqual(parse_speech_request({"input": "a", "speed": "1.5"}).speed, 1.5)
        for bad in (0.24, 4.01, "fast", float("nan")):
            with self.assertRaises(AudioApiError) as ctx:
                parse_speech_request({"input": "a", "speed": bad})
            self.assertEqual((ctx.exception.status, ctx.exception.param), (400, "speed"))

    def test_deprecated_speed_aliases_are_honoured_but_speed_wins(self):
        req = parse_speech_request({"input": "a", "utterance_speed": 0.8})
        self.assertEqual((req.speed, req.used_speed_alias), (0.8, True))
        req = parse_speech_request({"input": "a", "utteranceSpeed": 1.2})
        self.assertEqual((req.speed, req.used_speed_alias), (1.2, True))
        req = parse_speech_request({"input": "a", "speed": 2.0, "utterance_speed": 0.8})
        self.assertEqual((req.speed, req.used_speed_alias), (2.0, False))

    def test_only_wav_is_produced(self):
        self.assertEqual(parse_speech_request({"input": "a", "response_format": "WAV"}).response_format, "wav")
        with self.assertRaises(AudioApiError) as ctx:
            parse_speech_request({"input": "a", "response_format": "mp3"})
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("wav", ctx.exception.message)
        self.assertEqual(ctx.exception.payload()["param"], "response_format")

    def test_curl_equivalent(self):
        line = curl_for_speech("https://box:5000", {"input": "hi", "model": "supertonic"})
        self.assertTrue(line.startswith("curl -k -X POST https://box:5000/v1/audio/speech"))
        self.assertIn('"model": "supertonic"', line)


class TranscriptionFormTests(unittest.TestCase):
    def test_file_is_required(self):
        with self.assertRaises(AudioApiError) as ctx:
            parse_transcription_form({}, has_file=False, size=10)
        self.assertEqual((ctx.exception.status, ctx.exception.param), (400, "file"))

    def test_size_limit_is_413(self):
        with self.assertRaises(AudioApiError) as ctx:
            parse_transcription_form({}, has_file=True, size=MAX_TRANSCRIPTION_BYTES + 1)
        self.assertEqual(ctx.exception.status, 413)
        parse_transcription_form({}, has_file=True, size=MAX_TRANSCRIPTION_BYTES)
        parse_transcription_form({}, has_file=True, size=None)   # unknown length is checked later

    def test_defaults_and_formats(self):
        req = parse_transcription_form({}, has_file=True, size=1)
        self.assertEqual((req.model, req.language, req.response_format), (None, "auto", "json"))
        req = parse_transcription_form(
            {"model": " whisper-small-a16w8 ", "language": "DE", "response_format": "verbose_json"},
            has_file=True, size=1)
        self.assertEqual((req.model, req.language, req.response_format),
                         ("whisper-small-a16w8", "de", "verbose_json"))
        with self.assertRaises(AudioApiError) as ctx:
            parse_transcription_form({"response_format": "srt"}, has_file=True, size=1)
        self.assertEqual(ctx.exception.param, "response_format")


class FormatTranscriptionTests(unittest.TestCase):
    RESULT = {"text": "  hello world ", "language": "en", "no_speech_prob": 0.01, "avg_logprob": -0.2}

    def test_json_is_text_only(self):
        body, mimetype = format_transcription(self.RESULT, None, "json", model="m")
        self.assertEqual((body, mimetype), ({"text": "hello world"}, "application/json"))

    def test_text_is_plain(self):
        body, mimetype = format_transcription(self.RESULT, None, "text")
        self.assertEqual((body, mimetype), ("hello world\n", "text/plain; charset=utf-8"))

    def test_verbose_json_carries_the_studio_analysis(self):
        asr = analyze_transcription(self.RESULT, requested_language="auto",
                                    supported_tts_languages=("en", "de"))
        body, _ = format_transcription(self.RESULT, asr, "verbose_json", model="whisper")
        self.assertEqual(body["text"], "hello world")
        self.assertEqual(body["language"], "en")
        self.assertTrue(body["language_detected"])
        self.assertEqual(body["tts_language"], "en")
        self.assertAlmostEqual(body["no_speech_prob"], 0.01)
        self.assertFalse(body["ignored"])
        self.assertIsNone(body["reason"])
        self.assertEqual(body["model"], "whisper")

    def test_default_task_is_transcribe(self):
        body, _ = format_transcription(self.RESULT, None, "verbose_json")
        self.assertEqual(body["task"], "transcribe")

    def test_translate_keeps_source_language_and_speaks_english(self):
        result = {"text": " Good morning! ", "language": "de", "no_speech_prob": 0.02, "avg_logprob": -0.27}
        asr = analyze_transcription(result, requested_language="auto",
                                    supported_tts_languages=("en", "de"))
        body, _ = format_transcription(result, asr, "verbose_json", model="whisper", task="translate")
        self.assertEqual((body["text"], body["task"], body["language"], body["tts_language"]),
                         ("Good morning!", "translate", "de", "en"))

    def test_json_and_text_ignore_the_task(self):
        self.assertEqual(format_transcription(self.RESULT, None, "json", task="translate"),
                         ({"text": "hello world"}, "application/json"))
        self.assertEqual(format_transcription(self.RESULT, None, "text", task="translate")[0], "hello world\n")

    def test_verbose_json_without_analysis_falls_back_to_the_raw_fields(self):
        body, _ = format_transcription(self.RESULT, None, "verbose_json")
        self.assertEqual(body["language"], "en")
        self.assertAlmostEqual(body["avg_logprob"], -0.2)


class VoicesListingTests(unittest.TestCase):
    ENGINES = [
        {"key": "supertonic", "label": "Supertonic 3", "loaded": True},
        {"key": "piper-tts", "label": "piper-tts", "loaded": False},
        {"key": "browser", "label": "Browser", "loaded": True},
    ]

    def test_browser_is_dropped_and_languages_are_the_sorted_union(self):
        listing = build_voices_listing(
            self.ENGINES,
            supertonic={"languages": ["ko", "en"], "voices": [{"id": "M1", "label": "M1"}]},
            piper_tts={"languages": ["zh", "en"], "voices": [{"id": "zh_CN-huayan-medium", "label": "Huayan", "language": "zh"}]},
            default_engine="supertonic")
        self.assertEqual([e["key"] for e in listing["engines"]], ["supertonic", "piper-tts"])
        self.assertEqual(listing["languages"], ["en", "ko", "zh"])
        self.assertEqual(listing["engines"][0]["languages"], ["en", "ko"])
        self.assertFalse(listing["engines"][1]["loaded"])
        self.assertEqual(listing["engines"][1]["voices"][0]["language"], "zh")
        self.assertEqual(listing["default_engine"], "supertonic")

    def test_unknown_default_engine_is_null(self):
        listing = build_voices_listing(self.ENGINES, supertonic={"languages": [], "voices": []},
                                       default_engine="piper-plus")
        self.assertIsNone(listing["default_engine"])
        self.assertEqual(listing["engines"][0]["voices"], [])


if __name__ == "__main__":
    unittest.main()
