"""Unit tests for the Supertonic client module (no board, no worker process).

Covers the text segmenter that keeps every request inside the compiled
192-character contract, environment discovery, and the worker-free parts of the
client. Nothing here spawns the worker or needs pyneat.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import supertonic_tts
from supertonic_tts import MAX_MODEL_CHARS, segment_text


def _limit(language):
    return MAX_MODEL_CHARS - len(f"<{language}></{language}>")


def _assert_all_fit(case, segments, language):
    limit = _limit(language)
    for segment in segments:
        # The engine appends a period when a segment lacks ending punctuation.
        effective = len(segment) + (0 if supertonic_tts._ENDING_PUNCTUATION.search(segment) else 1)
        case.assertLessEqual(effective, limit, segment)
        case.assertEqual(segment, segment.strip())


class SegmentTextTests(unittest.TestCase):
    def test_blank_input_yields_no_segments(self):
        self.assertEqual(segment_text("", "en"), [])
        self.assertEqual(segment_text("   \n\t ", "en"), [])
        self.assertEqual(segment_text(None, "en"), [])

    def test_short_text_is_one_segment_and_whitespace_is_collapsed(self):
        self.assertEqual(
            segment_text("Hello   from\nModalix.", "en"), ["Hello from Modalix."]
        )

    def test_sentences_are_packed_while_they_fit(self):
        text = "First sentence. Second one! Third? Fourth: fifth; done."
        self.assertEqual(segment_text(text, "en"), [text])

    def test_sentences_are_split_at_boundaries_when_they_do_not_fit(self):
        sentence = "This sentence has exactly enough words to be long. "
        text = sentence * 8
        segments = segment_text(text, "en")
        self.assertGreater(len(segments), 1)
        _assert_all_fit(self, segments, "en")
        # Every segment ends on a sentence boundary, nothing is lost.
        for segment in segments:
            self.assertTrue(segment.endswith("."), segment)
        self.assertEqual(" ".join(segments), text.strip())

    def test_oversized_sentence_prefers_comma_then_whitespace(self):
        clause = "a clause with a few words in it, "
        text = (clause * 10).rstrip(", ") + "."
        segments = segment_text(text, "en")
        self.assertGreater(len(segments), 1)
        _assert_all_fit(self, segments, "en")
        self.assertTrue(all(s.endswith(",") for s in segments[:-1]), segments)

        words = "word " * 80
        segments = segment_text(words.strip(), "en")
        self.assertGreater(len(segments), 1)
        _assert_all_fit(self, segments, "en")
        self.assertEqual(" ".join(segments).split(), words.split())

    def test_unbroken_run_is_hard_cut(self):
        text = "x" * 500
        segments = segment_text(text, "en")
        _assert_all_fit(self, segments, "en")
        self.assertEqual("".join(segments), text)

    def test_cjk_boundaries_and_closers_stay_attached(self):
        # NFKD folds full-width ！？ to ASCII, as the engine's own preprocessing
        # does; sentences that fit are packed into one segment.
        import unicodedata
        text = "これは最初の文です。「二番目」です！三番目ですか？"
        expected = unicodedata.normalize(
            "NFKD", "これは最初の文です。 「二番目」です! 三番目ですか?"
        )
        self.assertEqual(segment_text(text, "ja"), [expected])
        long_text = "これは長い文です。" * 40
        segments = segment_text(long_text, "ja")
        self.assertGreater(len(segments), 1)
        _assert_all_fit(self, segments, "ja")
        for segment in segments:
            self.assertTrue(segment.endswith("。"), segment)

    def test_language_wrapper_length_reduces_budget(self):
        text = "w" * 186   # fits <en></en> (183) + period? no: 187 > 183
        en = segment_text(text, "en")
        self.assertEqual(len(en), 2)
        _assert_all_fit(self, en, "en")
        # A one-character code has a two-character-shorter wrapper.
        text = "w" * 180 + "."
        self.assertEqual(segment_text(text, "en"), [text])
        self.assertEqual(len(segment_text(text, "pt-br"), ), 2)


class EnvironmentDiscoveryTests(unittest.TestCase):
    def test_not_available_when_paths_are_missing(self):
        with mock.patch.dict(os.environ, {
            "SUPERTONIC_PYTHON": "/nonexistent/python",
            "SUPERTONIC_REPO_ROOT": "/nonexistent/repo",
            "SUPERTONIC_APP_ROOT": "/nonexistent/app",
        }):
            self.assertIsNone(supertonic_tts._supertonic_python())
            self.assertFalse(supertonic_tts.available())

    def test_available_requires_venv_checkout_and_models(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app"
            repo = root / "repo"
            (app / ".venv" / "bin").mkdir(parents=True)
            (app / ".venv" / "bin" / "python").write_text("")
            env = {
                "SUPERTONIC_REPO_ROOT": str(repo),
                "SUPERTONIC_APP_ROOT": str(app),
            }
            env.pop("SUPERTONIC_PYTHON", None)
            with mock.patch.dict(os.environ, env, clear=False):
                os.environ.pop("SUPERTONIC_PYTHON", None)
                self.assertEqual(
                    supertonic_tts._supertonic_python(),
                    str(app / ".venv" / "bin" / "python"),
                )
                self.assertFalse(supertonic_tts.available())   # no checkout yet
                (repo / "app" / "supertonic_sima").mkdir(parents=True)
                (repo / "app" / "supertonic_sima" / "__init__.py").write_text("")
                self.assertFalse(supertonic_tts.available())   # no models yet
                (app / "models" / "supertonic-3" / "onnx").mkdir(parents=True)
                (app / "models" / "supertonic-3" / "onnx" / "tts.json").write_text("{}")
                (app / "models" / "supertonic-3-sima").mkdir(parents=True)
                (app / "models" / "supertonic-3-sima"
                 / "supertonic_vector_field_sima_mpk.tar.gz").write_bytes(b"")
                self.assertTrue(supertonic_tts.available())

    def test_worker_spawn_fails_clearly_without_runtime(self):
        with mock.patch.dict(os.environ, {"SUPERTONIC_PYTHON": "/nonexistent/python"}):
            supertonic_tts.shutdown_worker()
            with self.assertRaises(RuntimeError) as ctx:
                supertonic_tts.SupertonicTTS()
            self.assertIn("Supertonic runtime venv not found", str(ctx.exception))


class ClientConfigurationTests(unittest.TestCase):
    def _client(self):
        # Build a client without touching the worker.
        with mock.patch.object(supertonic_tts, "_request", return_value=(
            b'{"languages": ["en", "ko", "ja"], "voices": ["F1", "M1"],'
            b' "sample_rate": 44100, "steps": 8, "min_speed": 0.7, "max_speed": 2.0}'
        )):
            return supertonic_tts.SupertonicTTS(voice="M1")

    def test_load_reply_configures_client(self):
        tts = self._client()
        self.assertEqual(tts.voice, "M1")
        self.assertEqual(tts.sample_rate, 44100)
        self.assertTrue(tts.supports("ko"))
        self.assertFalse(tts.supports("no"))
        self.assertFalse(tts.supports(None))

    def test_unknown_default_voice_falls_back_to_first(self):
        with mock.patch.object(supertonic_tts, "_request", return_value=(
            b'{"languages": ["en"], "voices": ["F3", "M2"]}'
        )):
            self.assertEqual(supertonic_tts.SupertonicTTS(voice="Z9").voice, "F3")

    def test_speed_is_clamped_to_model_contract(self):
        tts = self._client()
        tts.set_utterance_speed(0.5)      # Studio slider minimum
        self.assertAlmostEqual(tts.speed, 0.7)
        tts.set_utterance_speed(3)
        self.assertAlmostEqual(tts.speed, 2.0)
        tts.set_utterance_speed("nonsense")
        self.assertAlmostEqual(tts.speed, 1.0)

    def test_set_voice_rejects_unknown(self):
        tts = self._client()
        self.assertTrue(tts.set_voice("F1"))
        self.assertFalse(tts.set_voice("F9"))
        self.assertEqual(tts.voice, "F1")

    def test_stream_request_carries_segments_voice_and_language(self):
        tts = self._client()
        tts.set_voice("F1")
        captured = {}

        def fake_stream(req):
            captured.update(req)
            yield b"RIFF-one"
            yield b"RIFF-two"

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=fake_stream):
            chunks = [c.getvalue() for c in tts.synthesize_stream("Hi there. Bye.", language="ko")]
        self.assertEqual(chunks, [b"RIFF-one", b"RIFF-two"])
        self.assertEqual(captured["cmd"], "synth_stream")
        self.assertEqual(captured["segments"], ["Hi there. Bye."])
        self.assertEqual(captured["voice"], "F1")
        self.assertEqual(captured["language"], "ko")

    def test_per_call_voice_override_does_not_stick(self):
        tts = self._client()
        captured = {}

        def fake_stream(req):
            captured.update(req)
            yield b"RIFF"

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=fake_stream):
            list(tts.synthesize_stream("Hello.", language="en", voice="F1"))
            self.assertEqual(captured["voice"], "F1")
            list(tts.synthesize_stream("Hello.", language="en", voice="nope"))
            self.assertEqual(captured["voice"], "M1")
        self.assertEqual(tts.voice, "M1")

    def test_blank_text_makes_no_request(self):
        tts = self._client()
        with mock.patch.object(supertonic_tts, "_request_stream") as stream:
            self.assertEqual(list(tts.synthesize_stream("   ", language="en")), [])
            self.assertEqual(tts.synthesize("", language="en").getvalue(), b"")
        stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
