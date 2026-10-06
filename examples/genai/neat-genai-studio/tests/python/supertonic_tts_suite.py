"""Unit tests for the Supertonic client module (no board, no worker process).

Lives under tests/python (imported by test_unit.py) so the installed bundle,
which packages src/python wholesale, does not ship it.

Covers the text segmenter that keeps every request inside the compiled
192-character contract, environment discovery, and the worker-free parts of the
client. Nothing here spawns the worker or needs pyneat.
"""

from __future__ import annotations

import io
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

    def test_comma_at_the_budget_boundary_never_exceeds_the_contract(self):
        limit = _limit("en")
        for offset in (-2, -1, 0, 1, 2):
            text = "a" * (limit + offset) + "," + " more words follow here."
            segments = segment_text(text, "en")
            _assert_all_fit(self, segments, "en")
            self.assertEqual("".join(segments).replace(" ", ""), text.replace(" ", ""))

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


class DurationFallbackTests(unittest.TestCase):
    class Engine:
        """Rejects anything longer than `limit` the way the real engine does
        when the predicted latent length exceeds 192 frames."""
        def __init__(self, limit):
            self.limit, self.calls = limit, []

        def synthesize(self, text, **kwargs):
            self.calls.append(text)
            if len(text) > self.limit:
                raise ValueError("predicted latent length 234 exceeds static limit 192")
            return text

    def _run(self, engine, text):
        return list(supertonic_tts.synthesize_fitting(
            engine, text, voice="M1", language="en", speed=1.0, seed=1))

    def test_fitting_text_is_synthesized_once(self):
        engine = self.Engine(100)
        self.assertEqual(self._run(engine, "short enough"), ["short enough"])
        self.assertEqual(engine.calls, ["short enough"])

    def test_too_long_text_is_split_at_whitespace_until_it_fits(self):
        engine = self.Engine(12)
        pieces = self._run(engine, "one two three four five six seven")
        self.assertEqual(" ".join(pieces), "one two three four five six seven")
        self.assertTrue(all(len(p) <= 12 for p in pieces))

    def test_comma_is_preferred_and_kept(self):
        self.assertEqual(supertonic_tts._halves("first clause, second clause"),
                         ("first clause,", "second clause"))

    def test_unsplittable_word_is_hard_cut(self):
        engine = self.Engine(8)
        pieces = self._run(engine, "a" * 20)
        self.assertEqual("".join(pieces), "a" * 20)
        self.assertTrue(all(len(p) <= 8 for p in pieces))

    def test_other_errors_propagate(self):
        class Broken:
            def synthesize(self, text, **kwargs):
                raise ValueError("unsupported language 'xx'")
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            self._run(Broken(), "hello there")


class EnvironmentDiscoveryTests(unittest.TestCase):
    def test_not_available_when_paths_are_missing(self):
        with mock.patch.dict(os.environ, {
            "SUPERTONIC_PYTHON": "/nonexistent/python",
            "SUPERTONIC_MODELS_ROOT": "/nonexistent/models",
        }):
            self.assertIsNone(supertonic_tts._supertonic_python())
            self.assertFalse(supertonic_tts.available())

    def test_available_requires_venv_and_models(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv_py = root / ".venv-supertonic" / "bin" / "python"
            venv_py.parent.mkdir(parents=True)
            venv_py.write_text("")
            models = root / "models"
            with mock.patch.dict(os.environ, {
                "SUPERTONIC_PYTHON": str(venv_py),
                "SUPERTONIC_MODELS_ROOT": str(models),
            }):
                self.assertEqual(supertonic_tts._supertonic_python(), str(venv_py))
                self.assertEqual(supertonic_tts.models_root(), models)
                self.assertFalse(supertonic_tts.available())   # no models yet
                (models / "supertonic-3" / "onnx").mkdir(parents=True)
                (models / "supertonic-3" / "onnx" / "tts.json").write_text("{}")
                (models / "supertonic-3-sima").mkdir(parents=True)
                (models / "supertonic-3-sima"
                 / "supertonic_vector_field_sima_mpk.tar.gz").write_bytes(b"")
                self.assertTrue(supertonic_tts.available())

    def test_supertonic_venv_env_names_the_interpreter(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "custom-venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("")
            with mock.patch.dict(os.environ, {"SUPERTONIC_VENV": str(venv)}, clear=False):
                os.environ.pop("SUPERTONIC_PYTHON", None)
                self.assertEqual(supertonic_tts._supertonic_python(), str(venv / "bin" / "python"))
            with mock.patch.dict(os.environ, {"SUPERTONIC_VENV": str(Path(tmp) / "missing")}, clear=False):
                os.environ.pop("SUPERTONIC_PYTHON", None)
                self.assertIsNone(supertonic_tts._supertonic_python())

    def test_legacy_app_root_venv_is_used_until_migrated(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "supertonic-tts"
            (legacy / ".venv" / "bin").mkdir(parents=True)
            (legacy / ".venv" / "bin" / "python").write_text("")
            with mock.patch.object(supertonic_tts, "DEFAULT_VENV", Path(tmp) / "missing"), \
                 mock.patch.dict(os.environ, {"SUPERTONIC_APP_ROOT": str(legacy)}, clear=False):
                os.environ.pop("SUPERTONIC_PYTHON", None); os.environ.pop("SUPERTONIC_VENV", None)
                self.assertEqual(supertonic_tts._supertonic_python(), str(legacy / ".venv" / "bin" / "python"))

    def test_default_venv_is_beside_the_example(self):
        # src/python/ui/supertonic_tts.py -> <example>/.venv-supertonic
        self.assertEqual(supertonic_tts.DEFAULT_VENV.name, ".venv-supertonic")
        self.assertTrue((supertonic_tts.DEFAULT_VENV.parent / "setup.sh").is_file())

    def test_legacy_app_root_maps_to_its_models_subdir(self):
        env = {"SUPERTONIC_APP_ROOT": "/data/old-supertonic"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SUPERTONIC_MODELS_ROOT", None)
            self.assertEqual(supertonic_tts.models_root(), Path("/data/old-supertonic/models"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SUPERTONIC_MODELS_ROOT", None)
            os.environ.pop("SUPERTONIC_APP_ROOT", None)
            self.assertEqual(supertonic_tts.models_root(), Path(supertonic_tts.DEFAULT_MODELS_ROOT))

    def test_worker_spawn_fails_clearly_without_runtime(self):
        with mock.patch.dict(os.environ, {"SUPERTONIC_PYTHON": "/nonexistent/python"}):
            supertonic_tts.shutdown_worker()
            with self.assertRaises(RuntimeError) as ctx:
                supertonic_tts.SupertonicTTS()
            self.assertIn("Supertonic runtime venv not found", str(ctx.exception))


class VendoredRuntimeTests(unittest.TestCase):
    """The vendored supertonic_sima package: importable pieces without the
    runtime, and no absolute paths left behind."""

    PACKAGE = Path(supertonic_tts.__file__).resolve().parent / "supertonic_sima"

    def _load(self, name):
        # Load a single module by file so the package __init__ (which pulls in
        # the engine and therefore pyneat) is not executed on a host.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            f"vendored_{name}", self.PACKAGE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_runtime_closure_is_present_with_provenance_headers(self):
        for name in ("__init__", "audio", "engine", "inputs", "text"):
            path = self.PACKAGE / f"{name}.py"
            self.assertTrue(path.is_file(), path)
            head = path.read_text(encoding="utf-8")[:400]
            self.assertIn("Vendored from", head)
            self.assertIn("3b837b3e1b6a378ab8c24c3c04b079429b67e237", head)

    def test_text_contract_matches_the_client(self):
        text = self._load("text")
        self.assertEqual(tuple(text.AVAILABLE_VOICES), supertonic_tts.VOICES)
        self.assertAlmostEqual(text.MIN_SPEED, 0.7)
        self.assertAlmostEqual(text.MAX_SPEED, 2.0)
        self.assertIn("ko", text.AVAILABLE_LANGUAGES)
        self.assertNotIn("zh", text.AVAILABLE_LANGUAGES)

    def test_emoji_pattern_strips_emoji_and_keeps_text(self):
        # The class was rewritten as ordered ranges (CodeQL); same code points.
        text = self._load("text")
        strip = text._EMOJI_PATTERN.sub
        self.assertEqual(strip("", "hi \U0001f600\U0001f1fa\U0001f1f8 there \u2600"), "hi  there ")
        self.assertEqual(strip("", "caf\u00e9 \u2013 na\u00efve 100%"), "caf\u00e9 \u2013 na\u00efve 100%")
        for cp in (0x2600, 0x26ff, 0x2700, 0x27bf, 0x1f1e6, 0x1f1ff, 0x1f300, 0x1f5ff, 0x1f600, 0x1f64f,
                   0x1f680, 0x1f6ff, 0x1f700, 0x1f77f, 0x1f780, 0x1f7ff, 0x1f800, 0x1f8ff, 0x1f900, 0x1f9ff,
                   0x1fa00, 0x1fa6f, 0x1fa70, 0x1faff):
            self.assertTrue(text._EMOJI_PATTERN.fullmatch(chr(cp)), hex(cp))
        for cp in (0x25ff, 0x27c0, 0x1f1e5, 0x1f200, 0x1fb00):
            self.assertIsNone(text._EMOJI_PATTERN.fullmatch(chr(cp)), hex(cp))

    def test_no_absolute_paths_in_the_vendored_code(self):
        for path in self.PACKAGE.glob("*.py"):
            self.assertNotIn("/media/nvme", path.read_text(encoding="utf-8"), path)


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

    def test_dead_worker_is_retried_once_before_any_audio(self):
        tts = self._client()
        calls = []

        def failing_then_ok(req):
            calls.append(req)
            if len(calls) == 1:
                raise supertonic_tts.WorkerDied("NeatError: accelerator_execution_failed")
                yield  # pragma: no cover - makes this a generator
            yield b"RIFF-recovered"

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=failing_then_ok):
            chunks = [c.getvalue() for c in tts.synthesize_stream("Hello.", language="en")]
        self.assertEqual(chunks, [b"RIFF-recovered"])
        self.assertEqual(len(calls), 2)

    def test_failure_after_audio_or_with_live_worker_is_not_retried(self):
        tts = self._client()

        def mid_stream_failure(req):
            yield b"RIFF-first"
            raise supertonic_tts.WorkerDied("boom")

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=mid_stream_failure):
            with self.assertRaises(supertonic_tts.WorkerDied):
                list(tts.synthesize_stream("Hello.", language="en"))

        calls = []

        def bad_request(req):
            calls.append(req)
            raise RuntimeError("ValueError: text too long")   # status 1: worker alive
            yield  # pragma: no cover

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=bad_request):
            with self.assertRaises(RuntimeError):
                list(tts.synthesize_stream("Hello.", language="en"))
        self.assertEqual(len(calls), 1)

    def test_fatal_frame_reaps_worker_and_raises_worker_died(self):
        import struct

        class FakeProc:
            def __init__(self, frames):
                self.stdin = io.BytesIO()
                self.stdout = io.BytesIO(b"".join(
                    bytes([st]) + struct.pack(">I", len(body)) + body for st, body in frames))
                self.killed = False

            def poll(self):
                return None

            def kill(self):
                self.killed = True

            def wait(self, timeout=None):
                return 0

        proc = FakeProc([(2, b"RIFF-a"), (3, b"NeatError: accelerator_execution_failed")])
        with mock.patch.object(supertonic_tts, "_ensure_worker", return_value=proc):
            supertonic_tts._worker = proc
            stream = supertonic_tts._request_stream({"cmd": "synth_stream"})
            self.assertEqual(next(stream), b"RIFF-a")
            with self.assertRaises(supertonic_tts.WorkerDied):
                next(stream)
        self.assertTrue(proc.killed)
        self.assertIsNone(supertonic_tts._worker)

    def test_per_call_speed_is_clamped_and_does_not_stick(self):
        tts = self._client()
        tts.set_utterance_speed(1.0)
        captured = {}

        def fake_stream(req):
            captured.update(req)
            yield b"RIFF"

        with mock.patch.object(supertonic_tts, "_request_stream", side_effect=fake_stream):
            list(tts.synthesize_stream("Hello.", language="en", speed=3.5))
            self.assertAlmostEqual(captured["speed"], 2.0)     # clamped to the contract
            list(tts.synthesize_stream("Hello.", language="en", speed=0.25))
            self.assertAlmostEqual(captured["speed"], 0.7)
            list(tts.synthesize_stream("Hello.", language="en"))
            self.assertAlmostEqual(captured["speed"], 1.0)     # configured speed unchanged
        self.assertAlmostEqual(tts.speed, 1.0)
        self.assertAlmostEqual(tts.clamp_speed("garbage"), 1.0)

    def test_blank_text_makes_no_request(self):
        tts = self._client()
        with mock.patch.object(supertonic_tts, "_request_stream") as stream:
            self.assertEqual(list(tts.synthesize_stream("   ", language="en")), [])
            self.assertEqual(tts.synthesize("", language="en").getvalue(), b"")
        stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
