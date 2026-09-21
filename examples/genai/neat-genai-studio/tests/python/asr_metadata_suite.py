import unittest

from asr_metadata import analyze_transcription, normalize_language_code


class AsrMetadataTests(unittest.TestCase):
    def test_normalizes_language_names_and_bcp47_codes(self):
        self.assertEqual(normalize_language_code("English"), "en")
        self.assertEqual(normalize_language_code("pt-BR"), "pt")
        self.assertEqual(normalize_language_code("zh_CN"), "zh")

    def test_accepts_voiced_transcription_and_routes_detected_language(self):
        result = analyze_transcription(
            {
                "text": "Hola",
                "language": "es",
                "no_speech_prob": 0.02,
                "avg_logprob": -0.15,
            },
            requested_language="auto",
            supported_tts_languages=("en", "es"),
        )

        self.assertFalse(result["ignored"])
        self.assertTrue(result["language_detected"])
        self.assertEqual(result["tts_language"], "es")

    def test_rejects_low_logprob_even_when_no_speech_probability_is_low(self):
        result = analyze_transcription(
            {
                "text": "Thank you.",
                "language": "en",
                "no_speech_prob": 0.13,
                "avg_logprob": -1.4,
            },
            requested_language="auto",
            supported_tts_languages=("en",),
        )

        self.assertTrue(result["ignored"])
        self.assertEqual(result["reason"], "low_confidence")

    def test_rejects_high_no_speech_despite_good_logprob(self):
        result = analyze_transcription(
            {
                "text": "quiet but intelligible",
                "language": "en",
                "no_speech_prob": 0.91,
                "avg_logprob": -0.2,
            },
            requested_language="auto",
            supported_tts_languages=("en",),
        )

        self.assertTrue(result["ignored"])
        self.assertEqual(result["reason"], "no_speech")

    def test_logprob_at_threshold_does_not_rescue_no_speech(self):
        result = analyze_transcription(
            {
                "text": "borderline",
                "language": "en",
                "no_speech_prob": 0.91,
                "avg_logprob": -1.0,
            },
            requested_language="auto",
            supported_tts_languages=("en",),
        )

        self.assertTrue(result["ignored"])

    def test_thresholds_are_configurable(self):
        result = analyze_transcription(
            {
                "text": "quiet speech",
                "language": "en",
                "no_speech_prob": 0.7,
                "avg_logprob": -1.2,
            },
            requested_language="auto",
            supported_tts_languages=("en",),
            no_speech_threshold=0.8,
            logprob_threshold=-1.3,
        )

        self.assertFalse(result["ignored"])

    def test_high_no_speech_is_used_when_logprobe_is_unavailable(self):
        result = analyze_transcription(
            {
                "text": "hallucinated text",
                "language": "en",
                "no_speech_prob": 0.8,
            },
            requested_language="auto",
            supported_tts_languages=("en",),
        )

        self.assertTrue(result["ignored"])

    def test_explicit_language_is_not_reported_as_detected(self):
        result = analyze_transcription(
            {"text": "Bonjour", "language": "fr"},
            requested_language="fr",
            supported_tts_languages=("en", "fr"),
        )

        self.assertFalse(result["language_detected"])
        self.assertEqual(result["tts_language"], "fr")

    def test_unsupported_detected_language_disables_tts_routing(self):
        result = analyze_transcription(
            {"text": "Cze\u015b\u0107", "language": "pl"},
            requested_language="auto",
            supported_tts_languages=("en", "de"),
        )

        self.assertFalse(result["ignored"])
        self.assertIsNone(result["tts_language"])


if __name__ == "__main__":
    unittest.main()
