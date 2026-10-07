"""Backend-only mode policy (backend_mode.py): served paths, CORS allowlist,
headers and the /health payload. Pure Python: no Flask."""

from __future__ import annotations

import unittest

from backend_mode import (
    cors_headers,
    cors_path,
    effective_cors_setting,
    health_payload,
    normalize_origin,
    origin_allowed,
    parse_cors_origins,
    path_allowed,
)


class BackendPathTests(unittest.TestCase):
    def test_api_surface_is_served(self):
        for path in ("/", "/health", "/v1/chat/completions", "/v1/audio/speech",
                     "/v1/audio/translations", "/audio/transcriptions", "/models/status",
                     "/models/load", "/benchmark/run", "/tts/engine", "/voices",
                     "/voices/select", "/supertonic/voice", "/piperplus/voices", "/shutdown"):
            self.assertTrue(path_allowed(path), path)

    def test_web_ui_and_studio_chat_are_not(self):
        for path in ("/playground/", "/playground/index.html", "/solutions/", "/showcase",
                     "/config.js", "/static/newui.js", "/upload", "/stop", "/clear-history",
                     "/upload-to-rag", "/rag/inspect", "/board-camera/devices", "/voicesx",
                     "/v1", "/modelsx"):
            self.assertFalse(path_allowed(path), path)

    def test_shutdown_is_never_cross_origin(self):
        self.assertFalse(cors_path("/shutdown"))
        self.assertFalse(cors_path("/"))
        self.assertTrue(cors_path("/v1/audio/speech"))
        self.assertTrue(cors_path("/health"))


class CorsPolicyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_cors_origins(None), ())
        self.assertEqual(parse_cors_origins("  "), ())
        self.assertEqual(parse_cors_origins("*"), "*")
        self.assertEqual(parse_cors_origins("http://a:3000, https://B.example/ http://a:3000 junk ftp://x"),
                         ("http://a:3000", "https://b.example"))

    def test_normalize(self):
        self.assertEqual(normalize_origin("HTTPS://Host:8443/"), "https://host:8443")
        self.assertEqual(normalize_origin("null"), "")
        self.assertEqual(normalize_origin(""), "")

    def test_allowed(self):
        allow = parse_cors_origins("http://insight.local:8080")
        self.assertTrue(origin_allowed("http://insight.local:8080", allow))
        self.assertTrue(origin_allowed("http://INSIGHT.local:8080/", allow))
        self.assertFalse(origin_allowed("http://insight.local:9090", allow))
        self.assertFalse(origin_allowed("null", "*"))
        self.assertTrue(origin_allowed("https://anything", "*"))
        self.assertFalse(origin_allowed("https://anything", ()))

    def test_environment_overrides_the_config_even_when_empty(self):
        self.assertEqual(effective_cors_setting({}, "http://a:3000"), "http://a:3000")
        self.assertEqual(effective_cors_setting({"BACKEND_CORS_ORIGINS": "*"}, "http://a:3000"), "*")
        self.assertEqual(effective_cors_setting({"BACKEND_CORS_ORIGINS": ""}, "http://a:3000"), "")
        self.assertEqual(parse_cors_origins(effective_cors_setting({"BACKEND_CORS_ORIGINS": ""}, "http://a:3000")), ())
        self.assertEqual(effective_cors_setting({}, None), "")

    def test_headers_echo_origin(self):
        h = cors_headers("http://a:3000/", "content-type, x-custom")
        self.assertEqual(h["Access-Control-Allow-Origin"], "http://a:3000")
        self.assertEqual(h["Vary"], "Origin")
        self.assertEqual(h["Access-Control-Allow-Headers"], "content-type, x-custom")
        self.assertIn("X-Engine", h["Access-Control-Expose-Headers"])
        self.assertEqual(cors_headers("http://a:3000")["Access-Control-Allow-Headers"], "Content-Type")


class HealthPayloadTests(unittest.TestCase):
    STATUS = {"asrModel": "whisper-small-a16w8", "catalog": [
        {"name": "Gemma", "type": "chat", "loaded": True},
        {"name": "whisper-small-a16w8", "type": "asr", "loaded": True},
        {"name": "Llama", "type": "chat", "loaded": False}]}
    ENGINES = [{"key": "supertonic", "loaded": True}, {"key": "piper-tts", "loaded": False},
               {"key": "browser", "loaded": True}]

    def test_reachable(self):
        body = health_payload(mode="backend-only", version="1.2", status=self.STATUS, engines=self.ENGINES)
        self.assertTrue(body["ok"])
        self.assertTrue(body["model_server"]["reachable"])
        self.assertEqual(body["asr_model"], "whisper-small-a16w8")
        self.assertEqual(body["chat_models_loaded"], ["Gemma"])
        self.assertEqual([e["key"] for e in body["tts"]["engines"]], ["supertonic", "piper-tts"])

    def test_unreachable(self):
        body = health_payload(mode="studio", version="1.2", status=None, engines=[], error="refused")
        self.assertFalse(body["ok"])
        self.assertFalse(body["model_server"]["reachable"])
        self.assertEqual(body["model_server"]["error"], "refused")
        self.assertIsNone(body["asr_model"])


if __name__ == "__main__":
    unittest.main()
