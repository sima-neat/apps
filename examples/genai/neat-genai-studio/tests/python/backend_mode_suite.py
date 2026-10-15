"""Backend-only mode policy (backend_mode.py): served paths, CORS allowlist,
headers, the /health payload and the routes relayed to the control API. Pure
Python: no Flask."""

from __future__ import annotations

import json
import unittest
import urllib.parse
import urllib.request

from backend_mode import (
    CONTROL_ROUTES,
    control_payload,
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
                     "/voices/select", "/supertonic/voice", "/piperplus/voices", "/shutdown",
                     "/rag/status", "/rag/inspect", "/rag/search", "/rag/upload", "/rag/reset", "/rag/clear",
                     "/board-camera/devices", "/board-camera/snapshot"):
            self.assertTrue(path_allowed(path), path)

    def test_removed_web_ui_routes_are_not(self):
        for path in ("/favicon.ico", "/playground/", "/playground/index.html", "/solutions/", "/showcase",
                     "/config.js", "/static/newui.js", "/upload", "/stop", "/clear-history",
                     "/upload-to-rag", "/reset-rag", "/clear-rag", "/raghealth", "/board-camerax", "/voicesx",
                     "/v1", "/modelsx"):
            self.assertFalse(path_allowed(path), path)

    def test_shutdown_is_never_cross_origin(self):
        self.assertFalse(cors_path("/shutdown"))
        self.assertFalse(cors_path("/"))
        self.assertTrue(cors_path("/v1/audio/speech"))
        self.assertTrue(cors_path("/health"))
        self.assertTrue(cors_path("/rag/upload"))


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

    def test_reports_the_api_version(self):
        body = health_payload(mode="backend-only", version="1.2", status=self.STATUS, engines=self.ENGINES)
        self.assertEqual(body["api_version"], 1)

    def test_lists_optional_features(self):
        on = health_payload(mode="backend-only", version="1.2", status=self.STATUS, engines=self.ENGINES, rag_enabled=True)
        self.assertEqual(on["features"], {"rag": True, "benchmark": True, "board_camera": True})
        off = health_payload(mode="backend-only", version="1.2", status=self.STATUS, engines=self.ENGINES)
        self.assertFalse(off["features"]["rag"])
        self.assertEqual(off["api_version"], 1, "adding features is not a breaking change")

    def test_lists_an_engine_that_failed_to_load_with_its_error(self):
        engines = [{"key": "piper-tts", "loaded": True}, {"key": "browser", "loaded": True}]
        body = health_payload(mode="backend-only", version="1.2", status=self.STATUS, engines=engines,
                              engine_failures={"supertonic": "accelerator-unavailable"})
        self.assertIn({"key": "supertonic", "loaded": False, "error": "accelerator-unavailable"},
                      body["tts"]["engines"])
        self.assertTrue(body["ok"])  # the model server is fine; one voice engine is not

    def test_an_engine_that_loaded_is_not_also_listed_as_failed(self):
        body = health_payload(mode="studio", version="1.2", status=self.STATUS, engines=self.ENGINES,
                              engine_failures={"supertonic": "stale"})
        self.assertEqual([e["key"] for e in body["tts"]["engines"]], ["supertonic", "piper-tts"])

    def test_unreachable(self):
        body = health_payload(mode="studio", version="1.2", status=None, engines=[], error="refused")
        self.assertFalse(body["ok"])
        self.assertFalse(body["model_server"]["reachable"])
        self.assertEqual(body["model_server"]["error"], "refused")
        self.assertIsNone(body["asr_model"])


if __name__ == "__main__":
    unittest.main()


class _RecordingManager:
    """The control API's manager, recording what each request asked for."""

    max_resident = 1   # reported with the catalog

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return {"ok": name}
        return call


class ControlRelayTests(unittest.TestCase):
    # A request to each relayed route, and the control API call it must reach.
    CASES = {
        ("GET", "/models/status"): ({}, ("status", (), {})),
        ("GET", "/models/catalog"): ({}, ("scan_catalog", (), {})),
        ("GET", "/models/logs"): ({"after": "7"}, ("load_logs", (7,), {})),
        ("GET", "/models/card"): ({"name": "m"}, ("model_card", ("m",), {})),
        ("POST", "/models/load"): ({"name": "m", "unload": ["x"]}, ("load", ("m", ["x"]), {})),
        ("POST", "/models/unload"): ({"name": "m"}, ("unload", ("m",), {})),
        ("POST", "/models/delete"): ({"name": "m"}, ("delete", ("m",), {})),
        ("POST", "/benchmark/run"): (
            {"num_samples": 2, "prompt": "p"},
            ("benchmark_start", (), {"num_samples": 2, "max_new_tokens": 128, "prompt": "p", "model": ""})),
        ("GET", "/benchmark/status"): ({}, ("benchmark_status", (), {})),
        ("POST", "/benchmark/stop"): (None, ("benchmark_stop", (), {})),
    }

    def test_every_relayed_route_is_served_and_open_to_cors(self):
        self.assertEqual(set(CONTROL_ROUTES), set(self.CASES))
        for method, route in CONTROL_ROUTES:
            self.assertIn(method, ("GET", "POST"))
            self.assertTrue(path_allowed(route), route)
            self.assertTrue(cors_path(route), route)

    def test_get_relays_the_query_and_post_the_json_object(self):
        self.assertEqual(control_payload("GET", [("after", "3"), ("x", "y")], None),
                         ([("after", "3"), ("x", "y")], None))
        self.assertEqual(control_payload("POST", [("q", "1")], {"name": "m"}), (None, {"name": "m"}))
        self.assertEqual(control_payload("POST", [], None), (None, {}))
        for body in (["m"], "m", 3):
            with self.assertRaises(ValueError):
                control_payload("POST", [], body)

    def test_each_route_reaches_its_control_api_handler(self):
        from server.control_api import serve_control_api

        manager = _RecordingManager()
        server = serve_control_api(manager, "127.0.0.1", 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        for (method, route), (target, _timeout) in CONTROL_ROUTES.items():
            with self.subTest(route=route):
                sent, expected = self.CASES[(method, route)]
                query = list(sent.items()) if method == "GET" else []
                params, body = control_payload(method, query, None if method == "GET" else sent)
                url = base + target + ("?" + urllib.parse.urlencode(params) if params else "")
                data = json.dumps(body).encode() if body is not None else None
                req = urllib.request.Request(url, data=data, method=method,
                                             headers={"Content-Type": "application/json"})
                before = len(manager.calls)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    self.assertEqual(resp.status, 200)
                # The handler may read more than one thing (the catalog also
                # reports the resident models); the expected call is among them.
                self.assertIn(expected, manager.calls[before:])
