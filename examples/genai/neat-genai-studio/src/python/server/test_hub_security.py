import tempfile
import sys
import types
import unittest
from pathlib import Path

# The security helpers do not parse YAML; keep this host-only unit test runnable
# even when the example's runtime dependencies have not been installed.
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    sys.modules["yaml"] = types.ModuleType("yaml")

from server.hub import _catalog_target, safe_name, validated_repo_id
from server.model_manager import parse_param_count
from shared.config import HubConfig


class HubPathSecurityTests(unittest.TestCase):
    def setUp(self):
        self.hub = HubConfig(allow_download=True, orgs=("simaai", "TDoSiMa"))

    def test_accepts_existing_model_id_shapes(self):
        values = (
            "simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4",
            "TDoSiMa/gemma-4-E4B-it_GPTQ_INT4-emb-int8-8k",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(validated_repo_id(value, self.hub), value)
                self.assertEqual(safe_name(value), value.split("/", 1)[1])

    def test_rejects_traversal_and_unconfigured_organizations(self):
        values = (
            "simaai/..",
            "simaai/../outside",
            "simaai/model/extra",
            "/absolute",
            "simaai\\outside",
            "simaai/model%2Foutside",
            "untrusted/model",
        )
        for value in values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validated_repo_id(value, self.hub)

    def test_rejects_existing_symlink_that_escapes_catalog(self):
        with tempfile.TemporaryDirectory() as catalog, tempfile.TemporaryDirectory() as outside:
            Path(catalog, "model").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                _catalog_target(Path(catalog), "simaai/model")

    def test_parameter_count_parser_is_linear_and_preserves_catalog_formats(self):
        self.assertEqual(parse_param_count("Qwen3-4B-Instruct"), "4B")
        self.assertEqual(parse_param_count("LFM2-350M-a16w4"), "350M")
        self.assertEqual(parse_param_count("model-1.5B-int4"), "1.5B")
        self.assertIsNone(parse_param_count("model-000000000000000000000000000000000x"))


if __name__ == "__main__":
    unittest.main()
