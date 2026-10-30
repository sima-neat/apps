import tempfile
import json
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

# shared.config needs PyYAML; require it rather than stubbing the module
# process-wide (which would break later tests that parse YAML for real).
try:
    import yaml  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - environment check
    raise ModuleNotFoundError(
        "PyYAML is required to run the Studio unit tests (pip install PyYAML)"
    ) from exc

from server.hub import (_catalog_target, classify_hub_repo, hub_download_stream,
                        safe_name, validated_repo_id)
from server.model_manager import parse_param_count
from shared.config import HubConfig


class HubPathSecurityTests(unittest.TestCase):
    def setUp(self):
        self.hub = HubConfig(allow_download=True,
                             orgs=("simaai", "TDoSiMa", "florianvoss"))

    def test_accepts_existing_model_id_shapes(self):
        values = (
            "simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4",
            "TDoSiMa/gemma-4-E4B-it_GPTQ_INT4-emb-int8-8k",
            "simaai/whisper-large-v3-turbo-a16w8",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(validated_repo_id(value, self.hub), value)
        self.assertEqual(safe_name(values[0]), values[0].split("/", 1)[1])
        self.assertEqual(safe_name(values[1]), values[1].replace("/", "@", 1))

    def test_shipped_defaults_cover_every_searched_org(self):
        """The configured default must match the orgs the app advertises.

        An org missing here is silently unreachable: hub_search skips it and
        validated_repo_id refuses its repos, with no error naming the cause.
        """
        from shared.config import _load_hub_config
        self.assertEqual(_load_hub_config({}).orgs,
                         ("simaai", "TDoSiMa", "florianvoss"))

    def test_official_whisper_repos_are_typed_as_asr(self):
        for name in ("whisper-small-a16w8",
                     "whisper-medium-a16w8",
                     "whisper-large-v3-turbo-a16w8"):
            repo = f"simaai/{name}"
            with self.subTest(repo=repo):
                self.assertEqual(
                    classify_hub_repo(repo, "automatic-speech-recognition"), "asr")
                self.assertEqual(safe_name(repo), name)

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

    def test_download_path_uses_canonical_hub_metadata(self):
        captured = {}

        class FakeApi:
            def __init__(self, token=None):
                self.token = token

            def model_info(self, repo_id, files_metadata=False):
                self.requested_repo_id = repo_id
                return types.SimpleNamespace(id="simaai/model", siblings=[])

        def fake_download(**kwargs):
            captured.update(kwargs)
            Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)

        fake_hub = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_download)
        with tempfile.TemporaryDirectory() as catalog, patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), patch("server.hub.hub_enabled", return_value=True), patch(
            "server.hub.classify_model_dir", return_value={"name": "model"}
        ), patch("server.hub.repair_chat_template_files", return_value=[]):
            events = list(hub_download_stream(Path(catalog), self.hub, "simaai/model"))

        # Resolved: on macOS the temporary folder is /var/..., a link to /private/var/...
        self.assertEqual(Path(captured["local_dir"]).resolve(), Path(catalog, "model").resolve())
        self.assertEqual(captured["repo_id"], "simaai/model")
        self.assertIn('"state": "done"', events[-1])


    def download_events(self, model_info):
        class FakeApi:
            def __init__(self, token=None):
                pass

            def model_info(self, repo_id, files_metadata=False):
                return model_info(repo_id)

        def fake_download(**kwargs):
            raise AssertionError("must not download")

        fake_hub = types.SimpleNamespace(HfApi=FakeApi, snapshot_download=fake_download)
        with tempfile.TemporaryDirectory() as catalog, patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), patch("server.hub.hub_enabled", return_value=True):
            return [json.loads(e) for e in hub_download_stream(Path(catalog), self.hub, "simaai/model")]

    def test_a_renamed_repository_is_named_not_downloaded(self):
        events = self.download_events(lambda _: types.SimpleNamespace(id="simaai/model-renamed", siblings=[]))
        self.assertEqual(events[-1]["state"], "error")
        self.assertEqual(events[-1]["renamedTo"], "simaai/model-renamed")
        self.assertIn("now publishes this model as simaai/model-renamed", events[-1]["message"])

    def test_a_move_to_a_disallowed_account_is_said_without_the_id(self):
        events = self.download_events(lambda _: types.SimpleNamespace(id="someone-else/model", siblings=[]))
        self.assertNotIn("renamedTo", events[-1])
        self.assertIn("does not download from", events[-1]["message"])

    def test_each_kind_of_lookup_failure_reads_differently(self):
        def raising(name, base=Exception):
            error = type(name, (base,), {})

            def model_info(_):
                raise error("secret detail https://x?token=abc")
            return model_info

        messages = {name: self.download_events(raising(name))[-1]["message"]
                    for name in ("GatedRepoError", "RepositoryNotFoundError", "ConnectionError", "Whatever")}
        self.assertIn("needs approval", messages["GatedRepoError"])
        self.assertIn("no model with this id", messages["RepositoryNotFoundError"])
        self.assertIn("Could not reach Hugging Face", messages["ConnectionError"])
        self.assertIn("Could not resolve model repository", messages["Whatever"])
        self.assertTrue(all("secret detail" not in m for m in messages.values()), "error text never reaches the client")
        self.assertEqual(len(set(messages.values())), 4)

if __name__ == "__main__":
    unittest.main()
