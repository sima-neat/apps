import io
import shutil
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

# ModelManager itself does not parse YAML; keep this host-only unit test
# runnable even when the example's runtime dependencies are not installed.
try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    sys.modules["yaml"] = types.ModuleType("yaml")

from server.model_manager import ModelManager
from shared.config import HubConfig


class FakeServer:
    """Stands in for pyneat.GenAIServer: add_model registers, remove_model frees."""

    def __init__(self, names=()):
        self.names = list(names)
        self.added = []
        self.removed = []

    def model_names(self):
        return list(self.names)

    def add_model(self, path, name):
        self.added.append((str(path), name))
        self.names.append(name)
        return name

    def remove_model(self, name):
        if name in self.names:
            self.names.remove(name)
            self.removed.append(name)
            return True
        return False


def make_model_dir(root: Path, name: str, kind: str) -> Path:
    """Create a directory classify_model_dir() will type as `kind`."""
    model = root / name
    (model / "devkit").mkdir(parents=True)
    (model / "elf_files").mkdir()
    (model / "elf_files" / "stage0_mla.elf").write_bytes(b"x")
    (model / ".neat-complete").write_text("ok\n")
    config = "whisper_config.json" if kind == "asr" else "vlm_config.json"
    (model / "devkit" / config).write_text("{}")
    return model


class AsrSwitchingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        make_model_dir(self.tmp, "whisper-small-a16w8", "asr")
        make_model_dir(self.tmp, "whisper-medium-a16w8", "asr")
        make_model_dir(self.tmp, "Llama-3.2-3B-Instruct", "chat")
        # No warmup and no settle: both reach for the network or the clock.
        patcher = patch.object(ModelManager, "_stop_model_streams", lambda *a: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def manager(self, loaded=("whisper-small-a16w8",), asr="whisper-small-a16w8"):
        server = FakeServer(loaded)
        manager = ModelManager(
            server,
            catalog_dir=self.tmp,
            max_resident_chat_models=1,
            asr_name=asr,
            hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998",
            warmup=False,
            asr_warmup=False,
            switch_settle_s=0.0,
        )
        return manager, server

    def entry(self, manager, name):
        return next(e for e in manager.catalog() if e["name"] == name)

    def test_switching_asr_evicts_the_previous_asr_and_keeps_the_chat_model(self):
        manager, server = self.manager()
        manager.load("Llama-3.2-3B-Instruct")
        manager.set_active_asr("whisper-medium-a16w8")

        self.assertEqual(server.removed, ["whisper-small-a16w8"])
        self.assertEqual(manager.active_asr(), "whisper-medium-a16w8")
        self.assertIn("Llama-3.2-3B-Instruct", server.model_names())
        self.assertNotIn("whisper-small-a16w8", server.model_names())

    def test_loading_a_chat_model_never_evicts_the_active_asr(self):
        manager, server = self.manager()
        manager.set_active_asr("whisper-medium-a16w8")
        server.removed.clear()
        manager.load("Llama-3.2-3B-Instruct")

        # The switched-to ASR has a different name than the configured one; it
        # must still be excluded from the chat residency set.
        self.assertNotIn("whisper-medium-a16w8", server.removed)
        self.assertIn("whisper-medium-a16w8", server.model_names())
        self.assertEqual(manager.active_asr(), "whisper-medium-a16w8")

    def test_set_active_asr_rejects_a_chat_model(self):
        manager, _ = self.manager()
        with self.assertRaisesRegex(ValueError, "not a speech-to-text"):
            manager.set_active_asr("Llama-3.2-3B-Instruct")
        self.assertEqual(manager.active_asr(), "whisper-small-a16w8")

    def test_set_active_asr_rejects_an_unknown_model(self):
        manager, _ = self.manager()
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            manager.set_active_asr("whisper-enormous")

    def test_catalog_marks_active_and_configured_asr_independently(self):
        manager, _ = self.manager()
        manager.set_active_asr("whisper-medium-a16w8")

        small = self.entry(manager, "whisper-small-a16w8")
        medium = self.entry(manager, "whisper-medium-a16w8")
        self.assertTrue(small["pinned"])          # config re-selects it on restart
        self.assertFalse(small["activeAsr"])
        self.assertFalse(medium["pinned"])
        self.assertTrue(medium["activeAsr"])      # but this one transcribes now

    def test_active_asr_cannot_be_unloaded_or_deleted(self):
        manager, _ = self.manager()
        with self.assertRaisesRegex(ValueError, "cannot be unloaded"):
            manager.unload("whisper-small-a16w8")
        with self.assertRaisesRegex(ValueError, "cannot be deleted"):
            manager.delete("whisper-small-a16w8")

    def test_an_inactive_asr_can_be_deleted(self):
        manager, _ = self.manager()
        manager.set_active_asr("whisper-medium-a16w8")
        manager.delete("whisper-small-a16w8")

        self.assertFalse((self.tmp / "whisper-small-a16w8").exists())
        self.assertNotIn("whisper-small-a16w8", [e["name"] for e in manager.catalog()])

    def test_a_configured_asr_the_server_never_loaded_is_not_active(self):
        # main.py skips a configured ASR whose directory is missing — what you
        # get after switching away from the startup default and deleting it.
        manager, _ = self.manager(loaded=(), asr="whisper-small-a16w8")

        self.assertIsNone(manager.active_asr())
        status = manager.status()
        self.assertIsNone(status["asrModel"])
        self.assertEqual(status["configuredAsrModel"], "whisper-small-a16w8")
        self.assertFalse(self.entry(manager, "whisper-small-a16w8")["activeAsr"])

    def test_a_failed_eviction_aborts_the_switch(self):
        manager, server = self.manager()

        def refuse(name):
            raise RuntimeError("MLA busy")

        with patch.object(server, "remove_model", side_effect=refuse):
            with self.assertRaisesRegex(RuntimeError, "Could not unload"):
                manager.set_active_asr("whisper-medium-a16w8")

        # The old model is still serving, so it must still be reported active,
        # and the replacement must not have been added alongside it.
        self.assertEqual(manager.active_asr(), "whisper-small-a16w8")
        self.assertIn("whisper-small-a16w8", server.model_names())
        self.assertNotIn("whisper-medium-a16w8", server.model_names())

    def test_status_reports_active_and_configured_asr_separately(self):
        manager, _ = self.manager()
        manager.set_active_asr("whisper-medium-a16w8")

        status = manager.status()
        self.assertEqual(status["asrModel"], "whisper-medium-a16w8")
        self.assertEqual(status["configuredAsrModel"], "whisper-small-a16w8")


class AsrWarmupBehaviourTests(AsrSwitchingTests):
    """Warming forces the deferred MLA load; a non-MLA failure must not undo it."""

    def manager(self, loaded=("whisper-small-a16w8",), asr="whisper-small-a16w8"):
        server = FakeServer(loaded)
        manager = ModelManager(
            server,
            catalog_dir=self.tmp,
            max_resident_chat_models=1,
            asr_name=asr,
            hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998",
            warmup=True,
            asr_warmup=True,
            switch_settle_s=0.0,
        )
        return manager, server

    def setUp(self):
        super().setUp()
        # Every inherited case now runs with warming on, against a stub probe.
        self.warm = patch.object(ModelManager, "_warm_check_asr",
                                 return_value=(True, ""))
        self.warm.start()
        self.addCleanup(self.warm.stop)
        chat = patch.object(ModelManager, "_warm_check", return_value=(True, ""))
        chat.start()
        self.addCleanup(chat.stop)

    def test_switching_warms_the_new_model(self):
        manager, _ = self.manager()
        with patch.object(ModelManager, "_warm_check_asr",
                          return_value=(True, "")) as warm:
            manager.set_active_asr("whisper-medium-a16w8")
        warm.assert_called_once_with("whisper-medium-a16w8")

    def test_a_non_mla_warm_failure_leaves_the_model_active(self):
        manager, _ = self.manager()
        with patch.object(ModelManager, "_warm_check_asr",
                          return_value=(False, "HTTP 400: bad audio")):
            result = manager.set_active_asr("whisper-medium-a16w8")

        self.assertEqual(result["state"], "ready")
        self.assertIn("bad audio", result["warm_warning"])
        self.assertEqual(manager.active_asr(), "whisper-medium-a16w8")

    def test_an_mla_warm_failure_rolls_the_switch_back(self):
        manager, server = self.manager()
        with patch.object(ModelManager, "_warm_check_asr",
                          return_value=(False, "MLA_LOAD_FAILED: bulk load")):
            with self.assertRaisesRegex(RuntimeError, "accelerator"):
                manager.set_active_asr("whisper-medium-a16w8")

        self.assertIsNone(manager.active_asr())
        self.assertNotIn("whisper-medium-a16w8", server.model_names())


class AsrWarmupPayloadTests(unittest.TestCase):
    """The warm-up probe is built with the stdlib only (no requests in pyneat)."""

    def test_silence_payload_is_a_16k_mono_16bit_wav(self):
        with wave.open(io.BytesIO(ModelManager._silence_wav()), "rb") as clip:
            self.assertEqual(clip.getnchannels(), 1)
            self.assertEqual(clip.getsampwidth(), 2)
            self.assertEqual(clip.getframerate(), 16000)
            self.assertEqual(clip.getnframes(), 16000)

    def test_multipart_body_carries_the_model_name_and_clip(self):
        body, content_type = ModelManager._multipart_body(
            {"model": "whisper-medium-a16w8", "language": "en"},
            "file", "warmup.wav", b"RIFFdata", "audio/wav",
        )
        boundary = content_type.split("boundary=", 1)[1]
        self.assertTrue(content_type.startswith("multipart/form-data; "))
        self.assertIn(b'name="model"', body)
        self.assertIn(b"whisper-medium-a16w8", body)
        self.assertIn(b'filename="warmup.wav"', body)
        self.assertIn(b"RIFFdata", body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))


if __name__ == "__main__":
    unittest.main()
