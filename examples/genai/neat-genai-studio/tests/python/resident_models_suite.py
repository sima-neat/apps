"""Several chat/VLM models resident at once (max_resident_chat_models > 1).

Drives ModelManager against the fake GenAIServer from the ASR switching suite:
loads keep other chat models up to the limit, evict the least recently used
past it, and a failed load rolls back only the model that failed.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server.model_manager import ModelManager
from shared.config import HubConfig

CHAT_MODELS = ("Llama-3.2-3B-Instruct", "Qwen2.5-1.5B-Instruct", "gemma-3-4b-it")
ASR_MODEL = "whisper-small-a16w8"


class ResidentChatModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name in CHAT_MODELS:
            make_model_dir(self.tmp, name, "chat")
        make_model_dir(self.tmp, ASR_MODEL, "asr")
        patcher = patch.object(ModelManager, "_stop_model_streams", lambda *a: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def manager(self, limit, warmup=False):
        server = FakeServer((ASR_MODEL,))
        manager = ModelManager(
            server,
            catalog_dir=self.tmp,
            max_resident_chat_models=limit,
            asr_name=ASR_MODEL,
            hub=HubConfig(allow_download=False),
            openai_base_url="http://127.0.0.1:9998",
            warmup=warmup,
            asr_warmup=False,
            switch_settle_s=0.0,
        )
        return manager, server

    def test_a_limit_of_one_still_replaces_the_loaded_model(self):
        llama, qwen, _ = CHAT_MODELS
        manager, server = self.manager(limit=1)
        manager.load(llama)
        result = manager.load(qwen)

        self.assertEqual(result["evicted"], [llama])
        self.assertEqual(manager.resident(), [qwen])
        self.assertNotIn(llama, server.model_names())

    def test_models_up_to_the_limit_stay_loaded_side_by_side(self):
        llama, qwen, _ = CHAT_MODELS
        manager, server = self.manager(limit=2)
        manager.load(llama)
        result = manager.load(qwen)

        self.assertEqual(result["evicted"], [])
        self.assertEqual(manager.resident(), [qwen, llama])
        self.assertEqual(server.removed, [])
        self.assertIn(llama, server.model_names())
        self.assertIn(qwen, server.model_names())

    def test_past_the_limit_the_least_recently_used_model_is_evicted(self):
        llama, qwen, gemma = CHAT_MODELS
        manager, server = self.manager(limit=2)
        manager.load(llama)
        manager.load(qwen)
        # Re-selecting a resident model registers nothing and makes it the
        # most recently used, so the next load evicts the other one.
        again = manager.load(llama)
        self.assertEqual(again["load_seconds"], 0.0)
        self.assertEqual(len(server.added), 2)
        self.assertEqual(manager.resident(), [llama, qwen])

        result = manager.load(gemma)

        self.assertEqual(result["evicted"], [qwen])
        self.assertEqual(manager.resident(), [gemma, llama])
        self.assertIn(llama, server.model_names())
        self.assertNotIn(qwen, server.model_names())

    def test_the_asr_model_is_never_counted_or_evicted(self):
        llama, qwen, _ = CHAT_MODELS
        manager, server = self.manager(limit=2)
        manager.load(llama)
        manager.load(qwen)

        self.assertNotIn(ASR_MODEL, manager.resident())
        self.assertIn(ASR_MODEL, server.model_names())
        self.assertEqual(manager.active_asr(), ASR_MODEL)

    def test_a_model_that_does_not_fit_leaves_the_loaded_ones_alone(self):
        llama, qwen, _ = CHAT_MODELS
        manager, server = self.manager(limit=3, warmup=True)
        with patch.object(ModelManager, "_warm_check", lambda self, name: (True, "")):
            manager.load(llama)
        with patch.object(ModelManager, "_warm_check",
                          lambda self, name: (False, "Cannot allocate memory")):
            with self.assertRaisesRegex(RuntimeError, f"beside the other loaded models \\({llama}\\)"):
                manager.load(qwen)

        self.assertEqual(manager.resident(), [llama])
        self.assertIn(llama, server.model_names())
        self.assertNotIn(qwen, server.model_names())

    def test_a_failed_warm_up_rolls_back_only_the_new_model(self):
        llama, qwen, _ = CHAT_MODELS
        manager, server = self.manager(limit=2, warmup=True)
        with patch.object(ModelManager, "_warm_check", lambda self, name: (True, "")):
            manager.load(llama)
        with patch.object(ModelManager, "_warm_check",
                          lambda self, name: (False, "HTTP 400: bad chat template")):
            with self.assertRaisesRegex(RuntimeError, f"Model '{qwen}' failed to load"):
                manager.load(qwen)

        self.assertEqual(manager.resident(), [llama])
        self.assertIn(llama, server.model_names())
        self.assertNotIn(qwen, server.model_names())

    def test_status_reports_the_resident_order_and_limit(self):
        llama, qwen, _ = CHAT_MODELS
        manager, _ = self.manager(limit=2)
        manager.load(llama)
        manager.load(qwen)

        status = manager.status()
        self.assertEqual(status["resident"], [qwen, llama])
        self.assertEqual(status["maxResident"], 2)


if __name__ == "__main__":
    unittest.main()
