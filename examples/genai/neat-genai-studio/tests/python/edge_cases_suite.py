"""Edge cases the Studio handles instead of failing on the board.

Memory full (a load that won't fit beside the loaded models), storage full (a
download bigger than the free space), context full (a conversation longer than
the model's compiled window) and a busy accelerator (Platform 3.0's "no free
bank" refusal). No hardware: ModelManager runs against the fake server from
the ASR switching suite.
"""

import shutil
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server import hub as hub_module
from server import model_manager as mm
from server.model_manager import ModelDoesNotFit, ModelManager
from shared.chat_limits import (accelerator_busy, context_window_from_elfs, fit_to_window,
                                request_tokens, text_tokens, too_long_alone)
from shared.config import HubConfig

Usage = namedtuple("Usage", "total used free")


def sized_model(root: Path, name: str, size: int, window: int | None = None) -> Path:
    """A chat model whose ELF stages take `size` bytes, compiled for `window`."""
    model = make_model_dir(root, name, "chat")
    (model / "elf_files" / "stage0_mla.elf").write_bytes(b"x" * size)
    if window:
        (model / "elf_files" / f"m_language_n1_cache_token{window - 1}_stage1_mla.elf").write_bytes(b"")
    return model


class ContextWindowTests(unittest.TestCase):
    def test_the_window_comes_from_the_compiled_stage_names(self):
        self.assertEqual(context_window_from_elfs(
            ["x_language_n128_cache_token0_stage1_mla.elf", "x_language_n1_cache_token2047_stage1_mla.elf"]), 2048)
        self.assertIsNone(context_window_from_elfs(["stage0_mla.elf"]))

    def test_estimates_run_high_for_english_and_count_cjk_per_character(self):
        # The board's LFM2.5-230M (2048 tokens) answered 202 of these sentences.
        fits = " ".join(["The quick brown fox jumps over the lazy dog."] * 202)
        self.assertGreaterEqual(text_tokens(fits), 2048)
        self.assertEqual(text_tokens("你好世界"), 4)

    def test_the_oldest_turns_go_first_and_the_question_always_stays(self):
        system = {"role": "system", "content": "Be brief."}
        turn = "a" * 400   # about 100 tokens
        messages = [system]
        for i in range(10):
            messages += [{"role": "user", "content": f"{i} {turn}"}, {"role": "assistant", "content": turn}]
        messages.append({"role": "user", "content": "Last question?"})
        sent, dropped = fit_to_window(messages, 1024)
        self.assertGreater(dropped, 0)
        self.assertEqual(sent[0], system)
        self.assertEqual(sent[-1]["content"], "Last question?")
        self.assertEqual(sent[1]["role"], "user", "no answer is left without its question")
        self.assertLessEqual(request_tokens(sent), 1024)
        self.assertEqual(fit_to_window(messages, None), (messages, 0), "unknown window: unchanged")

    def test_a_message_too_long_on_its_own_is_reported(self):
        long = [{"role": "user", "content": "word " * 3000}]
        sent, dropped = fit_to_window(long, 2048)
        self.assertEqual(dropped, 0)
        self.assertTrue(too_long_alone(sent, 2048))
        self.assertFalse(too_long_alone([{"role": "user", "content": "Hi"}], 2048))

    def test_pictures_count_toward_the_window(self):
        with_image = [{"role": "user", "content": [{"type": "text", "text": "What is it?"},
                                                   {"type": "image", "image": "data:..."}]}]
        self.assertGreater(request_tokens(with_image), 300)


class BusyAcceleratorTests(unittest.TestCase):
    def test_only_the_driver_refusal_counts_as_busy(self):
        self.assertTrue(accelerator_busy(
            "MLA-RT queued wait failed for /x_mla.elf: rc=-11 (Resource temporarily unavailable)"))
        self.assertTrue(accelerator_busy("dispatch: no free bank for model"))
        self.assertFalse(accelerator_busy("HTTP 404"))
        self.assertFalse(accelerator_busy(None))


class MemoryFullTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        sized_model(self.tmp, "small-a", 30, window=2048)
        sized_model(self.tmp, "small-b", 30)
        sized_model(self.tmp, "big-c", 60, window=8192)
        for target, value in ((ModelManager, "_stop_model_streams"),):
            p = patch.object(target, value, lambda *a: None)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(mm, "ACCELERATOR_BYTES", 100)
        p.start()
        self.addCleanup(p.stop)
        self.server = FakeServer(())
        self.manager = ModelManager(self.server, catalog_dir=self.tmp, max_resident_chat_models=3, asr_name=None,
                                    hub=HubConfig(allow_download=False),
                                    openai_base_url="http://127.0.0.1:9998",
                                    warmup=False, asr_warmup=False, switch_settle_s=0.0)

    def test_the_catalog_reports_memory_and_window(self):
        entries = {e["name"]: e for e in self.manager.catalog()}
        self.assertEqual(entries["small-a"]["acceleratorBytes"], 30)
        self.assertEqual(entries["small-a"]["contextTokens"], 2048)
        self.assertEqual(entries["big-c"]["contextTokens"], 8192)
        self.assertIsNone(entries["small-b"]["contextTokens"])

    def test_a_load_that_wont_fit_is_refused_before_it_starts(self):
        self.manager.load("small-a")
        self.manager.load("small-b")
        with self.assertRaises(ModelDoesNotFit) as caught:
            self.manager.load("big-c")
        self.assertEqual(sorted(caught.exception.resident), ["small-a", "small-b"])
        self.assertEqual(caught.exception.need, 60)
        self.assertNotIn("big-c", self.server.model_names(), "nothing was sent to the accelerator")
        self.assertEqual(sorted(self.manager.resident()), ["small-a", "small-b"], "the loaded models stay")

    def test_unloading_one_makes_room(self):
        self.manager.load("small-a")
        self.manager.load("small-b")
        result = self.manager.load("big-c", unload=["small-a"])
        self.assertEqual(result["evicted"], ["small-a"])
        self.assertIn("big-c", self.server.model_names())


class StorageFullTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_download_bigger_than_the_free_space_is_refused(self):
        with patch.object(hub_module.shutil, "disk_usage", lambda p: Usage(10e9, 6e9, 4e9)):
            why = hub_module.disk_room(self.tmp / "model", int(3.5e9))
        self.assertIn("needs 3.5 GB and 4.0 GB is free", why)

    def test_room_to_spare_lets_it_through_and_partial_files_count(self):
        target = self.tmp / "model"
        target.mkdir()
        (target / "part.bin").write_bytes(b"x" * 1000)
        with patch.object(hub_module.shutil, "disk_usage", lambda p: Usage(10e9, 5e9, 1e9 + 500)):
            self.assertIsNone(hub_module.disk_room(target, 1000), "already downloaded")
        with patch.object(hub_module.shutil, "disk_usage", lambda p: Usage(10e9, 1e9, 9e9)):
            self.assertIsNone(hub_module.disk_room(target, int(2e9)))
        self.assertIsNone(hub_module.disk_room(target, None), "unknown size")


if __name__ == "__main__":
    unittest.main()
