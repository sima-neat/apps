"""Edge cases the Studio handles instead of failing on the board.

Memory full (a load that won't fit beside the loaded models), storage full (a
download bigger than the free space), context full (a conversation longer than
the model's compiled window) and a busy accelerator (Platform 3.0's "no free
bank" refusal). No hardware: ModelManager runs against the fake server from
the ASR switching suite.
"""

import errno
import shutil
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

from asr_switching_suite import FakeServer, make_model_dir
from server import hub as hub_module
from server import model_manager as mm
from server.model_manager import ModelDoesNotFit, ModelManager
from shared.chat_limits import (accelerator_busy, context_window_from_elfs, fit_to_window,
                                request_tokens, text_tokens, tokenizer_counter, too_long_alone,
                                whole_reply)
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

    def test_malformed_messages_are_left_for_the_model_server_not_a_crash(self):
        # A 500 from the trimming would hide the model server's own 4xx.
        for messages in ([{"role": "user", "content": 5}], [{"role": "user", "content": True}]):
            kept, dropped = fit_to_window(messages, 2048)
            self.assertEqual((kept, dropped), (messages, 0))
            self.assertFalse(too_long_alone(messages, 2048))
        self.assertEqual(fit_to_window("not a list", 2048), ("not a list", 0))
        self.assertFalse(too_long_alone("not a list", 2048))

    def test_the_models_own_token_count_trims_text_the_estimate_undercounts(self):
        # On the board, "lorem " filler costs more tokens than 4 characters
        # each: trimmed by the estimate, LFM2.5-230M answered with no text.
        lorem = lambda text: 3 * len(str(text).split())   # noqa: E731 - a stand-in tokenizer
        messages = []
        for i in range(14):
            messages += [{"role": "user", "content": f"note {i} " + "lorem " * 120},
                         {"role": "assistant", "content": "ok"}]
        messages.append({"role": "user", "content": "Say hi."})
        estimated, estimate_dropped = fit_to_window(messages, 2048)
        self.assertGreater(request_tokens(estimated, lorem), 2048 - 64, "the estimate leaves it too long")
        sent, dropped = fit_to_window(messages, 2048, lorem)
        self.assertGreater(dropped, estimate_dropped)
        self.assertLessEqual(request_tokens(sent, lorem), 2048 - 64)
        self.assertEqual(sent[-1]["content"], "Say hi.")
        self.assertTrue(too_long_alone([{"role": "user", "content": "lorem " * 700}], 2048, lorem))
        self.assertFalse(too_long_alone([{"role": "user", "content": "lorem " * 700}], 2048))

    def test_the_tokenizer_file_is_read_once_and_missing_ones_fall_back(self):
        tokenizer_counter.cache_clear()
        self.addCleanup(tokenizer_counter.cache_clear)
        opened = []

        class Tokenizer:
            @staticmethod
            def from_file(path):
                if "missing" in path:
                    raise OSError("no such file")
                opened.append(path)
                return types.SimpleNamespace(encode=lambda text, add_special_tokens: types.SimpleNamespace(
                    ids=text.split()))

        with patch.dict(sys.modules, {"tokenizers": types.SimpleNamespace(Tokenizer=Tokenizer)}):
            count = tokenizer_counter("/m/devkit/tokenizer.json")
            self.assertEqual(count("one two three"), 3)
            self.assertIs(tokenizer_counter("/m/devkit/tokenizer.json"), count)
            self.assertEqual(opened, ["/m/devkit/tokenizer.json"])
            self.assertIsNone(tokenizer_counter("/missing/tokenizer.json"))
        self.assertIsNone(tokenizer_counter(None))

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

    def test_a_whole_reply_refused_by_a_busy_accelerator_is_sent_once_more(self):
        class Reply:
            def __init__(self, status, body):
                self.status_code, self.content = status, body.encode()
                self.headers = {"Content-Type": "application/json"}
                self.closed = False

            def close(self):
                self.closed = True

        busy = Reply(500, '{"error": {"message": "MLA-RT queued wait failed: rc=-11"}}')
        good = Reply(200, '{"choices": [{"message": {"content": "hi"}}]}')
        sent = []
        status, ctype, body = whole_reply(busy, lambda: sent.append(1) or good, pause=0)
        self.assertEqual((status, body), (200, good.content))
        self.assertEqual(len(sent), 1, "sent once more")
        self.assertTrue(busy.closed and good.closed)
        # Busy twice: the second answer is passed on, with no third try.
        again = Reply(500, '{"error": "dispatch: no free bank"}')
        status, _, body = whole_reply(busy, lambda: again, pause=0)
        self.assertEqual((status, body), (500, again.content))
        # Any other answer goes straight back.
        other = Reply(400, '{"error": "bad request"}')
        self.assertEqual(whole_reply(other, lambda: self.fail("not retried"), pause=0)[0], 400)


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
        self.server = FakeServer(())
        self.manager = ModelManager(self.server, catalog_dir=self.tmp, max_resident_chat_models=3, asr_name=None,
                                    hub=HubConfig(allow_download=False),
                                    openai_base_url="http://127.0.0.1:9998",
                                    warmup=False, asr_warmup=False, switch_settle_s=0.0,
                                    accelerator_bytes=100)

    def test_the_catalog_reports_memory_and_window(self):
        (self.tmp / "small-a" / "devkit").mkdir(exist_ok=True)
        (self.tmp / "small-a" / "devkit" / "tokenizer.json").write_text("{}")
        entries = {e["name"]: e for e in self.manager.catalog()}
        self.assertEqual(entries["small-a"]["tokenizerPath"],
                         str(self.tmp / "small-a" / "devkit" / "tokenizer.json"))
        self.assertIsNone(entries["small-b"]["tokenizerPath"])
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

    def test_the_budget_comes_from_the_config(self):
        roomy = ModelManager(FakeServer(()), catalog_dir=self.tmp, max_resident_chat_models=3, asr_name=None,
                             hub=HubConfig(allow_download=False), openai_base_url="http://127.0.0.1:9998",
                             warmup=False, asr_warmup=False, switch_settle_s=0.0, accelerator_bytes=200)
        for name in ("small-a", "small-b", "big-c"):
            roomy.load(name)
        self.assertEqual(sorted(roomy.resident()), ["big-c", "small-a", "small-b"])

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

    def test_a_disk_that_fills_mid_download_is_recognised(self):
        try:
            try:
                raise OSError(errno.ENOSPC, "No space left on device")
            except OSError as inner:
                raise RuntimeError("download failed") from inner
        except RuntimeError as outer:
            self.assertTrue(hub_module._out_of_space(outer))
        self.assertFalse(hub_module._out_of_space(RuntimeError("HTTP 404")))


if __name__ == "__main__":
    unittest.main()
