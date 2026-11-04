"""Edge cases the Studio handles instead of failing on the board.

Memory full (a load that won't fit beside the loaded models), storage full (a
download bigger than the free space), Core's context errors and a busy accelerator
(Platform 3.0's "no free bank" refusal). No hardware: ModelManager runs against the fake server from
the ASR switching suite.
"""

import errno
import json
import os
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
from server.model_manager import ModelLoadFailed, ModelManager
from shared.chat_limits import accelerator_busy, context_window_from_elfs, first_chat_event, whole_reply
from shared.config import HubConfig, model_dir_complete
from shared import recovery

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

    def test_streamed_context_errors_and_completions_pass_through(self):
        error = {"error": {"code": "context_length_exceeded", "message": "Prompt uses 4096 tokens"}}
        for chunk in (error,
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                      {"choices": [{"delta": {}, "finish_reason": "length"}]},
                      {"choices": [{"delta": {"content": "hello"}}]}):
            with self.subTest(chunk=chunk):
                lines = [b"data: " + json.dumps(chunk).encode(), b"", b"data: [DONE]", b""]
                reply = types.SimpleNamespace(iter_lines=lambda **kwargs: iter(lines))
                held, remaining, message = first_chat_event(reply)
                self.assertEqual(held + list(remaining), lines)
                self.assertEqual(message, (chunk.get("error") or {}).get("message"))

class BusyAcceleratorTests(unittest.TestCase):
    def test_only_the_driver_refusal_counts_as_busy(self):
        self.assertTrue(accelerator_busy(
            "MLA-RT queued wait failed for /x_mla.elf: rc=-11 (Resource temporarily unavailable)"))
        self.assertTrue(accelerator_busy("dispatch: no free bank for model"))
        self.assertFalse(accelerator_busy("HTTP 404"))
        self.assertFalse(accelerator_busy("MLA-RT queued wait failed: rc=-5 (Input/output error)"))
        self.assertFalse(accelerator_busy("MLA-RT queued wait failed: rc=-110 (Connection timed out)"))
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

    def test_context_errors_and_completions_keep_the_core_status_and_body(self):
        for status, body in (
            (400, b'{"error":{"code":"context_length_exceeded","message":"Prompt too long"}}'),
            (200, b'{"choices":[{"message":{"content":""},"finish_reason":"stop"}]}'),
            (200, b'{"choices":[{"message":{"content":"hello"},"finish_reason":"length"}]}'),
        ):
            with self.subTest(status=status, body=body):
                reply = types.SimpleNamespace(
                    status_code=status, headers={"Content-Type": "application/json"},
                    close=lambda: None, content=body)
                self.assertEqual(whole_reply(reply, lambda: self.fail("not retried")),
                                 (status, "application/json", body))


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
                                    warmup=False, asr_warmup=False, switch_settle_s=0.0)

    def test_the_catalog_reports_elf_bytes_and_window(self):
        (self.tmp / "small-a" / "devkit").mkdir(exist_ok=True)
        (self.tmp / "small-a" / "devkit" / "tokenizer.json").write_text("{}")
        self.manager.scan_catalog()
        entries = {e["name"]: e for e in self.manager.catalog()}
        self.assertEqual(entries["small-a"]["tokenizerPath"],
                         str(self.tmp / "small-a" / "devkit" / "tokenizer.json"))
        self.assertIsNone(entries["small-b"]["tokenizerPath"])
        self.assertEqual(entries["small-a"]["acceleratorBytes"], 30)
        self.assertEqual(entries["small-a"]["contextTokens"], 2048)
        self.assertEqual(entries["big-c"]["contextTokens"], 8192)
        self.assertIsNone(entries["small-b"]["contextTokens"])

    def test_paired_packages_use_target_metadata_and_elf_bytes(self):
        for method in ("dflash", "gemma4_mtp"):
            with self.subTest(method=method):
                package = self.tmp / method
                for name, draft, size, window in (("a-draft", True, 20, 8192),
                                                  ("z-target", False, 40, 4096)):
                    component = make_model_dir(package, name, "chat")
                    (component / "elf_files" / "stage0_mla.elf").rename(
                        component / "elf_files" / f"language_n1_cache_token{window-1}_stage1_mla.elf")
                    next((component / "elf_files").glob("*.elf")).write_bytes(b"x" * size)
                    config = {"lm_cfg": {"speculative_decoding_cfg": {
                        "method": method, "is_draft": draft}}}
                    if method == "gemma4_mtp" and not draft:
                        config.update(vision_model_name="vision", vm_cfg={"image_size": [480, 480]})
                    (component / "devkit" / "vlm_config.json").write_text(json.dumps(config))
                tokenizer = package / "z-target/devkit/tokenizer.json"
                tokenizer.write_text("{}")
                self.manager.scan_catalog()
                entry = next(e for e in self.manager.catalog() if e["name"] == method)
                self.assertEqual(entry["acceleratorBytes"], 60)
                self.assertEqual(entry["stagesTotal"], 2)
                self.assertEqual(entry["contextTokens"], 4096)
                self.assertEqual(entry["tokenizerPath"], str(tokenizer))
                self.assertEqual(entry["supportsVision"], method == "gemma4_mtp")
                self.assertTrue(entry["complete"])
                self.manager.load("small-a")
                self.manager.load("small-b")
                self.manager.load(method, unload=["small-a", "small-b"])
                self.assertEqual(self.server.added[-1], (str(package), method))
                self.manager.unload(method)
                next((package / "a-draft/elf_files").glob("*.elf")).write_bytes(b"")
                self.assertFalse(model_dir_complete(package)[0])

    def test_elf_sizes_do_not_reject_a_load(self):
        for name in ("small-a", "small-b", "big-c"):
            self.manager.load(name)
        self.assertEqual(sorted(self.manager.resident()), ["big-c", "small-a", "small-b"])

    def test_native_allocation_failure_is_reported_as_out_of_memory(self):
        for detail in ("Cannot allocate memory", "Out of memory", "std::bad_alloc"):
            with self.subTest(detail=detail), \
                 patch.object(self.server, "add_model", side_effect=RuntimeError(detail)):
                with self.assertRaises(ModelLoadFailed) as caught:
                    self.manager.load("big-c")
                self.assertTrue(caught.exception.out_of_memory)
                self.assertIn("did not fit", str(caught.exception))
                self.assertIsNone(self.manager.status()["loading"])

    def test_native_stderr_identifies_memory_failure_without_reusing_old_errors(self):
        def fail_load(*args):
            os.write(2, b"ERROR: SECTION_ALLOC failed: Cannot allocate memory\n")
            raise RuntimeError("MLA-RT bulk load failed for model: stage_mla.elf")

        with patch.object(self.server, "add_model", side_effect=fail_load):
            with self.assertRaises(ModelLoadFailed) as first:
                self.manager.load("big-c")
        self.assertTrue(first.exception.out_of_memory)
        with patch.object(self.server, "add_model", side_effect=RuntimeError("MLA-RT bulk load failed")):
            with self.assertRaises(ModelLoadFailed) as second:
                self.manager.load("big-c")
        self.assertFalse(second.exception.out_of_memory)

    def test_stdout_log_reader_drains_allocation_errors_before_classifying(self):
        import subprocess
        from server.load_log import __file__ as log_source

        code = """import os
from server.load_log import LoadLogTap
tap = LoadLogTap()
assert tap.install()
try:
    baseline = tap.memory_errors
    os.write(1, b'ERROR: SECTION_ALLOC failed: Cannot allocate memory\\n')
    assert tap.memory_failed_since(baseline)
    baseline = tap.memory_errors
    os.write(1, b'MLA-RT bulk load failed\\n')
    assert not tap.memory_failed_since(baseline)
finally:
    tap.uninstall()
"""
        env = dict(os.environ, PYTHONPATH=str(Path(log_source).parents[1]))
        out = subprocess.run([sys.executable, '-c', code], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('Cannot allocate memory', out.stdout)

    def test_interrupted_load_reports_connection_loss_and_requests_restart(self):
        import signal
        import threading

        restarted = threading.Event()
        with patch.dict(os.environ, NEAT_RUN_PID='12345'), \
             patch.object(recovery.os, 'kill', side_effect=lambda *_: restarted.set()) as kill:
            failure = recovery.interrupted_load('large-model')
            self.assertEqual(failure['code'], 'model_server_lost')
            self.assertTrue(failure['restarting'])
            self.assertIn('large-model', failure['error'])
            self.assertFalse(restarted.is_set())
            self.assertTrue(restarted.wait(3))
            kill.assert_called_once_with(12345, signal.SIGUSR1)

    def test_connection_loss_without_launcher_requires_manual_restart(self):
        with patch.dict(os.environ, NEAT_RUN_PID=''):
            failure = recovery.interrupted_load('model')
            self.assertEqual(failure['code'], 'model_server_lost')
            self.assertFalse(failure['restarting'])
            self.assertIn('may have run out of memory', failure['error'])
            self.assertIn('Restart GenAI Studio', failure['error'])


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
