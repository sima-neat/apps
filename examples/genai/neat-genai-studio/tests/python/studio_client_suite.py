"""Shared request-side helpers used by both front ends (shared/studio_client.py).

Pure stdlib, no server or hardware. Collected through test_unit.py.
"""

from __future__ import annotations

import http.client
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from shared import studio_client as sc


class NoThinkTransformTests(unittest.TestCase):
    def test_appends_to_last_user_text_turn_without_mutating_input(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "again"}]
        out = sc.apply_no_think(msgs)
        self.assertEqual(out[-1]["content"], "again /no_think")
        self.assertEqual(out[1]["content"], "hi")        # earlier turn untouched
        self.assertEqual(msgs[-1]["content"], "again")   # input not mutated

    def test_multipart_turn_appends_to_its_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "see"},
                                             {"type": "image", "image": "x"}]}]
        out = sc.apply_no_think(msgs)
        self.assertEqual(out[0]["content"][0]["text"], "see /no_think")
        self.assertEqual(msgs[0]["content"][0]["text"], "see")

    def test_image_only_turn_gains_a_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "image", "image": "x"}]}]
        out = sc.apply_no_think(msgs)
        self.assertEqual(out[0]["content"][-1], {"type": "text", "text": "/no_think"})

    def test_empty_history_is_returned_unchanged(self):
        self.assertEqual(sc.apply_no_think([]), [])


class MlaResetPolicyTests(unittest.TestCase):
    def test_disabled_only_when_env_is_not_one(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLA_RESET", None)
            self.assertFalse(sc.mla_reset_disabled())
            os.environ["MLA_RESET"] = "1"
            self.assertFalse(sc.mla_reset_disabled())
            os.environ["MLA_RESET"] = "0"
            self.assertTrue(sc.mla_reset_disabled())

    def test_hand_off_reports_no_supervisor_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEAT_RESET_REQUEST_FILE", None)
            self.assertEqual(sc.hand_reset_to_supervisor(), "no_supervisor")

    def test_hand_off_writes_the_sentinel(self):
        with TemporaryDirectory() as d:
            req = Path(d) / "reset.request"
            with mock.patch.dict(os.environ, {"NEAT_RESET_REQUEST_FILE": str(req)}):
                self.assertEqual(sc.hand_reset_to_supervisor(), "handed")
            self.assertEqual(req.read_text(encoding="utf-8"), "reset\n")

    def test_hand_off_reports_error_when_unwritable(self):
        with mock.patch.dict(os.environ,
                             {"NEAT_RESET_REQUEST_FILE": "/no-such-dir-xyz/reset.request"}):
            self.assertEqual(sc.hand_reset_to_supervisor(), "error")

    def test_only_a_mid_reply_disconnect_counts_as_accepted(self):
        self.assertTrue(sc.is_reset_disconnect(http.client.RemoteDisconnected("gone")))
        self.assertTrue(sc.is_reset_disconnect(ConnectionResetError()))
        self.assertFalse(sc.is_reset_disconnect(ConnectionRefusedError()))
        self.assertFalse(sc.is_reset_disconnect(OSError("network unreachable")))


if __name__ == "__main__":
    unittest.main()
