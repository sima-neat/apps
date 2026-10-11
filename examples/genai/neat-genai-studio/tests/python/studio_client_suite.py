"""Shared request-side helpers used by both front ends (shared/studio_client.py).

Pure stdlib, no server or hardware. Collected through test_unit.py.
"""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
