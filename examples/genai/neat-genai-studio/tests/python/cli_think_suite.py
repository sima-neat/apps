"""CLI reasoning support: the <think> stream splitter and the /no_think rewrite.

Loads cli/main.py as a module (it only runs under __main__), no server needed.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

CLI_MAIN = Path(__file__).resolve().parents[2] / "src" / "python" / "cli" / "main.py"
_spec = importlib.util.spec_from_file_location("studio_cli_main", CLI_MAIN)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def _split(deltas):
    splitter = cli._ThinkSplitter()
    pieces = []
    for delta in deltas:
        pieces += splitter.feed(delta)
    pieces += splitter.flush()
    merged = []
    for kind, text in pieces:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


class ThinkSplitterTests(unittest.TestCase):
    def test_reasoning_block_is_separated_from_the_answer(self):
        self.assertEqual(
            _split(["<think>", "let me ", "reason", "</think>", "The answer."]),
            [("think", "let me reason"), ("answer", "The answer.")])

    def test_tags_split_across_deltas_are_resolved(self):
        self.assertEqual(_split(["<thi", "nk>abc</th", "ink>xyz"]),
                         [("think", "abc"), ("answer", "xyz")])

    def test_a_lone_angle_bracket_is_not_swallowed(self):
        self.assertEqual(_split(["plain ", "answer <", "3 done"]),
                         [("answer", "plain answer <3 done")])

    def test_unterminated_reasoning_is_flushed_as_reasoning(self):
        self.assertEqual(_split(["<think>unterminated"]), [("think", "unterminated")])

    def test_multiple_blocks(self):
        self.assertEqual(
            _split(["a<think>b</think>c<think>d</think>e"]),
            [("answer", "a"), ("think", "b"), ("answer", "c"), ("think", "d"), ("answer", "e")])


class NoThinkRewriteTests(unittest.TestCase):
    def test_last_user_text_turn_gets_the_switch_without_mutating_input(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "again"}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[-1]["content"], "again /no_think")
        self.assertEqual(out[1]["content"], "hi")           # earlier turns untouched
        self.assertEqual(msgs[-1]["content"], "again")      # input not mutated

    def test_multimodal_turn_appends_to_its_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "see"},
                                             {"type": "image", "image": "x"}]}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[0]["content"][0]["text"], "see /no_think")
        self.assertEqual(msgs[0]["content"][0]["text"], "see")

    def test_image_only_turn_gains_a_text_part(self):
        msgs = [{"role": "user", "content": [{"type": "image", "image": "x"}]}]
        out = cli._without_thinking(msgs)
        self.assertEqual(out[0]["content"][-1], {"type": "text", "text": "/no_think"})


if __name__ == "__main__":
    unittest.main()
