"""CLIP tokenizer for the CLIP ViT-B/16 text encoder, after OpenAI CLIP (MIT, see src/common/LICENSE-CLIP.txt).

The vocabulary and merges come from the model package's tokenizer.json. Text is expected in composed (NFC) form.
Letters such as é, ß, Cyrillic, and CJK tokenize as CLIP does. Digits must be ASCII 0-9, and superscripts,
combining marks, İ, and a word-final capital Σ can tokenize differently.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np

# tokenizer.json's pre-tokenizer split, with Python's classes for letters and digits.
WORDS = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+")
START, END = "<|startoftext|>", "<|endoftext|>"


def bytes_to_unicode() -> dict[int, str]:
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    others = [b for b in range(256) if b not in printable]
    return {**{b: chr(b) for b in printable}, **{b: chr(256 + n) for n, b in enumerate(others)}}


class ClipTokenizer:
    def __init__(self, path: str | Path) -> None:
        model = json.loads(Path(path).read_text(encoding="utf-8"))["model"]
        self.byte_chars = bytes_to_unicode()
        self.ids = model["vocab"]
        self.ranks = {tuple(merge.split()): i for i, merge in enumerate(model["merges"])}

    def pieces(self, word: str) -> list[str]:
        parts = [*word[:-1], word[-1] + "</w>"]
        while len(parts) > 1:
            pair = min(itertools.pairwise(parts), key=lambda p: self.ranks.get(p, len(self.ranks)))
            if pair not in self.ranks:
                break
            merged, i = [], 0
            while i < len(parts):
                if tuple(parts[i:i + 2]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        return parts

    def tokens(self, text: str) -> list[int]:
        """The token ids of the text, without the start and end tokens."""
        tokens = []
        for word in WORDS.findall(" ".join(text.lower().split())):
            tokens += [self.ids[piece] for piece in self.pieces("".join(self.byte_chars[b] for b in word.encode()))]
        return tokens

    def encode(self, text: str, length: int) -> np.ndarray:
        """The start token, the text's tokens and the end token, padded to length with the end token."""
        end = self.ids[END]
        ids = [self.ids[START], *self.tokens(text)][:length - 1] + [end]
        return np.pad(np.array(ids, np.int64), (0, length - len(ids)), constant_values=end)
