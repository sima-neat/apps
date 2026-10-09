"""CLIP tokenizer for the EfficientSAM3 text encoder, after OpenAI CLIP (MIT, see src/common/LICENSE-CLIP.txt)."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

VOCAB = Path(__file__).resolve().parents[1] / "common" / "bpe_simple_vocab_16e6.txt.gz"
WORDS = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+")


def bytes_to_unicode() -> dict[int, str]:
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    others = [b for b in range(256) if b not in printable]
    return {**{b: chr(b) for b in printable}, **{b: chr(256 + n) for n, b in enumerate(others)}}


class ClipTokenizer:
    def __init__(self) -> None:
        self.byte_chars = bytes_to_unicode()
        lines = gzip.decompress(VOCAB.read_bytes()).decode("utf-8").split("\n")
        merges = [tuple(line.split()) for line in lines[1:49152 - 256 - 2 + 1]]
        pieces = list(self.byte_chars.values())
        pieces += [piece + "</w>" for piece in pieces] + ["".join(merge) for merge in merges]
        pieces += ["<start_of_text>", "<end_of_text>"]
        self.ids = {piece: i for i, piece in enumerate(pieces)}
        self.ranks = {merge: i for i, merge in enumerate(merges)}

    def pieces(self, word: str) -> list[str]:
        parts = [*word[:-1], word[-1] + "</w>"]
        while len(parts) > 1:
            pair = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, len(self.ranks)))
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

    def encode(self, text: str, length: int) -> np.ndarray:
        ids = [self.ids["<start_of_text>"]]
        for word in WORDS.findall(" ".join(text.lower().split())):
            ids += [self.ids[piece] for piece in self.pieces("".join(self.byte_chars[b] for b in word.encode()))]
        ids = ids[:length - 1] + [self.ids["<end_of_text>"]]
        return np.pad(np.array(ids, np.int64), (0, length - len(ids)))
