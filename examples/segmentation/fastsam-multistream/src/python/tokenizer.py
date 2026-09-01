"""Vendored minimal CLIP BPE tokenizer (no torch / no open_clip).

Faithful port of open_clip's SimpleTokenizer for MobileCLIP-S2 (context_length=77, sot=49406,
eot=49407). Validated to match open_clip.get_tokenizer("MobileCLIP-S2") exactly:
    "the black dog"    -> [49406, 518, 1449, 1929, 49407]
    "a photo of a cat" -> [49406, 320, 1125, 539, 320, 2368, 49407]
"""
import gzip
import html
import os
from functools import lru_cache

import numpy as np
import regex as re

DEFAULT_CONTEXT_LENGTH = 77
DEFAULT_BPE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "bpe_simple_vocab_16e6.txt.gz")

try:
    import ftfy

    def _fix_text(text):
        return ftfy.fix_text(text)
except ImportError:  # ftfy is optional; for plain ASCII prompts it's a no-op
    def _fix_text(text):
        return text


@lru_cache()
def bytes_to_unicode():
    """utf-8 byte <-> unicode lookup so the reversible BPE works on str (CLIP-standard)."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


def basic_clean(text):
    text = _fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    return " ".join(text.split()).strip()


class SimpleTokenizer:
    def __init__(self, bpe_path=DEFAULT_BPE_PATH, context_length=DEFAULT_CONTEXT_LENGTH):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot_token_id = self.encoder[special_tokens[0]]
        self.eot_token_id = self.encoder[special_tokens[1]]
        self.context_length = context_length

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bt] for bt in self.bpe(token).split(" "))
        return bpe_tokens

    def __call__(self, texts, context_length=None):
        """texts: str or list[str] -> int32 numpy [M, context_length]."""
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length
        all_tokens = [[self.sot_token_id] + self.encode(t) + [self.eot_token_id] for t in texts]
        result = np.zeros((len(all_tokens), context_length), dtype=np.int32)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = tokens
        return result
