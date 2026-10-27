"""ResNet-50 ImageNet classification."""

from __future__ import annotations

import numpy as np
import pyneat

from hub_models import base


class Model(base.Model):
    def __init__(self, cfg: dict, packs: dict, labels: list[str], width: int, height: int):
        opt = base.image_options(width, height, pyneat.ResizeMode.Stretch, pyneat.NormalizePreset.ImageNet)
        self.neat_model = pyneat.Model(packs["model"], opt)
        self.labels = labels
        self.top_k = cfg["top_k"]

    def postprocess(self, sample) -> list[tuple[str, dict]]:
        logits = base.tensors(sample)[0].to_numpy(copy=False)
        return [("classification", {"top_classes": top_classes(logits, self.labels, self.top_k)})]


def top_classes(logits: np.ndarray, labels: list[str], top_k: int) -> list[dict]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    scores = np.exp(logits - logits.max())
    scores /= scores.sum()
    ranked = np.argsort(-scores, kind="stable")[:top_k]
    return [{"label": labels[i], "confidence": float(scores[i])} for i in ranked]
