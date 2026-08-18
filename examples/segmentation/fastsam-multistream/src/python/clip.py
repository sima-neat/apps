import cv2
import numpy as np
import pyneat as neat

from fastsam import lround
from tokenizer import SimpleTokenizer

CLIP_IMAGE_PX = 256
CLIP_BATCH = 16
TEXT_CONTEXT_LENGTH = 77
FEATURE_DIM = 512


def _floats(tensor):
    if tensor.dtype != neat.TensorDType.Float32:
        raise RuntimeError("expected Float32 tensor")
    return np.frombuffer(tensor.copy_dense_bytes_tight(), dtype=np.float32)


def _crop_into(dst, window_rgb, submask, px=CLIP_IMAGE_PX, bg=1.0):
    h, w = window_rgb.shape[:2]
    scale = px / min(h, w)
    nw = max(px, lround(w * scale))
    nh = max(px, lround(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    resized = cv2.resize(window_rgb, (nw, nh), interpolation=interp)
    keep = cv2.resize(submask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    y0 = (nh - px) // 2
    x0 = (nw - px) // 2
    dst[:] = resized[y0:y0 + px, x0:x0 + px] * np.float32(1.0 / 255.0)
    dst[keep[y0:y0 + px, x0:x0 + px] == 0] = bg


def _scores(image_features, text_query):
    if len(image_features) == 0:
        return np.zeros(0)
    query = np.asarray(text_query, dtype=np.float64)
    qnorm = np.linalg.norm(query)
    q = query / qnorm if qnorm > 0.0 else np.zeros_like(query)

    images = np.asarray(image_features, dtype=np.float64)
    inorms = np.linalg.norm(images, axis=1, keepdims=True)
    unit = np.divide(images, inorms, out=np.zeros_like(images), where=inorms > 0.0)

    return unit @ q


class ImageEncoder:
    def __init__(self, model_path, run_opt):
        self._stack = np.zeros((CLIP_BATCH, CLIP_IMAGE_PX, CLIP_IMAGE_PX, 3), np.float32)
        self._rows = [self._stack[i] for i in range(CLIP_BATCH)]
        self._model = neat.Model(model_path)
        self._runner = self._model.build([self._input()], neat.ModelRouteOptions(), run_opt)

    def _input(self):
        return neat.Tensor.from_numpy(self._stack, memory=neat.TensorMemory.EV74)

    def _encode(self, crops, timeout_ms):
        feats = []
        for start in range(0, len(crops), CLIP_BATCH):
            n = min(len(crops), start + CLIP_BATCH) - start
            if n < CLIP_BATCH:
                self._stack[n:] = 0.0
            for i in range(n):
                _crop_into(self._rows[i], crops[start + i].window, crops[start + i].submask)

            out = self._runner.run([self._input()], timeout_ms)
            if not out:
                raise RuntimeError("image encoder returned no output")
            vals = _floats(out[0])
            dim = vals.size // CLIP_BATCH
            rows = vals[:CLIP_BATCH * dim].reshape(CLIP_BATCH, dim)
            feats.extend(rows[i] for i in range(n))
        return feats

    def best_match(self, candidates, text_query, min_score, timeout_ms):
        if not candidates:
            return None
        scores = _scores(self._encode([crop for _, crop in candidates], timeout_ms), text_query)
        if len(scores) == 0:
            return None
        best = int(np.argmax(scores))
        if scores[best] < min_score:
            return None
        return candidates[best][0]

    def close(self):
        self._runner.close()


def _text_consts(path):
    data = np.load(str(path))
    return {k: data[k].astype(np.float32) for k in data.files}


class TextEncoder:
    def __init__(self, model_path, consts_path, run_opt):
        self._consts = _text_consts(consts_path)
        self._tokenizer = SimpleTokenizer(context_length=TEXT_CONTEXT_LENGTH)
        self._model = neat.Model(model_path)
        seed = self._trunk_input(np.zeros((TEXT_CONTEXT_LENGTH, FEATURE_DIM), np.float32))
        self._runner = self._model.build([seed], neat.ModelRouteOptions(), run_opt)

    def _trunk_input(self, emb_row):
        arr = np.ascontiguousarray(emb_row, dtype=np.float32).reshape(1, *emb_row.shape)
        return neat.Tensor.from_numpy(arr, copy=True, memory=neat.TensorMemory.EV74,
                                      layout=neat.TensorLayout.HWC)

    def encode(self, text, timeout_ms):
        tokens = self._tokenizer([text])
        te, pe = self._consts["token_embedding"], self._consts["positional_embedding"]
        emb = (te[tokens.astype(np.int64)] + pe[None]).astype(np.float32)

        seqs = []
        for i in range(emb.shape[0]):
            out = self._runner.run([self._trunk_input(emb[i])], timeout_ms)
            if not out:
                raise RuntimeError("text encoder returned no output")
            seqs.append(_floats(out[0]).reshape(TEXT_CONTEXT_LENGTH, FEATURE_DIM))

        seq = np.stack(seqs)
        idx = tokens.argmax(axis=-1)
        pooled = seq[np.arange(seq.shape[0]), idx]
        return pooled @ self._consts["text_projection"]

    def close(self):
        self._runner.close()
