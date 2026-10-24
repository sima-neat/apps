"""RF-DETR object detection: a backbone in the source graph and a transformer runner.

The host picks the top proposals between the two compiled stages. A bridge thread does this
while the main thread publishes transformer results, so both stages and the host overlap.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pyneat

from hub_models import base

NUM_CLASSES = 91
TOP_K = 300  # Proposals the transformer takes; fixed by the compiled model.


class Model(base.Model):
    def __init__(self, cfg: dict, packs: dict, labels: list[str], width: int, height: int):
        opt = base.image_options(width, height, pyneat.ResizeMode.Stretch, pyneat.NormalizePreset.ImageNet)
        opt.processcvu.pre_run_target = "EV74"
        opt.processcvu.post_run_target = "A65"
        self.neat_model = pyneat.Model(packs["backbone"], opt)

        opt = pyneat.ModelOptions()
        opt.preprocess.kind = pyneat.InputKind.Tensor
        opt.preprocess.enable = pyneat.AutoFlag.Off
        opt.processcvu.pre_run_target = "A65"
        opt.processcvu.post_run_target = "A65"
        self.transformer = pyneat.Model(packs["transformer"], opt)

        # The backbone's square feature grid sets the proposal count for every model size.
        side = int(self.neat_model.output_specs()[0].shape[1])
        self.proposals = side * side
        self.labels = labels
        self.width = width
        self.height = height
        self.min_score = cfg["min_score"]
        self.max_detections = cfg["max_detections"]

    def results(self, run, stop):
        runner = self.transformer.build(
            transformer_seed(self.transformer),
            route_options=pyneat.ModelRouteOptions(),
            run_options=transformer_run_options(),
        )
        shapes = [tuple(spec.shape) for spec in self.transformer.input_specs()]
        errors: list[BaseException] = []

        def bridge() -> None:
            try:
                for sample in base.Model.results(self, run, stop):
                    if not runner.push_samples(self.transformer_sample(sample, shapes)) and not stop.is_set():
                        raise RuntimeError("RF-DETR transformer rejected input")
            except BaseException as exc:  # Reported by the main thread.
                errors.append(exc)
                stop.set()

        worker = threading.Thread(target=bridge, name="rf-detr-bridge", daemon=True)
        worker.start()
        try:
            while not stop.is_set():
                sample = runner.pull(timeout_ms=500)
                if base.tensors(sample):  # A pull that times out returns an empty sample.
                    yield sample
            if errors:
                raise errors[0]
        finally:
            stop.set()
            runner.close_input()
            worker.join()
            runner.close()

    def transformer_sample(self, sample, shapes):
        """Gather the top proposals from a backbone result into a transformer input sample."""
        feature = scores = proposals = None
        for tensor in base.tensors(sample):
            shape = tuple(tensor.shape)
            if shape[-1] == 256:
                feature = tensor
            elif shape[-1] == 4:
                proposals = tensor
            elif math.prod(shape) == self.proposals:
                scores = tensor
        gathered = pyneat.Tensor.from_numpy(
            top_proposals(scores.to_numpy(copy=False), proposals.to_numpy(copy=False)),
            copy=True,
            memory=pyneat.TensorMemory.EV74,
        )
        if feature.storage.kind in (pyneat.StorageKind.CpuOwned, pyneat.StorageKind.CpuExternal):
            feature = feature.cvu()  # Keep SiMa-backed features zero-copy.
        inputs = []
        for shape in shapes:
            tensor = gathered if math.prod(shape) == TOP_K * 4 else feature
            if tuple(tensor.shape) == (1, *shape):  # Drop the batch axis the transformer omits.
                tensor.shape = list(shape)
                tensor.strides_bytes = list(tensor.strides_bytes)[1:]
                tensor.axis_semantics = list(tensor.axis_semantics)[1:]
            inputs.append(tensor)
        result = pyneat.Sample()
        result.kind = pyneat.SampleKind.TensorSet
        result.tensors = inputs
        for name in ("frame_id", "stream_id", "stream_label", "input_seq", "orig_input_seq",
                     "pts_ns", "dts_ns", "duration_ns"):
            setattr(result, name, getattr(sample, name))
        result.attributes = sample.attributes
        return result

    def postprocess(self, sample) -> list[tuple[str, dict]]:
        boxes = logits = None
        for tensor in base.tensors(sample):
            if math.prod(tensor.shape) == TOP_K * 4:
                boxes = tensor.to_numpy(copy=False)
            else:
                logits = tensor.to_numpy(copy=False)
        objects = detections(
            boxes, logits, self.labels, self.width, self.height, self.min_score, self.max_detections
        )
        return [("object-detection", {"objects": objects})]


def stable_top(values: np.ndarray, count: int) -> np.ndarray:
    """Indices of the ``count`` largest values, best first, lower index first on ties."""
    return np.argsort(-values, kind="stable")[:count]


def top_proposals(scores: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    proposals = np.asarray(proposals, dtype=np.float32).reshape(-1, 4)
    return proposals[stable_top(scores, TOP_K)].reshape(1, TOP_K, 4)


def detections(boxes, logits, labels, width, height, min_score, max_detections) -> list[dict]:
    """Decode the best query/class pairs; boxes are normalized centre x, y, width, height."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(TOP_K, 4)
    scores = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=np.float32), -80.0, 80.0)))
    scores = scores.reshape(-1)
    objects = []
    for flat in stable_top(scores, TOP_K):
        score = float(scores[flat])
        if score < min_score or len(objects) >= max_detections:
            break
        query, class_id = divmod(int(flat), NUM_CLASSES)
        if labels[class_id] == "unused":
            continue
        cx, cy, w, h = boxes[query]
        x0 = float(np.clip((cx - w / 2) * width, 0, width))
        y0 = float(np.clip((cy - h / 2) * height, 0, height))
        x1 = float(np.clip((cx + w / 2) * width, 0, width))
        y1 = float(np.clip((cy + h / 2) * height, 0, height))
        objects.append({"label": labels[class_id], "confidence": score, "bbox": [x0, y0, x1 - x0, y1 - y0]})
    return objects


def transformer_seed(transformer) -> list:
    return [
        pyneat.Tensor.from_numpy(
            np.zeros(tuple(spec.shape), dtype=np.float32), copy=True, memory=pyneat.TensorMemory.EV74
        )
        for spec in transformer.input_specs()
    ]


def transformer_run_options():
    """Every backbone result reaches the transformer; the bridge waits when it is full."""
    opt = pyneat.RunOptions()
    opt.preset = pyneat.RunPreset.Balanced
    opt.queue_depth = 4
    opt.overflow_policy = pyneat.OverflowPolicy.Block
    opt.output_memory = pyneat.OutputMemory.Owned
    return opt
