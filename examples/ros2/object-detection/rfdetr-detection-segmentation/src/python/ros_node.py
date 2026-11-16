"""RF-DETR: ROS image -> backbone -> native top-K proposals -> transformer."""
import json
import threading
import pyneat
from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor


class TwoStageRun:
    """Overlap the two models; retain backbone buffers until their result returns."""
    def __init__(self, backbone, transformer, input_shapes, proposal_count, top_k):
        self.backbone, self.transformer = backbone, transformer
        self.input_shapes = input_shapes
        self.proposal_count, self.top_k = proposal_count, top_k
        self.stop = threading.Event()
        self.error = None
        self.retained = {}
        self.lock = threading.Lock()
        self.worker = threading.Thread(target=self.bridge, name="rfdetr-proposals")
        self.worker.start()

    def push(self, samples):
        if self.error:
            raise self.error
        return self.backbone.push(samples)

    def bridge(self):
        try:
            while not self.stop.is_set():
                sample = self.backbone.pull(timeout_ms=100)
                if not sample:
                    continue
                feature, scores, proposals = native.split_backbone(sample, self.proposal_count)
                gathered = native.stable_topk_gather(scores.to_numpy(copy=False),
                                                     proposals.to_numpy(copy=False), self.top_k)
                tensor = pyneat.Tensor.from_numpy(gathered, copy=True, memory=pyneat.TensorMemory.EV74)
                request = pyneat.Sample()
                request.kind = pyneat.SampleKind.TensorSet
                request.tensors = native.transformer_inputs(self.input_shapes, feature, tensor, self.top_k)
                native.copy_identity(sample, request)
                with self.lock:
                    self.retained[int(sample.frame_id)] = (sample, request)
                if not self.transformer.push_samples(request):
                    raise RuntimeError("RF-DETR transformer rejected input")
        except BaseException as error:
            if not self.stop.is_set():
                self.error = error

    def pull(self, timeout_ms):
        if self.error:
            raise self.error
        output = self.transformer.pull(timeout_ms=timeout_ms)
        if output:
            with self.lock:
                if self.retained.pop(int(output.frame_id), None) is None:
                    raise RuntimeError("RF-DETR transformer returned an unknown frame")
        return output if output else None

    def close(self):
        self.stop.set()
        self.backbone.close()
        self.transformer.close()
        self.worker.join()
        self.retained.clear()


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        native.pyneat = pyneat
        self.config = native.load_config(config_path)
        self.labels = native.load_labels(self.config.labels)
        self.geometry = None

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("RF-DETR expects native nv12 source pixels")
        geometry = (message.width, message.height)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("RF-DETR source geometry changed")
        self.geometry = geometry
        return [image_tensor(message)]

    def build(self, seed):
        self.backbone = native.build_backbone(self.config, *self.geometry)
        self.transformer = native.build_transformer(self.config)
        side = native.feature_side(self.backbone)
        native.require_model_contract(self.config, self.backbone, self.transformer, side)
        options = native.transformer_run_options()
        # Retain the device-backed feature buffer through transformer completion,
        # as the native backbone graph does. Owned output can wrap a CPU copy in
        # a GstSample, which is unsuitable for the transformer's EV74 input.
        options.output_memory = pyneat.OutputMemory.ZeroCopy
        backbone_run = self.backbone.build(seed, run_options=options)
        try:
            transformer_run = self.transformer.build(native.transformer_seed(self.transformer),
                                                     run_options=native.transformer_run_options())
        except BaseException:
            backbone_run.close()
            raise
        return TwoStageRun(backbone_run, transformer_run,
                           [tuple(spec.shape) for spec in self.transformer.input_specs()],
                           side * side, self.config.top_k)

    def decode(self, sample, message):
        _, payload = native.result_metadata(self.config, sample, message.width, message.height, self.labels)
        return json.loads(payload)


if __name__ == "__main__":
    raise SystemExit(run("rfdetr_detection_segmentation", Pipeline))
