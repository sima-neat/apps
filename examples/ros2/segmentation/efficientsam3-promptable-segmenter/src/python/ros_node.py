"""EfficientSAM3: ROS NV12 image + configured text prompt -> segmentation."""
import json
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import planar_array, sample_tensors


class PromptRun:
    """The image and text are one request with one runtime identity."""
    def __init__(self, runner):
        self.runner = runner

    def push(self, samples):
        request = pyneat.Sample()
        request.kind = pyneat.SampleKind.TensorSet
        request.tensors = [sample.tensor for sample in samples]
        request.frame_id = samples[0].frame_id
        request.pts_ns = samples[0].pts_ns
        request.stream_id = samples[0].stream_id
        return self.runner.push_samples(request)

    def pull(self, timeout_ms):
        return self.runner.pull(timeout_ms=timeout_ms)

    def close(self):
        self.runner.close()


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        self.config = native.load_config(config_path)
        tokens = native.prompt_tokens(self.config.prompt)
        # Text encoding happens once. Every image owns separate prepared storage.
        self.text = native.ev74_tensor(native.encode_prompt(self.config, tokens))
        self.model = pyneat.Model(self.config.model_path)

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("EfficientSAM3 expects native nv12 source pixels")
        image = np.empty((native.MODEL_SIZE, native.MODEL_SIZE, 3), np.float32)
        native.preprocess(planar_array(message), image)
        return [native.ev74_tensor(image), self.text]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = native.PIPELINE_DEPTH
        options.overflow_policy = pyneat.OverflowPolicy.Block
        blank = np.zeros((native.MODEL_SIZE, native.MODEL_SIZE, 3), np.float32)
        return PromptRun(self.model.build([native.ev74_tensor(blank), self.text], run_options=options))

    def decode(self, sample, message):
        detections, masks = [tensor.to_numpy(copy=False) for tensor in sample_tensors(sample)]
        segments = native.segments_of(detections, masks, self.config, message.width, message.height)
        return json.loads(native.segments_json(segments))


if __name__ == "__main__":
    raise SystemExit(run("efficientsam3_promptable_segmenter", Pipeline))
