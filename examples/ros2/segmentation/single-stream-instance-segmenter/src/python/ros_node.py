"""Instance segmentation from ROS image topics, using the native model and decoder."""
import json
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor, sample_tensors


class Pipeline:
    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        self.model = None
        self.encoding = None
        self.labels = native.load_labels(self.config.labels_path)

    def prepare(self, message):
        if message.encoding not in ("nv12", "nv16"):
            raise ValueError("This pipeline expects the native decoder's nv12/nv16 pixels")
        if self.model is None:
            self.encoding = message.encoding
            self.model = native.make_model(self.config, message.width, message.height, message.encoding == "nv16")
        if message.encoding != self.encoding:
            raise ValueError("Input encoding changed after model construction")
        return [image_tensor(message)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.preset = pyneat.RunPreset.Realtime
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        options.output_memory = pyneat.OutputMemory.ZeroCopy
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        detections = native.decode_segments(self.config, sample_tensors(sample), message.width, message.height)
        segments = native.metadata_segments(detections, self.labels, (message.height, message.width),
                                            self.config.output.mask_threshold)
        # Preserve the native polygon precision and output budget.
        data, _ = native.encode_segments(segments)
        return json.loads(data)


if __name__ == "__main__":
    raise SystemExit(run("single_stream_instance_segmenter", Pipeline))
