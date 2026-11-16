"""Oriented-box detection from ROS image topics, using the native model and decoder."""
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
        if message.encoding != "nv12":
            raise ValueError("This pipeline expects the native decoder's nv12 pixels")
        if self.model is None:
            self.encoding = message.encoding
            self.model = native.build_model(self.config)
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
        detections = native.decode_detections(sample)
        return {"objects": native.insight_objects(detections, self.labels)}


if __name__ == "__main__":
    raise SystemExit(run("multi_stream_obb_detector", Pipeline))
