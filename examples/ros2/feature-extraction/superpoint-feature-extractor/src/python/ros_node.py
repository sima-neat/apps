"""SuperPoint ROS images -> native keypoint geometry in source-image pixels."""
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    def __init__(self, config, config_path=None):
        self.config = native.load_config(config_path)
        self.model = None

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("SuperPoint expects bgr8; set ros.source_encoding: bgr8")
        if self.model is None:
            self.model = pyneat.Model(str(self.config.model),
                                     native.model_options(pyneat, message.width, message.height))
            self.model_size = native.model_input_size(self.model)
        return [native.input_tensor(image_array(message), np, pyneat)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def render(self, frame, data):
        import cv2
        native.draw_points(frame, data["points"], cv2)
        return frame

    def decode(self, sample, message):
        source_size = native.FrameSize(message.width, message.height)
        points = native.feature_points(sample_tensors(sample), self.model_size,
                                        source_size, np, pyneat)
        return {"width": message.width, "height": message.height, "points": points}


if __name__ == "__main__":
    raise SystemExit(run("superpoint_feature_extractor", Pipeline))
