"""PCB inspection on ROS images, using the native options and box decoder."""
import cv2
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    def __init__(self, config, config_path=None):
        native.cv2, native.np, native.pyneat = cv2, np, pyneat
        self.config = native.build_app_config(config)
        self.labels = native.load_labels(self.config.labels_path)
        self.model = pyneat.Model(self.config.model_path,
                                 native.model_options(self.config, self.labels))

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("PCB input must be bgr8")
        return [native.to_bgr_tensor(image_array(message))]

    def build(self, seed):
        options = native.run_options(self.config)
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def render(self, frame, data):
        native.draw_boxes(frame, data["objects"], self.labels)
        return frame

    def decode(self, sample, message):
        detections = native.decode_detections(sample_tensors(sample), message.width,
                                              message.height, self.config)
        return {"width": message.width, "height": message.height, "objects": detections}


if __name__ == "__main__":
    raise SystemExit(run("pcb_defect_detector", Pipeline))
