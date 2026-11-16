"""FastFlow ROS node publishing native regions and a float anomaly-map Image."""
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
        self.config = native.load_config(config_path)
        self.model = None

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("FastFlow expects bgr8; set ros.source_encoding: bgr8")
        if self.model is None:
            self.model, self.map_side = native.build_model(self.config, message.width, message.height)
        return [pyneat.Tensor.from_numpy(np.ascontiguousarray(image_array(message)),
                                        copy=True, image_format=pyneat.PixelFormat.BGR)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def render(self, frame, data):
        import time
        now = time.monotonic()
        if not hasattr(self, "visual_started"):
            self.visual_started, self.visual_count = now, 0
        self.visual_count += 1
        elapsed = now - self.visual_started
        fps = (self.visual_count - 1) / elapsed if elapsed else 0.0
        images = {topic: values for topic, values, _ in data["images"]}
        native.render(frame, images["anomaly_map"], data["regions"],
                      images["anomaly_mask"], self.config, fps)
        return frame

    def decode(self, sample, message):
        raw = sample_tensors(sample)[0].to_numpy(copy=True)
        anomaly = np.asarray(raw, np.float32).reshape(self.map_side, self.map_side)
        regions, mask = native.regions_from_map(anomaly, self.config.threshold,
                                                self.config.min_region_px)
        return {"regions": regions, "threshold": self.config.threshold,
                "map_width": self.map_side, "map_height": self.map_side,
                "images": [("anomaly_map", anomaly, "32FC1"), ("anomaly_mask", mask, "mono8")]}


if __name__ == "__main__":
    raise SystemExit(run("fastflow_anomaly_detector", Pipeline))
