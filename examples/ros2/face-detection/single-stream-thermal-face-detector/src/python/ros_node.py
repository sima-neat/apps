"""Thermal-face split heads on ROS NV12 images, with the native landmark decoder."""
import json
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor


class Pipeline:
    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        self.labels = native.load_labels(self.config.labels_path)
        self.model = native.make_model(self.config)

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("Thermal-face input must be nv12")
        return [image_tensor(message)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        cfg = self.config
        boxes, scores, landmarks = native.decode_yolov5face_split(
            native.sample_to_outputs(sample), cfg.min_score, cfg.nms_iou, cfg.max_detections)
        scale, left, top = native.letterbox_params(message.width, message.height, 800, 800)
        _, landmarks = native.unletterbox(boxes, landmarks, scale, left, top,
                                          message.width, message.height)
        return json.loads(native.build_poses_json(landmarks, scores, self.labels[0]))


if __name__ == "__main__":
    raise SystemExit(run("single_stream_thermal_face_detector", Pipeline))
