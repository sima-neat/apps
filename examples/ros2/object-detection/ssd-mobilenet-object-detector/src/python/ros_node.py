"""SSD image-topic inference with the native normalization and device BoxDecode."""
import cv2
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    def __init__(self, config, config_path=None):
        self.label_path = config["model"].get("labels", "")
        self.visual_labels = None
        native.cv2, native.np, native.pyneat = cv2, np, pyneat
        model, decode = config["model"], config.get("decode", {})
        mean, stddev = native.normalization_for_profile(
            model.get("preprocessing_profile", native.DEFAULT_PREPROCESSING_PROFILE))
        self.maximum = int(decode.get("max_detections", 100))
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.resize.enable = pyneat.AutoFlag.On
        options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
        options.preprocess.normalize.enable = pyneat.AutoFlag.On
        options.preprocess.normalize.mean = mean
        options.preprocess.normalize.stddev = stddev
        options.preprocess.normalize.has_explicit_stats = True
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        options.preprocess.color_convert.output_format = pyneat.PreprocessColorFormat.RGB
        options.decode_type = pyneat.BoxDecodeType.Ssd
        options.num_classes = native.NUM_CLASSES
        options.score_threshold = float(decode.get("score_threshold", 0.55))
        options.nms_iou_threshold = float(decode.get("nms_iou", 0.60))
        options.top_k = self.maximum
        self.model = pyneat.Model(model["path"], options)

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("SSD input must be bgr8")
        return [native.image_to_tensor(image_array(message))]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, route_options=pyneat.ModelRouteOptions(), run_options=options)

    def render(self, frame, data):
        if self.visual_labels is None:
            self.visual_labels = native.load_labels(native._resolve_asset(self.label_path, "coco_labels.txt"))
        return native.visualize_detections(frame, data["objects"], self.visual_labels)

    def decode(self, sample, message):
        boxes = native.decode_detections(sample_tensors(sample), message.width,
                                          message.height, self.maximum)
        return {"width": message.width, "height": message.height, "objects": boxes}


if __name__ == "__main__":
    raise SystemExit(run("ssd_mobilenet_object_detector", Pipeline))
