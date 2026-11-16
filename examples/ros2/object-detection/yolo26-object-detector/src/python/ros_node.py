"""YOLO26: ROS Image -> the native image model route -> decoded ROS results."""
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array


class Pipeline:
    def __init__(self, config, config_path=None):
        self.label_path = config["model"].get("labels", "")
        self.visual_labels = None
        decode = config.get("decode", {})
        self.threshold = float(decode.get("score_threshold", native.MIN_SCORE))
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        options.decode_type = pyneat.BoxDecodeType.YoloV26
        options.score_threshold = self.threshold
        options.nms_iou_threshold = float(decode.get("nms_iou", native.NMS_IOU))
        options.top_k = int(decode.get("max_detections", native.MAX_DET))
        self.model = pyneat.Model(config["model"]["path"], options)

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("YOLO26 expects bgr8, matching the native folder input")
        tensor = pyneat.Tensor.from_numpy(np.ascontiguousarray(image_array(message)),
                                         copy=True, image_format=pyneat.PixelFormat.BGR,
                                         memory=pyneat.TensorMemory.EV74)
        return [tensor]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        options.preset = pyneat.RunPreset.Balanced
        return self.model.build(seed, route_options=pyneat.ModelRouteOptions(), run_options=options)

    def render(self, frame, data):
        import cv2
        from pathlib import Path
        native.cv2 = cv2
        if self.visual_labels is None:
            self.visual_labels = native.load_labels(Path(self.label_path))
        return native.draw_boxes(frame, data["objects"], self.visual_labels)

    def decode(self, sample, message):
        tensors = list(sample.tensors) if sample.tensors else [sample.tensor]
        payload = native.extract_bbox_payload_from_tensors(tensors)
        if not payload:
            raise RuntimeError("Model returned no BBOX payload")
        boxes = native.parse_bbox_payload(payload, message.width, message.height, self.threshold)
        return {"width": message.width, "height": message.height, "objects": boxes}


if __name__ == "__main__":
    raise SystemExit(run("yolo26_object_detector", Pipeline))
