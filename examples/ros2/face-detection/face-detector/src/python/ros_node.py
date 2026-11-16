"""RetinaFace ROS node preserving native BGR mean subtraction, padding and NMS."""
import cv2
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        native.cv2, native.np, native.pyneat = cv2, np, pyneat
        self.settings = config.get("decode", {})
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Tensor
        options.preprocess.input_max_width = native.INFER_WIDTH
        options.preprocess.input_max_height = native.INFER_HEIGHT
        options.preprocess.input_max_depth = 3
        self.model = pyneat.Model(config["model"]["path"], options)
        self.geometry = {}

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("RetinaFace input must be bgr8")
        pixels = image_array(message).astype(np.float32) - np.array([104, 117, 123], np.float32)
        padded, geometry = native.pad_image_bgr(pixels, message.height, message.width,
                                                native.INFER_WIDTH, native.INFER_HEIGHT)
        pixels = cv2.resize(padded, (native.INFER_WIDTH, native.INFER_HEIGHT),
                            interpolation=cv2.INTER_LINEAR)
        self.geometry[id(message)] = geometry
        return [native.tensor_from_hwc_f32(pixels)]

    def build(self, seed):
        graph = pyneat.Graph()
        graph.add(pyneat.nodes.input(self.model.input_appsrc_options(True)))
        graph.add(pyneat.nodes.quant_tess(pyneat.QuantTessOptions(self.model)))
        graph.add(pyneat.groups.mla(self.model))
        graph.add(pyneat.nodes.detess_dequant(pyneat.DetessDequantOptions(self.model)))
        graph.add(pyneat.nodes.output())
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        self.graph = graph
        dummy = native.tensor_from_hwc_f32(
            np.zeros((native.INFER_HEIGHT, native.INFER_WIDTH, 3), np.float32))
        return graph.build([dummy], options=options)

    def render(self, frame, data):
        return native.draw_detections(frame, data["faces"])

    def decode(self, sample, message):
        arrays = native.tensor_numpy_outputs(sample_tensors(sample))
        boxes, scores, landmarks = native.parse_retinaface_outputs(arrays)
        detections = native.postprocess_retinaface(
            boxes, scores, landmarks, self.geometry.pop(id(message)),
            confidence_threshold=float(self.settings.get("confidence_threshold", 0.40)),
            nms_threshold=float(self.settings.get("nms_iou", 0.90)),
            top_k=int(self.settings.get("top_k", 5000)),
            keep_top_k=int(self.settings.get("keep_top_k", 750)),
            with_landmarks=landmarks is not None)
        return {"width": message.width, "height": message.height, "faces": detections}


if __name__ == "__main__":
    raise SystemExit(run("face_detector", Pipeline))
