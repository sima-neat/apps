"""DETR image-topic inference with the native tensor graph and coordinate decoder."""
from pathlib import Path

import cv2
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    warmup_runs = 0  # Native folder mode does not discard a first inference.

    def __init__(self, config, config_path=None):
        native.cv2, native.np, native.pyneat = cv2, np, pyneat
        self.settings = config.get("decode", {})
        self.model = native.load_tensor_model(Path(config["model"]["path"]))
        self.geometry = {}

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("DETR input must be bgr8")
        pixels, geometry = native.preprocess_image_bgr(image_array(message))
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
        # The native explicit graph is initialized with a zero tensor, not a
        # real image. Preserve that initialization as well as the model options.
        dummy = native.tensor_from_hwc_f32(
            np.zeros((native.FRAME_HEIGHT, native.FRAME_WIDTH, 3), np.float32))
        return graph.build([dummy], options=options)

    def render(self, frame, data):
        return native.visualize_detections(frame, data["objects"],
                                           max_draw=int(self.settings.get("max_draw", 50)))

    def decode(self, sample, message):
        arrays = native.tensor_numpy_outputs(sample_tensors(sample))
        logits, boxes = native.extract_logits_and_boxes(arrays)
        detections = native.process_detr_output(
            boxes, logits, self.geometry.pop(id(message)),
            confidence_threshold=float(self.settings.get("confidence_threshold", 0.70)),
            detect_only_person=bool(self.settings.get("person_only", False)))
        return {"width": message.width, "height": message.height, "objects": detections}


if __name__ == "__main__":
    raise SystemExit(run("detr_object_detector", Pipeline))
