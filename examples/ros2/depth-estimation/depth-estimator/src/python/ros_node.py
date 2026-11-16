"""Depth Anything V2 image-topic node with the native RGB resize and depth layout."""
import cv2
import numpy as np
import pyneat

from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        runtime = config.get("runtime", {})
        self.size = int(runtime.get("infer_size", 518))
        self.queue_depth = int(runtime.get("queue_depth", 4))
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.RGB
        options.preprocess.input_max_width = self.size
        options.preprocess.input_max_height = self.size
        options.preprocess.input_max_depth = 3
        self.model = pyneat.Model(config["model"]["path"], options)

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("Depth input must be bgr8, matching the native example")
        rgb = cv2.cvtColor(image_array(message), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return [pyneat.Tensor.from_numpy(np.ascontiguousarray(rgb), copy=True,
                                        image_format=pyneat.PixelFormat.RGB)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = self.queue_depth
        options.overflow_policy = pyneat.OverflowPolicy.Block
        dummy = pyneat.Tensor.from_numpy(np.zeros((self.size, self.size, 3), np.uint8),
                                        copy=True, image_format=pyneat.PixelFormat.RGB)
        return self.model.build([dummy], route_options=pyneat.ModelRouteOptions(), run_options=options)

    def render(self, frame, data):
        # Match the native saved subplot at the model's depth-map resolution.
        depth = data["images"][0][1]
        values = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colormap = cv2.applyColorMap(values, cv2.COLORMAP_INFERNO)
        resized = cv2.resize(frame, (colormap.shape[1], colormap.shape[0]))
        return np.concatenate((resized, colormap), axis=1)

    def decode(self, sample, message):
        tensor = sample_tensors(sample)[0]
        if not tensor.is_dense():
            raise ValueError("Depth output tensor is not dense")
        spatial = [int(d) for d in tensor.shape if d > 1]
        if len(spatial) < 2:
            raise ValueError(f"Unexpected depth shape: {tensor.shape}")
        height, width = spatial[:2]
        values = np.asarray(tensor.to_numpy(copy=True)).reshape(-1)
        # Exactly the native idx=x*height+y mapping, expressed as a transpose.
        depth = np.ascontiguousarray(values[:height * width].reshape(width, height).T,
                                     dtype=np.float32)
        return {"width": width, "height": height, "units": "relative_depth",
                "images": [("depth", depth, "32FC1")]}


if __name__ == "__main__":
    raise SystemExit(run("depth_estimator", Pipeline))
