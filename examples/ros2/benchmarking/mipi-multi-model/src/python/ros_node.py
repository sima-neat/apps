"""Run the selected native MIPI model profile on ROS NV12 images."""
import json
import numpy as np
import pyneat
from neat_apps.runtime import load_native
native = load_native(__file__)
from model_profiles import inspect_package
from neat_apps.image_node import run
from neat_apps.images import image_tensor, sample_tensors


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        self.config = native.load_config(config_path)
        self.profile = self.config.profile
        package = inspect_package(self.config.model_path, self.profile)
        options = native.model_options(pyneat, self.profile, package)
        if self.profile.preprocessing == "tensorflow_ssd":
            # Neat 0.6.1 needs an explicit SSD decoded-output capacity.
            options.top_k = 100
        self.model = pyneat.Model(str(package.path), options)
        self.labels = native.load_labels()

    def prepare(self, message):
        if (message.width, message.height, message.encoding) != (native.CAMERA_WIDTH, native.CAMERA_HEIGHT, "nv12"):
            raise ValueError("MIPI profiles expect 1920x1080 nv12 images, as in the native example")
        return [image_tensor(message)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def render(self, frame, data):
        from neat_apps.visual_output import render_result
        if self.profile.task == "pose":
            # The native MIPI profile reports pose counts, without defining an
            # image-coordinate overlay contract for these raw decoded tensors.
            import cv2
            count = sum(np.asarray(item["boxes"]).reshape(-1, 6).shape[0] for item in data["poses"])
            cv2.putText(frame, f"poses={count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0, 255, 0), 2)
            return frame
        return render_result(frame, data)

    def decode(self, sample, message):
        if self.profile.task == "detection":
            return {"objects": native.insight_objects(pyneat, self.profile, sample, self.labels)}
        if self.profile.task == "segmentation":
            payload, _ = native.encode_segments(native.insight_segments(pyneat, sample, self.labels))
            return json.loads(payload)
        if self.profile.task == "pose":
            poses = pyneat.decode_pose(sample_tensors(sample))
            return {"poses": [{"boxes": item.boxes.to_numpy(copy=True),
                               "keypoints": item.keypoints.to_numpy(copy=True)} for item in poses]}
        values = sample_tensors(sample)[0].to_numpy(copy=True)
        if self.profile.task == "classification":
            flat = values.reshape(-1)
            index = int(np.argmax(flat))
            return {"class_id": index, "value": float(flat[index]), "scores": flat}
        depth = np.squeeze(values).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape: {depth.shape}")
        return {"shape": list(depth.shape), "images": [("depth", depth, "32FC1")]}


if __name__ == "__main__":
    raise SystemExit(run("mipi_multi_model", Pipeline))
