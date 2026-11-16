"""Classify each ROS image with every configured native model profile."""
import cv2
import numpy as np
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_array, sample_tensors


class ModelRunners:
    """One runner per profile; join only outputs carrying the same frame ID."""
    def __init__(self, runners):
        self.runners = runners
        self.outputs = [None] * len(runners)

    def run(self, tensors):
        for runner, tensor in zip(self.runners, tensors):
            runner.run([tensor])

    def push(self, samples):
        return all(runner.push([sample]) for runner, sample in zip(self.runners, samples))

    def pull(self, timeout_ms):
        for index, runner in enumerate(self.runners):
            if not self.outputs[index]:
                self.outputs[index] = runner.pull(timeout_ms=timeout_ms)
            if not self.outputs[index]:
                return None
        first = self.outputs[0]
        if any((output.frame_id, output.pts_ns) != (first.frame_id, first.pts_ns)
               for output in self.outputs):
            raise RuntimeError("Classification model outputs disagree on frame identity")
        result = pyneat.Sample()
        result.kind = pyneat.SampleKind.Bundle
        result.frame_id, result.pts_ns = first.frame_id, first.pts_ns
        result.fields = self.outputs
        self.outputs = [None] * len(self.runners)
        return result

    def close(self):
        for runner in self.runners:
            runner.close()


class Pipeline:
    warmup_runs = 0  # build() warms every profile with the native blank image.

    def __init__(self, config, config_path=None):
        native.CONFIG_DIRECTORY = config_path.resolve().parent
        self.profiles = native.load_profiles(config)
        if not self.profiles:
            raise ValueError("At least one classification profile is required")
        for profile in self.profiles:
            profile.labels = native.load_label_map(profile.label_map, profile.num_classes)
        self.models = [native.build_model(profile) for profile in self.profiles]

    def prepare(self, message):
        if message.encoding != "bgr8":
            raise ValueError("Classification input must be bgr8")
        bgr = image_array(message)
        tensors = []
        for profile in self.profiles:
            resized = cv2.resize(bgr, (profile.input_width, profile.input_height),
                                  interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensors.append(pyneat.Tensor.from_numpy(rgb, copy=True,
                                                    image_format=pyneat.PixelFormat.RGB))
        return tensors

    def build(self, seed):
        runners = []
        try:
            for model, profile, tensor in zip(self.models, self.profiles, seed):
                options = pyneat.RunOptions()
                options.queue_depth = 8
                options.overflow_policy = pyneat.OverflowPolicy.Block
                runner = model.build([tensor], run_options=options)
                runners.append(runner)
                blank = np.zeros((profile.input_height, profile.input_width, 3), np.uint8)
                runner.run([pyneat.Tensor.from_numpy(blank, copy=True,
                                                     image_format=pyneat.PixelFormat.RGB)])
            return ModelRunners(runners)
        except BaseException:
            for runner in runners:
                runner.close()
            raise

    def decode(self, sample, message):
        predictions = {}
        for profile, output in zip(self.profiles, sample.fields):
            scores = native.tensor_to_numpy_dense(sample_tensors(output)[0]).flatten().astype(np.float32)
            if len(scores) < profile.num_classes or not np.isfinite(scores).all():
                raise ValueError("Invalid classification scores")
            scores = scores[:profile.num_classes]
            probabilities = native.softmax(scores)
            indices = np.lexsort((np.arange(scores.size), -scores))[:profile.top_k]
            predictions[profile.name] = [(int(i), profile.labels[int(i)], float(probabilities[i]))
                                         for i in indices]
        return {"predictions": predictions}


if __name__ == "__main__":
    raise SystemExit(run("image_classification_explorer", Pipeline))
