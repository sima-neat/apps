"""PatchCore image-topic inference with the native calibrated memory bank."""
# Import native first: it configures the same BLAS worker limits before NumPy.
from neat_apps.runtime import load_native
native = load_native(__file__)
import pyneat

from neat_apps.image_node import run
from neat_apps.images import image_array, image_tensor, sample_tensors


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        self.config = native.load_config(config_path)
        meta = native.load_bank_meta(self.config.bank_meta_path)
        native.verify_bank_matches_model(meta, self.config.model_path)
        native.verify_bank_hash(meta, self.config.memory_bank_path)
        self.bank = native.MemoryBank.load(self.config.memory_bank_path)
        self.threshold = float(meta["threshold"]["value"])
        self.neighbors = int(meta["num_neighbors"])
        self.visual_scale = (float(meta["patch_threshold"]["scale_min"]),
                             float(meta["patch_threshold"]["scale_max"]))
        self.model = None
        self.encoding = None

    def prepare(self, message):
        if self.model is None:
            self.encoding = message.encoding
            if message.encoding == "nv12":
                self.model = native.make_rtsp_graph_model(self.config, message.width, message.height)
            elif message.encoding == "bgr8":
                self.model = native.make_image_model(self.config.model_path)
            else:
                raise ValueError("PatchCore expects bgr8 files or nv12 live images")
        if message.encoding != self.encoding:
            raise ValueError("Input encoding changed after model construction")
        return [image_tensor(message) if self.encoding == "nv12"
                else native.model_input_tensor(image_array(message))]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = native.JOINED_BUFFERS
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def render(self, frame, data):
        import cv2
        score_map = data["images"][0][1]
        overlay = native.draw_overlay(frame, score_map, self.config.scoring.gaussian_sigma,
                                      self.config.output.overlay_alpha, *self.visual_scale)
        cv2.putText(overlay, f"{data['anomaly']['verdict']} {data['anomaly']['score']:.3f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        return overlay

    def decode(self, sample, message):
        embedding = native.extract_hwc(sample_tensors(sample)[0].to_numpy(copy=False))
        scored = self.bank.score(embedding, self.neighbors)
        verdict = "ANOMALOUS" if scored.image_score > self.threshold else "normal"
        return {"anomaly": {"verdict": verdict, "score": scored.image_score,
                            "threshold": self.threshold},
                "images": [("anomaly_map", scored.score_map, "32FC1")]}


if __name__ == "__main__":
    raise SystemExit(run("patchcore_anomaly_detector", Pipeline))
