"""One explicit model node per configured stream; launch starts them together."""
import pyneat
from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        index = int(config.get("ros", {}).get("stream_index", 0))
        if not 0 <= index < len(self.config.streams):
            raise ValueError("stream_index is outside the configured streams")
        self.stream = self.config.streams[index]
        self.labels = native.load_labels(self.config.labels_path)
        self.model = None
        self.geometry = None

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("Multi-model inference expects nv12 images")
        geometry = (message.width, message.height)
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("Stream geometry changed")
        self.geometry = geometry
        if self.model is None:
            self.model = native.build_model(self.config, self.stream, *geometry)
        return [image_tensor(message)]

    def build(self, seed):
        options = native.build_model_run_options(self.config.max_inflight_per_stream)
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        cfg = self.config
        if self.stream.task == native.TASK_DETECTION:
            boxes = native.parse_boxes_strict(native.extract_bbox_payload(sample),
                                              message.width, message.height, cfg.max_detections)
            data, dropped = native.detection_metadata_data(boxes, self.labels, message.width, message.height)
        elif self.stream.task == native.TASK_SEGMENTATION:
            segments = native.decode_segments(sample, message.width, message.height, cfg.max_detections)
            data, dropped = native.segmentation_metadata_data(segments, self.labels,
                                                               (message.height, message.width), cfg.mask_threshold)
        else:
            poses = native.decode_poses(sample, message.width, message.height, cfg.max_detections)
            data, dropped = native.pose_metadata_data(poses)
        if dropped:
            raise ValueError(f"Native metadata limits rejected {dropped} instances")
        return data


if __name__ == "__main__":
    raise SystemExit(run("multi_stream_multi_model", Pipeline))
