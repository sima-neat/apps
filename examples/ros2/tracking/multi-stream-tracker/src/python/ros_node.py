"""Image-topic tracking with independent native tracker state for each camera."""
import pyneat

from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor


class Pipeline:
    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        self.model = native.build_model(self.config)
        self.trackers = {}
        self.max_streams = len(self.config.rtsp_urls)

    def new_tracker(self):
        return native.MultiClassTracker(self.config.tracker_classes)

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("Tracking expects the native decoder's nv12 images")
        return [image_tensor(message)]

    def build(self, seed):
        options = native.build_run_options()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        camera = message.header.frame_id
        if camera not in self.trackers:
            if len(self.trackers) >= self.max_streams:
                raise ValueError("More camera frame_ids than configured streams")
            self.trackers[camera] = [self.new_tracker(), 0, -1]
        tracker, frame, previous_pts = self.trackers[camera]
        if sample.pts_ns <= previous_pts:
            raise ValueError("Tracking requires increasing timestamps per camera")
        boxes = native.parse_boxes_strict(native.extract_bbox_payload(sample), message.width,
                                         message.height, self.config.max_detections)
        tracks = tracker.update(boxes, frame)
        self.trackers[camera] = [tracker, frame + 1, sample.pts_ns]
        data = {"tracks": native.build_metadata_tracks(tracks, message.width, message.height)}
        if frame < self.config.warmup_frames:
            data["warming_up"] = True
        return data


if __name__ == "__main__":
    raise SystemExit(run("multi_stream_tracker", Pipeline))
