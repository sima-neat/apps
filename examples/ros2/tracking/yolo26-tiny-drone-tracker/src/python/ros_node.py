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
        cfg = self.config
        return native.ObjectTracker(native.TrackerConfig(
            high_score_threshold=cfg.tracker_high_score,
            new_track_threshold=cfg.tracker_new_track_score,
            match_iou_threshold=cfg.tracker_iou_threshold,
            max_center_distance=cfg.tracker_max_center_distance,
            velocity_momentum=cfg.tracker_velocity_momentum,
            max_missing_frames=cfg.tracker_max_missing,
            min_confirmed_hits=cfg.tracker_min_confirmed_hits,
            max_active_tracks=cfg.tracker_max_active,
            center_distance_enabled=cfg.tracker_center_distance_enabled))

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
        boxes = native.filter_target_class(boxes, self.config.target_class_id)
        tracks = tracker.update(boxes, frame)
        self.trackers[camera] = [tracker, frame + 1, sample.pts_ns]
        data = {"tracks": native.build_metadata_tracks(tracks, message.width, message.height, self.config.target_label)}
        if frame < self.config.warmup_frames:
            data["warming_up"] = True
        return data


if __name__ == "__main__":
    raise SystemExit(run("yolo26_tiny_drone_tracker", Pipeline))
