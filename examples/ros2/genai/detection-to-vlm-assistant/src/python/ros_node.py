"""ROS detection with bounded, asynchronous native person-crop VLM requests."""
import json
from queue import Queue, Empty, Full
import threading
import time
import cv2
import pyneat
from std_msgs.msg import String
from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor, planar_array, header_dict


class Pipeline:
    def __init__(self, config, config_path=None):
        self.config = native.load_config(config_path)
        self.labels = native.load_labels(self.config.labels_path)
        self.model = None
        self.requests = Queue(maxsize=self.config.genai_max_pending_requests)
        # Include the active request in the native pending-request bound.
        self.slots = threading.BoundedSemaphore(self.config.genai_max_pending_requests)
        self.stop = threading.Event()
        self.last_request = 0.0
        self.worker = None

    def attach(self, node):
        self.node = node
        self.captions = node.create_publisher(String, "captions", 8)
        if self.config.genai_enabled:
            self.worker = threading.Thread(target=self.comment, name="person-caption")
            self.worker.start()

    def prepare(self, message):
        if message.encoding != "nv12":
            raise ValueError("Detection-to-VLM expects native nv12 source pixels")
        if self.model is None:
            self.model = native.build_model(self.config, message.width, message.height)
        return [image_tensor(message)]

    def build(self, seed):
        options = pyneat.RunOptions()
        options.queue_depth = 8
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        boxes = native.parse_boxes(sample)
        now = time.monotonic()
        if self.config.genai_enabled and now - self.last_request >= self.config.genai_interval_seconds:
            person = native.best_box_for_label(boxes, self.labels, "person")
            if person is not None and self.slots.acquire(blocking=False):
                try:
                    rgb = cv2.cvtColor(planar_array(message), cv2.COLOR_YUV2RGB_NV12)
                    crop = native.crop_box(rgb, person).copy()
                    self.requests.put_nowait((crop, header_dict(message.header)))
                    self.last_request = now
                except BaseException:
                    self.slots.release()
                    raise
        return json.loads(native.metadata_json(boxes, self.labels, self.config.classes))

    def comment(self):
        while not self.stop.is_set():
            try:
                crop, header = self.requests.get(timeout=0.1)
            except Empty:
                continue
            try:
                text = native.request_person_crop_response(crop, self.config)
                result = {"header": header, "text": text, "error": None}
            except Exception as error:
                result = {"header": header, "text": "", "error": str(error)}
            finally:
                self.slots.release()
                self.requests.task_done()
            if not self.stop.is_set():
                self.captions.publish(String(data=json.dumps(result)))

    def close(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()


if __name__ == "__main__":
    raise SystemExit(run("detection_to_vlm_assistant", Pipeline))
