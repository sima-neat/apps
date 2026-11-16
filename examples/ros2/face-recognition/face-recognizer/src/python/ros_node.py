"""ROS image -> SCRFD faces -> native alignment and ArcFace gallery matching."""
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pyneat
from neat_apps.runtime import load_native
native = load_native(__file__)
from neat_apps.image_node import run
from neat_apps.images import image_tensor, planar_array, sample_tensors


class Pipeline:
    warmup_runs = 0

    def __init__(self, config, config_path=None):
        native.load_runtime_dependencies()
        self.config = native.load_app_config(config_path)
        self.people = native.load_gallery(self.config.gallery_path) if Path(self.config.gallery_path).exists() else []
        self.model = self.arcface = self.source = None
        self.encoding = None

    def prepare(self, message):
        if message.encoding not in ("nv12", "nv16"):
            raise ValueError("Face recognition expects native nv12/nv16 source pixels")
        if self.model is None:
            self.encoding = message.encoding
            usb = SimpleNamespace(decoded_format=pyneat.Format.NV16) if self.encoding == "nv16" else None
            self.source = native.Source(message.width, message.height, 0, usb)
            self.model = native.build_scrfd(self.config, self.source)
        if (message.width, message.height, message.encoding) != (self.source.width, self.source.height, self.encoding):
            raise ValueError("Face-recognition source geometry or encoding changed")
        return [image_tensor(message)]

    def build(self, seed):
        self.arcface = native.build_tensor_run(self.config.arcface_model, native.ARCFACE_SIZE)
        options = pyneat.RunOptions()
        options.queue_depth = 4
        options.overflow_policy = pyneat.OverflowPolicy.Block
        return self.model.build(seed, run_options=options)

    def decode(self, sample, message):
        faces = native.decode_scrfd([t.to_numpy(copy=False) for t in sample_tensors(sample)],
                                    self.config, native.letterbox_for(message.width, message.height))
        objects = []
        if faces:
            packed = planar_array(message)
            bgr = (cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12) if message.encoding == "nv12"
                   else cv2.cvtColor(np.dstack([packed[:message.height], packed[message.height:]]),
                                     cv2.COLOR_YUV2BGR_YUY2))
            for index, face in enumerate(faces, 1):
                embedding = native.embed_face(self.arcface, bgr, face.landmarks)
                name, similarity = native.match_embedding(embedding, self.people, self.config)
                objects.append(native.insight_object(index, face, name, similarity, self.source))
        return {"objects": objects}

    def close(self):
        if self.arcface is not None:
            self.arcface.close()


if __name__ == "__main__":
    raise SystemExit(run("face_recognizer", Pipeline))
