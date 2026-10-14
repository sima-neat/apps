"""YOLO26 object detection: Neat decodes the boxes on A65; YOLO26 needs no NMS."""

from __future__ import annotations

import struct

import pyneat

from hub_models import base

BOX = struct.Struct("<iiiifi")  # x, y, width, height, score, class id in frame pixels


class Model(base.Model):
    def __init__(self, cfg: dict, packs: dict, labels: list[str], width: int, height: int):
        opt = base.image_options(
            width, height, pyneat.ResizeMode.Letterbox, pyneat.NormalizePreset.COCO_YOLO
        )
        opt.decode_type = pyneat.BoxDecodeType.YoloV26
        opt.score_threshold = cfg["min_score"]
        opt.top_k = cfg["max_detections"]
        self.neat_model = pyneat.Model(packs["model"], opt)
        self.labels = labels
        self.width = width
        self.height = height

    def postprocess(self, sample) -> list[tuple[str, dict]]:
        payload = base.tensors(sample)[0].copy_payload_bytes()
        return [("object-detection", {"objects": parse_boxes(payload, self.labels, self.width, self.height)})]


def parse_boxes(payload: bytes, labels: list[str], width: int, height: int) -> list[dict]:
    """Read Neat's BBOX payload: a uint32 count followed by that many BOX records."""
    (count,) = struct.unpack_from("<I", payload, 0)
    if 4 + count * BOX.size > len(payload):
        raise RuntimeError(f"BBOX payload holds fewer than {count} boxes")
    objects = []
    for index in range(count):
        x, y, w, h, score, class_id = BOX.unpack_from(payload, 4 + index * BOX.size)
        x0, y0 = min(max(x, 0), width), min(max(y, 0), height)
        x1, y1 = min(max(x + w, 0), width), min(max(y + h, 0), height)
        objects.append(
            {
                "label": labels[class_id] if 0 <= class_id < len(labels) else str(class_id),
                "confidence": float(score),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
            }
        )
    return objects
