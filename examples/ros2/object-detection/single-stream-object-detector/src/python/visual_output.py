"""Optional outputs for an already matched source image and decoded result.

No model work or topic matching lives here. Rendering runs only when a selected
output is due. The inference node calls write before publishing its acknowledgment, so a
finite folder run cannot exit before the last selected image has been saved.
"""
from pathlib import Path
from image_transport import image_array, planar_array, header_dict
import hashlib
import json
import math
import tempfile
import time


def positive_int(settings, name, default):
    value = settings.get(name, default)
    if type(value) is not int or value < 1:
        raise ValueError(f"ros.visualization.{name} must be a positive integer")
    return value


def enabled(settings):
    value = settings.get("enabled", False)
    if type(value) is not bool:
        raise ValueError("Visualization enabled flags must be YAML true or false")
    return value


def image_bgr(message):
    import cv2
    import numpy as np
    if message.encoding == "bgr8":
        return image_array(message).copy()
    if message.encoding == "rgb8":
        return cv2.cvtColor(image_array(message), cv2.COLOR_RGB2BGR)
    if message.encoding == "mono8":
        return cv2.cvtColor(image_array(message), cv2.COLOR_GRAY2BGR)
    pixels = planar_array(message)
    if message.encoding == "nv12":
        return cv2.cvtColor(pixels, cv2.COLOR_YUV2BGR_NV12)
    if message.encoding == "nv16":
        # NV16: full-height Y and interleaved UV planes -> packed YUYV.
        return cv2.cvtColor(np.dstack((pixels[:message.height], pixels[message.height:])),
                            cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f"Cannot visualize image encoding {message.encoding}")


# Names used by the native COCO/BlazePose metadata, not guessed array indices.
SKELETON = [(a, b) for a, b in (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"))]


def render_result(frame, data):
    """Render the native JSON schemas used by live ROS pipelines, in source pixels.

    Folder apps with an existing renderer define their own render() method instead.
    Dense arrays remain in data['images'] until this call finishes.
    """
    import cv2
    import numpy as np
    color = (0, 220, 0)
    def point(x, y):
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Nonfinite visualization coordinates")
        return (int(np.clip(round(float(x)), -100000, 100000)),
                int(np.clip(round(float(y)), -100000, 100000)))
    def label(text, x=8, y=24):
        cv2.putText(frame, str(text), point(x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    for item in [*data.get("objects", []), *data.get("tracks", []),
                 *data.get("faces", []), *data.get("segments", []), *data.get("poses", [])]:
        if "boxes" in item:
            raise ValueError("Raw pose tensors need the pipeline's render method")
        if "oriented_bbox" in item:
            cx, cy, w, h, angle = item["oriented_bbox"]
            corners = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)),
                                     math.degrees(float(angle))))
            cv2.polylines(frame, [np.int32(corners)], True, color, 2)
        if "bbox" in item:
            x, y, w, h = item["bbox"]
            box = (x, y, x+w, y+h)
        elif "box" in item:
            box = item["box"]
        elif "x1" in item:
            box = [item[k] for k in ("x1", "y1", "x2", "y2")]
        else:
            box = None
        if box is not None:
            x1, y1, x2, y2 = box
            if "oriented_bbox" not in item:
                cv2.rectangle(frame, point(x1, y1), point(x2, y2), color, 2)
            text = item.get("label", item.get("class_id", ""))
            identity = item.get("track_id", item.get("id", ""))
            score = item.get("confidence", item.get("score"))
            label(f"{text} {identity}" + (f" {score:.2f}" if score is not None else ""), x1, y1-5)
        mask = item.get("mask", [])
        if len(mask):
            contour = np.asarray(mask, np.int32).reshape(-1, 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [contour], color)
            frame[:] = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
            cv2.polylines(frame, [contour], True, color, 1)
        points = item.get("keypoints", [])
        visible = {p.get("name", str(i)): point(p["x"], p["y"])
                   for i, p in enumerate(points) if p.get("confidence", 1) >= 0.5}
        for a, b in SKELETON:
            if a in visible and b in visible:
                cv2.line(frame, visible[a], visible[b], color, 2)
        for xy in visible.values():
            cv2.circle(frame, xy, 2, (0, 0, 255), -1)
    for x, y, *_ in data.get("points", []):
        cv2.circle(frame, point(x, y), 2, color, -1)
    row = 1
    for model, predictions in data.get("predictions", {}).items():
        for index, name, confidence in predictions:
            label(f"{model}: {name} ({index}) {confidence:.3f}", y=24*row)
            row += 1
    if "class_id" in data:
        label(f"class {data['class_id']}  {data.get('value', 0):.3f}")
    for topic, values, _encoding in data.get("images", []):
        if topic not in ("depth", "anomaly_map"):
            continue
        values = np.asarray(values, np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Visualization expects a finite two-dimensional dense map")
        normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO if topic == "depth" else cv2.COLORMAP_JET)
        heat = cv2.resize(heat, frame.shape[1::-1])
        frame = np.concatenate((frame, heat), axis=1) if topic == "depth" else cv2.addWeighted(frame, .5, heat, .5, 0)
    return frame


def fit_video_frame(frame, shape):
    """Fit a rendered image into a fixed video canvas without changing its aspect ratio."""
    import cv2
    height, width = shape[:2]
    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, min(width, round(source_w * scale)))
    resized_h = max(1, min(height, round(source_h * scale)))
    if (resized_h, resized_w) != (source_h, source_w):
        frame = cv2.resize(frame, (resized_w, resized_h),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top, left = (height - resized_h) // 2, (width - resized_w) // 2
    return cv2.copyMakeBorder(frame, top, height - resized_h - top,
                              left, width - resized_w - left, cv2.BORDER_CONSTANT,
                              value=(0, 0, 0))


class InsightVideo:
    """Public Neat input -> hardware H.264 encoder -> Insight VideoSender graph.

    Overlays are already drawn on their exact source frame. No separately timed
    metadata is required for the 2D view, including external ROS camera inputs.
    """
    def __init__(self, settings, channel, shape):
        import numpy as np
        import pyneat
        height, width = shape[:2]
        # Keep one encoder canvas per stream, including mixed-size image folders.
        height, width = height + height % 2, width + width % 2
        self.shape = (height, width, 3)
        self.settings, self.channel = settings, channel
        self.metadata = None
        self.sequence = 0
        self.started_ns = time.monotonic_ns()
        source = pyneat.InputOptions()
        source.payload_type = pyneat.PayloadType.Image
        source.format = pyneat.Format.BGR
        source.width, source.height, source.depth = width, height, 3
        source.fps_n, source.fps_d = settings["fps"], 1
        encode = pyneat.SimaEncodeOptions()
        encode.fps = settings["fps"]
        options = pyneat.VideoSenderOptions.from_raw(encode)
        options.host, options.channel = settings["host"], channel
        options.video_port_base = settings["video_port_base"]
        options.encoder.bitrate_kbps = settings["bitrate_kbps"]
        self.graph = pyneat.Graph("ros_visual_output")
        self.graph.add(pyneat.nodes.input(source))
        self.graph.add(pyneat.groups.video_sender(options))
        run_options = pyneat.RunOptions()
        run_options.queue_depth = 2
        run_options.input_timeout_ms = 1000
        run_options.overflow_policy = pyneat.OverflowPolicy.Block
        self.run = self.graph.build([self.tensor(np.zeros(self.shape, np.uint8))], run_options)

    @staticmethod
    def tensor(frame):
        import pyneat
        return pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.BGR,
                                        memory=pyneat.TensorMemory.CPU)

    def send(self, frame, auxiliary=None):
        if frame.shape != self.shape:
            frame = fit_video_frame(frame, self.shape)
        tensor = self.tensor(frame)
        if auxiliary is None:
            inputs = [tensor]
        else:
            import pyneat
            # A local output clock keeps re-encoded video and auxiliary metadata
            # aligned even when the external ROS camera uses a different clock.
            pts = time.monotonic_ns() - self.started_ns + 1_000_000_000
            self.sequence += 1
            sample = pyneat.make_tensor_sample("", tensor)
            sample.payload_type = pyneat.PayloadType.Image
            sample.media_type, sample.format, sample.payload_tag = "video/x-raw", "BGR", "BGR"
            sample.pts_ns, sample.frame_id = pts, self.sequence
            inputs = [sample]
        if not self.run.push(inputs):
            raise RuntimeError("Insight video push failed")
        if auxiliary is not None:
            if self.metadata is None:
                options = pyneat.MetadataSenderOptions()
                options.host, options.channel = self.settings["host"], self.channel
                options.metadata_port_base = self.settings["metadata_port_base"]
                self.metadata = pyneat.MetadataSender(options)
            payload = {"type": "auxiliary-visualization", "data": auxiliary,
                       "timestamp": pts // 1_000_000, "pts_ns": pts,
                       "rtp_timestamp": ((pts * 90_000) // 1_000_000_000) & 0xFFFFFFFF,
                       "frame_id": str(self.sequence)}
            if not self.metadata.send_raw_json(json.dumps(payload, allow_nan=False, separators=(",", ":"))):
                raise RuntimeError("Insight auxiliary metadata send failed")

    def close(self):
        self.run.close()


class VisualOutput:
    """Synchronous, opt-in sink; no extra queue or inference copies when disabled."""
    def __init__(self, settings, render=None, video_factory=InsightVideo, auxiliary=None):
        self.save = dict(settings.get("save", {}))
        self.insight = dict(settings.get("insight", {}))
        self.save_enabled, self.insight_enabled = enabled(self.save), enabled(self.insight)
        self.enabled = self.save_enabled or self.insight_enabled
        self.render = render or render_result
        use_auxiliary = self.insight.get("auxiliary", True)
        if type(use_auxiliary) is not bool:
            raise ValueError("insight.auxiliary must be true or false")
        self.auxiliary = auxiliary if use_auxiliary else None
        self.video_factory = video_factory
        self.videos, self.counts, self.last_sent = {}, {}, {}
        self.directory = None
        self.saved = self.sent = 0
        self.max_streams = positive_int(settings, "max_streams", 4)
        self.max_visible = positive_int(self.insight, "max_visible_streams", 4)
        self.save_every = positive_int(self.save, "every_n_frames", 1)
        self.channels = dict(self.insight.get("channels", {}))
        if any(not isinstance(k, str) or type(v) is not int or v < 0 for k,v in self.channels.items()):
            raise ValueError("Insight channels must map frame_id strings to nonnegative channel integers")
        if len(set(self.channels.values())) != len(self.channels):
            raise ValueError("Insight channels must be distinct")
        for key, default in (("fps", 30), ("video_port_base", 9000), ("metadata_port_base", 9100),
                             ("bitrate_kbps", 4000)):
            self.insight[key] = positive_int(self.insight, key, default)
        self.channel_base = self.insight.get("channel_base", 0)
        if type(self.channel_base) is not int or self.channel_base < 0:
            raise ValueError("Insight channel_base must be a nonnegative integer")
        if self.insight_enabled:
            host = self.insight.get("host", "")
            if not isinstance(host, str) or not host.strip() or '<' in host:
                raise ValueError("Set ros.visualization.insight.host to a reachable Insight host")
            maximum = max(self.channels.values(), default=self.channel_base + self.max_streams - 1)
            if self.insight["video_port_base"] + maximum > 65535:
                raise ValueError("Insight video channel port exceeds 65535")
            if self.auxiliary and self.insight["metadata_port_base"] + maximum > 65535:
                raise ValueError("Insight metadata channel port exceeds 65535")
        if self.save_enabled and not self.save.get("directory"):
            raise ValueError("Set ros.visualization.save.directory")

    def write(self, message, data, sequence):
        if not self.enabled:
            return
        camera = message.header.frame_id
        if camera not in self.counts:
            if len(self.counts) >= self.max_streams:
                raise ValueError("Visualization max_streams exceeded")
            self.counts[camera] = 0
        count = self.counts[camera]
        self.counts[camera] += 1
        save = self.save_enabled and count % self.save_every == 0
        now = time.monotonic()
        send = self.insight_enabled and now - self.last_sent.get(camera, -math.inf) >= 1 / self.insight["fps"]
        stream_index = list(self.counts).index(camera)
        if self.channels:
            send = send and camera in self.channels
        else:
            send = send and stream_index < self.max_visible
        if not save and not send:
            return
        frame = self.render(image_bgr(message), data)
        if save:
            import cv2
            if self.directory is None:
                parent = Path(self.save["directory"])
                parent.mkdir(parents=True, exist_ok=True)
                self.directory = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
            digest = hashlib.sha256(camera.encode()).hexdigest()[:12]
            target = self.directory / f"{digest}-{sequence:08d}.png"
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f"Could not save visualization: {target}")
            record = {"sequence": sequence, "header": header_dict(message.header), "image": target.name}
            with (self.directory / "frames.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            self.saved += 1
        if send:
            if camera not in self.videos:
                channel = self.channels.get(camera, self.channel_base + stream_index)
                self.videos[camera] = self.video_factory(self.insight, channel, frame.shape)
            if self.auxiliary is None:
                self.videos[camera].send(frame)
            else:
                self.videos[camera].send(frame, self.auxiliary(data))
            self.last_sent[camera] = now
            self.sent += 1

    def close(self):
        error = None
        for video in self.videos.values():
            try:
                video.close()
            except Exception as exc:
                error = error or exc
        self.videos.clear()
        if self.enabled:
            print(json.dumps({"visualization_saved": self.saved, "visualization_sent": self.sent,
                              "visualization_directory": str(self.directory) if self.directory else None}), flush=True)
        if error is not None:
            raise error
