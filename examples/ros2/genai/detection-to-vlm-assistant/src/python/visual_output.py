"""Optional detection overlays, saved images and Insight video for this app."""
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


def render_result(frame, data):
    """Draw this detector's source-coordinate boxes and labels."""
    import cv2
    import numpy as np
    color = (0, 220, 0)

    def point(x, y):
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("Nonfinite visualization coordinates")
        return (int(np.clip(round(float(x)), -100000, 100000)),
                int(np.clip(round(float(y)), -100000, 100000)))

    for item in data.get("objects", []):
        x, y, w, h = item["bbox"]
        cv2.rectangle(frame, point(x, y), point(x + w, y + h), color, 2)
        text = f"{item['label']} {item['id']} {item['confidence']:.2f}"
        cv2.putText(frame, text, point(x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


class InsightVideo:
    """Public Neat input -> hardware H.264 encoder -> Insight VideoSender graph.

    Overlays are already drawn on their exact source frame. No separately timed
    metadata is required for the 2D view, including external ROS camera inputs.
    """
    def __init__(self, settings, channel, shape):
        import numpy as np
        import pyneat
        height, width = shape[:2]
        if width % 2 or height % 2:
            raise ValueError("Insight H.264 video requires even width and height")
        self.shape = shape
        self.settings, self.channel = settings, channel
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
        self.run = self.graph.build([self.tensor(np.zeros(shape, np.uint8))], run_options)

    @staticmethod
    def tensor(frame):
        import pyneat
        return pyneat.Tensor.from_numpy(frame, copy=True, image_format=pyneat.PixelFormat.BGR,
                                        memory=pyneat.TensorMemory.CPU)

    def send(self, frame):
        if frame.shape != self.shape:
            raise ValueError("Insight geometry changed within one camera stream; restart the app")
        if not self.run.push([self.tensor(frame)]):
            raise RuntimeError("Insight video push failed")

    def close(self):
        self.run.close()


class VisualOutput:
    """Synchronous, opt-in sink; no extra queue or inference copies when disabled."""
    def __init__(self, settings, render=None, video_factory=InsightVideo):
        self.save = dict(settings.get("save", {}))
        self.insight = dict(settings.get("insight", {}))
        self.save_enabled, self.insight_enabled = enabled(self.save), enabled(self.insight)
        self.enabled = self.save_enabled or self.insight_enabled
        self.render = render or render_result
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
        for key, default in (("fps", 30), ("video_port_base", 9000), ("bitrate_kbps", 4000)):
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
            self.videos[camera].send(frame)
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
