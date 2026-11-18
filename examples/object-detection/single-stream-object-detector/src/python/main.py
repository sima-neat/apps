"""Single-stream object detector: YOLO26 on one RTSP, HTTP MJPEG or USB camera source, with the
video and the detections sent to Insight.

main() builds the whole graph. The helpers it calls are in utils/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from utils.app_config import AppConfig, load_app_config
from utils.config import load_labels
from utils.insight import build_metadata_boxes, send_metadata
from utils.runtime import pull_result_has_sample, time_ms, use_system_packages

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"

# pyneat, OpenCV and the helpers built on them are imported by load_runtime_dependencies(),
# only for a real run, so --help and --validate-config-only work without them.
pyneat = None
detections = None
detector = None


def load_runtime_dependencies() -> None:
    global pyneat, detections, detector
    use_system_packages()
    from utils import detections, detector
    import pyneat


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera RTSP YOLO26 Insight example")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    return parser.parse_args(argv)


class ProfileWindow:
    """With runtime.profile, prints averages over every profile_interval published frames."""

    def __init__(self, enabled: bool, interval: int) -> None:
        self.enabled = enabled
        self.interval = interval
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0

    def add(self, detection_pull_ms: float, metadata_send_ms: float, box_count: int) -> None:
        if not self.enabled:
            return
        if self.frames == 0:
            self.start_ms = time_ms()
        self.frames += 1
        self.boxes += box_count
        self.detection_pull_ms += detection_pull_ms
        self.metadata_send_ms += metadata_send_ms
        if self.frames >= self.interval:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or self.frames == 0:
            return
        elapsed = time_ms() - self.start_ms
        output_fps = self.frames * 1000.0 / elapsed if elapsed > 0.0 else 0.0
        print(
            f"[profile] frames={self.frames} "
            f"output_fps={output_fps} "
            f"avg_detection_pull_ms={self.detection_pull_ms / self.frames} "
            f"avg_metadata_send_ms={self.metadata_send_ms / self.frames} "
            f"avg_boxes={self.boxes / self.frames}",
            flush=True,
        )
        self.frames = 0
        self.boxes = 0
        self.start_ms = 0.0
        self.detection_pull_ms = 0.0
        self.metadata_send_ms = 0.0


@dataclass
class PipelineRuntime:
    """What the pull loop needs from the graph main() built."""

    run: object
    metadata_sender: object
    labels: list[str]
    #: Run output the loop pulls: the detections alone, or the frame-joined bundle when saving.
    output_name: str
    frame_w: int
    frame_h: int
    video_port: int


def run_pipeline(runtime: PipelineRuntime, cfg: AppConfig) -> int:
    """Pull each frame's detections, send them to Insight, and save sampled frames."""
    profile = ProfileWindow(cfg.profile, cfg.profile_interval)
    processed = 0
    while cfg.frames <= 0 or processed < cfg.frames:
        pull_start = time_ms()
        detection_sample = runtime.run.pull(runtime.output_name, 20000)
        pull_end = time_ms()
        if not pull_result_has_sample(runtime.run, detection_sample, runtime.output_name):
            print("[warn] timed out waiting for detections", file=sys.stderr)
            continue

        payload = detections.extract_bbox_payload(detection_sample)
        boxes = detections.parse_boxes_strict(
            payload, runtime.frame_w, runtime.frame_h, cfg.max_detections
        )

        metadata_start = time_ms()
        objects = build_metadata_boxes(boxes, runtime.labels, runtime.frame_w, runtime.frame_h)
        send_metadata(
            runtime.metadata_sender, "object-detection", {"objects": objects}, detection_sample
        )
        metadata_end = time_ms()

        processed += 1
        detector.maybe_save_debug_frame(cfg, processed, detection_sample, boxes)
        profile.add(pull_end - pull_start, metadata_end - metadata_start, len(boxes))

    profile.flush()
    print(f"processed={processed} video_sender={cfg.insight_host}:{runtime.video_port}")
    return processed


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        cfg = load_app_config(args.config)
        if args.validate_config_only:
            # Reports which key supplied the source, not its value: a URL can
            # carry credentials and this line ends up in terminal and CI logs.
            print(f"Config validated: {args.config} (source={cfg.source_key})")
            return 0

        load_runtime_dependencies()
        if cfg.profile:
            os.environ.setdefault("SIMA_GST_ELEMENT_TIMINGS", "1")
            os.environ.setdefault("SIMA_GST_FLOW_DEBUG", "1")
            os.environ.setdefault("SIMA_GST_BOUNDARY_PROBES", "1")
        if cfg.save_dir:
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        frame_w, frame_h, fps = detector.resolve_source_geometry(cfg)
        if fps <= 0:
            raise RuntimeError(
                f"failed to resolve source frame rate; set source.{cfg.source_type}.fps or use a "
                "source with probeable FPS metadata"
            )
        # A USB camera's JPEG goes to Insight unchanged and is decoded only for the model, in the
        # camera's own sampling: most webcams send 4:2:2 MJPEG, which decodes to NV16, and the model
        # reads NV16 directly. Other sources are decoded once and re-encoded as H.264 for Insight.
        usb = cfg.source_type == "usb"
        # The camera is checked once; the same info gives the decoded layout and builds the node.
        usb_camera = pyneat.usb_camera_info(detector.make_usb_camera_options(cfg)) if usb else None
        decoded_format = usb_camera.decoded_format if usb else pyneat.Format.NV12

        # The model resizes and normalizes the decoded frame itself and decodes YOLO26 boxes.
        model_options = pyneat.ModelOptions()
        model_options.preprocess.kind = pyneat.InputKind.Image
        model_options.preprocess.enable = pyneat.AutoFlag.On
        model_options.preprocess.color_convert.input_format = (
            pyneat.PreprocessColorFormat.NV16
            if decoded_format == pyneat.Format.NV16
            else pyneat.PreprocessColorFormat.NV12
        )
        if frame_w > 0 and frame_h > 0:
            model_options.preprocess.input_max_width = frame_w
            model_options.preprocess.input_max_height = frame_h
        model_options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        model_options.decode_type = pyneat.BoxDecodeType.YoloV26
        model_options.score_threshold = cfg.min_score
        model_options.nms_iou_threshold = cfg.nms_iou
        model_options.top_k = cfg.max_detections
        model = pyneat.Model(cfg.model_path, model_options)
        labels = load_labels(cfg.labels_path)

        # Insight video: the USB camera's JPEG as it is, or the decoded frame encoded as H.264.
        if usb:
            video_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
        else:
            encode = pyneat.SimaEncodeOptions()
            encode.fps = fps
            video_options = pyneat.VideoSenderOptions.from_raw(encode)
            video_options.encoder.bitrate_kbps = 1000
        video_options.host = cfg.insight_host
        video_options.channel = 0
        video_options.video_port_base = cfg.video_port
        video_graph = pyneat.Graph("video")
        video_graph.connect(pyneat.nodes.input("video"), pyneat.groups.video_sender(video_options))

        # Insight correlates the RTP timestamp with the metadata timestamp, so the video and the
        # detections must stay in one Run and therefore on one GStreamer timeline. The source
        # branches to the video sender, the model and, with output.save_dir, the saved frames.
        save_frames = bool(cfg.save_dir)
        source = detector.make_source_graph(cfg, fps, frame_w, frame_h, usb_camera)
        decoded_outputs = ["model", "frame"] if save_frames else ["model"]
        branch = pyneat.graphs.branch(
            "decoded" if usb else "source", decoded_outputs if usb else ["video", *decoded_outputs]
        )

        model_graph = pyneat.Graph("model")
        model_graph.connect(pyneat.nodes.input("model"), model)

        detections_graph = pyneat.Graph("detections")
        detections_graph.add(pyneat.nodes.output("detections", pyneat.OutputOptions.every_frame(4)))

        graph = pyneat.Graph()
        if usb:
            # The camera's JPEG splits before decoding: one copy to Insight, one to the decoder.
            decode_graph = pyneat.Graph("decode")
            decode_graph.connect(
                pyneat.nodes.input("decode"),
                pyneat.nodes.sima_decode(detector.make_usb_decode_options(cfg, decoded_format)),
            )
            video_link = pyneat.GraphLinkOptions()
            video_link.policy = pyneat.GraphLinkPolicy.RealtimeLatestByStream
            split = pyneat.graphs.branch("source", ["video", "decode"])
            graph.connect(source, split)
            graph.connect(split, video_graph, video_link)
            graph.connect(split, decode_graph)
            graph.connect(decode_graph, branch)
        else:
            graph.connect(source, branch)
            graph.connect(branch, video_graph)
        graph.connect(branch, model_graph)
        graph.connect(model_graph, detections_graph)
        if save_frames:
            # Each saved frame is joined with the detections from the same frame.
            frame_graph = pyneat.Graph("frame")
            frame_graph.add(pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
            joined = pyneat.graphs.combine(
                ["frame", "detections"], "detector_output", pyneat.CombinePolicy.ByFrame
            )
            graph.connect(branch, frame_graph)
            graph.connect(frame_graph, joined)
            graph.connect(detections_graph, joined)
        output_name = "detector_output" if save_frames else "detections"
        if cfg.profile:
            print(f"Backend:\n{graph.describe_backend()}")

        run_options = pyneat.RunOptions()
        run_options.preset = pyneat.RunPreset.Realtime
        run_options.queue_depth = 3
        run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        run_options.output_memory = pyneat.OutputMemory.ZeroCopy
        run = graph.build(run_options)

        metadata_options = pyneat.MetadataSenderOptions()
        metadata_options.host = cfg.insight_host
        metadata_options.channel = 0
        metadata_options.metadata_port_base = cfg.metadata_port
        metadata_sender = pyneat.MetadataSender(metadata_options)

        print(
            f"source={cfg.camera if cfg.source_type == 'usb' else cfg.source_url} "
            f"type={cfg.source_type} codec={cfg.source_codec} "
            f"stream={frame_w}x{frame_h}@{fps} "
            f"insight={cfg.insight_host} video={video_options.video_port} "
            f"metadata={metadata_sender.metadata_port()} channel=0"
        )
        runtime = PipelineRuntime(
            run=run,
            metadata_sender=metadata_sender,
            labels=labels,
            output_name=output_name,
            frame_w=frame_w,
            frame_h=frame_h,
            video_port=video_options.video_port,
        )
        try:
            run_pipeline(runtime, cfg)
        finally:
            run.close()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
