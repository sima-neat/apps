"""Single-stream RTSP FastFlow anomaly detection Insight example using pyneat."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "common" / "config.yaml"
PULL_TIMEOUT_MS = 20000
DEFAULT_MEAN = [0.3893437, 0.35421189, 0.36142577]
DEFAULT_STDDEV = [1.0, 1.0, 1.0]

# Board-only modules, imported in main() so the config checks run anywhere.
cv2 = None
np = None
pyneat = None


@dataclass(frozen=True)
class Config:
    model_path: str
    mean: list[float]        # input normalisation the package expects, RGB, on 0..1 pixels
    stddev: list[float]
    rtsp_url: str
    tcp: bool
    latency_ms: int
    frames: int              # 0 runs until stopped
    threshold: float         # map probability that counts as anomalous
    min_region_px: int       # smallest connected region, in map pixels, that counts as a defect
    profile: bool
    profile_interval: int
    insight_host: str
    video_port: int
    heat_max: float          # map probability drawn at full heatmap intensity
    alpha: float             # heatmap opacity over the frame
    save_dir: str
    save_every: int


def rgb_stats(value, key: str) -> list[float]:
    """One normalisation value per channel, in R, G, B order."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(v) for v in value]
        except (TypeError, ValueError):
            pass
    raise ValueError(f"{key} must be three numbers, one per RGB channel")


def bool_or(raw: dict, key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be true or false")
    return value


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    model = raw.get("model") or {}
    source = raw.get("source") or {}
    inference = raw.get("inference") or {}
    runtime = raw.get("runtime") or {}
    output = raw.get("output") or {}
    insight = output.get("insight") or {}
    normalize = model.get("normalize") or {}
    cfg = Config(
        model_path=str(model.get("path") or ""),
        mean=rgb_stats(normalize.get("mean", DEFAULT_MEAN), "model.normalize.mean"),
        stddev=rgb_stats(normalize.get("stddev", DEFAULT_STDDEV), "model.normalize.stddev"),
        rtsp_url=str(source.get("rtsp_url") or ""),
        tcp=bool_or(source, "tcp", True),
        latency_ms=int(source.get("latency_ms", 100)),
        frames=int(inference.get("frames", 0)),
        threshold=float(inference.get("threshold", 0.5)),
        min_region_px=int(inference.get("min_region_px", 300)),
        profile=bool_or(runtime, "profile", False),
        profile_interval=int(runtime.get("profile_interval", 100)),
        insight_host=str(insight.get("host") or ""),
        video_port=int(insight.get("video_port", 9000)),
        heat_max=float(output.get("heat_max", 0.7)),
        alpha=float(output.get("alpha", 0.55)),
        save_dir=str(output.get("save_dir") or ""),
        save_every=int(output.get("save_every", 0)),
    )
    required = (
        ("model.path", cfg.model_path),
        ("source.rtsp_url", cfg.rtsp_url),
        ("output.insight.host", cfg.insight_host),
    )
    for key, value in required:
        if not value:
            raise ValueError(f"{key} must be set")
    if not all(s > 0 for s in cfg.stddev):
        raise ValueError("model.normalize.stddev must be > 0")
    if cfg.frames < 0:
        raise ValueError("inference.frames must be >= 0")
    if not 0.0 <= cfg.threshold <= 1.0:
        raise ValueError("inference.threshold must be between 0 and 1")
    if cfg.min_region_px <= 0:
        raise ValueError("inference.min_region_px must be > 0")
    if cfg.profile_interval <= 0:
        raise ValueError("runtime.profile_interval must be > 0")
    if cfg.save_every < 0:
        raise ValueError("output.save_every must be >= 0")
    if not 0.0 <= cfg.alpha <= 1.0:
        raise ValueError("output.alpha must be between 0 and 1")
    if cfg.heat_max <= cfg.threshold:
        raise ValueError("output.heat_max must be greater than inference.threshold")
    return cfg


def probe_stream(url: str, tcp: bool) -> tuple[int, int, int]:
    """Frame size and rate, which the graph needs before it starts."""
    capture_options_key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    previous_capture_options = os.environ.get(capture_options_key)
    os.environ[capture_options_key] = f"rtsp_transport;{'tcp' if tcp else 'udp'}"
    try:
        capture = cv2.VideoCapture(url)
    finally:
        if previous_capture_options is None:
            os.environ.pop(capture_options_key, None)
        else:
            os.environ[capture_options_key] = previous_capture_options
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"could not read the frame size and rate of {url}")
    return width, height, fps


def build_source(cfg: Config, width: int, height: int, fps: int):
    """Graph: RTSP decode only, one "frame" output.

    The model is not attached to the decoder: the CVU resize would pull the decoder's row
    padding into the frame border and flag it as a defect.
    """
    source = pyneat.RtspDecodedInputOptions()
    source.url = cfg.rtsp_url
    source.tcp = cfg.tcp
    source.latency_ms = cfg.latency_ms
    source.payload_type = 96
    source.insert_queue = True
    source.decoder_name = "decoder"
    source.decoder_raw_output = True
    source.auto_caps_from_stream = True
    source.fallback_h264_width = width
    source.fallback_h264_height = height
    source.fallback_h264_fps = fps
    source.output_caps.enable = True
    source.output_caps.format = pyneat.Format.NV12
    source.output_caps.width = width
    source.output_caps.height = height
    source.output_caps.fps = fps
    source.output_caps.memory = pyneat.CapsMemory.Any

    graph = pyneat.Graph()
    graph.connect(pyneat.groups.rtsp_decoded_input(source),
                  pyneat.nodes.output("frame", pyneat.OutputOptions.every_frame(4)))
    run_options = pyneat.RunOptions()
    run_options.preset = pyneat.RunPreset.Realtime
    run_options.queue_depth = 3
    run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
    run_options.output_memory = pyneat.OutputMemory.ZeroCopy
    return graph, graph.build(run_options)


def build_model(cfg: Config, width: int, height: int):
    """A BGR frame in, the anomaly map out. Also returns the side of the square map."""
    options = pyneat.ModelOptions()
    options.preprocess.kind = pyneat.InputKind.Image
    options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
    options.preprocess.normalize.mean = cfg.mean
    options.preprocess.normalize.stddev = cfg.stddev
    options.preprocess.normalize.has_explicit_stats = True
    options.preprocess.input_max_width = width
    options.preprocess.input_max_height = height
    options.preprocess.input_max_depth = 3
    model = pyneat.Model(cfg.model_path, options)
    return model, int(model.output_specs()[0].shape[1])


class InsightVideo:
    """A push graph: annotated frames in, H.264 RTP/UDP out to the Insight viewer."""

    def __init__(self, cfg: Config, width: int, height: int, fps: int) -> None:
        source = pyneat.InputOptions()
        source.payload_type = pyneat.PayloadType.Image
        source.format = pyneat.Format.RGB
        source.width = width
        source.height = height
        source.depth = 3
        source.fps_n = fps
        source.fps_d = 1
        source.memory_policy = pyneat.InputMemoryPolicy.Ev74
        options = pyneat.VideoSenderOptions.h264_rtp_udp_from_raw(width, height, fps)
        options.host = cfg.insight_host
        options.channel = 0
        options.video_port_base = cfg.video_port
        options.encoder.bitrate_kbps = 4000
        self.port = options.video_port
        self._graph = pyneat.Graph("insight")
        self._graph.add(pyneat.nodes.input(source))
        self._graph.add(pyneat.groups.video_sender(options))
        # The graph negotiates its caps from a first sample.
        self._run = self._graph.build([self._tensor(np.zeros((height, width, 3), np.uint8))])

    @staticmethod
    def _tensor(rgb):
        return pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB,
                                        memory=pyneat.TensorMemory.EV74)

    def send(self, frame) -> None:
        if not self._run.push([self._tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]):
            raise RuntimeError("Insight video push failed")

    def close(self) -> None:
        self._run.close()


def regions_from_map(anomaly_map, threshold: float, min_region_px: int):
    """Regions above the threshold that cover at least min_region_px, and their mask."""
    above = (anomaly_map >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(above, 8)
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_region_px]
    regions = [{"bbox": tuple(int(v) for v in stats[i, :4]),
                "score": float(anomaly_map[labels == i].max())}
               for i in keep]
    return regions, np.isin(labels, keep).astype(np.uint8)


def render(frame, anomaly_map, regions, mask, cfg: Config, fps: float) -> None:
    """Blend the heatmap over the defect regions, box them and add the verdict banner."""
    height, width = frame.shape[:2]
    if regions:
        # Fixed display span, so the overlay brightness does not flicker between frames.
        span = cfg.heat_max - cfg.threshold
        intensity = np.clip((anomaly_map - cfg.threshold) / span, 0.0, 1.0)
        scaled = cv2.resize((intensity * 255).astype(np.uint8), (width, height))
        heat = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1.0 - cfg.alpha, heat, cfg.alpha, 0.0)
        full_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.copyTo(blended, full_mask, frame)
        sx = width / anomaly_map.shape[1]
        sy = height / anomaly_map.shape[0]
        for region in regions:
            x, y, w, h = region["bbox"]
            cv2.rectangle(frame, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)), (0, 0, 255), 2)
    verdict, color = ("ANOMALY", (0, 0, 255)) if regions else ("OK", (0, 200, 0))
    cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{verdict}  score={float(anomaly_map.max()):.3f}  regions={len(regions)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:5.1f} fps", (width - 130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def frame_bgr(tensor):
    """A decoded NV12 frame as a BGR image, without the decoder's row padding."""
    width = tensor.width()
    height = tensor.height()
    raw = np.frombuffer(tensor.copy_payload_bytes(), dtype=np.uint8)
    packed = np.empty((height * 3 // 2, width), dtype=np.uint8)
    planes = sorted(tensor.planes, key=lambda plane: int(plane.byte_offset))
    for plane, rows, row0 in zip(planes, (height, height // 2), (0, height)):
        offset = int(plane.byte_offset)
        stride = int(plane.strides_bytes[0]) if plane.strides_bytes else width
        plane_rows = raw[offset:offset + stride * rows].reshape(rows, stride)
        packed[row0:row0 + rows] = plane_rows[:, :width]
    return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_NV12)


def frame_tensor(sample):
    """The frame tensor of a pulled sample, bare or in a labelled field."""
    if not sample.fields:
        return sample.tensor if sample.kind == pyneat.SampleKind.Tensor else sample.tensors[0]
    field = next(part for part in sample.fields if part.stream_label == "frame")
    return field.tensor if field.kind == pyneat.SampleKind.Tensor else field.tensors[0]


def run(cfg: Config) -> None:
    width, height, fps = probe_stream(cfg.rtsp_url, cfg.tcp)
    if width != height:
        print(f"[warn] stream is {width}x{height}; the model letterboxes non-square frames, "
              "which flattens the map", file=sys.stderr)
    model, map_side = build_model(cfg, width, height)
    video = InsightVideo(cfg, width, height, fps)
    source_graph, graph_run = build_source(cfg, width, height, fps)
    print(f"rtsp={cfg.rtsp_url} stream={width}x{height}@{fps} map={map_side}x{map_side} "
          f"threshold={cfg.threshold} min_region_px={cfg.min_region_px} "
          f"insight={cfg.insight_host} video={video.port} channel=0", flush=True)

    processed = 0
    flagged = 0
    window_regions = 0
    window_start = time.perf_counter()
    stamps: deque[float] = deque(maxlen=30)  # on-screen frame rate window
    try:
        while cfg.frames <= 0 or processed < cfg.frames:
            sample = graph_run.pull("frame", PULL_TIMEOUT_MS)
            if sample is None:
                print("[warn] timed out waiting for a frame", file=sys.stderr)
                continue
            # Release the sample right away: the decoder's small buffer pool stalls otherwise.
            frame = frame_bgr(frame_tensor(sample))
            sample = None
            tensor = pyneat.Tensor.from_numpy(np.ascontiguousarray(frame), copy=True,
                                              image_format=pyneat.PixelFormat.BGR)
            raw_map = model.run([tensor], PULL_TIMEOUT_MS)[0].to_numpy(copy=True)
            anomaly_map = np.asarray(raw_map, dtype=np.float32).reshape(map_side, map_side)
            regions, mask = regions_from_map(anomaly_map, cfg.threshold, cfg.min_region_px)

            stamps.append(time.perf_counter())
            live_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            render(frame, anomaly_map, regions, mask, cfg, live_fps)
            video.send(frame)

            processed += 1
            flagged += bool(regions)
            window_regions += len(regions)
            if cfg.save_dir and cfg.save_every and processed % cfg.save_every == 0:
                out_path = Path(cfg.save_dir) / f"frame_{processed}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    print(f"[warn] failed to write output frame: {out_path}", file=sys.stderr)
            if cfg.profile and processed % cfg.profile_interval == 0:
                elapsed = time.perf_counter() - window_start
                print(f"[profile] frames={cfg.profile_interval} "
                      f"output_fps={cfg.profile_interval / elapsed:.1f} "
                      f"avg_regions={window_regions / cfg.profile_interval:.2f}", flush=True)
                window_start = time.perf_counter()
                window_regions = 0
    finally:
        graph_run.close()
        video.close()
        print(f"processed={processed} flagged={flagged} "
              f"video_sender={cfg.insight_host}:{video.port}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="FastFlow anomaly detection on one RTSP stream, heatmap video to Insight")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-config-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.validate_config_only:
            print(f"Config validated: {args.config}")
            return 0
        global cv2, np, pyneat
        import cv2
        import numpy as np
        import pyneat
        if cfg.save_dir:
            Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        run(cfg)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
