#!/usr/bin/env python3
# Copyright 2026 SiMa Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate moving App16 video and live boxes in a dedicated Insight tab.

This is an active visual gate: it creates its own Chromium target and therefore
must be the only Insight viewer while it runs (Insight currently gives metadata
to one viewer).  The target is always closed before exit; no existing operator
tab is navigated or reused.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def channel_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channel IDs must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("channel IDs must be nonnegative and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdp-host", default="127.0.0.1")
    parser.add_argument("--cdp-port", type=int, default=9222)
    parser.add_argument("--viewer-url", required=True)
    parser.add_argument(
        "--channel-ids",
        type=channel_ids,
        default=list(range(24)),
        help="comma-separated Insight channel IDs (default: 0..23)",
    )
    parser.add_argument("--layout", type=int, default=24)
    parser.add_argument("--wait-seconds", type=float, default=15.0)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument(
        "--temporal-samples",
        type=int,
        default=7,
        help="snapshots across sample window; use one every five seconds for the long gate",
    )
    parser.add_argument("--expected-fps", type=float, default=20.0)
    parser.add_argument("--minimum-fps-ratio", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        help=(
            "optional channel-marker manifest; proves each tile received its assigned media, "
            "not merely some moving stream"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("insight-visual-gate"),
        help="output path without .json or .png suffix",
    )
    parser.add_argument(
        "--keep-target-on-success",
        action="store_true",
        help="leave the dedicated viewer target open after a passing gate",
    )
    args = parser.parse_args()
    if args.layout <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--layout and dimensions must be positive")
    if args.wait_seconds < 0 or args.sample_seconds <= 0 or args.expected_fps <= 0:
        parser.error("wait must be nonnegative; sample time and FPS must be positive")
    if args.temporal_samples < 3:
        parser.error("--temporal-samples must be at least 3")
    if not 0 < args.minimum_fps_ratio <= 1:
        parser.error("--minimum-fps-ratio must be in (0, 1]")
    return args


def fetch_json(url: str, method: str = "GET") -> dict[str, Any]:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


class Cdp:
    def __init__(self, websocket_url: str, origin: str) -> None:
        try:
            import websocket
        except ImportError as exc:  # pragma: no cover - host dependency check
            raise RuntimeError(
                "websocket-client is required; install it with "
                "'python3 -m pip install websocket-client'"
            ) from exc
        self._socket = websocket.create_connection(
            websocket_url, timeout=15, origin=origin
        )
        self._sequence = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._sequence += 1
        request_id = self._sequence
        self._socket.send(
            json.dumps({"id": request_id, "method": method, "params": params or {}})
        )
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})

    def close(self) -> None:
        self._socket.close()


class DedicatedTarget:
    def __init__(self, host: str, port: int) -> None:
        self._base = f"http://{host}:{port}"
        create_url = f"{self._base}/json/new?{urllib.parse.quote('about:blank', safe='')}"
        target = fetch_json(create_url, method="PUT")
        self.id = str(target["id"])
        self.cdp = Cdp(target["webSocketDebuggerUrl"], f"http://{host}:{port}")

    def close(self) -> None:
        try:
            self.cdp.close()
        finally:
            try:
                with urllib.request.urlopen(
                    f"{self._base}/json/close/{self.id}", timeout=5
                ) as response:
                    response.read()
            except OSError:
                # The page may already have closed itself; do not mask gate data.
                pass


# Install before viewer JavaScript. Paint operations count real overlay work;
# clearRect is tracked separately because Insight clears every animation frame
# even when no metadata is arriving.
CANVAS_HOOK_JS = r"""
(() => {
  const proto = CanvasRenderingContext2D.prototype;
  const wrap = (name, field) => {
    const original = proto[name];
    if (!original || original.__app16Wrapped) return;
    function wrapped(...args) {
      const canvas = this && this.canvas;
      if (canvas) canvas[field] = (canvas[field] || 0) + 1;
      return original.apply(this, args);
    }
    wrapped.__app16Wrapped = true;
    proto[name] = wrapped;
  };
  wrap('clearRect', '__app16ClearOps');
  for (const name of ['stroke', 'strokeRect', 'fill', 'fillRect', 'fillText']) {
    wrap(name, '__app16PaintOps');
  }

  // Track the RTP timestamp of every frame actually presented by each video
  // element. This must be installed before Insight creates its tiles so the
  // App metadata RTP timestamp can be compared without assuming that the
  // publisher and pipeline PTS origins both started at frame zero.
  const videoFrameMetadata = new WeakMap();
  const trackedVideos = new WeakSet();
  const originalVideoFrameCallback = HTMLVideoElement.prototype.requestVideoFrameCallback;
  const trackVideo = video => {
    if (!originalVideoFrameCallback || trackedVideos.has(video)) return;
    trackedVideos.add(video);
    const onFrame = (_, metadata) => {
      const previous = videoFrameMetadata.get(video) || {callbackCount: 0};
      videoFrameMetadata.set(video, {
        callbackCount: previous.callbackCount + 1,
        rtpTimestamp: Number.isFinite(metadata.rtpTimestamp)
          ? metadata.rtpTimestamp : null,
        presentedFrames: Number.isFinite(metadata.presentedFrames)
          ? metadata.presentedFrames : null,
        mediaTime: Number.isFinite(metadata.mediaTime) ? metadata.mediaTime : null,
      });
      originalVideoFrameCallback.call(video, onFrame);
    };
    originalVideoFrameCallback.call(video, onFrame);
  };
  const scanVideos = node => {
    if (node instanceof HTMLVideoElement) trackVideo(node);
    if (node && node.querySelectorAll) {
      for (const video of node.querySelectorAll('video')) trackVideo(video);
    }
  };
  new MutationObserver(records => {
    for (const record of records) {
      for (const node of record.addedNodes) scanVideos(node);
    }
  }).observe(document, {childList: true, subtree: true});
  document.addEventListener('DOMContentLoaded', () => scanVideos(document));
  window.__app16VideoFrameMetadataFor = video => videoFrameMetadata.get(video) || null;

  window.__app16MetadataByChannel = Object.create(null);
  window.__app16Peers = [];
  const observedChannels = new WeakSet();
  const remoteHookedPeers = new WeakSet();
  const observeMetadataChannel = channel => {
    if (!channel || observedChannels.has(channel)) return channel;
    observedChannels.add(channel);
    channel.addEventListener('message', event => {
      try {
        const payload = JSON.parse(event.data);
        const stream = Number.parseInt(payload.stream_index, 10);
        if (!Number.isInteger(stream)) return;
        const previous = window.__app16MetadataByChannel[stream] || {count: 0};
        window.__app16MetadataByChannel[stream] = {
          count: previous.count + 1,
          ptsNs: Number.isFinite(payload.pts_ns) ? payload.pts_ns : null,
          rtpTimestamp: Number.isFinite(payload.rtp_timestamp)
            ? payload.rtp_timestamp : null,
          frameId: payload.frame_id ?? null,
          receivedAtMs: performance.now(),
        };
      } catch (_) {}
    });
    return channel;
  };

  const pcProto = RTCPeerConnection.prototype;
  const originalCreateDataChannel = RTCPeerConnection.prototype.createDataChannel;
  RTCPeerConnection.prototype.createDataChannel = function(...args) {
    const channel = originalCreateDataChannel.apply(this, args);
    return observeMetadataChannel(channel);
  };

  // Also cover a peer where Insight creates the channel remotely. Installing
  // the listener immediately before setRemoteDescription guarantees it is in
  // place before the browser can emit the negotiated `datachannel` event.
  const originalSetRemoteDescription = pcProto.setRemoteDescription;
  pcProto.setRemoteDescription = function(...args) {
    if (!remoteHookedPeers.has(this)) {
      remoteHookedPeers.add(this);
      window.__app16Peers.push(this);
      this.addEventListener('datachannel', event => {
        observeMetadataChannel(event.channel);
      });
    }
    return originalSetRemoteDescription.apply(this, args);
  };
})();
"""


def sample_js(
    expected_ids: list[int],
    marker: dict[str, int] | None,
    temporal: dict[str, Any] | None,
) -> str:
    return r"""
(() => {
  const expected = EXPECTED_IDS;
  const identityMarker = IDENTITY_MARKER;
  const temporalMarker = TEMPORAL_MARKER;
  return Array.from(document.querySelectorAll('.video-tile')).map((tile, index) => {
    const video = tile.querySelector('video');
    const canvas = tile.querySelector('canvas');
    const banner = tile.querySelector('.tile-banner-text');
    const bannerText = banner ? banner.textContent : '';
    const match = bannerText.match(/Channel\s+(\d+)/i);
    const channelId = match ? Number.parseInt(match[1], 10) : null;
    let overlayAlphaSamples = 0;
    if (canvas && canvas.width && canvas.height) {
      const pixels = canvas.getContext('2d').getImageData(
        0, 0, canvas.width, canvas.height
      ).data;
      for (let offset = 3; offset < pixels.length; offset += 4) {
        if (pixels[offset]) {
          overlayAlphaSamples += 1;
          if (overlayAlphaSamples >= 1024) break;
        }
      }
    }
    let pixelHash = null;
    let nonBlackSamples = 0;
    let identityRgb = null;
    let temporalCode = null;
    if (video && video.readyState >= 2 && video.videoWidth && video.videoHeight) {
      const probe = document.createElement('canvas');
      probe.width = 32;
      probe.height = 18;
      const ctx = probe.getContext('2d', {willReadFrequently: true});
      ctx.drawImage(video, 0, 0, probe.width, probe.height);
      const pixels = ctx.getImageData(0, 0, probe.width, probe.height).data;
      let hash = 2166136261;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        const rgb = pixels[offset] | (pixels[offset + 1] << 8) | (pixels[offset + 2] << 16);
        if (rgb !== 0) nonBlackSamples += 1;
        hash ^= rgb;
        hash = Math.imul(hash, 16777619) >>> 0;
      }
      pixelHash = hash;
      if (identityMarker) {
        const markerProbe = document.createElement('canvas');
        markerProbe.width = 8;
        markerProbe.height = 8;
        const markerCtx = markerProbe.getContext('2d', {willReadFrequently: true});
        markerCtx.drawImage(
          video,
          identityMarker.x, identityMarker.y,
          identityMarker.width, identityMarker.height,
          0, 0, markerProbe.width, markerProbe.height
        );
        const markerPixels = markerCtx.getImageData(
          0, 0, markerProbe.width, markerProbe.height
        ).data;
        const sum = [0, 0, 0];
        for (let offset = 0; offset < markerPixels.length; offset += 4) {
          sum[0] += markerPixels[offset];
          sum[1] += markerPixels[offset + 1];
          sum[2] += markerPixels[offset + 2];
        }
        const count = markerPixels.length / 4;
        identityRgb = sum.map(value => value / count);
      }
      if (temporalMarker) {
        const bitProbe = document.createElement('canvas');
        const stripWidth = (temporalMarker.bits - 1) * temporalMarker.bit_stride
          + temporalMarker.bit_width;
        bitProbe.width = stripWidth;
        bitProbe.height = temporalMarker.bit_height;
        const bitCtx = bitProbe.getContext('2d', {willReadFrequently: true});
        // One draw freezes the complete strip from one composited video frame.
        // Per-bit drawImage calls can straddle two frames and synthesize a code
        // that never existed in the source.
        bitCtx.drawImage(
          video,
          temporalMarker.x, temporalMarker.y,
          stripWidth, temporalMarker.bit_height,
          0, 0, stripWidth, temporalMarker.bit_height
        );
        const bitPixels = bitCtx.getImageData(
          0, 0, stripWidth, temporalMarker.bit_height
        ).data;
        let code = 0;
        for (let bit = 0; bit < temporalMarker.bits; bit += 1) {
          const x = bit * temporalMarker.bit_stride
            + Math.floor(temporalMarker.bit_width / 2);
          const y = Math.floor(temporalMarker.bit_height / 2);
          const offset = (y * stripWidth + x) * 4;
          const luma = 0.2126 * bitPixels[offset]
            + 0.7152 * bitPixels[offset + 1]
            + 0.0722 * bitPixels[offset + 2];
          if (luma >= temporalMarker.luma_threshold) code += 2 ** bit;
        }
        temporalCode = code;
      }
    }
    const quality = video && video.getVideoPlaybackQuality
      ? video.getVideoPlaybackQuality() : null;
    const decodedFrames = quality && Number.isFinite(quality.totalVideoFrames)
      ? quality.totalVideoFrames
      : (video && Number.isFinite(video.webkitDecodedFrameCount)
          ? video.webkitDecodedFrameCount : null);
    const presentedFrame = video && window.__app16VideoFrameMetadataFor
      ? window.__app16VideoFrameMetadataFor(video) : null;
    let videoRtp = null;
    if (video && video.srcObject && video.srcObject.getVideoTracks) {
      const track = video.srcObject.getVideoTracks()[0] || null;
      if (track) {
        for (const pc of (window.__app16Peers || [])) {
          const receiver = pc.getReceivers().find(candidate => candidate.track === track);
          if (!receiver || !receiver.getSynchronizationSources) continue;
          const sources = receiver.getSynchronizationSources();
          const latest = sources.reduce((selected, source) => {
            if (!Number.isFinite(source.rtpTimestamp)) return selected;
            if (!selected || Number(source.timestamp) > Number(selected.timestamp)) return source;
            return selected;
          }, null);
          if (latest) {
            videoRtp = {
              rtpTimestamp: latest.rtpTimestamp,
              sourceTimestamp: Number.isFinite(latest.timestamp) ? latest.timestamp : null,
              source: 'receiver-synchronization-source',
            };
            break;
          }
        }
      }
    }
    if (!videoRtp && presentedFrame && Number.isFinite(presentedFrame.rtpTimestamp)) {
      videoRtp = {
        rtpTimestamp: presentedFrame.rtpTimestamp,
        sourceTimestamp: null,
        source: 'presented-video-frame',
      };
    }
    return {
      index,
      expectedChannelId: expected[index] ?? null,
      channelId,
      active: tile.dataset.active || '',
      banner: bannerText,
      currentTime: video ? video.currentTime : 0,
      decodedFrames,
      width: video ? video.videoWidth : 0,
      height: video ? video.videoHeight : 0,
      pixelHash,
      nonBlackSamples,
      identityRgb,
      temporalCode,
      videoFrame: presentedFrame,
      videoRtp,
      metadata: channelId === null
        ? null
        : (window.__app16MetadataByChannel[channelId] || null),
      overlayAlphaSamples,
      overlayPaintOps: canvas ? (canvas.__app16PaintOps || 0) : 0,
      overlayClearOps: canvas ? (canvas.__app16ClearOps || 0) : 0,
    };
  });
})()
""".replace("EXPECTED_IDS", json.dumps(expected_ids)).replace(
        "IDENTITY_MARKER", json.dumps(marker)
    ).replace("TEMPORAL_MARKER", json.dumps(temporal))


def evaluate_tiles(
    cdp: Cdp,
    expected_ids: list[int],
    marker: dict[str, int] | None,
    temporal: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    result = cdp.call(
        "Runtime.evaluate",
        {"expression": sample_js(expected_ids, marker, temporal), "returnByValue": True},
    )
    return result["result"]["value"]


def is_active(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "active"}


def load_identity_manifest(
    path: Path | None, expected_ids: list[int], width: int, height: int
) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if payload.get("width") != width or payload.get("height") != height:
        raise ValueError(
            "identity manifest dimensions do not match --width/--height: "
            f"{payload.get('width')}x{payload.get('height')} vs {width}x{height}"
        )
    marker = payload.get("marker")
    required_marker = {"x", "y", "width", "height"}
    if not isinstance(marker, dict) or not required_marker <= set(marker):
        raise ValueError("identity manifest requires marker x/y/width/height")
    marker = {key: int(marker[key]) for key in required_marker}
    if (
        marker["x"] < 0
        or marker["y"] < 0
        or marker["width"] <= 0
        or marker["height"] <= 0
        or marker["x"] + marker["width"] > width
        or marker["y"] + marker["height"] > height
    ):
        raise ValueError("identity marker rectangle is outside the video frame")
    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, dict):
        raise ValueError("identity manifest requires a channels mapping")
    colors: dict[int, list[float]] = {}
    for channel in expected_ids:
        row = raw_channels.get(str(channel))
        rgb = row.get("rgb") if isinstance(row, dict) else None
        if (
            not isinstance(rgb, list)
            or len(rgb) != 3
            or any(not isinstance(value, (int, float)) or not 0 <= value <= 255 for value in rgb)
        ):
            raise ValueError(f"identity manifest has no valid RGB marker for channel {channel}")
        colors[channel] = [float(value) for value in rgb]
    temporal = payload.get("temporal")
    if temporal is not None:
        required_temporal = {
            "x",
            "y",
            "bit_width",
            "bit_height",
            "bit_stride",
            "bits",
            "period_frames",
            "fps",
            "luma_threshold",
            "sync_tolerance_frames",
        }
        if not isinstance(temporal, dict) or not required_temporal <= set(temporal):
            raise ValueError("identity manifest temporal marker is incomplete")
        temporal = {
            key: float(temporal[key])
            if key in {"fps", "luma_threshold", "sync_tolerance_frames"}
            else int(temporal[key])
            for key in required_temporal
        }
        if (
            temporal["bits"] < 2
            or temporal["bits"] > 20
            or temporal["period_frames"] <= 1
            or temporal["period_frames"] > 2 ** temporal["bits"]
            or temporal["fps"] <= 0
            or temporal["bit_width"] <= 0
            or temporal["bit_height"] <= 0
            or temporal["bit_stride"] < temporal["bit_width"]
            or temporal["x"] < 0
            or temporal["y"] < 0
            or temporal["x"] + (temporal["bits"] - 1) * temporal["bit_stride"]
            + temporal["bit_width"]
            > width
            or temporal["y"] + temporal["bit_height"] > height
        ):
            raise ValueError("identity manifest temporal marker is invalid or outside frame")
    return {
        "marker": marker,
        "colors": colors,
        "tolerance": float(payload.get("tolerance", 45.0)),
        "temporal": temporal,
    }


def rgb_distance(lhs: list[float] | None, rhs: list[float]) -> float | None:
    if not isinstance(lhs, list) or len(lhs) != 3:
        return None
    return math.sqrt(sum((float(lhs[index]) - rhs[index]) ** 2 for index in range(3)))


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rate_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def circular_distance(lhs: int, rhs: int, modulus: int) -> int:
    delta = abs(lhs - rhs) % modulus
    return min(delta, modulus - delta)


def signed_circular_delta(value: int, baseline: int, modulus: int) -> int:
    delta = (value - baseline) % modulus
    return delta - modulus if delta >= modulus // 2 else delta


def analyze_temporal_samples(
    tile_samples: list[dict[str, Any]],
    sample_times: list[float],
    temporal: dict[str, Any],
) -> dict[str, Any]:
    period = int(temporal["period_frames"])
    fps = float(temporal["fps"])
    sync_tolerance = float(temporal["sync_tolerance_frames"])
    codes = [sample.get("temporalCode") for sample in tile_samples]
    metadata = [sample.get("metadata") for sample in tile_samples]
    video_frames = [sample.get("videoFrame") for sample in tile_samples]
    video_rtp = [sample.get("videoRtp") for sample in tile_samples]
    segments: list[dict[str, Any]] = []
    media_origin_offsets: list[int] = []
    rtp_translation_offsets: list[int] = []
    all_codes_valid = all(isinstance(code, int) and 0 <= code < period for code in codes)
    all_metadata_valid = all(
        isinstance(row, dict)
        and isinstance(row.get("count"), (int, float))
        and isinstance(row.get("ptsNs"), (int, float))
        and isinstance(row.get("rtpTimestamp"), (int, float))
        for row in metadata
    )
    all_video_frames_valid = all(
        isinstance(row, dict)
        and isinstance(row.get("callbackCount"), (int, float))
        for row in video_frames
    )
    all_video_rtp_valid = all(
        isinstance(row, dict) and isinstance(row.get("rtpTimestamp"), (int, float))
        for row in video_rtp
    )

    if all_codes_valid and all_metadata_valid:
        for code, row in zip(codes, metadata):
            pts_ns = int(row["ptsNs"])
            predicted_frame = round(pts_ns * fps / 1_000_000_000) % period
            media_origin_offsets.append((predicted_frame - int(code)) % period)
    if all_metadata_valid and all_video_rtp_valid:
        for metadata_row, video_row in zip(metadata, video_rtp):
            rtp_translation_offsets.append(
                (
                    int(video_row["rtpTimestamp"])
                    - int(metadata_row["rtpTimestamp"])
                )
                & 0xFFFFFFFF
            )

    for index in range(1, len(tile_samples)):
        elapsed = sample_times[index] - sample_times[index - 1]
        expected_frames = elapsed * fps
        tolerance = max(3.0, expected_frames * 0.20)
        code_delta = None
        metadata_frame_delta = None
        metadata_count_delta = None
        video_callback_delta = None
        rtp_matches_pts = False
        if all_codes_valid:
            code_delta = (int(codes[index]) - int(codes[index - 1])) % period
        if all_metadata_valid:
            before_metadata = metadata[index - 1]
            after_metadata = metadata[index]
            metadata_count_delta = int(after_metadata["count"]) - int(before_metadata["count"])
            before_frame = round(int(before_metadata["ptsNs"]) * fps / 1_000_000_000) % period
            after_frame = round(int(after_metadata["ptsNs"]) * fps / 1_000_000_000) % period
            metadata_frame_delta = (after_frame - before_frame) % period
            expected_rtp = (
                int(after_metadata["ptsNs"]) * 90000 // 1_000_000_000
            ) & 0xFFFFFFFF
            rtp_matches_pts = (
                circular_distance(
                    int(after_metadata["rtpTimestamp"]) & 0xFFFFFFFF,
                    expected_rtp,
                    1 << 32,
                )
                <= 1
            )
        if all_video_frames_valid:
            video_callback_delta = int(video_frames[index]["callbackCount"]) - int(
                video_frames[index - 1]["callbackCount"]
            )
        video_forward = (
            code_delta is not None
            and abs(code_delta - expected_frames) <= tolerance
            and code_delta < period / 2
        )
        metadata_forward = (
            metadata_frame_delta is not None
            and abs(metadata_frame_delta - expected_frames) <= tolerance
            and metadata_frame_delta < period / 2
            and metadata_count_delta is not None
            and metadata_count_delta > 0
            and rtp_matches_pts
        )
        segments.append(
            {
                "elapsed_s": elapsed,
                "expected_frames": expected_frames,
                "tolerance_frames": tolerance,
                "video_code_delta": code_delta,
                "metadata_pts_frame_delta": metadata_frame_delta,
                "metadata_count_delta": metadata_count_delta,
                "video_callback_delta": video_callback_delta,
                "rtp_matches_pts": rtp_matches_pts,
                "video_forward": video_forward,
                "metadata_forward": metadata_forward,
                "passed": video_forward and metadata_forward,
            }
        )

    media_origin_offset_stable = bool(media_origin_offsets) and all(
        circular_distance(offset, media_origin_offsets[0], period) <= sync_tolerance
        for offset in media_origin_offsets
    )
    video_callbacks_forward = all_video_frames_valid and bool(segments) and all(
        isinstance(segment["video_callback_delta"], int)
        and segment["video_callback_delta"] > 0
        for segment in segments
    )
    rtp_sync_tolerance_ticks = round(sync_tolerance * 90000 / fps)
    rtp_translation_baseline = (
        rtp_translation_offsets[0] if rtp_translation_offsets else None
    )
    rtp_translation_deviations = (
        [
            signed_circular_delta(offset, rtp_translation_baseline, 1 << 32)
            for offset in rtp_translation_offsets
        ]
        if rtp_translation_baseline is not None
        else []
    )
    rtp_translation_stable = bool(rtp_translation_deviations) and all(
        abs(deviation) <= rtp_sync_tolerance_ticks
        for deviation in rtp_translation_deviations
    )
    return {
        "codes": codes,
        "metadata": metadata,
        "video_frames": video_frames,
        "video_rtp": video_rtp,
        "media_origin_offsets_frames": media_origin_offsets,
        "video_metadata_rtp_translation_offsets": rtp_translation_offsets,
        "video_metadata_rtp_translation_baseline": rtp_translation_baseline,
        "video_metadata_rtp_translation_deviations": rtp_translation_deviations,
        "video_metadata_rtp_translation_range_ticks": (
            max(rtp_translation_deviations) - min(rtp_translation_deviations)
            if rtp_translation_deviations
            else None
        ),
        "rtp_sync_tolerance_ticks": rtp_sync_tolerance_ticks,
        "segments": segments,
        "codes_valid": all_codes_valid,
        "metadata_valid": all_metadata_valid,
        "video_frames_valid": all_video_frames_valid,
        "video_rtp_valid": all_video_rtp_valid,
        "media_origin_offset_stable": media_origin_offset_stable,
        "video_callbacks_forward": video_callbacks_forward,
        "video_metadata_rtp_translation_stable": rtp_translation_stable,
        "video_forward": all_codes_valid
        and bool(segments)
        and all(segment["video_forward"] for segment in segments),
        "metadata_forward": all_metadata_valid
        and bool(segments)
        and all(segment["metadata_forward"] for segment in segments),
        "passed": all_codes_valid
        and all_metadata_valid
        and bool(segments)
        and all(segment["passed"] for segment in segments)
        and video_callbacks_forward
        and rtp_translation_stable,
    }


def main() -> int:
    args = parse_args()
    expected_ids = args.channel_ids
    identity = load_identity_manifest(
        args.identity_manifest, expected_ids, args.width, args.height
    )
    identity_marker = identity["marker"] if identity else None
    temporal_marker = identity["temporal"] if identity else None
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    target = DedicatedTarget(args.cdp_host, args.cdp_port)
    keep_target = False
    try:
        cdp = target.cdp
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": CANVAS_HOOK_JS})
        cdp.call(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": f"localStorage.setItem('layoutCount', '{args.layout}');"},
        )
        cdp.call("Page.navigate", {"url": args.viewer_url})
        time.sleep(args.wait_seconds)
        snapshots: list[list[dict[str, Any]]] = []
        sample_times: list[float] = []
        started = time.monotonic()
        interval = args.sample_seconds / (args.temporal_samples - 1)
        for sample_index in range(args.temporal_samples):
            target_time = started + sample_index * interval
            if sample_index > 0:
                time.sleep(max(0.0, target_time - time.monotonic()))
            snapshots.append(
                evaluate_tiles(cdp, expected_ids, identity_marker, temporal_marker)
            )
            sample_times.append(time.monotonic())
        elapsed = sample_times[-1] - sample_times[0]
        first = snapshots[0]
        second = snapshots[-1]

        screenshot = cdp.call(
            "Page.captureScreenshot",
            {"format": "png", "captureBeyondViewport": True},
        )
        prefix.with_suffix(".png").write_bytes(base64.b64decode(screenshot["data"]))

        records: list[dict[str, Any]] = []
        for index, expected_id in enumerate(expected_ids):
            tile_samples = [
                snapshot[index] if index < len(snapshot) else {} for snapshot in snapshots
            ]
            before = first[index] if index < len(first) else {}
            after = second[index] if index < len(second) else {}
            decoded_before = before.get("decodedFrames")
            decoded_after = after.get("decodedFrames")
            decoded_delta = (
                decoded_after - decoded_before
                if isinstance(decoded_before, (int, float))
                and isinstance(decoded_after, (int, float))
                else None
            )
            measured_identity = after.get("identityRgb")
            expected_identity = identity["colors"][expected_id] if identity else None
            identity_distance = (
                rgb_distance(measured_identity, expected_identity)
                if expected_identity is not None
                else None
            )
            nearest_identity = None
            if identity and isinstance(measured_identity, list):
                nearest_identity = min(
                    expected_ids,
                    key=lambda channel: float(
                        rgb_distance(measured_identity, identity["colors"][channel])
                    ),
                )
            temporal_result = (
                analyze_temporal_samples(tile_samples, sample_times, temporal_marker)
                if temporal_marker
                else None
            )
            records.append(
                {
                    "expected_channel_id": expected_id,
                    "reported_channel_id": after.get("channelId"),
                    "active": is_active(after.get("active")),
                    "time_delta_s": float(after.get("currentTime", 0))
                    - float(before.get("currentTime", 0)),
                    "decoded_frame_delta": decoded_delta,
                    "decoded_fps": decoded_delta / elapsed if decoded_delta is not None else None,
                    "dimensions_ok": after.get("width") == args.width
                    and after.get("height") == args.height,
                    "pixels_changed": len(
                        {
                            sample.get("pixelHash")
                            for sample in tile_samples
                            if sample.get("pixelHash") is not None
                        }
                    )
                    >= 2,
                    "nonblack_video": after.get("nonBlackSamples", 0) > 0,
                    "measured_identity_rgb": measured_identity,
                    "expected_identity_rgb": expected_identity,
                    "identity_distance": identity_distance,
                    "nearest_identity_channel": nearest_identity,
                    "temporal": temporal_result,
                    "overlay_visible": after.get("overlayAlphaSamples", 0) > 0,
                    "overlay_paint_delta": after.get("overlayPaintOps", 0)
                    - before.get("overlayPaintOps", 0),
                    "overlay_clear_delta": after.get("overlayClearOps", 0)
                    - before.get("overlayClearOps", 0),
                }
            )

        minimum_frames = args.expected_fps * elapsed * args.minimum_fps_ratio
        reported_ids = [record["reported_channel_id"] for record in records]
        checks = {
            "tile_count": len(first) == len(expected_ids) and len(second) == len(expected_ids),
            "expected_channel_ids": reported_ids == expected_ids,
            "unique_channel_ids": len(set(reported_ids)) == len(expected_ids),
            "all_active": all(record["active"] for record in records),
            "all_time_advancing": all(
                record["time_delta_s"] >= elapsed * 0.6 for record in records
            ),
            "all_decoded_fps": all(
                record["decoded_frame_delta"] is not None
                and record["decoded_frame_delta"] >= minimum_frames
                for record in records
            ),
            "all_dimensions": all(record["dimensions_ok"] for record in records),
            "all_pixels_moving": all(
                record["pixels_changed"] and record["nonblack_video"] for record in records
            ),
            "all_overlays_visible": all(record["overlay_visible"] for record in records),
            "all_overlays_redrawing": all(
                record["overlay_paint_delta"] > 0 for record in records
            ),
        }
        if identity:
            checks["all_media_identities_match"] = all(
                record["identity_distance"] is not None
                and record["identity_distance"] <= identity["tolerance"]
                and record["nearest_identity_channel"] == record["expected_channel_id"]
                for record in records
            )
            checks["unique_media_identities"] = len(
                {record["nearest_identity_channel"] for record in records}
            ) == len(expected_ids)
        if temporal_marker:
            checks["all_video_temporal_codes_forward"] = all(
                record["temporal"] and record["temporal"]["video_forward"]
                for record in records
            )
            checks["all_video_frame_callbacks_forward"] = all(
                record["temporal"] and record["temporal"]["video_callbacks_forward"]
                for record in records
            )
            checks["all_metadata_pts_rtp_forward"] = all(
                record["temporal"]
                and record["temporal"]["metadata_forward"]
                for record in records
            )
            checks["all_video_metadata_rtp_translation_stable"] = all(
                record["temporal"]
                and record["temporal"]["video_metadata_rtp_translation_stable"]
                for record in records
            )
        summary = {
            "expected_channel_ids": expected_ids,
            "elapsed_s": elapsed,
            "expected_fps": args.expected_fps,
            "minimum_decoded_frames": minimum_frames,
            "decoded_fps": rate_summary(
                [
                    float(record["decoded_fps"])
                    for record in records
                    if record["decoded_fps"] is not None
                ]
            ),
            "decoded_rate_miss_channels": [
                record["expected_channel_id"]
                for record in records
                if record["decoded_frame_delta"] is None
                or record["decoded_frame_delta"] < minimum_frames
            ],
            "identity_manifest": str(args.identity_manifest) if identity else None,
            "identity_tolerance": identity["tolerance"] if identity else None,
            "temporal_samples": args.temporal_samples,
            "temporal_marker": temporal_marker,
            "checks": checks,
            "passed": all(checks.values()),
            "dedicated_target_id": target.id,
        }
        keep_on_success = args.keep_target_on_success and summary["passed"]
        summary["target_kept_open"] = keep_on_success
        payload = {
            "summary": summary,
            "records": records,
            "sample_times": sample_times,
            "snapshots": snapshots,
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True))
        keep_target = keep_on_success
        return 0 if summary["passed"] else 1
    finally:
        if keep_target:
            target.cdp.close()
        else:
            target.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"visual gate failed: {exc}", file=sys.stderr)
        sys.exit(2)
