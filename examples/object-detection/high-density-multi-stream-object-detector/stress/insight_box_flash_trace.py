#!/usr/bin/env python3
# Copyright 2026 SiMa Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace exactly why an Insight object-detection overlay disappears.

This is a diagnostic companion to ``insight_visual_gate.py``.  It opens a
dedicated viewer target and records, for every presented video frame:

* the browser-visible RTP timestamp;
* the matching metadata data-channel arrival, including object count;
* whether the overlay canvas was cleared and subsequently painted; and
* Insight's cumulative UDP-ingest/forward/drop counters.

Insight metadata is currently delivered to one viewer.  Do not run this while
an operator is using another Insight viewer; the script never navigates or
reuses an existing target, but its dedicated target can become that one
metadata viewer for the duration of the trace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from insight_visual_gate import DedicatedTarget, channel_ids


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
    parser.add_argument("--wait-seconds", type=float, default=12.0)
    parser.add_argument("--trace-seconds", type=float, default=15.0)
    parser.add_argument("--expected-fps", type=float, default=20.0)
    parser.add_argument(
        "--stats-poll-ms",
        type=int,
        default=250,
        help="poll /ingest/stats once per interval; 0 disables polling",
    )
    parser.add_argument(
        "--event-limit-per-channel",
        type=int,
        default=4096,
        help="bounded metadata and render-event history for each channel",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("insight-box-flash-trace.json"),
    )
    args = parser.parse_args()
    if args.layout <= 0 or args.trace_seconds <= 0 or args.wait_seconds < 0:
        parser.error("layout/trace must be positive and wait must be nonnegative")
    if args.expected_fps <= 0 or args.stats_poll_ms < 0:
        parser.error("expected FPS must be positive and stats poll must be nonnegative")
    if args.event_limit_per_channel < 64:
        parser.error("event limit per channel must be at least 64")
    return args


TRACE_HOOK_JS = r"""
(() => {
  const LIMIT = __TRACE_LIMIT__;
  const STATS_POLL_MS = __STATS_POLL_MS__;
  const channels = Object.create(null);
  const latestVideoFrame = new WeakMap();
  const activeRender = new WeakMap();
  let stats = [];
  let statsErrors = [];
  let startedAtMs = performance.now();

  const boundedPush = (array, value, limit = LIMIT) => {
    array.push(value);
    if (array.length > limit) array.splice(0, array.length - limit);
  };
  const channelState = channel => {
    const key = String(channel);
    if (!channels[key]) {
      channels[key] = {
        metadata: [],
        renders: [],
        malformedMetadata: 0,
        metadataBytes: 0,
      };
    }
    return channels[key];
  };
  const parseChannelFromCanvas = canvas => {
    const tile = canvas?.closest?.('.video-tile');
    const text = tile?.querySelector?.('.tile-banner-text')?.textContent || '';
    const match = text.match(/Channel\s+(\d+)/i);
    return match ? Number.parseInt(match[1], 10) : null;
  };
  const signedRtpDelta = (current, previous) => {
    if (!Number.isInteger(current) || !Number.isInteger(previous)) return null;
    const delta = ((current >>> 0) - (previous >>> 0)) >>> 0;
    return delta >= 0x80000000 ? delta - 0x100000000 : delta;
  };

  // Wrap the callback registered by ViewerApp itself.  Updating the map before
  // invoking it makes the canvas clear/paint below refer to this exact decoded
  // frame rather than to an independently sampled "latest" frame.
  const videoProto = HTMLVideoElement.prototype;
  const originalRequestVideoFrameCallback = videoProto.requestVideoFrameCallback;
  if (originalRequestVideoFrameCallback) {
    videoProto.requestVideoFrameCallback = function(callback) {
      const video = this;
      return originalRequestVideoFrameCallback.call(video, (now, metadata) => {
        latestVideoFrame.set(video, {
          callbackAtMs: now,
          rtpTimestamp: Number.isFinite(metadata?.rtpTimestamp)
            ? metadata.rtpTimestamp >>> 0 : null,
          presentedFrames: Number.isFinite(metadata?.presentedFrames)
            ? metadata.presentedFrames : null,
          mediaTime: Number.isFinite(metadata?.mediaTime) ? metadata.mediaTime : null,
          expectedDisplayTime: Number.isFinite(metadata?.expectedDisplayTime)
            ? metadata.expectedDisplayTime : null,
        });
        return callback.call(this, now, metadata);
      });
    };
  }

  const recordMetadata = event => {
    try {
      const raw = typeof event.data === 'string' ? event.data : String(event.data);
      const payload = JSON.parse(raw);
      const channel = Number.parseInt(payload.stream_index, 10);
      if (!Number.isInteger(channel)) return;
      const state = channelState(channel);
      const objects = payload?.data?.objects;
      const confidenceThreshold = typeof window.resolveTypeSettings === 'function'
        ? (window.resolveTypeSettings(channel, 'object-detection')?.type
            ?.confidenceThreshold ?? 0)
        : 0;
      const drawableObjects = Array.isArray(objects)
        ? objects.filter(object => (object?.confidence ?? 1) >= confidenceThreshold)
        : null;
      const confidences = Array.isArray(objects)
        ? objects.map(object => object?.confidence).filter(Number.isFinite)
        : [];
      const previous = state.metadata[state.metadata.length - 1] || null;
      const sourceRtp = Number.isInteger(payload.rtp_timestamp)
        ? payload.rtp_timestamp >>> 0 : null;
      const outgoingRtp = Number.isInteger(payload?._insight?.rtp_timestamp)
        ? payload._insight.rtp_timestamp >>> 0 : null;
      const row = {
        receivedAtMs: performance.now(),
        type: payload.type ?? null,
        objectsField: Array.isArray(objects) ? 'array' : typeof objects,
        objectCount: Array.isArray(objects) ? objects.length : null,
        confidenceThreshold,
        drawableObjectCount: drawableObjects ? drawableObjects.length : null,
        minimumConfidence: confidences.length ? Math.min(...confidences) : null,
        maximumConfidence: confidences.length ? Math.max(...confidences) : null,
        ptsNs: Number.isFinite(payload.pts_ns) ? payload.pts_ns : null,
        dtsNs: Number.isFinite(payload.dts_ns) ? payload.dts_ns : null,
        frameId: payload.frame_id ?? null,
        inputSeq: Number.isFinite(payload.input_seq) ? payload.input_seq : null,
        origInputSeq: Number.isFinite(payload.orig_input_seq)
          ? payload.orig_input_seq : null,
        sourceRtp,
        outgoingRtp,
        bytes: raw.length,
        sincePreviousMs: previous
          ? performance.now() - previous.receivedAtMs : null,
        sourceRtpDelta: previous
          ? signedRtpDelta(sourceRtp, previous.sourceRtp) : null,
        outgoingRtpDelta: previous
          ? signedRtpDelta(outgoingRtp, previous.outgoingRtp) : null,
        ptsDeltaNs: previous && Number.isFinite(payload.pts_ns)
          && Number.isFinite(previous.ptsNs)
          ? payload.pts_ns - previous.ptsNs : null,
      };
      state.metadataBytes += raw.length;
      boundedPush(state.metadata, row);
    } catch (_) {
      // A malformed payload cannot be assigned to a channel reliably.  Keep a
      // global count as well as per-channel counters for valid envelopes.
      window.__neatBoxFlashMalformedMetadata =
        (window.__neatBoxFlashMalformedMetadata || 0) + 1;
    }
  };
  const observedChannels = new WeakSet();
  const observeDataChannel = channel => {
    if (!channel || observedChannels.has(channel)) return channel;
    observedChannels.add(channel);
    channel.addEventListener('message', recordMetadata);
    return channel;
  };
  const pcProto = RTCPeerConnection.prototype;
  const originalCreateDataChannel = pcProto.createDataChannel;
  pcProto.createDataChannel = function(...args) {
    return observeDataChannel(originalCreateDataChannel.apply(this, args));
  };
  const originalSetRemoteDescription = pcProto.setRemoteDescription;
  const remoteHooked = new WeakSet();
  pcProto.setRemoteDescription = function(...args) {
    if (!remoteHooked.has(this)) {
      remoteHooked.add(this);
      this.addEventListener('datachannel', event => observeDataChannel(event.channel));
    }
    return originalSetRemoteDescription.apply(this, args);
  };

  const finalizeRender = (canvas, row) => {
    if (activeRender.get(canvas) !== row) return;
    activeRender.delete(canvas);
    row.finalizedAtMs = performance.now();
    const state = channelState(row.channel);
    boundedPush(state.renders, row);
  };
  const canvasProto = CanvasRenderingContext2D.prototype;
  const paintMethods = ['stroke', 'strokeRect', 'fill', 'fillRect', 'fillText'];
  for (const name of paintMethods) {
    const original = canvasProto[name];
    if (!original) continue;
    canvasProto[name] = function(...args) {
      const row = this?.canvas ? activeRender.get(this.canvas) : null;
      if (row) {
        row.paintOps += 1;
        row.paintByMethod[name] = (row.paintByMethod[name] || 0) + 1;
      }
      return original.apply(this, args);
    };
  }
  const originalClearRect = canvasProto.clearRect;
  canvasProto.clearRect = function(...args) {
    const result = originalClearRect.apply(this, args);
    const canvas = this?.canvas;
    if (!canvas) return result;
    const [x, y, width, height] = args.map(Number);
    if (x !== 0 || y !== 0 || width < canvas.width || height < canvas.height) {
      return result;
    }
    const channel = parseChannelFromCanvas(canvas);
    const video = canvas.closest?.('.video-tile')?.querySelector?.('video') || null;
    const frame = video ? latestVideoFrame.get(video) || null : null;
    const previous = activeRender.get(canvas);
    if (previous) finalizeRender(canvas, previous);
    const row = {
      channel,
      clearAtMs: performance.now(),
      frameRtp: Number.isInteger(frame?.rtpTimestamp) ? frame.rtpTimestamp : null,
      presentedFrames: frame?.presentedFrames ?? null,
      mediaTime: frame?.mediaTime ?? null,
      expectedDisplayTime: frame?.expectedDisplayTime ?? null,
      paintOps: 0,
      paintByMethod: {},
    };
    if (Number.isInteger(channel)) {
      activeRender.set(canvas, row);
      // ViewerApp draws synchronously after clearRect.  A microtask therefore
      // observes the complete outcome of this one video-frame callback.
      queueMicrotask(() => finalizeRender(canvas, row));
    }
    return result;
  };

  const pollStats = async () => {
    if (!STATS_POLL_MS) return;
    try {
      const response = await fetch('/ingest/stats?all=1', {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      boundedPush(stats, {
        receivedAtMs: performance.now(),
        serverTime: payload.time ?? null,
        channels: (payload.channels || []).map(row => ({
          channel: row.channel,
          metadata: row.metadata,
          forwarding: row.forwarding,
        })),
      }, Math.max(64, Math.ceil(600000 / STATS_POLL_MS)));
    } catch (error) {
      boundedPush(statsErrors, {
        receivedAtMs: performance.now(),
        error: String(error?.message || error),
      }, 64);
    }
  };
  if (STATS_POLL_MS) {
    window.addEventListener('DOMContentLoaded', () => {
      pollStats();
      window.setInterval(pollStats, STATS_POLL_MS);
    });
  }

  window.__neatBoxFlashTraceReset = () => {
    for (const key of Object.keys(channels)) delete channels[key];
    stats = [];
    statsErrors = [];
    window.__neatBoxFlashMalformedMetadata = 0;
    startedAtMs = performance.now();
    pollStats();
    return startedAtMs;
  };
  window.__neatBoxFlashTraceExport = () => ({
    startedAtMs,
    endedAtMs: performance.now(),
    malformedMetadata: window.__neatBoxFlashMalformedMetadata || 0,
    channels,
    stats,
    statsErrors,
  });
})();
"""


def signed_rtp_delta(current: int, previous: int) -> int:
    delta = ((current & 0xFFFFFFFF) - (previous & 0xFFFFFFFF)) & 0xFFFFFFFF
    return delta - 0x100000000 if delta >= 0x80000000 else delta


def nearest_metadata(
    metadata: list[dict[str, Any]], frame_rtp: int, clear_at_ms: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    exact = [
        row
        for row in metadata
        if row.get("outgoingRtp") == frame_rtp
        and isinstance(row.get("receivedAtMs"), (int, float))
    ]
    before = [row for row in exact if row["receivedAtMs"] <= clear_at_ms]
    after = [row for row in exact if row["receivedAtMs"] > clear_at_ms]
    return (before[-1] if before else None, after[0] if after else None)


def classify_render(
    render: dict[str, Any], metadata: list[dict[str, Any]]
) -> str:
    frame_rtp = render.get("frameRtp")
    # Object-detection boxes use strokeRect.  Generic paintOps can include an
    # ROI polygon, which must not mask a one-frame box disappearance.
    painted = int(render.get("paintByMethod", {}).get("strokeRect", 0)) > 0
    if not isinstance(frame_rtp, int):
        return "painted_without_frame_rtp" if painted else "cleared_without_frame_rtp"
    before, after = nearest_metadata(metadata, frame_rtp, float(render["clearAtMs"]))
    if before is not None:
        object_count = before.get("objectCount")
        drawable_count = before.get("drawableObjectCount")
        if object_count == 0:
            return "exact_empty_metadata_painted" if painted else "exact_empty_metadata_cleared"
        if isinstance(drawable_count, int) and drawable_count == 0:
            return "exact_objects_below_threshold_painted" if painted else "exact_objects_below_threshold_cleared"
        if isinstance(object_count, int) and object_count > 0:
            return "exact_nonempty_rendered" if painted else "exact_nonempty_not_painted"
        return "exact_metadata_without_objects_painted" if painted else "exact_metadata_without_objects_cleared"
    if after is not None:
        return "late_exact_metadata_overlay_held" if painted else "late_exact_metadata_cleared"
    return "no_exact_metadata_overlay_held" if painted else "no_exact_metadata_cleared"


def counter_delta(
    snapshots: list[dict[str, Any]], channel: int, field: str
) -> int | None:
    values: list[int] = []
    for snapshot in snapshots:
        for row in snapshot.get("channels", []):
            if row.get("channel") != channel:
                continue
            value = row.get("metadata", {}).get(field)
            if isinstance(value, int):
                values.append(value)
            break
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def analyze_channel(
    channel: int, state: dict[str, Any], snapshots: list[dict[str, Any]], expected_fps: float
) -> dict[str, Any]:
    metadata = list(state.get("metadata", []))
    renders = list(state.get("renders", []))
    classifications = [classify_render(row, metadata) for row in renders]
    rendered_rtps = {
        row["frameRtp"] for row in renders if isinstance(row.get("frameRtp"), int)
    }
    unmatched_timestamped_metadata = [
        row
        for row in metadata
        if isinstance(row.get("outgoingRtp"), int)
        and row["outgoingRtp"] not in rendered_rtps
    ]
    flash_indices = [
        index
        for index in range(1, len(renders) - 1)
        if int(renders[index - 1].get("paintByMethod", {}).get("strokeRect", 0)) > 0
        and int(renders[index].get("paintByMethod", {}).get("strokeRect", 0)) == 0
        and int(renders[index + 1].get("paintByMethod", {}).get("strokeRect", 0)) > 0
    ]
    expected_ticks = 90000.0 / expected_fps
    unexpected_metadata_gaps = []
    for index in range(1, len(metadata)):
        before = metadata[index - 1]
        after = metadata[index]
        if not isinstance(before.get("sourceRtp"), int) or not isinstance(after.get("sourceRtp"), int):
            continue
        delta = signed_rtp_delta(after["sourceRtp"], before["sourceRtp"])
        if delta <= 0 or not math.isclose(delta, expected_ticks, abs_tol=2.0):
            unexpected_metadata_gaps.append(
                {
                    "metadata_index": index,
                    "source_rtp_delta": delta,
                    "expected_ticks": expected_ticks,
                    "before": before,
                    "after": after,
                }
            )
    stats_fields = [
        "messages_received",
        "messages_forwarded",
        "dropped_no_data_channel",
        "dropped_queue_full",
        "send_errors",
        "invalid_json",
        "chunk_datagrams_received",
        "messages_reassembled",
        "reassembly_drops",
    ]
    stats_delta = {
        field: counter_delta(snapshots, channel, field) for field in stats_fields
    }
    return {
        "channel": channel,
        "metadata_messages_seen_in_browser": len(metadata),
        "rendered_video_frames_seen": len(renders),
        "render_classifications": dict(Counter(classifications)),
        "single_frame_flash_count": len(flash_indices),
        "single_frame_flashes": [
            {
                "render_index": index,
                "classification": classifications[index],
                "render": renders[index],
            }
            for index in flash_indices
        ],
        "unexpected_metadata_gap_count": len(unexpected_metadata_gaps),
        "unexpected_metadata_gaps": unexpected_metadata_gaps,
        # The first/last few entries can be trace-window boundary effects.  A
        # sustained interior population means Insight delivered metadata whose
        # correlated egress RTP timestamp was never presented by this viewer.
        "unmatched_timestamped_metadata_count": len(unmatched_timestamped_metadata),
        "unmatched_timestamped_metadata": unmatched_timestamped_metadata,
        "insight_ingest_stats_delta": stats_delta,
    }


def evaluate(cdp: Any, expression: str) -> Any:
    result = cdp.call(
        "Runtime.evaluate",
        {"expression": expression, "returnByValue": True, "awaitPromise": True},
    )
    if result.get("exceptionDetails"):
        raise RuntimeError(f"trace JavaScript failed: {result['exceptionDetails']}")
    return result["result"].get("value")


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    hook = TRACE_HOOK_JS.replace("__TRACE_LIMIT__", str(args.event_limit_per_channel)).replace(
        "__STATS_POLL_MS__", str(args.stats_poll_ms)
    )

    target = DedicatedTarget(args.cdp_host, args.cdp_port)
    try:
        cdp = target.cdp
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Page.addScriptToEvaluateOnNewDocument", {"source": hook})
        cdp.call(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": f"localStorage.setItem('layoutCount', '{args.layout}');"},
        )
        cdp.call("Page.navigate", {"url": args.viewer_url})
        time.sleep(args.wait_seconds)
        evaluate(cdp, "window.__neatBoxFlashTraceReset()")
        time.sleep(args.trace_seconds)
        trace = evaluate(cdp, "window.__neatBoxFlashTraceExport()")
    finally:
        target.close()

    states = trace.get("channels", {}) if isinstance(trace, dict) else {}
    snapshots = trace.get("stats", []) if isinstance(trace, dict) else []
    analysis = [
        analyze_channel(
            channel,
            states.get(str(channel), {}),
            snapshots,
            args.expected_fps,
        )
        for channel in args.channel_ids
    ]
    overall_reasons: Counter[str] = Counter()
    for row in analysis:
        overall_reasons.update(row["render_classifications"])
    payload = {
        "viewer_url": args.viewer_url,
        "expected_channel_ids": args.channel_ids,
        "expected_fps": args.expected_fps,
        "requested_trace_seconds": args.trace_seconds,
        "trace": trace,
        "analysis": analysis,
        "summary": {
            "metadata_messages_seen_in_browser": sum(
                row["metadata_messages_seen_in_browser"] for row in analysis
            ),
            "rendered_video_frames_seen": sum(
                row["rendered_video_frames_seen"] for row in analysis
            ),
            "single_frame_flash_count": sum(
                row["single_frame_flash_count"] for row in analysis
            ),
            "render_classifications": dict(overall_reasons),
            "channels_with_single_frame_flashes": [
                row["channel"] for row in analysis if row["single_frame_flash_count"]
            ],
            "channels_with_metadata_gaps": [
                row["channel"] for row in analysis if row["unexpected_metadata_gap_count"]
            ],
            "stats_poll_errors": trace.get("statsErrors", []),
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    print(f"trace written to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        print(f"box flash trace failed: {exc}", file=sys.stderr)
        sys.exit(2)
