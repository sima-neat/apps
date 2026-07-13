#!/usr/bin/env python3
# Copyright 2026 SiMa Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only 30-minute Insight ingest/browser stability gate for App16.

Only GET requests are issued. The script observes the operator viewer that
already owns metadata; it never calls /offer, creates a peer, or navigates a
tab. Every requested channel must advance video and metadata on every sample.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def channel_ids(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("channel IDs must be comma-separated integers") from exc
    if not result or any(item < 0 for item in result) or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("channel IDs must be nonnegative and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://127.0.0.1:9900")
    parser.add_argument("--channel-ids", type=channel_ids, default=list(range(24)))
    parser.add_argument("--duration-seconds", type=float, default=1800.0)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--expected-video-fps", type=float, default=20.0)
    parser.add_argument("--expected-metadata-fps", type=float, default=20.0)
    parser.add_argument("--minimum-fps-ratio", type=float, default=0.9)
    parser.add_argument(
        "--minimum-median-fps-ratio",
        type=float,
        default=0.95,
        help="per-channel full-run median floor relative to each requested rate",
    )
    parser.add_argument("--max-bad-samples", type=int, default=0)
    parser.add_argument("--verify-tls", action="store_true")
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("insight-stability-gate")
    )
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("duration and interval must be positive")
    if args.expected_video_fps <= 0 or args.expected_metadata_fps <= 0:
        parser.error("expected rates must be positive")
    if not 0 < args.minimum_fps_ratio <= 1:
        parser.error("--minimum-fps-ratio must be in (0, 1]")
    if not 0 < args.minimum_median_fps_ratio <= 1:
        parser.error("--minimum-median-fps-ratio must be in (0, 1]")
    if args.max_bad_samples < 0:
        parser.error("--max-bad-samples must be nonnegative")
    return args


def fetch(url: str, context: ssl.SSLContext) -> dict[str, Any]:
    # Deliberately no Request(method=POST): this stability gate is read-only.
    with urllib.request.urlopen(url, context=context, timeout=10) as response:
        return json.load(response)


def by_channel(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["channel"]): row
        for row in payload.get("channels", [])
        if isinstance(row, dict) and "channel" in row
    }


def timestamp_sort_key(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        return dt.datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def browser_peer_selection_key(peer: dict[str, Any]) -> tuple[Any, ...]:
    metadata_at = timestamp_sort_key((peer.get("metadata") or {}).get("last_sent_at"))
    browser_at = timestamp_sort_key(peer.get("last_browser_report_at"))
    peer_id = peer.get("id")
    return (
        metadata_at is not None,
        metadata_at if metadata_at is not None else float("-inf"),
        browser_at is not None,
        browser_at if browser_at is not None else float("-inf"),
        peer_id if isinstance(peer_id, int) else -1,
    )


def selected_browser_peers(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in payload.get("channels", []):
        candidates = [
            peer
            for peer in row.get("peers", [])
            if peer.get("active")
            and peer.get("connection_state") == "connected"
            and peer.get("data_channel_state") == "open"
            and peer.get("browser")
        ]
        if candidates:
            selected[int(row["channel"])] = max(candidates, key=browser_peer_selection_key)
    return selected


def counter(row: dict[str, Any] | None, *path: str) -> int:
    value: Any = row or {}
    for key in path:
        if not isinstance(value, dict):
            return 0
        value = value.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def browser_report_interval(
    previous_peer: dict[str, Any] | None,
    current_peer: dict[str, Any] | None,
    wall_elapsed: float,
) -> tuple[float, str, bool]:
    """Return the interval covered by the selected browser counters.

    Insight may return cached reports while the viewer publishes its per-tile
    statistics in a staggered wave.  The counters therefore cover the browser
    report interval, which can differ from the gate's wall-clock interval.
    ``browser.time`` is part of the report containing those counters; the
    server receive timestamp is the next-best clock when the browser timestamp
    is unavailable.  Wall time is diagnostic only: an invalid report clock
    must not satisfy the browser-rate gate.
    """
    if not (
        previous_peer
        and current_peer
        and previous_peer.get("id") == current_peer.get("id")
    ):
        return wall_elapsed, "wall_elapsed", False

    candidates = (
        (
            "browser.time",
            (previous_peer.get("browser") or {}).get("time"),
            (current_peer.get("browser") or {}).get("time"),
        ),
        (
            "last_browser_report_at",
            previous_peer.get("last_browser_report_at"),
            current_peer.get("last_browser_report_at"),
        ),
    )
    for source, before_value, after_value in candidates:
        before = timestamp_sort_key(before_value)
        after = timestamp_sort_key(after_value)
        if before is not None and after is not None and after > before:
            return after - before, source, True
    return wall_elapsed, "wall_elapsed", False


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


def rate_summary(values: list[float], floor: float) -> dict[str, Any]:
    return {
        "samples": len(values),
        "min": min(values) if values else None,
        "p05": percentile(values, 0.05),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "rate_misses": sum(value < floor for value in values),
    }


def sample(
    channel_list: list[int],
    previous_ingest: dict[int, dict[str, Any]],
    current_ingest: dict[int, dict[str, Any]],
    previous_peers: dict[int, dict[str, Any]],
    current_peers: dict[int, dict[str, Any]],
    elapsed: float,
    minimum_video_fps: float,
    minimum_metadata_fps: float,
) -> dict[str, Any]:
    rows = []
    for channel in channel_list:
        ingest_before = previous_ingest.get(channel)
        ingest_after = current_ingest.get(channel)
        peer_before = previous_peers.get(channel)
        peer_after = current_peers.get(channel)
        peer_stable = bool(
            peer_before
            and peer_after
            and peer_before.get("id") == peer_after.get("id")
        )
        packet_delta = counter(ingest_after, "rtp", "packets_received") - counter(
            ingest_before, "rtp", "packets_received"
        )
        ingest_metadata_delta = counter(
            ingest_after, "metadata", "messages_received"
        ) - counter(ingest_before, "metadata", "messages_received")
        decoded_delta = counter(
            peer_after, "browser", "inbound_rtp", "frames_decoded"
        ) - counter(peer_before, "browser", "inbound_rtp", "frames_decoded")
        egress_metadata_delta = counter(
            peer_after, "metadata", "messages_sent"
        ) - counter(peer_before, "metadata", "messages_sent")
        (
            browser_elapsed,
            browser_time_source,
            browser_time_valid,
        ) = browser_report_interval(peer_before, peer_after, elapsed)
        video_fps = decoded_delta / browser_elapsed
        ingest_metadata_fps = ingest_metadata_delta / elapsed
        egress_metadata_fps = egress_metadata_delta / elapsed
        browser = (peer_after or {}).get("browser", {})
        checks = {
            "ingest_present": ingest_after is not None,
            "ingest_video_active": bool((ingest_after or {}).get("active")),
            "ingest_metadata_active": bool(
                (ingest_after or {}).get("metadata", {}).get("active")
            ),
            "ingest_video_advancing": packet_delta > 0,
            "ingest_metadata_rate": ingest_metadata_fps >= minimum_metadata_fps,
            "operator_peer_stable": peer_stable,
            "browser_video_active": bool(browser.get("video", {}).get("active")),
            "browser_video_rate": peer_stable
            and browser_time_valid
            and video_fps >= minimum_video_fps,
            "browser_metadata_rate": peer_stable
            and egress_metadata_fps >= minimum_metadata_fps,
        }
        rows.append(
            {
                "channel": channel,
                "peer_id": (peer_after or {}).get("id"),
                "packet_delta": packet_delta,
                "ingest_metadata_delta": ingest_metadata_delta,
                "decoded_frame_delta": decoded_delta,
                "egress_metadata_delta": egress_metadata_delta,
                "browser_report_elapsed_s": browser_elapsed,
                "browser_report_time_source": browser_time_source,
                "browser_report_time_valid": browser_time_valid,
                "browser_video_fps": video_fps,
                "ingest_metadata_fps": ingest_metadata_fps,
                "egress_metadata_fps": egress_metadata_fps,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    failed = [row["channel"] for row in rows if not row["passed"]]
    return {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_s": elapsed,
        "channels": rows,
        "failed_channels": failed,
        "passed": not failed,
    }


def main() -> int:
    args = parse_args()
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = prefix.with_suffix(".jsonl")
    summary_path = prefix.with_suffix(".summary.json")
    context = (
        ssl.create_default_context()
        if args.verify_tls
        else ssl._create_unverified_context()  # noqa: SLF001 - local self-signed Insight
    )
    base_url = args.base_url.rstrip("/")
    ingest_url = f"{base_url}/api/ingest/stats?all=1&verbose=1"
    egress_url = f"{base_url}/api/egress/stats?all=1&verbose=1"

    previous_ingest = by_channel(fetch(ingest_url, context))
    previous_peers = selected_browser_peers(fetch(egress_url, context))
    expected_set = set(args.channel_ids)
    missing_ingest = sorted(expected_set - set(previous_ingest))
    missing_peers = sorted(expected_set - set(previous_peers))
    if missing_ingest or missing_peers:
        raise RuntimeError(
            f"initial Insight snapshot incomplete: missing_ingest={missing_ingest} "
            f"missing_operator_peers={missing_peers}"
        )

    minimum_video_fps = args.expected_video_fps * args.minimum_fps_ratio
    minimum_metadata_fps = args.expected_metadata_fps * args.minimum_fps_ratio
    started = time.monotonic()
    deadline = started + args.duration_seconds
    previous_time = started
    samples = 0
    bad_samples = 0
    failed_channel_samples: dict[int, int] = {channel: 0 for channel in args.channel_ids}
    rate_samples: dict[int, dict[str, list[float]]] = {
        channel: {
            "browser_video_fps": [],
            "ingest_metadata_fps": [],
            "egress_metadata_fps": [],
        }
        for channel in args.channel_ids
    }
    interrupted = False

    with jsonl_path.open("w", buffering=1, encoding="utf-8") as output:
        try:
            while time.monotonic() < deadline:
                time.sleep(min(args.interval_seconds, max(0.0, deadline - time.monotonic())))
                now = time.monotonic()
                current_ingest = by_channel(fetch(ingest_url, context))
                current_peers = selected_browser_peers(fetch(egress_url, context))
                record = sample(
                    args.channel_ids,
                    previous_ingest,
                    current_ingest,
                    previous_peers,
                    current_peers,
                    now - previous_time,
                    minimum_video_fps,
                    minimum_metadata_fps,
                )
                output.write(json.dumps(record, sort_keys=True) + "\n")
                samples += 1
                if not record["passed"]:
                    bad_samples += 1
                    for channel in record["failed_channels"]:
                        failed_channel_samples[channel] += 1
                for row in record["channels"]:
                    channel_rates = rate_samples[row["channel"]]
                    for metric in channel_rates:
                        channel_rates[metric].append(float(row[metric]))
                print(
                    json.dumps(
                        {
                            "sample": samples,
                            "elapsed_s": round(now - started, 1),
                            "passed": record["passed"],
                            "failed_channels": record["failed_channels"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                previous_ingest = current_ingest
                previous_peers = current_peers
                previous_time = now
        except KeyboardInterrupt:
            interrupted = True

    elapsed_total = time.monotonic() - started
    per_channel_rates: dict[int, dict[str, Any]] = {}
    median_rate_miss_channels: list[int] = []
    median_video_floor = args.expected_video_fps * args.minimum_median_fps_ratio
    median_metadata_floor = args.expected_metadata_fps * args.minimum_median_fps_ratio
    for channel, metrics in rate_samples.items():
        summaries = {
            "browser_video_fps": rate_summary(metrics["browser_video_fps"], minimum_video_fps),
            "ingest_metadata_fps": rate_summary(
                metrics["ingest_metadata_fps"], minimum_metadata_fps
            ),
            "egress_metadata_fps": rate_summary(
                metrics["egress_metadata_fps"], minimum_metadata_fps
            ),
        }
        per_channel_rates[channel] = summaries
        medians = [
            summaries["browser_video_fps"]["median"],
            summaries["ingest_metadata_fps"]["median"],
            summaries["egress_metadata_fps"]["median"],
        ]
        if (
            any(value is None for value in medians)
            or float(medians[0]) < median_video_floor
            or float(medians[1]) < median_metadata_floor
            or float(medians[2]) < median_metadata_floor
        ):
            median_rate_miss_channels.append(channel)
    summary = {
        "channel_ids": args.channel_ids,
        "requested_duration_s": args.duration_seconds,
        "elapsed_s": elapsed_total,
        "interval_s": args.interval_seconds,
        "samples": samples,
        "bad_samples": bad_samples,
        "max_bad_samples": args.max_bad_samples,
        "failed_channel_samples": failed_channel_samples,
        "minimum_video_fps": minimum_video_fps,
        "minimum_metadata_fps": minimum_metadata_fps,
        "minimum_median_video_fps": median_video_floor,
        "minimum_median_metadata_fps": median_metadata_floor,
        "expected_video_fps": args.expected_video_fps,
        "expected_metadata_fps": args.expected_metadata_fps,
        "per_channel_rates": per_channel_rates,
        "median_rate_miss_channels": median_rate_miss_channels,
        "interrupted": interrupted,
        "read_only": True,
        "passed": not interrupted
        and elapsed_total >= args.duration_seconds * 0.99
        and samples > 0
        and bad_samples <= args.max_bad_samples
        and not median_rate_miss_channels,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"stability gate failed: {exc}", file=sys.stderr)
        sys.exit(2)
