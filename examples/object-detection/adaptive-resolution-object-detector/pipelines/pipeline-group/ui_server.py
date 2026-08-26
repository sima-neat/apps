#!/usr/bin/env python3
r"""Browser control panel for the adaptive-resolution detection pipeline.

Runs ON the DevKit (192.168.135.72), where it can control the detector as a
local process and still reach the Insight API on its mapped host port. It reuses
pipeline.py for every primitive - the CLI and this UI share one brain.

  Browser  --http-->  ui_server.py (DevKit:8090)  --local-->  detector app
                                    \--https 9900--> Insight (media + stats)

Start it (detached, survives your shell):
  ssh sima@192.168.135.72 \
    'setsid nohup python3 <bundle>/pipeline-<mode>/ui_server.py \
       > <bundle>/logs/ui_server.log 2>&1 < /dev/null &'

Then open  http://192.168.135.72:8090

Stream model (persisted in ui-state.json). Every stream is one of:
  auto      Insight video, resolution auto-managed by total stream count
  pinned    a specific Insight video (e.g. an upload); fixed at its native size
  external  an external camera rtsp:// URL; fixed at whatever the camera sends

Only `auto` streams re-tier as the count changes. `pinned` and `external`
sources cannot be down-scaled by us (we do not own the encoder), so they hold
their native resolution and count against the shared decode budget as-is.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline  # noqa: E402  (path set above)

PORT = {"live": 8091, "group": 8092}.get(pipeline.PIPELINE, 8090)
WEB = HERE / "web" / "index.html"
STATE_FILE = HERE / "ui-state.json"
INSIGHT_UI = "https://192.168.131.68:9900/"   # main Neat Insight UI


# --------------------------------------------------------------------------
# persisted stream model
# --------------------------------------------------------------------------


GROUP_SIZE = pipeline.GROUP_SIZE
MAX_GROUPS = pipeline.MAX_GROUPS


def load_groups() -> list[list[dict]]:
    """Streams bucketed by group: groups[g][pos] is one stream.

    Position within a group is meaningful - it fixes that stream's Insight
    channel (group*GROUP_SIZE + pos), so the list order is the wire layout.
    """
    if STATE_FILE.exists():
        try:
            groups = json.loads(STATE_FILE.read_text()).get("groups", [])
            return [list(g) for g in groups][:MAX_GROUPS]
        except json.JSONDecodeError:
            return []
    return []


def save_groups(groups: list[list[dict]]) -> None:
    STATE_FILE.write_text(json.dumps({"groups": groups}, indent=2))


def flat(groups: list[list[dict]]) -> list[dict]:
    """Every stream across every group - for the shared-hardware budget check."""
    return [s for g in groups for s in g]


def insight_slot(group: int, pos: int) -> int:
    """Insight media slot (1-based) backing one position.

    Derived from the fixed channel, so a group always stages into the same
    slots and can never disturb another group's sources.
    """
    return pipeline.channel_for(group, pos) + 1


def plan_group(group: int, streams: list[dict]) -> tuple[list[str], list[tuple[int, str]]]:
    """Resolve ONE group's streams into (config URLs, Insight staging)."""
    urls: list[str] = []
    staging: list[tuple[int, str]] = []
    for pos, s in enumerate(streams):
        if s["kind"] == "external":
            urls.append(s["url"])
            continue
        slot = insight_slot(group, pos)
        staging.append((slot, s["video"]))
        urls.append(pipeline.insight_url(slot))
    return urls, staging


def _stop_slot(slot: int) -> None:
    """Stop one Insight source, tolerating 'already stopped' (which 500s)."""
    try:
        pipeline.api("/api/mediasrc/stop", {"index": slot})
    except Exception:  # noqa: BLE001 - stopping an idle slot is not an error here
        pass


def stage_group(group: int, staging: list[tuple[int, str]], drop_from: int) -> None:
    """Start this group's Insight sources; stop only the ones it no longer uses.

    Deliberately never calls mediasrc/stop-all - that is global and would cut
    every other group's sources, which is exactly the coupling this pipeline
    exists to avoid. Every slot touched here is inside this group's own fixed
    range, so a rebuild is invisible to the other groups.

    Each slot is stopped before being (re)assigned: Insight returns 500 when
    /mediasrc/start is called on a source that is already playing, which is
    what the global stop-all used to mask.
    """
    for slot, _ in staging:
        _stop_slot(slot)
    for slot, video in staging:
        pipeline.api("/api/mediasrc/assign", {"index": slot, "file": video})
    for slot, _ in staging:
        pipeline.api("/api/mediasrc/start", {"index": slot})
    # positions this group no longer occupies (after a removal / shrink)
    for pos in range(drop_from, GROUP_SIZE):
        _stop_slot(insight_slot(group, pos))
    if staging:
        time.sleep(3)  # let RTSP mounts come up before the app connects


def rebuild_group(group: int, streams: list[dict]) -> None:
    """Apply one group: stage its sources, write its config, restart its app.

    Touches nothing outside this group - no sibling process is signalled, no
    sibling config rewritten, no sibling Insight slot stopped.
    """
    urls, staging = plan_group(group, streams)
    stage_group(group, staging, drop_from=len(streams))
    if streams:
        pipeline.write_config_group(group, urls)
        pipeline.start_group(group)
        pipeline.wait_for_group(group, len(streams), timeout_s=300)
    else:
        pipeline.stop_group(group)


def add_many(news: list[dict]) -> str:
    """Add streams, filling groups in order and rebuilding only what changed.

    Newcomers go into whichever groups have free positions. A group that gains
    a stream is rebuilt; every other group keeps running untouched - that is
    the whole point of this pipeline.
    """
    groups = load_groups()
    touched: list[int] = []
    for s in news:
        placed = False
        for g in range(MAX_GROUPS):
            while len(groups) <= g:
                groups.append([])
            if len(groups[g]) < GROUP_SIZE:
                groups[g].append(s)
                if g not in touched:
                    touched.append(g)
                placed = True
                break
        if not placed:
            raise RuntimeError(
                f"no free slot: all {MAX_GROUPS} groups hold {GROUP_SIZE} streams")
    save_groups(groups)
    for g in touched:
        rebuild_group(g, groups[g])
    return f"rebuilt group(s) {', '.join(map(str, touched))}"


def remove_positions(targets: list[tuple[int, int]]) -> str:
    """Remove (group, pos) pairs; rebuild only the groups that lost a stream."""
    groups = load_groups()
    by_group: dict[int, set[int]] = {}
    for g, pos in targets:
        by_group.setdefault(g, set()).add(pos)
    for g, positions in by_group.items():
        if g < len(groups):
            groups[g] = [s for i, s in enumerate(groups[g]) if i not in positions]
    while groups and not groups[-1]:
        groups.pop()
    save_groups(groups)
    for g in sorted(by_group):
        rebuild_group(g, groups[g] if g < len(groups) else [])
    return f"rebuilt group(s) {', '.join(map(str, sorted(by_group)))}"


# --------------------------------------------------------------------------
# runnability guard - can this configuration allocate and run?
# --------------------------------------------------------------------------
# Runnability is decided by DECODER POOL MEMORY, nothing else.
#
# The decoder daemon reserves per stream: an input pool, a hidden/DPB pool and a
# visible/output pool, each sized from the frame geometry with its own alignment.
# The expression below reconstructs its own accounting (the daemon logs
# "[decoder-admission] ... input=8x... hidden=20x... request=..." to syslog) and
# reproduces its logged totals exactly: 720p=61562880, 1080p=105191424,
# 2160p=425779200 bytes.
#
# Frame rate is deliberately absent. Three independent reasons: the daemon's
# request expression has no rate term; the app leaves dec_fps=-1 so the true rate
# is never even transmitted (every stream is admitted as "@30"); and empirically
# 720p@500fps (461 MP/s) allocated fine while 2x2160p@25 (415 MP/s) failed - a
# rate model is not merely imprecise, it is non-monotone. Frame rate costs
# throughput, and degraded FPS is a success here, so it must never block or warn.
# Decoder pool bytes are the *metric*, but they are not the whole cost: this app
# also runs one hardware ENCODER per stream, and at high stream counts the encoder
# is what actually runs out first. Measured directly at 9x720p: all nine decoder
# pools fit (554 MB) yet the ninth stream died with
#   element='n4_encoder_76' error='Allocate output buffers failed'
#   cma: __cma_alloc: alloc failed, req-size: 360 pages, ret: -16
# (360 pages = 1,474,560 B = exactly one 720p NV12 frame).
#
# So the budget is calibrated as a decoder-byte PROXY for total pool pressure,
# set between the largest configuration that ran (531 MB: 1x4K + 1x1080p) and the
# smallest that failed (554 MB: 9x720p). It reproduces every measured boundary:
#   pass: 1x4K(426) 8x720p(493) 5x1080p(526) 1x4K+1x1080p(531)
#   fail: 9x720p(554) 6x1080p(631) 2x4K(852)
#
# Note ret:-16 is EBUSY (CMA fragmentation), not ENOMEM - so the true ceiling can
# drift with fragmentation and after an unclean shutdown. Treat it as a good
# predictor, not a hard physical constant. Dropping the per-stream re-encode
# (encoded passthrough) would remove the encoder pools and raise this materially.
# Scale pipeline: encoded passthrough, NO per-stream encoder, so the binding
# constraint is the decoder's own ~1.8 GB CMA budget (CmaTotal 1830912 kB). The
# high-density example fits 16x720p in it with num-buffers=18, which is what this
# pipeline now uses. Headroom kept below the full 1.8 GB for fragmentation.
POOL_BUDGET_BYTES = 1_780_000_000
MAX_DECODE_W, MAX_DECODE_H = 3840, 2160  # policy/DecoderPolicy.h:29-32

# Memory (above) decides what ALLOCATES; throughput (here) decides what actually
# DETECTS. Measured 2026-07-24, @25fps, per-stream detection fps read from the
# app's own [profile] log after 35 s of steady state:
#   HOLD  (every stream a rock-solid 25.3 fps, jitter <=0.2):
#         16x720p = 368 MP/s   8x1080p = 414 MP/s   2x4K = 414 MP/s
#   COLLAPSE (detection drops to ZERO on every stream - video still passes):
#         32x720p = 736 MP/s  16x1080p = 830 MP/s   4x4K = 830 MP/s
# All three collapses booted and allocated fine, then delivered no boxes. The
# binding limit is the decode-leg MEGAPIXEL RATE (the ~442 MP/s decoder ceiling),
# not pool bytes and not stream count alone. Past it the shared detector is
# oversubscribed and stalls to nothing - the opposite of graceful degradation, so
# it MUST be blocked: a config that delivers no boxes is not "degraded", it is
# broken. Ceiling set at 450 (just above the 414 that held, well under 736).
# A second, independent cap: the MLA sustains ~404 detections/s total, i.e.
# 16 streams at 25 fps - beyond ~16 streams the detector oversubscribes the same
# way regardless of resolution. Both gates apply only to 2+ streams: a SINGLE
# stream over the ceiling merely runs slow (720p@500 gave 130 det/s), which is
# the allowed "degraded fps is a success" case and must never be blocked.
#
# The per-pipeline stream cap is pipeline.MAX_STREAMS (scale=16, the MLA/decoder
# ceiling above; live=6, where the per-stream-graph app's metadata forwarding
# breaks). The MP/s ceiling is the same hardware limit for both.
DECODE_MPS_CEILING = 450.0
MAX_DET_STREAMS = pipeline.GROUP_SIZE * pipeline.MAX_GROUPS

# Per-pipeline switch (see the live copy): live runs without a stream-count
# cap by request; scale and group keep theirs. Defined here too so the guard
# body stays identical across all three deployments.
ENFORCE_STREAM_CAP = pipeline.PIPELINE != "live"

# Master switch for the whole capacity guard, OFF by request (2026-07-29): no
# add is ever blocked and no alert is raised, whatever the decode load, stream
# count or pool maths say. Everything the checks encode is measured and still
# documented above - setting this back to True restores all of them unchanged,
# which is why the code is switched rather than deleted.
#
# What running unguarded means, so it is a choice and not a surprise: past the
# limits the failure is not a graceful slowdown. The shared detector stalls to
# ZERO boxes on every stream while video keeps flowing (looks like it works, no
# error anywhere), and on the live pipeline an exhausted MLA pool kills the
# whole process, taking already-running streams with it.
ENFORCE_LIMITS = False
# Filename prefix -> label, for grouping the picker only. Adding a resolution
# here changes how clips are FILED, never how they are measured: geometry comes
# from Insight itself (see video_spec), so an unrecognised name still works and
# simply lands under "other".
_LIB_GROUPS = [("2160p_", "2160p"), ("1440p_", "1440p"), ("1080p_", "1080p"),
               ("480p_", "480p"), ("video", "720p")]
_SPEC_CACHE: dict[str, tuple[int, int, float]] = {}


def _parse_rate(value) -> float:
    """Insight reports frame_rate as '25/1'; fall back to the source contract."""
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            return float(num) / (float(den) or 1.0)
        return float(value)
    except (TypeError, ValueError):
        return float(pipeline.FPS)


def video_spec(name: str) -> tuple[int, int, float]:
    """(w, h, fps) of an Insight clip, measured - never guessed from the name.

    This used to shortcut known filename prefixes to hardcoded geometry, which
    made the pipeline silently wrong for anything outside that ladder and
    pinned the frame rate to an assumption. Asking Insight makes every
    resolution work with no table to maintain (2K and 640x480 included), and
    keeps the capacity guard honest - a wrong fps here is what let a config
    through that then collapsed to zero detections. Cached per file, so the
    round-trip is paid once.
    """
    if name in _SPEC_CACHE:
        return _SPEC_CACHE[name]
    info = pipeline.api("/api/media-info", {"path": name}, timeout=15) or {}
    spec = (info.get("width") or 1920, info.get("height") or 1080,
            _parse_rate(info.get("frame_rate")))
    _SPEC_CACHE[name] = spec
    return spec


def stream_spec(s: dict, total: int) -> tuple[int, int, float]:
    """Source (decoded) width, height and frame rate of one stream."""
    if s["kind"] == "auto":
        t = pipeline.tier_for(total)
        return (t.width, t.height, float(pipeline.FPS))
    if s["kind"] == "pinned":
        return video_spec(s["video"])
    return (s.get("w", 1920), s.get("h", 1080), float(s.get("fps", pipeline.FPS)))


def _align(x: int, n: int) -> int:
    return -(-x // n) * n


def stream_pool_bytes(w: int, h: int) -> int:
    """Decoder input + hidden(DPB) + visible(output) pool reservation for one stream."""
    n_hidden = 24 if h <= 720 else 20
    n_visible = 18  # scale app runs decoder num_buffers=18 (smooth, matches high-density)
    inp = 2 * (w * h * 3 // 4)   # decoder_input_buffers=2
    hidden = n_hidden * (_align(w, 64) * _align(h, 64) * 3 // 2)
    visible = n_visible * (_align(w, 256) * _align(h, 64) * 3 // 2)
    return inp + hidden + visible


def budget_report(streams: list[dict]) -> dict:
    """Summed decoder pool bytes AND decode megapixel-rate vs their ceilings.

    Memory decides allocation; MP/s decides whether the shared detector keeps up.
    Both are reported so the UI can show whichever is the binding headroom.
    """
    total = len(streams)
    specs = [stream_spec(s, total) for s in streams]
    used = sum(stream_pool_bytes(w, h) for w, h, _f in specs)
    mps = sum(w * h * f for w, h, f in specs) / 1e6
    mem_ok = used <= POOL_BUDGET_BYTES
    # Single-stream throughput never blocks (degraded fps is a success); the
    # collapse effect is a 2+-stream aggregate. So the MP/s term only "fails" the
    # report when there is more than one stream.
    thr_ok = total < 2 or mps <= DECODE_MPS_CEILING
    return {"used_mb": round(used / 1e6, 1),
            "budget_mb": round(POOL_BUDGET_BYTES / 1e6, 1),
            "mps": round(mps, 1), "mps_ceiling": DECODE_MPS_CEILING,
            "count": total, "ok": mem_ok and thr_ok}


def blocked_if_over(prospective: list[dict]):
    """A blocked-response dict if the set genuinely cannot run, else None.

    Blocks a config only when it would fail to DELIVER DETECTIONS, on any of:
      1. a single stream past the decoder's resolution ceiling;
      2. more than MAX_DET_STREAMS streams (the shared MLA oversubscribes);
      3. combined decode megapixel-rate past DECODE_MPS_CEILING (the detector
         stalls to zero - measured, not modelled);
      4. not enough decoder pool memory to allocate.
    (2) and (3) never fire on a single stream - one stream that can't keep up
    just runs at a lower fps, which is a success here, never a block or warning.
    """
    if not ENFORCE_LIMITS:
        return None
    total = len(prospective)
    specs = [stream_spec(s, total) for s in prospective]
    for i, (w, h, _f) in enumerate(specs):
        if w > MAX_DECODE_W or h > MAX_DECODE_H:
            return {"blocked": True, "budget": budget_report(prospective),
                    "error": (f"Stream {i + 1} is {w}x{h}, beyond the decoder's "
                              f"{MAX_DECODE_W}x{MAX_DECODE_H} ceiling. Nothing was changed.")}

    rep = budget_report(prospective)
    if ENFORCE_STREAM_CAP and total > MAX_DET_STREAMS:
        return {"blocked": True, "budget": rep,
                "error": (f"{total} streams exceeds this pipeline's {MAX_DET_STREAMS}-stream "
                          f"ceiling; past it streams stop delivering boxes. Nothing was changed - "
                          f"use up to {MAX_DET_STREAMS} streams.")}
    if total >= 2 and rep["mps"] > DECODE_MPS_CEILING:
        return {"blocked": True, "budget": rep,
                "error": (f"Combined decode load is {rep['mps']} MP/s, over the "
                          f"~{DECODE_MPS_CEILING:.0f} MP/s the decoder sustains; past it the "
                          f"detector stalls and delivers no boxes. Nothing was changed - use "
                          f"fewer streams or lower resolutions so the total stays under "
                          f"~{DECODE_MPS_CEILING:.0f} MP/s (e.g. 16x720p, 8x1080p or 2x4K on scale).")}
    if rep["used_mb"] > rep["budget_mb"]:
        return {"blocked": True, "budget": rep,
                "error": (f"Not enough decoder memory: {total} stream(s) need "
                          f"{rep['used_mb']} MB of pool, but only {rep['budget_mb']} MB is "
                          f"available. Nothing was changed - use fewer streams or lower "
                          f"resolutions. (Frame rate is not a factor.)")}
    return None


# --------------------------------------------------------------------------
# background job runner (one pipeline op at a time)
# --------------------------------------------------------------------------

_busy = threading.Lock()
STATUS = {"busy": False, "message": "idle", "error": None}


def submit(fn, desc: str) -> bool:
    if not _busy.acquire(blocking=False):
        return False
    STATUS.update(busy=True, message=desc, error=None)

    def worker():
        try:
            fn()
            STATUS.update(message=f"done: {desc}")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            STATUS.update(error=str(exc), message=f"failed: {desc}")
            traceback.print_exc()
        finally:
            STATUS["busy"] = False
            _busy.release()

    threading.Thread(target=worker, daemon=True).start()
    return True


# --------------------------------------------------------------------------
# live metrics sampler (per-channel fps + bitrate, refreshed every few seconds)
# --------------------------------------------------------------------------

METRICS: dict[int, dict] = {}
VIEWERS = {"count": 0}  # max browser peers on any channel; >1 breaks metadata overlays


def sample_viewers():
    """How many browser viewers are connected (max peers across active channels)."""
    try:
        eg = pipeline.api("/api/egress/stats", timeout=10)
        chs = eg.get("channels", eg) if isinstance(eg, dict) else eg
        if isinstance(chs, dict):
            chs = list(chs.values())
        VIEWERS["count"] = max(
            (len(c.get("peers") or []) for c in chs
             if isinstance(c, dict) and (c.get("peers"))), default=0)
    except Exception:  # noqa: BLE001 - best effort
        pass


def metrics_loop():
    prev: dict[int, tuple[int, float]] = {}
    while True:
        try:
            sample_viewers()
            now = time.time()
            cur = {}
            for c in pipeline.ingest():
                ch = c.get("channel")
                if ch is None:
                    continue
                meta = (c.get("metadata") or {}).get("messages_received") or 0
                kbps = round((c.get("rtp", {}).get("bitrate_bps") or 0) / 1000)
                fps = 0.0
                if ch in prev:
                    pm, pt = prev[ch]
                    dt = now - pt
                    if dt > 0:
                        fps = round((meta - pm) / dt, 1)
                cur[ch] = {"fps": fps, "kbps": kbps,
                           "active": bool((c.get("rtp", {}).get("packets_received") or 0))}
                prev[ch] = (meta, now)
            METRICS.clear()
            METRICS.update(cur)
        except Exception:  # noqa: BLE001 - metrics are best-effort
            pass
        time.sleep(5)


# --------------------------------------------------------------------------
# state assembly for the UI
# --------------------------------------------------------------------------


_VIDEO_CACHE: dict = {"at": 0.0, "groups": None}
_EMPTY_LIB = {label: [] for _p, label in _LIB_GROUPS} | {"other": []}


def video_library() -> dict[str, list[str]]:
    """Insight videos grouped for the picker - cached, short timeout.

    /api/state is polled every 2.5s and this is the only part of it that calls
    out to Insight. With the API default of 60s, an Insight that stops
    answering made EVERY poll block for a minute, which reads as "the whole UI
    is frozen" when in reality only the video list was unavailable (observed
    2026-07-28: Insight hung, and all three pipelines' UIs appeared dead).
    Caching the list and failing fast keeps the rest of the panel - stream
    table, fps, remove buttons - working while Insight is unwell.
    """
    now = time.time()
    if _VIDEO_CACHE["groups"] is not None and now - _VIDEO_CACHE["at"] < 30:
        return _VIDEO_CACHE["groups"]
    try:
        all_files = sorted(pipeline.api("/api/mediasrc/videos", timeout=5) or [])
    except Exception:  # noqa: BLE001 - a slow Insight must not stall the panel
        return _VIDEO_CACHE["groups"] or dict(_EMPTY_LIB)
    groups = {label: [] for _p, label in _LIB_GROUPS}
    groups["other"] = []
    for f in all_files:
        label = next((lab for pre, lab in _LIB_GROUPS if f.startswith(pre)), "other")
        groups[label].append(f)
    _VIDEO_CACHE.update(at=now, groups=groups)
    return groups


def full_state() -> dict:
    groups = load_groups()
    streams = flat(groups)
    total = len(streams)

    rows = []
    running_any = False
    for g, members in enumerate(groups):
        delivered = pipeline.delivered_group(g)     # pos -> "WxH", per group
        g_running = pipeline.group_running(g)
        running_any = running_any or g_running
        for pos, s in enumerate(members):
            ch = pipeline.channel_for(g, pos)
            source = s["url"] if s["kind"] == "external" else s.get("video", "auto")
            w, h, sfps = stream_spec(s, total)
            m = METRICS.get(ch, {})
            rows.append({
                "cam": f"cam-{ch + 1}", "channel": ch,
                "group": g, "pos": pos, "group_running": g_running,
                "kind": s["kind"], "source": source,
                "source_res": f"{w}x{h}",
                "source_fps": round(sfps, 1),
                "delivered": delivered.get(pos, "-"),
                "fps": m.get("fps", 0.0), "kbps": m.get("kbps", 0),
                "active": m.get("active", False),
            })

    return {
        "busy": STATUS["busy"], "message": STATUS["message"], "error": STATUS["error"],
        "app_running": running_any,
        "count": total,
        "tier": "-",
        "streams": rows,
        "budget": budget_report(streams),
        "viewers": VIEWERS["count"],
        "viewer_url": INSIGHT_UI,
        "videos": video_library(),
        "ceilings": {t.name: t.max_streams for t in pipeline.TIERS},
        "pipeline": pipeline.PIPELINE,
        "max_streams": MAX_DET_STREAMS,
        "live_add": False,
        "group_size": GROUP_SIZE,
        "max_groups": MAX_GROUPS,
        "groups": [{"group": g, "count": len(m), "running": pipeline.group_running(g)}
                   for g, m in enumerate(groups)],
    }


# --------------------------------------------------------------------------
# shared static assets (SiMa logo + the fonts Neat Insight itself uses)
# --------------------------------------------------------------------------
# Kept in one place beside the pipelines rather than duplicated per web dir,
# and served from here because these pages come from THIS server, not from
# Insight - a cross-origin link to Insight's copies would break on every IP
# change and trip its self-signed certificate.
ASSETS = HERE.parent / "web-assets"
_MIME = {".png": "image/png", ".woff2": "font/woff2", ".svg": "image/svg+xml",
         ".ico": "image/x-icon"}


def serve_asset(handler, name: str) -> bool:
    """Write ASSETS/name to the client. False if it is missing or unsafe."""
    # Look the request up in an index built by enumerating web-assets, so the
    # path handed to the filesystem is one WE produced rather than one derived
    # from the request. The request string is only ever a dictionary key, which
    # leaves no traversal to defend against: a name carrying "/" or ".." simply
    # is not a key. Rebuilt per request so a new asset needs no restart.
    try:
        index = {entry.name: entry for entry in ASSETS.iterdir() if entry.is_file()}
    except OSError:
        return False
    path = index.get(name)
    if path is None:
        return False
    body = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", _MIME.get(path.suffix, "application/octet-stream"))
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.end_headers()
    handler.wfile.write(body)
    return True


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------


def parse_multipart_file(body: bytes, content_type: str):
    """Extract (filename, bytes) from a single-file multipart/form-data body."""
    if "boundary=" not in content_type:
        return None, None
    boundary = content_type.split("boundary=", 1)[1].strip().encode()
    for part in body.split(b"--" + boundary):
        if b"filename=" in part and b"\r\n\r\n" in part:
            head, data = part.split(b"\r\n\r\n", 1)
            fname = head.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
            if not fname:
                continue
            return fname, data.rsplit(b"\r\n", 1)[0]  # strip trailing CRLF
    return None, None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ---- GET ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not WEB.exists():
                self._json({"error": "web/index.html missing"}, 500)
                return
            body = WEB.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/assets/"):
            if not serve_asset(self, path[len("/assets/"):]):
                self._json({"error": "not found"}, 404)
        elif path == "/api/state":
            self._json(full_state())
        elif path == "/api/logs":
            # Truncate each line: the app prints whole GStreamer pipelines on one
            # line (thousands of chars), which bloats the payload and is unreadable
            # in a status panel.
            out = pipeline.exec_devkit(
                f"tail -n 40 {pipeline.LOG} 2>/dev/null | cut -c1-200")
            self._json({"log": out.stdout})
        else:
            self._json({"error": "not found"}, 404)

    # ---- POST ----
    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/upload":
            ctype = self.headers.get("Content-Type", "")
            n = int(self.headers.get("Content-Length", 0))
            fname, data = parse_multipart_file(self.rfile.read(n), ctype)
            if not fname:
                self._json({"error": "no file in upload"}, 400)
                return
            # Reduce the client-supplied name to a bare basename before it touches
            # a filesystem path - an absolute path or ../ components in `fname`
            # would otherwise let this unauthenticated (0.0.0.0) endpoint overwrite
            # then delete any file the service account can write. The bytes land in
            # a private tempfile; the sanitised name only travels as multipart
            # metadata to Insight.
            safe_name = Path(fname).name
            if not safe_name or safe_name in (".", ".."):
                self._json({"error": "invalid filename"}, 400)
                return
            with tempfile.NamedTemporaryFile(dir="/tmp", prefix="upload-") as tmp:
                tmp.write(data)
                tmp.flush()
                r = subprocess.run(
                    ["curl", "-sk", "--max-time", "600", "-F",
                     f"file=@{tmp.name};filename={safe_name}",
                     f"{pipeline.INSIGHT_API}/api/upload/media"],
                    capture_output=True, text=True)
            self._json({"ok": "complete" in r.stdout.lower(), "detail": r.stdout.strip()[:200]})
            return

        body = self._body()

        # Stop bypasses the job queue so it always works, even mid-operation.
        if path == "/api/down":
            def do_down():
                # See the note in the scale/live copy: clearing our own saved
                # state must not depend on Insight being reachable.
                err = None
                try:
                    pipeline.stop_all_groups()
                except Exception as exc:      # noqa: BLE001
                    err = f"stopping the detectors failed: {exc}"
                try:
                    pipeline.api("/api/mediasrc/stop-all", {})
                except Exception as exc:      # noqa: BLE001
                    err = err or f"Insight sources may still be playing: {exc}"
                save_groups([])
                STATUS.update(busy=False, message="pipeline stopped", error=err)
            threading.Thread(target=do_down, daemon=True).start()
            self._json({"accepted": True})
            return

        # Build the prospective stream set and the action to run for it.
        current = flat(load_groups())
        if path == "/api/add_videos":
            vids = body.get("videos", [])
            if not vids:
                self._json({"error": "no videos selected"}, 400); return
            news = [{"kind": "pinned", "video": v} for v in vids]
            prospective = current + news
            action = (lambda: add_many(news), f"adding {len(vids)} video(s)")
        elif path == "/api/add_rtsps":
            # Resolution and frame rate are PROBED from each camera, never
            # typed. The detector app probes its sources itself and decodes at
            # whatever it actually finds, so a typed value never affected the
            # output - it only fed the capacity guard, where a wrong guess
            # silently mis-counted the load (calling a 4K@60 camera "720p" let
            # through a config that then collapsed to zero detections).
            urls_in = [str(u).strip() for u in body.get("urls", []) if str(u).strip()]
            if not urls_in:
                self._json({"error": "no rtsp:// urls given"}, 400); return
            bad = next((u for u in urls_in if not u.startswith("rtsp://")), None)
            if bad:
                self._json({"error": f"not an rtsp:// url: {bad}"}, 400); return
            with ThreadPoolExecutor(max_workers=min(8, len(urls_in))) as pool:
                probed = list(pool.map(pipeline.probe_source, urls_in))
            unreachable = [u for u, p in zip(urls_in, probed) if p is None]
            if unreachable:
                self._json({"error": "could not read video from: " + ", ".join(unreachable)
                                     + " - check the camera is reachable and streaming"}, 400)
                return
            news = [{"kind": "external", "url": u, "w": p[0], "h": p[1], "fps": p[2]}
                    for u, p in zip(urls_in, probed)]
            prospective = current + news
            action = (lambda: add_many(news), f"adding {len(news)} camera(s)")
        elif path in ("/api/remove_stream", "/api/remove_streams"):
            # The UI identifies a stream by its Insight channel, which encodes
            # its position exactly: channel = group*GROUP_SIZE + pos. Only the
            # groups owning these channels get rebuilt; the rest keep running.
            raw = ([body.get("index", -1)] if path == "/api/remove_stream"
                   else body.get("indices", []))
            chans = sorted({int(i) for i in raw})
            groups_now = load_groups()
            targets: list[tuple[int, int]] = []
            for ch in chans:
                g, pos = divmod(ch, GROUP_SIZE)
                if not (0 <= g < len(groups_now) and 0 <= pos < len(groups_now[g])):
                    self._json({"error": f"invalid stream channel {ch}"}, 400); return
                targets.append((g, pos))
            if not targets:
                self._json({"error": "no streams selected"}, 400); return
            drop = set(targets)
            prospective = [s for g, members in enumerate(groups_now)
                           for pos, s in enumerate(members) if (g, pos) not in drop]
            touched = sorted({g for g, _ in targets})
            action = (lambda: remove_positions(targets),
                      f"removing {len(targets)} stream(s) from group(s) "
                      f"{', '.join(map(str, touched))}")
        else:
            self._json({"error": "not found"}, 404); return

        # Budget guard - removing is always allowed; everything else must fit.
        if path not in ("/api/remove_stream", "/api/remove_streams"):
            blocked = blocked_if_over(prospective)
            if blocked:
                self._json(blocked)
                return

        ok = submit(*action)
        self._json({"accepted": ok, "busy": not ok})


def main():
    WEB.parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=metrics_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"adaptive pipeline UI on http://0.0.0.0:{PORT}  (INSIGHT_API={pipeline.INSIGHT_API})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
