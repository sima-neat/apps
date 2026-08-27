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
import os
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

PORT = 8091 if pipeline.PIPELINE == 'live' else 8090
WEB = HERE / "web" / "index.html"
STATE_FILE = HERE / "ui-state.json"
INSIGHT_UI = "https://192.168.131.68:9900/"   # main Neat Insight UI


# --------------------------------------------------------------------------
# persisted stream model
# --------------------------------------------------------------------------


def load_streams() -> list[dict]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text()).get("streams", [])
        except json.JSONDecodeError:
            return []
    return []


def save_streams(streams: list[dict]) -> None:
    STATE_FILE.write_text(json.dumps({"streams": streams}, indent=2))


def plan(streams: list[dict]) -> tuple[list[str], list[tuple[int, str]], pipeline.Tier]:
    """Resolve a stream list into (config URLs, Insight staging, managed tier).

    Managed (`auto`) streams take tier-appropriate clips for the *total* count;
    `pinned` streams keep their chosen clip; `external` streams pass their URL
    straight through with no Insight slot.
    """
    total = len(streams)
    tier = pipeline.tier_for(total)
    tier_files = pipeline.media_files(tier.prefix)
    urls: list[str] = []
    staging: list[tuple[int, str]] = []
    slot = 0
    auto_i = 0
    for s in streams:
        if s["kind"] == "external":
            urls.append(s["url"])
            continue
        slot += 1
        if s["kind"] == "pinned":
            video = s["video"]
        else:  # auto
            if not tier_files:
                raise RuntimeError(f"no '{tier.prefix}*' media in Insight for tier {tier.name}")
            video = tier_files[auto_i % len(tier_files)]
            auto_i += 1
        staging.append((slot, video))
        urls.append(pipeline.insight_url(slot))
    return urls, staging, tier


def stage_insight(staging: list[tuple[int, str]]) -> None:
    pipeline.api("/api/mediasrc/stop-all", {})
    for slot, video in staging:
        pipeline.api("/api/mediasrc/assign", {"index": slot, "file": video})
    for slot, _ in staging:
        pipeline.api("/api/mediasrc/start", {"index": slot})
    if staging:
        time.sleep(3)  # let RTSP mounts come up before the app connects


def rebuild(streams: list[dict]) -> None:
    """Full apply: stage all Insight sources, write config, restart the app."""
    urls, staging, tier = plan(streams)
    stage_insight(staging)
    pipeline.write_config_urls(urls, header=f"{len(streams)} stream(s), managed tier {tier.name}")
    if streams:
        pipeline.start_app()
        pipeline.wait_for_streams(len(streams), timeout_s=300)
    else:
        pipeline.stop_app()
    save_streams(streams)


def live_or_rebuild_add(new: dict) -> str:
    """Add one stream live (no restart) where possible. See add_many()."""
    return live_or_rebuild_add_many([new])


def live_or_rebuild_add_many(news: list[dict]) -> str:
    """Add one OR MORE streams. Live (no restart) when the managed tier does not
    change and the app is already up; otherwise one full rebuild for the batch.

    Batch-capable on purpose: the whole point of the live pipeline is that adding
    a stream does not disturb the ones already running, and that has to hold when
    several are added at once too. Only the newcomers are staged and appended -
    existing streams are never touched.
    """
    streams = load_streams()
    old_total = len(streams)
    combined = streams + news

    # Only the per-stream "live" mode can add without a full restart - its app
    # watches the config file. The fused "scale" app has no config watch, so an
    # append would never be picked up; it must rebuild.
    if not pipeline.LIVE_ADD:
        rebuild(combined)
        return "rebuild"

    # The managed tier only decides what resolution `auto` streams are STAGED at,
    # so a tier change is only a reason to restart when there is an auto stream
    # whose source would actually have to change. `pinned` and `external` streams
    # hold their native resolution at any count, so with none of them auto (the
    # UI only creates pinned/external now) a count-boundary crossing must not
    # cost a restart - that would defeat the whole point of the live pipeline.
    has_auto = any(s["kind"] == "auto" for s in combined)
    tier_unchanged = (not has_auto) or (
        pipeline.tier_for(len(combined)).name == pipeline.tier_for(old_total).name)
    if old_total and news and pipeline.app_running() and tier_unchanged:
        # plan() stages every non-external stream in list order, so the newcomers
        # are exactly the last N entries; the existing ones are left alone.
        _, staging, _ = plan(combined)
        n_new_slots = sum(1 for s in news if s["kind"] != "external")
        new_slots = staging[-n_new_slots:] if n_new_slots else []
        urls: list[str] = []
        slot_i = 0
        for s in news:
            if s["kind"] == "external":
                urls.append(s["url"])
                continue
            slot, video = new_slots[slot_i]
            slot_i += 1
            pipeline.api("/api/mediasrc/assign", {"index": slot, "file": video})
            pipeline.api("/api/mediasrc/start", {"index": slot})
            urls.append(pipeline.insight_url(slot))
        if n_new_slots:
            time.sleep(5)  # let the new RTSP mounts come up before config-watch builds
        for url in urls:
            pipeline.append_source(url)
        save_streams(combined)
        pipeline.wait_for_streams(len(combined), timeout_s=180)
        return "live"

    rebuild(combined)
    return "rebuild"


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
MAX_DET_STREAMS = pipeline.MAX_STREAMS

# The live pipeline does NOT enforce a stream-count cap: requested 2026-07-28.
# Kept as a per-pipeline switch rather than deleting the check, so scale and
# group keep theirs and this file stays identical across all three deployments.
# Note what this gives up: live allocates a model + MLA output pool + encoder
# PER STREAM, and the shared MLA pool has been seen to run out (measured
# 4x1080p + 2x720p -> "MLA_0_s5... Unable to allocate output memory"), which
# kills the whole process and takes every already-running stream down with it.
# Past the cap that failure is now unguarded - the MP/s and memory checks below
# still apply, but they do not model per-stream MLA pool exhaustion.
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
    n_visible = pipeline.DECODER_BUFFERS  # the decoder pool size this pipeline actually runs
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
      2. more than MAX_DET_STREAMS streams (the shared MLA oversubscribes) -
         skipped entirely when ENFORCE_STREAM_CAP is false, as on live;
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

# Bumped by every Stop. Stop deliberately bypasses _busy so it stays responsive
# during a long rebuild, but that let it race the operation it interrupted: the
# worker would resume after Stop finished, start the detector again and re-save
# its stale stream list, leaving "pipeline stopped" on screen over a pipeline
# that had come back up. A worker that finishes holding a stale token knows a
# Stop landed while it ran, and re-asserts the stopped state so Stop wins.
_stop_token = 0
_stop_token_lock = threading.Lock()


def current_stop_token() -> int:
    with _stop_token_lock:
        return _stop_token


def begin_stop() -> None:
    """Invalidate any operation currently in flight."""
    global _stop_token
    with _stop_token_lock:
        _stop_token += 1


def submit(fn, desc: str) -> bool:
    if not _busy.acquire(blocking=False):
        return False
    token = current_stop_token()
    STATUS.update(busy=True, message=desc, error=None)

    def worker():
        try:
            fn()
            STATUS.update(message=f"done: {desc}")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            STATUS.update(error=str(exc), message=f"failed: {desc}")
            traceback.print_exc()
        finally:
            if current_stop_token() != token:
                # A Stop landed while this ran - including one that finished
                # before we did. Undo whatever we brought back up.
                reassert_stopped()
                STATUS.update(message="pipeline stopped", error=None)
            STATUS["busy"] = False
            _busy.release()

    threading.Thread(target=worker, daemon=True).start()
    return True

def reassert_stopped() -> None:
    """Re-apply Stop after an operation that raced it. Best effort by design:
    this runs on the worker's way out and must not raise into it."""
    try:
        pipeline.stop_app()
    except Exception:  # noqa: BLE001
        pass
    try:
        save_streams([])
    except Exception:  # noqa: BLE001
        pass



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
    streams = load_streams()
    delivered = pipeline.delivered()  # {cam-id: "WxH"} from the app log
    tier = pipeline.tier_for(len(streams)) if streams else None

    rows = []
    total = len(streams)
    for i, s in enumerate(streams):
        cam = f"cam-{i + 1}"
        ch = i  # channel index follows stream order
        source = s["url"] if s["kind"] == "external" else s.get("video", "auto")
        w, h, sfps = stream_spec(s, total)
        m = METRICS.get(ch, {})
        rows.append({
            "cam": cam, "channel": ch, "kind": s["kind"], "source": source,
            "source_res": f"{w}x{h}",
            "source_fps": round(sfps, 1),
            "delivered": delivered.get(cam, "-"),
            "fps": m.get("fps", 0.0), "kbps": m.get("kbps", 0),
            "active": m.get("active", False),
        })

    return {
        "busy": STATUS["busy"], "message": STATUS["message"], "error": STATUS["error"],
        "app_running": pipeline.app_running(),
        "count": len(streams),
        "tier": tier.name if tier else "-",
        "streams": rows,
        "budget": budget_report(streams),
        "viewers": VIEWERS["count"],
        "viewer_url": INSIGHT_UI,
        "videos": video_library(),
        "ceilings": {t.name: t.max_streams for t in pipeline.TIERS},
        "pipeline": pipeline.PIPELINE,
        "max_streams": pipeline.MAX_STREAMS,
        "live_add": pipeline.LIVE_ADD,
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


# Ceilings for this unauthenticated 0.0.0.0 server. Without them a single POST is
# read whole into DevKit memory, which the media pools have already claimed most
# of, so one oversized upload - or a few concurrent ones, since this is a
# ThreadingHTTPServer - can push the board into OOM before a byte reaches disk.
MAX_UPLOAD_BYTES = int(os.environ.get("UI_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))
MAX_JSON_BYTES = int(os.environ.get("UI_MAX_JSON_BYTES", 1024 * 1024))
MAX_PART_HEADER_BYTES = 64 * 1024
READ_CHUNK_BYTES = 1024 * 1024


def stream_multipart_file(rfile, length: int, content_type: str, dest) -> str | None:
    """Copy a single-file multipart/form-data body to `dest`, returning its filename.

    The body is consumed in chunks and the file bytes go straight to `dest`, so
    peak memory is the chunk size rather than the payload size. Reading the whole
    request first - and splitting it, which copies it again - is what made a large
    upload a memory event on the DevKit. Returns None - having possibly written
    part of the body to `dest`, which the caller discards - when the body carries
    no named file part or never reaches its closing boundary.
    """
    if "boundary=" not in content_type:
        return None
    # The boundary parameter may be quoted and may be followed by further
    # parameters (RFC 2045). Taking the rest of the header verbatim leaves the
    # quotes in, and a boundary that never matches means the closing marker is
    # never found - which used to look like a successful upload.
    raw_boundary = content_type.split("boundary=", 1)[1].split(";", 1)[0].strip()
    if len(raw_boundary) >= 2 and raw_boundary[0] == '"' and raw_boundary[-1] == '"':
        raw_boundary = raw_boundary[1:-1]
    if not raw_boundary:
        return None
    boundary = b"--" + raw_boundary.encode()
    remaining = length
    buf = b""

    # The part headers are small: read until the blank line that ends them.
    while remaining > 0 and b"\r\n\r\n" not in buf:
        chunk = rfile.read(min(remaining, READ_CHUNK_BYTES))
        if not chunk:
            break
        remaining -= len(chunk)
        buf += chunk
        # Only a header that never terminates is oversized - a chunk that ran on
        # past the blank line is carrying file bytes, not headers.
        if b"\r\n\r\n" not in buf and len(buf) > MAX_PART_HEADER_BYTES:
            return None
    if b"\r\n\r\n" not in buf:
        return None
    head, buf = buf.split(b"\r\n\r\n", 1)
    if b"filename=" not in head:
        return None
    fname = head.split(b'filename="', 1)[1].split(b'"', 1)[0].decode(errors="replace")
    if not fname:
        return None

    # Then the file bytes. Hold back enough tail to recognise the closing boundary
    # even when it straddles two chunks, and write everything before it.
    tail = len(boundary) + 8
    while True:
        cut = buf.find(b"\r\n" + boundary)
        if cut != -1:
            dest.write(buf[:cut])
            return fname
        if len(buf) > tail:
            dest.write(buf[:-tail])
            buf = buf[-tail:]
        if remaining <= 0:
            break
        chunk = rfile.read(min(remaining, READ_CHUNK_BYTES))
        if not chunk:
            break
        remaining -= len(chunk)
        buf += chunk
    # No closing boundary. The body was truncated, or its boundary parameter did
    # not match the one we parsed - either way what reached `dest` is a partial
    # file, or a whole one with the multipart trailer stuck to the end. Refuse it
    # rather than forward it to Insight and report the upload as complete.
    return None


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
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return {}
        if n <= 0:
            return {}
        if n > MAX_JSON_BYTES:
            # Treated as an unusable body, the same as malformed JSON below. No
            # control endpoint here takes a payload anywhere near this size.
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
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = -1
            if n < 0:
                self._json({"error": "bad Content-Length"}, 400)
                return
            if n > MAX_UPLOAD_BYTES:
                # Refused on the header, before the body is read at all.
                self.close_connection = True
                self._json({"error": f"upload exceeds {MAX_UPLOAD_BYTES} bytes"}, 413)
                return
            with tempfile.NamedTemporaryFile(dir="/tmp", prefix="upload-") as tmp:
                fname = stream_multipart_file(self.rfile, n, ctype, tmp)
                if not fname:
                    self._json({"error": "no file in upload"}, 400)
                    return
                # Reduce the client-supplied name to a bare basename before it
                # touches a filesystem path - an absolute path or ../ components in
                # `fname` would otherwise let this unauthenticated (0.0.0.0)
                # endpoint overwrite then delete any file the service account can
                # write. The bytes land in a private tempfile; the sanitised name
                # only travels as multipart metadata to Insight.
                safe_name = Path(fname).name
                if not safe_name or safe_name in (".", ".."):
                    self._json({"error": "invalid filename"}, 400)
                    return
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
            # Invalidate any in-flight operation BEFORE doing anything, so a
            # worker that finishes mid-Stop sees a stale token and undoes itself.
            begin_stop()

            def do_down():
                # Order and error-handling matter here. Stopping the DETECTOR and
                # clearing our own saved list must both happen even when Insight
                # is unreachable - it lives at an address that changes, and an
                # unhandled URLError used to kill this thread before the list was
                # cleared, so Stop silently did nothing and left a phantom count
                # on screen with no error anywhere.
                err = None
                try:
                    pipeline.stop_app()
                except Exception as exc:      # noqa: BLE001
                    err = f"stopping the detector failed: {exc}"
                try:
                    pipeline.api("/api/mediasrc/stop-all", {})
                except Exception as exc:      # noqa: BLE001
                    err = err or f"Insight sources may still be playing: {exc}"
                save_streams([])
                STATUS.update(busy=False, message="pipeline stopped", error=err)
            threading.Thread(target=do_down, daemon=True).start()
            self._json({"accepted": True})
            return

        # Build the prospective stream set and the action to run for it.
        current = load_streams()
        if path == "/api/add_videos":
            vids = body.get("videos", [])
            if not vids:
                self._json({"error": "no videos selected"}, 400); return
            news = [{"kind": "pinned", "video": v} for v in vids]
            prospective = current + news
            action = (lambda: live_or_rebuild_add_many(news), f"adding {len(vids)} video(s)")
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
            action = (lambda: live_or_rebuild_add_many(news), f"adding {len(news)} camera(s)")
        elif path == "/api/remove_stream":
            idx = int(body.get("index", -1))
            if not (0 <= idx < len(current)):
                self._json({"error": "invalid stream index"}, 400); return
            remaining = current[:idx] + current[idx + 1:]
            prospective = remaining
            action = (lambda: rebuild(remaining), f"removing stream cam-{idx + 1}")
        elif path == "/api/remove_streams":
            # Remove several streams in ONE rebuild, so the user waits once.
            idxs = sorted({int(i) for i in body.get("indices", [])})
            if not idxs or any(not (0 <= i < len(current)) for i in idxs):
                self._json({"error": "invalid stream indices"}, 400); return
            drop = set(idxs)
            remaining = [s for i, s in enumerate(current) if i not in drop]
            prospective = remaining
            action = (lambda: rebuild(remaining), f"removing {len(idxs)} stream(s)")
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
