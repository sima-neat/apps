#!/usr/bin/env python3
"""Adaptive-resolution detection pipeline: one control script, one knob.

Insight (in the SDK container) serves the RTSP sources and receives video and
detection metadata. The detector application runs on the Modalix DevKit and
reads its stream list from a YAML config it polls while running.

One thing adapts as the stream count changes:

  source resolution     this script swaps which Insight media file each slot
                        streams. The application decodes and delivers at the
                        source's native size, so feeding it smaller sources is
                        the only way to cut decoder and buffer memory - what
                        actually runs out first.

The tier table below is measured on this DevKit, not derived. See README.md.

Usage:
  python3 pipeline.py up 1        start with 1 stream (4K)
  python3 pipeline.py add         add one stream, restarting to rebuild
  python3 pipeline.py set 5       change to 5 streams, restart for a uniform tier
  python3 pipeline.py status      per-channel resolution, bitrate, live FPS
  python3 pipeline.py down        stop application and sources
"""

from __future__ import annotations

import argparse
import json
import shlex
import ssl
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
#
# This module drives the pipeline from two places and detects which at import:
#
#   from the SDK container  Insight is container-internal (127.0.0.1:9900) and
#                           the detector is controlled over SSH to the DevKit.
#   on the DevKit itself    Insight is reached on its mapped host port
#                           (used by ui_server.py), and the detector is a local
#                           process - no SSH hop.
#
# The discriminator is the DevKit-only pyneat venv installed by `neat update`.

_ON_DEVKIT = Path("/home/sima/pyneat/bin/python").exists()

INSIGHT_HOST = "192.168.131.68"             # Insight, as the DevKit and browser see it
RTSP_PORT = 8554
DEVKIT = "sima@192.168.135.72"

# Insight API base: the mapped host port from the DevKit, the internal port
# from inside the SDK container.
INSIGHT_API = "https://192.168.131.68:9900" if _ON_DEVKIT else "https://127.0.0.1:9900"

PYTHON = "/home/sima/pyneat/bin/python"     # stock pyneat venv installed by neat update

# ==========================================================================
# WHICH PIPELINE THIS DIRECTORY DRIVES  (the only line that differs between the
# two deployments; the live copy sets PIPELINE = "live")
# ==========================================================================
#
#   scale  ONE fused graph, one shared detector, encoded passthrough. Boxes stay
#          correct at high stream counts (the fused fan-in avoids the restamping
#          bridge that broke metadata past ~6 streams). Adding a stream restarts
#          the whole pipeline (~10-30 s). Target: 16-32 streams, mixed 4K/1080p/720p.
#
#   live   ONE graph PER stream (adaptive app), re-encode topology. A new stream
#          builds live while the others keep running (no full restart), but the
#          per-stream bridges cap reliable boxes at ~6 streams.
#
# The two are separate apps with opposite trade-offs, deployed as two independent
# pipelines (separate directories, separate UI ports). Run one at a time.
PIPELINE = "scale"

# --------------------------------------------------------------------------
# where this bundle lives
# --------------------------------------------------------------------------
#
# Every path below is derived from this file's own location, so the bundle
# works from any clone path on any machine. The layout it expects:
#
#   <apps>/examples/object-detection/adaptive-resolution-object-detector/
#       src/ tests/                     the live app + its labels
#       pipelines/                      <- this bundle
#           pipeline-{scale,live,group}/
#           launcher.py                 the chooser on :8080
#
# Both the SDK container and the DevKit must see this at the SAME absolute
# path - they do, because the DevKit NFS-mounts the container's workspace.
HERE = Path(__file__).resolve().parent          # pipeline-<mode>/
BUNDLE = HERE.parent                            # pipelines/
EXAMPLE = BUNDLE.parent                         # adaptive-resolution-object-detector/
APPS_ROOT = EXAMPLE.parents[2]                  # the apps repo root

# ==========================================================================
# WHICH DETECTOR, IN WHICH LANGUAGE
# ==========================================================================
#
# There is ONE entry point per language, and both take the same flags:
#
#   src/python/main.py                            --mode {adaptive,fused} --config X
#   src/cpp/pre-built/<binary>                    --mode {adaptive,fused} --config X
#
# The PIPELINE fixes the mode; the UI toggles the LANGUAGE. Because the flags
# are identical, flipping the toggle changes nothing else in this file.
#
#   scale / group  -> fused     one graph, one shared detector, passthrough
#   live           -> adaptive  one graph per stream, built/torn down live
_MODES = {"scale": "fused", "live": "adaptive", "group": "fused"}
APP_MODE = _MODES[PIPELINE]
MAX_STREAMS = 6 if PIPELINE == "live" else 16
LIVE_ADD = PIPELINE == "live"

PY_APP = EXAMPLE / "src" / "python" / "main.py"
# Packaged location first - that is what `sima-cli neat install apps` ships -
# then the from-source build tree.
CPP_APP_CANDIDATES = (
    EXAMPLE / "src" / "cpp" / "pre-built" / "adaptive-resolution-object-detector",
    APPS_ROOT / "build" / "examples" / "object-detection"
    / "adaptive-resolution-object-detector" / "adaptive-resolution-object-detector",
)

LANGUAGES = ("python", "cpp")
# One file, shared by all three pipelines and the UI, so the toggle is global:
# picking C++ in any panel applies everywhere. Gitignored - it is user state.
LANGUAGE_FILE = BUNDLE / "language"


def language() -> str:
    """The selected implementation language; python unless set otherwise."""
    try:
        value = LANGUAGE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "python"
    return value if value in LANGUAGES else "python"


def set_language(value: str) -> None:
    if value not in LANGUAGES:
        raise ValueError(f"language must be one of {LANGUAGES}")
    LANGUAGE_FILE.write_text(value + "\n", encoding="utf-8")


def cpp_binary():
    """First existing C++ binary, or None when it has not been built."""
    for candidate in CPP_APP_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def app_command() -> str:
    """How to start this pipeline's detector, minus --config."""
    if language() == "cpp":
        binary = cpp_binary()
        if binary is None:
            raise SystemExit(
                "language is 'cpp' but no binary was found. Build it with "
                "./build.sh --clean, or switch back to Python. Looked in:\n  "
                + "\n  ".join(str(c) for c in CPP_APP_CANDIDATES))
        return f"{shlex.quote(str(binary))} --mode {APP_MODE}"
    return f"{shlex.quote(PYTHON)} -u {shlex.quote(str(PY_APP))} --mode {APP_MODE}"


# pgrep pattern for THIS pipeline's detector in EITHER language. Matching the
# config filename rather than the program is what makes it language-agnostic:
# the path appears in argv for both, and is unique per pipeline. The bracket
# stops the pattern matching the pgrep command that carries it.
APP_PATTERN = ("[" + PIPELINE[0] + "]" + PIPELINE[1:]
               + ("[0-9]*" if PIPELINE == "group" else "") + "-run.yaml")

# Patterns for ALL pipelines - used to stop the others before starting this one,
# since they share one MLA and one set of Insight ports.
ALL_PIPELINE_PATTERNS = (
    "[s]cale-run.yaml",
    "[l]ive-run.yaml",
    "[g]roup[0-9]*-run.yaml",
)

# Per-pipeline files, so the two deployments never clobber each other's state.
CONFIG = HERE / f"{PIPELINE}-run.yaml"
LOG = str(BUNDLE / "logs" / f"{PIPELINE}-py.log")
# The detector is started with `> {LOG}` on the DevKit, and a redirect will
# not create the directory - make sure it exists before any run.
Path(LOG).parent.mkdir(parents=True, exist_ok=True)

# `models/` is where download_models.sh and the READMEs put packs. The older
# assets/models/ is still accepted so an existing checkout keeps working.
_MODEL_DIRS = (APPS_ROOT / "models", APPS_ROOT / "assets" / "models")
MODELS = str(next((d for d in _MODEL_DIRS if d.is_dir()), _MODEL_DIRS[0]))
LABELS = str(EXAMPLE / "src" / "common" / "coco_label.txt")

_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


# --------------------------------------------------------------------------
# measured tier table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    name: str
    width: int
    height: int
    prefix: str          # Insight media filename prefix
    max_streams: int     # highest count that stayed up on this DevKit


# Ordered best-first. max_streams is the highest count that DETECTS at full rate
# (every stream 25.3 fps, jitter <=0.2), measured 2026-07-24 with the encoded-
# passthrough topology. One more than that is where the shared detector stalls to
# zero boxes (decode leg past ~442 MP/s), so these are the usable ceilings:
#   2 x 2160p = 414 MP/s hold   ; 4 x 2160p = 830 MP/s collapse
#   8 x 1080p = 414 MP/s hold   ; 16 x 1080p = 830 MP/s collapse
#  16 x  720p = 368 MP/s hold   ; 32 x  720p = 736 MP/s collapse
# (Earlier, smaller limits were the pre-passthrough re-encode topology, which ran
# out of encoder pools far sooner; passthrough removed that and lifted them here.)
TIERS: list[Tier] = [
    Tier("2160p", 3840, 2160, "2160p_", 1),
    Tier("1080p", 1920, 1080, "1080p_", 8),
    Tier("720p", 1280, 720, "video", 16),
]

FPS = 25

# Named so pipelines/pipeline-scale/ui_server.py's memory estimate can read the
# real value instead of carrying its own guess - that drift (an estimate of 18
# against this actual 8) is what let ENFORCE_LIMITS's budget check reject
# configurations the app can actually run, had it ever been switched on.
DECODER_BUFFERS = 8

# The app's own output budget fair-shares DELIVERED resolution and forces a
# high-res source (e.g. a pinned 4K) to downscale when other streams exist - and
# that 4K re-adapt rebuild is broken (it leaves the channel dead / blank). We
# adapt at the SOURCE instead (tier swapping + the pinned/external choice), so
# set this high enough that the app always delivers the source as-is. Egress
# stays bounded because the UI decode budget caps total pixels at 10.4 MP, and
# 10.4 MP x 25 fps ~= 260 MP/s, under the ~280 MP/s the platform sustains.


def tier_for(streams: int) -> Tier:
    """Highest-resolution tier that survives `streams` concurrent decodes."""
    for tier in TIERS:
        if streams <= tier.max_streams:
            return tier
    return TIERS[-1]


# --------------------------------------------------------------------------
# Insight
# --------------------------------------------------------------------------


def api(path: str, payload: dict | None = None, timeout: int = 60):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(f"{INSIGHT_API}{path}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        body = r.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def media_files(prefix: str) -> list[str]:
    return sorted(f for f in (api("/api/mediasrc/videos") or []) if f.startswith(prefix))


def stage_sources(tier: Tier, count: int) -> None:
    """Point the first `count` Insight slots at this tier's media and start them.

    Tier media sets are smaller than the stream ceiling, so files repeat once
    exhausted. Decoder load is identical; only the picture repeats.
    """
    files = media_files(tier.prefix)
    if not files:
        sys.exit(f"no '{tier.prefix}*' media in Insight - see README.md (Media)")

    api("/api/mediasrc/stop-all", {})
    for i in range(count):
        api("/api/mediasrc/assign", {"index": i + 1, "file": files[i % len(files)]})
    for i in range(count):
        api("/api/mediasrc/start", {"index": i + 1})
    time.sleep(3)  # let the RTSP mounts come up before the app connects


def ingest() -> list[dict]:
    d = api("/api/ingest/stats", timeout=30)
    ch = d.get("channels", d) if isinstance(d, dict) else d
    if isinstance(ch, dict):
        ch = list(ch.values())
    return [c for c in (ch or []) if isinstance(c, dict)]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def insight_url(index: int) -> str:
    """RTSP URL of Insight media slot `index` (1-based)."""
    return f"rtsp://{INSIGHT_HOST}:{RTSP_PORT}/src{index}"


def _config_scale(urls: list[str], header: str) -> str:
    """fused_main.py schema: bare stream list, per-stream probe."""
    sources = "\n".join(f"  - {url}" for url in urls)
    return f"""# Generated by pipeline.py - {header} [mode: scale]
model:
  path: {MODELS}/yolo26n-det-int8-b1.tar.gz
  labels: {LABELS}

streams:
{sources}

input:
  tcp: true
  latency_ms: 100
  # Decoder pools. The app's loader defaults to 4 buffers +
  # throughput-low-latency (memory_opt on), which its own AppConfig comment
  # calls the cause of stutter/freezes under jitter. These are the values the
  # proven 16-stream high-density profile runs. "auto" is not a literal the
  # decoder sees: core treats auto/default/empty as "resolve tuning from the
  # admission lease" and leaves memory_opt off.
  decoder_buffers: {DECODER_BUFFERS}
  decoder_input_buffers: 2
  decoder_tuning: auto

runtime:
  profile: false
  warmup_frames: 30

inference:
  frames: 0
  fps: 0                      # 0 = follow each source's own rate; never capped
  # Shallow, not deep. Every stream shares ONE detector input port here, so
  # depth buys staleness rather than throughput: queued frames age out and get
  # dropped by KeepLatest, which shows up as detections stopping. 1/8 matches
  # the high-density profile that sustains 16 streams.
  max_inflight_per_stream: 1
  max_inflight_total: 8
  min_score: 0.30
  nms_iou: 0.60
  max_detections: 50

output:
  insight:
    host: {INSIGHT_HOST}
    video_port_base: 9000
    metadata_port_base: 9100
  video_enabled: true
  debug_dir: null
  save_every: 0
"""


def _config_live(urls: list[str], header: str) -> str:
    """adaptive-resolution-object-detector schema, re-encode topology.

    encoded_passthrough:false + metadata_rtp_timestamp:on keep the video and
    metadata on one clock so Insight's exact match succeeds (measured 100% up to
    6 streams). budget high so the app delivers each source as-is; our layer owns
    resolution.
    """
    sources = "\n".join(f"    - id: cam-{i + 1}\n      rtsp_url: {u}"
                        for i, u in enumerate(urls))
    return f"""# Generated by pipeline.py - {header} [mode: live]
model:
  path: {MODELS}/yolo26n-det-int8-b1.tar.gz
  labels: {LABELS}

streams:
  max_streams: 16
  sources:
{sources}

input:
  tcp: true
  latency_ms: 100

runtime:
  profile: false
  warmup_frames: 30
  config_watch_seconds: 1.0

inference:
  frames: 0
  fps: 0
  min_score: 0.30
  nms_iou: 0.60
  max_detections: 50

output:
  encoded_passthrough: false
  metadata_rtp_timestamp: "on"
  insight:
    host: {INSIGHT_HOST}
    video_port: 9000
    metadata_port: 9100
  video_enabled: true
  debug_dir: null
  save_every: 0
"""


def write_config_urls(urls: list[str], header: str = "mixed streams") -> None:
    """Write the detector config for the active mode's schema."""
    builder = _config_live if PIPELINE == "live" else _config_scale
    CONFIG.write_text(builder(urls, header))


def write_config(tier: Tier, streams: int) -> None:
    """Managed-only config: `streams` Insight slots, all at one tier."""
    write_config_urls(
        [insight_url(i + 1) for i in range(streams)],
        header=f"Tier {tier.name} ({tier.width}x{tier.height}) x {streams} stream(s)")


def config_streams() -> list[str]:
    """Streams in the running config, in `- url` order.

    _config_scale() writes a bare list under `streams:`, one URL per line, with
    no `id:` key at all - unlike the adaptive pipeline's rich schema. Counting
    "- id:" lines here always found zero, so cmd_add() below always believed no
    stream was running and clobbered slot 1 on every call.
    """
    if not CONFIG.exists():
        return []
    lines = CONFIG.read_text().splitlines()
    try:
        start = lines.index("streams:") + 1
    except ValueError:
        return []
    out = []
    for ln in lines[start:]:
        if ln.startswith("  - "):
            out.append(ln[len("  - "):].strip())
        elif ln.strip() == "" or not ln.startswith("  "):
            break
    return out


# --------------------------------------------------------------------------
# DevKit
# --------------------------------------------------------------------------


def exec_devkit(cmd: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a shell command on the DevKit: locally if we are the DevKit, else SSH."""
    if _ON_DEVKIT:
        argv = ["bash", "-lc", cmd]
    else:
        argv = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", DEVKIT, cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


# Back-compat alias; existing call sites read cleaner as ssh() from the SDK.
ssh = exec_devkit


def app_running() -> bool:
    # Match the detector by its argv, not a bare "python": on the DevKit the UI
    # server is itself a python process and a "grep python" would count it too.
    return bool(exec_devkit(f'pgrep -f "{APP_PATTERN}" | head -1').stdout.strip())


def stop_app(grace_s: int = 20) -> bool:
    """Stop the detector, giving it time to tear its graphs down first.

    SIGKILL leaves decoder and CVU buffer pools allocated in the reserved
    region, and the next run then fails to allocate even at a stream count that
    worked before. SIGTERM lets the application release them, so a clean stop is
    what keeps the next start clean.

    Returns True if it exited on its own, False if it had to be killed.
    """
    pids = ssh(f'pgrep -f "{APP_PATTERN}" | tr "\\n" " "').stdout.strip()
    if not pids:
        return True

    ssh(f'kill -TERM {pids} 2>/dev/null || true')
    for _ in range(grace_s):
        if not ssh(f'pgrep -f "{APP_PATTERN}" | tr "\\n" " "').stdout.strip():
            return True
        time.sleep(1)

    ssh(f'kill -9 {pids} 2>/dev/null || true; sleep 2')
    mark_runtime_dirty("this pipeline's detector")
    return False


# A SIGKILLed detector leaves decoder and CVU pools allocated in the reserved
# region, and the next start must run fix_devkit_runtime.sh to reclaim them.
# But by then the process is gone, so no probe can discover that it was killed -
# `down` used to print that the next `up` would reclaim, and the next `up` then
# saw no PID, judged the stop clean and skipped the reclaim. Record it where the
# next invocation will see it, on the DevKit itself, so the promise holds across
# separate CLI runs and across the CLI/panel split.
RUNTIME_DIRTY_MARKER = "/tmp/sima-detector-unclean-stop"


def mark_runtime_dirty(what: str) -> None:
    exec_devkit(f"touch {RUNTIME_DIRTY_MARKER}")
    print(f"warning: {what} had to be killed - decoder/MLA pools may be stranded; "
          f"the next start will reclaim them", flush=True)


def runtime_dirty() -> bool:
    return bool(exec_devkit(
        f"test -f {RUNTIME_DIRTY_MARKER} && echo dirty").stdout.strip())


def reset_runtime() -> None:
    """Reclaim MLA/decoder memory left behind by a previous run.

    Decoder pools are carved from a reserved region that is not always fully
    released when an application exits, so a second run can fail to allocate
    even at a stream count that worked before. This is the recovery the DevKit
    ships for it, and the app's README calls for it after any earlier ML/video
    app has used the runtime.
    """
    ssh("bash /usr/bin/fix_devkit_runtime.sh >/dev/null 2>&1 || true", timeout=600)
    exec_devkit(f"rm -f {RUNTIME_DIRTY_MARKER}")


def _term_then_kill(pattern: str, grace_s: int = 20) -> bool:
    """SIGTERM a pattern's processes and WAIT for them, SIGKILL only if they hang.

    Same contract as stop_app(): the app releases decoder/CVU pools during its
    SIGTERM teardown, and killing before that finishes strands them in the
    reserved region so the next start fails to allocate. A fixed two-second
    sleep was long enough only while SIGTERM was unhandled and killed instantly;
    now that the detector actually tears down on TERM, graph teardown needs the
    full grace period. Returns True if the processes exited on their own.
    """
    pids = ssh(f'pgrep -f "{pattern}" | tr "\n" " "').stdout.strip()
    if not pids:
        return True
    ssh(f'kill -TERM {pids} 2>/dev/null || true')
    for _ in range(grace_s):
        if not ssh(f'pgrep -f "{pattern}" | tr "\n" " "').stdout.strip():
            return True
        time.sleep(1)
    ssh(f'kill -9 {pids} 2>/dev/null || true; sleep 2')
    mark_runtime_dirty(pattern)
    return False


def stop_any_detector() -> bool:
    """Stop BOTH pipelines' detector apps.

    The two pipelines share one MLA and one set of Insight ports, so only one
    detector may run at a time. stop_app() only targets THIS pipeline's app, so
    before starting we also clear the OTHER pipeline's app - otherwise starting
    the live pipeline while the scale detector runs would collide on the MLA.
    """
    # Every pipeline's config filename, in either language. Kept as literals
    # rather than derived from one mapping, because each pipeline module only
    # knows its own PIPELINE name.
    all_clean = True
    for pattern in ALL_PIPELINE_PATTERNS:
        if not _term_then_kill(pattern):
            all_clean = False
            print(f"warning: {pattern} did not exit in time and was killed; "
                  f"its decoder/MLA pools may need fix_devkit_runtime.sh", flush=True)
    return all_clean


def start_app() -> None:
    # fix_devkit_runtime.sh costs ~60 s. It only matters when pools were leaked,
    # which happens on an unclean (SIGKILL) stop - so pay for it only then.
    clean = stop_app()
    # A FOREIGN detector that had to be killed strands pools just as ours does,
    # and once it is gone no later probe can discover that - so its result has
    # to be folded in here, not just warned about.
    foreign_clean = stop_any_detector()  # shared MLA: only one detector at a time
    # runtime_dirty() carries a kill from a PREVIOUS invocation - notably
    # `down`, which killed the detector and then exited, leaving nothing for the
    # two checks above to see.
    if not clean or not foreign_clean or runtime_dirty():
        why = ("previous stop" if not clean
               else "another pipeline's detector" if not foreign_clean
               else "an earlier forced stop")
        print(f"{why} was unclean - reclaiming decoder/MLA pools", flush=True)
        reset_runtime()
    log_q = shlex.quote(str(LOG))
    exec_devkit(f'rm -f {log_q}; setsid nohup {app_command()} '
                f'--config {shlex.quote(str(CONFIG))} '
                f'> {log_q} 2>&1 < /dev/null & sleep 2', timeout=180)


def wait_for_streams(n: int, timeout_s: int = 300) -> bool:
    """Block until n streams report a delivered resolution, or the app dies."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        log = Path(LOG)
        text = log.read_text() if log.exists() else ""
        # The fused app prints each stream's "] rtsp=" banner BEFORE building the
        # shared graph, so banners alone meant a build failure still looked like
        # a successful run. Require its post-build marker too. The adaptive app's
        # "] channel=" banner is printed after that stream's own build, so it
        # stands on its own.
        if text.count("] channel=") >= n:
            return True
        if "[app] graph running" in text and text.count("] rtsp=") >= n:
            return True
        if "Traceback" in text or "Decoder plugin pool" in text:
            return False
        time.sleep(2)
    return False


def delivered() -> dict[str, str]:
    """Map cam id -> delivered WxH, parsed from the application log."""
    log = Path(LOG)
    out: dict[str, str] = {}
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        if "] rtsp=" in line and "stream=" in line:
            idx = line.split("[stream ")[1].split("]")[0]
            geom = line.split("stream=")[1].split()[0]      # WxH@fps
            out[f"cam-{int(idx) + 1}"] = geom.split("@")[0]
    return out


# --------------------------------------------------------------------------
# source probing
# --------------------------------------------------------------------------


_PROBE_CODE = """
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    print("FAIL"); raise SystemExit(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
f = cap.get(cv2.CAP_PROP_FPS) or 0
cap.release()
print("OK", w, h, f)
"""


def probe_source(url: str, timeout_s: int = 25):
    """True (width, height, fps) of an RTSP source, or None if unreachable.

    The declared resolution a user types is NEVER what the pipeline decodes at -
    the detector app probes each source itself and uses the real geometry. That
    made a typed resolution both redundant and dangerous: it fed the capacity
    guard, so calling a 4K camera "720p" let a config through that then collapsed
    to zero detections. Probing here makes the guard agree with reality.

    Runs under the pyneat venv python, the only interpreter on the DevKit with
    cv2, and is wrapped in `timeout` so an unreachable camera cannot hang the
    request.
    """
    import shlex
    code = shlex.quote(_PROBE_CODE)
    cmd = f"timeout {timeout_s} {PYTHON} -c {code} {shlex.quote(url)} 2>/dev/null | tail -1"
    try:
        out = exec_devkit(cmd, timeout=timeout_s + 20).stdout.strip()
    except Exception:  # noqa: BLE001 - a failed probe is reported, never raised
        return None
    if not out.startswith("OK"):
        return None
    try:
        _, w, h, f = out.split()
        w, h, f = int(w), int(h), float(f)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h, (f if f > 0 else float(FPS))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_up(n: int) -> None:
    tier = tier_for(n)
    print(f"tier {tier.name} ({tier.width}x{tier.height}) for {n} stream(s)")
    if n > tier.max_streams:
        print(f"WARNING: {n} exceeds the measured ceiling of {tier.max_streams}")
    stage_sources(tier, n)
    write_config(tier, n)
    start_app()
    print("waiting for streams ...")
    if wait_for_streams(n):
        for cam, res in sorted(delivered().items()):
            print(f"  {cam}: delivered {res}")
    else:
        print("FAILED - check the log:")
        print(ssh(f"grep -E 'ERR|Traceback' {LOG} | head -3").stdout[:600])


def cmd_add() -> None:
    # The fused app builds ONE graph for every stream and has no config watch
    # (see src/python/fused_app.py / fused_app.h): unlike the adaptive pipeline,
    # there is no live add here. Appending a source into the running config would
    # neither restart the app nor even parse next time - _config_scale()'s bare
    # list has no `id:` key to append onto. A full rebuild is the only option.
    n = len(config_streams()) + 1
    print(f"scale mode has no live add - restarting for {n} stream(s)")
    cmd_up(n)


def cmd_set(n: int) -> None:
    print(f"restarting for a uniform tier across {n} stream(s)")
    cmd_up(n)


def cmd_status() -> None:
    cams = config_streams()
    print(f"config streams : {len(cams)}")
    print(f"app running    : {app_running()}")
    for cam, res in sorted(delivered().items()):
        print(f"  {cam}: delivered {res}")

    before = {c.get("channel"): (c.get("metadata") or {}).get("messages_received") or 0
              for c in ingest()}
    time.sleep(10)
    rows = ingest()
    if not rows:
        print("no active Insight channels")
        return
    print(f"\n{'ch':>3} {'kbps':>7} {'fps':>6}")
    for c in sorted(rows, key=lambda x: x.get("channel", 0)):
        ch = c.get("channel")
        now = (c.get("metadata") or {}).get("messages_received") or 0
        fps = (now - before.get(ch, now)) / 10
        kbps = round((c.get("rtp", {}).get("bitrate_bps") or 0) / 1000)
        print(f"{ch:>3} {kbps:>7} {fps:>6.1f}")


def cmd_down() -> None:
    clean = stop_app()
    print("detector stopped cleanly" if clean
          else "detector did not exit in time and was killed; "
               "the next `up` runs fix_devkit_runtime.sh to reclaim its pools")
    stopped = api("/api/mediasrc/stop-all", {}) or {}
    print(f"insight sources stopped: {stopped.get('stopped_count', '?')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_up = sub.add_parser("up", help="start with N streams")
    p_up.add_argument("streams", type=int)
    sub.add_parser("add", help="add one stream (restarts to rebuild)")
    p_set = sub.add_parser("set", help="change stream count, restart for a uniform tier")
    p_set.add_argument("streams", type=int)
    sub.add_parser("status", help="show delivered resolution, bitrate and live FPS")
    sub.add_parser("down", help="stop application and sources")
    args = ap.parse_args()

    {"up": lambda: cmd_up(args.streams),
     "add": cmd_add,
     "set": lambda: cmd_set(args.streams),
     "status": cmd_status,
     "down": cmd_down}[args.cmd]()


if __name__ == "__main__":
    main()
