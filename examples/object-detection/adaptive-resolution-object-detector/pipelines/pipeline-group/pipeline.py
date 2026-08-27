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
PIPELINE = "group"

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


# ==========================================================================
# GROUPED (hybrid) MODE
# ==========================================================================
# Scale runs ONE fused graph for every stream: cheap, but any change rebuilds
# all of them. Live runs ONE graph PER stream: independent, but each stream
# carries its own model + MLA output pool + encoder, and the shared MLA pool
# runs out around 5-6 streams (measured: 4x1080p+2x720p died with
# "MLA_0_s5... Unable to allocate output memory", taking every running stream
# with it).
#
# Grouped mode splits the difference: N INDEPENDENT PROCESSES of the fused
# scale app, GROUP_SIZE streams each. One model per GROUP rather than per
# stream, so 16 streams cost 4 MLA contexts instead of live's 16 - while a
# rebuild only ever touches the one group that changed.
#
# Validated on this DevKit before this module was written: two concurrent
# instances came up clean with no MLA conflict, all four channels detected at
# 25/s, and killing group 0 left group 1 running at an unchanged 25/s.
#
# Each group owns a FIXED, non-overlapping Insight channel range:
#   group g, position j  ->  channel  g*GROUP_SIZE + j
#                            video    9000 + channel
#                            metadata 9100 + channel
# The app derives both ports as (port_base + its own stream index), so giving
# group g a port_base offset by g*GROUP_SIZE lands its streams exactly on that
# range. Fixed ranges are what make groups independent: a change inside one
# group never renumbers a channel belonging to another.
GROUP_SIZE = 4
MAX_GROUPS = 4            # GROUP_SIZE * MAX_GROUPS is the total stream ceiling
VIDEO_PORT_BASE = 9000
METADATA_PORT_BASE = 9100

# Named so pipelines/pipeline-group/ui_server.py's memory estimate can read the
# real value instead of carrying its own guess - that drift (an estimate of 18
# against this actual 8) is what let ENFORCE_LIMITS's budget check reject
# configurations the app can actually run, had it ever been switched on.
DECODER_BUFFERS = 8

def config_path(group: int) -> Path:
    """Config file for one group's app instance."""
    return HERE / f"{PIPELINE}{group}-run.yaml"


def log_path(group: int) -> str:
    """Log file for one group's app instance."""
    return str(BUNDLE / "logs" / f"{PIPELINE}{group}-py.log")


def group_pattern(group: int) -> str:
    """pgrep pattern matching ONLY this group's app instance.

    Matching on the config filename is what keeps the groups addressable
    individually - every group runs the identical app binary, so the argv
    path cannot tell them apart. The bracket makes the pattern never match
    the pgrep/ssh command carrying it.
    """
    return f"[{PIPELINE[0]}]{PIPELINE[1:]}{group}-run.yaml"

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
    Tier("1080p", 1920, 1080, "1080p_", 3),
    Tier("720p", 1280, 720, "video", 8),
]

FPS = 25

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


def _config_scale(urls: list[str], header: str, group: int = 0) -> str:
    """fused_main.py schema: bare stream list, per-stream probe.

    `group` shifts this instance's Insight port bases so its streams land on
    channels [group*GROUP_SIZE, ...] and never collide with another group's.
    group=0 reproduces the plain single-instance layout exactly.
    """
    sources = "\n".join(f"  - {url}" for url in urls)
    v_base = VIDEO_PORT_BASE + group * GROUP_SIZE
    m_base = METADATA_PORT_BASE + group * GROUP_SIZE
    return f"""# Generated by pipeline.py - {header} [mode: scale]
model:
  path: {MODELS}/yolo26n-det-int8-b1.tar.gz
  labels: {LABELS}

streams:
{sources}

input:
  tcp: true
  latency_ms: 100
  # Same fused app as `scale`, so it needs the same decoder pools. Omitting
  # these keys is not neutral: the loader then falls back to 4 buffers +
  # throughput-low-latency, which turns memory_opt ON - the setting the app's
  # own AppConfig comment blames for stutter/freezes under jitter. That is what
  # made a group lag on a 1080p source while `scale` ran the same stream clean.
  # "auto" is not a literal the decoder sees: core treats auto/default/empty as
  # "resolve tuning from the admission lease" and leaves memory_opt off.
  decoder_buffers: {DECODER_BUFFERS}
  decoder_input_buffers: 2
  decoder_tuning: auto

runtime:
  profile: false
  warmup_frames: 30

inference:
  frames: 0
  fps: 0                      # 0 = follow each source's own rate; never capped
  # Shallow, not deep - every stream in THIS group shares one detector input
  # port, so depth buys staleness rather than throughput: queued frames age out
  # and get dropped by KeepLatest, which shows up as detections stopping.
  # per_stream:1 is the binding cap here (a group holds at most GROUP_SIZE
  # streams), matching the profile that sustains 16 streams under `scale`.
  max_inflight_per_stream: 1
  max_inflight_total: 8
  min_score: 0.30
  nms_iou: 0.60
  max_detections: 50

output:
  insight:
    host: {INSIGHT_HOST}
    video_port_base: {v_base}
    metadata_port_base: {m_base}
  video_enabled: true
  debug_dir: null
  save_every: 0
"""


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


def reset_runtime() -> None:
    """Reclaim MLA/decoder memory left behind by a previous run.

    Decoder pools are carved from a reserved region that is not always fully
    released when an application exits, so a second run can fail to allocate
    even at a stream count that worked before. This is the recovery the DevKit
    ships for it, and the app's README calls for it after any earlier ML/video
    app has used the runtime.
    """
    ssh("bash /usr/bin/fix_devkit_runtime.sh >/dev/null 2>&1 || true", timeout=600)


# --------------------------------------------------------------------------
# grouped mode: one app instance per group, addressed individually
# --------------------------------------------------------------------------
# Everything below is scoped to a SINGLE group. Nothing here may touch another
# group's process, config, log or Insight channels - that isolation is the
# entire point of this pipeline.


def channel_for(group: int, pos: int) -> int:
    """Global Insight channel of position `pos` within `group`."""
    return group * GROUP_SIZE + pos


def write_config_group(group: int, urls: list[str], header: str = "") -> None:
    """Write one group's config, with its channel range baked into the ports."""
    text = _config_scale(urls, header or f"group {group}, {len(urls)} stream(s)", group)
    config_path(group).write_text(text)


def group_running(group: int) -> bool:
    return bool(exec_devkit(f'pgrep -f "{group_pattern(group)}" | head -1').stdout.strip())


def running_groups() -> list[int]:
    return [g for g in range(MAX_GROUPS) if group_running(g)]


def stop_group(group: int, grace_s: int = 20) -> bool:
    """Stop ONE group's app. Returns True if it exited cleanly on its own.

    SIGTERM first: SIGKILL leaves decoder/CVU pools allocated in the reserved
    region and the next start can then fail to allocate. Siblings are never
    signalled - the pattern is this group's own config filename.
    """
    pat = group_pattern(group)
    pids = ssh(f'pgrep -f "{pat}" | tr "\\n" " "').stdout.strip()
    if not pids:
        return True
    ssh(f'kill -TERM {pids} 2>/dev/null || true')
    for _ in range(grace_s):
        if not ssh(f'pgrep -f "{pat}" | tr "\\n" " "').stdout.strip():
            return True
        time.sleep(1)
    ssh(f'kill -9 {pids} 2>/dev/null || true; sleep 2')
    return False


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
    return False


def stop_foreign_detectors() -> None:
    """Stop the scale and live pipelines' detectors - never a sibling group.

    Grouped mode runs the same binary as the scale pipeline, so matching on the
    app path alone would kill our own groups. Both are therefore identified by
    their CONFIG filename instead, which is unique per deployment.
    """
    for pattern in ("[s]cale-run.yaml", "[l]ive-run.yaml"):
        if not _term_then_kill(pattern):
            print(f"warning: {pattern} did not exit in time and was killed; "
                  f"its decoder/MLA pools may need fix_devkit_runtime.sh", flush=True)


def stop_all_groups() -> None:
    for g in range(MAX_GROUPS):
        stop_group(g)


def start_group(group: int) -> None:
    """(Re)start ONE group's app instance, leaving every sibling untouched."""
    clean = stop_group(group)
    stop_foreign_detectors()
    # reset_runtime() restarts the shared runtime services, which would disrupt
    # any sibling group mid-flight. Only safe when nothing else is running, so
    # an unclean stop with siblings up is left for the next idle moment.
    if not clean and not running_groups():
        print(f"group {group} stop was unclean - reclaiming decoder/MLA pools", flush=True)
        reset_runtime()
    log_q = shlex.quote(log_path(group))
    exec_devkit(f'rm -f {log_q}; setsid nohup {app_command()} '
                f'--config {shlex.quote(str(config_path(group)))} '
                f'> {log_q} 2>&1 < /dev/null & sleep 2',
                timeout=180)


def wait_for_group(group: int, n: int, timeout_s: int = 300) -> bool:
    """Block until this group's app reports n streams, or it dies."""
    deadline = time.time() + timeout_s
    log = Path(log_path(group))
    while time.time() < deadline:
        text = log.read_text() if log.exists() else ""
        # Requires the fused app's post-build marker, not just the per-stream
        # banners it prints before building the shared graph - see the note in
        # wait_for_streams().
        if "[app] graph running" in text and text.count("] rtsp=") >= n:
            return True
        if "Traceback" in text or "Decoder plugin pool" in text:
            return False
        time.sleep(2)
    return False


def delivered_group(group: int) -> dict[int, str]:
    """Map position-within-group -> delivered WxH, from this group's log."""
    log = Path(log_path(group))
    out: dict[int, str] = {}
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        if "] rtsp=" in line and "stream=" in line:
            pos = int(line.split("[stream ")[1].split("]")[0])
            out[pos] = line.split("stream=")[1].split()[0].split("@")[0]
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


def group_plan(n: int) -> list[int]:
    """Stream count for each of the MAX_GROUPS groups, filled in order.

    Always length MAX_GROUPS; a trailing 0 means that group should be stopped
    if it is holding anything from a previous, larger run.
    """
    counts = []
    remaining = max(n, 0)
    for _ in range(MAX_GROUPS):
        count = min(GROUP_SIZE, remaining)
        counts.append(count)
        remaining -= count
    return counts


def total_streams() -> int:
    """Streams across every RUNNING group's own config.

    A stopped group's leftover config file does not count - it is not part of
    "how many streams are up right now", which is what cmd_add() below needs.
    """
    total = 0
    for group in running_groups():
        path = config_path(group)
        if not path.exists():
            continue
        lines_ = path.read_text().splitlines()
        try:
            i = lines_.index("streams:") + 1
        except ValueError:
            continue
        for ln in lines_[i:]:
            if ln.startswith("  - "):
                total += 1
            elif ln.strip() == "" or not ln.startswith("  "):
                break
    return total


def _stop_insight_slot(slot: int) -> None:
    """Stop one Insight source, tolerating 'already stopped' (which 500s)."""
    try:
        api("/api/mediasrc/stop", {"index": slot})
    except Exception:  # noqa: BLE001 - stopping an idle slot is not an error here
        pass


def release_group_slots(group: int) -> None:
    """Stop every Insight slot in this group's fixed range.

    A shrinking group (4 streams -> 2) or one going to zero must give back the
    positions it no longer uses, not just the ones its NEW count still needs -
    those higher positions are still playing in Insight with no detector
    referencing them otherwise, exactly the leak the web panel's stage_group()
    was fixed to avoid.
    """
    for pos in range(GROUP_SIZE):
        _stop_insight_slot(channel_for(group, pos) + 1)


def stage_group_sources(group: int, tier: Tier, count: int) -> None:
    """Point this group's own Insight slots at this tier's media and start them.

    Never calls mediasrc/stop-all - that is global and would cut every other
    group's sources, which is exactly the coupling grouped mode exists to avoid.
    """
    files = media_files(tier.prefix)
    if not files:
        sys.exit(f"no '{tier.prefix}*' media in Insight - see README.md (Media)")
    # Stop the FULL range, not just the positions the new count still needs -
    # see release_group_slots(). This also covers the same-count case: Insight
    # 500s on /start for a slot already playing.
    release_group_slots(group)
    slots = [channel_for(group, pos) + 1 for pos in range(count)]
    for i, slot in enumerate(slots):
        api("/api/mediasrc/assign", {"index": slot, "file": files[i % len(files)]})
    for slot in slots:
        api("/api/mediasrc/start", {"index": slot})
    if slots:
        time.sleep(3)  # let the RTSP mounts come up before the app connects


def cmd_up(n: int) -> None:
    ceiling = GROUP_SIZE * MAX_GROUPS
    if n > ceiling:
        # Refuse rather than warn: group_plan() can only lay out `ceiling`
        # streams, so continuing would silently run a SMALLER experiment than
        # asked for and then report it as a success at that lower count.
        sys.exit(f"{n} streams exceeds the {ceiling}-stream ceiling "
                 f"({MAX_GROUPS} groups x {GROUP_SIZE} streams each). "
                 f"Nothing was changed - ask for at most {ceiling}.")
    counts = group_plan(n)
    active = [g for g, count in enumerate(counts) if count > 0]
    for group, count in enumerate(counts):
        if count == 0:
            stop_group(group)
            release_group_slots(group)
            continue
        tier = tier_for(count)
        print(f"group {group}: tier {tier.name} ({tier.width}x{tier.height}) "
              f"for {count} stream(s)")
        if count > tier.max_streams:
            print(f"WARNING: group {group}: {count} exceeds the measured "
                  f"ceiling of {tier.max_streams}")
        stage_group_sources(group, tier, count)
        urls = [insight_url(channel_for(group, pos) + 1) for pos in range(count)]
        write_config_group(
            group, urls,
            header=f"Tier {tier.name} ({tier.width}x{tier.height}) x {count} stream(s)")
        start_group(group)
    print("waiting for streams ...")
    if all(wait_for_group(g, counts[g]) for g in active):
        for g in active:
            for cam, res in sorted(delivered_group(g).items()):
                print(f"  group{g} cam-{cam}: delivered {res}")
    else:
        print("FAILED - check the logs:")
        for g in active:
            print(ssh(f"grep -E 'ERR|Traceback' {log_path(g)} | head -3").stdout[:600])


def cmd_add() -> None:
    # Grouped mode runs the same fused app as `scale` in each of its processes,
    # and that app has no config watch - see src/python/fused_app.py /
    # fused_app.h. There is no live add here any more than there is for `scale`;
    # cmd_up() restages and restarts every active group, not just the one that
    # picks up the new stream - simpler than tracking which group changed, at
    # the cost of a brief interruption to every running group on every add.
    n = total_streams() + 1
    print(f"grouped mode has no live add - restarting for {n} stream(s)")
    cmd_up(n)


def cmd_set(n: int) -> None:
    print(f"restarting for a uniform tier across {n} stream(s)")
    cmd_up(n)


def cmd_status() -> None:
    active = running_groups()
    print(f"config streams : {total_streams()}")
    print(f"groups running : {active if active else 'none'}")
    for g in active:
        for cam, res in sorted(delivered_group(g).items()):
            print(f"  group{g} cam-{cam}: delivered {res}")

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
    # Evaluate every group BEFORE aggregating: all() over a generator stops at
    # the first False, so one group needing a kill used to leave every
    # higher-numbered group running while `down` went on to stop the Insight
    # sources and report only that "some groups were killed".
    results = [stop_group(g) for g in range(MAX_GROUPS)]
    clean = all(results)
    print("all groups stopped cleanly" if clean
          else "some groups did not exit in time and were killed; "
               "the next `up` runs fix_devkit_runtime.sh to reclaim their pools")
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
