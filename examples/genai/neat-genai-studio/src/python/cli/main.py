#!/usr/bin/env python3
"""Neat GenAI Studio — terminal chat client (``./run.sh --cli``).

Talks directly to the running Neat model server — the control API to list and
load catalog models, and the OpenAI-compatible endpoint to stream chat — so you
can use the studio from a terminal with no browser. Uses only the standard
library (+ PyYAML to read the config).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

# Make `server` / `shared` importable so the CLI can reuse the same Hugging Face
# search/download the UI uses (server/hub.py) — no pyneat needed for that.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- colour (matches run.sh; degrades on non-TTY / NO_COLOR / dumb) ----------
def _use_color():
    return (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


if _use_color():
    RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
    ACCENT = "\033[38;2;154;190;30m"      # lime
    TEAL = "\033[38;2;61;179;138m"
    MUTED = "\033[38;2;140;150;160m"
    OK = "\033[38;2;53;196;137m"
    ERR = "\033[38;2;239;91;98m"
    CODE = "\033[38;2;120;200;230m"       # inline code
    ITAL = "\033[3m"
    ULINE = "\033[4m"
else:
    RESET = BOLD = DIM = ACCENT = TEAL = MUTED = OK = ERR = CODE = ITAL = ULINE = ""


_CTRL_TOKENS = re.compile(r"</s>|<pad>|<0x[0-9A-Fa-f]+>")


def type_label(t):
    """Human label for a model type — 'chat' is shown as LLM."""
    return {"chat": "LLM", "vlm": "VLM", "asr": "ASR"}.get((t or "chat").lower(), (t or "").upper())


def fmt_bytes(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if (n < 10 and unit != "B") else f"{n:.0f} {unit}"
        n /= 1024
    return ""


def _inline_md(s):
    if not _use_color():
        return s
    s = re.sub(r"`([^`]+)`", lambda m: f"{CODE}{m.group(1)}{RESET}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
    s = re.sub(r"__([^_]+)__", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", lambda m: f"{ITAL}{m.group(1)}{RESET}", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f"{ULINE}{m.group(1)}{RESET} {MUTED}({m.group(2)}){RESET}", s)
    return s


def render_md_line(ln, state):
    """Render one Markdown line to ANSI. `state` is a 1-item list tracking whether
    we're inside a ``` code fence across lines (so it works line-by-line while
    streaming)."""
    if not _use_color():
        return ln
    stripped = ln.strip()
    if stripped.startswith("```"):
        state[0] = not state[0]
        return f"{DIM}{'─' * 40}{RESET}"
    if state[0]:
        return f"{CODE}{ln}{RESET}"
    m = re.match(r"^(#{1,6})\s+(.*)$", ln)
    if m:
        return f"{BOLD}{ACCENT}{m.group(2)}{RESET}"
    if stripped.startswith(">"):
        return f"{DIM}│{RESET} {_inline_md(stripped.lstrip('>').strip())}"
    m = re.match(r"^(\s*)[-*+]\s+(.*)$", ln)
    if m:
        return f"{m.group(1)}{ACCENT}•{RESET} {_inline_md(m.group(2))}"
    m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", ln)
    if m:
        return f"{m.group(1)}{ACCENT}{m.group(2)}.{RESET} {_inline_md(m.group(3))}"
    return _inline_md(ln)


def render_markdown_ansi(text):
    """Render a whole Markdown block to ANSI. Plain text on non-TTY."""
    if not _use_color():
        return text
    state = [False]
    return "\n".join(render_md_line(ln, state) for ln in text.split("\n"))


def load_config(config_path):
    """Return ((ctrl_host, ctrl_port), (oai_host, oai_port), max_tokens)."""
    ctrl = ("127.0.0.1", 9997)
    oai = ("127.0.0.1", 9998)
    max_tokens = 256
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        app = cfg.get("app", {}) or {}
        c = app.get("control", {}) or {}
        o = app.get("openai", {}) or {}
        ctrl = (c.get("client_host", ctrl[0]), int(c.get("port", ctrl[1])))
        oai = (o.get("client_host", oai[0]), int(o.get("port", oai[1])))
        max_tokens = int((app.get("request", {}) or {}).get("max_tokens", max_tokens))
    except Exception:
        pass
    return ctrl, oai, max_tokens


def _http(url, data=None, method=None, timeout=60):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(
        url, data=body, headers=headers,
        method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8") or "{}")


def ctrl_get(ctrl, path):
    return _http(f"http://{ctrl[0]}:{ctrl[1]}{path}")


def ctrl_post(ctrl, path, payload, timeout=600):
    return _http(f"http://{ctrl[0]}:{ctrl[1]}{path}", data=payload, timeout=timeout)


def catalog(ctrl):
    try:
        return ctrl_get(ctrl, "/control/catalog").get("catalog", []) or []
    except Exception:
        return []


def wait_ready(oai, timeout=90):
    deadline = time.monotonic() + timeout
    url = f"http://{oai[0]}:{oai[1]}/v1/models"
    while time.monotonic() < deadline:
        try:
            _http(url, timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def reset_mla_wait(ctrl, oai, announce=True):
    """Reset the accelerator (MLA). The model server exits and the run.sh
    supervisor relaunches it; returns True once the new server accepts requests."""
    if announce:
        print(f"{MUTED}  resetting the accelerator (MLA) — the server will restart, "
              f"this takes a few seconds…{RESET}")
    try:
        ctrl_post(ctrl, "/control/reset_mla", {}, timeout=10)
    except Exception:
        pass   # the server exits mid-request, so this call often won't return
    time.sleep(3)               # let the old server exit before polling
    return wait_ready(oai, timeout=120)


def stream_chat(oai, model, messages, max_tokens, render=False):
    """Stream a completion. Returns (text, ttft_seconds, tps, tokens).
    With render=True the response is shown live, rendered as Markdown line-by-line
    (complete lines are rendered as they arrive; the trailing partial line is
    flushed at the end) so the reply is visible while it streams. A live token
    count trails the current line. With render=False raw tokens are printed as
    they arrive."""
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "stream": True}
    req = urllib.request.Request(
        f"http://{oai[0]}:{oai[1]}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    parts, ttft, tps, tokens = [], None, None, 0
    buf, md_state = "", [False]   # render mode: current incomplete line + code-fence state

    def _redraw_partial():
        # Show the in-progress line live with a trailing "… N tok" count, redrawn
        # in place. Tail-truncate to one terminal row so a long line never wraps
        # (which would corrupt the in-place redraw); the full text is preserved in
        # `parts` and rendered in full once the line completes.
        width = shutil.get_terminal_size((80, 24)).columns
        count = f"  … {tokens} tok"
        avail = max(0, width - len(count) - 1)
        shown = buf
        if len(shown) > avail:
            shown = ("…" + shown[-(avail - 1):]) if avail > 1 else ""
        sys.stdout.write(f"\r\x1b[K{shown}{DIM}{count}{RESET}")
        sys.stdout.flush()

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if "ttft" in obj:
                try:
                    ttft = float(obj["ttft"])
                except (TypeError, ValueError):
                    pass
            if "tps" in obj:
                try:
                    tps = float(obj["tps"])
                except (TypeError, ValueError):
                    pass
            choices = obj.get("choices") or []
            delta = choices[0].get("delta", {}).get("content") if choices else None
            if not delta:
                continue
            clean = _CTRL_TOKENS.sub("", delta)
            if clean:
                parts.append(clean)
                tokens += 1
                if render:
                    buf += clean
                    while "\n" in buf:   # flush every completed line as rendered Markdown
                        line_text, buf = buf.split("\n", 1)
                        sys.stdout.write("\r\x1b[K")   # drop the partial+counter line
                        print(render_md_line(line_text, md_state))
                    _redraw_partial()   # keep the in-progress line + live count visible
                else:
                    sys.stdout.write(clean)
                    sys.stdout.flush()
    if render:
        sys.stdout.write("\r\x1b[K")   # clear the trailing partial+counter
        if buf:
            print(render_md_line(buf, md_state))   # render the last (unterminated) line
        else:
            sys.stdout.flush()
    return "".join(parts), ttft, tps, tokens


HELP = f"""{MUTED}Commands:
  /models            list catalog models with metadata (● loaded, ○ not)
  /load [name]       load a model (no name → arrow-key menu)
  /download          browse Hugging Face, download a model, and load it
  /unload [name]     unload a model (no name → unload the loaded LLM/VLM)
  /image [path]      attach an image to your next message (VLM only; no path → prompt)
  /camera [device]   arm the board camera: every message then auto-sends a fresh
                     frame to the VLM. /camera off to stop; device defaults to
                     /dev/video0 (aliases /cam, /webcam)
  /system <text>     set a system prompt (empty clears it)
  /new               clear the conversation history
  /reset             reset the accelerator (MLA) and restart the model server
  /tokens <n>        set the max response tokens
  /benchmark [sel] [runs] [tok]   TTFT/TPS benchmark. sel: blank=active model,
                     'all', or a comma-list. e.g. /benchmark all 5 128 (aliases /bench, /perf)
  /help              show this help
  /quit              exit (aliases: /exit, /bye, /q, or Ctrl+D)
Anything else is sent to the model as a chat message.{RESET}"""


def _cam_label(device):
    """Human-friendly device label: a bare index becomes /dev/videoN."""
    dev = str(device)
    return f"/dev/video{dev}" if dev.isdigit() else dev


def _read_if_nonempty(path):
    """Return the file's bytes if it exists and is non-empty, else None."""
    try:
        if os.path.getsize(path) > 0:
            with open(path, "rb") as fh:
                return fh.read()
    except OSError:
        pass
    return None


def capture_camera_frame(device=None, timeout=12):
    """Grab a single still frame from the board's camera and return
    ``(jpeg_bytes, tool_label)``.

    The Studio runs on the Modalix board, so this reads a camera physically
    attached to the board (unlike the web UI, which uses the browser's camera).
    To avoid a heavy dependency it shells out to whichever capture tool the board
    already has — ffmpeg, fswebcam, or libcamera/rpicam — and only falls back to
    OpenCV if it happens to be importable. Raises ``RuntimeError`` with guidance
    if nothing on the box can grab a frame.

    ``device`` may be an index (``0`` → ``/dev/video0``) or a node path; it
    defaults to ``$NEAT_CAMERA_DEVICE`` or ``/dev/video0``.
    """
    import shutil
    import subprocess
    import tempfile

    dev = (device or os.environ.get("NEAT_CAMERA_DEVICE") or "/dev/video0").strip()
    dev_node = f"/dev/video{dev}" if dev.isdigit() else dev

    errors = []
    fd, tmp_path = tempfile.mkstemp(prefix="neat-cam-", suffix=".jpg")
    os.close(fd)

    def _run(cmd):
        subprocess.run(cmd, check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 1) ffmpeg via V4L2 — near-ubiquitous, honors a timeout, grabs one frame.
        ff = shutil.which("ffmpeg")
        if ff:
            try:
                _run([ff, "-nostdin", "-hide_banner", "-loglevel", "error",
                      "-f", "v4l2", "-i", dev_node,
                      "-frames:v", "1", "-q:v", "2", "-y", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, "ffmpeg"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"ffmpeg: {exc}")

        # 2) fswebcam — the classic USB-webcam still grabber.
        fs = shutil.which("fswebcam")
        if fs:
            try:
                _run([fs, "-q", "--no-banner", "-d", dev_node,
                      "-r", "1280x720", "-S", "3", "--jpeg", "90", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, "fswebcam"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"fswebcam: {exc}")

        # 3) libcamera / rpicam still — CSI cameras on ARM boards.
        for tool in ("libcamera-still", "rpicam-still"):
            binp = shutil.which(tool)
            if not binp:
                continue
            try:
                _run([binp, "-n", "-t", "800",
                      "--width", "1280", "--height", "720", "-o", tmp_path])
                data = _read_if_nonempty(tmp_path)
                if data:
                    return data, tool
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{tool}: {exc}")

        # 4) OpenCV — only if it is already installed (not a declared dependency).
        try:
            import cv2  # type: ignore  # noqa: I001
            # cv2 takes an integer index, so map "/dev/video2" → 2 (matching the
            # dev_node the other tools use); refuse a node we can't map rather
            # than silently opening index 0 (the wrong camera).
            if dev.isdigit():
                idx = int(dev)
            else:
                mnum = re.search(r"(\d+)$", dev)
                if mnum is None:
                    raise RuntimeError(f"cannot map {dev_node} to a camera index")
                idx = int(mnum.group(1))
            cap = cv2.VideoCapture(idx)
            try:
                ok, frame = False, None
                for _ in range(5):   # let auto-exposure settle for a frame or two
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        break
                if ok and frame is not None:
                    ok2, buf = cv2.imencode(".jpg", frame)
                    if ok2:
                        return bytes(buf), "opencv"
                errors.append("opencv: no frame")
            finally:
                cap.release()
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"opencv: {exc}")

        detail = "; ".join(errors) if errors else "no capture tool found on PATH"
        raise RuntimeError(
            f"could not grab a frame from {dev_node} ({detail}). "
            "Install ffmpeg, fswebcam, or libcamera-apps on the board, or set "
            "NEAT_CAMERA_DEVICE to the right /dev/video* node.")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def select_menu(items, prompt="Select"):
    """Pick one of ``items`` (a list of ``(label, value)``). Returns the chosen
    value, or None if cancelled. Uses an arrow-key menu on a real terminal and a
    numbered prompt otherwise (or if termios is unavailable)."""
    items = list(items)
    if not items:
        return None
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_menu(items, prompt)
        except Exception:  # noqa: BLE001 - fall back to a numbered prompt
            pass
    return _numbered_menu(items, prompt)


def _numbered_menu(items, prompt):
    print(f"{MUTED}{prompt}{RESET}")
    for i, (label, _v) in enumerate(items, 1):
        print(f"  {DIM}{i}.{RESET} {label}")
    try:
        raw = input(f"{MUTED}  number (blank to cancel) ▸ {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if raw.isdigit() and 1 <= int(raw) <= len(items):
        return items[int(raw) - 1][1]
    return None


def _arrow_menu(items, prompt):
    import select as _select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    idx = 0

    def draw(first):
        if not first:
            sys.stdout.write(f"\x1b[{len(items)}A")   # move up over the previous list
        for i, (label, _v) in enumerate(items):
            if i == idx:
                sys.stdout.write(f"\x1b[2K  {ACCENT}{BOLD}▸ {label}{RESET}\n")
            else:
                sys.stdout.write(f"\x1b[2K    {MUTED}{label}{RESET}\n")
        sys.stdout.flush()

    print(f"{MUTED}{prompt} — ↑/↓ move · Enter select · Esc cancel{RESET}")
    draw(True)
    sel = None
    try:
        tty.setcbreak(fd)   # char-at-a-time input, keeps \n->\r\n output translation
        while True:
            ch = os.read(fd, 1)
            if ch in (b"\r", b"\n"):
                sel = items[idx][1]
                break
            if ch == b"q":
                break
            if ch == b"\x1b":
                ready, _, _ = _select.select([fd], [], [], 0.05)
                if not ready:
                    break                      # bare Esc -> cancel
                seq = os.read(fd, 2)
                if seq == b"[A":
                    idx = (idx - 1) % len(items)
                    draw(False)
                elif seq == b"[B":
                    idx = (idx + 1) % len(items)
                    draw(False)
    except (KeyboardInterrupt, OSError):
        sel = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return sel


# ---- model loading + Hugging Face (reuses server/hub.py, like the UI) --------
def load_model(ctrl, name, oai=None, auto_retry=True):
    """Load a model via the control API. Returns True on success. Loading a
    chat/VLM model evicts any other resident one, so that is made explicit.

    If the load fails and ``oai`` is given, auto-recover once: reset the
    accelerator (MLA) and retry before notifying the user — mirrors the UI, where
    a first load can fail if the MLA is in a bad state."""
    try:
        resident = [m.get("name") for m in catalog(ctrl)
                    if m.get("loaded") and m.get("type", "chat") != "asr"
                    and m.get("name") != name]
    except Exception:
        resident = []
    if resident:
        print(f"{MUTED}  unloading {', '.join(resident)}, then loading {name}…{RESET}")
    else:
        print(f"{MUTED}  loading {name}…{RESET}")

    def _attempt():
        r = ctrl_post(ctrl, "/control/load", {"name": name})
        if isinstance(r, dict) and r.get("error"):
            raise RuntimeError(r["error"])
        return r

    try:
        r = _attempt()
    except Exception as exc:  # noqa: BLE001
        if auto_retry and oai is not None:
            # Auto-recover: reset the MLA once and retry before giving up.
            print(f"{MUTED}  load failed ({exc}) — resetting the accelerator (MLA) "
                  f"and retrying once…{RESET}")
            if not reset_mla_wait(ctrl, oai, announce=False):
                print(f"{ERR}  the server did not come back after the reset.{RESET}")
                return False
            try:
                print(f"{MUTED}  retrying load of {name}…{RESET}")
                r = _attempt()
            except Exception as exc2:  # noqa: BLE001
                print(f"{ERR}  {exc2}{RESET}")
                return False
        else:
            print(f"{ERR}  {exc}{RESET}")
            return False

    ev = r.get("evicted") if isinstance(r, dict) else None
    ev = ev if isinstance(ev, list) else ([ev] if ev else [])
    if ev:
        print(f"{MUTED}  unloaded {', '.join(str(x) for x in ev)}{RESET}")
    secs = r.get("load_seconds") if isinstance(r, dict) else None
    tnote = f" {DIM}(in {secs:.1f}s){RESET}" if isinstance(secs, (int, float)) and secs > 0 else ""
    print(f"{OK}✔ active model: {name}{RESET}{tnote}")
    return True


def _hub_config(config_path):
    """Read catalog_dir + Hugging Face settings from the config."""
    from pathlib import Path
    catalog_dir, allow, orgs = None, True, ("simaai", "TDoSiMa")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        models = (cfg.get("server", {}) or {}).get("models", {}) or {}
        if models.get("catalog_dir"):
            catalog_dir = Path(str(models["catalog_dir"])).expanduser()
        hub = (cfg.get("server", {}) or {}).get("hub", {}) or {}
        allow = bool(hub.get("allow_download", True))
        raw = hub.get("orgs", hub.get("org", list(orgs)))
        orgs = tuple(x for x in ([raw] if isinstance(raw, str) else raw) if x)
    except Exception:
        pass
    from shared.config import HubConfig
    return catalog_dir, HubConfig(allow_download=allow, orgs=orgs)


def hub_available(config_path):
    """True when Hugging Face downloads are allowed and the board is online."""
    try:
        from server.hub import hub_enabled
        _, hub = _hub_config(config_path)
        return hub_enabled(hub)
    except Exception:
        return False


def browse_and_download(ctrl, config_path, oai=None):
    """Browse Hugging Face (same models the UI lists), download one with a live
    progress bar, then load it. Returns the loaded model name, or None."""
    try:
        from server.hub import hub_download_stream, hub_search, safe_name
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  Hugging Face support unavailable: {exc}{RESET}")
        return None
    catalog_dir, hub = _hub_config(config_path)
    print(f"{MUTED}  searching Hugging Face…{RESET}")
    try:
        res = hub_search(catalog_dir, hub, "")
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  search failed: {exc}{RESET}")
        return None
    if not res.get("enabled"):
        print(f"{MUTED}  Hugging Face is offline or disabled.{RESET}")
        return None
    results = res.get("results", []) or []
    if not results:
        print(f"{MUTED}  no models found.{RESET}")
        return None
    items = []
    for m in results:
        meta = []
        if m.get("sizeBytes"):
            meta.append(fmt_bytes(m["sizeBytes"]))
        if m.get("downloads") is not None:
            meta.append(f"{m['downloads']:,}↓")
        if m.get("alreadyInCatalog"):
            meta.append("in catalog")
        label = m.get("repoId") + (f"  {DIM}{' · '.join(meta)}{RESET}" if meta else "")
        items.append((label, m.get("repoId")))
    repo_id = select_menu(items, "Download which model from Hugging Face?")
    if not repo_id:
        print(f"{MUTED}  (cancelled){RESET}")
        return None
    name = safe_name(repo_id)
    print(f"{MUTED}  downloading {repo_id}…{RESET}")
    try:
        for line in hub_download_stream(catalog_dir, hub, repo_id):
            try:
                evt = json.loads(line)
            except Exception:
                continue
            state = evt.get("state")
            if state == "downloading":
                pct = evt.get("pct")
                sz = (f"{fmt_bytes(evt.get('downloaded'))} / {fmt_bytes(evt.get('total'))}"
                      if evt.get("total") else fmt_bytes(evt.get("downloaded")))
                bar = ""
                if pct is not None:
                    filled = int(pct / 5)
                    bar = f"[{'█' * filled}{'░' * (20 - filled)}] {pct}%"
                sys.stdout.write(f"\r\x1b[K{MUTED}  {bar} {sz}{RESET}")
                sys.stdout.flush()
            elif state == "resolving":
                sys.stdout.write(f"\r\x1b[K{MUTED}  resolving…{RESET}")
                sys.stdout.flush()
            elif state == "done":
                name = evt.get("name") or name
                sys.stdout.write(f"\r\x1b[K{OK}✔ downloaded {name}{RESET}\n")
            elif state == "error":
                sys.stdout.write(f"\r\x1b[K{ERR}  download failed: {evt.get('message')}{RESET}\n")
                return None
    except Exception as exc:  # noqa: BLE001
        print(f"\n{ERR}  download error: {exc}{RESET}")
        return None
    try:
        ctrl_post(ctrl, "/control/rescan", {})   # make the server aware of it
    except Exception:
        pass
    return name if load_model(ctrl, name, oai=oai) else None


def choose_and_load_model(ctrl, config_path, oai=None):
    """Startup prompt: pick a catalog model to load. When the catalog is empty,
    go straight into Hugging Face to download one. Returns the active name or ''."""
    cat = catalog(ctrl)
    loadable = [m for m in cat if m.get("type", "chat") != "asr"]

    # No models to load — go straight to Hugging Face if it's available.
    if not loadable:
        if hub_available(config_path):
            print(f"{MUTED}  No models in the catalog — let's download one from "
                  f"Hugging Face.{RESET}")
            return browse_and_download(ctrl, config_path, oai=oai) or ""
        print(f"{MUTED}  No models in the catalog and Hugging Face is offline — "
              f"use /load or /download once one is available.{RESET}")
        return ""

    # Models exist — prompt to load one (with a Hugging Face option when online).
    items = []
    for m in loadable:
        meta = [type_label(m.get("type"))]
        if m.get("sizeBytes"):
            meta.append(fmt_bytes(m["sizeBytes"]))
        if m.get("complete") is False:
            meta.append("incomplete")
        mark = "●" if m.get("loaded") else "○"
        items.append((f"{mark} {m['name']}  ({' · '.join(meta)})", ("load", m["name"])))
    if hub_available(config_path):
        items.append(("⬇ Download a model from Hugging Face…", ("hub", None)))
    choice = select_menu(items, "Load a model to get started")
    if not choice:
        return ""
    kind, value = choice
    if kind == "load":
        return value if load_model(ctrl, value, oai=oai) else ""
    return browse_and_download(ctrl, config_path, oai=oai) or ""


# ---- Benchmark (MoLE perf: TTFT / TPS) --------------------------------------

def run_one_benchmark(ctrl, oai, model, runs, max_tokens, prompt):
    """Load `model` (if needed), run a TTFT/TPS benchmark, stream live progress,
    and return the summary dict (or None). Ctrl+C stops it."""
    if not any(m.get("name") == model and m.get("loaded") for m in catalog(ctrl)):
        if not load_model(ctrl, model, oai=oai):
            return None
    try:
        ctrl_post(ctrl, "/control/benchmark",
                  {"num_samples": runs, "max_new_tokens": max_tokens,
                   "prompt": prompt, "model": model}, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  could not start benchmark: {exc}{RESET}")
        return None
    while True:
        try:
            time.sleep(0.4)
            d = ctrl_get(ctrl, "/control/benchmark/status")
        except KeyboardInterrupt:
            try:
                ctrl_post(ctrl, "/control/benchmark/stop", {}, timeout=5)
            except Exception:
                pass
            sys.stdout.write("\r\x1b[K")
            print(f"{MUTED}  (benchmark stopped){RESET}")
            return None
        except Exception:
            continue
        total, done, cur = d.get("total", 0), d.get("done", 0), d.get("current")
        if cur:
            bits = f"{cur.get('tokens', 0)} tok"
            if cur.get("ttftMs") is not None:
                bits += f" · TTFT {cur['ttftMs']}ms"
            if cur.get("tps") is not None:
                bits += f" · {cur['tps']} tok/s"
            sys.stdout.write(f"\r{DIM}  run {cur.get('index', 0) + 1}/{total} · {bits}{RESET}\x1b[K")
            sys.stdout.flush()
        elif total:
            sys.stdout.write(f"\r{DIM}  {done}/{total} runs done{RESET}\x1b[K")
            sys.stdout.flush()
        if not d.get("running"):
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
            return d.get("summary")


def _bench_stat_row(label, m, unit=""):
    return (f"  {label:<13} min {m['min']:>8}  max {m['max']:>8}  avg {m['mean']:>8}  "
            f"med {m['median']:>8}  σ {m['stdev']:>7}  p90 {m['p90']:>8}{unit}")


def print_benchmark_summary(summary):
    if not summary:
        print(f"{ERR}  no valid runs.{RESET}")
        return
    note = f" · {summary['errors']} error(s)" if summary.get("errors") else ""
    print(f"{OK}  {summary['count']} run(s){note}{RESET}")
    print(f"{DIM}{_bench_stat_row('TTFT (ms)', summary['ttftMs'])}{RESET}")
    print(f"{DIM}{_bench_stat_row('Throughput', summary['tps'], ' tok/s')}{RESET}")
    print(f"{DIM}{_bench_stat_row('Tokens/run', summary['tokens'])}{RESET}")


def print_benchmark_comparison(results):
    valid = [r for r in results if r["summary"]]
    if len(valid) < 2:
        return
    best_tps = max(r["summary"]["tps"]["mean"] for r in valid)
    best_ttft = min(r["summary"]["ttftMs"]["mean"] for r in valid)
    print(f"\n{ACCENT}{BOLD}  Comparison  {DIM}(★ = best){RESET}")
    print(f"{MUTED}  {'Model':<26} {'TPS':>8} {'p90':>7} {'TTFT ms':>9} {'tok':>6} {'σ TPS':>7} {'runs':>5}{RESET}")
    for r in results:
        s = r["summary"]
        name = (r["model"][:25] + "…") if len(r["model"]) > 26 else r["model"]
        if not s:
            print(f"  {name:<26} {ERR}failed{RESET}")
            continue
        tps, ttft = s["tps"]["mean"], s["ttftMs"]["mean"]
        tm = f"{OK}★{RESET}" if tps == best_tps else " "
        fm = f"{OK}★{RESET}" if ttft == best_ttft else " "
        print(f"  {name:<26} {tps:>7}{tm} {s['tps']['p90']:>7} {ttft:>8}{fm} "
              f"{s['tokens']['mean']:>6} {s['tps']['stdev']:>7} {s['count']:>5}")


def export_benchmark(results, config):
    try:
        path = input(f"{MUTED}  export results? (path.csv / path.json, blank to skip) ▸ {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if not path:
        return
    path = os.path.expanduser(path)
    try:
        if path.lower().endswith(".json"):
            payload = {"config": config,
                       "results": [{"model": r["model"], "summary": r["summary"]} for r in results]}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        else:
            cols = ["model", "tpsMean", "tpsP90", "tpsStdev", "ttftMeanMs",
                    "tokensMean", "validRuns", "errors"]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(",".join(cols) + "\n")
                for r in results:
                    s = r["summary"]
                    if not s:
                        fh.write(f"{r['model']},,,,,,,\n")
                        continue
                    fh.write(f"{r['model']},{s['tps']['mean']},{s['tps']['p90']},"
                             f"{s['tps']['stdev']},{s['ttftMs']['mean']},"
                             f"{s['tokens']['mean']},{s['count']},{s.get('errors', 0)}\n")
        print(f"{OK}✔ saved to {path}{RESET}")
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  export failed: {exc}{RESET}")


def do_benchmark(ctrl, oai, active, arg):
    """Handle the /benchmark command. `arg` = "[selector] [runs] [tokens]" where
    selector is 'all', a comma-list of models, or empty (active model)."""
    runs, max_new, words = 5, 128, []
    nums = []
    for p in arg.split():
        (nums if p.isdigit() else words).append(p)
    selector = " ".join(words)
    if len(nums) >= 1:
        runs = max(1, min(50, int(nums[0])))
    if len(nums) >= 2:
        max_new = max(8, min(2048, int(nums[1])))

    names = [m["name"] for m in catalog(ctrl) if m.get("type", "chat") != "asr"]
    if selector.lower() == "all":
        models = names
    elif selector:
        models = [s.strip() for s in selector.split(",") if s.strip()]
        unknown = [m for m in models if m not in names]
        if unknown:
            print(f"{ERR}  unknown model(s): {', '.join(unknown)}{RESET}")
            return
    elif active:
        models = [active]
    else:
        print(f"{ERR}  no model loaded — /load one, or /benchmark all{RESET}")
        return
    if not models:
        print(f"{MUTED}  no models to benchmark.{RESET}")
        return

    print(f"{MUTED}  benchmarking {len(models)} model(s) · {runs} runs · {max_new} tokens each…{RESET}")
    results = []
    for i, model in enumerate(models):
        if len(models) > 1:
            print(f"\n{ACCENT}▸ Model {i + 1}/{len(models)}: {model}{RESET}")
        summary = run_one_benchmark(ctrl, oai, model, runs, max_new, "")
        results.append({"model": model, "summary": summary})
        if summary:
            print_benchmark_summary(summary)
        elif len(models) > 1:
            continue   # keep going through the rest even if one failed
    print_benchmark_comparison(results)
    if any(r["summary"] for r in results):
        export_benchmark(results, {"runs": runs, "maxNewTokens": max_new})


def startup_menu(ctrl, oai, config_path, active):
    """Ask the user what they want to do on start. Returns the active model name
    (possibly changed by loading/downloading), then the REPL begins."""
    hint = f"  {DIM}(loaded: {active}){RESET}" if active else ""
    items = [
        (f"💬  Chat with a model{hint}", "chat"),
        ("📊  Benchmark model(s)", "bench"),
    ]
    if hub_available(config_path):
        items.append(("⬇  Download a model from Hugging Face", "download"))
    items.append(("⌨   Go straight to the prompt", "prompt"))
    choice = select_menu(items, "What would you like to do?")

    if not choice or choice == "prompt":
        return active
    if choice == "chat":
        if active and any(m.get("name") == active and m.get("loaded") for m in catalog(ctrl)):
            return active
        return choose_and_load_model(ctrl, config_path, oai=oai)
    if choice == "download":
        return browse_and_download(ctrl, config_path, oai=oai) or active
    if choice == "bench":
        try:
            sel = input(f"{MUTED}  which models? (blank = active · 'all' · comma-list) ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return active
        do_benchmark(ctrl, oai, active, sel)
        loaded = [m for m in catalog(ctrl)
                  if m.get("loaded") and m.get("type", "chat") != "asr"]
        return loaded[0]["name"] if loaded else active
    return active


def main():
    ap = argparse.ArgumentParser(description="Neat GenAI Studio terminal chat")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="", help="model to make active on start")
    ap.add_argument("--max-tokens", type=int, default=0)
    args = ap.parse_args()

    ctrl, oai, cfg_max = load_config(args.config)
    max_tokens = args.max_tokens or cfg_max

    print(f"{ACCENT}{BOLD}▸{RESET} Neat GenAI Studio — terminal chat")
    print(f"{MUTED}  connecting to the model server ({oai[0]}:{oai[1]})…{RESET}")
    if not wait_ready(oai):
        print(f"{ERR}✘ The model server never became ready.{RESET}")
        return 1

    cat = catalog(ctrl)
    active = args.model.strip()
    if not active:
        loaded = [m for m in cat if m.get("loaded") and (m.get("type", "chat") != "asr")]
        if loaded:
            active = loaded[0]["name"]

    if active and not any(m.get("name") == active and m.get("loaded") for m in cat):
        if not load_model(ctrl, active, oai=oai):
            active = ""
    elif active:
        print(f"{OK}✔ active model: {active}{RESET}")

    # On an interactive start (no explicit --model), ask what the user wants to do:
    # chat, benchmark, download, or drop straight to the prompt. Otherwise fall
    # back to the load/download prompt when nothing is loaded.
    if not args.model.strip() and sys.stdin.isatty():
        active = startup_menu(ctrl, oai, args.config, active)
    elif not active:
        active = choose_and_load_model(ctrl, args.config, oai=oai)

    print(f"{MUTED}  /help for commands, /quit to exit.{RESET}\n")

    messages, system, pending_image, camera_device = [], None, None, None
    while True:
        # The prompt shows the loaded model, 🖼 if a one-shot image is queued,
        # and 📷 if the live camera is armed (a frame is grabbed every message).
        mlabel = active if active else "no model"
        marks = ""
        if pending_image:
            marks += f" {ACCENT}🖼{RESET}"
        if camera_device is not None:
            marks += f" {ACCENT}📷{RESET}"
        try:
            line = input(f"{ACCENT}{BOLD}you{RESET} {DIM}[{mlabel}]{RESET}{marks} "
                         f"{ACCENT}{BOLD}▸{RESET} ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(f"\n{MUTED}(Ctrl+D or /quit to exit){RESET}")
            continue
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd, arg = cmd.lower(), arg.strip()
            if cmd in ("quit", "exit", "q", "bye"):
                break
            elif cmd in ("help", "h", "?"):
                print(HELP)
            elif cmd in ("models", "ls"):
                cat = catalog(ctrl)
                if not cat:
                    print(f"{MUTED}  (catalog empty — use /download to fetch one){RESET}")
                for m in cat:
                    mark = f"{OK}●{RESET}" if m.get("loaded") else f"{MUTED}○{RESET}"
                    meta = [type_label(m.get("type"))]
                    if m.get("supportsVision") and (m.get("type", "chat").lower() != "vlm"):
                        meta.append("vision")
                    if m.get("sizeBytes"):
                        meta.append(fmt_bytes(m["sizeBytes"]))
                    if m.get("complete") is False:
                        meta.append("incomplete")
                    act = f" {ACCENT}(active){RESET}" if m.get("name") == active else ""
                    print(f"  {mark} {m.get('name')}  {DIM}{' · '.join(str(x) for x in meta)}{RESET}{act}")
            elif cmd in ("load", "use"):
                if not arg:
                    items = []
                    for m in catalog(ctrl):
                        if m.get("type", "chat") == "asr":
                            continue
                        meta = [type_label(m.get("type"))]
                        if m.get("sizeBytes"):
                            meta.append(fmt_bytes(m["sizeBytes"]))
                        mark = "●" if m.get("loaded") else "○"
                        items.append((f"{mark} {m.get('name')}  ({' · '.join(str(x) for x in meta)})",
                                      m.get("name")))
                    if not items:
                        print(f"{MUTED}  no models in the catalog — try /download.{RESET}")
                        continue
                    arg = select_menu(items, "Load which model?")
                    if not arg:
                        print(f"{MUTED}  (cancelled){RESET}")
                        continue
                if load_model(ctrl, arg, oai=oai):
                    active, messages, pending_image = arg, [], None
            elif cmd in ("download", "hub"):
                new = browse_and_download(ctrl, args.config, oai=oai)
                if new:
                    active, messages, pending_image = new, [], None
            elif cmd in ("image", "img"):
                cur = next((m for m in catalog(ctrl) if m.get("name") == active), None)
                if not active or not (cur and cur.get("supportsVision")):
                    print(f"{MUTED}  /image needs a vision (VLM) model loaded.{RESET}")
                    continue
                path = arg
                if not path:
                    try:
                        path = input(f"{MUTED}  image path (blank to clear) ▸ {RESET}").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        path = ""
                if not path:
                    pending_image = None
                    print(f"{MUTED}  image cleared.{RESET}")
                    continue
                path = os.path.expanduser(path)
                if not os.path.isfile(path):
                    print(f"{ERR}  file not found: {path}{RESET}")
                    continue
                try:
                    import base64
                    import mimetypes
                    with open(path, "rb") as fh:
                        b64 = base64.b64encode(fh.read()).decode("utf-8")
                    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
                    pending_image = f"data:{mime};base64,{b64}"
                    print(f"{OK}✔ image attached ({os.path.basename(path)}) — sent with your next message.{RESET}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{ERR}  {exc}{RESET}")
            elif cmd in ("camera", "cam", "webcam"):
                # Live-camera mode. Once armed to a device, EVERY message auto-grabs
                # a fresh frame from the board camera and sends it to the VLM — no
                # per-message /image needed. It reuses the same image attach path.
                #   /camera <index|node>  arm on that device (validated by a grab)
                #   /camera               toggle: arm on the default, or disarm
                #   /camera off|stop|none disarm
                if arg.lower() in ("off", "stop", "none", "clear") or \
                        (not arg and camera_device is not None):
                    camera_device = None
                    print(f"{OK}✔ live camera off.{RESET}")
                    continue
                target = arg if arg else (os.environ.get("NEAT_CAMERA_DEVICE") or "0")
                print(f"{MUTED}  testing camera on {_cam_label(target)}…{RESET}")
                try:
                    jpeg, tool = capture_camera_frame(target)   # validate before arming
                except KeyboardInterrupt:
                    print(f"\n{MUTED}  (camera test cancelled){RESET}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    # Leave any existing arming untouched, and say so.
                    if camera_device is not None:
                        print(f"{ERR}  could not switch camera: {exc}{RESET}")
                        print(f"{MUTED}  kept live camera on {_cam_label(camera_device)}.{RESET}")
                    else:
                        print(f"{ERR}  camera not armed: {exc}{RESET}")
                    continue
                camera_device = target
                print(f"{OK}✔ live camera armed on {_cam_label(target)} via {tool} "
                      f"({len(jpeg) // 1024} KB/frame) — every message now sends a fresh "
                      f"frame. /camera off to stop.{RESET}")
                cur = next((m for m in catalog(ctrl) if m.get("name") == active), None)
                if not (active and cur and cur.get("supportsVision")):
                    print(f"{MUTED}  note: the current model isn't a VLM — frames send "
                          f"once you load one.{RESET}")
            elif cmd == "unload":
                names = [arg] if arg else [
                    m.get('name') for m in catalog(ctrl)
                    if m.get('loaded') and m.get('type', 'chat') != 'asr']
                if not names:
                    print(f"{MUTED}  no LLM/VLM is loaded.{RESET}")
                    continue
                for name in names:
                    try:
                        ctrl_post(ctrl, "/control/unload", {"name": name})
                        print(f"{OK}✔ unloaded {name}{RESET}")
                        if name == active:
                            active = ""
                            camera_device = None   # no model → live camera can't send
                    except Exception as exc:  # noqa: BLE001
                        print(f"{ERR}  {exc}{RESET}")
            elif cmd == "system":
                system = arg or None
                messages = []
                print(f"{OK}✔ system prompt {'set' if system else 'cleared'}.{RESET}")
            elif cmd in ("new", "clear"):
                messages = []
                print(f"{OK}✔ conversation cleared.{RESET}")
            elif cmd in ("reset", "reset-mla"):
                if reset_mla_wait(ctrl, oai):
                    print(f"{OK}✔ MLA reset — server back online. Load a model to continue.{RESET}")
                    active, messages, pending_image, camera_device = "", [], None, None
                else:
                    print(f"{ERR}  the server did not come back after the reset.{RESET}")
            elif cmd in ("tokens", "max"):
                try:
                    max_tokens = max(1, int(arg))
                    print(f"{OK}✔ max_tokens = {max_tokens}{RESET}")
                except ValueError:
                    print(f"{MUTED}  usage: /tokens <n>{RESET}")
            elif cmd in ("benchmark", "bench", "perf"):
                do_benchmark(ctrl, oai, active, arg)
                # Benchmarking loads models, so the resident one may have changed;
                # resync the active model (and clear the chat if it did).
                loaded = [m for m in catalog(ctrl)
                          if m.get("loaded") and m.get("type", "chat") != "asr"]
                new_active = loaded[0]["name"] if loaded else active
                if new_active != active:
                    active, messages, pending_image = new_active, [], None
            else:
                print(f"{MUTED}  unknown command /{cmd} — /help for the list.{RESET}")
            continue

        if not active:
            print(f"{ERR}  no model loaded — /models then /load <name>.{RESET}")
            continue

        # Resolve the image for this turn. An explicit one-shot /image wins;
        # otherwise, if the live camera is armed and a VLM is loaded, grab a
        # fresh frame so every message goes out with what the camera sees.
        turn_image = pending_image
        if turn_image is None and camera_device is not None:
            cur = next((m for m in catalog(ctrl) if m.get("name") == active), None)
            if cur and cur.get("supportsVision"):
                print(f"{MUTED}  📷 grabbing a frame…{RESET}")
                try:
                    jpeg, _tool = capture_camera_frame(camera_device)
                    import base64
                    turn_image = "data:image/jpeg;base64," + \
                        base64.b64encode(jpeg).decode("utf-8")
                except KeyboardInterrupt:
                    print(f"\n{MUTED}  (frame grab cancelled){RESET}")
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"{ERR}  live frame failed ({exc}); sending text only.{RESET}")
            else:
                # Armed but the active model can't take images — keep the marker
                # honest by saying why the frame isn't going out this turn.
                print(f"{MUTED}  📷 camera armed, but the active model isn't a VLM — "
                      f"sending text only.{RESET}")
        if turn_image is not None:
            user_content = [{"type": "text", "text": line},
                            {"type": "image", "image": turn_image}]
        else:
            user_content = line
        msgs = ([{"role": "system", "content": system}] if system else []) \
            + messages + [{"role": "user", "content": user_content}]
        render = sys.stdout.isatty()
        if render:
            print(f"{TEAL}{BOLD}neat ◂{RESET}")
        else:
            print(f"{TEAL}{BOLD}neat ◂{RESET} ", end="", flush=True)
        try:
            text, ttft, tps, tokens = stream_chat(oai, active, msgs, max_tokens, render=render)
        except KeyboardInterrupt:
            print(f"\n{MUTED}(stopped){RESET}")
            continue
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:200]
            print(f"\n{ERR}  HTTP {exc.code}: {detail}{RESET}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n{ERR}  {exc}{RESET}")
            continue
        if not render:
            print()   # render mode already printed the reply live, line-by-line
        bits = []
        if tokens:
            bits.append(f"{tokens} tok")
        if ttft is not None:
            bits.append(f"ttft {ttft * 1000:.0f}ms")
        if tps is not None:
            bits.append(f"{tps:.1f} tok/s")
        if bits:
            print(f"{DIM}{'  ·  '.join(bits)}{RESET}")
        messages.append({"role": "user", "content": line})   # text only; the image is one-shot
        messages.append({"role": "assistant", "content": text})
        pending_image = None
        print()

    print(f"{MUTED}bye.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
