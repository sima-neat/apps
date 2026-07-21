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

# Board-camera capture is shared with the Flask UI backend.
from shared.board_camera import capture_camera_frame, cam_label as _cam_label  # noqa: E402


# ---- colour (matches run.sh; degrades on non-TTY / NO_COLOR / dumb) ----------
def _in_tmux():
    """True when running inside tmux or GNU screen. They multiplex the terminal
    and, unless explicitly configured for RGB, swallow 24-bit truecolor escapes."""
    return (bool(os.environ.get("TMUX"))
            or os.environ.get("TERM", "").startswith(("tmux", "screen")))


def _use_color():
    return (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM") != "dumb")


def _truecolor():
    """Whether emitting 24-bit colour is safe. Inside tmux/screen it is only safe
    when the terminal advertises it (COLORTERM); otherwise fall back to 256-colour
    so the CLI stays colourful instead of rendering as no colour. Outside a
    multiplexer, keep truecolor (unchanged behaviour)."""
    if _in_tmux():
        return os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
    return True


if _use_color():
    RESET, BOLD, DIM, ITAL, ULINE = "\033[0m", "\033[1m", "\033[2m", "\033[3m", "\033[4m"
    if _truecolor():
        ACCENT = "\033[38;2;154;190;30m"      # lime
        TEAL = "\033[38;2;61;179;138m"
        MUTED = "\033[38;2;140;150;160m"
        OK = "\033[38;2;53;196;137m"
        ERR = "\033[38;2;239;91;98m"
        CODE = "\033[38;2;120;200;230m"       # inline code
    else:
        # 256-colour approximations — render in tmux/screen without truecolor.
        ACCENT = "\033[38;5;148m"
        TEAL = "\033[38;5;36m"
        MUTED = "\033[38;5;245m"
        OK = "\033[38;5;42m"
        ERR = "\033[38;5;203m"
        CODE = "\033[38;5;117m"
else:
    RESET = BOLD = DIM = ACCENT = TEAL = MUTED = OK = ERR = CODE = ITAL = ULINE = ""


# Optional readline: gives the chat prompt up/down history recall + line editing.
# We record only the main chat prompts (not sub-prompts like confirmations), and
# persist history across sessions in a dotfile.
try:
    import readline as _readline
    _HAS_READLINE = True
    try:
        _readline.set_auto_history(False)   # we add only the main prompts, by hand
        _MANUAL_HISTORY = True
    except Exception:  # noqa: BLE001 — libedit may lack it; fall back to auto history
        _MANUAL_HISTORY = False
except Exception:  # noqa: BLE001 — no readline (rare); prompt still works, no history
    _readline = None
    _HAS_READLINE = False
    _MANUAL_HISTORY = False

_HISTORY_FILE = os.path.expanduser("~/.neat_ai_history")


def _rl(code):
    """Wrap a non-printing ANSI escape with readline's ignore markers so it
    measures the prompt width correctly. Only when readline is actually driving
    an interactive prompt (and the code is non-empty)."""
    return f"\001{code}\002" if (_HAS_READLINE and code and sys.stdin.isatty()) else code


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


def write_chat_log(path, active, system, messages):
    """Write the current conversation to a plain-text .log file and return the path."""
    lines = ["Neat GenAI Studio — chat export",
             f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Model: {active or '(none)'}",
             f"System: {system if system else '(none)'}",
             "=" * 60, ""]
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Defensive: multimodal content (text + image parts) — keep the text.
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif isinstance(p, dict) and p.get("type") == "image":
                    parts.append("[image]")
            content = "\n".join(parts)
        who = {"user": "You", "assistant": "Assistant", "system": "System"}.get(
            msg.get("role", ""), str(msg.get("role", "")).capitalize())
        lines.append(f"{who}:")
        lines.append(str(content).rstrip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def _inline_md(s):
    if not _use_color():
        return s
    s = re.sub(r"`([^`]+)`", lambda m: f"{CODE}{m.group(1)}{RESET}", s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
    s = re.sub(r"__([^_]+)__", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
    s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", lambda m: f"{ITAL}{m.group(1)}{RESET}", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f"{ULINE}{m.group(1)}{RESET} {MUTED}({m.group(2)}){RESET}", s)
    return s


# ---- LaTeX → Unicode (terminal-friendly math) --------------------------------
# We can't render real math in a terminal, but converting the common LaTeX a chat
# model emits ($x^2$, \frac, \alpha, \sqrt, \sum, …) to Unicode makes replies far
# more readable. This is a best-effort transform, not a TeX engine.
_SUP = {c: u for c, u in zip(
    "0123456789+-=()niabcdefghklmoprstuvwxyzABDEGHIJKLMNOPRTUVW",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉᶠᵍʰᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻᴬᴮᴰᴱᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾᴿᵀᵁⱽᵂ")}
_SUB = {c: u for c, u in zip(
    "0123456789+-=()aehijklmnoprstuvx",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")}
_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η", r"\theta": "θ", r"\vartheta": "ϑ",
    r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ",
    r"\varphi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ", r"\Xi": "Ξ",
    r"\Pi": "Π", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
}
_SYMBOLS = {
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\ne": "≠",
    r"\ll": "≪", r"\gg": "≫", r"\approx": "≈", r"\equiv": "≡", r"\cong": "≅",
    r"\sim": "∼", r"\simeq": "≃", r"\propto": "∝", r"\infty": "∞", r"\partial": "∂",
    r"\nabla": "∇", r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\oint": "∮",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄", r"\in": "∈", r"\notin": "∉",
    r"\ni": "∋", r"\subset": "⊂", r"\subseteq": "⊆", r"\supset": "⊃", r"\supseteq": "⊇",
    r"\cup": "∪", r"\cap": "∩", r"\setminus": "∖", r"\emptyset": "∅", r"\varnothing": "∅",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐", r"\leftrightarrow": "↔", r"\Leftrightarrow": "⇔", r"\iff": "⇔",
    r"\mapsto": "↦", r"\langle": "⟨", r"\rangle": "⟩", r"\lceil": "⌈", r"\rceil": "⌉",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\cdots": "⋯", r"\ldots": "…", r"\dots": "…",
    r"\vdots": "⋮", r"\ddots": "⋱", r"\prime": "′", r"\ast": "∗", r"\star": "⋆",
    r"\circ": "∘", r"\bullet": "•", r"\deg": "°", r"\angle": "∠", r"\perp": "⊥",
    r"\parallel": "∥", r"\hbar": "ℏ", r"\ell": "ℓ", r"\Re": "ℜ", r"\Im": "ℑ",
    r"\aleph": "ℵ", r"\wedge": "∧", r"\land": "∧", r"\vee": "∨", r"\lor": "∨",
    r"\neg": "¬", r"\lnot": "¬", r"\oplus": "⊕", r"\otimes": "⊗", r"\odot": "⊙",
    r"\mathbb{R}": "ℝ", r"\mathbb{Z}": "ℤ", r"\mathbb{N}": "ℕ", r"\mathbb{Q}": "ℚ",
    r"\mathbb{C}": "ℂ",
}
_MATH_CMDS = {**_GREEK, **_SYMBOLS}
_CMD_RE = re.compile(r"\\[a-zA-Z]+")


def _script(content, table):
    """Map ``content`` to Unicode super/subscripts; fall back to ^(..)/_(..)."""
    content = content.strip()
    if content and all(c in table for c in content):
        return "".join(table[c] for c in content)
    marker = "^" if table is _SUP else "_"
    return f"{marker}({content})" if len(content) > 1 else f"{marker}{content}"


def _frac(a, b):
    a, b = a.strip(), b.strip()
    aw = a if len(a) <= 1 else f"({a})"
    bw = b if len(b) <= 1 else f"({b})"
    return f"{aw}/{bw}"


def _convert_math(s):
    """Convert a LaTeX math fragment (no delimiters) to a Unicode approximation."""
    # \mathbb{R} etc. that map to a single glyph first (before brace stripping).
    for cmd, sym in _SYMBOLS.items():
        if "{" in cmd and cmd in s:
            s = s.replace(cmd, sym)
    # size/spacing wrappers and accents that just wrap their argument.
    s = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathbb|mathcal|mathit|mathsf|operatorname|mbox)"
               r"\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:vec|hat|bar|tilde|dot|ddot|overline|underline|widehat|widetilde)"
               r"\s*\{([^{}]*)\}", r"\1", s)
    # Known Greek/symbol commands first (longest match is intrinsic to \[a-zA-Z]+).
    s = _CMD_RE.sub(lambda m: _MATH_CMDS.get(m.group(0), m.group(0)), s)
    # \sqrt[n]{x}, \sqrt{x} and \frac{a}{b} → (a)/(b). Each regex only matches a
    # brace group with no nested braces, so run them together to a fixpoint: the
    # innermost construct converts first each pass, exposing the next one out —
    # this handles either nesting order (frac-in-sqrt or sqrt-in-frac). Done BEFORE
    # scripts so ^/​_ applied to a \frac/\sqrt macro doesn't eat the backslash.
    # Each pass strips ≥1 macro, so the command count bounds the iterations.
    for _ in range(s.count("\\frac") + s.count("\\sqrt") + 1):
        before = s
        s = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}",
                   lambda m: f"{_script(m.group(1), _SUP)}√({m.group(2)})", s)
        s = re.sub(r"\\sqrt\s*\{([^{}]*)\}",
                   lambda m: (f"√({m.group(1)})" if len(m.group(1)) > 1 else f"√{m.group(1)}"), s)
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}",
                   lambda m: _frac(m.group(1), m.group(2)), s)
        if s == before:
            break
    # Superscripts / subscripts (braced form first, then single token). The single
    # token class excludes () so it won't re-mangle a ^(…)/_(…) fallback, and
    # excludes \ so a script on an unbraced macro can't swallow its backslash.
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP), s)
    s = re.sub(r"\^([^\s{}()^_\\])", lambda m: _script(m.group(1), _SUP), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB), s)
    s = re.sub(r"_([^\s{}()^_\\])", lambda m: _script(m.group(1), _SUB), s)
    # Spacing commands and math line breaks → a space.
    s = re.sub(r"\\(?:quad|qquad|,|;|:|!|\s)", " ", s)
    s = s.replace("\\\\", " ")
    # Remaining unknown \command → its bare name; then drop grouping braces.
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"[ \t]{2,}", " ", s)


def _looks_mathy(inner):
    """Heuristic: is ``inner`` (between $…$) real math, not currency/prose?"""
    if re.search(r"[\\^_{}]", inner):          # LaTeX structure → definitely math
        return True
    s = inner.strip()
    if not s or re.fullmatch(r"[\s\d.,]+", s):  # empty / pure number → currency
        return False
    words = s.split()
    if re.search(r"[=<>+\-*/]", s):            # has an operator → a compact expression
        # A dangling operator at either end means the closing '$' was really the
        # next currency amount's '$' ("$5 + $10" → inner "5 +", "$5-$10" → "5-");
        # reject so the '$' pair is left intact. Real math won't start/end on a
        # bare operator (a leading '-' is a valid unary minus, so it's excluded).
        if re.match(r"^[=<>+*/]", s) or re.search(r"[=<>+\-*/]$", s):
            return False
        # Only multi-letter words count as prose; chained operators and lone
        # single-letter variables ("a + b + c + d") shouldn't be penalized.
        prose = [w for w in words if re.fullmatch(r"[A-Za-z]{2,}", w)]
        return len(s) <= 60 and len(prose) < 4
    # No operator/structure: accept only a single short token (a lone variable),
    # so prose or "$5 and $10"-style currency runs are left untouched.
    return len(words) == 1 and len(s) <= 12 and bool(re.search(r"[A-Za-z]", s))


def latex_to_unicode(text):
    """Convert inline/single-line LaTeX math in ``text`` to Unicode (best effort)."""
    if "$" not in text and "\\(" not in text and "\\[" not in text:
        return text
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: _convert_math(m.group(1)), text, flags=re.S)
    text = re.sub(
        r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)",
        lambda m: _convert_math(m.group(1)) if _looks_mathy(m.group(1)) else m.group(0),
        text)
    return text


def render_md_line(ln, state):
    """Render one Markdown line to ANSI. `state` tracks cross-line context so it
    works while streaming: state[0] = inside a ``` code fence, state[1] = inside a
    $$…$$ / \\[…\\] display-math block."""
    if len(state) < 2:          # tolerate the older 1-item state shape
        state.append(False)
    stripped = ln.strip()
    if stripped.startswith("```"):
        state[0] = not state[0]
        return f"{DIM}{'─' * 40}{RESET}" if _use_color() else ln
    if state[0]:
        return f"{CODE}{ln}{RESET}" if _use_color() else ln
    # Display-math block: lone $$ or \[ … \] on their own lines.
    if state[1]:
        if stripped in ("$$", "\\]"):
            state[1] = False
            return ""
        # Guard against an unterminated opener (e.g. the reply is truncated
        # mid-equation): a line that is clearly Markdown structure, or has no math
        # markers at all, implicitly closes the block and renders normally — so a
        # stray $$ can't turn the rest of the reply into indented pseudo-math.
        if stripped and (stripped.startswith(("```", "#", ">", "|"))
                         or re.match(r"^\s*[-*+]\s", ln)
                         or re.match(r"^\s*\d+[.)]\s", ln)
                         or not re.search(r"[\\^_{}=]", stripped)):
            state[1] = False
            # fall through to normal rendering below
        else:
            conv = _convert_math(ln)
            return f"  {ACCENT}{conv}{RESET}" if _use_color() else f"  {conv}"
    if stripped in ("$$", "\\["):
        state[1] = True
        return ""
    # Convert inline / single-line LaTeX math to Unicode (also on non-TTY).
    ln = latex_to_unicode(ln)
    if not _use_color():
        return ln
    stripped = ln.strip()
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
    """Render a whole Markdown block. ANSI styling on a TTY; either way LaTeX math
    is converted to Unicode for readability (render_md_line handles both cases)."""
    state = [False, False]
    return "\n".join(render_md_line(ln, state) for ln in text.split("\n"))


def load_config(config_path):
    """Return ((ctrl_host, ctrl_port), (oai_host, oai_port), max_tokens)."""
    ctrl = ("127.0.0.1", 9997)
    oai = ("127.0.0.1", 9998)
    max_tokens = 512
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
    buf, md_state = "", [False, False]   # render: [in code fence, in $$ math block]

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
  /download          browse Hugging Face — pick one, several, or all models to
                     download (Space to multi-select, 'a' for all), then load one
  /unload [name]     unload a model (no name → unload the loaded LLM/VLM)
  /delete [name]     delete a model's weights from disk (no name → menu; asks to
                     confirm; aliases /rm, /remove)
  /image [path]      attach an image to your next message (VLM only; no path → prompt)
  /camera [device]   arm the board camera: every message then auto-sends a fresh
                     frame to the VLM. /camera off to stop; device defaults to
                     /dev/video0 (aliases /cam, /webcam)
  /system <text>     set a system prompt (empty clears it)
  /new               clear the conversation history
  /export [file]     save this chat to a .log file (default neat-chat-<time>.log)
  /reset             reset the accelerator (MLA) and restart the model server
  /tokens <n>        set the max response tokens
  /benchmark [sel] [runs] [tok]   TTFT/TPS benchmark. sel: blank=active model,
                     'all', or a comma-list. e.g. /benchmark all 5 128 (aliases /bench, /perf)
  /rag [filter]      inspect the RAG database — list ingested chunks (aliases /docs)
  /rag on|off        toggle RAG-augmented chat (retrieved passages added to prompts)
  /rag search <q>    semantic search — show top matches without asking the model
  /rag db [path]     show, or switch to, which milvus.db is served ('default' reverts)
  /rag status        show the RAG toggle, active database and service state
  /rag reset|clear   rebuild from the default document, or clear all RAG documents
  /help              show this help
  /quit              exit (aliases: /exit, /bye, /q, or Ctrl+D)
Anything else is sent to the model as a chat message.{RESET}"""


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


def select_multi(items, prompt="Select"):
    """Pick one *or more* of ``items`` (a list of ``(label, value)``). Returns a
    list of chosen values (order preserved), or None if cancelled. Arrow-key
    checkbox menu on a real terminal, numbered multi-select otherwise."""
    items = list(items)
    if not items:
        return None
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_multiselect(items, prompt)
        except Exception:  # noqa: BLE001 - fall back to a numbered prompt
            pass
    return _numbered_multi(items, prompt)


def _numbered_multi(items, prompt):
    print(f"{MUTED}{prompt}{RESET}")
    for i, (label, _v) in enumerate(items, 1):
        print(f"  {DIM}{i}.{RESET} {label}")
    try:
        raw = input(f"{MUTED}  numbers (e.g. 1,3-5), 'all', blank to cancel ▸ {RESET}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    if raw.lower() in ("all", "a", "*"):
        return [v for _l, v in items]
    picked, seen = [], set()
    for tok in re.split(r"[,\s]+", raw):
        if not tok:
            continue
        rng = re.fullmatch(r"(\d+)-(\d+)", tok)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            if lo > hi:
                lo, hi = hi, lo
            lo, hi = max(lo, 1), min(hi, len(items))   # clamp: "1-999999" can't freeze
            nums = range(lo, hi + 1)
        elif tok.isdigit():
            nums = [int(tok)]
        else:
            continue
        for n in nums:
            if 1 <= n <= len(items) and n not in seen:
                seen.add(n)
                picked.append(items[n - 1][1])
    return picked or None


def _arrow_multiselect(items, prompt):
    import select as _select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    idx, chosen = 0, set()

    def draw(first):
        if not first:
            sys.stdout.write(f"\x1b[{len(items)}A")   # move up over the previous list
        for i, (label, _v) in enumerate(items):
            box = f"{ACCENT}[x]{RESET}" if i in chosen else f"{DIM}[ ]{RESET}"
            if i == idx:
                sys.stdout.write(f"\x1b[2K  {ACCENT}{BOLD}▸{RESET} {box} {BOLD}{label}{RESET}\n")
            else:
                sys.stdout.write(f"\x1b[2K    {box} {MUTED}{label}{RESET}\n")
        sys.stdout.flush()

    print(f"{MUTED}{prompt}\n  ↑/↓ move · Space toggle · a all · Enter confirm · Esc cancel{RESET}")
    draw(True)
    result = None
    try:
        tty.setcbreak(fd)
        while True:
            ch = os.read(fd, 1)
            if ch in (b"\r", b"\n"):
                # Nothing ticked → treat like single-select on the cursor item.
                result = [items[i][1] for i in sorted(chosen)] if chosen else [items[idx][1]]
                break
            if ch == b" ":
                chosen.discard(idx) if idx in chosen else chosen.add(idx)
                draw(False)
            elif ch in (b"a", b"A"):
                chosen = set() if len(chosen) == len(items) else set(range(len(items)))
                draw(False)
            elif ch == b"q":
                break
            elif ch == b"\x1b":
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
        result = None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return result


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


def _download_one(catalog_dir, hub, repo_id):
    """Download a single Hugging Face repo with a live progress bar. Returns the
    catalog model name on success, or None on failure."""
    from server.hub import hub_download_stream, safe_name
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
                sys.stdout.write(f"\r\x1b[K{ERR}  download failed ({repo_id}): "
                                 f"{evt.get('message')}{RESET}\n")
                return None
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write("\n")
        print(f"{ERR}  download error ({repo_id}): {exc}{RESET}")
        return None
    return name


def download_specific(ctrl, config_path, repo_id, oai=None):
    """Download one specific Hugging Face repo (no browsing), rescan, and load it.
    Returns the loaded model name, or None."""
    try:
        from server.hub import hub_search  # noqa: F401 - probe availability early
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  Hugging Face support unavailable: {exc}{RESET}")
        return None
    if not hub_available(config_path):
        print(f"{MUTED}  Hugging Face is offline or disabled.{RESET}")
        return None
    catalog_dir, hub = _hub_config(config_path)
    try:
        name = _download_one(catalog_dir, hub, repo_id)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print(f"{MUTED}  (download cancelled){RESET}")
        return None
    if not name:
        return None
    try:
        ctrl_post(ctrl, "/control/rescan", {})   # make the server aware of it
    except Exception:
        pass
    return name if load_model(ctrl, name, oai=oai) else None


def browse_and_download(ctrl, config_path, oai=None):
    """Browse Hugging Face (same models the UI lists), download one, several, or
    all with a live progress bar, then load one. Returns the loaded model name (the
    first successfully downloaded), or None."""
    try:
        from server.hub import hub_search  # noqa: F401 - probe availability early
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
    repo_ids = select_multi(items, "Download which model(s) from Hugging Face?")
    if not repo_ids:
        print(f"{MUTED}  (cancelled){RESET}")
        return None
    total = len(repo_ids)
    downloaded, failed = [], []
    for i, repo_id in enumerate(repo_ids, 1):
        if total > 1:
            print(f"{MUTED}  [{i}/{total}] {repo_id}{RESET}")
        try:
            name = _download_one(catalog_dir, hub, repo_id)
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            print(f"{MUTED}  (download cancelled — keeping what finished){RESET}")
            break
        (downloaded if name else failed).append(name or repo_id)
    if not downloaded:
        print(f"{ERR}  nothing downloaded.{RESET}")
        return None
    try:
        ctrl_post(ctrl, "/control/rescan", {})   # make the server aware of them
    except Exception:
        pass
    if total > 1:
        print(f"{OK}✔ downloaded {len(downloaded)}/{total}: {', '.join(downloaded)}{RESET}")
        if failed:
            print(f"{ERR}  failed: {', '.join(failed)}{RESET}")
        print(f"{MUTED}  loading {downloaded[0]} (only one model is resident; "
              f"/load to switch).{RESET}")
    first = downloaded[0]
    return first if load_model(ctrl, first, oai=oai) else None


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


def _startup_chat(ctrl, config_path, active, oai=None):
    """Ensure a model is loaded so the REPL can chat; returns the active name."""
    if active and any(m.get("name") == active and m.get("loaded") for m in catalog(ctrl)):
        return active
    return choose_and_load_model(ctrl, config_path, oai=oai)


def _load_named_model(ctrl, config_path, name, active, oai=None):
    """Load a specific catalog model by name and return it. Falls back to the
    picker if the name isn't in the catalog, or to the prior active on load error."""
    if name not in [m.get("name") for m in catalog(ctrl)]:
        print(f"{ERR}  no model named '{name}' in the catalog — pick one:{RESET}")
        return choose_and_load_model(ctrl, config_path, oai=oai)
    return name if load_model(ctrl, name, oai=oai) else active


def _startup_benchmark(ctrl, oai, active, sel=None):
    """Run a benchmark and return the (possibly changed) active model so the REPL
    can continue. ``sel`` (blank / 'all' / comma-list / a model name) is prompted
    for when None."""
    if sel is None:
        try:
            sel = input(f"{MUTED}  which models? (blank = active · 'all' · comma-list) ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return active
    do_benchmark(ctrl, oai, active, sel)
    loaded = [m for m in catalog(ctrl)
              if m.get("loaded") and m.get("type", "chat") != "asr"]
    return loaded[0]["name"] if loaded else active


def _rag_service_up():
    """True if the RAG VectorDB service (owns the DB file) is reachable."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:9100/", timeout=1):
            return True
    except Exception:  # noqa: BLE001
        return False


def _rag_embedding_dir(config_path):
    """Read app.rag.embedding_model_dir from the config, or ''."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return (((cfg.get("app") or {}).get("rag") or {}).get("embedding_model_dir")) or ""
    except Exception:  # noqa: BLE001
        return ""


def _default_rag_source():
    """Path to the bundled default RAG Markdown (src/common/rag/neat.md)."""
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(src, "common", "rag", "neat.md")


# ---------------------------------------------------------------------------
# RAG-augmented chat (CLI). The web UI owns a VectorDB service (127.0.0.1:9100)
# for semantic search; in CLI mode we start that same worker ourselves so the
# user can toggle retrieval on/off and switch which milvus.db it serves.
# ---------------------------------------------------------------------------
_RAG = {"on": False, "proc": None, "db": None, "k": 3}
_RAG_ATEXIT = False
# The compiled on-board model has a small fixed context window, so bound how much
# retrieved text we inject: each passage and the whole block are capped.
_RAG_PASSAGE_CHARS = 1200
_RAG_CONTEXT_CHARS = 3000


def _rag_active_db():
    """The milvus.db the CLI is currently pointed at (a /rag db override, else the
    app default)."""
    if _RAG["db"]:
        return _RAG["db"]
    from rag.inspect_db import default_db_path
    return default_db_path()


def _rag_worker_path():
    """Path to rag/vectordb_worker.py (spawned as the CLI's RAG service)."""
    py = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/python
    return os.path.join(py, "rag", "vectordb_worker.py")


def _rag_service_db():
    """The milvus.db the running service reports serving (None if unreachable or
    unknown). Lets us tell whether a *reused* service actually serves our DB."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:9100/", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        served = data.get("db_path")
        return os.path.abspath(served) if served else None
    except Exception:  # noqa: BLE001
        return None


def _rag_display_db():
    """The DB actually in use: a reused (external) service's served DB wins over our
    override, since that's what queries hit. Falls back to the active/default DB."""
    if _rag_service_up() and _RAG["proc"] is None:
        served = _rag_service_db()
        if served:
            return served
    return os.path.abspath(_rag_active_db())


def _rag_context_block(hits):
    """Turn search hits into a size-bounded context block (per-passage and total
    caps) so retrieval can't overflow the small on-board context window. Returns
    (block_text, passages_used)."""
    parts, total = [], 0
    for h in hits:
        text = str(h.get("content", "")).strip()
        if not text:
            continue
        if len(text) > _RAG_PASSAGE_CHARS:
            text = text[:_RAG_PASSAGE_CHARS].rstrip() + "…"
        if total + len(text) > _RAG_CONTEXT_CHARS:
            room = _RAG_CONTEXT_CHARS - total
            if room > 200:                # only append a worthwhile remainder
                parts.append(text[:room].rstrip() + "…")
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts), len(parts)


def _rag_config(config_path):
    """Return (enabled, embedding_model_dir) from the app config."""
    enabled, emb = True, ""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        rag = ((cfg.get("app") or {}).get("rag") or {})
        enabled = bool(rag.get("enabled", True))
        emb = rag.get("embedding_model_dir") or ""
    except Exception:  # noqa: BLE001
        pass
    return enabled, emb


def _wait_rag_ready(timeout=120, proc=None):
    """Poll until the RAG service answers (it only binds the port after the
    embedding model + DB have loaded). Bails early if ``proc`` exits first."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False                  # worker died during startup
        if _rag_service_up():
            return True
        time.sleep(0.5)
    return False


def _start_rag_service(config_path):
    """Ensure a RAG service is available. Reuses one that's already up (e.g. the
    web UI's); otherwise spawns our own worker pointed at the active DB. Returns
    True when a service is reachable."""
    if _rag_service_up():                 # someone already serves 9100 — use it
        served, want = _rag_service_db(), os.path.abspath(_rag_active_db())
        if served and served != want:
            print(f"{MUTED}  note: the running RAG service serves {served} (not the "
                  f"selected {want}) — using it until the studio stops.{RESET}")
        return True
    enabled, emb = _rag_config(config_path)
    if not emb:
        print(f"{ERR}  no RAG embedding model configured (app.rag.embedding_model_dir).{RESET}")
        return False
    worker = _rag_worker_path()
    if not os.path.isfile(worker):
        print(f"{ERR}  RAG worker not found: {worker}{RESET}")
        return False
    db = _rag_active_db()
    if not os.path.isfile(db):
        print(f"{ERR}  no RAG database at {db} — build one (upload in the UI) or "
              f"'/rag db <path>' to point at an existing milvus.db.{RESET}")
        return False
    import subprocess
    env = os.environ.copy()
    env["VDB_EMBED_MODEL_DIR"] = emb
    env["VECTOR_DB_PATH"] = db
    print(f"{MUTED}  starting the RAG service (loads the embedding model — a moment)…{RESET}")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", worker], env=env, start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  could not start the RAG service: {exc}{RESET}")
        return False
    _RAG["proc"] = proc
    global _RAG_ATEXIT
    if not _RAG_ATEXIT:                    # stop our worker when the CLI exits
        import atexit
        atexit.register(_stop_rag_service)
        _RAG_ATEXIT = True
    try:
        ready = _wait_rag_ready(proc=proc)
    except KeyboardInterrupt:              # Ctrl+C while it loads — cancel cleanly
        print(f"\n{MUTED}  (cancelled — stopping the RAG service){RESET}")
        _stop_rag_service()
        return False
    if not ready:
        print(f"{ERR}  the RAG service did not become ready — check the embedding model.{RESET}")
        _stop_rag_service()
        return False
    return True


def _stop_rag_service():
    """Stop the RAG service *we* started (leaves an external/UI one alone)."""
    proc = _RAG.get("proc")
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    _RAG["proc"] = None


def _rag_search(query, k=3, min_score=-1.0):
    """Semantic search via the running VectorDB service. Returns a list of
    ``{content, metadata, score}`` on success ([] is a genuine zero-hit result),
    or ``None`` if the service could not be reached — so callers can tell a broken
    retrieval apart from an empty one."""
    try:
        import urllib.parse
        qs = urllib.parse.urlencode({"query": query, "k": k, "min_score": min_score})
        with urllib.request.urlopen(f"http://127.0.0.1:9100/search?{qs}", timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", []) or []
    except Exception:  # noqa: BLE001 - service down/unreachable/error
        return None


def do_rag_on(config_path):
    """Turn on RAG augmentation: retrieved passages get prepended to each prompt."""
    enabled, _emb = _rag_config(config_path)
    if not enabled:
        print(f"{MUTED}  RAG is disabled in config (app.rag.enabled = false).{RESET}")
        return
    if _start_rag_service(config_path):
        _RAG["on"] = True
        who = "" if _RAG["proc"] else " (shared with the running studio)"
        print(f"{OK}✔ RAG search ON{who} — messages are augmented with the top "
              f"{_RAG['k']} passages.{RESET}")


def do_rag_off():
    """Turn off RAG augmentation (leaves any service running)."""
    _RAG["on"] = False
    print(f"{OK}✔ RAG search OFF.{RESET}")


def do_rag_search_cmd(query, config_path):
    """One-off semantic search — print the top matches without sending to the model."""
    if not query:
        print(f"{MUTED}  usage: /rag search <query>{RESET}")
        return
    if not _start_rag_service(config_path):
        return
    try:
        hits = _rag_search(query, k=_RAG["k"])
    except KeyboardInterrupt:
        print(f"\n{MUTED}  (search cancelled){RESET}")
        return
    if hits is None:
        print(f"{ERR}  RAG service unreachable — try /rag on.{RESET}")
        return
    if not hits:
        print(f"{MUTED}  no matches.{RESET}")
        return
    print(f"\n{ACCENT}{BOLD}RAG search{RESET} {DIM}“{query}”{RESET}\n")
    for i, h in enumerate(hits, 1):
        score = h.get("score")
        tag = f" {DIM}score {score:.3f}{RESET}" if isinstance(score, (int, float)) else ""
        text = str(h.get("content", "")).strip()
        preview = text if len(text) <= 400 else text[:400].rstrip() + "…"
        print(f"{ACCENT}{i}.{RESET}{tag}")
        for line in preview.split("\n"):
            print(f"   {line}")
        print()


def do_rag_db(path, config_path):
    """Show or switch the milvus.db the CLI serves. Switching restarts our service
    (if any) against the new file; won't touch a service owned by the web UI."""
    if not path:
        print(f"{MUTED}  current RAG database:{RESET} {_rag_active_db()}")
        print(f"{MUTED}  usage: /rag db <path-to-milvus.db>   (or 'default' to revert){RESET}")
        return
    if path.strip().lower() in ("default", "reset", "-"):
        target = None
    else:
        target = os.path.abspath(os.path.expanduser(path.strip()))
        if not os.path.isfile(target):
            print(f"{ERR}  no such file: {target}{RESET}")
            return
    if _rag_service_up() and _RAG["proc"] is None:
        print(f"{MUTED}  the RAG service is running elsewhere (web UI) — stop the studio "
              f"first, or switch the database there.{RESET}")
        return
    restart = _RAG["on"] or _RAG["proc"] is not None
    _stop_rag_service()
    _RAG["db"] = target
    print(f"{OK}✔ RAG database → {_rag_active_db()}{RESET}")
    if restart:
        _start_rag_service(config_path)


def do_rag_status():
    """Show the RAG toggle, active database, service state and chunk count."""
    up = _rag_service_up()
    served = _rag_display_db()            # what queries actually hit
    want = os.path.abspath(_rag_active_db())
    print(f"\n{ACCENT}{BOLD}RAG status{RESET}")
    print(f"{MUTED}  augmentation:{RESET} {'ON' if _RAG['on'] else 'off'}")
    print(f"{MUTED}  database:{RESET} {served}")
    if served != want:                    # a reused service serves a different DB
        print(f"{MUTED}  selected:{RESET} {want} {DIM}(applies once the running service stops){RESET}")
    owner = " (started by this CLI)" if _RAG["proc"] else (" (external / web UI)" if up else "")
    print(f"{MUTED}  service:{RESET} {'up' if up else 'down'}{owner}")
    print(f"{MUTED}  top-k:{RESET} {_RAG['k']}\n")


def do_rag_clear():
    """Clear the RAG database (delete its files). Only when the RAG service isn't
    running — it owns the milvus-lite file."""
    if _rag_service_up() and _RAG["proc"] is None:
        print(f"{MUTED}  the RAG service is running — use the web UI's 'Clear RAG DB' "
              f"(or stop the studio first).{RESET}")
        return
    db_path = _rag_active_db()
    try:
        ans = input(f"{ERR}  clear the RAG database (delete all ingested documents)? "
                    f"[y/N] ▸ {RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        print(f"{MUTED}  cancelled — nothing removed.{RESET}")
        return
    _stop_rag_service()                   # release our handle only after confirming
    removed = 0
    for p in (db_path, os.path.splitext(db_path)[0] + ".meta.json"):
        try:
            os.remove(p)
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"{ERR}  {exc}{RESET}")
    print(f"{OK}✔ RAG database cleared ({removed} file(s) removed).{RESET}")


def do_rag_reset(config_path):
    """Rebuild the RAG database from the bundled default document. Only when the
    RAG service isn't running (it owns the file)."""
    if _rag_service_up() and _RAG["proc"] is None:
        print(f"{MUTED}  the RAG service is running — use the web UI's 'Reset to Default' "
              f"(or stop the studio first).{RESET}")
        return
    default_md = _default_rag_source()
    if not os.path.isfile(default_md):
        print(f"{ERR}  default RAG source not found: {default_md}{RESET}")
        return
    emb = _rag_embedding_dir(config_path)
    if not emb:
        print(f"{ERR}  no RAG embedding model configured (app.rag.embedding_model_dir).{RESET}")
        return
    try:
        ans = input(f"{MUTED}  reset the RAG database to the default document? "
                    f"[y/N] ▸ {RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if ans not in ("y", "yes"):
        print(f"{MUTED}  cancelled.{RESET}")
        return
    _stop_rag_service()                   # release the file if the CLI owns the service
    print(f"{MUTED}  rebuilding from the default document (loads the embedding model — "
          f"this can take a moment)…{RESET}")
    try:
        from rag.create_db import create_markdown_vectordb
        create_markdown_vectordb(input_path=default_md, output_db=_rag_active_db(),
                                 embedding_model=emb)
        print(f"{OK}✔ RAG database reset to default.{RESET}")
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  reset failed: {exc}{RESET}")


def do_rag_inspect(arg=""):
    """Inspect the RAG database: a summary (source, chunk count, embedding model)
    plus the ingested chunks. Prefers the running VectorDB service (which owns the
    DB file); only when it is unreachable does it read the DB file directly, so we
    never open a second handle to milvus-lite. Optional ``arg`` filters chunks by
    a case-insensitive substring."""
    filt = arg.strip().lower()
    try:
        from rag.inspect_db import inspect_rag, read_rag_meta
    except Exception as exc:  # noqa: BLE001
        print(f"{ERR}  RAG inspect unavailable: {exc}{RESET}")
        return
    db_path = _rag_active_db()
    meta, docs, err = None, [], None

    # 1) Try the running VectorDB service. If it answers (even with an error) it
    #    owns the file — do NOT then read the file directly.
    service_reachable = False
    try:
        import urllib.error
        with urllib.request.urlopen(
                "http://127.0.0.1:9100/documents?limit=16383", timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        docs = data.get("documents", []) or []
        service_reachable = True
    except urllib.error.HTTPError as exc:
        service_reachable = True
        err = f"RAG service error: HTTP {exc.code}"
    except Exception:  # noqa: BLE001 - service down; fall back to a direct read
        service_reachable = False
    if service_reachable:
        served = _rag_service_db()        # the chunks came from whatever it serves
        if served:
            db_path = served              # keep the header honest about the source
        meta = read_rag_meta(db_path)
    else:
        res = inspect_rag(db_path)
        docs, err = res.get("documents", []) or [], res.get("error")
        meta = res.get("meta") or read_rag_meta(db_path)
    meta = meta or {}

    # ---- summary ----
    print(f"\n{ACCENT}{BOLD}RAG database{RESET}")
    if meta.get("input"):
        print(f"{MUTED}  source:{RESET} {meta['input']}")
    if meta.get("embedding_model"):
        print(f"{MUTED}  embedding:{RESET} {os.path.basename(str(meta['embedding_model']).rstrip('/'))}")
    print(f"{MUTED}  file:{RESET} {db_path}")
    if not os.path.isfile(db_path):
        print(f"{ERR}  no RAG database file found — upload a Markdown file to build one.{RESET}\n")
        return
    if err and not docs:
        print(f"{ERR}  {err}{RESET}")
        if meta.get("chunks"):
            print(f"{MUTED}  (the sidecar reports {meta['chunks']} chunks){RESET}")
        print()
        return

    # ---- chunk list (optionally filtered) ----
    def _hay(d):
        md = d.get("metadata") or {}
        return (str(d.get("text", "")) + " " + " ".join(str(v) for v in md.values())).lower()
    shown = [d for d in docs if filt in _hay(d)] if filt else docs
    tail = f" ({len(shown)} match '{arg.strip()}')" if filt else ""
    print(f"{MUTED}  chunks:{RESET} {len(docs)}{tail}\n")
    for i, d in enumerate(shown, 1):
        md = d.get("metadata") or {}
        headers = [str(md[k]) for k in sorted(md)
                   if str(k).lower().startswith("header") and md.get(k)]
        crumb = " › ".join(headers) if headers else "(no header)"
        text = str(d.get("text", "")).strip()
        preview = text if len(text) <= 400 else text[:400].rstrip() + "…"
        print(f"{ACCENT}{i}.{RESET} {DIM}{crumb}{RESET}")
        for line in preview.split("\n"):
            print(f"   {line}")
        print()


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
        return _startup_chat(ctrl, config_path, active, oai=oai)
    if choice == "download":
        return browse_and_download(ctrl, config_path, oai=oai) or active
    if choice == "bench":
        return _startup_benchmark(ctrl, oai, active)
    return active


def main():
    ap = argparse.ArgumentParser(description="Neat GenAI Studio terminal chat")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="", help="model to make active on start")
    ap.add_argument("--max-tokens", type=int, default=0)
    # Jump straight to an action, skipping the interactive startup menu. Each
    # takes an optional model: `--chat MODEL` loads it and chats, `--download REPO`
    # downloads it, `--benchmark MODEL` benchmarks it (bare flag prompts/menus).
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--chat", nargs="?", const="", default=None, metavar="MODEL",
                      help="go straight into chat, optionally loading MODEL first")
    mode.add_argument("--download", nargs="?", const="", default=None, metavar="REPO",
                      help="download a model (REPO, else prompt), then chat")
    mode.add_argument("--benchmark", "--bench", nargs="?", const="", default=None,
                      dest="benchmark", metavar="MODEL",
                      help="benchmark MODEL (else prompt for which), then chat")
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

    # A --chat/--download/--benchmark flag jumps straight to that action; then the
    # REPL begins. Otherwise, on an interactive start (no explicit --model), ask
    # what the user wants to do; else fall back to the load prompt when idle.
    if args.download is not None:
        repo = args.download.strip()
        active = ((download_specific(ctrl, args.config, repo, oai=oai) if repo
                   else browse_and_download(ctrl, args.config, oai=oai)) or active)
    elif args.benchmark is not None:
        active = _startup_benchmark(ctrl, oai, active, sel=(args.benchmark.strip() or None))
    elif args.chat is not None:
        model = args.chat.strip()
        active = (_load_named_model(ctrl, args.config, model, active, oai=oai) if model
                  else _startup_chat(ctrl, args.config, active, oai=oai))
    elif not args.model.strip() and sys.stdin.isatty():
        active = startup_menu(ctrl, oai, args.config, active)
    elif not active:
        active = choose_and_load_model(ctrl, args.config, oai=oai)

    hist_hint = "  ·  ↑/↓ recalls previous prompts" if (_HAS_READLINE and sys.stdin.isatty()) else ""
    print(f"{MUTED}  /help for commands, /quit to exit.{hist_hint}{RESET}\n")

    # Restore prior-session prompt history for up/down recall.
    if _HAS_READLINE:
        try:
            _readline.read_history_file(_HISTORY_FILE)
        except OSError:
            pass
        try:
            _readline.set_history_length(1000)
        except Exception:  # noqa: BLE001
            pass

    messages, system, pending_image, camera_device = [], None, None, None
    while True:
        # The prompt shows the loaded model, 🖼 if a one-shot image is queued,
        # and 📷 if the live camera is armed (a frame is grabbed every message).
        mlabel = active if active else "no model"
        marks = ""
        if pending_image:
            marks += f" {_rl(ACCENT)}🖼{_rl(RESET)}"
        if camera_device is not None:
            marks += f" {_rl(ACCENT)}📷{_rl(RESET)}"
        try:
            line = input(f"{_rl(ACCENT)}{_rl(BOLD)}you{_rl(RESET)} "
                         f"{_rl(DIM)}[{mlabel}]{_rl(RESET)}{marks} "
                         f"{_rl(ACCENT)}{_rl(BOLD)}▸{_rl(RESET)} ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print(f"\n{MUTED}(Ctrl+D or /quit to exit){RESET}")
            continue
        if not line:
            continue
        # Record the main chat prompt for up/down recall (sub-prompts are excluded).
        if _HAS_READLINE and _MANUAL_HISTORY:
            _readline.add_history(line)

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
            elif cmd in ("delete", "rm", "remove"):
                # Delete a model's weights from disk (server unloads it first if
                # resident and refuses the pinned ASR model). Irreversible, so
                # pick from a menu when no name is given and always confirm.
                cat = catalog(ctrl)
                if arg:
                    name = arg
                    if name not in [m.get("name") for m in cat]:
                        print(f"{ERR}  no model named '{name}' in the catalog — /models to list.{RESET}")
                        continue
                else:
                    items = [
                        (f"{m.get('name')}  [{type_label(m.get('type'))}]"
                         f"{'  ● loaded' if m.get('loaded') else ''}", m.get("name"))
                        for m in cat if m.get("type", "chat") != "asr"]
                    if not items:
                        print(f"{MUTED}  no deletable models in the catalog.{RESET}")
                        continue
                    name = select_menu(items, "Delete which model from disk?")
                    if not name:
                        continue
                try:
                    ans = input(f"{ERR}  delete '{name}' from disk? this removes the "
                                f"weights and cannot be undone [y/N] ▸ {RESET}").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
                if ans not in ("y", "yes"):
                    print(f"{MUTED}  cancelled — nothing deleted.{RESET}")
                    continue
                try:
                    ctrl_post(ctrl, "/control/delete", {"name": name})
                    print(f"{OK}✔ deleted {name} from disk.{RESET}")
                    if name == active:
                        active, messages, pending_image = "", [], None
                        camera_device = None   # no model → live camera can't send
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", "replace")
                    try:
                        detail = json.loads(detail).get("error", detail)
                    except Exception:  # noqa: BLE001
                        pass
                    print(f"{ERR}  delete failed: {detail[:200]}{RESET}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{ERR}  {exc}{RESET}")
            elif cmd == "system":
                system = arg or None
                messages = []
                print(f"{OK}✔ system prompt {'set' if system else 'cleared'}.{RESET}")
            elif cmd in ("new", "clear"):
                messages = []
                print(f"{OK}✔ conversation cleared.{RESET}")
            elif cmd in ("export", "save"):
                if not messages:
                    print(f"{MUTED}  nothing to export yet — have a conversation first.{RESET}")
                    continue
                path = arg.strip() if arg else f"neat-chat-{time.strftime('%Y%m%d-%H%M%S')}.log"
                path = os.path.expanduser(path)
                if not path.lower().endswith(".log"):
                    path += ".log"
                try:
                    write_chat_log(path, active, system, messages)
                    turns = sum(1 for m in messages if m.get("role") == "user")
                    print(f"{OK}✔ exported {turns} turn(s) to {path}{RESET}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{ERR}  export failed: {exc}{RESET}")
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
            elif cmd in ("rag", "docs"):
                parts = arg.split(None, 1)
                sub = parts[0].lower() if parts else ""
                rest = parts[1].strip() if len(parts) > 1 else ""
                if sub == "clear":
                    do_rag_clear()
                elif sub == "reset":
                    do_rag_reset(args.config)
                elif sub == "on":
                    do_rag_on(args.config)
                elif sub == "off":
                    do_rag_off()
                elif sub in ("search", "query", "q"):
                    do_rag_search_cmd(rest, args.config)
                elif sub == "db":
                    do_rag_db(rest, args.config)
                elif sub == "status":
                    do_rag_status()
                else:
                    do_rag_inspect(arg)
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
        # RAG augmentation: retrieve context and prepend it to *this* turn's text.
        # History (below) keeps the clean prompt so context isn't re-fed each turn.
        sent_text = line
        if _RAG["on"]:
            try:
                hits = _rag_search(line, k=_RAG["k"])
            except KeyboardInterrupt:
                print(f"\n{MUTED}  (RAG retrieval cancelled){RESET}")
                continue
            if hits is None:
                print(f"{ERR}  📚 RAG retrieval failed — the service is unreachable "
                      f"(/rag status; /rag on to restart). Asking without context.{RESET}")
            else:
                block, used = _rag_context_block(hits)
                if used:
                    sent_text = (f"{line}\n\nUse the following retrieved context to answer "
                                 f"if it is relevant:\n\n{block}")
                    print(f"{MUTED}  📚 added {used} RAG passage(s) as context.{RESET}")
                else:
                    print(f"{MUTED}  📚 RAG on, but no passages matched — asking without context.{RESET}")
        if turn_image is not None:
            user_content = [{"type": "text", "text": sent_text},
                            {"type": "image", "image": turn_image}]
        else:
            user_content = sent_text
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
    _stop_rag_service()                   # stop the RAG worker the CLI may have started
    if _HAS_READLINE:
        try:
            _readline.write_history_file(_HISTORY_FILE)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
