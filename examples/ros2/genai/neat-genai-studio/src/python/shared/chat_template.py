"""Self-heal chat-template files in model directories.

The LLiMa runtime picks the first chat template it finds, in this order:
``devkit/chat_template.jinja``, ``devkit/chat_template.json``, then the legacy
embedded ``devkit/tokenizer_config.json["chat_template"]``. Two things break
this with repos updated on the Hugging Face Hub:

* A repo's ``chat_template.jinja`` holds the Hub's HTML *blob viewer page*
  (saved from a ``/blob/`` URL instead of ``/resolve/``), so minja fails with
  ``Expected value expression at row ..., column ...`` while parsing HTML.
* Newer ``transformers`` saves the template as a standalone
  ``chat_template.jinja`` at the repo root and drops the key from
  ``tokenizer_config.json`` — but the runtime only looks inside ``devkit/``.
* Updated templates use Jinja that real Jinja2 accepts but minja cannot parse
  — notably adjacent string-literal concatenation (``"a"  "b"``), which minja
  rejects with ``Expected closing parenthesis in call args at row ...``.

``repair_chat_template_files`` fixes all of these in place so old- and
new-layout repos load the same way: corrupt (HTML) template files are renamed
aside so the runtime falls through to the older embedded template, surviving
templates are rewritten into minja-parseable form (originals backed up), and
when ``devkit/`` ends up with no usable template a valid root-level one is
mirrored into it. Stdlib only, safe to call from both the UI and model-server
processes.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

# An HTML document declares itself within the first couple of KB; the Hub's
# blob-viewer markers can appear anywhere in its (single-line) body.
_HTML_PREFIXES = ("<!doctype", "<html", "<head", "<body", "<meta")
_HUB_PAGE_MARKERS = ("svelte_hydrater", "viewerblobpage", "data-target=")


def _looks_like_html(text: str) -> bool:
    head = (text or "").lstrip()[:2048].lower()
    if any(head.startswith(prefix) for prefix in _HTML_PREFIXES):
        return True
    body = (text or "")[:262144].lower()
    return any(marker in body for marker in _HUB_PAGE_MARKERS)


def looks_like_chat_template(text: str | None) -> bool:
    """Plausibly a Jinja chat template: has Jinja tags and is not an HTML page."""
    if not text or _looks_like_html(text):
        return False
    return "{{" in text or "{%" in text


# A {{ ... }} or {% ... %} block (statements and expressions; comments hold no
# string literals worth merging). Non-greedy: assumes no literal "}}" inside.
_JINJA_BLOCK_RE = re.compile(r"{[{%].*?[%}]}", re.S)
# Two adjacent string literals of the same quote kind, e.g. "a"  "b" or 'a' 'b'
# (Python/Jinja2-style implicit concatenation, split across lines or not).
_ADJACENT_DQ_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s+"((?:[^"\\]|\\.)*)"')
_ADJACENT_SQ_RE = re.compile(r"'((?:[^'\\]|\\.)*)'\s+'((?:[^'\\]|\\.)*)'")


def _merge_adjacent_strings(block: str) -> str:
    """Collapse implicit string-literal concatenation into single literals."""
    prev = None
    while prev != block:
        prev = block
        block = _ADJACENT_DQ_RE.sub(lambda m: f'"{m.group(1)}{m.group(2)}"', block)
        block = _ADJACENT_SQ_RE.sub(lambda m: f"'{m.group(1)}{m.group(2)}'", block)
    return block


def normalize_template_for_runtime(text: str) -> str:
    """Rewrite Jinja constructs the runtime's minja parser rejects.

    Currently: adjacent string-literal concatenation inside ``{{ }}`` / ``{% %}``
    blocks (Jinja2 merges them like Python; minja errors out). Literal template
    text outside Jinja blocks is never touched. Idempotent — a no-op for
    templates that were already parseable.
    """
    if not text:
        return text
    return _JINJA_BLOCK_RE.sub(lambda m: _merge_adjacent_strings(m.group(0)), text)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _rewrite_with_backup(path: Path, new_text: str, actions: list[str]) -> None:
    """Replace a file's content, keeping the original next to it as ``.orig``."""
    backup = path.with_name(path.name + ".orig")
    try:
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(new_text, encoding="utf-8")
        actions.append(
            f"rewrote {path.parent.name}/{path.name} for the runtime's jinja parser "
            "(merged split string literals)")
    except OSError as exc:
        actions.append(f"could not rewrite {path}: {exc}")


def _quarantine(path: Path, actions: list[str]) -> None:
    """Rename a bad template file aside so the runtime falls back past it."""
    target = path.with_name(path.name + ".corrupt")
    try:
        if target.exists():
            target.unlink()
        path.rename(target)
        actions.append(
            f"set aside {path.parent.name}/{path.name} (HTML page, not a chat template)")
    except OSError as exc:
        actions.append(f"could not set aside {path}: {exc}")


def _embedded_template(config_path: Path) -> str | None:
    """The chat_template string embedded in a tokenizer_config.json, if any."""
    text = _read_text(config_path)
    if text is None:
        return None
    try:
        value = json.loads(text).get("chat_template")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def _has_usable_template(devkit: Path) -> bool:
    if (devkit / "chat_template.jinja").is_file():
        return True
    if (devkit / "chat_template.json").is_file():
        return True
    return looks_like_chat_template(_embedded_template(devkit / "tokenizer_config.json"))


def repair_chat_template_files(model_dir) -> list[str]:
    """Sanitize a model directory's chat-template files in place.

    Returns human-readable descriptions of what was changed (empty when the
    directory was already fine). Never raises for I/O problems — a directory
    it cannot fix is left for the runtime to report.
    """
    model_dir = Path(model_dir)
    actions: list[str] = []
    devkit = model_dir / "devkit"

    # 1) Quarantine template files that are actually HTML pages, wherever they
    #    sit, so minja never sees them and the runtime's fallback order works.
    #    Survivors are rewritten into minja-parseable form (original kept).
    for base in (devkit, model_dir):
        if not base.is_dir():
            continue
        jinja_file = base / "chat_template.jinja"
        if jinja_file.is_file():
            text = _read_text(jinja_file)
            if text is not None and not looks_like_chat_template(text):
                _quarantine(jinja_file, actions)
            elif text is not None:
                fixed = normalize_template_for_runtime(text)
                if fixed != text:
                    _rewrite_with_backup(jinja_file, fixed, actions)
        json_file = base / "chat_template.json"
        if json_file.is_file():
            text = _read_text(json_file)
            template = None
            if text is not None:
                try:
                    value = json.loads(text).get("chat_template")
                    template = value if isinstance(value, str) else None
                except (json.JSONDecodeError, AttributeError):
                    template = None
            if text is not None and not looks_like_chat_template(template):
                _quarantine(json_file, actions)
            elif template is not None:
                fixed = normalize_template_for_runtime(template)
                if fixed != template:
                    try:
                        data = json.loads(text)
                        data["chat_template"] = fixed
                        _rewrite_with_backup(
                            json_file,
                            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                            actions)
                    except (json.JSONDecodeError, OSError):
                        pass
        # The legacy embedded template gets the same minja normalization.
        config_file = base / "tokenizer_config.json"
        embedded = _embedded_template(config_file)
        if embedded is not None and looks_like_chat_template(embedded):
            fixed = normalize_template_for_runtime(embedded)
            if fixed != embedded:
                try:
                    data = json.loads(_read_text(config_file) or "")
                    data["chat_template"] = fixed
                    _rewrite_with_backup(
                        config_file,
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        actions)
                except (json.JSONDecodeError, OSError):
                    pass

    if not devkit.is_dir() or _has_usable_template(devkit):
        return actions

    # 2) devkit/ has no usable template left. Bridge the new repo layout: a
    #    valid template at the repo root gets mirrored to where the runtime looks.
    for name in ("chat_template.jinja", "chat_template.json"):
        src = model_dir / name
        if src.is_file():  # survived the quarantine pass above, so it's valid
            try:
                shutil.copy2(src, devkit / name)
                actions.append(f"copied {name} into devkit/ (new Hub repo layout)")
                return actions
            except OSError as exc:
                actions.append(f"could not copy {name} into devkit/: {exc}")

    # 3) Last resort: recover the legacy template embedded in the repo root's
    #    tokenizer_config.json into devkit's copy (which the runtime reads).
    root_template = _embedded_template(model_dir / "tokenizer_config.json")
    devkit_config = devkit / "tokenizer_config.json"
    if looks_like_chat_template(root_template) and devkit_config.is_file():
        text = _read_text(devkit_config)
        try:
            config = json.loads(text) if text else None
        except json.JSONDecodeError:
            config = None
        if isinstance(config, dict):
            try:
                shutil.copy2(devkit_config, devkit_config.with_suffix(".json.orig"))
                config["chat_template"] = root_template
                devkit_config.write_text(
                    json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
                actions.append(
                    "restored legacy chat_template into devkit/tokenizer_config.json")
            except OSError as exc:
                actions.append(f"could not update devkit/tokenizer_config.json: {exc}")
    return actions
