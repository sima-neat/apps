"""Reviewed TTS voice catalog and verified downloader.

The catalog is deliberately local and commit-pinned. Runtime discovery ignores
every model not present in this allowlist, including stale files left by older
Studio versions.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("voice_catalog.json")
ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"  # voice assets stay in api/assets
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# Hugging Face rate-limits anonymous downloads per IP address, and an office
# network shares one, so a busy moment answers 429. Retry that, server errors
# and dropped connections a few times before giving up.
_DOWNLOAD_TIMEOUT_S = 60
_DOWNLOAD_ATTEMPTS = 4
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRY_WAIT_S = 30
_sleep = time.sleep
# ssl.SSLError: an HTTPS connection dropped mid-transfer (it is not a ConnectionError).
_TRANSIENT_ERRORS = (
    urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError, ssl.SSLError,
)


class VoiceCatalogError(ValueError):
    pass


def load_catalog(path=CATALOG_PATH):
    with Path(path).open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("schema_version") != 1:
        raise VoiceCatalogError("unsupported voice catalog schema")

    blocked_licenses = {
        re.sub(r"[^A-Z0-9]+", "-", str(value).strip().upper()).strip("-")
        for value in catalog.get("policy", {}).get("blocked_licenses", [])
    }
    ids = set()
    for voice in catalog.get("voices", []):
        voice_id = voice.get("id")
        if not voice_id or voice_id in ids:
            raise VoiceCatalogError(f"invalid or duplicate voice id: {voice_id!r}")
        ids.add(voice_id)
        if voice.get("engine") not in {"piper-plus", "piper-tts"}:
            raise VoiceCatalogError(f"unsupported engine for {voice_id}")
        license_name = voice.get("license")
        if not isinstance(license_name, str) or not license_name.strip():
            raise VoiceCatalogError(f"missing license for {voice_id}")
        normalized_license = re.sub(
            r"[^A-Z0-9]+", "-", license_name.strip().upper()
        ).strip("-")
        if any(blocked in normalized_license for blocked in blocked_licenses):
            raise VoiceCatalogError(f"blocked license for {voice_id}: {license_name}")
        if not _REVISION_RE.fullmatch(str(voice.get("revision", ""))):
            raise VoiceCatalogError(f"voice revision must be a pinned commit: {voice_id}")
        for file_info in voice.get("files", []):
            if not _SHA256_RE.fullmatch(str(file_info.get("sha256", ""))):
                raise VoiceCatalogError(f"invalid checksum for {voice_id}")
    return catalog


def catalog_voices(catalog=None, *, engine=None, language=None):
    voices = (catalog or load_catalog()).get("voices", [])
    return [
        voice for voice in voices
        if (engine is None or voice["engine"] == engine)
        and (language is None or language in voice["languages"])
    ]


def voice_by_id(voice_id, catalog=None):
    return next(
        (voice for voice in catalog_voices(catalog) if voice["id"] == voice_id),
        None,
    )


def asset_paths(voice, assets_path=ASSETS_PATH):
    base = Path(assets_path) / voice.get("asset_dir", ".")
    return [base / file_info["target"] for file_info in voice["files"]]


def installed_voices(catalog=None, *, assets_path=ASSETS_PATH, engine=None, language=None):
    return [
        voice for voice in catalog_voices(catalog, engine=engine, language=language)
        if all(path.is_file() for path in asset_paths(voice, assets_path))
    ]


def installation_plan(languages, optional_ids=(), catalog=None):
    wanted = set(languages)
    optional_ids = set(optional_ids)
    result = []
    for voice in catalog_voices(catalog):
        if not wanted.intersection(voice["languages"]):
            continue
        if voice.get("optional") and voice["id"] not in optional_ids:
            continue
        result.append(voice)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_wait(error, attempt):
    """Seconds to wait before the next attempt, or None when not worth retrying."""
    if isinstance(error, urllib.error.HTTPError):
        if error.code not in _RETRY_STATUSES:
            return None
        retry_after = (error.headers or {}).get("Retry-After", "")
        if retry_after.strip().isdigit():
            return min(int(retry_after), _MAX_RETRY_WAIT_S)
    elif not isinstance(error, _TRANSIENT_ERRORS):
        return None
    return min(2 ** attempt, _MAX_RETRY_WAIT_S)


def _fetch(url, temp_path):
    name = Path(urllib.parse.urlsplit(url).path).name
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response, \
                    Path(temp_path).open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            return
        except _TRANSIENT_ERRORS as error:
            wait = _retry_wait(error, attempt)
            if wait is None or attempt == _DOWNLOAD_ATTEMPTS:
                raise VoiceCatalogError(
                    f"could not download {name} ({error}). Run setup.sh again to "
                    "resume; files already verified are kept."
                ) from error
            print(f"⏳ {name}: {error}; retrying in {wait}s "
                  f"(attempt {attempt + 1} of {_DOWNLOAD_ATTEMPTS})")
            _sleep(wait)


def _download(url, target, expected_sha256):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _sha256(target) == expected_sha256:
        print(f"✅ verified: {target}")
        return

    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        print(f"⬇️  downloading: {target.name}")
        _fetch(url, temp_path)
        actual = _sha256(temp_path)
        if actual != expected_sha256:
            raise VoiceCatalogError(
                f"checksum mismatch for {target.name}: {actual} != {expected_sha256}"
            )
        os.replace(temp_path, target)
        print(f"✅ installed: {target}")
    finally:
        temp_path.unlink(missing_ok=True)


def install_voices(languages, optional_ids=(), *, assets_path=ASSETS_PATH, catalog=None):
    catalog = catalog or load_catalog()
    plan = installation_plan(languages, optional_ids, catalog)
    for voice in plan:
        print(
            f"\n📦 {voice['label']} [{voice['license']}] "
            f"from {voice['repository']}@{voice['revision'][:12]}"
        )
        base_url = (
            f"https://huggingface.co/{voice['repository']}/resolve/{voice['revision']}"
        )
        targets = asset_paths(voice, assets_path)
        for file_info, target in zip(voice["files"], targets):
            source = file_info["source"]
            _download(f"{base_url}/{source}?download=true", target, file_info["sha256"])
    return plan


def repair_voices(*, assets_path=ASSETS_PATH, catalog=None):
    """Complete every installed voice that is missing a file, e.g. a config a
    git update removed, and verify its files. A voice counts as installed when
    its model files (everything but the .json configs) are there; a config on
    its own, as older packages shipped, does not install the voice. Returns
    the ids of the voices completed."""
    repaired = []
    for voice in catalog_voices(catalog or load_catalog()):
        paths = asset_paths(voice, assets_path)
        models = [path for path in paths if not path.name.endswith(".json")]
        if models and all(path.is_file() for path in models) and not all(path.is_file() for path in paths):
            install_voice(voice, assets_path=assets_path)
            repaired.append(voice["id"])
    return repaired


def install_voice(voice, *, assets_path=ASSETS_PATH):
    """Install one already-validated catalog entry."""
    base_url = (
        f"https://huggingface.co/{voice['repository']}/resolve/{voice['revision']}"
    )
    targets = asset_paths(voice, assets_path)
    for file_info, target in zip(voice["files"], targets):
        _download(
            f"{base_url}/{file_info['source']}?download=true",
            target,
            file_info["sha256"],
        )
    return targets


def _codes(value):
    if value.strip().lower() in {"", "none", "off"}:
        return []
    return [item for item in re.split(r"[\s,]+", value.strip()) if item]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    repair = subparsers.add_parser("repair")
    repair.add_argument("--assets", type=Path, default=ASSETS_PATH)
    install = subparsers.add_parser("install")
    install.add_argument("--languages", required=True)
    install.add_argument("--optional", default="")
    install.add_argument("--assets", type=Path, default=ASSETS_PATH)
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except VoiceCatalogError as error:
        print(f"❌ {error}")
        return 1


def _run(args):
    catalog = load_catalog()
    if args.command == "repair":
        repaired = repair_voices(assets_path=args.assets, catalog=catalog)
        print(f"Restored missing files for: {', '.join(repaired)}" if repaired else "Installed voices are complete.")
        return 0
    if args.command == "validate":
        print(f"voice catalog valid: {len(catalog['voices'])} reviewed voices")
        return 0

    languages = _codes(args.languages)
    optional_ids = _codes(args.optional)
    plan = install_voices(languages, optional_ids, assets_path=args.assets, catalog=catalog)
    covered = {lang for voice in plan for lang in voice["languages"]}
    for language in sorted(set(languages) - covered):
        print(f"⚠️  no catalogued server-side voice for '{language}'; browser/text only")
    print(f"\n✅ Installed/verified {len(plan)} catalogued voice(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
