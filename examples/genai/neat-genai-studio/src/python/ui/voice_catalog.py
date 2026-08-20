"""Reviewed TTS voice catalog and verified downloader.

The catalog is deliberately local and commit-pinned. Runtime discovery ignores
every model not present in this allowlist, including stale files left by older
Studio versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path


CATALOG_PATH = Path(__file__).with_name("voice_catalog.json")
ASSETS_PATH = Path(__file__).with_name("assets")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


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
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
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
    install = subparsers.add_parser("install")
    install.add_argument("--languages", required=True)
    install.add_argument("--optional", default="")
    install.add_argument("--assets", type=Path, default=ASSETS_PATH)
    args = parser.parse_args(argv)

    catalog = load_catalog()
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
