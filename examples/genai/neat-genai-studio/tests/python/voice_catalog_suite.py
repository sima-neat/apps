import hashlib
import http.client
import io
import json
import ssl
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

import voice_catalog

from voice_catalog import (
    VoiceCatalogError,
    installation_plan,
    load_catalog,
    voice_by_id,
)


def test_catalog_has_simple_licenses_and_pinned_sources():
    catalog = load_catalog()
    assert catalog["voices"]
    assert all(isinstance(v["license"], str) for v in catalog["voices"])
    assert all(len(v["revision"]) == 40 for v in catalog["voices"])
    assert voice_by_id("css10", catalog)["default"] is True


def test_only_cc_by_nc_sa_voices_are_excluded():
    catalog = load_catalog()
    ids = {voice["id"] for voice in catalog["voices"]}
    assert "ja_JA-hi_fi_captain-medium" not in ids
    assert "ko_KR-kss-medium" not in ids
    assert "en_US-hfc_female-medium" not in ids
    assert "en_US-hfc_male-medium" not in ids
    assert "zh_CN-huayan-medium" in ids
    assert "zh_CN-chaowen-medium" in ids
    assert voice_by_id("zh_CN-huayan-medium", catalog)["license"] == "Unknown"


def test_korean_has_no_server_install_plan():
    assert installation_plan(["ko"], catalog=load_catalog()) == []


def test_chinese_has_default_and_optional_dedicated_voices():
    catalog = load_catalog()
    default_ids = {v["id"] for v in installation_plan(["zh"], catalog=catalog)}
    selected_ids = {
        v["id"]
        for v in installation_plan(
            ["zh"], optional_ids=["zh_CN-chaowen-medium"], catalog=catalog
        )
    }
    assert "zh_CN-huayan-medium" in default_ids
    assert "zh_CN-chaowen-medium" not in default_ids
    assert "zh_CN-chaowen-medium" in selected_ids


def test_optional_voices_require_explicit_selection():
    catalog = load_catalog()
    default_ids = {v["id"] for v in installation_plan(["en"], catalog=catalog)}
    selected_ids = {
        v["id"]
        for v in installation_plan(
            ["en"], optional_ids=["mera", "en_US-ljspeech-medium"], catalog=catalog
        )
    }
    assert "css10" in default_ids
    assert "en_US-kristin-medium" in default_ids
    assert "mera" not in default_ids
    assert "en_US-ljspeech-medium" not in default_ids
    assert {"mera", "en_US-ljspeech-medium"}.issubset(selected_ids)


def test_catalog_rejects_blocked_license(tmp_path):
    catalog = load_catalog()
    catalog["voices"][0]["license"] = "CC BY-NC-SA 4.0"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(VoiceCatalogError, match="blocked license"):
        load_catalog(path)


def test_repair_completes_only_partly_installed_voices(tmp_path, monkeypatch):
    # A git update removed thorsten's config while its downloaded model stayed:
    # repair fetches the config again. Paola was never installed: left alone.
    catalog = load_catalog()
    thorsten = voice_by_id("de_DE-thorsten-medium", catalog)
    model, config = voice_catalog.asset_paths(thorsten, tmp_path)
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    fetched = []

    def fake_download(url, target, expected_sha256):
        fetched.append(Path(target).name)
        Path(target).write_bytes(b"x")

    monkeypatch.setattr(voice_catalog, "_download", fake_download)
    assert voice_catalog.repair_voices(assets_path=tmp_path, catalog=catalog) == ["de_DE-thorsten-medium"]
    assert config.name in fetched
    assert not any("paola" in name for name in fetched)
    assert voice_catalog.repair_voices(assets_path=tmp_path, catalog=catalog) == []
    # A config on its own (older packages shipped Norwegian's) installs nothing.
    norwegian = voice_by_id("no_NO-talesyntese-medium", catalog)
    voice_catalog.asset_paths(norwegian, tmp_path)[1].write_bytes(b"config only")
    fetched.clear()
    assert voice_catalog.repair_voices(assets_path=tmp_path, catalog=catalog) == []
    assert fetched == []


def _http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError("https://hf.test/config.json", code, "err", headers, None)


def _serve(monkeypatch, replies):
    """Make urlopen answer each call with the next reply: bytes, or an exception."""
    calls, sleeps = [], []

    def urlopen(url, timeout=None):
        calls.append(timeout)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply)

    monkeypatch.setattr(voice_catalog.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(voice_catalog, "_sleep", sleeps.append)
    return calls, sleeps


def test_download_retries_a_rate_limit_and_honours_retry_after(monkeypatch, tmp_path):
    body = b"voice config"
    calls, sleeps = _serve(monkeypatch, [_http_error(429, retry_after=7), body])
    target = tmp_path / "config.json"
    voice_catalog._download("https://hf.test/config.json", target, hashlib.sha256(body).hexdigest())
    assert target.read_bytes() == body
    assert sleeps == [7]
    assert calls == [voice_catalog._DOWNLOAD_TIMEOUT_S] * 2


def test_download_retries_a_tls_connection_dropped_mid_transfer(monkeypatch, tmp_path):
    body = b"voice config"

    class Dropped(io.BytesIO):
        def read(self, *args):
            raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")

    replies, sleeps = [Dropped(b""), io.BytesIO(body)], []
    monkeypatch.setattr(voice_catalog.urllib.request, "urlopen", lambda url, timeout=None: replies.pop(0))
    monkeypatch.setattr(voice_catalog, "_sleep", sleeps.append)
    target = tmp_path / "config.json"
    voice_catalog._download("https://hf.test/config.json", target, hashlib.sha256(body).hexdigest())
    assert target.read_bytes() == body
    assert sleeps == [2]


def test_download_does_not_retry_a_missing_file(monkeypatch, tmp_path):
    calls, sleeps = _serve(monkeypatch, [_http_error(404)])
    with pytest.raises(VoiceCatalogError, match="could not download config.json"):
        voice_catalog._download("https://hf.test/config.json", tmp_path / "config.json", "0" * 64)
    assert (len(calls), sleeps) == (1, [])
    assert list(tmp_path.iterdir()) == []


def test_download_gives_up_after_the_last_attempt_and_says_how_to_resume(monkeypatch, tmp_path):
    attempts = voice_catalog._DOWNLOAD_ATTEMPTS
    errors = ([_http_error(503), http.client.IncompleteRead(b"partial")] + [_http_error(503)] * attempts)[: attempts - 1]
    errors.append(urllib.error.URLError("connection reset"))
    calls, sleeps = _serve(monkeypatch, errors)
    with pytest.raises(VoiceCatalogError, match="Run setup.sh again to resume"):
        voice_catalog._download("https://hf.test/config.json", tmp_path / "config.json", "0" * 64)
    assert len(calls) == attempts
    assert sleeps == [2, 4, 8][: attempts - 1]
    assert list(tmp_path.iterdir()) == []


def test_install_reports_a_failed_download_without_a_traceback(monkeypatch, tmp_path, capsys):
    _serve(monkeypatch, [_http_error(403)] * 50)
    assert voice_catalog.main(["install", "--languages", "en", "--assets", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "❌ could not download" in out
    assert "Traceback" not in out
