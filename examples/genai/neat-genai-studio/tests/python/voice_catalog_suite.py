import json
from pathlib import Path

import pytest

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
