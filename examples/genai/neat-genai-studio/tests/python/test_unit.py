"""Unit tests for neat-genai-studio (Python).

The suites live beside this module as ``*_suite.py`` (deliberately not
``test_*.py``, so an ad-hoc ``pytest`` over the tree does not collect them a
second time). The installed bundle packages ``src/python`` wholesale, so tests
must not live there. This module is what the repository harness collects: it
puts ``src/python`` and ``src/python/ui`` on the path, re-exports the TestCase
classes, and marks them as unit coverage so ``./tests/test.sh --unit`` runs
them. Run one suite in place with ``python -m unittest asr_switching_suite``
from this directory (with the same two paths on PYTHONPATH).

No hardware, no model downloads, no live server — the ASR switching suite drives
a fake GenAIServer against model directories built in a temp dir.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_PYTHON = EXAMPLE_DIR / "src" / "python"

if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

UI_PYTHON = SRC_PYTHON / "ui"
if str(UI_PYTHON) not in sys.path:
    sys.path.insert(0, str(UI_PYTHON))   # the ui suites import their modules bare

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from asr_switching_suite import (  # noqa: E402
    AsrSwitchingTests,
    AsrWarmupBehaviourTests,
    AsrWarmupPayloadTests,
    EncoderLayoutIsNotJudgedLocallyTests,
    MlaFailureClassificationTests,
)
from hub_security_suite import HubPathSecurityTests  # noqa: E402,F401
from cli_think_suite import (  # noqa: E402,F401
    NoThinkRewriteTests as CliNoThinkRewriteTests,
    ResetDisconnectClassificationTests as CliResetDisconnectClassificationTests,
    StreamTokenCountTests as CliStreamTokenCountTests,
    ThinkSplitterTests as CliThinkSplitterTests,
)
from asr_metadata_suite import AsrMetadataTests  # noqa: E402,F401
from audio_api_suite import (  # noqa: E402,F401
    FormatTranscriptionTests as AudioApiFormatTranscriptionTests,
    SpeechRequestTests as AudioApiSpeechRequestTests,
    TranscriptionFormTests as AudioApiTranscriptionFormTests,
    VoicesListingTests as AudioApiVoicesListingTests,
)
from shell_config_suite import ShellConfigValueTests, ShellWebConfigTests  # noqa: E402,F401
from backend_mode_suite import (  # noqa: E402,F401
    BackendPathTests,
    CorsPolicyTests as BackendCorsPolicyTests,
    HealthPayloadTests as BackendHealthPayloadTests,
)
from supertonic_tts_suite import (  # noqa: E402,F401
    ClientConfigurationTests as SupertonicClientConfigurationTests,
    DurationFallbackTests as SupertonicDurationFallbackTests,
    EnvironmentDiscoveryTests as SupertonicEnvironmentDiscoveryTests,
    SegmentTextTests as SupertonicSegmentTextTests,
    VendoredRuntimeTests as SupertonicVendoredRuntimeTests,
)
from voice_catalog_suite import (  # noqa: E402,F401
    test_catalog_has_simple_licenses_and_pinned_sources,
    test_catalog_rejects_blocked_license,
    test_chinese_has_default_and_optional_dedicated_voices,
    test_korean_has_no_server_install_plan,
    test_only_cc_by_nc_sa_voices_are_excluded,
    test_optional_voices_require_explicit_selection,
)

# Applies to every TestCase collected from this module.
pytestmark = pytest.mark.unit


@pytest.mark.unit
def test_ui_config_reads_backend_settings(tmp_path) -> None:
    """app.web.headless and the persisted backend CORS allowlist (string or list)."""
    from shared.config import load_ui_config

    plain = tmp_path / "plain.yaml"
    plain.write_text("app:\n  web:\n    port: 5000\n", encoding="utf-8")
    cfg = load_ui_config(plain, tmp_path)
    assert (cfg.web.headless, cfg.web.cors_origins) == (False, "")

    text = tmp_path / "text.yaml"
    text.write_text('app:\n  web:\n    port: 5000\n    headless: true\n'
                    '    cors_origins: "http://a:3000, https://b"\n', encoding="utf-8")
    cfg = load_ui_config(text, tmp_path)
    assert (cfg.web.headless, cfg.web.cors_origins) == (True, "http://a:3000, https://b")

    listed = tmp_path / "list.yaml"
    listed.write_text("app:\n  web:\n    port: 5000\n    cors_origins:\n      - http://a:3000\n      - '*'\n",
                      encoding="utf-8")
    assert load_ui_config(listed, tmp_path).web.cors_origins == "http://a:3000,*"


def test_ui_config_reads_supertonic_paths(tmp_path) -> None:
    """app.tts.supertonic persists the machine-specific Supertonic models root
    that setup.sh wrote; a pre-vendoring app_root maps to its models/ subdir and
    defaults apply when the section is absent."""
    from shared.config import load_ui_config

    base = "app:\n  web:\n    port: 5000\n"
    with_paths = tmp_path / "with.yaml"
    with_paths.write_text(
        base + "  tts:\n    supertonic:\n      models_root: /data/st-models\n", encoding="utf-8")
    cfg = load_ui_config(with_paths, tmp_path)
    assert cfg.supertonic.models_root == "/data/st-models"
    assert cfg.supertonic.venv == ""

    custom_venv = tmp_path / "venv.yaml"
    custom_venv.write_text(
        base + "  tts:\n    supertonic:\n      models_root: /data/st-models\n"
        "      venv: /data/st-venv\n", encoding="utf-8")
    assert load_ui_config(custom_venv, tmp_path).supertonic.venv == "/data/st-venv"

    # A config written before the runtime was vendored named the parent dir.
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        base + "  tts:\n    supertonic:\n      repo_root: /data/st-repo\n"
        "      app_root: /data/st-app\n", encoding="utf-8")
    cfg = load_ui_config(legacy, tmp_path)
    assert cfg.supertonic.models_root == "/data/st-app/models"
    assert cfg.supertonic.venv == ""          # no <app_root>/.venv on this host

    # ...and its runtime venv is carried over while it still exists.
    app_root = tmp_path / "st-app"
    (app_root / ".venv" / "bin").mkdir(parents=True)
    (app_root / ".venv" / "bin" / "python").write_text("")
    legacy_venv = tmp_path / "legacy-venv.yaml"
    legacy_venv.write_text(base + f"  tts:\n    supertonic:\n      app_root: {app_root}\n", encoding="utf-8")
    assert load_ui_config(legacy_venv, tmp_path).supertonic.venv == str(app_root / ".venv")

    without = tmp_path / "without.yaml"
    without.write_text(base, encoding="utf-8")
    cfg = load_ui_config(without, tmp_path)
    assert cfg.supertonic.models_root == ""   # unset: the runtime default decides


@pytest.mark.unit
def test_tts_text_sanitizer() -> None:
    """Run the TTS sanitizer suite.

    It is a script rather than a TestCase — it asserts by printing and calling
    sys.exit — so run it as a subprocess and check the exit code instead of
    importing it, which would execute it at collection time.
    """
    script = TESTS_DIR / "tts_text_check.py"
    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

__all__ = [
    "AsrSwitchingTests",
    "AsrWarmupBehaviourTests",
    "AsrWarmupPayloadTests",
    "EncoderLayoutIsNotJudgedLocallyTests",
    "MlaFailureClassificationTests",
    "HubPathSecurityTests",
    "CliNoThinkRewriteTests",
    "CliResetDisconnectClassificationTests",
    "CliStreamTokenCountTests",
    "CliThinkSplitterTests",
    "AsrMetadataTests",
    "AudioApiFormatTranscriptionTests",
    "AudioApiSpeechRequestTests",
    "AudioApiTranscriptionFormTests",
    "AudioApiVoicesListingTests",
    "SupertonicClientConfigurationTests",
    "SupertonicDurationFallbackTests",
    "SupertonicEnvironmentDiscoveryTests",
    "SupertonicSegmentTextTests",
    "SupertonicVendoredRuntimeTests",
]
