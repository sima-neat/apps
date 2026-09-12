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
    MlaFailureClassificationTests,
)
from hub_security_suite import HubPathSecurityTests  # noqa: E402,F401
from cli_think_suite import (  # noqa: E402,F401
    NoThinkRewriteTests as CliNoThinkRewriteTests,
    ThinkSplitterTests as CliThinkSplitterTests,
)
from asr_metadata_suite import AsrMetadataTests  # noqa: E402,F401
from supertonic_tts_suite import (  # noqa: E402,F401
    ClientConfigurationTests as SupertonicClientConfigurationTests,
    EnvironmentDiscoveryTests as SupertonicEnvironmentDiscoveryTests,
    SegmentTextTests as SupertonicSegmentTextTests,
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
def test_ui_config_reads_supertonic_paths(tmp_path) -> None:
    """app.tts.supertonic persists the machine-specific Supertonic paths that
    setup.sh wrote, and defaults apply when the section is absent."""
    from shared.config import load_ui_config

    base = "app:\n  web:\n    port: 5000\n"
    with_paths = tmp_path / "with.yaml"
    with_paths.write_text(
        base + "  tts:\n    supertonic:\n      repo_root: /data/st-repo\n"
        "      app_root: /data/st-app\n", encoding="utf-8")
    cfg = load_ui_config(with_paths, tmp_path)
    assert cfg.supertonic.repo_root == "/data/st-repo"
    assert cfg.supertonic.app_root == "/data/st-app"

    without = tmp_path / "without.yaml"
    without.write_text(base, encoding="utf-8")
    cfg = load_ui_config(without, tmp_path)
    assert cfg.supertonic.repo_root == "/media/nvme/repos/supertonic-sima"
    assert cfg.supertonic.app_root == "/media/nvme/supertonic-tts"


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
    "MlaFailureClassificationTests",
    "HubPathSecurityTests",
    "CliNoThinkRewriteTests",
    "CliThinkSplitterTests",
    "AsrMetadataTests",
    "SupertonicClientConfigurationTests",
    "SupertonicEnvironmentDiscoveryTests",
    "SupertonicSegmentTextTests",
]
