"""Unit tests for neat-genai-studio (Python).

The suites live next to the code they cover, under ``src/python``, so they stay
runnable in place with ``python -m unittest``. This module is what the
repository harness collects: it puts ``src/python`` on the path, re-exports the
TestCase classes, and marks them as unit coverage so ``./tests/test.sh --unit``
runs them.

No hardware, no model downloads, no live server — the ASR switching suite drives
a fake GenAIServer against model directories built in a temp dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_PYTHON = EXAMPLE_DIR / "src" / "python"

if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from server.test_asr_switching import (  # noqa: E402
    AsrSwitchingTests,
    AsrWarmupBehaviourTests,
    AsrWarmupPayloadTests,
    MlaFailureClassificationTests,
)

# Applies to every TestCase collected from this module.
pytestmark = pytest.mark.unit

__all__ = [
    "AsrSwitchingTests",
    "AsrWarmupBehaviourTests",
    "AsrWarmupPayloadTests",
    "MlaFailureClassificationTests",
]
