"""Unit tests for detection-to-vlm-assistant (Python)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML = EXAMPLE_DIR / "src" / "common" / "config.yaml"
DETECTOR_APP_PY = EXAMPLE_DIR / "src" / "python" / "detector_app.py"
GENAI_SERVER_PY = EXAMPLE_DIR / "src" / "python" / "genai_server.py"


@pytest.mark.unit
def test_input_options_format_uses_pyneat_enum() -> None:
    tree = ast.parse(DETECTOR_APP_PY.read_text(encoding="utf-8"))
    bad_lines: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "format":
                bad_lines.append(node.lineno)

    assert bad_lines == []


@pytest.mark.unit
def test_genai_config_and_scripts_use_genai_names() -> None:
    raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))

    assert raw["genai_server"]["host"]
    assert raw["genai_server"]["port"] > 0
    assert raw["genai_server"]["model"]["name"]
    assert raw["genai_server"]["model"]["path"]
    assert raw["genai"]["host"]
    assert raw["genai"]["port"] > 0
    assert raw["genai"]["system_prompt"]
    assert raw["genai"]["user_prompt"]

    assert "factual" in raw["genai"]["system_prompt"].lower()
    assert "what is the person doing" in raw["genai"]["user_prompt"].lower()
