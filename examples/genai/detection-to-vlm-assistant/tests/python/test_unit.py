"""Unit tests for detection-to-vlm-assistant (Python)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parent.parent.parent
MAIN_PY = EXAMPLE_DIR / "src" / "python" / "main.py"


@pytest.mark.unit
def test_input_options_format_uses_pyneat_enum() -> None:
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
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
