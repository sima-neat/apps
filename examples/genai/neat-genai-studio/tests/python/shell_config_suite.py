"""The shared shell helper setup.sh and run.sh use to read the persisted
Supertonic paths (src/common/config_value.sh), exercised through bash."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2] / "src" / "common" / "config_value.sh"


def _read(config_text: str, key: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        config = Path(tmp) / "config.yaml"
        config.write_text(config_text, encoding="utf-8")
        out = subprocess.run(
            ["bash", "-c", 'source "$1"; supertonic_config_value "$2" "$3"', "_",
             str(HELPER), str(config), key],
            check=True, capture_output=True, text=True)
        return out.stdout.strip()


CONFIG = """app:
  web:
    port: 5000
  tts:
    supertonic:
      models_root: "/data/st # models"
      venv: '/opt/st-venv'
      app_root: /legacy/root   # old key
  rag:
    enabled: true
"""


class ShellConfigValueTests(unittest.TestCase):
    def test_quoted_and_plain_values(self):
        self.assertEqual(_read(CONFIG, "models_root"), "/data/st # models")
        self.assertEqual(_read(CONFIG, "venv"), "/opt/st-venv")
        self.assertEqual(_read(CONFIG, "app_root"), "/legacy/root")

    def test_missing_key_section_or_file_is_empty(self):
        self.assertEqual(_read(CONFIG, "nope"), "")
        self.assertEqual(_read("app:\n  web:\n    port: 5000\n", "models_root"), "")
        out = subprocess.run(
            ["bash", "-c", 'source "$1"; supertonic_config_value /does/not/exist venv', "_", str(HELPER)],
            check=True, capture_output=True, text=True)
        self.assertEqual(out.stdout, "")

    def test_keys_outside_the_supertonic_section_are_ignored(self):
        text = "app:\n  web:\n    venv: /wrong\n  tts:\n    piper:\n      venv: /wrong2\n"
        self.assertEqual(_read(text, "venv"), "")


if __name__ == "__main__":
    unittest.main()
