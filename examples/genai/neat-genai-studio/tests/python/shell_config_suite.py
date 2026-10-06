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


import os
import sys


def _run(script: str, *args: str, python: str | None = None) -> str:
    env = dict(os.environ, STUDIO_CONFIG_PYTHON=python or sys.executable)
    out = subprocess.run(["bash", "-c", f'source "$1"; shift; {script}', "_", str(HELPER), *args],
                         check=True, capture_output=True, text=True, env=env)
    return out.stdout.strip()


WEB = """app:
  web:
    host: "0.0.0.0"
    port: 5000   # the UI
    https: 'false'
    headless: "True"
    cors_origins: "http://a:3000, https://b"   # allowlist
  tts:
    supertonic:
      models_root: /m
"""


class ShellWebConfigTests(unittest.TestCase):
    def _web(self, key):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text(WEB, encoding="utf-8")
            return _run('web_config_scalar "$1" "$2"', str(config), key)

    def test_quoted_and_commented_scalars(self):
        self.assertEqual(self._web("host"), "0.0.0.0")
        self.assertEqual(self._web("port"), "5000")
        self.assertEqual(self._web("https"), "false")
        self.assertEqual(self._web("headless"), "true")       # normalized by the loader
        self.assertEqual(self._web("cors_origins"), "http://a:3000, https://b")
        self.assertEqual(self._web("models_root"), "")      # other section

    def test_list_valued_setting_is_joined(self):
        text = ("app:\n  web:\n    port: 5000\n    cors_origins:\n      - http://a:3000\n"
                "      - 'https://b'   # second\n      - \"*\"\n    https: true\n")
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text(text, encoding="utf-8")
            self.assertEqual(_run('web_config_list "$1" cors_origins', str(config)), "http://a:3000,https://b,*")
            self.assertEqual(_run('web_config_scalar "$1" cors_origins', str(config)), "http://a:3000,https://b,*")
            self.assertEqual(_run('web_config_list "$1" https', str(config)), "true")   # loader value

    def test_any_indentation_and_no_app_root_like_the_loader(self):
        four = ("app:\n    web:\n        headless: true\n        https: false\n"
                "    tts:\n        supertonic:\n            models_root: /m4\n")
        top = "web:\n  headless: yes\ntts:\n  supertonic:\n    venv: /v\n"
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "four.yaml"; a.write_text(four, encoding="utf-8")
            b = Path(tmp) / "top.yaml"; b.write_text(top, encoding="utf-8")
            self.assertEqual(_run('web_config_scalar "$1" headless', str(a)), "true")
            self.assertEqual(_run('web_config_scalar "$1" https', str(a)), "false")
            self.assertEqual(_run('supertonic_config_value "$1" models_root', str(a)), "/m4")
            self.assertEqual(_run('web_config_scalar "$1" headless', str(b)), "true")
            self.assertEqual(_run('supertonic_config_value "$1" venv', str(b)), "/v")
        # the Python loader agrees on the same files
        import sys
        sys.path.insert(0, str(HELPER.parents[2] / "src" / "python"))
        from shared.config import load_ui_config
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "four.yaml"; a.write_text(four, encoding="utf-8")
            cfg = load_ui_config(a, Path(tmp))
            self.assertEqual((cfg.web.headless, cfg.web.https, cfg.supertonic.models_root), (True, False, "/m4"))

    def test_flow_style_mapping_reads_like_the_loader(self):
        text = ('app:\n  web: {host: 0.0.0.0, port: 5000, headless: true, https: "false",'
                ' cors_origins: [http://a:3000, "https://b"]}\n'
                '  tts: {supertonic: {models_root: /flow, venv: /fv}}\n')
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "flow.yaml"; c.write_text(text, encoding="utf-8")
            self.assertEqual(_run('web_config_scalar "$1" headless', str(c)), "true")
            self.assertEqual(_run('web_config_scalar "$1" https', str(c)), "false")
            self.assertEqual(_run('web_config_scalar "$1" port', str(c)), "5000")
            self.assertEqual(_run('web_config_list "$1" cors_origins', str(c)), "http://a:3000,https://b")
            self.assertEqual(_run('supertonic_config_value "$1" models_root', str(c)), "/flow")
            self.assertEqual(_run('supertonic_config_value "$1" venv', str(c)), "/fv")

    def test_fallback_reader_without_a_yaml_python(self):
        # No usable Python: the block-style awk reader still answers.
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "c.yaml"; c.write_text(WEB, encoding="utf-8")
            env = dict(os.environ, STUDIO_CONFIG_PYTHON="/nonexistent/python", PATH="/usr/bin:/bin")
            script = 'source "$1"; _studio_config_python() { return 1; }; web_config_scalar "$2" headless'
            out = subprocess.run(["bash", "-c", script, "_", str(HELPER), str(c)],
                                 check=True, capture_output=True, text=True, env=env)
            self.assertEqual(out.stdout.strip(), "True")

    def test_truthiness_matches_the_python_loader(self):
        for value in ("1", "true", "True", "YES", "on"):
            self.assertEqual(_run('config_true "$1" && echo y || echo n', value), "y", value)
        for value in ("0", "false", "", "no", "off", "'true'"):
            self.assertEqual(_run('config_true "$1" && echo y || echo n', value), "n", value)


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
