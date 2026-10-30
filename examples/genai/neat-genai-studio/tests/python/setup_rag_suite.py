"""setup.sh's SETUP_RAG switch: what a run installs, downloads and writes with
document search (RAG) on and off, and how a re-run keeps the choice. python3,
the venv tools and PyNeat are stubs that record their arguments, so nothing is
installed or fetched."""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE = Path(__file__).resolve().parents[2]
SETUP = EXAMPLE / "setup.sh"

# A venv's python and hf: record the call; config_query.py (the config reader
# setup.sh asks) runs for real so a re-run sees the written config.
VENV_TOOL = """#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "${SETUP_LOG}"
case "$*" in *config_query.py*) exec "${REAL_PYTHON}" "$@" ;; esac
exit 0
"""
# python3 on PATH: "-m venv DIR" makes DIR/bin/{python,hf} stubs; anything else
# is the real interpreter.
PYTHON3 = """#!/bin/bash
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
  mkdir -p "$3/bin"
  cp "${VENV_TOOL}" "$3/bin/python"
  cp "${VENV_TOOL}" "$3/bin/hf"
  exit 0
fi
exec "${REAL_PYTHON}" "$@"
"""

RAG_STEPS = ("download.pytorch.org", "requirements-rag.txt", "hf download thenlper/gte-small",
             "rag/create_db.py")


class RagDatabaseCleanupTests(unittest.TestCase):
    def test_command_releases_local_servers_on_success_and_failure(self):
        for error in (None, RuntimeError("database write failed")):
            with self.subTest(error=error):
                manager = Mock()
                dependencies = {
                    "langchain_huggingface": SimpleNamespace(HuggingFaceEmbeddings=Mock()),
                    "langchain_milvus": SimpleNamespace(Milvus=Mock()),
                    "langchain_text_splitters": SimpleNamespace(MarkdownHeaderTextSplitter=Mock()),
                    "milvus_lite.server_manager": SimpleNamespace(server_manager_instance=manager),
                }
                spec = importlib.util.spec_from_file_location(
                    "rag_create_db_test", EXAMPLE / "src/python/rag/create_db.py"
                )
                module = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, dependencies):
                    spec.loader.exec_module(module)
                args = SimpleNamespace(input="doc.md", output="db", embedding_model="model", collection="docs")
                with patch.object(module, "parse_args", return_value=args), patch.object(
                    module, "create_markdown_vectordb", side_effect=error
                ):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "database write failed"):
                            module.main()
                    else:
                        self.assertEqual(module.main(), 0)
                manager.release_all.assert_called_once_with()


class SetupRagSwitchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name, text in (("python3", PYTHON3), ("venv-tool", VENV_TOOL),
                           ("pyneat-python", "#!/bin/sh\nexit 0\n")):
            path = bin_dir / name
            path.write_text(text, encoding="utf-8")
            path.chmod(0o755)
        self.config = self.tmp / "config.local.yaml"
        self.log = self.tmp / "calls.log"
        self.env = dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            REAL_PYTHON=sys.executable, VENV_TOOL=str(bin_dir / "venv-tool"),
            SETUP_LOG=str(self.log), PYNEAT_PYTHON=str(bin_dir / "pyneat-python"),
            APP_VENV=str(self.tmp / "venv"), PIPERTTS_VENV=str(self.tmp / "venv-piper"),
            CONFIG_PATH=str(self.config), LLIMA_MODELS_PATH=str(self.tmp / "models"),
            ASR_MODEL_REPO="", INSTALL_TTS_VOICES="0", INSTALL_SUPERTONIC="0", NO_COLOR="1",
        )
        for name in ("SETUP_RAG", "STUDIO_CONFIG_PYTHON", "CHAT_MODEL_REPO", "CATALOG_MODEL_REPOS",
                     "ASR_CATALOG_MODEL_REPOS", "SKIP_MODEL_DOWNLOAD"):
            self.env.pop(name, None)

    def tearDown(self):
        self._tmp.cleanup()

    def _setup(self, *args, **env):
        self.log.write_text("", encoding="utf-8")
        result = subprocess.run(["bash", str(SETUP), *args], env={**self.env, **env},
                                stdin=subprocess.DEVNULL, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.log.read_text(encoding="utf-8")

    def _rag_config(self):
        sys.path.insert(0, str(EXAMPLE / "src" / "python"))
        from shared.config import load_ui_config
        return load_ui_config(self.config, self.tmp).rag

    def _assert_rag_steps(self, calls, present):
        for step in RAG_STEPS:
            (self.assertIn if present else self.assertNotIn)(step, calls)

    def test_default_installs_document_search(self):
        calls = self._setup()
        self.assertIn("requirements.txt", calls)
        self._assert_rag_steps(calls, present=True)
        self.assertTrue(self._rag_config().enabled)

    def test_setup_rag_0_skips_torch_packages_model_and_database(self):
        calls = self._setup(SETUP_RAG="0")
        self.assertIn("-r " + str(EXAMPLE / "src/python/requirements.txt"), calls)
        self._assert_rag_steps(calls, present=False)
        rag = self._rag_config()
        self.assertFalse(rag.enabled)
        self.assertEqual(rag.embedding_model_dir, str(self.tmp / "models" / "gte-small"))
        # The hint names the script itself, not a path relative to some directory.
        self.assertIn(f"SETUP_RAG=1 {SETUP}", self.config.read_text(encoding="utf-8"))

    def test_rerun_and_dependency_refresh_keep_the_choice(self):
        self._setup(SETUP_RAG="0")
        self._assert_rag_steps(self._setup(), present=False)
        self.assertFalse(self._rag_config().enabled)
        refresh = self._setup("--dependencies-only")
        self.assertIn("requirements.txt", refresh)
        self.assertNotIn("requirements-rag.txt", refresh)
        self.assertNotIn("download.pytorch.org", refresh)
        # The environment wins over the config.
        self._assert_rag_steps(self._setup(SETUP_RAG="1"), present=True)
        self.assertTrue(self._rag_config().enabled)
        self.assertIn("requirements-rag.txt", self._setup("--dependencies-only"))


if __name__ == "__main__":
    unittest.main()
