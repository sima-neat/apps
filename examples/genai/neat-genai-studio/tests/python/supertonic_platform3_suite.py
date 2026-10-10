"""Supertonic on Platform 3.0 (Python 3.13, PyNeat 0.6 in ~/pyneat): the numpy
pin, the PyNeat link into .venv-supertonic, and the engine's tolerance of
PyNeat 0.6's leading batch axis.

The setup.sh pieces run through bash against throwaway venvs; nothing here
needs pyneat, a board or a network."""

from __future__ import annotations

import ast
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2]
SETUP = EXAMPLE / "setup.sh"
ENGINE = EXAMPLE / "src" / "python" / "ui" / "tts" / "supertonic_sima" / "engine.py"
REQUIREMENTS = EXAMPLE / "src" / "python" / "requirements-supertonic.txt"


def _setup_function(name: str) -> str:
    """The body of one shell function from setup.sh, from its header to the
    first line that is a lone closing brace."""
    lines = SETUP.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


def _engine_function(name: str):
    """One pure helper from the vendored engine, without importing the engine
    (which needs numpy, onnxruntime and pyneat)."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ENGINE), "exec"), namespace)
    return namespace[name]


def _executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class BatchAxisTests(unittest.TestCase):
    def setUp(self):
        self.drop = _engine_function("_drop_batch_axes")

    def test_leading_batch_axis_of_one_is_dropped(self):
        self.assertEqual(self.drop((1, 1, 192, 144), 3), (1, 192, 144))

    def test_public_shapes_are_unchanged(self):
        self.assertEqual(self.drop((1, 192, 144), 3), (1, 192, 144))
        self.assertEqual(self.drop((1, 50, 256), 3), (1, 50, 256))

    def test_a_real_batch_is_kept(self):
        # A batch other than one is a different contract and must still fail
        # the engine's shape check.
        self.assertEqual(self.drop((2, 1, 192, 144), 3), (2, 1, 192, 144))

    def test_engine_uses_it_for_specs_and_outputs(self):
        source = ENGINE.read_text(encoding="utf-8")
        self.assertIn("return _drop_batch_axes(tuple(int(value) for value in shape), 3)", source)
        self.assertIn("value.reshape(_drop_batch_axes(value.shape, len(expected_public_shape)))", source)


class NumpyPinTests(unittest.TestCase):
    def test_numpy_is_pinned_to_a_2x_release(self):
        # numpy 1.x has no Python 3.13 wheels, so Platform 3.0 cannot install it.
        pins = re.findall(r"^numpy==(\S+)$", REQUIREMENTS.read_text(encoding="utf-8"), re.M)
        self.assertEqual(len(pins), 1, pins)
        self.assertEqual(pins[0].split(".")[0], "2")


class PyneatLinkTests(unittest.TestCase):
    """_link_installed_pyneat against a throwaway venv and a fake Neat
    environment whose pyneat distribution lives on PYTHONPATH."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.venv = root / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(self.venv)], check=True)
        self.purelib = Path(subprocess.run(
            [str(self.venv / "bin" / "python"), "-c",
             'import sysconfig; print(sysconfig.get_paths()["purelib"])'],
            check=True, capture_output=True, text=True).stdout.strip())
        site = root / "neat-site"
        (site / "pyneat").mkdir(parents=True)
        (site / "pyneat" / "__init__.py").write_text("", encoding="utf-8")
        dist = site / "pyneat-0.6.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: pyneat\nVersion: 0.6.0\n", encoding="utf-8")
        (dist / "RECORD").write_text(
            "pyneat/__init__.py,,\npyneat-0.6.0.dist-info/METADATA,,\npyneat-0.6.0.dist-info/RECORD,,\n"
            "../../../bin/neat-tool,,\n", encoding="utf-8")
        self.site = site
        self.neat_python = _executable(
            root / "neat-python",
            f'#!/bin/sh\nPYTHONPATH="{site}" exec "{sys.executable}" "$@"\n')

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, runtime_ok: bool, venv_python: Path | None = None) -> int:
        script = textwrap.dedent(f"""
            _supertonic_runtime_ok() {{ return {0 if runtime_ok else 1}; }}
            {_setup_function("_link_installed_pyneat")}
            _link_installed_pyneat
        """)
        env = dict(os.environ, SUPERTONIC_VENV=str(self.venv), PYNEAT_PYTHON=str(self.neat_python))
        if venv_python is not None:
            env["SUPERTONIC_VENV"] = str(venv_python)
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True).returncode

    def test_links_the_package_and_its_metadata(self):
        self.assertEqual(self._run(runtime_ok=True), 0)
        for name in ("pyneat", "pyneat-0.6.0.dist-info"):
            link = self.purelib / name
            self.assertTrue(link.is_symlink(), name)
            self.assertEqual(link.resolve(), (self.site / name).resolve())
        # RECORD entries outside site-packages (console scripts) are not linked.
        self.assertEqual(sorted(p.name for p in self.purelib.iterdir() if p.is_symlink()),
                         ["pyneat", "pyneat-0.6.0.dist-info"])

    def test_links_are_removed_when_the_runtime_does_not_import(self):
        self.assertEqual(self._run(runtime_ok=False), 1)
        self.assertFalse((self.purelib / "pyneat").exists())
        self.assertFalse((self.purelib / "pyneat-0.6.0.dist-info").exists())

    def test_nothing_is_linked_when_python_versions_differ(self):
        fake = Path(self._tmp.name) / "other-venv"
        (fake / "bin").mkdir(parents=True)
        _executable(fake / "bin" / "python", textwrap.dedent(f"""\
            #!/bin/sh
            case "$2" in
              *version_info*) echo 2.7 ;;
              *) exec "{self.venv / 'bin' / 'python'}" "$@" ;;
            esac
            """))
        self.assertEqual(self._run(runtime_ok=True, venv_python=fake), 1)
        self.assertFalse((self.purelib / "pyneat").exists())


if __name__ == "__main__":
    unittest.main()
