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
VECTOR_PIN = EXAMPLE / "src" / "python" / "ui" / "tts" / "supertonic_sima" / "vector_estimator.json"
WORKER = EXAMPLE / "src" / "python" / "ui" / "tts" / "supertonic_worker.py"


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


class VectorEstimatorTests(unittest.TestCase):
    """Platform 3.0's vector estimator comes from the SiMa model zoo; the URL and
    checksum in vector_estimator.json are what setup.sh downloads and checks
    and what the engine accepts."""

    def setUp(self):
        import json
        self.pin = json.loads(VECTOR_PIN.read_text(encoding="utf-8"))

    def test_the_pin_points_at_the_model_zoo(self):
        self.assertTrue(self.pin["url"].startswith("https://docs.sima.ai/pkg_downloads/"))
        self.assertRegex(self.pin["sha256"], r"^[0-9a-f]{64}$")

    def test_setup_checks_the_same_checksum_and_no_longer_fetches_it_from_hugging_face(self):
        sums = _setup_function("_supertonic_checksums")
        self.assertIn(f"{self.pin['sha256']}  supertonic-3-sima/supertonic_vector_field_sima_mpk.tar.gz", sums)
        models = _setup_function("_supertonic_models")
        self.assertNotIn("supertonic_vector_field_sima_mpk.tar.gz", models)
        self.assertIn("_supertonic_vector_estimator", models)
        self.assertIn('sima-cli download "${url}"', _setup_function("_supertonic_vector_estimator"))

    def test_setup_finds_sima_cli_outside_a_login_shell(self):
        # The DevKit puts sima-cli on PATH only for login shells; both sima-cli
        # steps must find it when setup.sh runs over ssh.
        self.assertIn("if ! _find_sima_cli; then", _setup_function("_supertonic_vector_estimator"))
        self.assertIn("_find_sima_cli", _setup_function("_supertonic_venv"))
        helper = _setup_function("_find_sima_cli")
        with tempfile.TemporaryDirectory() as tmp:
            venv_bin = Path(tmp) / "bin"
            venv_bin.mkdir()
            cli = venv_bin / "sima-cli"
            cli.write_text("#!/bin/sh\n")
            cli.chmod(0o755)
            run = lambda bin_dir: subprocess.run(
                ["bash", "-c", f"{helper}\n_find_sima_cli && command -v sima-cli"],
                env={"PATH": "/usr/bin:/bin", "SIMA_CLI_VENV_BIN": str(bin_dir)},
                capture_output=True, text=True,
            )
            found = run(venv_bin)
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertEqual(found.stdout.strip(), str(cli))
            self.assertNotEqual(run(Path(tmp) / "missing").returncode, 0)

    def test_setup_skips_supertonic_on_a_platform_older_than_3_0(self):
        # The pinned vector estimator is the Platform 3.0 compile; an older board
        # keeps its Supertonic files and skips the install.
        script = "\n".join([
            "section() { :; }", "warn() { echo \"WARN $*\"; }",
            "_supertonic_runtime_ok() { echo REACHED; return 1; }", "_supertonic_venv() { return 1; }",
            _setup_function("_platform_version"), _setup_function("install_supertonic"),
            "install_supertonic",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            def run(version):
                info = Path(tmp) / "buildinfo"
                if version is None:
                    info.unlink(missing_ok=True)
                else:
                    info.write_text(f"DISTRO = eLxr\nDISTRO_VERSION = {version}\nCODENAME = agate\n")
                return subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin", "SIMA_BUILDINFO": str(info)},
                                      capture_output=True, text=True).stdout
            older = run("2.1.3")
            self.assertIn("needs Platform 3.0 or newer (this board: 2.1.3)", older)
            self.assertNotIn("REACHED", older)
            self.assertIn("REACHED", run("3.0.0"))
            self.assertIn("REACHED", run(None))                 # unknown platform: try the install

    def test_the_worker_passes_the_pinned_checksum_to_the_engine(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn('expected_mpk_sha256=json.loads(VECTOR_ESTIMATOR.read_text())["sha256"]', source)

    def test_the_engine_checks_the_pinned_checksum_instead_of_the_manifest(self):
        import hashlib
        import json
        import types
        tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SupertonicModalix")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_artifacts")
        namespace = {"json": json, "_sha256": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENGINE), "exec"), namespace)
        verify = namespace["_verify_artifacts"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mpk, runtime, manifest = root / "mpk", root / "runtime", root / "manifest.json"
            mpk.write_bytes(b"platform 3.0 compile")
            runtime.write_bytes(b"runtime")
            sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest.write_text(json.dumps({"compiled": {"archive": {"sha256": "0" * 64}},
                                            "graph_surgery": {"runtime_data": {"sha256": sha(runtime)}}}))
            engine = types.SimpleNamespace(artifact_manifest_path=manifest, mpk_path=mpk, runtime_data_path=runtime,
                                           vocoder_backend="cpu", expected_mpk_sha256=sha(mpk))
            verify(engine)                                  # the pin wins over the manifest's old checksum
            engine.expected_mpk_sha256 = None
            with self.assertRaisesRegex(ValueError, "MPK SHA-256 mismatch"):
                verify(engine)                              # without a pin, the manifest still decides
            engine.expected_mpk_sha256 = "1" * 64
            with self.assertRaisesRegex(ValueError, "MPK SHA-256 mismatch"):
                verify(engine)                              # a wrong file is still refused


if __name__ == "__main__":
    unittest.main()
