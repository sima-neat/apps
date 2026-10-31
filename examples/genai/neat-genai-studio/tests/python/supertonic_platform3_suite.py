"""Supertonic on Platform 3.0 (Python 3.13, PyNeat 0.6 in ~/pyneat): the numpy
pin, the PyNeat link into .venv-supertonic, and batched guidance bindings.

The setup.sh pieces run through bash against throwaway venvs; nothing here
needs pyneat, a board or a network. Numerical adapter checks use numpy."""

from __future__ import annotations

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
ENGINE = EXAMPLE / "src" / "python" / "api" / "tts" / "supertonic_sima" / "engine.py"
REQUIREMENTS = EXAMPLE / "src" / "python" / "requirements-supertonic.txt"


def _setup_function(name: str, script: Path = SETUP) -> str:
    """The body of one shell function, from its header to the
    first line that is a lone closing brace."""
    lines = script.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start:end + 1])


def _executable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class BatchAxisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from types import ModuleType, SimpleNamespace
        from unittest import mock
        import numpy as np

        cls.np = np
        cls.tensor = staticmethod(lambda value: SimpleNamespace(to_numpy=lambda **_: value))
        package = ModuleType("studio_supertonic_test")
        package.__path__ = [str(ENGINE.parent)]
        spec = importlib.util.spec_from_file_location("studio_supertonic_test.engine", ENGINE)
        cls.engine = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {
            "studio_supertonic_test": package,
            "studio_supertonic_test.engine": cls.engine,
            "pyneat": SimpleNamespace(
                Tensor=SimpleNamespace(from_numpy=lambda value, **_: cls.tensor(value)),
                TensorLayout=SimpleNamespace(HWC="HWC")),
            "onnxruntime": ModuleType("onnxruntime"),
        }):
            spec.loader.exec_module(cls.engine)

    def test_batched_layout_and_vocoder_waveform_order(self):
        from types import SimpleNamespace
        np, runtime = self.np, self.engine
        physical = np.arange(2 * 144 * 192, dtype=np.float32).reshape(2, 144, 1, 192)
        shape = runtime._public_shape(physical.shape)
        self.assertEqual(shape, (2, 1, 192, 144))
        output = [runtime._tensor_from_public(physical.transpose(0, 2, 3, 1))]
        np.testing.assert_array_equal(
            runtime._first_output_array(output, expected_public_shape=shape), physical)

        engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
        engine.vocoder_backend, engine.timeout_ms, engine.chunk_size = "mla", 1000, 3072
        engine.vocoder_public_input = runtime._public_shape(runtime.VOCODER_PHYSICAL_INPUT[1])
        engine.vocoder_public_output = runtime._public_shape(runtime.VOCODER_PHYSICAL_OUTPUT[1])
        engine._vocoder_input_buffer = np.zeros(engine.vocoder_public_input, dtype=np.float32)
        for shape in (engine.vocoder_public_input, engine.vocoder_public_input[1:]):
            engine.vocoder_model = SimpleNamespace(
                compiled_batch_size=lambda: 1,
                input_specs=lambda: [SimpleNamespace(shape=shape)],
                output_specs=lambda: [SimpleNamespace(shape=engine.vocoder_public_output)])
            engine._validate_vocoder_contract()
        frames = np.arange(512 * 1152, dtype=np.float32).reshape(engine.vocoder_public_output)
        latent = physical[0, :, 0, :][None, ...]

        def run(tensors, **kwargs):
            np.testing.assert_array_equal(
                tensors[0].to_numpy(), latent.transpose(0, 2, 1)[:, None, :, :])
            return [self.tensor(frames)]

        engine.vocoder_runner = SimpleNamespace(run=run)
        actual = engine._run_vocoder(latent, 2)
        np.testing.assert_array_equal(actual, np.arange(6144, dtype=np.float32))
        self.assertFalse(np.shares_memory(actual, frames))

    def test_batched_guidance_uses_one_call_per_step(self):
        from types import SimpleNamespace
        np, runtime = self.np, self.engine
        engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
        engine.steps, engine.timeout_ms = 8, 1000
        engine.sample_rate, engine.chunk_size, engine.latent_dim = 8, 2, 144
        engine.time_table = runtime.build_time_table(engine.steps)
        engine.physical_inputs = runtime.VECTOR_PHYSICAL_INPUTS
        engine.public_output = ("velocity", runtime._public_shape(runtime.VECTOR_PHYSICAL_OUTPUT[1]))
        engine._input_buffers = [np.zeros(runtime._public_shape(shape), dtype=np.float32)
                                 for _, shape in engine.physical_inputs]
        engine.constants = {
            runtime.CONDITIONAL_STYLE_KEY: np.full((1, 50, 256), 3, dtype=np.float32),
            runtime.UNCONDITIONAL_STYLE_KEY: np.full((1, 50, 256), -2, dtype=np.float32),
            runtime.UNCONDITIONAL_STYLE_VALUE: np.full((1, 50, 256), -3, dtype=np.float32),
            runtime.UNCONDITIONAL_TEXT: np.full((1, 256, 1), -1, dtype=np.float32),
        }
        engine.rope_table = np.zeros((192, 1, 64, 1, 192), dtype=np.float32)
        engine._encode_text = lambda *_: (np.array([[1, 2, 3]]), np.ones((1, 1, 3)), "hello")
        engine._load_style = lambda *_: (np.full((1, 50, 256), 2), np.zeros((1, 8, 16)))
        engine.duration_session = SimpleNamespace(run=lambda *_: [np.array([1.0])])
        engine.text_session = SimpleNamespace(run=lambda *_: [np.full((1, 256, 3), .5)])
        engine._run_vocoder = lambda latent, length: np.repeat(latent[0, 0, :length], 2)
        calls = 0

        def run(tensors, **kwargs):
            nonlocal calls
            calls += 1
            values = [tensor.to_numpy() for tensor in tensors]
            for index in (0, 4, 5, 6, 7):
                np.testing.assert_array_equal(values[index][0], values[index][1])
            velocity = (values[1].mean(axis=-1, keepdims=True)
                        + 2 * values[2].mean(axis=(1, 2, 3), keepdims=True)
                        + 3 * values[3].mean(axis=(1, 2, 3), keepdims=True))
            return [self.tensor(np.broadcast_to(velocity, engine.public_output[1]))]

        engine.runner = SimpleNamespace(run=run)
        result = engine.synthesize("hello", voice="F3", language="en", speed=1, seed=17)
        # Conditional velocity is 13.5 for text and 13 for padding; unconditional is -13.
        # Eight Euler updates total 4 * conditional - 3 * unconditional.
        initial = np.random.RandomState(17).randn(1, 144, 4).astype(np.float32)
        expected = initial[0, 0] + np.array([93, 93, 93, 91], dtype=np.float32)
        np.testing.assert_allclose(result.waveform, np.repeat(expected, 2), rtol=1e-6)
        self.assertEqual(calls, engine.steps)


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


class ModelAssetsTests(unittest.TestCase):
    """The installer and worker consume one set of compiled-model pins."""

    def test_setup_downloads_both_archives_reuses_them_and_rejects_bad_bytes(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pin = root / "app/src/python/api/tts/supertonic_sima/model_assets.json"
            pin.parent.mkdir(parents=True)
            sources = root / "sources"
            sources.mkdir()
            assets = {}
            for name in ("vector_estimator", "vocoder"):
                filename = name + ".tar.gz"
                data = name.encode()
                (sources / filename).write_bytes(data)
                assets[name] = {"file": filename, "url": "https://example.test/" + filename,
                                "sha256": hashlib.sha256(data).hexdigest()}
            pin.write_text(json.dumps(assets))
            script = "\n".join([
                "ok() { :; }; step() { :; }; warn() { echo \"$*\" >&2; }",
                "_find_sima_cli() { return 0; }",
                'sima-cli() { echo "$2" >> "$CALLS"; cp "$SOURCES/${2##*/}" "$4/${2##*/}"; }',
                _setup_function("_supertonic_mla_models"), "_supertonic_mla_models",
            ])
            env = {**os.environ, "EXAMPLE_DIR": str(root / "app"), "SOURCES": str(sources),
                   "SUPERTONIC_MODELS_ROOT": str(root / "models"), "CALLS": str(root / "calls")}
            def run():
                return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
            first = run()
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(len((root / "calls").read_text().splitlines()), 2)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(len((root / "calls").read_text().splitlines()), 2)
            target = root / "models/supertonic-3-sima/vector_estimator.tar.gz"
            target.write_bytes(b"truncated")
            self.assertEqual(run().returncode, 0)
            self.assertEqual(target.read_bytes(), b"vector_estimator")
            self.assertEqual(len((root / "calls").read_text().splitlines()), 3)
            target.unlink()
            (sources / "vector_estimator.tar.gz").write_bytes(b"wrong model")
            rejected = run()
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("checksum mismatch", rejected.stderr)
            self.assertFalse(target.exists())
            self.assertEqual(list((root / "models").glob(".supertonic-model.*")), [])

    def test_update_fingerprint_includes_model_pins(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            python_dir = Path(tmp)
            pin = python_dir / "api/tts/supertonic_sima/model_assets.json"
            pin.parent.mkdir(parents=True)
            assets = json.loads((EXAMPLE / "src/python/api/tts/supertonic_sima/model_assets.json").read_text())
            pin.write_text(json.dumps(assets))
            (python_dir / "requirements-supertonic.txt").write_text("")
            script = _setup_function("supertonic_fingerprint", EXAMPLE / "run.sh") + "\nsupertonic_fingerprint"
            env = {**os.environ, "EXAMPLE_DIR": str(EXAMPLE), "PYTHON_DIR": str(python_dir)}
            before = subprocess.check_output(["bash", "-c", script], env=env)
            assets["vector_estimator"]["sha256"] = "0" * 64
            pin.write_text(json.dumps(assets))
            self.assertNotEqual(before, subprocess.check_output(["bash", "-c", script], env=env))

    def test_setup_finds_sima_cli_outside_a_login_shell(self):
        # The DevKit puts sima-cli on PATH only for login shells; both sima-cli
        # steps must find it when setup.sh runs over ssh.
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


if __name__ == "__main__":
    unittest.main()
