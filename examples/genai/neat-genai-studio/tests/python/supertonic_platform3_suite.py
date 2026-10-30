"""Supertonic on Platform 3.0 (Python 3.13, PyNeat 0.6 in ~/pyneat): the numpy
pin, the PyNeat link into .venv-supertonic, and batched guidance bindings.

The setup.sh pieces run through bash against throwaway venvs; nothing here
needs pyneat, a board or a network. Numerical adapter checks use numpy."""

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
ENGINE = EXAMPLE / "src" / "python" / "api" / "tts" / "supertonic_sima" / "engine.py"
REQUIREMENTS = EXAMPLE / "src" / "python" / "requirements-supertonic.txt"
MODEL_ASSETS = EXAMPLE / "src" / "python" / "api" / "tts" / "supertonic_sima" / "model_assets.json"
WORKER = EXAMPLE / "src" / "python" / "api" / "tts" / "supertonic_worker.py"


def _setup_function(name: str) -> str:
    """The body of one shell function from setup.sh, from its header to the
    first line that is a lone closing brace."""
    lines = SETUP.read_text(encoding="utf-8").splitlines()
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
        import types
        from unittest import mock
        import numpy as np

        cls.np = np

        class Tensor:
            def __init__(self, value):
                self.value = np.array(value, dtype=np.float32, copy=True)

            @classmethod
            def from_numpy(cls, value, **kwargs):
                return cls(value)

            def to_numpy(self, copy=False):
                return self.value.copy() if copy else self.value

        cls.Tensor = Tensor
        package = types.ModuleType("studio_supertonic_test")
        package.__path__ = [str(ENGINE.parent)]
        spec = importlib.util.spec_from_file_location("studio_supertonic_test.engine", ENGINE)
        module = importlib.util.module_from_spec(spec)
        # Exercise the real adapter and Euler loop; only the execution providers
        # are replaced, so unit tests need neither hardware nor model assets.
        with mock.patch.dict(sys.modules, {
            "studio_supertonic_test": package,
            "studio_supertonic_test.engine": module,
            "pyneat": types.SimpleNamespace(
                Tensor=Tensor, TensorLayout=types.SimpleNamespace(HWC="HWC")
            ),
            "onnxruntime": types.ModuleType("onnxruntime"),
        }):
            spec.loader.exec_module(module)
        cls.engine = module

    def test_batch_axes_are_preserved_in_contracts_and_layout_conversion(self):
        from types import SimpleNamespace
        np, engine = self.np, self.engine
        physical = np.arange(2 * 144 * 192, dtype=np.float32).reshape(2, 144, 1, 192)
        shape = engine._public_shape(physical.shape)
        self.assertEqual(shape, (2, 1, 192, 144))
        tensor = engine._tensor_from_public(physical.transpose(0, 2, 3, 1))
        actual = engine._first_output_array(tensor, expected_public_shape=shape)
        np.testing.assert_array_equal(actual, physical)
        self.assertEqual(engine._public_shape((1, 512, 1, 1152)), (1, 1, 1152, 512))
        with self.assertRaisesRegex(RuntimeError, "MLA output shape"):
            engine._first_public_output_array(
                self.Tensor(physical[0].transpose(1, 2, 0)), expected_public_shape=shape
            )

    def test_model_contract_rejects_a_batch_one_vector_estimator(self):
        from types import SimpleNamespace
        runtime = self.engine
        engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
        engine.public_inputs = tuple(
            (name, runtime._public_shape(shape)) for name, shape in runtime.VECTOR_PHYSICAL_INPUTS
        )
        engine.public_output = ("velocity", runtime._public_shape(runtime.VECTOR_PHYSICAL_OUTPUT[1]))
        inputs = [SimpleNamespace(shape=shape) for _, shape in engine.public_inputs]
        engine.model = SimpleNamespace(
            input_specs=lambda: inputs,
            output_specs=lambda: [SimpleNamespace(shape=engine.public_output[1])],
        )
        engine._validate_model_contract()
        inputs[0].shape = (1, *inputs[0].shape[1:])
        with self.assertRaisesRegex(RuntimeError, "MPK input 0"):
            engine._validate_model_contract()

    def test_vocoder_contract_preserves_authored_batch_axis(self):
        from types import SimpleNamespace
        runtime = self.engine
        engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
        engine.vocoder_public_input = runtime._public_shape(runtime.VOCODER_PHYSICAL_INPUT[1])
        engine.vocoder_public_output = runtime._public_shape(runtime.VOCODER_PHYSICAL_OUTPUT[1])
        engine.vocoder_model = SimpleNamespace(
            compiled_batch_size=lambda: 1,
            input_specs=lambda: [SimpleNamespace(shape=(1, 1, 192, 144))],
            output_specs=lambda: [SimpleNamespace(shape=(1, 1, 1152, 512))],
        )
        engine._validate_vocoder_contract()
        self.assertEqual(engine.vocoder_public_input, (1, 1, 192, 144))
        for shape in ((1, 192, 144), (1, 1, 193, 144), (2, 1, 192, 144)):
            with self.subTest(shape=shape):
                engine.vocoder_model.input_specs = lambda: [SimpleNamespace(shape=shape)]
                with self.assertRaisesRegex(RuntimeError, "vocoder input shape"):
                    engine._validate_vocoder_contract()

    def test_vocoder_preserves_batch_one_and_waveform_order(self):
        from types import SimpleNamespace
        np, runtime = self.np, self.engine
        engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
        engine.vocoder_backend = "mla"
        engine.timeout_ms, engine.chunk_size = 1000, 3072
        engine.vocoder_public_input = runtime._public_shape(runtime.VOCODER_PHYSICAL_INPUT[1])
        engine.vocoder_public_output = runtime._public_shape(runtime.VOCODER_PHYSICAL_OUTPUT[1])
        engine._vocoder_input_buffer = np.zeros(engine.vocoder_public_input, dtype=np.float32)
        latent = np.arange(144 * 192, dtype=np.float32).reshape(1, 144, 192)
        frames = np.arange(512 * 1152, dtype=np.float32).reshape(engine.vocoder_public_output)

        def run(tensors, **kwargs):
            np.testing.assert_array_equal(
                tensors[0].to_numpy(), latent.transpose(0, 2, 1)[:, None, :, :]
            )
            return [self.Tensor(frames)]

        engine.vocoder_runner = SimpleNamespace(run=run)
        actual = engine._run_vocoder(latent, 2)
        np.testing.assert_array_equal(actual, np.arange(6144, dtype=np.float32))
        self.assertFalse(np.shares_memory(actual, frames))

    def test_one_call_per_step_matches_two_independent_guidance_branches(self):
        from types import SimpleNamespace
        np, runtime = self.np, self.engine
        for steps in (5, 8, 12):
            with self.subTest(steps=steps):
                engine = runtime.SupertonicModalix.__new__(runtime.SupertonicModalix)
                engine.steps = steps
                engine.time_table = runtime.build_time_table(steps)
                engine.timeout_ms = 1000
                engine.sample_rate, engine.chunk_size, engine.latent_dim = 8, 2, 144
                engine.physical_inputs = runtime.VECTOR_PHYSICAL_INPUTS
                engine.public_output = (
                    "velocity", runtime._public_shape(runtime.VECTOR_PHYSICAL_OUTPUT[1])
                )
                engine._input_buffers = [
                    np.zeros(runtime._public_shape(shape), dtype=np.float32)
                    for _, shape in engine.physical_inputs
                ]
                engine.constants = {
                    runtime.CONDITIONAL_STYLE_KEY: np.full((1, 50, 256), 3, dtype=np.float32),
                    runtime.UNCONDITIONAL_STYLE_KEY: np.full((1, 50, 256), -2, dtype=np.float32),
                    runtime.UNCONDITIONAL_STYLE_VALUE: np.full((1, 50, 256), -3, dtype=np.float32),
                    runtime.UNCONDITIONAL_TEXT: np.full((1, 256, 1), -1, dtype=np.float32),
                }
                engine.rope_table = np.zeros((192, 1, 64, 1, 192), dtype=np.float32)
                text = np.full((1, 256, 3), .5, dtype=np.float32)
                engine._encode_text = lambda *_: (
                    np.array([[1, 2, 3]]), np.ones((1, 1, 3), dtype=np.float32), "hello"
                )
                engine._load_style = lambda *_: (
                    np.full((1, 50, 256), 2, dtype=np.float32), np.zeros((1, 8, 16))
                )
                engine.duration_session = SimpleNamespace(
                    run=lambda *_: [np.array([1.0], dtype=np.float32)]
                )
                engine.text_session = SimpleNamespace(run=lambda *_: [text])
                engine._run_vocoder = lambda latent, length: np.repeat(latent[0, 0, :length], 2)
                calls = []

                def run(tensors, **kwargs):
                    values = [t.to_numpy().copy() for t in tensors]
                    calls.append(values)
                    latent, texts, styles, keys = values[:4]
                    velocity = (latent * .1 + texts.mean(axis=-1, keepdims=True) * .01
                                + styles.mean(axis=(1, 2, 3), keepdims=True) * .02
                                + keys.mean(axis=(1, 2, 3), keepdims=True) * .03
                                + values[6].mean(axis=(1, 2, 3), keepdims=True) * .001)
                    return [self.Tensor(velocity)]

                engine.runner = SimpleNamespace(run=run)
                actual = engine.synthesize("hello", voice="F3", language="en", speed=1, seed=17)
                # Independent NCHW reference: evaluate each guidance branch
                # separately, then apply the same mask and Euler schedule.
                latent = runtime._pad_last(
                    np.random.RandomState(17).randn(1, 144, 4).astype(np.float32)
                )
                mask = runtime._pad_last(np.ones((1, 1, 4), dtype=np.float32))
                conditional_text = runtime._pad_last(text).mean(axis=1, keepdims=True)
                for step in range(steps):
                    time = engine.time_table[step].mean() * .001
                    conditional = latent * .1 + conditional_text * .01 + 2 * .02 + 3 * .03 + time
                    unconditional = latent * .1 - .01 - 3 * .02 - 2 * .03 + time
                    latent = ((latent + (4 * conditional - 3 * unconditional) / steps)
                              * mask).astype(np.float32)
                np.testing.assert_allclose(actual.waveform, np.repeat(latent[0, 0, :4], 2),
                                           rtol=1e-6, atol=1e-6)
                self.assertEqual(len(calls), steps)
                for values in calls:
                    for index in (0, 4, 5, 6, 7):
                        np.testing.assert_array_equal(values[index][0], values[index][1])
                    self.assertTrue(np.all(values[2][0] == 2))
                    self.assertTrue(np.all(values[2][1] == -3))
                    self.assertTrue(np.all(values[3][0] == 3))
                    self.assertTrue(np.all(values[3][1] == -2))


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

    def setUp(self):
        import json
        self.assets = json.loads(MODEL_ASSETS.read_text(encoding="utf-8"))

    def test_both_model_zoo_archives_and_runtime_data_have_checksum_pins(self):
        for name, asset in self.assets.items():
            self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
            if name != "runtime_data":
                self.assertTrue(asset["url"].startswith("https://docs.sima.ai/pkg_downloads/"))
        script = _setup_function("_supertonic_checksums") + "\n_supertonic_checksums"
        result = subprocess.run(
            ["bash", "-c", script], env={**os.environ, "EXAMPLE_DIR": str(EXAMPLE)},
            capture_output=True, text=True, check=True,
        )
        for asset in self.assets.values():
            self.assertIn(f"{asset['sha256']}  supertonic-3-sima/{asset['file']}", result.stdout)

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

    def test_worker_passes_model_paths_and_checksums_from_the_shared_pins(self):
        import json
        import types
        from unittest import mock
        tree = ast.parse(WORKER.read_text())
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_engine")
        namespace = {"os": os, "Path": Path, "json": json, "MODEL_ASSETS": MODEL_ASSETS}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(WORKER), "exec"), namespace)
        with mock.patch.dict(os.environ, {"SUPERTONIC_MODELS_ROOT": "/models", "SUPERTONIC_STEPS": "8"}):
            options = namespace["_build_engine"](types.SimpleNamespace(
                SupertonicModalix=lambda **kwargs: kwargs
            ))
        for name, field in (("vector_estimator", "mpk_path"), ("vocoder", "vocoder_mpk_path"),
                            ("runtime_data", "runtime_data_path")):
            self.assertEqual(options[field], Path("/models/supertonic-3-sima") / self.assets[name]["file"])
            self.assertEqual(options["artifact_sha256"][name], self.assets[name]["sha256"])
        self.assertTrue(options["verify_hashes"])

    def test_engine_rejects_a_checksum_mismatch_for_each_asset(self):
        import hashlib
        import types
        tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SupertonicModalix")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_artifacts")
        sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        namespace = {"_sha256": sha}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENGINE), "exec"), namespace)
        verify = namespace["_verify_artifacts"]
        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: Path(tmp) / name for name in self.assets}
            for name, path in paths.items():
                path.write_bytes(name.encode())
            engine = types.SimpleNamespace(
                mpk_path=paths["vector_estimator"], runtime_data_path=paths["runtime_data"],
                vocoder_mpk_path=paths["vocoder"], vocoder_backend="mla",
                artifact_sha256={name: sha(path) for name, path in paths.items()},
            )
            verify(engine)
            for name, path in paths.items():
                with self.subTest(asset=name):
                    path.write_bytes(b"changed")
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        verify(engine)
                    path.write_bytes(name.encode())



if __name__ == "__main__":
    unittest.main()
