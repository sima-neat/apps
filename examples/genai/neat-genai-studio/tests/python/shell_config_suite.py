"""The shared shell helper setup.sh and run.sh use to read the persisted
Supertonic paths (src/common/config_value.sh), exercised through bash."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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
    port: 5000   # the backend API
    https: "True"
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
        self.assertEqual(self._web("https"), "true")          # normalized by the loader
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
        four = ("app:\n    web:\n        https: false\n"
                "    tts:\n        supertonic:\n            models_root: /m4\n")
        top = "web:\n  https: yes\ntts:\n  supertonic:\n    venv: /v\n"
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "four.yaml"; a.write_text(four, encoding="utf-8")
            b = Path(tmp) / "top.yaml"; b.write_text(top, encoding="utf-8")
            self.assertEqual(_run('web_config_scalar "$1" https', str(a)), "false")
            self.assertEqual(_run('supertonic_config_value "$1" models_root', str(a)), "/m4")
            self.assertEqual(_run('web_config_scalar "$1" https', str(b)), "true")
            self.assertEqual(_run('supertonic_config_value "$1" venv', str(b)), "/v")
        # the Python loader agrees on the same files
        import sys
        sys.path.insert(0, str(HELPER.parents[2] / "src" / "python"))
        from shared.config import load_ui_config
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "four.yaml"; a.write_text(four, encoding="utf-8")
            cfg = load_ui_config(a, Path(tmp))
            self.assertEqual((cfg.web.https, cfg.supertonic.models_root), (False, "/m4"))

    def test_flow_style_mapping_reads_like_the_loader(self):
        text = ('app:\n  web: {host: 0.0.0.0, port: 5000, https: "false",'
                ' cors_origins: [http://a:3000, "https://b"]}\n'
                '  tts: {supertonic: {models_root: /flow, venv: /fv}}\n')
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "flow.yaml"; c.write_text(text, encoding="utf-8")
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
            script = 'source "$1"; _studio_config_python() { return 1; }; web_config_scalar "$2" https'
            out = subprocess.run(["bash", "-c", script, "_", str(HELPER), str(c)],
                                 check=True, capture_output=True, text=True, env=env)
            self.assertEqual(out.stdout.strip(), "True")

    def test_truthiness_matches_the_python_loader(self):
        for value in ("1", "true", "True", "YES", "on"):
            self.assertEqual(_run('config_true "$1" && echo y || echo n', value), "y", value)
        for value in ("0", "false", "", "no", "off", "'true'"):
            self.assertEqual(_run('config_true "$1" && echo y || echo n', value), "n", value)


class ShellRagConfigTests(unittest.TestCase):
    """rag_config_enabled, which setup.sh reads to keep a SETUP_RAG choice."""

    def _enabled(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text(text, encoding="utf-8")
            return _run('rag_config_enabled "$1"', str(config))

    def test_values_read_like_the_loader(self):
        self.assertEqual(self._enabled("app:\n  rag:\n    enabled: true\n"), "true")
        self.assertEqual(self._enabled("app:\n  rag:\n    enabled: false   # SETUP_RAG=1\n"), "false")
        self.assertEqual(self._enabled("app:\n  rag: {enabled: 'yes'}\n"), "true")
        self.assertEqual(self._enabled("rag:\n  enabled: on\n"), "true")             # no app: root
        self.assertEqual(self._enabled("app:\n  web:\n    port: 5000\n"), "false")   # loader default

    def test_fallback_reader_without_a_yaml_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "c.yaml"
            c.write_text("app:\n  rag:\n    enabled: false   # SETUP_RAG=1\n", encoding="utf-8")
            script = 'source "$1"; _studio_config_python() { return 1; }; rag_config_enabled "$2"'
            out = subprocess.run(["bash", "-c", script, "_", str(HELPER), str(c)],
                                 check=True, capture_output=True, text=True)
            self.assertEqual(out.stdout.strip(), "false")


class ShellResidentLimitConfigTests(unittest.TestCase):
    """resident_config_limit, which setup.sh reads to keep the chat model limit."""

    def _limit(self, text, script='resident_config_limit "$1"', python=None):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.yaml"
            config.write_text(text, encoding="utf-8")
            return _run(script, str(config), python=python)

    def test_values_read_like_the_loader(self):
        self.assertEqual(self._limit("server:\n  models:\n    max_resident_chat_models: 3\n"), "3")
        self.assertEqual(self._limit("server:\n  models: {max_resident_chat_models: 4}\n"), "4")
        self.assertEqual(self._limit("server:\n  models:\n    catalog_dir: /m\n"), "2")   # loader default

    def test_fallback_reader_without_a_yaml_python(self):
        script = '_studio_config_python() { return 1; }; resident_config_limit "$1"'
        text = "server:\n  models:\n    max_resident_chat_models: 3   # small models\napp:\n  web:\n    port: 5000\n"
        self.assertEqual(self._limit(text, script), "3")
        self.assertEqual(self._limit("server:\n  models:\n    catalog_dir: /m\n", script), "")


class LegacyUiDirMigrationTests(unittest.TestCase):
    """run.sh moves an updated install's own files from src/python/ui/ (the
    backend folder before it became api/) into api/ on the next start."""

    EXAMPLE = Path(__file__).resolve().parents[2]

    def _migrate(self, python_dir: Path) -> subprocess.CompletedProcess:
        run_sh = (self.EXAMPLE / "run.sh").read_text(encoding="utf-8")
        start = run_sh.index("migrate_legacy_ui_dir() {")
        end = run_sh.index("\n}\n", start) + 3
        script = ('ok() { echo "OK $*"; }; warn() { echo "WARN $*" >&2; }\n'
                  f'PYTHON_DIR="$1"\n{run_sh[start:end]}\nmigrate_legacy_ui_dir\n')
        return subprocess.run(["bash", "-c", script, "_", str(python_dir)],
                              check=True, capture_output=True, text=True)

    def test_install_files_move_and_regenerated_ones_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp)
            files = {"assets/en_US-x.onnx": "voice", "assets/piper-plus/ja/m.onnx": "ja",
                     "milvus.db": "db", "milvus.meta.json": "{}", "uploads/doc.md": "# doc",
                     "server.log": "log", ".milvus.db.lock": "", "__pycache__/x.pyc": "c"}
            for rel, text in files.items():
                (py / "ui" / rel).parent.mkdir(parents=True, exist_ok=True)
                (py / "ui" / rel).write_text(text)
            (py / "api" / "assets").mkdir(parents=True)

            out = self._migrate(py)

            self.assertFalse((py / "ui").exists())
            for rel in ("assets/en_US-x.onnx", "assets/piper-plus/ja/m.onnx", "milvus.db",
                        "milvus.meta.json", "uploads/doc.md", "server.log"):
                self.assertEqual((py / "api" / rel).read_text(), files[rel], rel)
            self.assertFalse((py / "api" / ".milvus.db.lock").exists())
            self.assertFalse((py / "api" / "__pycache__").exists())
            self.assertIn("OK Moved 6 file(s)", out.stdout)

    def test_existing_api_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp)
            for side, db, voice in (("ui", "old documents", "same"), ("api", "new default", "same")):
                (py / side / "assets").mkdir(parents=True)
                (py / side / "milvus.db").write_text(db)
                (py / side / "assets" / "v.onnx").write_text(voice)

            out = self._migrate(py)

            self.assertEqual((py / "api" / "milvus.db").read_text(), "new default")
            self.assertEqual((py / "ui" / "milvus.db").read_text(), "old documents")
            self.assertFalse((py / "ui" / "assets").exists())          # identical copy dropped
            self.assertIn("ui/milvus.db", out.stderr)

    def test_the_split_voices_shim_in_ui_tts_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp)
            (py / "ui" / "tts").mkdir(parents=True)
            (py / "ui" / "tts" / "split_voices.py").write_text("shim")
            (py / "ui" / "milvus.db").write_text("db")
            (py / "api").mkdir()

            out = self._migrate(py)

            self.assertEqual((py / "ui" / "tts" / "split_voices.py").read_text(), "shim")
            self.assertFalse((py / "api" / "tts").exists())
            self.assertEqual((py / "api" / "milvus.db").read_text(), "db")
            self.assertEqual(out.stderr, "")

    def test_update_keeps_install_files_in_ui_until_they_move(self):
        # Every install-owned api/ path the update mirror protects is also
        # protected under ui/, so a second update before the first start
        # cannot delete files that have not moved yet.
        run_sh = (self.EXAMPLE / "run.sh").read_text(encoding="utf-8")
        api = re.findall(r"--exclude='/src/python/api/([^']+)'", run_sh)
        ui = re.findall(r"--exclude='/src/python/ui/([^']+)'", run_sh)
        self.assertTrue(api)
        self.assertEqual(sorted(api), sorted(ui))

    def test_the_old_updater_path_forwards_to_the_splitter(self):
        shim = self.EXAMPLE / "src" / "python" / "ui" / "tts" / "split_voices.py"
        with tempfile.TemporaryDirectory() as tmp:
            py = Path(tmp)
            (py / "ui" / "tts").mkdir(parents=True)
            (py / "api" / "tts").mkdir(parents=True)
            (py / "ui" / "tts" / "split_voices.py").write_text(shim.read_text(encoding="utf-8"))
            (py / "api" / "tts" / "split_voices.py").write_text(
                "import sys\nprint('split', sys.argv[1:])\n")
            out = subprocess.run([sys.executable, str(py / "ui" / "tts" / "split_voices.py"), "/assets"],
                                 check=True, capture_output=True, text=True)
            self.assertEqual(out.stdout.strip(), "split ['/assets']")

    def test_nothing_happens_without_a_ui_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "api").mkdir()
            out = self._migrate(Path(tmp))
            self.assertEqual((out.stdout, out.stderr), ("", ""))


# The usage of get-example.sh before it fetched the published bundle, as its --help printed it.
SOURCE_ARCHIVE_SCRIPT = '''#!/usr/bin/env bash
usage() {
  cat <<'USAGE'
Usage:
  get-example.sh [--branch <branch>] [--dest <dir>] [--force] <example>

Examples:
  get-example.sh neat-genai-studio
  get-example.sh genai/neat-genai-studio
  get-example.sh examples/genai/neat-genai-studio

Environment:
  NEAT_APPS_REPO_URL          Apps Git repository URL
                              default: https://github.com/sima-neat/apps.git
  NEAT_APPS_ARCHIVE_URL       Prebuilt source archive URL or local .tar.gz path
                              default: derived from NEAT_APPS_REPO_URL and branch
  NEAT_APPS_BRANCH            Git branch to fetch
                              default: main
  NEAT_APPS_EXAMPLE_DEST_DIR  Destination directory
                              default: ./<example-name>
USAGE
}
case "$1" in -h|--help) usage; exit 0 ;; esac
exit 1
'''


class RawTreeServer:
    """Serves a raw tree over HTTP, 404 for a missing file, as raw.githubusercontent.com does."""

    def __init__(self, root: Path) -> None:
        directory = str(root)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def __enter__(self) -> "RawTreeServer":
        return self

    def __exit__(self, *args) -> None:
        self.httpd.shutdown()


class StandaloneUpdateTests(unittest.TestCase):
    """run.sh update outside a checkout: which ref it fetches and which
    get-example.sh it runs, against a local install, raw tree and index."""

    EXAMPLE = Path(__file__).resolve().parents[2]
    GET_EXAMPLE = EXAMPLE.parents[2] / "scripts" / "get-example.sh"

    def _functions(self) -> str:
        run_sh = (self.EXAMPLE / "run.sh").read_text(encoding="utf-8")
        start = run_sh.index("apps_root() {")
        end = run_sh.index("\ndo_update() {", start)
        return 'errln() { echo "ERR $*" >&2; }\n' + run_sh[start:end] + "\n"

    def _install(self, root: Path, installed_ref: str | None, local_copy: bool) -> Path:
        """A Studio in root/out laid out as get-example.sh and sima-cli leave it."""
        apps = root / "out"
        studio = apps / "examples" / "genai" / "neat-genai-studio"
        studio.mkdir(parents=True)
        if installed_ref:
            (apps / "deps").mkdir()
            (apps / "deps" / "neat-apps.json").write_text(json.dumps(
                {"neat-apps": {"ref": installed_ref, "spec": "0" * 40}}, indent=2))
        if local_copy:
            (apps / "scripts").mkdir()
            (apps / "scripts" / "get-example.sh").write_text(self.GET_EXAMPLE.read_text())
        return studio

    def _raw(self, root: Path, scripts: dict[str, str]) -> Path:
        for ref, text in scripts.items():
            path = root / "raw" / ref / "scripts" / "get-example.sh"
            path.parent.mkdir(parents=True)
            path.write_text(text)
        return root / "raw"

    def _select(self, root: Path, studio: Path, raw: Path | str, **env: str):
        script = (self._functions() + 'EXAMPLE_DIR="$1"\nref="$(update_ref)"\necho "ref=${ref}"\n'
                  'select_get_example "${ref}" "$2/get-example.sh"\n')
        run_env = {"PATH": os.environ["PATH"], "NEAT_APPS_RAW_URL": str(raw), **env}
        return subprocess.run(["bash", "-c", script, "_", str(studio), str(root)],
                              capture_output=True, text=True, env=run_env)

    def test_the_installed_copy_fetches_the_installed_ref(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, "team/demo", local_copy=True)
            raw = self._raw(root, {"team/demo": SOURCE_ARCHIVE_SCRIPT})

            out = self._select(root, studio, raw)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.splitlines(),
                             ["ref=team/demo", str(root / "out" / "scripts" / "get-example.sh")])

    def test_a_named_script_wins_over_the_installed_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, "team/demo", local_copy=True)
            mine = root / "mine.sh"
            mine.write_text(self.GET_EXAMPLE.read_text())

            out = self._select(root, studio, root / "raw", NEAT_APPS_GET_EXAMPLE=str(mine))
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.splitlines(), ["ref=team/demo", str(mine)])

    def test_the_installed_copy_fetches_a_tag_from_its_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, "develop", local_copy=True)
            tree = root / "tree" / "prebuilt-apps"
            (tree / "examples" / "genai" / "neat-genai-studio").mkdir(parents=True)
            (tree / "examples" / "genai" / "neat-genai-studio" / "run.sh").write_text("new")
            (tree / "scripts").mkdir()
            (tree / "scripts" / "get-example.sh").write_text(self.GET_EXAMPLE.read_text())
            build = root / "index" / "apps" / "v1.0" / "abc1234def56"
            build.mkdir(parents=True)
            bundle = build / "neat-apps-v1.0-1.0.tar.gz"
            with tarfile.open(bundle, "w:gz") as tar:
                tar.add(tree, arcname="prebuilt-apps")
            digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
            (build / "metadata.json").write_text(json.dumps(
                {"resources-checksum": {bundle.name: digest}}, indent=4))
            (build.parent / "latest.tag").write_text("abc1234def56\n")

            out = self._select(root, studio, root / "raw", NEAT_APPS_BRANCH="v1.0")
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.splitlines()[0], "ref=v1.0")
            fetched = subprocess.run(
                ["bash", str(root / "get-example.sh"), "--ref", "v1.0", "--dest", "update", "neat-genai-studio"],
                cwd=root, capture_output=True, text=True,
                env={"PATH": os.environ["PATH"], "HOME": str(root / "home"),
                     "NEAT_APPS_ARTIFACTS_URL": str(root / "index")})
            self.assertEqual(fetched.returncode, 0, fetched.stderr)
            update = root / "update"
            self.assertEqual((update / "examples" / "genai" / "neat-genai-studio" / "run.sh").read_text(), "new")
            self.assertTrue((update / "scripts" / "get-example.sh").is_file())

    def test_without_a_copy_the_target_refs_script_is_downloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, "team/demo", local_copy=False)
            raw = self._raw(root, {"v1.0": self.GET_EXAMPLE.read_text(), "team/demo": SOURCE_ARCHIVE_SCRIPT})

            for branch in ("v1.0", "v1.0:" + "a" * 40):
                with self.subTest(branch=branch), RawTreeServer(raw) as server:
                    out = self._select(root, studio, server.url, NEAT_APPS_BRANCH=branch)
                    self.assertEqual(out.returncode, 0, out.stderr)
                    self.assertEqual(out.stdout.splitlines(),
                                     [f"ref={branch}", f"{server.url}/v1.0/scripts/get-example.sh"])

    def test_a_missing_named_script_is_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, "team/demo", local_copy=True)

            out = self._select(root, studio, root / "raw", NEAT_APPS_GET_EXAMPLE=str(root / "none.sh"))
            self.assertEqual(out.returncode, 1)
            self.assertIn(f"ERR Not found: {root / 'none.sh'}", out.stderr)

    def test_without_curl_the_script_is_downloaded_with_wget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, None, local_copy=False)
            raw = self._raw(root, {"v1.0": self.GET_EXAMPLE.read_text()})
            no_curl = root / "bin"
            no_curl.mkdir()
            for folder in ("/usr/bin", "/bin"):
                for tool in Path(folder).iterdir():
                    if tool.name != "curl" and not (no_curl / tool.name).exists():
                        (no_curl / tool.name).symlink_to(tool)

            with RawTreeServer(raw) as server:
                out = self._select(root, studio, server.url, NEAT_APPS_BRANCH="v1.0", PATH=str(no_curl))
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.splitlines()[-1], f"{server.url}/v1.0/scripts/get-example.sh")

    def test_a_missing_download_fails_as_a_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._install(root, None, local_copy=False)
            raw = self._raw(root, {"v1.0": self.GET_EXAMPLE.read_text()})
            no_curl = root / "bin"
            no_curl.mkdir()
            for folder in ("/usr/bin", "/bin"):
                for tool in Path(folder).iterdir():
                    if tool.name != "curl" and not (no_curl / tool.name).exists():
                        (no_curl / tool.name).symlink_to(tool)

            for path in (os.environ["PATH"], str(no_curl)):
                with self.subTest(curl=path != str(no_curl)), RawTreeServer(raw) as server:
                    out = self._select(root, studio, server.url, NEAT_APPS_BRANCH="gone", PATH=path)
                    self.assertEqual(out.returncode, 1)
                    self.assertIn(f"ERR Download failed: {server.url}/gone/scripts/get-example.sh", out.stderr)

    def test_an_old_layout_ignores_folders_above_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # A Studio fetched on its own into <dest>/, inside a user folder that
            # happens to have scripts/ and deps/ three levels up.
            studio = root / "user" / "a" / "b" / "neat-genai-studio"
            studio.mkdir(parents=True)
            (root / "user" / "scripts").mkdir()
            (root / "user" / "scripts" / "get-example.sh").write_text(self.GET_EXAMPLE.read_text())
            (root / "user" / "deps").mkdir()
            (root / "user" / "deps" / "neat-apps.json").write_text('{"neat-apps": {"ref": "decoy"}}')
            raw = self._raw(root, {"main": self.GET_EXAMPLE.read_text()})

            out = self._select(root, studio, raw)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.splitlines(), ["ref=main", f"{raw}/main/scripts/get-example.sh"])

    def test_a_source_archive_script_stops_with_the_way_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            installed = self._install(root / "installed", None, local_copy=False)
            alone = root / "alone" / "neat-genai-studio"
            alone.mkdir(parents=True)
            raw = self._raw(root, {"main": SOURCE_ARCHIVE_SCRIPT, "develop": self.GET_EXAMPLE.read_text()})

            for studio, hint in ((installed, "sima-cli neat install apps"),
                                 (alone, "Move this folder away and fetch the Studio again with get-example.sh")):
                with self.subTest(studio=studio.parent.name):
                    out = self._select(root, studio, raw)
                    self.assertEqual(out.returncode, 1)
                    self.assertEqual(out.stdout.splitlines(), ["ref=main"])
                    self.assertIn("does not fetch the published bundle", out.stderr)
                    self.assertIn(hint, out.stderr)
                    # main is the ref that failed, so the hint names another one.
                    self.assertIn("NEAT_APPS_BRANCH=<branch or tag> ./run.sh update", out.stderr)
                    self.assertNotIn("NEAT_APPS_BRANCH=main", out.stderr)


class StandaloneUpdateRunTests(unittest.TestCase):
    """The whole `run.sh update` on a Studio laid out as sima-cli and
    get-example.sh leave it, against a local artifacts index."""

    EXAMPLE = Path(__file__).resolve().parents[2]
    GET_EXAMPLE = EXAMPLE.parents[2] / "scripts" / "get-example.sh"
    # The Studio's source, without tests or anything an install creates.
    IGNORE = staticmethod(shutil.ignore_patterns("tests", "__pycache__", ".venv*", "config.local.yaml*", "*.log",
                                    "*.onnx", "*.onnx.json", "milvus.*", "uploads", "piper-plus",
                                    "mms-tts-kor", ".local-certs", ".neat-genai-studio.*"))

    def _setup(self, root: Path) -> Path:
        apps = root / "apps"
        studio = apps / "examples" / "genai" / "neat-genai-studio"
        shutil.copytree(self.EXAMPLE, studio, ignore=self.IGNORE)
        (studio / "config.local.yaml").write_text("app: {}\n")
        (studio / "src" / "python" / "stale.py").write_text("")
        (apps / "scripts").mkdir()
        shutil.copy2(self.GET_EXAMPLE, apps / "scripts" / "get-example.sh")
        (apps / "deps").mkdir()
        (apps / "deps" / "neat-apps.json").write_text(json.dumps(
            {"neat-apps": {"ref": "team/demo", "spec": "0" * 40}}))
        # The published bundle for team/demo: the same Studio with a marker.
        tree = root / "tree" / "prebuilt-apps"
        shutil.copytree(self.EXAMPLE, tree / "examples" / "genai" / "neat-genai-studio", ignore=self.IGNORE)
        (tree / "examples" / "genai" / "neat-genai-studio" / "MARKER").write_text("new")
        build = root / "index" / "apps" / "team%252Fdemo" / "abc1234def56"
        build.mkdir(parents=True)
        bundle = build / "neat-apps-team-demo-abc1234.tar.gz"
        with tarfile.open(bundle, "w:gz") as tar:
            tar.add(tree, arcname="prebuilt-apps")
        (build / "metadata.json").write_text(json.dumps(
            {"resources-checksum": {bundle.name: hashlib.sha256(bundle.read_bytes()).hexdigest()}}))
        (build.parent / "latest.tag").write_text("abc1234def56\n")
        return studio

    def _update(self, root: Path, studio: Path, path: str = "", **env: str):
        run_env = {"PATH": path + os.environ["PATH"], "HOME": str(root / "home"),
                   "NEAT_APPS_ARTIFACTS_URL": str(root / "index"), **env}
        return subprocess.run(["bash", str(studio / "run.sh"), "update"], cwd=studio,
                              capture_output=True, text=True, env=run_env)

    @unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
    def test_update_mirrors_the_bundle_and_keeps_the_install_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._setup(root)

            out = self._update(root, studio)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("Source updated from team/demo.", out.stdout)
            self.assertEqual((studio / "MARKER").read_text(), "new")
            self.assertEqual((studio / "config.local.yaml").read_text(), "app: {}\n")
            self.assertFalse((studio / "src" / "python" / "stale.py").exists())

    @unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
    def test_update_through_a_symlink_finds_the_install_and_ignores_fetch_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "link").symlink_to(self._setup(root))
            studio = root / "link"

            out = self._update(root, studio, NEAT_APPS_ARCHIVE_URL=str(root / "missing.tar.gz"),
                               NEAT_APPS_EXAMPLE_DEST_DIR=str(root / "elsewhere"),
                               NEAT_APPS_REPO_URL="https://github.com/sima-neat/apps.git")
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            self.assertIn("Source updated from team/demo.", out.stdout)
            self.assertEqual((studio / "MARKER").read_text(), "new")

    @unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
    def test_a_failed_fetch_leaves_the_studio_and_names_the_way_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._setup(root)

            (root / "t").mkdir()

            out = self._update(root, studio, NEAT_APPS_BRANCH="gone", TMPDIR=str(root / "t"))
            self.assertEqual(out.returncode, 1)
            self.assertIn("Cannot fetch neat-genai-studio for gone.", out.stderr)
            self.assertIn("Pick another ref with NEAT_APPS_BRANCH=main ./run.sh update", out.stderr)
            self.assertIn("NEAT_APPS_BRANCH=main", out.stderr)
            self.assertIn("NEAT_APPS_GET_EXAMPLE", out.stderr)
            self.assertFalse((studio / "MARKER").exists())
            self.assertTrue((studio / "src" / "python" / "stale.py").exists())
            self.assertEqual(list((root / "t").iterdir()), [])

    @unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
    def test_a_failed_script_choice_leaves_no_temporary_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._setup(root)
            (root / "t").mkdir()

            out = self._update(root, studio, NEAT_APPS_GET_EXAMPLE=str(root / "none.sh"), TMPDIR=str(root / "t"))
            self.assertEqual(out.returncode, 1)
            self.assertIn("Not found:", out.stderr)
            self.assertEqual(list((root / "t").iterdir()), [])

    def test_a_stopped_update_leaves_no_temporary_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = self._setup(root)
            (root / "t").mkdir()
            # A get-example.sh that stops run.sh with SIGTERM while it fetches.
            stop = root / "stop.sh"
            stop.write_text('#!/bin/bash\ncase "$1" in --help) echo "--ref"; exit 0 ;; esac\nkill -TERM $PPID\nexit 1\n')

            out = self._update(root, studio, NEAT_APPS_GET_EXAMPLE=str(stop), TMPDIR=str(root / "t"))
            self.assertNotEqual(out.returncode, 0)
            self.assertEqual(list((root / "t").iterdir()), [])

    def test_a_studio_fetched_alone_ignores_a_git_folder_above_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            studio = root / "user" / "a" / "b" / "neat-genai-studio"
            shutil.copytree(self.EXAMPLE, studio, ignore=self.IGNORE)
            (root / "user" / ".git").mkdir()
            fail = root / "fail.sh"
            fail.write_text('#!/bin/bash\ncase "$1" in --help) echo "--ref"; exit 0 ;; esac\nexit 1\n')

            out = self._update(root, studio, NEAT_APPS_GET_EXAMPLE=str(fail))
            self.assertEqual(out.returncode, 1)
            self.assertNotIn("git cannot read it", out.stderr)
            self.assertIn("Cannot fetch neat-genai-studio for main.", out.stderr)

    def test_a_checkout_git_cannot_read_is_never_mirrored(self):
        for git_entry in ("folder", "file"):
            with self.subTest(git=git_entry), tempfile.TemporaryDirectory() as tmp:
                self._check_unreadable_checkout(Path(tmp).resolve(), git_entry)

    def _check_unreadable_checkout(self, root: Path, git_entry: str):
        studio = self._setup(root)
        if git_entry == "folder":
            (root / "apps" / ".git").mkdir()
        else:
            # A worktree's .git is a file that names the real repository.
            (root / "apps" / ".git").write_text("gitdir: /elsewhere\n")
        fake = root / "fakebin"
        fake.mkdir()
        (fake / "git").write_text("#!/bin/sh\necho 'fatal: detected dubious ownership' >&2\nexit 128\n")
        (fake / "git").chmod(0o755)

        out = self._update(root, studio, path=f"{fake}:")
        self.assertEqual(out.returncode, 1)
        self.assertIn("git cannot read it", out.stderr)
        self.assertFalse((studio / "MARKER").exists())
        self.assertTrue((studio / "src" / "python" / "stale.py").exists())


class HfDownloadRetryTests(unittest.TestCase):
    """setup.sh's hf_download retries what the Hub may answer differently a
    moment later (#599) and stops at once on anything else."""

    EXAMPLE = Path(__file__).resolve().parents[2]
    # A fake `hf`: each call answers with the next line of the plan, either
    # "ok" (prints the path) or an error message on stderr with exit 1.
    FAKE_HF = (
        '#!/bin/bash\n'
        'n=$(( $(cat "$STATE/calls" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$STATE/calls"\n'
        'line=$(sed -n "${n}p" "$STATE/plan")\n'
        'if [[ "$line" == ok ]]; then echo "/models/$2"; exit 0; fi\n'
        'printf "%b\\n" "$line" >&2; exit 1\n'
    )

    def _download(self, plan):
        setup = (self.EXAMPLE / "setup.sh").read_text(encoding="utf-8")
        start = setup.index("hf_download() {")
        end = setup.index("\n}\n", start) + 3
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / "plan").write_text("\n".join(plan) + "\n")
            hf = state / "hf"
            hf.write_text(self.FAKE_HF)
            hf.chmod(0o755)
            script = ('set -euo pipefail\n'
                      'warn() { echo "WARN $*" >&2; }\n'
                      'sleep() { echo "$1" >> "$STATE/sleeps"; }\n'
                      f'{setup[start:end]}\n'
                      'if hf_download "$STATE/hf" simaai/m --local-dir /x; then echo RC=0; else echo RC=1; fi\n')
            out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                                 env=dict(os.environ, STATE=str(state)))
            calls = int((state / "calls").read_text())
            sleeps = (state / "sleeps").read_text().split() if (state / "sleeps").exists() else []
            return out, calls, [int(x) for x in sleeps]

    def test_a_rate_limit_is_retried_until_the_download_succeeds(self):
        out, calls, sleeps = self._download(
            ["429 Client Error: Too Many Requests for url: https://huggingface.co/x", "ok"])
        self.assertIn("RC=0", out.stdout)
        self.assertIn("/models/simaai/m", out.stdout)       # hf's output still reaches stdout
        self.assertIn("429 Client Error", out.stderr)       # and its messages the terminal
        self.assertEqual((calls, sleeps), (2, [15]))

    def test_server_errors_and_dropped_connections_are_retried(self):
        out, calls, sleeps = self._download(
            ["503 Server Error: Service Unavailable", "ConnectionError: Connection reset by peer", "ok"])
        self.assertIn("RC=0", out.stdout)
        self.assertEqual((calls, sleeps), (3, [15, 30]))

    def test_a_dropped_tls_connection_is_retried(self):
        out, calls, sleeps = self._download(
            ["requests.exceptions.SSLError: EOF occurred in violation of protocol (_ssl.c:2427)", "ok"])
        self.assertEqual((out.stdout.split()[-1], calls, sleeps), ("RC=0", 2, [15]))

    def test_a_wrong_repo_fails_at_once(self):
        out, calls, sleeps = self._download(["Repository Not Found for url: https://huggingface.co/x"])
        self.assertIn("RC=1", out.stdout)
        self.assertEqual((calls, sleeps), (1, []))

    def test_hf_tracebacks_are_classified_by_their_error_lines(self):
        # A real hf failure prints a traceback whose indented source lines
        # mention timeouts and ConnectionError even for a missing repo.
        trace = r"Traceback (most recent call last):\n    url=url, timeout=etag_timeout\n    except ConnectionError:\n"
        out, calls, _ = self._download([trace + "huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error"])
        self.assertEqual((out.stdout.split()[-1], calls), ("RC=1", 1))
        out, calls, sleeps = self._download(
            [trace + "huggingface_hub.errors.LocalEntryNotFoundError: An error happened", "ok"])
        self.assertEqual((out.stdout.split()[-1], calls, sleeps), ("RC=0", 2, [15]))

    def test_it_gives_up_after_the_last_attempt(self):
        out, calls, sleeps = self._download(["429 Client Error: Too Many Requests"] * 6)
        self.assertIn("RC=1", out.stdout)
        self.assertEqual((calls, sleeps), (4, [15, 30, 60]))
        self.assertIn("attempt 3 of 4", out.stderr)

    def test_every_hugging_face_download_in_setup_goes_through_it(self):
        setup = (self.EXAMPLE / "setup.sh").read_text(encoding="utf-8")
        body = setup.replace(setup[setup.index("hf_download() {"):setup.index("write_repo_marker() {")], "")
        self.assertNotIn("/bin/hf\" download", body)
        self.assertNotIn('"${hf}" download', body)
        self.assertEqual(body.count("hf_download "), 7)


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



class ServerConfigWithoutPyyamlTests(unittest.TestCase):
    """The model server runs in the Neat environment, which may lack PyYAML
    (Platform 3.0's ~/pyneat). run.sh converts the YAML config to JSON with the
    app's own Python, and the server reads that JSON without importing yaml."""

    EXAMPLE = Path(__file__).resolve().parents[2]

    def _convert(self, config: Path, target: Path) -> None:
        run_sh = (self.EXAMPLE / "run.sh").read_text(encoding="utf-8")
        start = run_sh.index("write_server_config() {")
        end = run_sh.index("\n}\n", start) + 3
        script = (f'APP_PYTHON="{sys.executable}"; CONFIG_PATH="$1"; SERVER_CONFIG_JSON="$2"\n'
                  + run_sh[start:end] + "\nwrite_server_config\n")
        subprocess.run(["bash", "-c", script, "_", str(config), str(target)], check=True)

    def test_server_config_loads_from_json_without_pyyaml(self):
        config = self.EXAMPLE / "src" / "common" / "config.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "server-config.json"
            self._convert(config, target)
            probe = (
                "import sys; sys.modules['yaml'] = None\n"
                "from pathlib import Path\n"
                "from shared.config import load_server_config\n"
                f"cfg = load_server_config(Path({str(target)!r}))\n"
                "print(repr(cfg))\n"
            )
            out = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True,
                                 cwd=self.EXAMPLE / "src" / "python")
            from shared.config import load_server_config
            expected = load_server_config(config)
            # The whole parsed configuration, not only that it loads.
            self.assertEqual(out.stdout.strip(), repr(expected))

    def test_run_sh_starts_the_server_with_the_json_copy(self):
        run_sh = (self.EXAMPLE / "run.sh").read_text(encoding="utf-8")
        self.assertIn('server/main.py" --config "${SERVER_CONFIG_JSON}"', run_sh)
        self.assertNotIn('server/main.py" --config "${CONFIG_PATH}"', run_sh)

    def test_setup_does_not_modify_the_neat_environment(self):
        setup = (self.EXAMPLE / "setup.sh").read_text(encoding="utf-8")
        self.assertNotIn("import yaml'", setup)
        self.assertNotIn('pip install "PyYAML', setup)


class StudioSupervisorTests(unittest.TestCase):
    SUPERVISOR = Path(__file__).resolve().parents[2] / 'src/common/studio_supervisor.sh'

    def test_worker_failure_restarts_children_and_explicit_stop_stays_stopped(self):
        import signal
        import time

        for target, sent_signal in (('supervisor', signal.SIGUSR1),
                                    ('model', signal.SIGKILL),
                                    ('api', signal.SIGKILL),
                                    ('supervisor', signal.SIGTERM)):
            with self.subTest(target=target, signal=sent_signal), tempfile.TemporaryDirectory() as folder:
                ready = Path(folder) / 'ready'
                children = Path(folder) / 'children'
                script = """set -euo pipefail
source "$2"
pids=(); process_groups=(); restart_requested=0
SHUTDOWN_GRACE_SECONDS=1
APP_PYTHON=/bin/true
_self_url=unused; _url=unused
RAG_WORKER_PATTERN="$3/rag-worker"
warn() { :; }; errln() { :; }
ok() { echo ready >> "$READY_FILE"; }; studio_ui_hint() { :; }
start_children() {
  setsid sleep 60 & server_pid=$!; pids[0]=$server_pid
  setsid sleep 60 & api_pid=$!; pids[1]=$api_pid
  setsid "$TEST_PYTHON" -c 'import time; time.sleep(60)' "$RAG_WORKER_PATTERN" &
  rag_pid=$!
  remember_process_group "$server_pid"
  remember_process_group "$api_pid"
  printf '%s %s %s\\n' "$server_pid" "$api_pid" "$rag_pid" >> "$CHILD_FILE"
}
trap 'stop_children' EXIT
trap 'exit 143' TERM
trap 'restart_requested=1' USR1
start_children
supervise_children
"""
                proc = subprocess.Popen(['bash', '-c', script, '_', str(children),
                                         str(self.SUPERVISOR), folder],
                                        env=dict(os.environ, READY_FILE=str(ready),
                                                 CHILD_FILE=str(children), TEST_PYTHON=sys.executable))
                try:
                    def generation(count):
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            lines = ready.read_text().splitlines() if ready.exists() else []
                            if len(lines) >= count:
                                return list(map(int, children.read_text().splitlines()[count - 1].split()))
                            time.sleep(0.05)
                        self.fail('supervisor did not become ready')

                    old = generation(1)
                    pid = {'supervisor': proc.pid, 'model': old[0], 'api': old[1]}[target]
                    os.kill(pid, sent_signal)
                    generations = 1
                    if sent_signal != signal.SIGTERM:
                        new = generation(2)
                        generations = 2
                        self.assertNotEqual(old, new)
                        for pid in old:
                            with self.assertRaises(ProcessLookupError):
                                os.kill(pid, 0)
                        for pid in new:
                            os.kill(pid, 0)
                        old = new
                        proc.terminate()
                    self.assertEqual(proc.wait(timeout=10), 143)
                    self.assertEqual(len(children.read_text().splitlines()), generations)
                    for pid in old:
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=10)

    def test_cleanup_kills_remaining_worker_after_group_leader_dies(self):
        import signal
        import time

        with tempfile.TemporaryDirectory() as folder:
            ready = Path(folder) / 'ready'
            leader = subprocess.Popen(['sleep', '60'], preexec_fn=os.setpgrp)
            self.addCleanup(lambda: leader.poll() is None and leader.kill())
            worker = subprocess.Popen(
                [sys.executable, '-c',
                 'import signal,time,sys; from pathlib import Path; '
                 'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                 'Path(sys.argv[1]).touch(); time.sleep(60)', str(ready)],
                preexec_fn=lambda: os.setpgid(0, leader.pid))
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(ready.exists())
                leader.kill()
                leader.wait(timeout=5)
                script = """set -euo pipefail
source "$1"
pids=(); process_groups=("$2")
RAG_WORKER_PATTERN="$3/unused-rag-worker"
SHUTDOWN_GRACE_SECONDS=0
warn() { :; }
stop_children
"""
                subprocess.run(['bash', '-c', script, '_', str(self.SUPERVISOR),
                                str(leader.pid), folder], check=True, timeout=5)
                self.assertEqual(worker.wait(timeout=5), -signal.SIGKILL)
            finally:
                if worker.poll() is None:
                    worker.kill()
                worker.wait(timeout=5)
                if leader.poll() is None:
                    leader.kill()
                leader.wait(timeout=5)
