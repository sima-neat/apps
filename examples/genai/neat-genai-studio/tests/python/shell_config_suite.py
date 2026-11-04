"""The shared shell helper setup.sh and run.sh use to read the persisted
Supertonic paths (src/common/config_value.sh), exercised through bash."""

from __future__ import annotations

import re
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
