"""Stage test models from a persistent NVMe cache, validating source and bytes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_identity(model, modelzoo_version):
    # Registry latest/variants and unversioned modelzoo acquisition must resolve
    # on every run. Never turn a mutable catalog reference into a filename hit.
    if model.get("source") != "url":
        return None
    url = model["url"].replace("{modelzoo_version}", modelzoo_version)
    try:
        with urlopen(Request(url, method="HEAD"), timeout=20) as response:
            headers = response.headers
            validator = headers.get("ETag")
            if (
                not validator
                and headers.get("Last-Modified")
                and headers.get("Content-Length")
            ):
                validator = [headers["Last-Modified"], headers["Content-Length"]]
            if not validator:
                return None
            return {
                "model": model,
                "url": response.url,
                "validator": validator,
                "modelzoo_version": modelzoo_version,
            }
    except OSError:
        # A failed metadata probe cannot authorize stale reuse. The normal
        # downloader below still handles the requested artifact or fails.
        return None


def probe_sources(models, modelzoo_version):
    """Use the CLI's existing authenticated session for protected model URLs."""
    urls = {
        filename: model["url"].replace("{modelzoo_version}", modelzoo_version)
        for filename, model in models.items()
        if model.get("source") == "url"
        and urlparse(model["url"]).hostname in {"docs.sima.ai", "docs-dev.sima.ai"}
    }
    heads = {}
    if urls:
        executable = os.environ.get("SIMA_CLI_BIN") or shutil.which("sima-cli")
        try:
            # The CLI is installed in its own Python environment. Reuse its
            # authenticated downloader dependencies without copying credentials
            # into PyNeat or installing a second CLI into the test environment.
            with open(executable, "rb") as handle:
                shebang = handle.readline(512).decode().strip()
            if not shebang.startswith("#!"):
                raise ValueError("sima-cli has no Python interpreter shebang")
            interpreter = shlex.split(shebang[2:])
            response = subprocess.run(
                [*interpreter, str(Path(__file__).resolve()), "--probe"],
                input=json.dumps(list(urls.values())),
                text=True,
                capture_output=True,
                timeout=30 + 15 * len(urls),
                check=False,
            )
            if response.returncode == 78:
                raise RuntimeError(
                    "Developer Portal model access is not provisioned on this runner; "
                    "configure it through Kerrigan before admitting model tests"
                )
            response.check_returncode()
            heads = json.loads(response.stdout)
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            print(
                "[cache bypass] authenticated source metadata unavailable", flush=True
            )
    identities = {}
    for filename, model in models.items():
        if filename not in urls:
            identities[filename] = source_identity(model, modelzoo_version)
            continue
        head = heads.get(urls[filename])
        identities[filename] = (
            {"model": model, "modelzoo_version": modelzoo_version, **head}
            if head
            else None
        )
    return identities


def authenticated_probe():
    import contextlib
    import io

    from sima_cli.auth.auth0 import get_auth_config, load_tokens, refresh_access_token
    from sima_cli.auth.devportal import validate_session

    urls = json.load(sys.stdin)
    result = {}
    with contextlib.redirect_stdout(io.StringIO()):
        session, valid = validate_session()
        if not valid:
            tokens = load_tokens() or {}
            if tokens.get("refresh_token"):
                refresh_access_token(get_auth_config(), tokens["refresh_token"])
                session, valid = validate_session()
        if not valid:
            # Hardware CI must never wait for an interactive device-login flow.
            raise SystemExit(78)
        if session is not None:
            for url in urls:
                try:
                    with session.head(
                        url, timeout=10, allow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        headers = response.headers
                        validator = headers.get("ETag")
                        if (
                            not validator
                            and headers.get("Last-Modified")
                            and headers.get("Content-Length")
                        ):
                            validator = [
                                headers["Last-Modified"],
                                headers["Content-Length"],
                            ]
                        if validator and "text/html" not in headers.get(
                            "Content-Type", ""
                        ):
                            result[url] = {"url": response.url, "validator": validator}
                except OSError:
                    pass
    print(json.dumps(result))


def prepare_models(root, scope, scope_file, destination, cache_root):
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "deps/manifest.json").read_text())
    variant_mode = os.environ.get("SIMANEAT_APPS_TEST_MODEL_VARIANTS") == "1"
    models = {}
    languages = set()
    for entry in scope["examples"].values():
        for language, config in entry.get("e2e", {}).items():
            if not config.get("enabled"):
                continue
            languages.add(language)
            selected = config.get("models", []) + (
                config.get("variants", []) if variant_mode else []
            )
            for model_id in selected:
                model = entry["models"][model_id]
                filename = model.get("file")
                if not filename:
                    continue  # Still acquired by the normal scoped downloader.
                if Path(filename).name != filename or filename in {".", ".."}:
                    raise ValueError("Model cache destinations must be plain filenames")
                if filename in models and models[filename] != model:
                    raise ValueError(
                        f"Conflicting acquisition identities for {filename}"
                    )
                models[filename] = model
    identities = {}
    hits = 0
    hit_files = set()
    source_identities = probe_sources(models, str(manifest["modelzoo-version"]))
    for filename, model in models.items():
        identity = source_identities[filename]
        if identity is None:
            print(
                f"[cache bypass] {filename}: mutable source or no remote validator",
                flush=True,
            )
            continue
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        identities[filename] = (key, identity)
        entry = cache_root / key
        with (cache_root / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                metadata = json.loads((entry / "metadata.json").read_text())
                cached = entry / "model.tar.gz"
                if (
                    metadata["identity"] == identity
                    and digest(cached) == metadata["sha256"]
                ):
                    shutil.copyfile(cached, destination / filename)
                    hits += 1
                    hit_files.add(filename)
                    print(f"[cache hit] {filename}", flush=True)
            except (OSError, ValueError, KeyError):
                pass
    if languages:
        command = [
            "bash",
            str(root / "scripts/download_models.sh"),
            "--scope-file",
            str(scope_file),
            "--kind",
            "e2e",
        ]
        for language in sorted(languages):
            command.extend(["--language", language])
        subprocess.run(
            command,
            check=True,
            cwd=root,
            env={**os.environ, "MODELS_DIR": str(destination)},
        )
    rechecked = probe_sources(
        {f: models[f] for f in identities if f not in hit_files},
        str(manifest["modelzoo-version"]),
    )
    for filename, (key, identity) in identities.items():
        if filename in hit_files:
            continue
        source = destination / filename
        if not source.is_file():
            raise ValueError(f"Downloader did not materialize {filename}")
        # Revalidate after transfer: if a mutable URL changed while downloading,
        # use the acquired file for this run but do not publish it under an old key.
        if rechecked[filename] != identity:
            print(
                f"[cache bypass] {filename}: source changed during acquisition",
                flush=True,
            )
            continue
        with (cache_root / (key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entry = cache_root / key
            entry.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=cache_root, prefix=".stage-") as stage:
                staged = Path(stage)
                shutil.copyfile(source, staged / "model.tar.gz")
                (staged / "metadata.json").write_text(
                    json.dumps(
                        {
                            "identity": identity,
                            "sha256": digest(staged / "model.tar.gz"),
                        }
                    )
                )
                os.replace(staged / "model.tar.gz", entry / "model.tar.gz")
                os.replace(staged / "metadata.json", entry / "metadata.json")
    result = {
        "hits": hits,
        "models": len(models),
        "seconds": time.monotonic() - started,
    }
    print(
        f"Model cache: {hits}/{len(models)} hits in {result['seconds']:.1f}s",
        flush=True,
    )
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit("This module is called by run_runtime_shard.py")
    authenticated_probe()
