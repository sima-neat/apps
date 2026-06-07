#!/usr/bin/env python3
"""End-to-end vectordb test:
1) generate markdown,
2) upload to Gradio and wait for DB sync,
3) restart vectordb service,
4) verify expected search result.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vectordb.gradioclient import upload_and_process_file
from vectordb.vectordb import RagDbClient


USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _log(msg: str) -> None:
    print(f"{_c('[e2e]', '1;36')} {msg}", flush=True)


def _log_request(method: str, url: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    _log(f"{_c('REQUEST', '1;34')} {method} {url}{suffix}")


def _log_response(status: str, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    _log(f"{_c('RESPONSE', '1;32')} {status}{suffix}")


def generate_markdown_file(output_dir: Path, expected_phrase: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "generated_test_doc.md"
    content = f"""---
title: SIMA Modalix VectorDB Validation Note
owner: platform-integration
environment: edge-devkit
---

# SIMA Edge AI Validation Document

This markdown file is generated for automated ingestion and retrieval testing in a SIMA workflow.
It simulates a realistic operator handoff note for Modalix deployment and local RAG verification.

## Platform Context

The pipeline under test includes:
- a Gradio-based ingestion backend for file upload and vector DB generation
- a local Milvus Lite database file (`milvus.db`)
- a Flask-based retrieval endpoint in `vectordb.py`
- model assets typically mounted alongside SIMA application bundles

## Deployment Notes (SIMA Specific)

1. The validation was executed on a local machine configured similarly to a SIMA devkit flow.
2. Runtime behavior should tolerate service restart after DB replacement.
3. Retrieval must preserve key operational terms after conversion.
4. Target use case is edge inferencing support with custom customer documentation.

## Retrieval Anchor

The canonical verification phrase for this run is:
**{expected_phrase}**

This phrase should be discoverable via semantic search and survive:
- upload
- conversion
- database download
- service restart

## Operational Checklist

- Confirm Gradio endpoint availability.
- Upload markdown artifact.
- Wait for queue completion.
- Download and replace local `milvus.db`.
- Restart vectordb server.
- Run search query and validate response payload.

## Distractor Content

To avoid trivial retrieval by single-token overlap, this file also includes unrelated terms:
`latency budget`, `thermal envelope`, `token batching`, `camera mux`, `audio stream jitter`,
`edge quantization`, and `mixed precision fallback`.
"""
    file_path.write_text(content, encoding="utf-8")
    return file_path


def upload_and_wait_for_db(
    gradio_url: str,
    markdown_file: Path,
    db_path: Path,
    sync_url: str | None = None,
    download_url: str | None = None,
) -> None:
    _log(f"Uploading markdown to Gradio: {markdown_file}")
    _log_request("POST/SSE", f"{gradio_url}/gradio_api/*", f"file={markdown_file.name}")
    for progress in upload_and_process_file(
        file_path=str(markdown_file),
        base_url=gradio_url,
        sync_url=sync_url,
        download_url=download_url,
    ):
        _log(f"gradio: {progress}")

    if not db_path.exists():
        raise RuntimeError(f"Database file not found after upload: {db_path}")
    _log_response("200 OK", f"db={db_path}")
    _log(f"Database ready: {db_path}")


def start_vectordb_server(host: str, port: int, db_path: Path) -> subprocess.Popen:
    cmd = [sys.executable, "-u", str(SCRIPT_DIR / "vectordb.py"), "--host", host, "--port", str(port)]
    env = os.environ.copy()
    env["VECTOR_DB_PATH"] = str(db_path)
    _log(f"Starting vectordb server: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(SCRIPT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def _stream_logs() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            print(f"{_c('[vectordb]', '1;35')} {line}", end="")

    threading.Thread(target=_stream_logs, daemon=True).start()
    wait_for_vectordb_ready(host, port, proc=proc, timeout_s=60)
    return proc


def stop_vectordb_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    _log("Stopping vectordb server")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        _log("Force-killing vectordb server")
        proc.kill()
        proc.wait(timeout=10)


def restart_vectordb_server(proc: subprocess.Popen, host: str, port: int, db_path: Path) -> subprocess.Popen:
    stop_vectordb_server(proc)
    return start_vectordb_server(host, port, db_path)


def wait_for_vectordb_ready(
    host: str, port: int, proc: subprocess.Popen | None = None, timeout_s: int = 60
) -> None:
    url = f"http://{host}:{port}/search"
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vectordb process exited early with code {proc.returncode}. "
                f"Check logs above (common cause: port {port} already in use)."
            )
        try:
            _log_request("GET", url, "query=health check k=1")
            resp = requests.get(url, params={"query": "health check", "k": 1}, timeout=2)
            resp.raise_for_status()
            _log_response(f"{resp.status_code}", "startup probe")
            _log(f"vectordb is ready on http://{host}:{port}")
            return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise TimeoutError(f"vectordb did not become ready within {timeout_s}s. Last error: {last_error}")


def verify_expected_result(host: str, port: int, query: str, expected_phrase: str) -> None:
    client = RagDbClient(host=host, port=port, timeout=10)
    _log_request("GET", f"http://{host}:{port}/search", f"query={query} k=3")
    results = client.search(query, k=3)
    _log_response("200", f"hits={len(results)}")
    if not results:
        raise AssertionError("vectordb returned no results")

    normalized_phrase = expected_phrase.lower()
    matched = any(normalized_phrase in str(item.get("content", "")).lower() for item in results)
    if not matched:
        raise AssertionError(
            f"Expected phrase not found in vectordb results.\n"
            f"Expected: {expected_phrase}\n"
            f"Results: {[r.get('content', '') for r in results]}"
        )
    _log("Verification passed: expected phrase found in vectordb response")
    for idx, item in enumerate(results, start=1):
        content = str(item.get("content", "")).replace("\n", " ").strip()
        if len(content) > 220:
            content = content[:220] + "..."
        score = item.get("score")
        metadata = item.get("metadata", {})
        _log(f"result[{idx}] score={score} content='{content}' metadata={metadata}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run e2e vectordb test against local Gradio + vectordb.")
    parser.add_argument("--gradio-url", default="http://127.0.0.1:7860", help="Gradio base URL")
    parser.add_argument("--vectordb-host", default="127.0.0.1", help="vectordb host")
    parser.add_argument("--vectordb-port", type=int, default=9100, help="vectordb port")
    parser.add_argument("--sync-url", default=None, help="Optional FastAPI sync endpoint (POST)")
    parser.add_argument("--download-url", default=None, help="Optional DB download endpoint (GET)")
    parser.add_argument(
        "--expected-phrase",
        default="amber nebula corridor",
        help="Phrase that must appear in retrieval results",
    )
    parser.add_argument(
        "--query",
        default="amber nebula corridor",
        help="Query sent to vectordb for verification",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _log(f"Expected phrase: {args.expected_phrase}")
    _log(f"Query phrase: {args.query}")
    if args.sync_url:
        _log(f"Sync URL: {args.sync_url}")
    if args.download_url:
        _log(f"Download URL: {args.download_url}")

    # Keep DB location deterministic for both gradioclient sync and vectordb reads.
    os.chdir(PROJECT_DIR)
    db_path = PROJECT_DIR / "milvus.db"

    with tempfile.TemporaryDirectory(prefix="vectordb-e2e-") as tmpdir:
        markdown_file = generate_markdown_file(Path(tmpdir), args.expected_phrase)
        _log(f"Generated markdown file: {markdown_file}")
        upload_and_wait_for_db(
            args.gradio_url,
            markdown_file,
            db_path,
            sync_url=args.sync_url,
            download_url=args.download_url,
        )

    # Start then restart to match requested flow and verify the service can reload DB.
    proc = start_vectordb_server(args.vectordb_host, args.vectordb_port, db_path)
    try:
        proc = restart_vectordb_server(proc, args.vectordb_host, args.vectordb_port, db_path)
        verify_expected_result(args.vectordb_host, args.vectordb_port, args.query, args.expected_phrase)
    finally:
        stop_vectordb_server(proc)

    _log("E2E test completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
