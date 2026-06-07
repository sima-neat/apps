# VectorDB E2E Testing Guide

This folder contains the standalone VectorDB service and E2E test tooling.

## What this tests

The E2E flow verifies:

1. Upload a generated markdown file to the RAGFPS Gradio server.
2. Wait for ingestion/processing completion.
3. Download the latest `milvus.db` from the sync API.
4. Restart local `vectordb` service using that DB.
5. Query `/search` and verify expected content is returned.

## Prerequisites

- Python virtual environment available at `apps-genai-demo/.venv`
- Dependencies installed
- Reachable RAGFPS endpoints:
  - Gradio API (typically `:7860`)
  - DB download API (typically `:8000/download_db`)

## Install dependencies

From project root:

```bash
cd /media/nvme/apps-genai-demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run E2E test (local RAGFPS)

```bash
cd /media/nvme/apps-genai-demo
.venv/bin/python vectordb/test_vectordb_e2e.py \
  --gradio-url http://127.0.0.1:7860 \
  --download-url http://127.0.0.1:8000/download_db \
  --vectordb-host 127.0.0.1 \
  --vectordb-port 9100
```

## Run E2E test (remote RAGFPS)

Example with RAGFPS at `10.0.0.163`:

```bash
cd /media/nvme/apps-genai-demo
.venv/bin/python vectordb/test_vectordb_e2e.py \
  --gradio-url http://10.0.0.163:7860 \
  --download-url http://10.0.0.163:8000/download_db \
  --vectordb-host 127.0.0.1 \
  --vectordb-port 9100
```

## Run VectorDB service manually

```bash
cd /media/nvme/apps-genai-demo
.venv/bin/python vectordb/vectordb.py --host 127.0.0.1 --port 9100
```

Optional init-only check:

```bash
.venv/bin/python vectordb/vectordb.py --init-only
```

## Manual query checks

```bash
VDB="http://127.0.0.1:9100"
curl -sS --get "$VDB/search" --data-urlencode "query=SIMA Edge AI Validation Document" --data "k=5" --data "min_score=-1" | jq .
curl -sS --get "$VDB/search" --data-urlencode "query=Modalix deployment local RAG verification" --data "k=5" --data "min_score=-1" | jq .
```

## Inspect Milvus DB content

Use the inspector to confirm collections/rows exist:

```bash
cd /media/nvme/apps-genai-demo
.venv/bin/python vectordb/inspect_milvus_db.py \
  --db-path /media/nvme/apps-genai-demo/milvus.db \
  --limit 5 \
  --max-text 220
```

## Troubleshooting

- Empty results from `/search`:
  - Ensure DB is populated via inspector.
  - Use `min_score=-1` while debugging filters.
  - Confirm service is reading the correct DB path.

- Noisy third-party warnings:
  - `vectordb.py` already suppresses known `milvus_lite` deprecation noise.

- Download issues:
  - Verify `http://<ragfps-host>:8000/download_db` is reachable.
  - If needed, pass custom endpoints via:
    - `--download-url`
    - `--sync-url` (only if your adapter exposes a sync trigger endpoint)
