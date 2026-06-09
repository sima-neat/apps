# RAG Helper

This folder contains the optional local VectorDB service and helper scripts used
by the Multimodal Assistant RAG flow.

The customer-facing setup commands are documented in the example root
`README.md`. This folder is only needed when RAG is enabled.

## Files

- `create_db.py`: create `milvus.db` from a Markdown document.
- `vectordb.py`: local VectorDB Flask service on port `9100`.
- `vectordb_worker.py`: worker entrypoint started by the Flask app.
- `inspect_milvus_db.py`: inspect a generated Milvus DB.
- `gradioclient.py`, `test_vectordb_e2e.py`, `vectordb_test.py`: RAG helper
  and test utilities kept for further validation.

## Create A Local DB

```bash
cd /workspace/sima-neat/apps/examples/genai/multimodal-assistant/python
source ~/multimodal-assistant-app/bin/activate

EMBED_DIR="/path/to/local/embedding-model-dir"
python rag/create_db.py \
  --input /workspace/sima-neat/apps/examples/genai/multimodal-assistant/common/rag/neat.md \
  --output ./milvus.db \
  --embedding-model "${EMBED_DIR}"
```

## Run VectorDB Manually

The Flask UI starts this service automatically when RAG is enabled. For a
manual check:

```bash
cd /workspace/sima-neat/apps/examples/genai/multimodal-assistant/python
source ~/multimodal-assistant-app/bin/activate

export VDB_EMBED_MODEL_DIR="/path/to/local/embedding-model-dir"
python rag/vectordb.py --host 127.0.0.1 --port 9100
```

Query it:

```bash
curl -sG http://127.0.0.1:9100/search \
  --data-urlencode "query=What is Neat?" \
  --data "k=3" \
  --data "min_score=-1" | python3 -m json.tool
```
