# vector_db_service.py

import os
import asyncio
import logging
import subprocess
import time
import warnings
from typing import List, Dict, Union
from pathlib import Path
from flask import Flask, request, jsonify, abort
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
import threading
import requests
from typing import Any, Dict, List

# milvus-lite emits a pkg_resources deprecation warning on import.
# We suppress this known third-party warning to keep service logs clean.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"milvus_lite(\..*)?",
)

# For starting/stopping this same script as a service
PYTHON = os.sys.executable
SCRIPT = Path(__file__).resolve()
WORKER_SCRIPT = SCRIPT.parent / "vectordb_worker.py"
PYTHON_DIR = SCRIPT.parent.parent
UI_DIR = PYTHON_DIR / "ui"
STARTUP_DELAY = 2
K_RETRIEVED_DOCS = 1
DEFAULT_MIN_SCORE = float(os.environ.get("VDB_MIN_SCORE", "0.0"))
DEFAULT_VDB_HOST = '0.0.0.0'
DEFAULT_VDB_PORT = 9100

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("VectorDB")

RAG_DB_PATH = str((UI_DIR / "milvus.db").resolve())

# --- VectorDB definition ---
class VectorDB:
    def __init__(
        self,
        db_path: Union[str, Path],
        collection_name: str = "demo_collection",
        embedding_model: str = "thenlper/gte-small",
        embedding_device: str = "cpu",
    ):
        self.db_path = str(db_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_device = embedding_device

        self.embedding_model = self._load_embedding_model()
        self.vector_store = self._load_vector_store()

    def _load_embedding_model(self):
            model_dir = self._resolve_local_model_path()
            model_kwargs: Dict[str, Any] = {"device": self.embedding_device}
            model_kwargs["local_files_only"] = True
            # Force offline behavior even if parent process exported online-friendly values.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            log.info(
                "Offline flags: HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s HF_DATASETS_OFFLINE=%s",
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
                os.environ.get("HF_DATASETS_OFFLINE"),
            )

            log.info(f"📦 Loading embedding model from local folder: {model_dir}")
            model = HuggingFaceEmbeddings(model_name=model_dir, model_kwargs=model_kwargs)
            log.info("✅ Loaded embedding model.")
            return model

    def _resolve_local_model_path(self) -> str:
        configured_dir = os.environ.get("VDB_EMBED_MODEL_DIR")
        if not configured_dir:
            raise FileNotFoundError(
                "Embedding model directory is not configured.\n"
                "Set app.rag.embedding_model_dir in src/common/config.yaml."
            )

        candidate_dirs = [Path(configured_dir)]

        for candidate in candidate_dirs:
            if candidate.is_dir():
                return str(candidate.resolve())

        searched = ", ".join(str(p) for p in candidate_dirs)
        raise FileNotFoundError(
            "Embedding model directory was not found.\n"
            f"Searched: {searched}\n"
            "Set app.rag.embedding_model_dir in src/common/config.yaml."
        )

    def _load_vector_store(self):
        log.info(f"Loading vector store from {self.db_path}")
        try:
            return self._create_vector_store()
        except Exception as e:
            # Newer pymilvus/langchain-milvus may require an active event loop during
            # AsyncMilvusClient setup, even when we only use sync APIs.
            if "no running event loop" not in str(e).lower():
                raise
            log.warning(
                "Milvus init required a running event loop; retrying initialization in an async context."
            )
            return asyncio.run(self._create_vector_store_async())

    def _create_vector_store(self):
        return Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"uri": self.db_path},
            index_params={"index_type": "FLAT", "metric_type": "IP"},
            drop_old=False
        )

    async def _create_vector_store_async(self):
        return self._create_vector_store()

    def search(self, query: str, k: int = 5, min_score: float = DEFAULT_MIN_SCORE) -> List[Dict[str, Union[str, dict]]]:
        results = self.vector_store.similarity_search_with_score(query, k=k)
        filtered = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results if score >= min_score
        ]
        return filtered

    def list_documents(self, limit: int = 16384) -> List[Dict[str, Any]]:
        """Enumerate every chunk in the collection using the already-open store
        (the service owns the DB file, so we never open a second handle). Returns
        ``[{id, text, metadata}]``."""
        try:
            from rag.inspect_db import rows_to_docs
        except ImportError:                      # loaded as a top-level module by the worker
            from inspect_db import rows_to_docs
        store = self.vector_store
        client = getattr(store, "client", None)
        if client is not None:                   # langchain-milvus MilvusClient path
            pk = getattr(store, "_primary_field", None) or "pk"
            try:
                rows = client.query(collection_name=self.collection_name,
                                    filter=f"{pk} >= 0", output_fields=["*"], limit=limit)
            except Exception:                    # noqa: BLE001 - try a match-all filter
                rows = client.query(collection_name=self.collection_name,
                                    filter="", output_fields=["*"], limit=limit)
            return rows_to_docs(rows, pk)
        col = getattr(store, "col", None)
        if col is not None:                      # older pymilvus Collection path
            try:
                col.load()
            except Exception:                    # noqa: BLE001
                pass
            pk = next((f.name for f in col.schema.fields
                       if getattr(f, "is_primary", False)), "pk")
            rows = col.query(expr=f"{pk} >= 0", output_fields=["*"], limit=limit)
            return rows_to_docs(rows, pk)
        raise RuntimeError("Vector store does not expose a queryable client")



class RagDbClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 9100, timeout: int = 10):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def search(self, query: str, k: int = 1, min_score: float = DEFAULT_MIN_SCORE) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/search"
        params = {"query": query, "k": k, "min_score": min_score}
        try:
            log.info(f"🔍 Searching: '{query}' (k={k}, min_score={min_score})")
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.HTTPError as http_err:
            log.error(f"HTTP error during search: {http_err} | Response: {resp.text}")
            raise
        except requests.RequestException as req_err:
            log.error(f"Network error during search: {req_err}")
            raise

        try:
            data = resp.json()
            results = data.get("results", [])
            log.info(f"Received {len(results)} results")
            return results
        except ValueError as json_err:
            log.error(f"Invalid JSON response: {json_err}")
            raise

    def list_documents(self, limit: int = 16384) -> Dict[str, Any]:
        """Fetch every ingested chunk from the running VectorDB service."""
        url = f"{self.base_url}/documents"
        resp = requests.get(url, params={"limit": limit}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_server_up(self) -> bool:
        """Check if the RAG server is responsive by making a test search."""
        try:
            test_results = self.search("ping", k=1)
            if isinstance(test_results, list):
                log.info("RAG server is up")
                return True
        except Exception as e:
            log.warning(f"RAG server check failed: {e}")
        return False


# --- Flask app setup ---
app = Flask(__name__)
ragdb: VectorDB = None

@app.route("/", methods=["GET"])
def index_endpoint():
    return jsonify({
        "service": "rag-vectordb",
        "message": "This is the RAG VectorDB API, not the Multimodal Assistant UI.",
        "search": "/search?query=<text>&k=3&min_score=-1",
    })


def initialize_db():
    global ragdb
    db_path = os.environ.get("VECTOR_DB_PATH", RAG_DB_PATH)
    log.info(f'Start initialization for {db_path}')
    try:
        ragdb = VectorDB(db_path=db_path)
        log.info(f"Initialized VectorDB with '{db_path}'.")
    except Exception as e:
        log.error(f"Error initializing VectorDB: {e}")
        raise

@app.route("/search", methods=["GET"])
def search_endpoint():
    if not ragdb:
        abort(500, description="DB not initialized.")
    query = request.args.get("query", "")
    k = int(request.args.get("k", K_RETRIEVED_DOCS))
    min_score = float(request.args.get("min_score", DEFAULT_MIN_SCORE))
    try:
        results = ragdb.search(query, k, min_score=min_score)
        return jsonify({"results": results})
    except Exception as e:
        log.error(f"Search error: {e}")
        abort(500, description=str(e))

@app.route("/documents", methods=["GET"])
def documents_endpoint():
    """List every ingested chunk (for RAG-database inspection)."""
    if not ragdb:
        abort(500, description="DB not initialized.")
    limit = int(request.args.get("limit", 16384))
    try:
        docs = ragdb.list_documents(limit=limit)
        return jsonify({"documents": docs, "count": len(docs),
                        "collection": ragdb.collection_name})
    except Exception as e:
        log.error(f"Documents error: {e}")
        abort(500, description=str(e))

def main():
    host = os.environ.get("HOST", DEFAULT_VDB_HOST)
    port = int(os.environ.get("PORT", DEFAULT_VDB_PORT))
    log.info("RAG VectorDB API: http://127.0.0.1:%s/search", port)
    log.info("Multimodal Assistant UI runs on the Flask app port, not this service.")
    app.run(host=host, port=port, debug=False)


def start_service():
    env = os.environ.copy()
    cmd = [PYTHON, "-u", WORKER_SCRIPT]
    log.info(f"Starting vector DB service with: {cmd}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
        bufsize=1
    )

    def stream_output():
        for line in iter(proc.stdout.readline, ''):
            print(line, end="")

    output_thread = threading.Thread(target=stream_output, daemon=True)
    output_thread.start()

    time.sleep(STARTUP_DELAY)
    return proc

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the standalone VectorDB service.")
    parser.add_argument(
        "--run-service",
        action="store_true",
        help="Backward-compatible flag. Service starts with or without this flag.",
    )
    parser.add_argument("--host", default=None, help="Override host (default: env HOST or 0.0.0.0).")
    parser.add_argument("--port", type=int, default=None, help="Override port (default: env PORT or 9100).")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Initialize VectorDB and exit without starting Flask.",
    )
    args = parser.parse_args()

    if args.host:
        os.environ["HOST"] = args.host
    if args.port is not None:
        os.environ["PORT"] = str(args.port)

    initialize_db()
    if args.init_only:
        log.info("VectorDB initialization succeeded (--init-only).")
    else:
        main()
