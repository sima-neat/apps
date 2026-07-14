"""Read-only inspection of the RAG Milvus database.

Enumerates the ingested chunks and reads the build-metadata sidecar without
loading the embedding model. Two entry points:

  * ``inspect_rag(...)``  — opens the DB file directly (fresh MilvusClient). Use
    this only when nothing else holds the file open (e.g. the CLI, where the RAG
    VectorDB service is not running). milvus-lite is single-writer per file.
  * ``rows_to_docs(...)`` — shared row→document shaping, reused by the VectorDB
    service so the UI can enumerate through the process that already owns the DB.

Everything degrades gracefully: a missing DB, a missing collection, or an
enumeration error still returns the sidecar summary with an ``error`` note.
"""

import json
import os
from pathlib import Path

DEFAULT_COLLECTION = "demo_collection"

# Fields that are never part of a user-visible chunk (the vector, the text
# itself, and internal keys) are dropped from the per-chunk metadata.
_SKIP_META = {"vector", "embedding", "sparse", "text", "page_content"}


def default_db_path():
    """The bundled RAG DB location (src/python/ui/milvus.db)."""
    return str((Path(__file__).resolve().parent.parent / "ui" / "milvus.db").resolve())


def read_rag_meta(db_path=None):
    """Return the build-metadata sidecar (``<db>.meta.json``) as a dict, or {}."""
    db_path = db_path or default_db_path()
    meta_path = Path(db_path).with_suffix(".meta.json")
    if meta_path.is_file():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def rows_to_docs(rows, pk="pk"):
    """Shape raw Milvus query rows into ``[{id, text, metadata}]``."""
    docs = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        text = r.get("text") or r.get("page_content") or ""
        meta = {k: v for k, v in r.items()
                if k != pk and k not in _SKIP_META and not str(k).startswith("$")}
        docs.append({"id": r.get(pk), "text": text, "metadata": meta})
    return docs


def _primary_key(client, collection):
    try:
        desc = client.describe_collection(collection)
        for f in desc.get("fields", []):
            if f.get("is_primary"):
                return f.get("name", "pk")
    except Exception:  # noqa: BLE001
        pass
    return "pk"


def read_rag_documents(db_path=None, collection=DEFAULT_COLLECTION, limit=16384):
    """Open the DB file directly and return every chunk as ``[{id, text, metadata}]``.

    Raises FileNotFoundError / ValueError / ImportError on a missing DB, missing
    collection, or absent pymilvus. Only call when no other process holds the DB.
    """
    db_path = db_path or default_db_path()
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"No RAG database at {db_path}")
    from pymilvus import MilvusClient
    client = MilvusClient(uri=db_path)
    try:
        cols = client.list_collections()
        if collection not in cols:
            raise ValueError(
                f"Collection '{collection}' not found (have: {', '.join(cols) or 'none'})")
        try:
            client.load_collection(collection)
        except Exception:  # noqa: BLE001 - milvus-lite may not require/allow this
            pass
        pk = _primary_key(client, collection)
        rows = client.query(collection_name=collection, filter=f"{pk} >= 0",
                            output_fields=["*"], limit=limit)
        return rows_to_docs(rows, pk)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def inspect_rag(db_path=None, collection=DEFAULT_COLLECTION, limit=16384):
    """Assemble a full inspection result: sidecar meta + enumerated chunks.

    Never raises — enumeration failures are reported in ``error`` while the
    sidecar summary (source file, chunk count, embedding model) is still returned.
    """
    db_path = db_path or default_db_path()
    result = {
        "path": db_path,
        "exists": os.path.isfile(db_path),
        "collection": collection,
        "meta": read_rag_meta(db_path),
        "documents": [],
        "count": 0,
        "error": None,
    }
    if not result["exists"]:
        result["error"] = "No RAG database file found."
        return result
    try:
        docs = read_rag_documents(db_path, collection, limit)
        result["documents"] = docs
        result["count"] = len(docs)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result
