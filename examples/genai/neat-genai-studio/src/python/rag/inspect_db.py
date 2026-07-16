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

# milvus-lite's bounds check rejects limit=16384 ("limit should be in range
# [1, 16384]" is exclusive at the top despite the message), so the largest
# usable query limit is 16383.
MAX_QUERY_LIMIT = 16383

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


def rows_to_docs(rows, pk="pk", text_field=None):
    """Shape raw Milvus query rows into ``[{id, text, metadata}]``.

    The chunk text is taken from ``text_field`` when known, else a commonly-named
    field, else the longest string value in the row — so the content shows up
    regardless of how the collection named its fields."""
    candidates = ([text_field] if text_field else []) + \
        ["text", "page_content", "content", "chunk", "document"]
    docs = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        text = ""
        for k in candidates:
            if k and r.get(k):
                text = r.get(k)
                break
        # Metadata = everything that isn't the pk, the text, a vector, or internal.
        meta, longest_key, longest_len = {}, None, -1
        for k, v in r.items():
            if k == pk or k == text_field or k in _SKIP_META or str(k).startswith("$"):
                continue
            if isinstance(v, (list, tuple)):   # a vector / embedding
                continue
            if isinstance(v, str) and len(v) > longest_len:
                longest_key, longest_len = k, len(v)
            meta[k] = v
        if not text and longest_key is not None:   # fallback: longest string is the chunk
            text = meta.pop(longest_key)
        docs.append({"id": r.get(pk), "text": str(text or ""), "metadata": meta})
    return docs


def schema_fields(client, collection):
    """Return ``(pk, text_field, output_fields)`` for a collection. output_fields
    are all NON-vector fields — milvus-lite does not reliably expand
    ``output_fields=["*"]`` to scalar fields, so the chunk text and metadata are
    only returned when we name the fields explicitly."""
    pk, text_field, out = "pk", None, []
    try:
        fields = client.describe_collection(collection).get("fields", [])
    except Exception:  # noqa: BLE001
        return pk, text_field, ["*"]
    for f in fields:
        name = f.get("name")
        if not name:
            continue
        if f.get("is_primary"):
            pk = name
        is_vector = ("dim" in (f.get("params") or {})
                     or "VECTOR" in str(f.get("type", "")).upper())
        if is_vector:
            continue
        out.append(name)
        if text_field is None and name in ("text", "page_content", "content"):
            text_field = name
    return pk, text_field, (out or ["*"])


def read_rag_documents(db_path=None, collection=DEFAULT_COLLECTION, limit=MAX_QUERY_LIMIT):
    """Open the DB file directly and return every chunk as ``[{id, text, metadata}]``.

    Raises FileNotFoundError / ValueError / ImportError on a missing DB, missing
    collection, or absent pymilvus. Only call when no other process holds the DB.
    """
    db_path = db_path or default_db_path()
    limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
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
        pk, text_field, out_fields = schema_fields(client, collection)
        rows = client.query(collection_name=collection, filter=f"{pk} >= 0",
                            output_fields=out_fields, limit=limit)
        return rows_to_docs(rows, pk, text_field)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


def inspect_rag(db_path=None, collection=DEFAULT_COLLECTION, limit=MAX_QUERY_LIMIT):
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
