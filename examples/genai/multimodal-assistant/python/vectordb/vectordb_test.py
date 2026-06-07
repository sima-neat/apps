#!/usr/bin/env python3
"""Offline VectorDB sanity test: initialize local DB and run queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_DB_PATH = str((PROJECT_DIR / "milvus.db").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize local VectorDB and run offline search queries."
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("VECTOR_DB_PATH", DEFAULT_DB_PATH),
        help="Path to local milvus.db file.",
    )
    parser.add_argument(
        "--collection",
        default="demo_collection",
        help="Collection name to open.",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List collections in the DB before initialization.",
    )
    parser.add_argument(
        "--auto-collection",
        action="store_true",
        help="If --collection is missing, auto-pick the first available collection.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Query to run. Repeat --query for multiple checks.",
    )
    parser.add_argument("--k", type=int, default=3, help="Top-k results per query.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=-1.0,
        help="Minimum score filter applied after retrieval.",
    )
    parser.add_argument(
        "--max-text",
        type=int,
        default=220,
        help="Max preview length for result content.",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Only initialize VectorDB, skip queries.",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="Fail if any query returns zero results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed traces for parity debugging.",
    )
    return parser.parse_args()


def _preview(text: str, limit: int) -> str:
    compact = text.replace("\n", " ").strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def _print_result(idx: int, item: dict[str, Any], max_text: int) -> None:
    score = item.get("score")
    content = _preview(str(item.get("content", "")), max_text)
    metadata = item.get("metadata", {})
    print(f"  [{idx}] score={score} content='{content}'")
    print(f"      metadata={json.dumps(metadata, ensure_ascii=True, default=str)}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _print_debug_header(db_path: Path) -> None:
    stat = db_path.stat()
    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    print("\n[debug] environment")
    print(f"  cwd={Path.cwd()}")
    print(f"  python={sys.executable}")
    print(f"  HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}")
    print(f"  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"  VECTOR_DB_PATH={os.environ.get('VECTOR_DB_PATH')}")
    print(f"  VDB_EMBED_MODEL_DIR={os.environ.get('VDB_EMBED_MODEL_DIR')}")
    print("\n[debug] db file")
    print(f"  size_bytes={stat.st_size}")
    print(f"  mtime_utc={mtime}")
    print(f"  sha256={_sha256_file(db_path)}")


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()

    print("=== VectorDB Offline Test ===")
    print(f"db_path:     {db_path}")
    print(f"exists:      {db_path.exists()}")
    print(f"collection:  {args.collection}")
    print(f"k:           {args.k}")
    print(f"min_score:   {args.min_score}")

    if not db_path.exists():
        print("error: db file does not exist")
        return 2
    if args.debug:
        _print_debug_header(db_path)

    collections: list[str] = []
    collection_stats: dict[str, Any] = {}
    collection_schemas: dict[str, Any] = {}
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=str(db_path))
        collections = client.list_collections()
        for name in collections:
            try:
                collection_stats[name] = client.get_collection_stats(name)
            except Exception as exc:
                collection_stats[name] = {"error": str(exc)}
            try:
                collection_schemas[name] = client.describe_collection(name)
            except Exception as exc:
                collection_schemas[name] = {"error": str(exc)}
    except Exception as exc:
        print(f"warning: unable to list collections before init: {exc}")

    if args.list_collections:
        print("\n[collections]")
        if collections:
            for i, name in enumerate(collections, start=1):
                print(f"  {i}. {name}")
        else:
            print("  (none)")
    if args.debug and collections:
        print("\n[debug] collection details")
        for name in collections:
            stats = collection_stats.get(name, {})
            schema = collection_schemas.get(name, {})
            fields = schema.get("fields", []) if isinstance(schema, dict) else []
            field_summary = []
            for field in fields:
                fname = field.get("name")
                ftype = field.get("type")
                dim = field.get("params", {}).get("dim")
                if dim is None:
                    field_summary.append(f"{fname}:{ftype}")
                else:
                    field_summary.append(f"{fname}:{ftype}:dim={dim}")
            print(f"  - {name}")
            print(f"    stats={json.dumps(stats, ensure_ascii=True, default=str)}")
            print(f"    fields={field_summary}")

    selected_collection = args.collection
    if collections and selected_collection not in collections:
        if args.auto_collection:
            selected_collection = collections[0]
            print(
                f"\n[warn] collection '{args.collection}' not found; "
                f"using '{selected_collection}'"
            )
        else:
            print(f"\nerror: collection '{args.collection}' not found in DB")
            print(f"available: {collections}")
            print("hint: pass --collection <name> or use --auto-collection")
            return 7

    print("\n[step] initializing VectorDB...")
    try:
        from vectordb.vectordb import VectorDB
        ragdb = VectorDB(db_path=str(db_path), collection_name=selected_collection)
    except ModuleNotFoundError as exc:
        print(f"error: missing dependency during import: {exc}")
        print("hint: install runtime deps, e.g. `.venv/bin/pip install -r requirements.txt`")
        return 6
    except Exception as exc:
        print(f"error: initialization failed: {exc}")
        return 3
    print("[ok] VectorDB initialized")
    if args.debug:
        try:
            resolved_model_dir = ragdb._resolve_local_model_path()
        except Exception as exc:
            resolved_model_dir = f"<error: {exc}>"
        print("\n[debug] initialized config")
        print(f"  selected_collection={selected_collection}")
        print(f"  ragdb.collection_name={ragdb.collection_name}")
        print(f"  ragdb.db_path={ragdb.db_path}")
        print(f"  embedding_model_name={ragdb.embedding_model_name}")
        print(f"  embedding_device={ragdb.embedding_device}")
        print(f"  resolved_model_dir={resolved_model_dir}")

    if args.init_only:
        return 0

    queries = args.query or ["health check"]
    if not args.query:
        print("\n[hint] default query is 'health check'; pass --query with domain terms for real validation.")
    has_empty = False
    for q in queries:
        print(f"\n[query] {q}")
        try:
            if args.debug:
                raw_results = ragdb.vector_store.similarity_search_with_score(q, k=args.k)
                raw_scores = [score for _, score in raw_results]
                print(
                    f"[debug] raw_hits={len(raw_results)} raw_scores={json.dumps(raw_scores, ensure_ascii=True, default=str)}"
                )
            results = ragdb.search(q, k=args.k, min_score=args.min_score)
        except Exception as exc:
            print(f"error: search failed: {exc}")
            return 4

        print(f"hits: {len(results)}")
        if not results:
            has_empty = True
            continue
        for i, item in enumerate(results, start=1):
            _print_result(i, item, args.max_text)

    if args.require_results and has_empty:
        print("\nerror: one or more queries returned zero results")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
