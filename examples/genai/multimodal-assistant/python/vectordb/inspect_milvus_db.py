#!/usr/bin/env python3
"""Inspect a local milvus-lite database file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect collections and sample rows in a milvus-lite DB file.")
    p.add_argument("--db-path", default="milvus.db", help="Path to milvus db file")
    p.add_argument("--collection", default=None, help="Optional collection name to inspect")
    p.add_argument("--limit", type=int, default=3, help="Sample rows per collection")
    p.add_argument("--max-text", type=int, default=180, help="Max text preview length in sample rows")
    return p.parse_args()


def hr(title: str) -> None:
    print(f"\n=== {title} ===")


def fmt_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{num_bytes}B"


def compact_value(value: Any, max_text: int) -> Any:
    if isinstance(value, str):
        value = value.replace("\n", " ").strip()
        return value if len(value) <= max_text else value[:max_text] + "..."
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float)):
            return f"<vector len={len(value)}>"
        return value
    return value


def print_json_block(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=True, default=str))


def list_collections_safe(client: MilvusClient) -> list[str]:
    try:
        return client.list_collections()
    except Exception as exc:
        print(f"warning: failed to list collections via pymilvus: {exc}")
        return []


def connect_explicit_milvus_lite(db_path: Path) -> tuple[MilvusClient | None, Any]:
    try:
        from milvus_lite.server_manager import server_manager_instance
    except Exception as exc:
        print(f"warning: milvus_lite import failed for explicit local fallback: {exc}")
        return None, None

    try:
        uds_uri = server_manager_instance.start_and_get_uri(str(db_path))
        if not uds_uri:
            print("warning: explicit local milvus-lite start returned no URI")
            return None, None
        client = MilvusClient(uri=uds_uri)
        return client, server_manager_instance
    except Exception as exc:
        print(f"warning: explicit local milvus-lite connection failed: {exc}")
        return None, None


def main() -> int:
    args = parse_args()
    db_path = Path(args.db_path).expanduser().resolve()

    hr("Database")
    print(f"path:   {db_path}")
    print(f"exists: {db_path.exists()}")
    print(f"size:   {fmt_size(db_path.stat().st_size) if db_path.exists() else '0B'}")

    if not db_path.exists():
        print("error: db file does not exist")
        return 2

    client = MilvusClient(uri=str(db_path))
    collections = list_collections_safe(client)
    lite_server_manager = None

    if not collections and db_path.stat().st_size > 0:
        print("\nwarning: no collections via direct path URI; retrying with explicit local milvus-lite URI...")
        explicit_client, lite_server_manager = connect_explicit_milvus_lite(db_path)
        if explicit_client is not None:
            client = explicit_client
            collections = list_collections_safe(client)

    hr("Collections")
    if collections:
        for i, name in enumerate(collections, start=1):
            print(f"{i}. {name}")
    else:
        print("(none)")
    if not collections:
        print("\nNo collections found via Milvus API.")
        if lite_server_manager is not None:
            lite_server_manager.release_server(str(db_path))
        return 1

    targets = [args.collection] if args.collection else collections
    for name in targets:
        if name not in collections:
            hr(f"Collection: {name}")
            print("not found")
            continue

        hr(f"Collection: {name}")
        stats = client.get_collection_stats(name)
        print("stats:")
        print_json_block(stats)

        schema = client.describe_collection(name)
        fields = schema.get("fields", [])
        print("\nschema:")
        print(f"- auto_id: {schema.get('auto_id')}")
        print(f"- enable_dynamic_field: {schema.get('enable_dynamic_field')}")
        print("- fields:")
        for field in fields:
            print(
                f"  - {field.get('name')} | type={field.get('type')} | "
                f"primary={field.get('is_primary', False)} | "
                f"dim={field.get('params', {}).get('dim', '-')}"
            )

        rows = client.query(
            collection_name=name,
            filter="",
            output_fields=["*"],
            limit=args.limit,
        )
        print(f"\nsample rows ({len(rows)}):")
        if not rows:
            print("  (none)")
            continue
        for i, row in enumerate(rows, start=1):
            compact_row = {k: compact_value(v, args.max_text) for k, v in row.items()}
            print(f"- row[{i}]")
            print_json_block(compact_row)

    if lite_server_manager is not None:
        lite_server_manager.release_server(str(db_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
