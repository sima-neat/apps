#!/usr/bin/env python3
"""Generate and sync Apps Portal Algolia records."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SOURCE = "examples"
SUPPORTED_LANGUAGES = ["en", "ko", "ja", "zh-Hant", "uk"]
DEFAULT_MAX_RECORD_BYTES = 9_500
DEFAULT_BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or sync Apps Portal Algolia records.")
    parser.add_argument("--catalog-json", default="portal/public/catalog.json", help="Portal catalog JSON path.")
    parser.add_argument("--site-base-url", required=True, help="Public examples route base URL.")
    parser.add_argument("--records-json", help="Write generated records to this JSON file.")
    parser.add_argument("--input-records-json", help="Read records from this JSON file instead of generating them.")
    parser.add_argument("--summary-json", help="Write generation summary to this JSON file.")
    parser.add_argument("--app-id", help="Algolia app ID.")
    parser.add_argument("--api-key", help="Algolia write API key.")
    parser.add_argument("--index-name", help="Algolia index name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-record-bytes", type=int, default=DEFAULT_MAX_RECORD_BYTES)
    parser.add_argument("--generate-only", action="store_true", help="Only generate records; do not upload.")
    parser.add_argument("--sync", action="store_true", help="Upload records and remove stale records for this source.")
    return parser.parse_args()


def clean_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def record_size(record: dict) -> int:
    return len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def fit_record(record: dict, max_record_bytes: int) -> tuple[dict, bool]:
    if record_size(record) <= max_record_bytes:
        return record, False

    content = str(record.get("content", ""))
    lo = 0
    hi = len(content)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = dict(record)
        candidate["content"] = content[:mid]
        candidate["content_truncated"] = True
        if record_size(candidate) <= max_record_bytes:
            best = candidate["content"]
            lo = mid + 1
        else:
            hi = mid - 1

    record["content"] = best
    record["content_truncated"] = True
    if record_size(record) > max_record_bytes:
        record["title"] = str(record.get("title", ""))[:120]
        record["content"] = str(record.get("content", ""))[:400]
        record["content_truncated"] = True
    if record_size(record) > max_record_bytes:
        raise RuntimeError(f"unable to fit record under {max_record_bytes} bytes: {record.get('path')}")
    return record, True


def app_url(site_base_url: str, app_id: str) -> str:
    encoded_id = urllib.parse.quote(app_id, safe="/")
    return f"{site_base_url.rstrip('/')}/app/{encoded_id}"


def generate_records(catalog_path: Path, site_base_url: str, max_record_bytes: int) -> tuple[list[dict], dict]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    examples = catalog.get("examples") or []
    records: list[dict] = []
    by_category: dict[str, int] = {}
    trimmed = 0
    max_seen = 0

    for example in examples:
        app_id = str(example.get("id") or "")
        if not app_id:
            continue

        title = str(example.get("name") or example.get("binary_name") or app_id)
        category = str(example.get("category") or "Examples")
        section_text = " ".join(
            clean_markdown(section.get("markdown") or "")
            for section in example.get("sections") or []
            if isinstance(section, dict)
        )
        tags = [str(tag) for tag in example.get("tags") or []]
        content = " ".join(
            part
            for part in [
                str(example.get("summary") or ""),
                str(example.get("binary_name") or ""),
                str(example.get("model") or ""),
                str(example.get("languages") or ""),
                " ".join(tags),
                section_text,
            ]
            if part
        )
        url = app_url(site_base_url, app_id)
        route = urllib.parse.urlparse(url).path
        record = {
            "objectID": f"{SOURCE}:{hashlib.sha1(app_id.encode('utf-8')).hexdigest()}",
            "source": SOURCE,
            # Example catalog metadata is not localized, so it remains useful
            # in every documentation language rather than disappearing from
            # language-filtered Developer Center searches.
            "language": SUPPORTED_LANGUAGES,
            "url": url,
            "route": route,
            "path": str(example.get("readme_path") or example.get("source_path") or app_id),
            "section": category,
            "category": category,
            "title": title,
            "content": clean_markdown(content)[:20_000],
            "hierarchy": {"lvl0": "Examples", "lvl1": category, "lvl2": title},
            "app_id": app_id,
            "tags": tags,
            "model": str(example.get("model") or ""),
            "languages": str(example.get("languages") or ""),
        }
        record, was_trimmed = fit_record(record, max_record_bytes)
        trimmed += int(was_trimmed)
        max_seen = max(max_seen, record_size(record))
        records.append(record)
        by_category[category] = by_category.get(category, 0) + 1

    summary = {
        "source": SOURCE,
        "count": len(records),
        "by_category": by_category,
        "trimmed_records": trimmed,
        "max_record_bytes": max_seen,
    }
    return records, summary


class AlgoliaClient:
    def __init__(self, app_id: str, api_key: str, index_name: str) -> None:
        self.host = f"https://{app_id}.algolia.net"
        self.index = urllib.parse.quote(index_name, safe="")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Algolia-Application-Id": app_id,
            "X-Algolia-API-Key": api_key,
        }

    def post(self, path: str, payload: dict) -> dict:
        return self.request("POST", path, payload)

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers=self.headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 404 and (
                path.endswith("/browse") or (method == "GET" and path.endswith("/settings"))
            ):
                return {"hits": []}
            raise SystemExit(f"Algolia API error: HTTP {error.code} on {path}\n{body}") from error

    def ensure_language_filter(self) -> None:
        path = f"/1/indexes/{self.index}/settings"
        settings = self.request("GET", path)
        attributes = list(settings.get("attributesForFaceting") or [])
        configured = any(re.fullmatch(r"(?:(?:filterOnly|searchable)\()?language\)?", item) for item in attributes)
        if not configured:
            self.request("PUT", path, {"attributesForFaceting": [*attributes, "filterOnly(language)"]})
            print("[algolia-index] configured filterOnly(language)")

    def browse_source_object_ids(self, source: str) -> list[str]:
        object_ids: list[str] = []
        payload = {"query": "", "attributesToRetrieve": ["objectID", "source"], "hitsPerPage": 1000}
        while True:
            result = self.post(f"/1/indexes/{self.index}/browse", payload)
            for hit in result.get("hits", []):
                if hit.get("source") == source and hit.get("objectID"):
                    object_ids.append(hit["objectID"])
            cursor = result.get("cursor")
            if not cursor:
                return object_ids
            payload = {"cursor": cursor}

    def batch(self, requests: list[dict]) -> None:
        if not requests:
            return
        result = self.post(f"/1/indexes/{self.index}/batch", {"requests": requests})
        print(f"[algolia-index] taskID={result.get('taskID')} requests={len(requests)}")


def sync_records(args: argparse.Namespace, records: list[dict]) -> None:
    if not args.app_id or not args.api_key or not args.index_name:
        raise SystemExit("--app-id, --api-key, and --index-name are required for --sync")

    client = AlgoliaClient(args.app_id, args.api_key, args.index_name)
    client.ensure_language_filter()
    desired_ids = {record["objectID"] for record in records}
    existing_ids = set(client.browse_source_object_ids(SOURCE))
    stale_ids = sorted(existing_ids - desired_ids)

    print(f"[algolia-index] existing {SOURCE} records={len(existing_ids)} stale={len(stale_ids)}")
    for start in range(0, len(stale_ids), args.batch_size):
        chunk = stale_ids[start : start + args.batch_size]
        client.batch([{"action": "deleteObject", "body": {"objectID": object_id}} for object_id in chunk])

    print(f"[algolia-index] uploading {len(records)} {SOURCE} records...")
    for start in range(0, len(records), args.batch_size):
        chunk = records[start : start + args.batch_size]
        client.batch([{"action": "addObject", "body": record} for record in chunk])


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_record_bytes < 2_000:
        raise SystemExit("--max-record-bytes must be at least 2000")
    if not args.generate_only and not args.sync:
        args.generate_only = True

    if args.input_records_json:
        records = json.loads(Path(args.input_records_json).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise SystemExit("--input-records-json must contain a JSON list")
        summary = {"source": SOURCE, "count": len(records), "from": args.input_records_json}
        print("[algolia-index] loaded records:")
    else:
        records, summary = generate_records(Path(args.catalog_json), args.site_base_url, args.max_record_bytes)
        print("[algolia-index] generated records:")
    print(json.dumps(summary, indent=2))

    if args.records_json and not args.input_records_json:
        Path(args.records_json).write_text(json.dumps(records, ensure_ascii=True), encoding="utf-8")
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.sync:
        sync_records(args, records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
