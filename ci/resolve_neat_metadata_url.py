#!/usr/bin/env python3
import json
import sys
import urllib.request


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def url_exists(url: str) -> bool:
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req):
                return True
        except Exception:
            continue
    return False


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: resolve_neat_metadata_url.py <ci/neat-core-link.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = json.load(f)

    core = cfg["neat_core"]
    base_url = core["base_url"].rstrip("/")
    selector = core["selector"].strip()
    metadata_preference = core.get("metadata_preference", ["metadata-all.json", "metadata.json"])

    if ":" not in selector:
        print(f"Invalid selector '{selector}' (expected <branch>:latest or <branch>:<tag>)",
              file=sys.stderr)
        return 2

    branch, ref = selector.split(":", 1)
    branch = branch.strip()
    ref = ref.strip()
    if not branch or not ref:
        print(f"Invalid selector '{selector}'", file=sys.stderr)
        return 2

    if ref == "latest":
        tag_url = f"{base_url}/{branch}/latest.tag"
        ref = fetch_text(tag_url).strip()
        if not ref:
            print(f"Empty latest.tag for branch '{branch}'", file=sys.stderr)
            return 1

    for metadata_name in metadata_preference:
        url = f"{base_url}/{branch}/{ref}/{metadata_name}"
        if url_exists(url):
            print(url)
            return 0

    print(f"No metadata file found for {branch}/{ref} using {metadata_preference}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
