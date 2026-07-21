#!/usr/bin/env bash
set -euo pipefail

archive="${1:-}"
if [[ -z "${archive}" || ! -f "${archive}" ]]; then
  echo "Usage: validate_apps_runtime_archive.sh <runtime-archive>" >&2
  exit 1
fi

runtime_root="prebuilt-apps"

python3 - "${archive}" "${runtime_root}" <<'PY'
import json
import sys
import tarfile
from pathlib import PurePosixPath

archive, root = sys.argv[1:]
prefix = f"{root}/"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    by_name = {member.name: member for member in members}
    text_suffixes = {
        ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
        ".json", ".md", ".py", ".sh", ".yaml", ".yml",
    }
    model_suffixes = (".tar.gz", ".mpk", ".onnx")
    allowed_example_root_executables = {"run.sh", "setup.sh"}
    obsolete_references = (
        b"assets/models/",
        b"assets/test_images/",
        b"assets/test_images_classification/",
        b"neat-apps-runtime/",
        b"/usr/bin/fix_devkit_runtime.sh",
    )

    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            fail(f"runtime archive contains an unsafe path: {member.name}")
        if member.name != root and not member.name.startswith(prefix):
            fail(f"runtime archive member is outside {root}/: {member.name}")
        if member.isfile() and member.name.endswith(model_suffixes):
            fail(f"runtime archive contains a packaged model: {member.name}")
        if (
            member.isfile()
            and member.mode & 0o111
            and len(path.parts) == 5
            and path.parts[:2] == (root, "examples")
            and path.name not in allowed_example_root_executables
        ):
            fail(f"executable is at an example root: {member.name}")
        if (
            member.isfile()
            and member.mode & 0o111
            and "/src/cpp/" in member.name
            and "/src/cpp/pre-built/" not in member.name
        ):
            fail(f"C++ executable is outside src/cpp/pre-built/: {member.name}")
        if member.isfile() and PurePosixPath(member.name).suffix in text_suffixes:
            source = bundle.extractfile(member)
            content = source.read() if source is not None else b""
            if any(reference in content for reference in obsolete_references):
                fail(f"runtime archive contains an obsolete path reference: {member.name}")

    manifest_name = f"{root}/manifest.json"
    manifest_member = by_name.get(manifest_name)
    if manifest_member is None or not manifest_member.isfile():
        fail(f"runtime archive is missing {manifest_name}")
    source = bundle.extractfile(manifest_member)
    if source is None:
        fail(f"unable to read {manifest_name}")
    try:
        manifest = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        fail(f"{manifest_name} is not valid JSON")
    if not isinstance(manifest, dict):
        fail(f"{manifest_name} must contain a JSON object")

    core = manifest.get("neat-core")
    if not isinstance(core, dict) or core.get("policy") != "snap":
        fail("manifest.json must set neat-core policy to snap")
    if "insight" in manifest:
        fail("manifest.json must not define an Insight install target")
    platform_version = manifest.get("platform-version")
    if not isinstance(platform_version, str) or not platform_version.strip():
        fail("manifest.json must define platform-version")
PY

members="$(tar -tzf "${archive}")"

forbidden='(^|/)(models|portal|tests|test-scope\.yaml|CMakeLists\.txt|CTestTestfile\.cmake|cmake_install\.cmake|Makefile|[^/]+_(unit|e2e)_test|sandbox[^/]*|__pycache__)(/|$)|\.(a|pyc|pyo|log|db|lock)$'
if grep -E "${forbidden}" <<<"${members}"; then
  echo "Runtime archive contains non-runtime or generated files." >&2
  exit 1
fi

invalid_assets="$(
  grep "^${runtime_root}/assets/" <<<"${members}" \
    | grep -Ev "^${runtime_root}/assets/$|^${runtime_root}/assets/datasets(/|$)" \
    || true
)"
if [[ -n "${invalid_assets}" ]]; then
  printf '%s\n' "${invalid_assets}" >&2
  echo "Runtime archive contains non-runtime assets." >&2
  exit 1
fi

echo "Runtime archive contains production files only: ${archive}"
