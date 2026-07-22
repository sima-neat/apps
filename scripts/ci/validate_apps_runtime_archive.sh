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
    core_metadata_name = f"{root}/deps/neat-core.json"
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
        if path.name == "neat-core.json" and member.name != core_metadata_name:
            fail(f"Core metadata is outside {core_metadata_name}: {member.name}")
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

    try:
        core_metadata = bundle.getmember(core_metadata_name)
    except KeyError:
        fail(f"runtime archive is missing {core_metadata_name}")
    if not core_metadata.isfile():
        fail(f"runtime archive is missing {core_metadata_name}")
    source = bundle.extractfile(core_metadata)
    data = json.load(source)
    core = data.get("neat-core")
    if not isinstance(core, dict):
        fail(f"invalid Core metadata in {core_metadata_name}")
    ref = core.get("ref")
    spec = core.get("spec")
    if not isinstance(ref, str) or not ref.strip():
        fail(f"invalid Core ref in {core_metadata_name}")
    if not isinstance(spec, str) or not spec.strip() or spec == "latest":
        fail(f"invalid Core spec in {core_metadata_name}")
PY

members="$(tar -tzf "${archive}")"

forbidden='(^|/)(manifest\.json|models|portal|tests|test-scope\.yaml|CMakeLists\.txt|CTestTestfile\.cmake|cmake_install\.cmake|Makefile|[^/]+_(unit|e2e)_test|sandbox[^/]*|__pycache__)(/|$)|\.(a|pyc|pyo|log|db|lock)$'
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
