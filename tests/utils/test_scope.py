"""Validate and query the apps test-scope contract."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml


APPS_ROOT = Path(__file__).resolve().parents[2]
VALID_SOURCES = {"modelzoo", "url", "huggingface"}
VALID_LANGUAGES = {"python", "cpp"}
VALID_KINDS = {"unit", "e2e"}


def load_scope(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), dict):
        raise ValueError(f"{path} must contain an examples mapping")
    return payload


def example_paths(apps_root: Path) -> set[str]:
    return {
        str(readme.parent.relative_to(apps_root / "examples"))
        for readme in (apps_root / "examples").glob("*/*/README.md")
    }


def example_name(example_key: str) -> str:
    return example_key.split("/", 1)[1]


def test_source_path(apps_root: Path, example_key: str, language: str, kind: str) -> Path:
    if language == "python":
        filename = "test_unit.py" if kind == "unit" else "test_e2e.py"
        return apps_root / "examples" / example_key / "python" / "tests" / filename
    filename = "test_unit.cpp" if kind == "unit" else "test_e2e.cpp"
    return apps_root / "examples" / example_key / "cpp" / "tests" / filename


def e2e_config(entry: dict[str, Any], language: str) -> dict[str, Any]:
    e2e = entry.get("e2e", {})
    config = e2e.get(language, {})
    if not isinstance(config, dict):
        raise ValueError(f"e2e.{language} must be a mapping")
    return config


def is_enabled(entry: dict[str, Any], language: str, kind: str) -> bool:
    if kind == "unit":
        unit = entry.get("unit", {})
        if not isinstance(unit, dict):
            raise ValueError("unit must be a mapping")
        return bool(unit.get(language, False))
    return bool(e2e_config(entry, language).get("enabled", False))


def enabled_models(entry: dict[str, Any], language: str) -> list[str]:
    raw = e2e_config(entry, language).get("models", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("e2e model selection must be a list of strings")
    return raw


def validate_scope(scope: dict[str, Any], apps_root: Path) -> list[str]:
    errors: list[str] = []
    scoped_examples = set(scope["examples"])
    actual_examples = example_paths(apps_root)

    for missing in sorted(actual_examples - scoped_examples):
        errors.append(f"missing scope entry for examples/{missing}")
    for extra in sorted(scoped_examples - actual_examples):
        errors.append(f"scope references missing example: examples/{extra}")

    for example_key, entry in sorted(scope["examples"].items()):
        if not isinstance(entry, dict):
            errors.append(f"{example_key}: entry must be a mapping")
            continue
        models = entry.get("models", {})
        if not isinstance(models, dict):
            errors.append(f"{example_key}: models must be a mapping")
            models = {}
        for model_id, model in models.items():
            if not isinstance(model, dict):
                errors.append(f"{example_key}: model {model_id} must be a mapping")
                continue
            source = model.get("source")
            if source not in VALID_SOURCES:
                errors.append(f"{example_key}: model {model_id} has invalid source {source!r}")
            if source == "modelzoo" and not str(model.get("name", "")).strip():
                errors.append(f"{example_key}: modelzoo model {model_id} needs name")
            if source == "url" and not str(model.get("url", "")).strip():
                errors.append(f"{example_key}: url model {model_id} needs url")
        for language in sorted(VALID_LANGUAGES):
            for kind in sorted(VALID_KINDS):
                try:
                    enabled = is_enabled(entry, language, kind)
                except ValueError as exc:
                    errors.append(f"{example_key}: {exc}")
                    continue
                if enabled and not test_source_path(apps_root, example_key, language, kind).exists():
                    errors.append(f"{example_key}: {language} {kind} is enabled but test file is missing")
            try:
                for model_id in enabled_models(entry, language):
                    if model_id not in models:
                        errors.append(
                            f"{example_key}: e2e.{language} references undefined model {model_id}"
                        )
                if is_enabled(entry, language, "e2e") and not enabled_models(entry, language):
                    errors.append(f"{example_key}: e2e.{language} is enabled but models is empty")
            except ValueError as exc:
                errors.append(f"{example_key}: {exc}")
    return errors


def enabled_examples(scope: dict[str, Any], language: str, kind: str) -> list[str]:
    return [
        example_key
        for example_key, entry in sorted(scope["examples"].items())
        if is_enabled(entry, language, kind)
    ]


def ctest_regex(scope: dict[str, Any], kind: str) -> str:
    names: list[str] = []
    for example_key in enabled_examples(scope, "cpp", kind):
        name = re.escape(example_name(example_key))
        if kind == "unit":
            names.append(f"{name}\\.unit")
            names.append(f"{name}\\.requires_rtsp")
        else:
            names.append(f"{name}\\.e2e")
            names.append(f"{name}\\.metadata_json_e2e")
    return "^(" + "|".join(names) + ")$" if names else ""


def python_files(scope: dict[str, Any], kind: str) -> list[Path]:
    return [
        test_source_path(APPS_ROOT, example_key, "python", kind)
        for example_key in enabled_examples(scope, "python", kind)
    ]


def scoped_models(
    scope: dict[str, Any], languages: list[str], kind: str
) -> list[tuple[str, dict[str, Any]]]:
    if kind == "unit":
        return []

    result: dict[str, dict[str, Any]] = {}
    for example_key, entry in sorted(scope["examples"].items()):
        models = entry.get("models", {})
        for language in languages:
            if not is_enabled(entry, language, "e2e"):
                continue
            for model_id in enabled_models(entry, language):
                result.setdefault(model_id, models[model_id])
    return sorted(result.items())


def model_field(model: dict[str, Any], key: str) -> str:
    return str(model.get(key, "") or "").replace("\t", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-file", type=Path, default=APPS_ROOT / "tests/configs/test-scope.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--quiet", action="store_true")

    py_files = sub.add_parser("python-files")
    py_files.add_argument("--kind", choices=sorted(VALID_KINDS), required=True)

    regex = sub.add_parser("ctest-regex")
    regex.add_argument("--kind", choices=sorted(VALID_KINDS), required=True)

    examples = sub.add_parser("enabled-examples")
    examples.add_argument("--kind", choices=sorted(VALID_KINDS), required=True)
    examples.add_argument("--language", choices=sorted(VALID_LANGUAGES), required=True)

    models = sub.add_parser("models")
    models.add_argument("--kind", choices=sorted(VALID_KINDS), default="e2e")
    models.add_argument("--language", choices=sorted(VALID_LANGUAGES), action="append", required=True)

    args = parser.parse_args()
    scope = load_scope(args.scope_file)
    errors = validate_scope(scope, APPS_ROOT)
    if errors:
        for error in errors:
            print(f"[scope] {error}", file=sys.stderr)
        return 1

    if args.command == "validate":
        if not args.quiet:
            print(f"validated {args.scope_file}")
        return 0
    if args.command == "python-files":
        for path in python_files(scope, args.kind):
            print(path)
        return 0
    if args.command == "ctest-regex":
        print(ctest_regex(scope, args.kind))
        return 0
    if args.command == "enabled-examples":
        for example_key in enabled_examples(scope, args.language, args.kind):
            print(example_key)
        return 0
    if args.command == "models":
        for model_id, model in scoped_models(scope, args.language, args.kind):
            fields = [
                model_id,
                model_field(model, "source"),
                model_field(model, "name"),
                model_field(model, "url"),
                model_field(model, "file"),
                model_field(model, "repo"),
                model_field(model, "path"),
            ]
            print("\t".join(fields))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
