# Contributing to Neat Apps

This repo hosts runnable application examples for the Neat Library. Contributions
should preserve that purpose: clear examples, current commands, explicit runtime
configuration, and tests that match the supported behavior.

## Example Contract

Every example lives under `examples/<category>/<example>/`.

Required for each example:
- `README.md`
- `tests/test-scope.yaml`
- `src/common/` when shared config, labels, or assets are needed
- `src/cpp/` when the example has a C++ implementation
- `src/python/` when the example has a Python implementation
- `tests/cpp/` or `tests/python/` when the matching coverage is enabled

Use the current `src/` and `tests/` layout. Do not add legacy root-level paths
such as `main.py`, `main.cpp`, `common/`, `python/`, or `cpp/`.

## Implementation

Prefer both C++ and Python for standard runtime examples. Python-only examples
are acceptable when the example is inherently Python-first, such as GenAI UI,
benchmarking, orchestration, or tooling.

Keep the Neat Library flow visible in the entrypoint. A contributor should be
able to read `src/cpp/main.cpp` or `src/python/main.py` and identify setup,
model loading, preprocessing, inference, postprocessing, and teardown without
chasing opaque wrappers.

Keep runtime inputs explicit:
- no local-only absolute paths
- no hardcoded board, RTSP, or Insight host values
- no hidden model defaults that differ from the README
- no stale product names in user-facing documentation

Use official names in documentation:
- Neat Library
- Neat Development Environment
- Model Compiler
- LLiMa

## Test Scope

`tests/test-scope.yaml` is the source of truth for enabled tests and e2e model
artifacts.

Use this shape:

```yaml
models:
  model_id:
    source: modelzoo
    name: model_name
    file: model_name_mpk.tar.gz
unit:
  python: true
  cpp: true
e2e:
  python:
    enabled: true
    models:
      - model_id
  cpp:
    enabled: true
    models:
      - model_id
```

If a language or e2e path is not supported, disable it in `test-scope.yaml`.
For disabled e2e coverage, include a short `reason`. Do not add placeholder
tests and enable them as real coverage.

Enabled coverage must have the matching files:
- Python unit: `tests/python/test_unit.py`
- Python e2e: `tests/python/test_e2e.py`
- C++ unit: `tests/cpp/test_unit.cpp`
- C++ e2e: `tests/cpp/test_e2e.cpp`

## README

Start from `examples/TEMPLATE_README.md`. Every example README must stay aligned
with the current source, config, supported models, and commands.

The README validator enforces:
- metadata fields: `Category`, `Difficulty`, `Tags`, `Languages`, `Status`,
  `Binary Name`, `Model`
- sections: `Metadata`, `Concept`, `Prerequisites`, `Get The Apps Repo`, `Run`,
  `Source Files`

For portal-facing examples, add a `## Preview` section after `## Concept` and
store the image at `portal/assets/examples/<category>/<example>/image.*`.

Use `models/` for model artifact examples. When documenting model
downloads, use the platform version placeholder and tell users to use the
platform version.

## Build and Test

`build.sh` builds. `tests/test.sh` tests. Keep those responsibilities separate.

Standard validation from the apps repo:

```bash
./build.sh --clean
./tests/test.sh --unit
python3 scripts/validate_readmes.py
python3 scripts/generate_catalog.py >/tmp/apps-catalog.json
```

Run e2e validation when the required hardware, model, RTSP, and service
prerequisites are available:

```bash
./tests/test.sh --e2e
./tests/test.sh --e2e --strict
```

Compile C++ examples in the Neat Development Environment. Run Neat runtime and
e2e checks on SiMa hardware such as Modalix or DevKit. Host checks are useful
for documentation, source inspection, and light static validation only.

## CMake and Scripts

When adding a C++ implementation:
- add `src/cpp/CMakeLists.txt`
- register the example through the category `CMakeLists.txt`
- keep C++ test filenames aligned with `cmake/ExampleModule.cmake`

When changing layout, commands, models, or test behavior, check the affected
repo contracts:
- `build.sh`
- `tests/test.sh`
- `tests/utils/test_scope.py`
- `scripts/validate_readmes.py`
- `scripts/generate_catalog.py`
- `.github/workflows/*.yml`

## Review Checklist

Before asking for review:
- README commands match the source and runtime behavior.
- `tests/test-scope.yaml` matches implemented coverage.
- Enabled tests have real test files.
- Disabled e2e paths have a reason.
- Model names, paths, and download commands are current.
- Portal examples have `## Preview` and `portal/assets/examples/.../image.*`.
- `./build.sh --clean` succeeds when source changes require a build.
- README validation and catalog generation pass for documentation changes.
