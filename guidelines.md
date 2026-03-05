# NEAT Apps Contributor Guidelines

This document defines how contributors should add and maintain examples in `apps/examples`.

## Goals
- Preserve runtime behavior unless a change is explicitly requested.
- Keep examples pedagogical: users should understand NEAT API usage by reading `main.py` and `main.cpp`.
- Keep every example consistent in layout, testing, and documentation.

## Required Example Layout
Every example under `examples/<category>/<example>` must follow:

```text
<example>/
  README.md
  cpp/
    CMakeLists.txt
    main.cpp
    tests/
      CMakeLists.txt
      unit_test.cpp
      e2e_test.cpp
  python/
    main.py
    requirements.txt
    tests/
      test_unit.py
      test_e2e.py
  common/
```

Notes:
- Do not keep legacy paths (`main.py`, `main.cpp`, `tests/python`, `tests/cpp`) at example root.
- Keep shared labels/config/assets in `common/` so both C++ and Python use the same files.

## Templates and Scaffolding
Use these sources of truth:
- README template: `examples/TEMPLATE_README.md`
- Example scaffold script: `scripts/create_example_scaffold.sh`

To add a new example:
1. Run `scripts/create_example_scaffold.sh`.
2. Fill in both implementations (`cpp/main.cpp`, `python/main.py`).
3. Replace scaffold placeholder tests with real unit/e2e tests.
4. Update `README.md` using `examples/TEMPLATE_README.md` sections.

## Implementation Expectations
- Prefer implementing both C++ and Python paths for each example.
- Keep core API calls explicit and visible (do not hide all logic behind opaque wrappers).
- Keep argument parsing and runtime wiring obvious to readers.
- Keep lifecycle stages clear: setup/build, run loop, teardown/reporting.

## Testing Requirements (Mandatory)
For each example, both languages must have:
- Unit tests
- End-to-end tests

Expected files:
- Python: `python/tests/test_unit.py`, `python/tests/test_e2e.py`
- C++: `cpp/tests/unit_test.cpp`, `cpp/tests/e2e_test.cpp`

Minimum quality bar:
- Unit tests validate CLI and argument/error behavior.
- E2E tests validate real execution path and output contracts.
- Tests should be deterministic and skip only when external prerequisites are unavailable.

## Build and Test Workflow
From `apps/`:

```bash
./build.sh --clean
source tests/scripts/testing/setup_test_env.sh
./tests/test.sh --unit
./tests/test.sh --e2e
python3 scripts/validate_readmes.py
```

`build.sh` compiles examples and tests.

## README Requirements
Each example README must be updated when paths or behavior-relevant commands change.

At minimum, include:
- Metadata
- Concept
- Prerequisites
- Command-line options
- Build
- Run
- Debugging notes
- Source files

Path examples in README must use current layout:
- `examples/<category>/<example>/python/main.py`
- `examples/<category>/<example>/python/requirements.txt`
- CMake direct build from `.../<example>/cpp`

## CMake and CI/CD Notes
- Category CMake files must register examples via `<example>/cpp`.
- Keep `cmake/ExampleModule.cmake` paths aligned with `cpp/tests/*`.
- If layout changes affect build/test/package scripts, update:
  - `build.sh`
  - `tests/test.sh`
  - `.github/workflows/*.yml` as needed

Always verify CI entrypoints still work after structural changes.

## Contributor Checklist
Before asking for review:
1. Layout follows the required structure.
2. C++ and Python implementations are both present (preferred and expected).
3. Unit + e2e tests exist and run.
4. `README.md` commands and file paths are current.
5. `./build.sh --clean` succeeds.
6. `./tests/test.sh --unit` and `./tests/test.sh --e2e` succeed (or clearly report environmental skips).
7. `python3 scripts/validate_readmes.py` succeeds.
