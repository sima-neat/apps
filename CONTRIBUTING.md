# Contributing to Neat Apps

Neat Apps contains runnable application examples for the Neat Library.
Contributions must keep the public examples readable, executable, testable, and
consistent with the installed `prebuilt-apps/` bundle.

## Example Layout

Every example lives under `examples/<category>/<example>/`.

Required:

- `README.md`
- `tests/test-scope.yaml`
- `src/common/` when C++ and Python share configuration or runtime data
- `src/cpp/` for C++ implementations
- `src/python/` for Python implementations
- matching files under `tests/cpp/` or `tests/python/` for enabled coverage

Do not add legacy root-level paths such as `main.py`, `main.cpp`, `common/`,
`python/`, or `cpp/`.

## Implementation

Prefer both C++ and Python for standard runtime examples. Python-only examples
are appropriate for GenAI UI, benchmarking, orchestration, and tooling.

Keep the Neat Library flow visible in each entrypoint. Readers should be able to
identify setup, model loading, preprocessing, inference, postprocessing, and
teardown without following opaque wrappers.

Keep runtime inputs explicit:

- no local absolute paths
- no hardcoded board, RTSP, or Insight hosts
- no hidden model defaults that disagree with the README
- no stale product names in user-facing text

Use the official product names: Neat Library, Neat Development Environment,
Model Compiler, and LLiMa.

## README Contract

Start from `examples/TEMPLATE_README.md`. Write the README as one path from
release installation to a successful run. Keep source development secondary.

The validator requires these sections in order:

1. `Metadata`
2. `Concept`
3. `Preview`
4. `Prerequisites`
5. `Install Apps`
6. `Prepare the Model`
7. `Configure`
8. `Run`
9. `Source Files`
10. `Development From Source`

Optional sections such as `Expected Result` and `Troubleshooting` belong where
they support that sequence.

The installed workflow must:

- install the latest Apps runtime with `sima-cli neat install apps`
- run subsequent commands from `prebuilt-apps/`
- use `src/cpp/pre-built/<binary>` for C++
- use packaged `src/python/` entrypoints or supported wrappers for Python
- keep clone, CMake, build, and test commands out of the primary workflow

The installed package excludes CMake files and tests. List only packaged files
under `Source Files`, and describe C++ source as implementation reference.
Link `Development From Source` back to this guide instead of copying build and
test instructions into every example.

## Model Documentation

Each example README must identify:

- its default model
- every additional model explicitly supported by its preprocessing and
  postprocessing path
- whether each model comes from Model Zoo, a direct artifact, or an
  example-specific installer
- the corresponding `sima-cli` command when that source supports it
- the downloaded package selected by `model.path`

Use `models/` for user-managed packages. Model downloads use the Model Zoo
version, which can differ from the installed platform version. Take it from
`modelzoo-version` in `deps/manifest.json`, and keep the literal documented in
example READMEs in sync with that field.

Do not infer application support from every model declared in
`tests/test-scope.yaml`. That file owns test acquisition. Confirm supported
models against the config and implementation, and distinguish them from the
default model.

GenAI examples may use their own setup scripts for Hugging Face model
directories. Do not document a `sima-cli` model download when the command is
not supported.

## Portal Metadata

README metadata fields are:

- `Category`
- `Difficulty`
- `Tags`
- `Languages`
- `Status`
- `Binary Name`
- `Model`

The first paragraph under `Concept` becomes the portal card summary. Write one
or two plain-English sentences, no more than 200 normalized characters, that
say what the application does and produces. Name the main model family when it
helps users understand the application. Do not use Markdown, executable names,
implementation details, or filler such as "This example" in the summary.

Every portal example needs a `Preview` section immediately after `Concept` and
an application-specific image under
`portal/assets/examples/<category>/<example>/image.*`. Check that it is clear in
both the catalog card and detail page and contains no sensitive information.

Treat the packaged `src/common/config.yaml` as the source of truth. Point users
to it and name only the settings they must change. Do not copy the full file
into the README. A focused snippet is appropriate only for an override or
generated configuration that the package does not provide.

Bundle instructions run from `prebuilt-apps/`. Define
`APP_DIR=examples/<category>/<example>` once and reuse it so commands remain
readable without changing relative model, asset, label, or output paths.
Pretty-print structured output examples.

## Test Scope

`tests/test-scope.yaml` is the source of truth for enabled tests and model
artifacts required by e2e coverage.

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

Disable unsupported coverage in `test-scope.yaml` and include a short reason
for disabled e2e paths. Enabled coverage must have the matching test file:

- Python unit: `tests/python/test_unit.py`
- Python e2e: `tests/python/test_e2e.py`
- C++ unit: `tests/cpp/test_unit.cpp`
- C++ e2e: `tests/cpp/test_e2e.cpp`

## Build and Test

`build.sh` builds. `tests/test.sh` tests.

```bash
./build.sh --clean
./tests/test.sh --unit
python3 scripts/validate_readmes.py
python3 scripts/generate_catalog.py >/tmp/apps-catalog.json
```

Run e2e validation when its hardware, model, RTSP, and service prerequisites are
available:

```bash
./tests/test.sh --e2e
./tests/test.sh --e2e --strict
```

Compile C++ examples in the Neat Development Environment. Run runtime and e2e
checks on supported SiMa hardware. Host checks cover documentation, source
hygiene, and lightweight static validation.

When adding C++:

- add `src/cpp/CMakeLists.txt`
- register the example through its category `CMakeLists.txt`
- keep C++ test filenames aligned with `cmake/ExampleModule.cmake`

When changing layout, commands, models, or tests, inspect `build.sh`,
`tests/test.sh`, `tests/utils/test_scope.py`, README validation, catalog
generation, and affected CI workflows.

## Review Checklist

- Installed README commands match the packaged runtime.
- Default and supported models match the implementation and configuration.
- Every model has an accurate acquisition path.
- `tests/test-scope.yaml` matches enabled coverage.
- Disabled e2e paths explain why.
- Portal examples include their preview asset.
- Source changes build successfully.
- README validation and catalog generation pass.
