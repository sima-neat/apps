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

A Python implementation reaches the shared helpers through a relative symlink,
`src/python/sima_examples -> ../../../../../support/python/sima_examples`, so
`main.py` imports `sima_examples` with no path code. Create it with
`ln -s ../../../../../support/python/sima_examples src/python/sima_examples`
from the example folder; `scripts/create_example_scaffold.sh` creates it for a
new example. `build.sh` and `scripts/get-example.sh` copy the files the link
points to, so the installed bundle and a fetched example hold no links. On
Windows, clone with `git config core.symlinks true` and symlink permission, or
git writes the link as a text file and the import fails. The high-density and
RF-DETR examples still find `support/python` in a folder above `main.py`.

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

Each section is one step the customer takes, in that order, and every command
runs as written from `prebuilt-apps/`. Optional sections go in these places:

- a `Prepare <input>` section, such as `Prepare Insight`, after `Prepare the Model`
- `Expected Result`, `Troubleshooting`, and `Performance` after `Run`

Keep each section minimal and precise. `Concept` starts with the one- or
two-sentence portal summary, then shows the graph topology as a flowgraph in a
`text` code block, drawn as `agents/references/flowgraph.md` describes: the
inputs, decoders, models, outputs, and senders the entrypoint connects, and the
link policies between them. `Configure` names only the keys the
customer edits, one bullet each, with the value or where it comes from. Explain
behavior in `Concept`, and list error messages with their fixes in
`Troubleshooting`.

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
- whether each model comes from the model browser, Model Zoo, a direct
  artifact, or an example-specific installer
- the corresponding `sima-cli` command when that source supports it
- the downloaded package selected by `model.path`

Use `models/` for user-managed packages. When a model is in the
[model browser](https://developer.sima.ai/models/), download it from there
with `sima-cli models download --id <model-id> --variant <variant>` and list
only the variants the browser offers. Use the Model Zoo only for a model the
browser does not have. Model Zoo downloads use the Model Zoo version, which
can differ from the installed platform version. Take it from
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

Config files use the key names in `examples/config-keys.yaml` and put settings
only one example has under `app.`. Every key is required, and the application
rejects keys it does not read.

Bundle instructions run from `prebuilt-apps/`. Define
`APP_DIR=examples/<category>/<example>` once and reuse it so commands remain
readable without changing relative model, asset, label, or output paths.
Pretty-print structured output examples.

## Test Scope

`tests/test-scope.yaml` is the source of truth for enabled tests and model
artifacts required by e2e coverage.

```yaml
models:
  model_id:              # a model in the model browser
    source: model-registry
    name: model_name
    ref: main
    spec: latest
    variant: w8a8
    file: model_name_w8a8_mpk.tar.gz
  other_model_id:        # a model only the Model Zoo has
    source: modelzoo
    name: other_model_name
    file: other_model_name_mpk.tar.gz
unit:
  python: true
  cpp: true
e2e:
  python:
    enabled: true
    requires_rtsp: false # true for live-stream suites; omission defaults to true
    models:
      - model_id
    variants:          # optional: also run the suite with these models when
      - other_model_id # SIMANEAT_APPS_TEST_MODEL_VARIANTS=1, as the nightly run does
  cpp:
    enabled: true
    requires_rtsp: false
    models:
      - model_id
```

The first entry under `models` is the model the suite runs with; every entry is
downloaded. `variants` are exercised only when `SIMANEAT_APPS_TEST_MODEL_VARIANTS=1`,
which the nightly workflow sets, so a pull request run stays at one model per suite.

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

Both Python scripts need Python 3.10 or newer. Under 3.9 they fail at import
with `TypeError: unsupported operand type(s) for |`, which reads like a bug in
the script rather than a version mismatch. macOS ships 3.9 as the system
`python3`; run them with `python3.11` or another 3.10+ interpreter there.

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

## Installed Layout

| Path | Purpose |
| --- | --- |
| `examples/<category>/<example>/README.md` | Example setup and run instructions |
| `examples/<category>/<example>/src/cpp/pre-built/` | Packaged C++ executables |
| `examples/<category>/<example>/src/cpp/` | C++ implementation reference |
| `examples/<category>/<example>/src/python/` | Python implementation and dependencies |
| `examples/<category>/<example>/src/common/` | Shared runtime configuration and data |
| `support/README.md` | What each shared helper does |
| `support/python/` | Helpers the Python examples share. An example that links them from `src/python/sima_examples` in the repository holds its own copy in that folder here |
| `support/cpp/` | Source of the helpers the C++ executables are built with, as reference |
| `assets/datasets/` | Shipped runtime sample data |
| `models/` | User-managed model packages |
| `deps/neat-apps.json` | Installed Apps ref and exact commit |
| `deps/neat-core.json` | Core artifact selected for this Apps build |

The installed bundle does not contain CMake files or tests. Clone the repository
to compile source or run the Apps test suite.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `examples/` | Application examples organized by category |
| `examples/config-keys.yaml` | Config keys the examples share, checked by `tests/scripts/test_config_keys.py` |
| `support/` | The shared library: code two or more examples need, under the same file and function names in each language. `support/README.md` lists the modules, and `agents/rules.md` says what goes there instead of an example's `utils/` |
| `skills/` | Customer skills that sima-cli installs |
| `agents/` | Rules and skills for contributors' agents and the automatic review |
| `assets/datasets/` | Runtime datasets shipped with Apps |
| `assets/datasets-test/` | Repository-only test fixtures |
| `models/` | Ignored user-managed model packages |
| `tests/` | Test runner and repository test support |
| `deps/manifest.json` | Source dependency and platform declaration |
| `portal/` | Examples portal source |
| `portal/assets/readme/` | Root README hero image |

## Dependency Selection

`deps/manifest.json` selects Core and records the SDK channel, the platform
version, and the Model Zoo version used for model downloads. Development uses
the `snap` policy, while a release-preparation branch can select an exact Core
artifact. Vulcan records the resolved Core target under package metadata and
the Apps installer uses that same target. Insight is not pinned in the Apps
manifest.

Builds inherit the sysroot requirement from Core's bundled Scheduler. Official
releases require a matching cached SDK receipt, package inventory and package
identity, without overlays; they never run `sysroot update`. Development pins
retain the update path. CI uses the same Scheduler policy with the shared
compatibility resolver to select and verify DevKit firmware before
installation, then shards tests using its resolved labels.

## Review Checklist

- Installed README commands match the packaged runtime.
- Default and supported models match the implementation and configuration.
- Every model has an accurate acquisition path.
- `tests/test-scope.yaml` matches enabled coverage.
- Disabled e2e paths explain why.
- Portal examples include their preview asset.
- Source changes build successfully.
- README validation and catalog generation pass.
