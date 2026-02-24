# SiMa NEAT Apps

![Standard Build & Archive](https://img.shields.io/badge/Standard%20Build%20%26%20Archive-TBA-lightgrey)
![Fuzz Nightly](https://img.shields.io/badge/Fuzz%20Nightly-TBA-lightgrey)
![Crash Correctness Nightly](https://img.shields.io/badge/Crash%20Correctness%20Nightly-TBA-lightgrey)
![SDK](https://img.shields.io/badge/SDK-2.0-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)

Source-first example applications for SiMa NEAT.

This repo is intentionally separate from `core`:
- `core` installs the NEAT SDK/runtime/tooling
- `apps` holds customer-facing applications and sample code

## Customer Workflow (Early Release)

1. Install NEAT/core dependencies by following the official guide:
   https://neat.modalix.info/getting-started/install

2. Clone this repo and build/run examples.

This keeps examples editable and easy to customize.

## Repo Layout

- `examples/cpp/`: C++ examples organized by task/category
- `examples/python/`: reserved for future Python examples (same category model)
- `support/`: shared C++ helper code used by multiple examples
- `utils/rtsp/`: RTSP helper scripts used by streaming demos
- `scripts/ci/`: CI scripts (public include boundary check, quality gates)
- `scripts/cd/`: continuous delivery scripts (reserved)
- `scripts/release/`: release scripts (reserved)
- `schemas/`: JSON schemas (e.g. catalog validation)
- `neat-core.json`: NEAT core SDK dependency declaration (branch, version, install method)
- `catalog.json`: example catalog (for CI/release traceability)

## Build

Install NEAT core SDK and build all examples in one step:

```bash
./build.sh --all
```

If NEAT core is already installed, build only:

```bash
./build.sh
```

To install NEAT core SDK without building (useful for app developers):

```bash
./build.sh --only-install-neat-core
```

If NEAT is installed in a non-standard prefix, set `CMAKE_PREFIX_PATH`:

```bash
CMAKE_PREFIX_PATH=/opt/sima-neat ./build.sh
```

## Python Examples

Python examples are not implemented yet.
`examples/python/` will be populated as Python examples are migrated.

## NEAT Core Dependency

`neat-core.json` declares which NEAT core SDK branch and version this repo depends on.
`./build.sh --all` reads this file to install the correct SDK before building.
