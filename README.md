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
- `ci/`: apps CI configuration (including NEAT core metadata selector and schemas)
- `catalog.json`: example catalog (for CI/release traceability)

## C++ Build (requires installed NEAT SDK)

```bash
./build.sh
```

If NEAT is installed in a non-standard prefix, set `CMAKE_PREFIX_PATH`:

```bash
CMAKE_PREFIX_PATH=/opt/sima-neat ./build.sh
```

## Python Examples

Python examples are not implemented yet.
`examples/python/` will be populated as Python examples are migrated.

## CI NEAT Dependency Source

`ci/neat-core-link.json` controls which NEAT/core metadata manifest URL is used by apps CI.
