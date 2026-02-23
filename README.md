# SiMa NEAT Apps

Source-first example applications for SiMa NEAT.

This repo is intentionally separate from `core`:
- `core` installs the NEAT SDK/runtime/tooling
- `apps` holds customer-facing applications and sample code

## Customer Workflow (Early Release)

1. Install NEAT/core dependencies:

```bash
sima-cli install -m https://neat-artifacts.modalix.info/neat/main/latest/metadata-all.json
```

2. Clone this repo and build/run examples.

This keeps examples editable and easy to customize.

## Repo Layout

- `examples/cpp/`: C++ examples organized by task/category
- `examples/python/`: reserved for future Python examples (same category model)
- `support/`: shared C++ helper code used by multiple examples
- `utils/rtsp/`: RTSP helper scripts used by streaming demos
- `ci/`: apps CI configuration (including NEAT core metadata selector)
- `manifests/`: apps catalog and release metadata artifacts (for CI/release traceability)

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
