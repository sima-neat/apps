# Source of truth

## Locate the installed contract

Use the headers and Python bindings from the installation that will run the
application. Packaged Core source and docs must match that version when used to
explain behavior; an unrelated checkout or newer example cannot override it.

Run `neat` (or `neat --json`) in the Neat Development Environment and on the
DevKit to identify the installed Core, PyNeat, Scheduler, Runner, and LLiMa
versions. In the Neat Development Environment, public headers live under
`$SYSROOT/usr/include` (`/opt/toolchain/aarch64/modalix/usr/include`). Packaged
Core source lives under `/neat-resources/core-src` only in images whose
`/etc/sdk-release` reports `Neat Core = bundled`; otherwise use the Core source
at the installed version's tag. On the target, check `/usr/include` and the
installed Python package. Core docs live under `docs/develop-apps/` and
`docs/reference/` in Core source.

Search for the task's symbol in its relevant header, binding, or documentation
page. The API surface map identifies header families when the owner is unknown.
If packaged source is absent, use the available installed contract and report
what was inspected. If no matching installation is accessible, continue
independent application work and identify the API or runtime claims that remain
unverified.

## Locate examples

Packaged Apps examples live under `/neat-resources/apps-src/examples` in
Core-bundled images. If absent, use an available Apps checkout or the public
[Apps examples](https://github.com/sima-neat/apps/tree/main/examples).
Check compatibility with the installed contract before copying an example from
a different version. Where an example and the installed headers disagree, such
as an example calling a deprecated factory, follow the headers.
