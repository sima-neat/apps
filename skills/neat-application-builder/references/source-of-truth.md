# Source of truth

## Locate the installed contract

Use the headers and Python bindings from the installation that will run the
application. Packaged Core source and docs must match that version when used to
explain behavior; an unrelated checkout or newer example cannot override it.

In the Neat Development Environment, look for packaged Core source under
`/neat-resources/core-src` and public headers under `$SYSROOT/usr/include` or
`/opt/toolchain/aarch64/modalix/usr/include`. On the target, check `/usr/include`
and the installed Python package. Core docs live under
`docs/develop-apps/` and `docs/reference/` in packaged source.

Search for the task's symbol in its relevant header, binding, or documentation
page. The API surface map identifies header families when the owner is unknown.
If packaged source is absent, use the available installed contract and report
what was inspected. If no matching installation is accessible, continue
independent application work and identify the API or runtime claims that remain
unverified.

## Locate examples

Packaged Apps examples live under `/neat-resources/apps-src/examples`. If absent,
use an available Apps checkout or the public
[Apps examples](https://github.com/sima-neat/apps/tree/main/examples).
Check compatibility with the installed contract before copying an example from
a different version.
