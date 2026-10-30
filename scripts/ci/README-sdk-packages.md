# SDK package preparation

Apps uses `SDK_PACKAGE_HELPER`, provided by the shared
`sima-neat/.github/.github/workflows/vulcan-build.yml@main`, to configure the
selected SDK container and install missing Node.js/npm and ripgrep tools before
the CLI bootstrap and build.

The helper retries recognized dpkg/APT lock contention for 120 seconds by
default, reports lock-owner diagnostics, and preserves unrelated failures.
It never removes locks or terminates their owners. Package operations run as
root inside the selected container.

See the [shared helper contract](https://github.com/sima-neat/.github/tree/main/actions/setup-sdk-package-helper)
for timeout settings and regression coverage.
