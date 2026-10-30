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

## Build artifact handoff

Post-build validation stages both Apps archives and the checked Scheduler policy
under `_vulcan-build-artifacts/`. Pass that single directory to the shared
uploader, which prefixes its workspace only once. Multiple relative paths in
this input leave later lines relative to the Actions checkout and can omit the
policy or introduce host directory prefixes into the artifact.

The downloaded artifact exposes both archives at its root and the policy at
`deps/scheduler-sdk-policy.json`, as required by runner selection. A missing
policy fails post-build preparation before upload.
