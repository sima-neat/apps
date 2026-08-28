# Neat Apps

Read `CONTRIBUTING.md` before changing or reviewing an application. It is the
source of truth for application layout, README content, model documentation,
test registration, and validation commands.

## Application changes

- Keep the Neat Library flow easy to follow. Graph construction, model loading,
  execution, and teardown should be visible in the entrypoint. Do not hide
  public Neat API objects behind unnecessary helpers.
- Put customer-facing settings in `src/common/config.yaml`. Keep paths, models,
  hosts, ports, and machine-specific values configurable.
- Keep C++ and Python implementations aligned in models, thresholds, outputs,
  and metadata. Follow the language exceptions in `CONTRIBUTING.md`.
- Preserve CLI arguments, configuration keys, output files, and metadata unless
  the pull request intentionally changes them and updates their tests and docs.
- Treat missing runtime capabilities as Core work instead of adding an
  application-specific workaround.
- Measure before changing topology, queue policy, copies, or worker counts.
  Performance gains never excuse incorrect output, and claims must match what
  was measured.
- New applications normally include `src/common/config.yaml`, C++ and Python
  implementations, and unit and end-to-end tests. Explain every omission in the
  pull request.
- Keep credentials, customer information, internal material, generated files,
  and local artifacts out of this public repository.
- Keep agent plans, scratch notes, generated reports, and work records outside
  this repository.

## Tests and documentation

- Use `tests/test-scope.yaml` as the source of truth for enabled tests and test
  models. Register tests only there so the same coverage runs locally and in
  Vulcan CI.
- End-to-end tests must prove useful output, not only that the process starts.
  Unit tests must not download models or require live services.
- Fail clearly when a test prerequisite is broken. Never silently skip or fall
  back to different inputs.
- README commands, models, paths, configuration, and expected results must
  match the installed application. New portal applications need their own
  preview image.
- Run documented model-download and application commands exactly as written.

## Pull request reviews

- Review the complete affected application, including its tests, configuration,
  README, and preview. Do not review changed lines in isolation.
- Report only concrete problems introduced or exposed by the change. Anchor
  each finding to the smallest relevant changed line range and explain the
  condition, the breakage, and its impact.
- Report one finding per root cause. Skip style preferences, vague risks, and
  test requests that do not name the behavior that could regress.
- Flag changes to dependency pins, `build.sh`, or
  `.github/workflows/vulcan-ci.yml` for code owner review.

Run the applicable checks from `CONTRIBUTING.md` and check the latest Vulcan CI
run for the branch. Report anything not run or observed as `not verified`.
