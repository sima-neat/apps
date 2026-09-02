# Neat Apps

Read `CONTRIBUTING.md` before changing or reviewing an application. It is the
source of truth for application layout, README content, model documentation,
test registration, and validation commands.

## Application changes

- Start from the linked issue and the nearest shipped applications. Prefer one
  config-driven application flow, one entrypoint per supported language, and
  one optional UI or controller. Add another script, tool, process, or workflow
  only when it provides distinct required behavior in the packaged application.
- Keep application-specific files inside the owning application. Put files
  shared by its C++ and Python implementations under `src/common`, and use the
  existing repository-level directories for cross-application utilities.
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
- Match input and output handling to the application contract. Offline and
  batch applications may read files or directories and save or display results,
  including annotated images. When an application provides real-time
  visualization, use Insight as the only supported visualization UI and publish
  live video and metadata through the Neat APIs.
- Keep system provisioning and recovery outside application code. Platform
  tooling owns host, network, mount, device-recovery, SDK, and unrelated process
  management.
- Keep credentials, customer information, internal material, and disposable
  build or runtime artifacts out of this public repository.
- Keep agent plans, scratch notes, generated reports, and work records outside
  this repository.

## Tests and documentation

- Use `tests/test-scope.yaml` as the source of truth for enabled tests and test
  models. Ensure enabled C++ tests are also registered with CMake.
- End-to-end tests must prove useful output, not only that the process starts.
  Unit tests must not download models or require live services.
- Fail clearly when a test prerequisite is broken. Never silently skip or fall
  back to different inputs.
- README commands, models, paths, configuration, and expected results must
  match the installed application. New portal applications need their own
  preview image.
- Run documented model-download and application commands exactly as written.

## Pull request reviews

- Compare the pull request claims, linked issue, and implemented behavior.
  Report when the implementation solves a different problem, omits claimed
  behavior, or expands the scope without justification.
- Compare the design with the nearest shipped applications. Report duplicate
  controllers, UIs, process stacks, entrypoints, setup paths, misplaced
  application files, or complexity without distinct packaged customer value.
- Trace one supported customer workflow through the package: install,
  configure, run, and inspect the result. Report source-only paths, missing
  packaged resources, hardcoded environment setup, or customer options spread
  across configuration, CLI arguments, and environment variables.
- Check visualization against the application mode. File or directory input and
  saved or locally displayed output are valid for offline applications. When an
  application provides real-time visualization, require Insight as the only
  supported visualization UI and Neat video and metadata APIs for publishing
  live results.
- Report application code that performs platform provisioning or recovery,
  including host filesystem, network, mount, device-recovery, SDK, or unrelated
  process management.
- For stateful or dynamic applications, review lifecycle transitions, resource
  limits, cleanup, persistence safety, and the association of cached results
  with their source. Tests must cover failure, replacement, stop, and restart
  paths when those behaviors exist.
- Map every advertised behavior to an end-to-end assertion and to both language
  implementations when the application supports C++ and Python. A test that
  only starts the process is not evidence for the feature.
- When model-preparation files change, check whether the Model Registry should
  own the artifacts. Temporary application-specific preparation may remain
  under the owning application's `src/common`, but it must be one
  development-only flow and must not become a packaged runtime dependency.
- Before marking a new or materially changed application ready, verify or
  require evidence that the exact branch package was installed with
  `sima-cli neat install apps@<branch>` and that its installed README produced
  the advertised result. The pull request description must record this
  evidence.
- Separate correctness, structure, packaging, and customer-workflow blockers
  from optional performance follow-ups.
- Review the complete affected application, including its tests, configuration,
  README, and preview. Do not review changed lines in isolation.
- Report only concrete problems introduced or exposed by the change. Anchor
  each finding to the smallest relevant changed line range and explain the
  condition, the breakage, and its impact.
- Write comments in plain English. Avoid unexplained jargon and state the
  expected behavior and what should change so the contributor can act without
  guessing.
- Report one finding per root cause. Skip style preferences, vague risks, and
  test requests that do not name the behavior that could regress.
- Flag changes to dependency pins, `build.sh`, or
  `.github/workflows/vulcan-ci.yml` for code owner review.

Run the applicable checks from `CONTRIBUTING.md` and check the latest Vulcan CI
run for the branch. Report anything not run or observed as `not verified`.
