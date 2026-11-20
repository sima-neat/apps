---
name: apps-review-pr
description: Review a sima-neat/apps pull request for merge readiness. Use for a full review, for a review of one example's change, and for judging whether a change should update the customer skills.
---

# Review an Apps Pull Request

Review the complete affected example: entrypoints, configuration, tests, README, and preview.
A changed line can break code it does not touch. `agents/coding-style.md`, `agents/rules.md`,
and `CONTRIBUTING.md` own the rules. This skill says what to check and in which order, and
each finding cites the owning file and section.

## Steps

1. Read the linked issue and the pull request description. List every behavior the change
   claims. Check the description against `agents/skills/write-pr-description/SKILL.md` and
   report each section, graph, screenshot, or test row it lacks.
2. Read `agents/coding-style.md`, `agents/rules.md`, and the `CONTRIBUTING.md` sections for the
   areas the change touches.
3. Trace each mode the README advertises from source to result: the input, the model, every
   output, and the example's own logic. Individually valid API calls do not prove the topology
   fits the requirement, so judge it with `skills/neat-application-builder/SKILL.md`. The step
   is done when every claimed behavior maps to code you read, or is reported missing.
4. Check every item under Shape and under Evidence, and report each one that fails.
5. Compare the change with the customer skills, as `agents/rules.md` "Customer skills"
   defines, and name the skill file and the text to add or change.
6. Report the two verdicts `AGENTS.md` requires, then the findings.

## Shape

- **Scope:** the change solves the issue's problem, delivers every claimed behavior, and adds
  only justified scope. Compared with the nearest shipped example, it has one entrypoint per
  language, one config, and at most one controller or UI.
- **Simplest form:** every helper, abstraction, dependency, graph, process, and compatibility
  path has a stated reason. Ask whether the same behavior needs fewer parts.
- **Code:** the entrypoints follow `agents/coding-style.md`, and every piece of code sits where
  `agents/rules.md` "Where code goes" places it.
- **Behavior:** the example meets `agents/rules.md` "Behavior": matched languages, Insight for
  real-time visualization, and measured performance changes.
- **Config:** keys follow `agents/rules.md` "Config keys". Compare each key the change adds
  with every `meaning` in `examples/config-keys.yaml`, and report a key that renames or
  restates in another unit a setting a listed key covers, naming the key to use. A key that
  another example already has under `app.` moves to `examples/config-keys.yaml`.
- **Demonstration:** every prerequisite meets `agents/rules.md` "Documentation and
  publication". A fixture a test creates does not count.
- **Outputs:** what the customer sees is true. Inspect printed results, saved files, overlays,
  labels, and metadata, including a run with no detections when one applies. Each number names
  what it measures and its unit, and a failed operation reports a failure.
- **Models:** the README's model source matches the example's `tests/test-scope.yaml` and
  follows the order `CONTRIBUTING.md` gives.
- **Documentation:** read every added, changed, or deleted Markdown file as customer content.
  The README's `Concept` graph matches the parts and links `main` connects.
- **Tests:** they follow `agents/skills/write-example-tests/SKILL.md`. Each advertised behavior
  maps to an end-to-end assertion in both languages. Stateful examples test failure,
  replacement, stop, cleanup, and restart.
- **Core candidates:** report each piece of code that works around a missing or broken Core
  capability as its own finding labeled `Core candidate`: a helper every example would need,
  a format parsed by hand, a value Core should provide, or a comment describing a runtime
  limitation. Name the Core API that would replace it and the lines it would save. These
  findings are proposals for the maintainer, not blockers.
- **Code owners:** flag changes to dependency pins, `build.sh`, `scripts/ci/package-apps.sh`,
  or `.github/workflows/vulcan-ci.yml` for code owner review.

## Evidence

- **Package:** one customer workflow works from the package: install with
  `sima-cli neat install apps@<branch>`, configure, run, and inspect the result, and the
  description records it. Report source-only paths, missing packaged files, and customer
  options spread across config, CLI arguments, and environment variables.
- **Commands:** README commands, including every model download, ran exactly as written. A
  matching test entry or a file downloaded another way leaves the command unverified.
- **CI:** read the latest Vulcan CI run for the branch. A skipped test proves nothing, and a
  green run adds to the checks above.
- **Performance claims:** each names what was measured, where the results were counted, and
  the MLA clock. Keeping up with a source's frame rate shows no spare capacity, the same URL
  repeated proves capacity only for that URL, and a latency number follows an untimed warm-up.
- **Packaging and portal:** when packaging changes, `scripts/ci/validate_apps_runtime_archive.sh`
  passes. When a README or preview changes, catalog generation succeeds, only source files
  were edited, and the catalog and detail pages were checked at desktop and phone widths.

## Findings

- Anchor each finding to the smallest changed line range, and state the condition, the
  breakage, its impact, and what should change, in plain English.
- Separate correctness, structure, packaging, and customer-workflow blockers from optional
  naming, style, and performance follow-ups. A blocker can stand while CI passes.
- Report style only where `agents/coding-style.md` states the rule, and a test request only
  with the behavior that could regress.
