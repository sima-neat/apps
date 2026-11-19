---
name: apps-review-pr
description: Review a sima-neat/apps pull request for merge readiness and for the customer skill updates it should make.
---

# Review an Apps Pull Request

Review the complete affected application: entrypoints, configuration, tests, README, and
preview. A changed line can break code it does not touch.

`agents/rules.md`, `agents/coding-style.md`, and `CONTRIBUTING.md` own the rules. This skill
says what to check and in which order; cite the owning file in each finding.

## Steps

1. Read the linked issue and the pull request description. List every behavior the change
   claims.
2. Read `agents/rules.md`, `agents/coding-style.md`, the `CONTRIBUTING.md` sections for the
   areas the change touches, and both customer skills under `skills/`.
3. Trace each mode the README advertises from source to result: the input, the model, every
   output, and the processing in the application. Individually valid API calls do not prove
   the topology fits the requirement. The step is done when every claimed behavior maps to
   code you read, or is reported missing.
4. Check every item under Shape and under Evidence. Report each one that fails.
5. Compare the change with the customer skills, as Skill updates describes.
6. Report the two verdicts `AGENTS.md` requires, then the findings.

## Shape

- **Scope:** the implementation solves the issue's problem, delivers every claimed behavior,
  and adds no unjustified scope. Compared with the nearest shipped example, it adds no
  duplicate entrypoint, controller, UI, process, or setup path.
- **Simplest form:** every extra helper, abstraction, dependency, graph, process, and
  compatibility path has a stated reason. Ask whether the same behavior needs fewer parts.
- **Entrypoint:** it follows `agents/coding-style.md`: one function reads as the application
  flow and builds the whole graph, functions are named for what they return or do, and
  options restate no Core default.
- **Placement:** code sits where `agents/rules.md` places it. Report a Neat-free helper copied
  into an example that `support/` already has or that another example needs, and a Neat API
  call hidden in `support/`.
- **Config:** it uses the shared key names, keeps example-only settings under `app.`, comments
  each key, and the application rejects unread keys.
- **Demonstration:** every prerequisite is shipped, obtainable with a documented command,
  provided by the customer as the README states, or produced by an earlier documented step.
  A fixture a test creates does not count.
- **Outputs:** what the customer sees is true. Inspect the printed results, saved files,
  overlays, labels, and metadata, including a case with no detections when one applies. Each
  number names what it measures and its unit, and a failed operation is reported as a
  failure.
- **Models:** the README's model source matches the example's `tests/test-scope.yaml` and
  follows the order `CONTRIBUTING.md` gives. Model-preparation files follow the placement
  rule in `agents/rules.md`.
- **Documentation:** every added, changed, or deleted Markdown file is read as product
  content for a customer. A change to a rule every README follows accounts for every example
  and updates the template and validator.
- **Visualization:** real-time visualization uses Insight through the Neat video and metadata
  senders.
- **Tests:** each advertised behavior maps to an end-to-end assertion in both languages. A
  test that only starts the process is no evidence. Stateful applications test failure,
  replacement, stop, cleanup, and restart.
- **Code owners:** changes to dependency pins, `build.sh`, or
  `.github/workflows/vulcan-ci.yml` are flagged for code owner review.

## Evidence

- **Package:** one customer workflow works from the package: install with
  `sima-cli neat install apps@<branch>`, configure, run, and inspect the result. The pull
  request description records it. Report source-only paths, missing packaged files, and
  customer options spread across config, CLI arguments, and environment variables.
- **Commands:** README commands, including every model download, ran exactly as written. A
  matching test entry or a file downloaded another way does not prove the command.
- **CI:** read the latest Vulcan CI run for the branch. A skipped test proves nothing, and a
  green run does not replace the checks above.
- **Performance claims:** each names what was measured, where the results were counted, and
  the MLA clock. Keeping up with a source's frame rate is not a capacity measurement, and the
  same URL repeated does not prove capacity for that many distinct streams. A latency number
  follows an untimed warm-up.
- **Packaging and portal:** when packaging changes, `scripts/ci/validate_apps_runtime_archive.sh`
  passes. When a README or preview changes, catalog generation succeeds, only source files
  were edited, and the catalog and detail pages were checked at desktop and phone widths.

## Skill updates

The customer skills under `skills/` teach application developers' agents the Neat APIs. A pull
request updates them when it shows something they lack or contradict:

- a node, node group, or link policy used in a way the builder skill does not describe;
- a topology pattern, such as a shared model across streams, that no reference covers;
- a tuning choice with its measurement that the optimizer skill does not cover;
- a default, error, or pitfall the change uncovered;
- a skill statement the change proves wrong.

For each, name the skill file and the text to add or change, or give the verdict
`Skill updates: none` with the reason, as `AGENTS.md` requires.

## Findings

- Anchor each finding to the smallest changed line range.
- Separate correctness, structure, packaging, and customer-workflow blockers from optional
  naming, style, and performance follow-ups. A blocker can stand while CI passes.
- Skip style preferences outside `agents/coding-style.md`, vague risks, and test requests that
  name no behavior that could regress.
- Write in plain English and state what should change.
